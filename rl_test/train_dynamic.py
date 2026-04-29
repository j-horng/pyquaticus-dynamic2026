# SPDX-License-Identifier: BSD-3-Clause
"""
Train MARL policies on Dynamic PyQuaticus with our GNN policy only.

Uses graph observations and the custom GNN model (message passing, self-node embedding).

Usage:
  python rl_test/train_dynamic.py
  python rl_test/train_dynamic.py --render
  # Overnight: progress is logged to out_dir/train.log (disable with --no-log-file). Matchup lines go to stderr only.
  python rl_test/train_dynamic.py --speedup 8 --runners 16
  # Save more often so you can resume if you have to stop early (e.g. --save-every 100):
  python rl_test/train_dynamic.py --speedup 8 --runners 16 --save-every 100
  # Resume after a crash (continues from next iteration, saves to same out_dir).
  # You can change Red difficulty when resuming (e.g. --red-heuristic-mode medium); Blue is restored, Red is rebuilt from current args.
  python rl_test/train_dynamic.py --resume ./training/iter_1250
  # Train vs built-in heuristic (default: easy first, then resume with --red-heuristic-mode medium):
  python rl_test/train_dynamic.py --red-heuristic
  python rl_test/train_dynamic.py --resume ./training/iter_N --red-heuristic --red-heuristic-mode medium
  # Self-play: Red uses Blue from a previous checkpoint (e.g. 12 iters behind):
  python rl_test/train_dynamic.py --resume ./training/iter_700 --red-from-checkpoint ./training/iter_688
  # Quick smoke test before a long run:
  python rl_test/train_dynamic.py --iters 100

  # NRL action-map run (writes checkpoints to a dedicated folder):
  python rl_test/train_dynamic.py --out-dir ./training/ray_dynamic_v7_new_MAP --action-map nrl
  # Watch a saved checkpoint:
  python rl_test/train_dynamic.py --resume ./training/ray_dynamic_v7_new_MAP/iter_10 --watch --action-map nrl

  # Watch / deploy without duplicating CLI: same env factory and flags as training (no PPO):
  python rl_test/train_dynamic.py --watch --team-size-min 4 --team-size-max 4
  python rl_test/train_dynamic.py --watch --resume ./training/iter_500 --team-size-min 4 --team-size-max 4
  python rl_test/train_dynamic.py --watch --resume ./training/iter_500 --red-heuristic --team-size-min 4 --team-size-max 4

  # Save checkpoint right now (while training is running): create file SAVE_NOW in out_dir.
  # E.g. from another terminal:  echo. > training/SAVE_NOW   (Windows)
  #                             touch training/SAVE_NOW      (Linux/Mac)
  # Next completed iteration will save to iter_N and delete SAVE_NOW.
"""

# Default directory for checkpoints (iter_N/) and train.log (relative to cwd).
TRAINING_OUT_DIR = "./training/"
# When --train-batch-size 0 (headless): sized so workers usually finish before Ray sample_timeout (raise if stable).
DEFAULT_HEADLESS_TRAIN_BATCH_SIZE = 4000

import argparse
import logging
import os
import random
import re
import sys
import time
from collections.abc import Mapping

import numpy as np
import ray
from gymnasium.spaces import Discrete
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

try:
    from ray.rllib.algorithms.registry import POLICIES
except ImportError:
    try:
        from ray.rllib.policy.registry import POLICIES
    except ImportError:
        POLICIES = None
try:
    from ray.rllib.models.catalog import ModelCatalog
    from pyquaticus.models.gnn_model import GNNModel
    ModelCatalog.register_custom_model("gnn_model", GNNModel)
except Exception as e:
    raise RuntimeError("GNN model registration failed (need ray/rllib and pyquaticus.models.gnn_model).") from e

import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std, ACTION_MAP
from pyquaticus.action_map import apply_legacy_action_map, apply_nrl_action_map
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
from pyquaticus.structs import Team
from pyquaticus.envs.graph_obs_wrapper import GraphObsWrapper
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

# Max agents per team supported by graph obs (6v6 = 12). Must match graph_obs_wrapper.MAX_AGENTS // 2.
MAX_TEAM_CAP = 6

# Episode custom metrics like "3v3/win" are summarized as "3v3/win_mean" under env_runners on many Ray versions.
_MATCHUP_PREFIX_RE = re.compile(r"^(\d+v\d+)/")
_ACTION_MAP_MODE = "nrl"

def _default_action_space():
    """Fallback space if wrapped env does not expose per-agent spaces."""
    return Discrete(len(ACTION_MAP))

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except Exception:
    try:
        from ray.rllib.agents.callbacks import DefaultCallbacks
    except Exception:
        DefaultCallbacks = object


class RandPolicy(Policy):
    """Random policy for opponent agents."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        n = len(obs_batch)
        if hasattr(self.action_space, "n"):
            return [np.random.randint(0, self.action_space.n) for _ in range(n)], [], {}
        return [self.action_space.sample() for _ in range(n)], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


class DoNothingPolicy(Policy):
    """Deterministic policy that always selects the no-op action for opponents."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # In discrete mode, the last ACTION_MAP entry is defined as the "none" action.
        if hasattr(action_space, "n"):
            self._noop_action = len(ACTION_MAP) - 1
        else:
            # For continuous spaces, the environment treats [0, 0] as no movement.
            self._noop_action = np.array([0.0, 0.0], dtype=np.float32)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        n = len(obs_batch)
        if isinstance(self._noop_action, np.ndarray):
            actions = [self._noop_action.copy() for _ in range(n)]
        else:
            actions = [self._noop_action for _ in range(n)]
        return actions, [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


# Register so checkpoints can load custom policies when restoring (durable names).
if POLICIES is not None:
    POLICIES["RandPolicy"] = RandPolicy
    POLICIES["DoNothingPolicy"] = DoNothingPolicy


def make_env(
    config=None,
    render_mode=None,
    render_agent_ids=False,
    sim_speedup=4,
    red_gets_raw_obs=False,
    red_dummy=False,
    stationary_red=False,
    red_stationary=False,
    red_attack_hard=False,
    red_all_attack=False,
    red_all_defend=False,
    max_time=600,
    max_score=3,
    score_ends_episode=True,
    team_size_range=(1, 6),
    tag_removes_agent=False,
    reinforcement_interval=0,
    reinforcement_prob=0.5,
    dynamic_toggle_on=False,
    dynamic_toggle_interval=200,
    dynamic_toggle_remove_prob=0.5,
    dynamic_toggle_add_prob=0.5,
    fixed_spawn=True,
    stationary_red_active=None,
    stationary_red_active_random=False,
    stationary_red_random_min=None,
    stationary_red_random_max=None,
    stationary_red_midfield_spawn=False,
    stationary_red_block_anchor=None,
    stationary_red_block_anchor_random=False,
):
    if _ACTION_MAP_MODE == "nrl":
        apply_nrl_action_map()
    else:
        apply_legacy_action_map()
    cfg = config_dict_std.copy()
    cfg["sim_speedup_factor"] = sim_speedup
    cfg["max_score"] = max_score
    cfg["score_ends_episode"] = score_ends_episode
    cfg["max_time"] = max_time
    cfg["tagging_cooldown"] = 60
    cfg["tag_on_oob"] = True
    cfg["render_agent_ids"] = bool(render_agent_ids)
    # default_init True  = deterministic spawn-line placement (no random positions).
    # default_init False + on_sides_init True = random position on own side each reset.
    cfg["default_init"] = bool(fixed_spawn)
    cfg["on_sides_init"] = True
    if red_dummy:
        cfg["red_dummy_mode"] = True
    if stationary_red or red_stationary:
        cfg["stationary_red_mode"] = True
        if stationary_red_active is not None:
            cfg["stationary_red_active"] = int(stationary_red_active)
        if stationary_red_active_random:
            cfg["stationary_red_active_random"] = True
        if stationary_red_random_min is not None and stationary_red_random_max is not None:
            cfg["stationary_red_random_range"] = (int(stationary_red_random_min), int(stationary_red_random_max))
    if stationary_red_block_anchor_random:
        cfg["stationary_red_block_anchor_random"] = True
    elif stationary_red_block_anchor is not None:
        cfg["stationary_red_block_anchor"] = str(stationary_red_block_anchor).lower()
    elif stationary_red_midfield_spawn:
        cfg["stationary_red_block_anchor"] = "midfield"
    if red_attack_hard:
        # One hard attacker on Red, other Red slots disabled by forcing 1 active.
        cfg["force_num_red_active"] = 1
    max_team = team_size_range[1]
    if red_all_attack:
        cfg["force_num_red_active"] = max_team
    if red_all_defend:
        cfg["force_num_red_active"] = max_team

    reward_config = {f"agent_{i}": rew.caps_and_grabs for i in range(2 * max_team)}

    env = DynamicPyQuaticusEnv(
        team_size_range=team_size_range,
        tag_removes_agent=tag_removes_agent,
        reinforcement_interval=reinforcement_interval,
        reinforcement_prob=reinforcement_prob,
        dynamic_toggle_on=dynamic_toggle_on,
        dynamic_toggle_interval=dynamic_toggle_interval,
        dynamic_toggle_remove_prob=dynamic_toggle_remove_prob,
        dynamic_toggle_add_prob=dynamic_toggle_add_prob,
        config_dict=cfg,
        reward_config=reward_config,
        render_mode=render_mode,
    )
    # Graph obs for Blue (GNN); optionally pass raw obs for Red (heuristic policies)
    blue_ids = [f"agent_{i}" for i in range(max_team)]
    env = GraphObsWrapper(
        env, flatten_for_fc=False, red_gets_raw_obs=red_gets_raw_obs, blue_agent_ids=blue_ids
    )
    env = ParallelPettingZooWrapper(env)
    return env


def _resolve_blue_policy_path(checkpoint_dir):
    p = os.path.abspath(checkpoint_dir)
    if os.path.isdir(p) and not p.endswith("blue_policy"):
        return os.path.join(p, "policies", "blue_policy")
    return p


def _warn_ckpt_gnn_mismatch(blue_src, args, log_fn):
    try:
        ckpt_hidden = blue_src.model.config.get("custom_model_config", {}).get("gnn_hidden", 64)
        if int(ckpt_hidden) != int(args.gnn_hidden):
            log_fn(
                f"WARNING: checkpoint gnn_hidden={ckpt_hidden} but --gnn-hidden={args.gnn_hidden}. "
                f"Weight load will fail or silently mismatch. Pass --gnn-hidden {ckpt_hidden} to match."
            )
    except Exception:
        pass


def _get_action_space(env, agent_id):
    """Get action space for an agent; works with ParallelPettingZooWrapper and GraphObsWrapper."""
    if hasattr(env, "action_space") and callable(env.action_space):
        try:
            return env.action_space(agent_id)
        except Exception:
            pass
    spaces = getattr(env, "action_spaces", None)
    if spaces and isinstance(spaces, dict) and agent_id in spaces:
        return spaces[agent_id]
    par = getattr(env, "par_env", None)
    if par is not None:
        return _get_action_space(par, agent_id)
    return _default_action_space()


def _get_dynamic_pyquaticus(wrapped_env):
    """Unwrap ParallelPettingZooWrapper / GraphObsWrapper to DynamicPyQuaticusEnv."""
    e = wrapped_env
    for _ in range(6):
        if isinstance(e, DynamicPyQuaticusEnv):
            return e
        nxt = getattr(e, "par_env", None)
        if nxt is None:
            break
        e = nxt
    raise RuntimeError("Could not unwrap to DynamicPyQuaticusEnv")


def _callback_candidate_envs(episode, env, base_env, env_index):
    """Collect wrapped env objects RLlib may pass into callbacks."""
    candidates = []
    if env is not None:
        candidates.append(env)
    if base_env is not None:
        try:
            subs = base_env.get_sub_environments()
            if subs is not None and isinstance(env_index, int) and 0 <= env_index < len(subs):
                candidates.append(subs[env_index])
        except Exception:
            pass
        try:
            vec = getattr(base_env, "vector_env", None)
            if vec is not None and getattr(vec, "envs", None):
                if isinstance(env_index, int) and 0 <= env_index < len(vec.envs):
                    candidates.append(vec.envs[env_index])
        except Exception:
            pass
    return candidates


def _callback_try_get_dynamic_pyquaticus(episode, env=None, base_env=None, env_index=0):
    for e in _callback_candidate_envs(episode, env, base_env, env_index):
        try:
            return _get_dynamic_pyquaticus(e)
        except Exception:
            continue
    return None


def _callback_metrics_dict(result):
    """Collect per-episode custom metrics from all places RLlib may put them (Ray version dependent)."""
    out = {}
    if not isinstance(result, Mapping):
        return out
    cm = result.get("custom_metrics")
    if isinstance(cm, Mapping):
        out.update(cm)
    for block_name in ("env_runners", "sampler_results"):
        block = result.get(block_name)
        if not isinstance(block, Mapping):
            continue
        nested = block.get("custom_metrics")
        if isinstance(nested, Mapping):
            out.update(nested)
        for k, v in block.items():
            if k == "custom_metrics":
                continue
            if isinstance(k, str) and _MATCHUP_PREFIX_RE.match(k):
                out[k] = v
    return out


def _callback_episode_length(episode):
    for name in ("env_steps", "episode_length"):
        v = getattr(episode, name, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    try:
        return int(len(episode))
    except Exception:
        return 0


def _callback_read_active_sizes_from_info(episode, dynamic):
    """Prefer num_blue_active / num_red_active from env info (last step); fallback to env attributes."""
    info_dict = None
    if hasattr(episode, "get_infos"):
        try:
            last_infos = episode.get_infos(-1)
            if isinstance(last_infos, dict):
                for inf in last_infos.values():
                    if isinstance(inf, dict) and "num_blue_active" in inf:
                        info_dict = inf
                        break
        except Exception:
            pass
    if info_dict is None and hasattr(episode, "get_agents"):
        try:
            for aid in episode.get_agents():
                try:
                    inf = episode.last_info_for(aid)
                except Exception:
                    continue
                if isinstance(inf, dict) and "num_blue_active" in inf:
                    info_dict = inf
                    break
        except Exception:
            pass
    if info_dict is None and hasattr(episode, "last_info_for"):
        try:
            inf = episode.last_info_for("agent_0")
            if isinstance(inf, dict) and "num_blue_active" in inf:
                info_dict = inf
        except Exception:
            pass
    if info_dict is not None:
        return int(info_dict["num_blue_active"]), int(info_dict["num_red_active"])
    return int(getattr(dynamic, "num_blue_active", 0)), int(getattr(dynamic, "num_red_active", 0))


class DynamicPyQuaticusCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_end(self, *, episode, env=None, base_env=None, env_index=0, **kwargs):
        dynamic = None
        if base_env is not None:
            try:
                subs = base_env.get_sub_environments()
                if subs:
                    idx = env_index if env_index < len(subs) else 0
                    dynamic = _get_dynamic_pyquaticus(subs[idx])
            except Exception:
                pass
        if dynamic is None:
            dynamic = _callback_try_get_dynamic_pyquaticus(episode, env=env, base_env=base_env, env_index=env_index)
        if dynamic is None:
            return
        try:
            dynamic._set_game_events_from_state()
        except Exception:
            pass

        nb, nr = _callback_read_active_sizes_from_info(episode, dynamic)
        key = f"{nb}v{nr}"

        b, r = Team.BLUE_TEAM, Team.RED_TEAM
        ge = dynamic.game_events
        tags_b = int(ge[b]["tags"])
        tags_r = int(ge[r]["tags"])
        grabs_b = int(ge[b]["grabs"])
        grabs_r = int(ge[r]["grabs"])
        caps_b = int(dynamic.state["captures"][int(b)])
        caps_r = int(dynamic.state["captures"][int(r)])
        ep_len = float(max(1, _callback_episode_length(episode)))

        win = 1.0 if caps_b > caps_r else 0.0
        blue_oob = float(dynamic.state.get("blue_oob_count", 0))
        red_oob = float(dynamic.state.get("red_oob_count", 0))

        cm = getattr(episode, "custom_metrics", None)
        if cm is None:
            episode.custom_metrics = {}
            cm = episode.custom_metrics
        cm[f"{key}/win"] = win
        cm[f"{key}/ep_len"] = ep_len
        cm[f"{key}/blue_caps"] = float(caps_b)
        cm[f"{key}/red_caps"] = float(caps_r)
        cm[f"{key}/blue_grabs"] = float(grabs_b)
        cm[f"{key}/red_grabs"] = float(grabs_r)
        cm[f"{key}/blue_drops"] = float(grabs_b - caps_b)
        cm[f"{key}/red_drops"] = float(grabs_r - caps_r)
        cm[f"{key}/blue_tags"] = float(tags_b)
        cm[f"{key}/red_tags"] = float(tags_r)
        cm[f"{key}/blue_oob"] = blue_oob
        cm[f"{key}/red_oob"] = red_oob

        # If using random Red opponent modes (easy/medium/hard random), policy_mapping_fn stores
        # the chosen variant in episode.user_data["red_variant"]. Surface it as custom_metrics so
        # we can print a per-iteration distribution in on_train_result.
        try:
            ud = getattr(episode, "user_data", None)
            if isinstance(ud, dict):
                v = ud.get("red_variant")
                if isinstance(v, str) and v:
                    cm[f"red_variant/{v}"] = 1.0
        except Exception:
            pass

    def on_train_result(self, *, algorithm, metrics_logger=None, result=None, **kwargs):
        if result is None:
            return
        cm = _callback_metrics_dict(result)
        prefixes = set()
        for k in cm:
            if not isinstance(k, str) or not k.endswith("_mean"):
                continue
            m = _MATCHUP_PREFIX_RE.match(k)
            if m:
                prefixes.add(m.group(1))
        total = len(prefixes)
        result["matchup_distribution"] = {
            k: float(cm.get(f"{k}/win_mean", 0.0)) for k in sorted(prefixes)
        }
        # Only print on checkpoint cadence (same as main loop: i>0 and i%save_every==0).
        # Keep console output uncluttered by avoiding per-iter prints.
        loop_i = getattr(algorithm, "_matchup_train_loop_i", None)
        se = max(1, int(getattr(algorithm, "_matchup_save_every", 5) or 5))
        should_print = (loop_i is not None and loop_i > 0 and loop_i % se == 0)
        if not should_print:
            return
        if prefixes:
            parts = " ".join(
                f"{k} win_frac={float(cm.get(f'{k}/win_mean', 0.0)):.2f}" for k in sorted(prefixes)
            )
            # Stderr only: train.log is fed by log() (stdout); avoid mixing matchup into stdout-only redirects.
            print(
                f"matchup stats this iter ({total} types; win_frac = mean 1[blue_caps>red_caps]): {parts}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print("matchup stats this iter (0 types)", file=sys.stderr, flush=True)

        # Print which random Red variant we trained against this iter (fractions over episodes).
        red_vars = []
        for k in cm:
            if isinstance(k, str) and k.startswith("red_variant/") and k.endswith("_mean"):
                red_vars.append(k)
        if red_vars:
            parts = " ".join(
                f"{k[len('red_variant/'):-len('_mean')]}={float(cm.get(k, 0.0)):.2f}" for k in sorted(red_vars)
            )
            print(f"red random opponent this iter (episode_frac): {parts}", file=sys.stderr, flush=True)

def _int_action(action):
    if isinstance(action, (list, tuple)):
        action = action[0]
    if hasattr(action, "item"):
        return int(action.item())
    return int(action)


def _run_watch(args):
    """Render loop: same kwargs as env_creator make_env; no PPO."""
    logging.basicConfig(level=logging.ERROR)
    ray.init(ignore_reinit_error=True)

    import pyquaticus.utils.rewards as rew

    if getattr(args, "reward_debug", False):
        rew.REWARD_DEBUG = True

    team_max = int(args.team_size_max)
    team_min = int(args.team_size_min)
    team_size_range = (team_min, team_max)
    SPEEDUP = max(1, int(args.speedup))
    reinf_interval = max(0, int(args.reinforcement_interval))
    reinf_prob = max(0.0, min(1.0, float(args.reinforcement_prob)))
    dt_interval = max(0, int(getattr(args, "dynamic_toggle_interval", 200)))
    dt_rm = max(0.0, min(1.0, float(getattr(args, "dynamic_toggle_remove_prob", 0.5))))
    dt_add = max(0.0, min(1.0, float(getattr(args, "dynamic_toggle_add_prob", 0.5))))

    env = None
    try:
        env = make_env(
            None,
            render_mode="human",
            render_agent_ids=True,
            sim_speedup=SPEEDUP,
            red_gets_raw_obs=(
                args.red_heuristic
                or args.red_attack_hard
                or args.red_all_attack
                or args.red_all_defend
                or getattr(args, "red_easy_attack", False)
                or getattr(args, "red_easy_defend", False)
                or getattr(args, "red_easy_combined", False)
                or getattr(args, "red_medium_attack", False)
                or getattr(args, "red_medium_defend", False)
                or getattr(args, "red_medium_combined", False)
                or getattr(args, "red_hard_attack", False)
                or getattr(args, "red_hard_defend", False)
                or getattr(args, "red_hard_combined", False)
            ),
            red_dummy=args.red_dummy,
            stationary_red=args.red_stationary,
            red_stationary=args.red_stationary,
            red_attack_hard=args.red_attack_hard,
            red_all_attack=args.red_all_attack,
            red_all_defend=args.red_all_defend,
            max_time=args.max_time,
            max_score=args.max_score,
            score_ends_episode=not args.no_score_end,
            team_size_range=team_size_range,
            tag_removes_agent=args.tag_removes_agent,
            reinforcement_interval=reinf_interval,
            reinforcement_prob=reinf_prob,
            dynamic_toggle_on=args.dynamic_toggle_on,
            dynamic_toggle_interval=dt_interval,
            dynamic_toggle_remove_prob=dt_rm,
            dynamic_toggle_add_prob=dt_add,
            fixed_spawn=args.fixed_spawn,
            stationary_red_active=args.stationary_red_active,
            stationary_red_active_random=args.stationary_red_active_random,
            stationary_red_random_min=args.stationary_red_random_min,
            stationary_red_random_max=args.stationary_red_random_max,
            stationary_red_block_anchor=getattr(args, "red_stationary_block_anchor", None),
            stationary_red_block_anchor_random=getattr(args, "red_stationary_block_anchor_random", False),
        )
    except Exception:
        ray.shutdown()
        raise

    dynamic_env = _get_dynamic_pyquaticus(env)
    dynamic_env.render_reward_thresholds = True
    dynamic_env.render_idle_dist_thresh_m = float(getattr(rew, "IDLE_DIST_THRESH", 0.0))
    dynamic_env.render_circle_dist_thresh_m = float(getattr(rew, "CIRCLE_DIST_THRESH", 0.0))
    dynamic_env.render_circle_heading_delta_deg = float(getattr(rew, "CIRCLE_HEADING_DELTA_DEG", 0.0))
    dynamic_env.render_idle_grace_steps = int(getattr(rew, "IDLE_GRACE_STEPS", 0))
    dynamic_env.render_circle_grace_steps = int(getattr(rew, "CIRCLE_GRACE_STEPS", 0))
    blue_ids = [f"agent_{i}" for i in range(team_max)]
    red_ids = [f"agent_{i}" for i in range(team_max, 2 * team_max)]

    blue_policy = None
    if args.resume:
        policy_path = _resolve_blue_policy_path(args.resume)
        if os.path.isdir(policy_path):
            print(f"Watch: loading blue policy from {policy_path}")
            blue_policy = Policy.from_checkpoint(policy_path)
        else:
            print(f"Watch: no policy at {policy_path}, using random actions for Blue")

    red_heuristics = {}
    red_ckpt_policy = None
    noop = len(ACTION_MAP) - 1
    red_random_pool = None

    def _pick_red_random_pool():
        if getattr(args, "red_easy_random", False):
            return [
                # Distribution variants (per-episode): match the explicit --red-easy-* distributions.
                ("easy_attack_dist", None, None),
                ("easy_defend_dist", None, None),
                ("easy_combined_dist", None, None),
            ]
        if getattr(args, "red_medium_random", False):
            return [
                # Distribution variants (per-episode): match the explicit --red-medium-* distributions.
                ("medium_attack_dist", None, None),
                ("medium_defend_dist", None, None),
                ("medium_combined_dist", None, None),
            ]
        if getattr(args, "red_hard_random", False):
            return [
                # Distribution variants (per-episode): match the explicit --red-hard-* distributions.
                ("hard_attack_dist", None, None),
                ("hard_defend_dist", None, None),
                ("hard_combined_dist", None, None),
            ]
        return None

    def _reroll_red_heuristics_for_episode():
        """When using --red-*-random, pick one variant per episode (at each env.reset())."""
        nonlocal red_heuristics
        if not red_random_pool:
            return None
        chosen_name, chosen_cls, chosen_kw = random.choice(red_random_pool)
        if chosen_cls is None and chosen_name in {
            "easy_attack_dist", "easy_defend_dist", "easy_combined_dist",
            "medium_attack_dist", "medium_defend_dist", "medium_combined_dist",
            "hard_attack_dist", "hard_defend_dist", "hard_combined_dist",
        }:
            if chosen_name.startswith("easy_"):
                mode = "easy"
            elif chosen_name.startswith("medium_"):
                mode = "medium"
            else:
                mode = "hard"
            red_heuristics = {}
            for aid in red_ids:
                try:
                    idx = int(str(aid).split("_", 1)[1])
                except Exception:
                    idx = 0
                slot = idx % 6
                if chosen_name.endswith("attack_dist"):
                    # 3 attackers + 1 combined + 2 defenders
                    if slot in (0, 1, 2):
                        red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode=mode)
                    elif slot == 3:
                        red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode=mode)
                    else:
                        red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode=mode)
                elif chosen_name.endswith("defend_dist"):
                    # 3 defenders + 1 combined + 2 attackers
                    if slot in (0, 1, 2):
                        red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode=mode)
                    elif slot == 3:
                        red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode=mode)
                    else:
                        red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode=mode)
                else:
                    # 2 combined + 2 attackers + 2 defenders
                    if slot in (0, 1):
                        red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode=mode)
                    elif slot in (2, 3):
                        red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode=mode)
                    else:
                        red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode=mode)
        else:
            red_heuristics = {aid: chosen_cls(aid, dynamic_env, **chosen_kw) for aid in red_ids}
        return chosen_name

    if args.red_heuristic:
        red_heuristics = {
            aid: Heuristic_CTF_Agent(aid, dynamic_env, mode=args.red_heuristic_mode) for aid in red_ids
        }
        print(f"Watch: Red heuristic (combined CTF, {args.red_heuristic_mode}).")
    elif getattr(args, "red_easy_random", False) or getattr(args, "red_medium_random", False) or getattr(args, "red_hard_random", False):
        red_random_pool = _pick_red_random_pool()
        chosen_name = _reroll_red_heuristics_for_episode()
        print(f"Watch: Red randomized each episode among attack/defend/combined -> chosen {chosen_name}.")
    elif getattr(args, "red_easy_attack", False):
        # Distribution: 3 attackers + 1 combined + 2 defenders (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1, 2):
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="easy")
            elif slot == 3:
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="easy")
            else:
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="easy")
        print(
            "Watch: Red easy-attack distribution (3x BaseAttacker easy, 1x Heuristic_CTF_Agent easy, 2x BaseDefender easy)."
        )
    elif getattr(args, "red_easy_defend", False):
        # Distribution: 3 defenders + 1 combined + 2 attackers (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1, 2):
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="easy")
            elif slot == 3:
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="easy")
            else:
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="easy")
        print(
            "Watch: Red easy-defend distribution (3x BaseDefender easy, 1x Heuristic_CTF_Agent easy, 2x BaseAttacker easy)."
        )
    elif getattr(args, "red_easy_combined", False):
        # Distribution: 2 combined + 2 attackers + 2 defenders (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1):
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="easy")
            elif slot in (2, 3):
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="easy")
            else:
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="easy")
        print(
            "Watch: Red easy-combined distribution (2x Heuristic_CTF_Agent easy, 2x BaseAttacker easy, 2x BaseDefender easy)."
        )
    elif getattr(args, "red_medium_attack", False):
        # Distribution: 3 attackers + 1 combined + 2 defenders (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1, 2):
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="medium")
            elif slot == 3:
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="medium")
            else:
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="medium")
        print(
            "Watch: Red medium-attack distribution (3x BaseAttacker medium, 1x Heuristic_CTF_Agent medium, 2x BaseDefender medium)."
        )
    elif getattr(args, "red_medium_defend", False):
        # Distribution: 3 defenders + 1 combined + 2 attackers (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1, 2):
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="medium")
            elif slot == 3:
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="medium")
            else:
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="medium")
        print(
            "Watch: Red medium-defend distribution (3x BaseDefender medium, 1x Heuristic_CTF_Agent medium, 2x BaseAttacker medium)."
        )
    elif getattr(args, "red_medium_combined", False):
        # Distribution: 2 combined + 2 attackers + 2 defenders (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1):
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="medium")
            elif slot in (2, 3):
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="medium")
            else:
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="medium")
        print(
            "Watch: Red medium-combined distribution (2x Heuristic_CTF_Agent medium, 2x BaseAttacker medium, 2x BaseDefender medium)."
        )
    elif getattr(args, "red_hard_attack", False):
        # Distribution: 3 attackers + 1 combined + 2 defenders (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1, 2):
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="hard")
            elif slot == 3:
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="hard")
            else:
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="hard")
        print(
            "Watch: Red hard-attack distribution (3x BaseAttacker hard, 1x Heuristic_CTF_Agent hard, 2x BaseDefender hard)."
        )
    elif getattr(args, "red_hard_defend", False):
        # Distribution: 3 defenders + 1 combined + 2 attackers (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1, 2):
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="hard")
            elif slot == 3:
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="hard")
            else:
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="hard")
        print(
            "Watch: Red hard-defend distribution (3x BaseDefender hard, 1x Heuristic_CTF_Agent hard, 2x BaseAttacker hard)."
        )
    elif getattr(args, "red_hard_combined", False):
        # Distribution: 2 combined + 2 attackers + 2 defenders (slot-based).
        red_heuristics = {}
        for aid in red_ids:
            try:
                idx = int(str(aid).split("_", 1)[1])
            except Exception:
                idx = 0
            slot = idx % 6
            if slot in (0, 1):
                red_heuristics[aid] = Heuristic_CTF_Agent(aid, dynamic_env, mode="hard")
            elif slot in (2, 3):
                red_heuristics[aid] = BaseAttacker(aid, dynamic_env, mode="hard")
            else:
                red_heuristics[aid] = BaseDefender(aid, dynamic_env, mode="hard")
        print(
            "Watch: Red hard-combined distribution (2x Heuristic_CTF_Agent hard, 2x BaseAttacker hard, 2x BaseDefender hard)."
        )
    elif args.red_from_checkpoint:
        red_path = _resolve_blue_policy_path(args.red_from_checkpoint)
        if os.path.isdir(red_path):
            print(f"Watch: loading red policy from {red_path}")
            red_ckpt_policy = Policy.from_checkpoint(red_path)
        else:
            print(f"Watch: red checkpoint not found at {red_path}, using random for Red")
    elif args.red_all_attack:
        red_heuristics = {aid: BaseAttacker(aid, dynamic_env, mode="hard") for aid in red_ids}
        print("Watch: Red all-attack heuristic (BaseAttacker hard).")
    elif args.red_all_defend:
        red_heuristics = {aid: BaseDefender(aid, dynamic_env, mode="hard") for aid in red_ids}
        print("Watch: Red all-defend heuristic (BaseDefender hard).")
    elif args.red_attack_hard:
        red_heuristics = {aid: BaseAttacker(aid, dynamic_env, mode="hard") for aid in red_ids}
        print("Watch: Red attack-hard (one active Red; BaseAttacker hard on each slot).")
    elif args.red_dummy or args.red_stationary:
        if args.red_stationary:
            if getattr(args, "stationary_red_active_random", False):
                _rn = getattr(args, "stationary_red_random_min", None)
                _rx = getattr(args, "stationary_red_random_max", None)
                if _rn is not None and _rx is not None:
                    _rng = f"{_rn}-{_rx}"
                else:
                    _rng = f"{args.team_size_min}-{args.team_size_max}"
                print(
                    f"Watch: Red stationary (no-op), random active count each episode {_rng} "
                    f"(Blue team size still {args.team_size_min}-{args.team_size_max})."
                )
            else:
                _nw = int(args.stationary_red_active) if args.stationary_red_active is not None else 2
                if getattr(args, "red_stationary_block_anchor_random", False):
                    print(
                        f"Watch: Red stationary block-random (no-op), target {_nw} active Red agents; "
                        f"each episode picks center / upper / lower spawn-row pair uniformly."
                    )
                else:
                    _ba = getattr(args, "red_stationary_block_anchor", None)
                    if _ba == "midfield":
                        print(
                            f"Watch: Red stationary at midfield (no-op), target {_nw} active Red agents "
                            f"(center pair on default Red spawn row)."
                        )
                    elif _ba == "topfield":
                        print(
                            f"Watch: Red stationary at topfield (no-op), target {_nw} active Red agents "
                            f"(upper pair on default Red spawn row)."
                        )
                    elif _ba == "bottomfield":
                        print(
                            f"Watch: Red stationary at bottomfield (no-op), target {_nw} active Red agents "
                            f"(lower pair on default Red spawn row)."
                        )
                    else:
                        print(f"Watch: Red stationary (no-op), target {_nw} active Red agents.")
        else:
            print("Watch: Red do-nothing (dummy).")
    else:
        print("Watch: Red random actions.")

    obs, info = env.reset()
    episode_idx, step = 0, 0

    try:
        while True:
            actions = {}
            global_state = dynamic_env._history_to_state()
            for aid in obs:
                if blue_policy is not None and aid in blue_ids:
                    out = blue_policy.compute_single_action(obs[aid], explore=False)
                    actions[aid] = _int_action(out[0] if isinstance(out, (list, tuple)) else out)
                elif aid in blue_ids:
                    samp = _get_action_space(env, aid).sample()
                    actions[aid] = _int_action(samp)
                elif aid in red_heuristics:
                    hinfo = {aid: {"global_state": global_state}}
                    act = red_heuristics[aid].compute_action(obs[aid], hinfo)
                    actions[aid] = _int_action(act)
                elif red_ckpt_policy is not None:
                    out = red_ckpt_policy.compute_single_action(obs[aid], explore=False)
                    actions[aid] = _int_action(out[0] if isinstance(out, (list, tuple)) else out)
                elif args.red_dummy or args.red_stationary:
                    actions[aid] = noop
                else:
                    samp = _get_action_space(env, aid).sample()
                    actions[aid] = _int_action(samp)

            obs, rewards, term, trunc, info = env.step(actions)
            step += 1

            done = any(term.values()) or any(trunc.values())
            if done:
                episode_idx += 1
                try:
                    dynamic_env._set_game_events_from_state()
                except Exception:
                    pass
                caps = np.asarray(dynamic_env.state["captures"]).flatten()
                cb, cr = int(caps[0]), int(caps[1])
                winner = "Blue" if cb > cr else ("Red" if cr > cb else "Draw")
                b, r = Team.BLUE_TEAM, Team.RED_TEAM
                ge = dynamic_env.game_events
                gb, gr = int(ge[b]["grabs"]), int(ge[r]["grabs"])
                tb, tr = int(ge[b]["tags"]), int(ge[r]["tags"])
                print(
                    f"Episode {episode_idx} done in {step} steps | "
                    f"Blue caps {cb} Red caps {cr} ({winner}) | "
                    f"grabs B/R {gb}/{gr} tags B/R {tb}/{tr}"
                )
                # Match training behavior: if --red-*-random, reroll opponent type per episode.
                if red_random_pool:
                    chosen_name = _reroll_red_heuristics_for_episode()
                    print(
                        f"Watch: Red randomized each episode among attack/defend/combined -> chosen {chosen_name}.",
                        flush=True,
                    )
                obs, info = env.reset()
                step = 0
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        ray.shutdown()


def _coerce_finite_scalar(v):
    """Convert RLlib / numpy / torch leaf metrics to float; None if missing or non-finite."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        try:
            xs = []
            for x in v:
                t = x.item() if hasattr(x, "item") and callable(getattr(x, "item", None)) else x
                xs.append(float(t))
            if xs and all(np.isfinite(x) for x in xs):
                return float(np.mean(xs))
        except (TypeError, ValueError):
            return None
        return None
    try:
        x = v.item() if hasattr(v, "item") and callable(getattr(v, "item", None)) else v
        fv = float(x)
        return fv if np.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def _stat_key_matches(k, stat):
    if k == stat:
        return True
    if not isinstance(k, str):
        return False
    if k.endswith("/" + stat) or k.endswith("." + stat):
        return True
    if "/" in k and k.rsplit("/", 1)[-1] == stat:
        return True
    if "." in k and k.rsplit(".", 1)[-1] == stat:
        return True
    return False


def _find_stat(d, stat, depth=0):
    """Walk nested dict/list/Mapping; match exact keys or Tune-style dotted/slashed metric paths."""
    if depth > 14 or d is None:
        return None
    if isinstance(d, (dict, Mapping)):
        if stat in d:
            fv = _coerce_finite_scalar(d[stat])
            if fv is not None:
                return fv
        for k, v in d.items():
            if _stat_key_matches(k, stat):
                fv = _coerce_finite_scalar(v)
                if fv is not None:
                    return fv
        for v in d.values():
            found = _find_stat(v, stat, depth + 1)
            if found is not None:
                return found
    elif isinstance(d, (list, tuple)):
        for v in d:
            found = _find_stat(v, stat, depth + 1)
            if found is not None:
                return found
    return None


def _extract_blue_learner_metrics(result):
    """Read entropy / policy_loss for ``blue_policy`` from known RLlib layouts, then generic search."""
    entropy, policy_loss = None, None
    if not isinstance(result, Mapping):
        return entropy, policy_loss

    def _from_learner_stats(ls):
        if ls is None:
            return None, None
        if isinstance(ls, (list, tuple)) and ls:
            ls = ls[-1]
        if not isinstance(ls, Mapping):
            return None, None
        ent = _coerce_finite_scalar(ls.get("entropy"))
        if ent is None:
            ent = _coerce_finite_scalar(ls.get("mean_entropy"))
        pl = _coerce_finite_scalar(ls.get("policy_loss"))
        if pl is None:
            pl = _coerce_finite_scalar(ls.get("mean_policy_loss"))
        if pl is None:
            pl = _coerce_finite_scalar(ls.get("total_loss"))
        return ent, pl

    info = result.get("info")
    if isinstance(info, Mapping):
        for root_key in ("learner", "learners"):
            root = info.get(root_key)
            if not isinstance(root, Mapping):
                continue
            bp = root.get("blue_policy")
            if isinstance(bp, (list, tuple)) and bp:
                bp = bp[-1]
            if not isinstance(bp, Mapping):
                continue
            e, p = _from_learner_stats(bp.get("learner_stats"))
            if e is not None:
                entropy = e
            if p is not None:
                policy_loss = p
            if entropy is None:
                entropy = _coerce_finite_scalar(bp.get("entropy"))
            if policy_loss is None:
                policy_loss = _coerce_finite_scalar(bp.get("policy_loss"))
            if policy_loss is None:
                policy_loss = _coerce_finite_scalar(bp.get("total_loss"))
            if entropy is not None or policy_loss is not None:
                return entropy, policy_loss

    learners = result.get("learners")
    if isinstance(learners, Mapping):
        bp = learners.get("blue_policy")
        if isinstance(bp, (list, tuple)) and bp:
            bp = bp[-1]
        if isinstance(bp, Mapping):
            e, p = _from_learner_stats(bp.get("learner_stats"))
            if e is not None:
                entropy = e
            if p is not None:
                policy_loss = p
            if entropy is None:
                entropy = _coerce_finite_scalar(bp.get("entropy"))
            if policy_loss is None:
                policy_loss = _coerce_finite_scalar(bp.get("policy_loss"))
            if policy_loss is None:
                policy_loss = _coerce_finite_scalar(bp.get("total_loss"))
            if entropy is not None or policy_loss is not None:
                return entropy, policy_loss

    bp = result.get("blue_policy")
    if isinstance(bp, (list, tuple)) and bp:
        bp = bp[-1]
    if isinstance(bp, Mapping):
        e, p = _from_learner_stats(bp.get("learner_stats"))
        entropy = e if entropy is None else entropy
        policy_loss = p if policy_loss is None else policy_loss
        if entropy is None:
            entropy = _coerce_finite_scalar(bp.get("entropy"))
        if policy_loss is None:
            policy_loss = _coerce_finite_scalar(bp.get("policy_loss"))
        if policy_loss is None:
            policy_loss = _coerce_finite_scalar(bp.get("total_loss"))

    if entropy is None:
        entropy = _find_stat(result, "entropy") or _find_stat(result, "mean_entropy")
    if policy_loss is None:
        policy_loss = (
            _find_stat(result, "policy_loss")
            or _find_stat(result, "mean_policy_loss")
            or _find_stat(result, "total_loss")
        )
    return entropy, policy_loss


def main():
    parser = argparse.ArgumentParser(description="Train on Dynamic PyQuaticus")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--action-map",
        type=str,
        default="nrl",
        choices=["nrl", "legacy"],
        help="Discrete action map layout (default: nrl).",
    )
    parser.add_argument("--iters", type=int, default=2000, help="Training iterations")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N iters")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=TRAINING_OUT_DIR,
        help="Output directory for checkpoints and train.log (default: ./training/)",
    )
    parser.add_argument("--runners", type=int, default=8, help="Number of parallel env runners (8=stable default; increase if PC has headroom)")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=0,
        help=(
            "PPO train_batch_size (0=auto: 500 with --render, else "
            f"{DEFAULT_HEADLESS_TRAIN_BATCH_SIZE} headless). "
            "Lower if workers hit sample_timeout; raise when runners are fast."
        ),
    )
    parser.add_argument(
        "--envs-per-runner",
        type=int,
        default=1,
        help="Rollout workers run this many env copies each (1=default; try 2–4 for more samples/iter if CPU/RAM allow; ignored with --render).",
    )
    parser.add_argument("--entropy-coeff", type=float, default=0.05,
        help="PPO entropy coefficient (default 0.05 for Phase 1; use 0.01 for Phase 2+)")
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="PPO learning rate (default 3e-4; lower to 1e-4 for fine-tuning)",
    )
    parser.add_argument(
        "--gnn-hidden",
        type=int,
        default=128,
        help="GNN hidden dim (default 128; use 64 for faster iteration)",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=2,
        help="GNN message passing layers (default 2)",
    )
    parser.add_argument("--speedup", type=int, default=8, help="Sim speedup factor (8=env steps 2x faster, minimal impact on learning)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="Resume from checkpoint (e.g. ./training/iter_1250)")
    parser.add_argument("--watch", action="store_true", help="Render instead of training")
    parser.add_argument("--reward-debug", action="store_true", help="With --watch: print per-event reward lines ([REWARD] ...) to console")
    parser.add_argument("--no-log-file", action="store_true", help="Disable writing progress to out_dir/train.log")
    parser.add_argument("--red-heuristic", action="store_true", help="Use built-in heuristic (combined CTF) for Red instead of random")
    parser.add_argument("--red-heuristic-mode", type=str, default="easy", choices=["easy", "medium", "hard"], help="Heuristic difficulty when --red-heuristic (default: easy)")
    parser.add_argument("--red-easy-attack", action="store_true")
    parser.add_argument("--red-easy-defend", action="store_true")
    parser.add_argument("--red-easy-combined", action="store_true")
    parser.add_argument("--red-easy-random", action="store_true", help="Randomize Red each episode among: easy attack/defend/combined")
    parser.add_argument("--red-medium-attack", action="store_true")
    parser.add_argument("--red-medium-defend", action="store_true")
    parser.add_argument("--red-medium-combined", action="store_true")
    parser.add_argument("--red-medium-random", action="store_true", help="Randomize Red each episode among: medium attack/defend/combined")
    parser.add_argument("--red-hard-attack", action="store_true")
    parser.add_argument("--red-hard-defend", action="store_true")
    parser.add_argument("--red-hard-combined", action="store_true")
    parser.add_argument("--red-hard-random", action="store_true", help="Randomize Red each episode among: hard attack/defend/combined")
    parser.add_argument("--red-dummy", action="store_true", help="Use do-nothing policy for Red (always no-op)")
    parser.add_argument("--red-stationary", action="store_true", help="Red agents are stationary (no-op); count via --stationary-red-active")
    parser.add_argument(
        "--red-stationary-midfield",
        action="store_true",
        help=(
            "Stationary Red (no-op), active reds on two **center** slots of the default Red spawn row "
            "(slightly split across the row). Uses --stationary-red-active (default 2). "
            "Not combinable with --red-stationary / --stationary-red."
        ),
    )
    parser.add_argument(
        "--stationary-red-midfield",
        action="store_true",
        help="Alias for --red-stationary-midfield (mirrors --stationary-red for --red-stationary).",
    )
    parser.add_argument(
        "--red-stationary-center",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--red-stationary-topfield",
        action="store_true",
        help=(
            "Stationary Red (no-op), active reds on the **upper** pair of default Red spawn-row slots "
            "(same line as training spawns). Not combinable with --red-stationary or other stationary-block flags."
        ),
    )
    parser.add_argument(
        "--stationary-red-topfield",
        action="store_true",
        help="Alias for --red-stationary-topfield.",
    )
    parser.add_argument(
        "--red-stationary-bottomfield",
        action="store_true",
        help=(
            "Stationary Red (no-op), active reds on the **lower** pair of default Red spawn-row slots. "
            "Not combinable with --red-stationary or other stationary-block flags."
        ),
    )
    parser.add_argument(
        "--stationary-red-bottomfield",
        action="store_true",
        help="Alias for --red-stationary-bottomfield.",
    )
    parser.add_argument(
        "--red-stationary-block-random",
        "--stationary-red-block-random",
        action="store_true",
        dest="red_stationary_block_anchor_random",
        help=(
            "Stationary Red (no-op); each episode uniformly picks spawn-row pair: "
            "center / upper / lower (midfield, topfield, bottomfield). Not combinable with --red-stationary or fixed block flags."
        ),
    )
    parser.add_argument("--red-attack-hard", action="store_true", help="Red uses 1 hard attacker-only heuristic (other red slots disabled)")
    parser.add_argument("--red-all-attack", action="store_true", help="All red agents use AttackGen hard heuristic")
    parser.add_argument("--red-all-defend", action="store_true", help="All red agents use DefendGen hard heuristic")
    parser.add_argument(
        "--stationary-red",
        action="store_true",
        help="Alias for --red-stationary (Red stays put; how many are active: --stationary-red-active).",
    )
    parser.add_argument(
        "--stationary-red-active",
        type=int,
        default=None,
        metavar="N",
        help="With stationary Red (line or block anchor modes): fixed number of active stationary Red agents (0-6; default 2). Ignored if --stationary-red-active-random.",
    )
    parser.add_argument(
        "--stationary-red-active-random",
        action="store_true",
        help="With stationary Red (line or block anchor): each episode pick a random active stationary Red count (default: use --team-size-min/max; override with --stationary-red-random-min/max).",
    )
    parser.add_argument(
        "--stationary-red-random-min",
        type=int,
        default=None,
        metavar="N",
        help="With --stationary-red-active-random: minimum active stationary Reds per episode (use with --stationary-red-random-max).",
    )
    parser.add_argument(
        "--stationary-red-random-max",
        type=int,
        default=None,
        metavar="N",
        help="With --stationary-red-active-random: maximum active stationary Reds per episode (use with --stationary-red-random-min).",
    )
    parser.add_argument("--red-from-checkpoint", type=str, default=None, metavar="PATH", help="Use Blue policy from this checkpoint for Red (self-play vs previous iteration)")
    parser.add_argument(
        "--max-time",
        type=float,
        default=600,
        help="Max episode time in seconds (default 600 = 10 min; use e.g. 120 for shorter episodes / faster Ray sampling)",
    )
    parser.add_argument("--max-score", type=int, default=3, help="Max score per team to end episode (default 3)")
    parser.add_argument(
        "--no-score-end",
        action="store_true",
        help="Do not end the episode when a team reaches --max-score (default: episodes end on max score)",
    )
    parser.add_argument(
        "--score-ends-episode",
        action="store_true",
        help="End episodes when a team reaches --max-score (default already on; use to override a prior --no-score-end in the same command).",
    )
    parser.add_argument("--team-size-min", type=int, default=1, help="Min agents per team at episode start (default 1)")
    parser.add_argument("--team-size-max", type=int, default=6, help="Max agents per team at episode start (default 6)")
    parser.add_argument("--tag-removes-agent", action="store_true", help="When tagged, agent is disabled (removed) until reinforcement")
    parser.add_argument("--reinforcement-interval", type=int, default=0, help="Steps between reinforcement spawn checks (0=off, e.g. 500)")
    parser.add_argument("--reinforcement-prob", type=float, default=0.5, help="Probability of spawning one reinforcement when interval hits (default 0.5)")
    parser.add_argument(
        "--dynamic-toggle-on",
        action="store_true",
        help="Random roster changes on an interval: remove only tagged agents (min 1 active/team); randomly revive disabled agents",
    )
    parser.add_argument(
        "--dynamic-toggle-interval",
        type=int,
        default=200,
        help="Steps between dynamic roster rolls when --dynamic-toggle-on (0=never)",
    )
    parser.add_argument(
        "--dynamic-toggle-remove-prob",
        type=float,
        default=0.5,
        help="Each dynamic tick: probability of attempting one eligible removal (default 0.5)",
    )
    parser.add_argument(
        "--dynamic-toggle-add-prob",
        type=float,
        default=0.5,
        help="Each dynamic tick: probability of attempting one random revival (default 0.5)",
    )
    parser.add_argument(
        "--random-spawn",
        action="store_true",
        help="Random positions on own side each episode (default_init=False). Omit for deterministic spawn-line placement (training default).",
    )
    args = parser.parse_args()
    global _ACTION_MAP_MODE
    _ACTION_MAP_MODE = args.action_map
    if args.action_map == "nrl":
        apply_nrl_action_map()
    else:
        apply_legacy_action_map()
    if getattr(args, "score_ends_episode", False):
        args.no_score_end = False
    # Backwards/alias support: treat --stationary-red as enabling --red-stationary behavior.
    args.red_stationary = bool(getattr(args, "red_stationary", False) or getattr(args, "stationary_red", False))
    args.red_stationary_midfield = bool(
        getattr(args, "red_stationary_midfield", False)
        or getattr(args, "stationary_red_midfield", False)
        or getattr(args, "red_stationary_center", False)
    )
    args.red_stationary_topfield = bool(
        getattr(args, "red_stationary_topfield", False) or getattr(args, "stationary_red_topfield", False)
    )
    args.red_stationary_bottomfield = bool(
        getattr(args, "red_stationary_bottomfield", False) or getattr(args, "stationary_red_bottomfield", False)
    )
    _block_modes = []
    if args.red_stationary_midfield:
        _block_modes.append("midfield")
    if args.red_stationary_topfield:
        _block_modes.append("topfield")
    if args.red_stationary_bottomfield:
        _block_modes.append("bottomfield")
    if len(_block_modes) > 1:
        raise SystemExit(
            "Use at most one of: --red-stationary-midfield, --red-stationary-topfield, --red-stationary-bottomfield "
            "(and their --stationary-red-* aliases)."
        )
    args.red_stationary_block_anchor = _block_modes[0] if len(_block_modes) == 1 else None
    args.red_stationary_block_anchor_random = bool(getattr(args, "red_stationary_block_anchor_random", False))
    if args.red_stationary_block_anchor_random and len(_block_modes) > 0:
        raise SystemExit(
            "Do not combine --red-stationary-block-random with --red-stationary-midfield, "
            "--red-stationary-topfield, or --red-stationary-bottomfield (or their --stationary-red-* aliases)."
        )
    if args.red_stationary_block_anchor:
        if args.red_stationary:
            raise SystemExit(
                "Use either --red-stationary / --stationary-red or a stationary **block** mode "
                "(--red-stationary-midfield, --red-stationary-topfield, --red-stationary-bottomfield), not both."
            )
        args.red_stationary = True
    elif args.red_stationary_block_anchor_random:
        if args.red_stationary:
            raise SystemExit(
                "Use either --red-stationary / --stationary-red or --red-stationary-block-random, not both."
            )
        args.red_stationary = True
    if getattr(args, "red_stationary_center", False):
        print(
            "WARNING: --red-stationary-center is deprecated; use --red-stationary-midfield or --stationary-red-midfield.",
            file=sys.stderr,
            flush=True,
        )
    # Default spawn behavior is fixed spawn-line; opt into random with --random-spawn.
    args.fixed_spawn = not bool(getattr(args, "random_spawn", False))

    team_min, team_max = args.team_size_min, args.team_size_max
    if team_min < 1 or team_max > MAX_TEAM_CAP or team_min > team_max:
        raise SystemExit(f"Require 1 <= --team-size-min <= --team-size-max <= {MAX_TEAM_CAP}.")

    if getattr(args, "stationary_red_active", None) is not None:
        n = int(args.stationary_red_active)
        if n < 0 or n > MAX_TEAM_CAP:
            raise SystemExit(f"Require 0 <= --stationary-red-active <= {MAX_TEAM_CAP}.")

    if getattr(args, "stationary_red_active_random", False) and not args.red_stationary:
        raise SystemExit(
            "--stationary-red-active-random requires stationary Red "
            "(--red-stationary, --stationary-red, a --red-stationary-{midfield|topfield|bottomfield} mode, "
            "or --red-stationary-block-random)."
        )

    _srrn = getattr(args, "stationary_red_random_min", None)
    _srrx = getattr(args, "stationary_red_random_max", None)
    if (_srrn is None) != (_srrx is None):
        raise SystemExit("Use both --stationary-red-random-min and --stationary-red-random-max together, or neither.")
    if _srrn is not None:
        if not getattr(args, "stationary_red_active_random", False):
            raise SystemExit("--stationary-red-random-min/max require --stationary-red-active-random.")
        if _srrn < 1 or _srrx > MAX_TEAM_CAP or _srrn > _srrx:
            raise SystemExit(
                f"Require 1 <= --stationary-red-random-min <= --stationary-red-random-max <= {MAX_TEAM_CAP}."
            )
        if _srrx > team_max:
            raise SystemExit("--stationary-red-random-max cannot exceed --team-size-max.")

    _sta_block = bool(args.red_stationary_block_anchor) or bool(args.red_stationary_block_anchor_random)
    _red_new_heur = any(
        [
            bool(getattr(args, "red_easy_attack", False)),
            bool(getattr(args, "red_easy_defend", False)),
            bool(getattr(args, "red_easy_combined", False)),
            bool(getattr(args, "red_easy_random", False)),
            bool(getattr(args, "red_medium_attack", False)),
            bool(getattr(args, "red_medium_defend", False)),
            bool(getattr(args, "red_medium_combined", False)),
            bool(getattr(args, "red_medium_random", False)),
            bool(getattr(args, "red_hard_attack", False)),
            bool(getattr(args, "red_hard_defend", False)),
            bool(getattr(args, "red_hard_combined", False)),
            bool(getattr(args, "red_hard_random", False)),
        ]
    )
    red_mode_count = sum(
        [
            bool(args.red_heuristic),
            _red_new_heur,
            bool(args.red_dummy),
            bool(args.red_stationary) and not _sta_block,
            _sta_block,
            bool(args.red_attack_hard),
            bool(args.red_all_attack),
            bool(args.red_all_defend),
            bool(args.red_from_checkpoint),
        ]
    )
    if red_mode_count > 1:
        raise SystemExit(
            "Use only one of: --red-heuristic, --red-dummy, --red-stationary, --red-stationary-midfield, "
            "--red-stationary-topfield, --red-stationary-bottomfield, --red-stationary-block-random, "
            "--red-attack-hard, --red-all-attack, --red-all-defend, --red-from-checkpoint, "
            "--red-easy-attack/--red-easy-defend/--red-easy-combined, "
            "--red-medium-attack/--red-medium-defend/--red-medium-combined, "
            "--red-hard-attack/--red-hard-defend/--red-hard-combined."
        )

    if args.watch:
        _run_watch(args)
        return

    # Out-dir: use parent of resume path if resuming and out-dir not explicitly set
    if args.resume and os.path.abspath(args.out_dir) == os.path.abspath(TRAINING_OUT_DIR):
        args.out_dir = os.path.dirname(os.path.normpath(os.path.abspath(args.resume))) or "."
    os.makedirs(args.out_dir, exist_ok=True)

    # Log file: same messages to stdout and to out_dir/train.log (unless --no-log-file)
    log_file = None
    if not args.no_log_file:
        log_path = os.path.join(args.out_dir, "train.log")
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except OSError:
            log_file = None

    def log(msg):
        print(msg)
        if log_file is not None:
            try:
                log_file.write(msg + "\n")
                log_file.flush()
            except OSError:
                pass

    logging.basicConfig(level=logging.ERROR)
    ray.init(ignore_reinit_error=True)

    RENDER = "human" if args.render else None
    SPEEDUP = max(1, int(args.speedup))

    team_size_range = (team_min, team_max)
    BLUE_AGENT_IDS = [f"agent_{i}" for i in range(team_max)]
    RED_AGENT_IDS = [f"agent_{i}" for i in range(team_max, 2 * team_max)]
    reinf_interval = max(0, int(args.reinforcement_interval))
    reinf_prob = max(0.0, min(1.0, args.reinforcement_prob))
    dt_interval = max(0, int(getattr(args, "dynamic_toggle_interval", 200)))
    dt_rm = max(0.0, min(1.0, float(getattr(args, "dynamic_toggle_remove_prob", 0.5))))
    dt_add = max(0.0, min(1.0, float(getattr(args, "dynamic_toggle_add_prob", 0.5))))
    train_bs = (
        int(args.train_batch_size)
        if args.train_batch_size > 0
        else (500 if args.render else DEFAULT_HEADLESS_TRAIN_BATCH_SIZE)
    )
    if args.envs_per_runner < 1:
        raise SystemExit("--envs-per-runner must be >= 1.")

    def env_creator(cfg=None):
        return make_env(
            cfg,
            render_mode=RENDER,
            sim_speedup=SPEEDUP,
            red_gets_raw_obs=(
                args.red_heuristic
                or args.red_attack_hard
                or args.red_all_attack
                or args.red_all_defend
                or getattr(args, "red_easy_attack", False)
                or getattr(args, "red_easy_defend", False)
                or getattr(args, "red_easy_combined", False)
                or getattr(args, "red_medium_attack", False)
                or getattr(args, "red_medium_defend", False)
                or getattr(args, "red_medium_combined", False)
                or getattr(args, "red_hard_attack", False)
                or getattr(args, "red_hard_defend", False)
                or getattr(args, "red_hard_combined", False)
            ),
            red_dummy=args.red_dummy,
            stationary_red=args.stationary_red,
            red_stationary=args.red_stationary,
            red_attack_hard=args.red_attack_hard,
            red_all_attack=args.red_all_attack,
            red_all_defend=args.red_all_defend,
            max_time=args.max_time,
            max_score=args.max_score,
            score_ends_episode=not args.no_score_end,
            team_size_range=team_size_range,
            tag_removes_agent=args.tag_removes_agent,
            reinforcement_interval=reinf_interval,
            reinforcement_prob=reinf_prob,
            dynamic_toggle_on=args.dynamic_toggle_on,
            dynamic_toggle_interval=dt_interval,
            dynamic_toggle_remove_prob=dt_rm,
            dynamic_toggle_add_prob=dt_add,
            fixed_spawn=args.fixed_spawn,
            stationary_red_active=args.stationary_red_active,
            stationary_red_active_random=args.stationary_red_active_random,
            stationary_red_random_min=args.stationary_red_random_min,
            stationary_red_random_max=args.stationary_red_random_max,
            stationary_red_block_anchor=getattr(args, "red_stationary_block_anchor", None),
            stationary_red_block_anchor_random=getattr(args, "red_stationary_block_anchor_random", False),
        )

    register_env("dynamic_pyquaticus", env_creator)
    env = make_env(
        render_mode=RENDER,
        sim_speedup=SPEEDUP,
        red_gets_raw_obs=(
            args.red_heuristic
            or args.red_attack_hard
            or args.red_all_attack
            or args.red_all_defend
            or getattr(args, "red_easy_attack", False)
            or getattr(args, "red_easy_defend", False)
            or getattr(args, "red_easy_combined", False)
            or getattr(args, "red_medium_attack", False)
            or getattr(args, "red_medium_defend", False)
            or getattr(args, "red_medium_combined", False)
            or getattr(args, "red_hard_attack", False)
            or getattr(args, "red_hard_defend", False)
            or getattr(args, "red_hard_combined", False)
        ),
        red_dummy=args.red_dummy,
        stationary_red=args.stationary_red,
        red_stationary=args.red_stationary,
        red_attack_hard=args.red_attack_hard,
        red_all_attack=args.red_all_attack,
        red_all_defend=args.red_all_defend,
        max_time=args.max_time,
        max_score=args.max_score,
        score_ends_episode=not args.no_score_end,
        team_size_range=team_size_range,
        tag_removes_agent=args.tag_removes_agent,
        reinforcement_interval=reinf_interval,
        reinforcement_prob=reinf_prob,
        dynamic_toggle_on=args.dynamic_toggle_on,
        dynamic_toggle_interval=dt_interval,
        dynamic_toggle_remove_prob=dt_rm,
        dynamic_toggle_add_prob=dt_add,
        fixed_spawn=args.fixed_spawn,
        stationary_red_active=args.stationary_red_active,
        stationary_red_active_random=args.stationary_red_active_random,
        stationary_red_random_min=args.stationary_red_random_min,
        stationary_red_random_max=args.stationary_red_random_max,
        stationary_red_block_anchor=getattr(args, "red_stationary_block_anchor", None),
        stationary_red_block_anchor_random=getattr(args, "red_stationary_block_anchor_random", False),
    )
    # Reset to ensure agents are initialized
    obs, info = env.reset()
    # Get spaces - Blue uses graph obs, Red uses raw obs when --red-heuristic
    agent_id_blue = "agent_0"
    agent_id_red = f"agent_{team_max}"
    if hasattr(env, "observation_space") and callable(env.observation_space):
        obs_space_blue = env.observation_space(agent_id_blue)
        obs_space_red = env.observation_space(agent_id_red)
    elif hasattr(env, "observation_spaces") and isinstance(env.observation_spaces, dict):
        obs_space_blue = env.observation_spaces[agent_id_blue]
        obs_space_red = env.observation_spaces.get(agent_id_red, env.observation_spaces[agent_id_blue])
    else:
        par_env = getattr(env, "par_env", env)
        if hasattr(par_env, "observation_space") and callable(par_env.observation_space):
            obs_space_blue = par_env.observation_space(agent_id_blue)
            obs_space_red = par_env.observation_space(agent_id_red)
        else:
            obs_space_blue = par_env.observation_spaces[agent_id_blue]
            obs_space_red = par_env.observation_spaces.get(agent_id_red, obs_space_blue)
    
    if hasattr(env, "action_space") and callable(env.action_space):
        act_space = env.action_space(agent_id_blue)
    elif hasattr(env, "action_spaces") and isinstance(env.action_spaces, dict):
        act_space = env.action_spaces[agent_id_blue]
    else:
        par_env = getattr(env, "par_env", env)
        if hasattr(par_env, "action_space") and callable(par_env.action_space):
            act_space = par_env.action_space(agent_id_blue)
        else:
            act_space = par_env.action_spaces[agent_id_blue]
    # Base env (for heuristic Red) = innermost PyQuaticus env, before env.close()
    base_env = (
        getattr(getattr(env, "par_env", env), "par_env", getattr(env, "par_env", env))
        if (
            args.red_heuristic
            or args.red_attack_hard
            or args.red_all_attack
            or args.red_all_defend
            or getattr(args, "red_easy_attack", False)
            or getattr(args, "red_easy_defend", False)
            or getattr(args, "red_easy_combined", False)
            or getattr(args, "red_easy_random", False)
            or getattr(args, "red_medium_attack", False)
            or getattr(args, "red_medium_defend", False)
            or getattr(args, "red_medium_combined", False)
            or getattr(args, "red_medium_random", False)
            or getattr(args, "red_hard_attack", False)
            or getattr(args, "red_hard_defend", False)
            or getattr(args, "red_hard_combined", False)
            or getattr(args, "red_hard_random", False)
        )
        else None
    )
    env.close()

    spawn_mode = "spawn_line (fixed)" if args.fixed_spawn else "random_on_own_side"
    log(
        f"Dynamic env: team_size={team_min}-{team_max} per team, init={spawn_mode}, "
        f"tag_removes_agent={args.tag_removes_agent}, reinforcement_interval={reinf_interval}, reinforcement_prob={reinf_prob}, "
        f"dynamic_toggle_on={args.dynamic_toggle_on}, dynamic_toggle_interval={dt_interval}, "
        f"dynamic_toggle_remove_prob={dt_rm}, dynamic_toggle_add_prob={dt_add}, "
        f"score_ends_episode={not args.no_score_end}, red_stationary={args.red_stationary}, "
        f"red_stationary_block_anchor={getattr(args, 'red_stationary_block_anchor', None)}, "
        f"red_stationary_block_anchor_random={getattr(args, 'red_stationary_block_anchor_random', False)}"
    )

    def _red_slot_id(agent_id_str: str) -> int:
        """Map global agent id (agent_6..agent_11) -> stable red slot 0..team_max-1."""
        try:
            n = int(agent_id_str.split("_", 1)[1])
        except Exception:
            return 0
        return int(n - team_max)

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in BLUE_AGENT_IDS:
            return "blue_policy"
        if _red_new_heur:
            if agent_id in RED_AGENT_IDS:
                slot = _red_slot_id(agent_id)
                # Randomize among attack/defend/combined per episode if requested.
                if getattr(args, "red_easy_random", False) or getattr(args, "red_medium_random", False) or getattr(args, "red_hard_random", False):
                    if episode is not None:
                        ud = getattr(episode, "user_data", None)
                        if isinstance(ud, dict):
                            key = "red_variant"
                            if key not in ud:
                                if getattr(args, "red_easy_random", False):
                                    ud[key] = random.choice(["easy_attack", "easy_defend", "easy_combined"])
                                elif getattr(args, "red_medium_random", False):
                                    ud[key] = random.choice(["medium_attack", "medium_defend", "medium_combined"])
                                else:
                                    ud[key] = random.choice(["hard_attack", "hard_defend", "hard_combined"])
                            variant = ud[key]
                        else:
                            variant = "easy_attack"
                    else:
                        variant = "easy_attack"
                    tier, sub = variant.split("_", 1)
                    pid = f"red_{tier}_{sub}_{slot}"
                    # Guard against mismatches between mapping and worker policy map.
                    try:
                        pm = getattr(worker, "policy_map", None)
                        keys = list(pm.keys()) if pm is not None else []
                    except Exception:
                        keys = []
                    if keys and pid not in keys:
                        pref = f"red_{tier}_{sub}_"
                        for k in keys:
                            if isinstance(k, str) and k.startswith(pref):
                                pid = k
                                break
                    return pid
                pid = f"red_heuristic_{slot}"
                try:
                    pm = getattr(worker, "policy_map", None)
                    keys = list(pm.keys()) if pm is not None else []
                except Exception:
                    keys = []
                if keys and pid not in keys:
                    pid = "red_heuristic_0"
                return pid
        if args.red_heuristic:
            if agent_id in RED_AGENT_IDS:
                slot = _red_slot_id(agent_id)
                return f"red_policy_{slot}"
        if args.red_all_attack:
            if agent_id in RED_AGENT_IDS:
                slot = _red_slot_id(agent_id)
                return f"red_attack_{slot}"
        if args.red_all_defend:
            if agent_id in RED_AGENT_IDS:
                slot = _red_slot_id(agent_id)
                return f"red_defend_{slot}"
        if args.red_attack_hard:
            if agent_id in RED_AGENT_IDS:
                slot = _red_slot_id(agent_id)
                return f"red_attack_{slot}"
        if args.red_dummy or args.red_stationary:
            return "red_dummy_policy"
        if args.red_from_checkpoint:
            return "red_prev_policy"
        return "red_policy"

    if _red_new_heur:
        from pyquaticus.base_policies.base_policy_wrappers import (
            EasyAttackGen,
            EasyDefendGen,
            EasyCombinedGen,
            MediumAttackGen,
            MediumDefendGen,
            MediumCombinedGen,
            HardAttackGen,
            HardDefendGen,
            HardCombinedGen,
        )
        policies = {"blue_policy": (None, obs_space_blue, act_space, {})}
        # For random modes, create all 3 variants and pick per-episode in policy_mapping_fn.
        if getattr(args, "red_easy_random", False):
            desc = "easy_random"
            for aid in RED_AGENT_IDS:
                slot = _red_slot_id(aid)
                for sub, gen in (("attack", EasyAttackGen), ("defend", EasyDefendGen), ("combined", EasyCombinedGen)):
                    rp = gen(aid, base_env)
                    rp.__name__ = f"RedHeuristic_easy_{sub}_{slot}"
                    if POLICIES is not None:
                        POLICIES[f"RedHeuristic_easy_{sub}_{slot}"] = rp
                    policies[f"red_easy_{sub}_{slot}"] = (rp, obs_space_red, act_space, {})
        elif getattr(args, "red_medium_random", False):
            desc = "medium_random"
            for aid in RED_AGENT_IDS:
                slot = _red_slot_id(aid)
                for sub, gen in (("attack", MediumAttackGen), ("defend", MediumDefendGen), ("combined", MediumCombinedGen)):
                    rp = gen(aid, base_env)
                    rp.__name__ = f"RedHeuristic_medium_{sub}_{slot}"
                    if POLICIES is not None:
                        POLICIES[f"RedHeuristic_medium_{sub}_{slot}"] = rp
                    policies[f"red_medium_{sub}_{slot}"] = (rp, obs_space_red, act_space, {})
        elif getattr(args, "red_hard_random", False):
            desc = "hard_random"
            for aid in RED_AGENT_IDS:
                slot = _red_slot_id(aid)
                for sub, gen in (("attack", HardAttackGen), ("defend", HardDefendGen), ("combined", HardCombinedGen)):
                    rp = gen(aid, base_env)
                    rp.__name__ = f"RedHeuristic_hard_{sub}_{slot}"
                    if POLICIES is not None:
                        POLICIES[f"RedHeuristic_hard_{sub}_{slot}"] = rp
                    policies[f"red_hard_{sub}_{slot}"] = (rp, obs_space_red, act_space, {})
        else:
            if getattr(args, "red_easy_attack", False):
                gen = EasyAttackGen
                desc = "easy_attack"
            elif getattr(args, "red_easy_defend", False):
                gen = EasyDefendGen
                desc = "easy_defend"
            elif getattr(args, "red_easy_combined", False):
                gen = EasyCombinedGen
                desc = "easy_combined"
            elif getattr(args, "red_medium_attack", False):
                gen = MediumAttackGen
                desc = "medium_attack"
            elif getattr(args, "red_medium_defend", False):
                gen = MediumDefendGen
                desc = "medium_defend"
            elif getattr(args, "red_medium_combined", False):
                gen = MediumCombinedGen
                desc = "medium_combined"
            elif getattr(args, "red_hard_attack", False):
                gen = HardAttackGen
                desc = "hard_attack"
            elif getattr(args, "red_hard_defend", False):
                gen = HardDefendGen
                desc = "hard_defend"
            elif getattr(args, "red_hard_combined", False):
                gen = HardCombinedGen
                desc = "hard_combined"
            else:
                raise SystemExit("Internal: _red_new_heur true but no specific flag set.")
        for aid in RED_AGENT_IDS:
            slot = _red_slot_id(aid)
            if getattr(args, "red_easy_random", False) or getattr(args, "red_medium_random", False) or getattr(args, "red_hard_random", False):
                # policy_mapping_fn will route to one of red_{tier}_{sub}_{n}
                continue
            rp = gen(aid, base_env)
            rp.__name__ = f"RedHeuristic_{desc}_{slot}"
            if POLICIES is not None:
                POLICIES[f"RedHeuristic_{desc}_{slot}"] = rp
            policies[f"red_heuristic_{slot}"] = (rp, obs_space_red, act_space, {})
        log(f"Red team using generator heuristic: {desc}.")
    elif args.red_heuristic:
        from pyquaticus.base_policies.base_policy_wrappers import CombinedGen
        mode = args.red_heuristic_mode
        policies = {"blue_policy": (None, obs_space_blue, act_space, {})}
        for aid in RED_AGENT_IDS:
            n = int(aid.split("_", 1)[1])
            rp = CombinedGen(aid, base_env, mode)
            rp.__name__ = f"HeuristicRed_{n}"
            if POLICIES is not None:
                POLICIES[f"HeuristicRed_{n}"] = rp
            policies[f"red_policy_{n}"] = (rp, obs_space_red, act_space, {})
        log(f"Red team using built-in heuristic (combined CTF, {mode} mode).")
    elif args.red_all_attack:
        from pyquaticus.base_policies.base_policy_wrappers import AttackGen
        policies = {"blue_policy": (None, obs_space_blue, act_space, {})}
        for aid in RED_AGENT_IDS:
            n = int(aid.split("_", 1)[1])
            policies[f"red_attack_{n}"] = (AttackGen(aid, base_env, "hard"), obs_space_red, act_space, {})
        log("Red team using all-attack heuristic (AttackGen hard, all red agents).")
    elif args.red_all_defend:
        from pyquaticus.base_policies.base_policy_wrappers import DefendGen
        policies = {"blue_policy": (None, obs_space_blue, act_space, {})}
        for aid in RED_AGENT_IDS:
            n = int(aid.split("_", 1)[1])
            policies[f"red_defend_{n}"] = (DefendGen(aid, base_env, "hard"), obs_space_red, act_space, {})
        log("Red team using all-defend heuristic (DefendGen hard, all red agents).")
    elif args.red_attack_hard:
        from pyquaticus.base_policies.base_policy_wrappers import AttackGen
        policies = {"blue_policy": (None, obs_space_blue, act_space, {})}
        for aid in RED_AGENT_IDS:
            n = int(aid.split("_", 1)[1])
            atk = AttackGen(aid, base_env, "hard")
            atk.__name__ = f"HeuristicRedAttackHard_{n}"
            if POLICIES is not None:
                POLICIES[f"HeuristicRedAttackHard_{n}"] = atk
            policies[f"red_attack_{n}"] = (atk, obs_space_red, act_space, {})
        log("Red team using 1 hard attacker-only heuristic (other red slots disabled).")
    elif args.red_dummy:
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_dummy_policy": (DoNothingPolicy, obs_space_blue, act_space, {}),
        }
        log("Red team using do-nothing policy (always no-op actions).")
    elif args.red_stationary:
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_dummy_policy": (DoNothingPolicy, obs_space_blue, act_space, {}),
        }
        if getattr(args, "stationary_red_active_random", False):
            _rn = getattr(args, "stationary_red_random_min", None)
            _rx = getattr(args, "stationary_red_random_max", None)
            if _rn is not None and _rx is not None:
                _rng = f"{_rn}-{_rx}"
            else:
                _rng = f"{args.team_size_min}-{args.team_size_max}"
            log(
                f"Red team stationary (do-nothing): random active Red count each episode {_rng} "
                f"(Blue roster {args.team_size_min}-{args.team_size_max}; use --stationary-red-random-min/max for Red-only range)."
            )
        else:
            _n = int(args.stationary_red_active) if args.stationary_red_active is not None else 2
            if getattr(args, "red_stationary_block_anchor_random", False):
                log(
                    f"Red team stationary block-random (do-nothing): {_n} active Red agents; "
                    f"each episode picks center/upper/lower spawn-row pair uniformly (see --stationary-red-active)."
                )
            else:
                _ba = getattr(args, "red_stationary_block_anchor", None)
                if _ba == "midfield":
                    log(
                        f"Red team stationary at midfield (do-nothing): {_n} active Red agents "
                        f"(center spawn-row pair; see --stationary-red-active)."
                    )
                elif _ba == "topfield":
                    log(
                        f"Red team stationary at topfield (do-nothing): {_n} active Red agents "
                        f"(upper spawn-row pair; see --stationary-red-active)."
                    )
                elif _ba == "bottomfield":
                    log(
                        f"Red team stationary at bottomfield (do-nothing): {_n} active Red agents "
                        f"(lower spawn-row pair; see --stationary-red-active)."
                    )
                else:
                    log(
                        f"Red team stationary (do-nothing): {_n} active Red agents "
                        f"(env clamps to team size; see --stationary-red-active)."
                    )
    elif args.red_from_checkpoint:
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_prev_policy": (None, obs_space_blue, act_space, {}),
        }
        log(f"Red team using Blue policy from checkpoint: {args.red_from_checkpoint}")
    else:
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_policy": (RandPolicy, obs_space_blue, act_space, {}),
        }

    def _load_red_prev_weights(algo, red_ckpt_path):
        path = _resolve_blue_policy_path(red_ckpt_path)
        if not os.path.isdir(path):
            log(f"ERROR: Red-from-checkpoint path not found: {path}")
            ray.shutdown()
            raise SystemExit(1)
        prev_policy = Policy.from_checkpoint(path)
        algo.get_policy("red_prev_policy").set_weights(prev_policy.get_weights())
        log("Red opponent weights loaded from checkpoint.")

    # Build or load algorithm
    i_start = 0
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.isdir(resume_path):
            log(f"ERROR: Resume path is not a directory: {resume_path}")
            ray.shutdown()
            raise SystemExit(1)
        # Parse iteration from path (e.g. .../iter_1250 -> 1250)
        basename = os.path.basename(resume_path)
        m = re.match(r"iter_(\d+)$", basename)
        if not m:
            log(f"ERROR: Resume path should end with iter_N (e.g. iter_1250), got: {basename}")
            ray.shutdown()
            raise SystemExit(1)
        i_start = int(m.group(1)) + 1
        log(f"Resuming from {resume_path} (next iteration {i_start})")

        # When resuming with --render, build a fresh algo with 0 workers so we get one responsive window,
        # then restore the checkpoint into it. Otherwise we'd keep the checkpoint's worker count (e.g. 8).
        if args.render:
            num_runners = 0
            env_runner_kw = {"num_env_runners": num_runners, "num_cpus_per_env_runner": 0.25, "num_envs_per_env_runner": 1}
            log("Rendering with resume: using 0 remote env runners so one game window stays responsive.")
            ppo_config = (
                PPOConfig()
                .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                .environment(env="dynamic_pyquaticus")
                .callbacks(callbacks_class=DynamicPyQuaticusCallbacks)
                .env_runners(**env_runner_kw)
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=["blue_policy"],
                )
            ).training(
                model={
                    "custom_model": "gnn_model",
                    "custom_model_config": {"gnn_hidden": args.gnn_hidden, "gnn_layers": args.gnn_layers},
                },
                train_batch_size=train_bs,
                lr=args.lr,
                minibatch_size=512,
                num_epochs=10,
                entropy_coeff=args.entropy_coeff, # allows it to explore early during the traiing process - tismailw
            )
            algo = ppo_config.build_algo()
            if args.red_from_checkpoint:
                # Don't restore full checkpoint (it has different Red policy). Load Blue from resume, Red from red_from_checkpoint.
                blue_path = _resolve_blue_policy_path(resume_path)
                if not os.path.isdir(blue_path):
                    log(f"ERROR: Resume checkpoint has no blue_policy at {blue_path}")
                    ray.shutdown()
                    raise SystemExit(1)
                blue_src = Policy.from_checkpoint(blue_path)
                _warn_ckpt_gnn_mismatch(blue_src, args, log)
                algo.get_policy("blue_policy").set_weights(blue_src.get_weights())
                log("Blue weights restored from resume checkpoint.")
                _load_red_prev_weights(algo, args.red_from_checkpoint)
            else:
                if hasattr(algo, "restore_from_path"):
                    algo.restore_from_path(resume_path)
                else:
                    algo.restore(resume_path)
        else:
            # Build a fresh algo with current args (and callbacks), then load only Blue weights from checkpoint.
            num_runners = args.runners
            env_runner_kw = {"num_env_runners": num_runners, "num_cpus_per_env_runner": 0.25}
            if args.envs_per_runner > 1:
                env_runner_kw["num_envs_per_env_runner"] = int(args.envs_per_runner)
            ppo_config = (
                PPOConfig()
                .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                .environment(env="dynamic_pyquaticus")
                .callbacks(callbacks_class=DynamicPyQuaticusCallbacks)
                .env_runners(**env_runner_kw)
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=["blue_policy"],
                )
            ).training(
                model={
                    "custom_model": "gnn_model",
                    "custom_model_config": {"gnn_hidden": args.gnn_hidden, "gnn_layers": args.gnn_layers},
                },
                train_batch_size=train_bs,
                lr=args.lr,
                minibatch_size=512,
                num_epochs=10,
                entropy_coeff=args.entropy_coeff,
            )
            algo = ppo_config.build_algo()

            blue_path = _resolve_blue_policy_path(resume_path)
            if not os.path.isdir(blue_path):
                log(f"ERROR: Resume checkpoint has no blue_policy at {blue_path}")
                ray.shutdown()
                raise SystemExit(1)
            blue_src = Policy.from_checkpoint(blue_path)
            _warn_ckpt_gnn_mismatch(blue_src, args, log)
            algo.get_policy("blue_policy").set_weights(blue_src.get_weights())
            log("Blue weights restored from resume checkpoint.")

            if args.red_from_checkpoint:
                _load_red_prev_weights(algo, args.red_from_checkpoint)
        if args.red_heuristic:
            mode_str = getattr(args, "red_heuristic_mode", "easy")
        elif _red_new_heur:
            if getattr(args, "red_easy_attack", False):
                mode_str = "easy_attack"
            elif getattr(args, "red_easy_defend", False):
                mode_str = "easy_defend"
            elif getattr(args, "red_easy_combined", False):
                mode_str = "easy_combined"
            elif getattr(args, "red_easy_random", False):
                mode_str = "easy_random"
            elif getattr(args, "red_medium_attack", False):
                mode_str = "medium_attack"
            elif getattr(args, "red_medium_defend", False):
                mode_str = "medium_defend"
            elif getattr(args, "red_medium_combined", False):
                mode_str = "medium_combined"
            elif getattr(args, "red_medium_random", False):
                mode_str = "medium_random"
            elif getattr(args, "red_hard_attack", False):
                mode_str = "hard_attack"
            elif getattr(args, "red_hard_defend", False):
                mode_str = "hard_defend"
            elif getattr(args, "red_hard_combined", False):
                mode_str = "hard_combined"
            elif getattr(args, "red_hard_random", False):
                mode_str = "hard_random"
            else:
                mode_str = "heuristic_custom"
        elif args.red_dummy:
            mode_str = "dummy"
        elif getattr(args, "red_stationary_block_anchor_random", False):
            mode_str = "stationary_block_random"
        elif getattr(args, "red_stationary_block_anchor", None):
            mode_str = f"stationary_{args.red_stationary_block_anchor}"
        elif args.red_stationary:
            mode_str = "stationary"
        elif args.red_attack_hard:
            mode_str = "attack_hard"
        elif args.red_all_attack:
            mode_str = "all_attack"
        elif args.red_all_defend:
            mode_str = "all_defend"
        elif args.red_from_checkpoint:
            mode_str = "prev_checkpoint"
        else:
            mode_str = "random"
        log(f"Red opponent: {mode_str} (current args). Blue weights restored from checkpoint.")
    else:
        # With --render, env has pygame Surface which can't be pickled; use 0 remote workers so rollout runs in driver.
        num_runners = 0 if args.render else args.runners
        env_runner_kw = {"num_env_runners": num_runners, "num_cpus_per_env_runner": 0.25}
        if args.render:
            if args.runners != 0:
                log("Rendering enabled: using 0 remote env runners (pygame cannot be pickled for Ray workers).")
            env_runner_kw["num_envs_per_env_runner"] = 1  # only one game window when rendering
        elif args.envs_per_runner > 1:
            env_runner_kw["num_envs_per_env_runner"] = int(args.envs_per_runner)
        if args.render:
            log(
                f"Rendering: train_batch_size={train_bs} (use --train-batch-size to change); "
                "window will still freeze briefly each iteration during PPO update."
            )
        else:
            log(
                f"PPO train_batch_size={train_bs} (0=auto: {DEFAULT_HEADLESS_TRAIN_BATCH_SIZE} headless, 500 with --render); "
                f"envs_per_env_runner={env_runner_kw.get('num_envs_per_env_runner', 1)}."
            )
        ppo_config = (
            PPOConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env="dynamic_pyquaticus")
            .callbacks(callbacks_class=DynamicPyQuaticusCallbacks)
            .env_runners(**env_runner_kw)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["blue_policy"],
            )
        ).training(
            model={
                "custom_model": "gnn_model",
                "custom_model_config": {"gnn_hidden": args.gnn_hidden, "gnn_layers": args.gnn_layers},
            },
            train_batch_size=train_bs,
            lr=args.lr,
            minibatch_size=512,
            num_epochs=10,
            entropy_coeff=args.entropy_coeff, # allows it to explore early during the traiing process - tismailw
        )
        algo = ppo_config.build_algo()
        if args.red_from_checkpoint:
            _load_red_prev_weights(algo, args.red_from_checkpoint)
        log("Training started (new run).")

    setattr(algo, "_matchup_save_every", max(1, int(args.save_every)))
    try:
        for i in range(i_start, args.iters + 1):
            start = time.time()
            try:
                save_now_file = os.path.join(args.out_dir, "SAVE_NOW")
                setattr(algo, "_matchup_force_print", os.path.isfile(save_now_file))
                setattr(algo, "_matchup_train_loop_i", i)
                result = algo.train()
                elapsed = time.time() - start
                ep_rew = result.get("env_runners", {}).get("episode_return_mean", 0)
                if ep_rew is None or (isinstance(ep_rew, float) and not np.isfinite(ep_rew)):
                    ep_rew_str = "n/a (no completed episodes yet)"
                else:
                    ep_rew_str = f"{float(ep_rew):.2f}"
                # Print every 25 iters (and iter 0)
                if i % args.save_every == 0 or i == 0:
                    entropy, policy_loss = _extract_blue_learner_metrics(result)
                    entropy_str = f"{entropy:.4f}" if entropy is not None else "n/a"
                    pol_str = f"{policy_loss:.4f}" if policy_loss is not None else "n/a"
                    log(
                        f"Iter {i}: return_mean={ep_rew_str}, entropy={entropy_str}, policy_loss={pol_str}, "
                        f"time={elapsed:.1f}s/iter (est. ~{50*elapsed:.0f}s per 50 iters)"
                    )
                if i > 0 and i % args.save_every == 0:
                    path = os.path.join(args.out_dir, f"iter_{i}")
                    algo.save(path)
                    log(f"Saved checkpoint to {path}")
                # "Save now" trigger: create out_dir/SAVE_NOW to save at end of current iter
                if os.path.isfile(save_now_file):
                    path = os.path.join(args.out_dir, f"iter_{i}")
                    algo.save(path)
                    log(f"Saved checkpoint (on request) to {path}")
                    try:
                        os.remove(save_now_file)
                    except OSError:
                        pass
            except Exception as e:
                log(f"ERROR at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                log("Check Ray worker logs for more details (RolloutWorker pid=... or: ray logs)")
                break
        else:
            log("Training complete.")
    finally:
        ray.shutdown()
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass


if __name__ == "__main__":
    main()
