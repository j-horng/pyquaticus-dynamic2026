# SPDX-License-Identifier: BSD-3-Clause
"""Parameterized tests for Dynamic PyQuaticus (team sizes 1–6)."""

import argparse

import numpy as np

try:
    from pytest import mark as _pytest_mark

    _parametrize = _pytest_mark.parametrize
except ImportError:

    def _parametrize(*_a, **_kw):
        def _decorator(fn):
            return fn

        return _decorator


import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.config import ACTION_MAP, config_dict_std
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
from pyquaticus.envs.graph_obs_wrapper import GraphObsWrapper, MAX_AGENTS, NODE_FEAT_DIM
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

_NOOP = len(ACTION_MAP) - 1
_EXPECTED_FLAT_GRAPH_SHAPE = (MAX_AGENTS * NODE_FEAT_DIM + MAX_AGENTS,)


def _reward_config(n: int) -> dict:
    return {f"agent_{i}": rew.caps_and_grabs for i in range(2 * n)}


def _base_cfg():
    cfg = config_dict_std.copy()
    cfg["max_score"] = 2
    cfg["max_time"] = 60
    cfg["sim_speedup_factor"] = 4
    return cfg


@_parametrize("max_team_size", [1, 2, 3, 4, 5, 6])
def test_dynamic_env(max_team_size: int):
    cfg = _base_cfg()
    reward_config = _reward_config(max_team_size)
    env = DynamicPyQuaticusEnv(
        team_size_range=(1, max_team_size),
        tag_removes_agent=False,
        config_dict=cfg,
        reward_config=reward_config,
    )
    obs, info = env.reset(seed=42)
    try:
        assert len(obs) == 2 * max_team_size
        assert "num_blue_active" in info["agent_0"]
        assert "disabled_agents" in info["agent_0"]
        nb = info["agent_0"]["num_blue_active"]
        nr = info["agent_0"]["num_red_active"]
        assert 1 <= nb <= max_team_size
        assert 1 <= nr <= max_team_size
        for _ in range(10):
            actions = {aid: _NOOP for aid in env.agents}
            obs, rewards, term, trunc, info = env.step(actions)
    finally:
        env.close()


@_parametrize("n", [1, 2, 3, 4, 5, 6])
def test_fixed_team_size(n: int):
    cfg = _base_cfg()
    reward_config = _reward_config(n)
    env = DynamicPyQuaticusEnv(
        team_size_range=(n, n),
        tag_removes_agent=False,
        config_dict=cfg,
        reward_config=reward_config,
    )
    obs, info = env.reset(seed=42)
    try:
        assert len(obs) == 2 * n
        disabled = env.state["disabled_agents"]
        assert int(np.sum(~disabled)) == 2 * n
        assert info["agent_0"]["num_blue_active"] == n
        assert info["agent_0"]["num_red_active"] == n
    finally:
        env.close()


@_parametrize("max_team_size", [1, 2, 3, 4, 5, 6])
def test_graph_wrapper(max_team_size: int):
    cfg = _base_cfg()
    reward_config = _reward_config(max_team_size)
    env = DynamicPyQuaticusEnv(
        team_size_range=(1, max_team_size),
        tag_removes_agent=False,
        config_dict=cfg,
        reward_config=reward_config,
    )
    env = ParallelPettingZooWrapper(env)
    blue_ids = [f"agent_{i}" for i in range(max_team_size)]
    env = GraphObsWrapper(env, flatten_for_fc=True, blue_agent_ids=blue_ids)
    obs, info = env.reset(seed=42)
    try:
        for aid in obs:
            assert obs[aid].shape == _EXPECTED_FLAT_GRAPH_SHAPE
        for _ in range(10):
            actions = {aid: _NOOP for aid in env.agents}
            obs, rewards, term, trunc, info = env.step(actions)
            for aid in obs:
                assert obs[aid].shape == _EXPECTED_FLAT_GRAPH_SHAPE
    finally:
        env.close()


def test_heuristic_disabled_filter():
    """(1, 4) env: four blue vs two red active; heuristic sees two opponents."""
    max_team = 4
    cfg = _base_cfg()
    reward_config = _reward_config(max_team)
    env = DynamicPyQuaticusEnv(
        team_size_range=(1, max_team),
        tag_removes_agent=False,
        config_dict=cfg,
        reward_config=reward_config,
    )
    env.reset(seed=0)
    env._set_initial_disabled([0, 1, 2, 3], [4, 5])
    env.num_blue_active = 4
    env.num_red_active = 2
    env.state["num_blue_active"] = 4
    env.state["num_red_active"] = 2
    env.state["global_state_hist_buffer"] = np.array(
        env.state_hist_buffer_len * [env.state_to_global_state(env.normalize_state)]
    )
    obs_agent0 = env._history_to_obs("agent_0", "obs_hist_buffer")
    gs = env._history_to_state()
    disabled = env.state["disabled_agents"]
    nb = int(np.sum(~disabled[: env.num_blue]))
    nr = int(np.sum(~disabled[env.num_blue : env.num_agents]))
    info = {
        "agent_0": {
            "global_state": gs,
            "num_blue_active": nb,
            "num_red_active": nr,
            "disabled_agents": disabled,
        }
    }
    agent = Heuristic_CTF_Agent("agent_0", env, mode="easy")
    agent.compute_action(obs_agent0, info)
    assert len(agent.opp_team_pos) == 2, f"expected 2 active red opponents, got {len(agent.opp_team_pos)}"
    env.close()


def test_dynamic_toggle_remove_eliminated_only_and_min_one():
    """Removal only targets tagged agents; at least one Blue remains active."""
    cfg = _base_cfg()
    n = 2
    reward_config = _reward_config(n)
    env = DynamicPyQuaticusEnv(
        team_size_range=(n, n),
        tag_removes_agent=False,
        dynamic_toggle_on=True,
        dynamic_toggle_interval=999,
        dynamic_toggle_remove_prob=1.0,
        dynamic_toggle_add_prob=0.0,
        config_dict=cfg,
        reward_config=reward_config,
    )
    try:
        env.reset(seed=0)
        env.players["agent_0"].is_tagged = True
        env.state["agent_is_tagged"][0] = 1
        env._try_random_remove_eliminated()
        assert bool(env.state["disabled_agents"][0])
        assert int(np.sum(~env.state["disabled_agents"][: env.num_blue])) == 1
        env._try_random_remove_eliminated()
        assert int(np.sum(~env.state["disabled_agents"][: env.num_blue])) == 1
    finally:
        env.close()


def test_dynamic_toggle_no_remove_if_not_eliminated():
    cfg = _base_cfg()
    n = 2
    reward_config = _reward_config(n)
    env = DynamicPyQuaticusEnv(
        team_size_range=(n, n),
        tag_removes_agent=False,
        dynamic_toggle_on=True,
        config_dict=cfg,
        reward_config=reward_config,
    )
    try:
        env.reset(seed=0)
        assert int(np.sum(~env.state["disabled_agents"][: env.num_blue])) == n
        env._try_random_remove_eliminated()
        assert int(np.sum(~env.state["disabled_agents"][: env.num_blue])) == n
    finally:
        env.close()


def _run_cli(single_size: int | None):
    sizes = [single_size] if single_size is not None else [1, 2, 3, 4, 5, 6]
    for n in sizes:
        test_dynamic_env(n)
        test_fixed_team_size(n)
        test_graph_wrapper(n)
    test_heuristic_disabled_filter()
    test_dynamic_toggle_remove_eliminated_only_and_min_one()
    test_dynamic_toggle_no_remove_if_not_eliminated()
    print(f"All tests passed (sizes={sizes}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test DynamicPyQuaticusEnv (sizes 1–6 by default).")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="N",
        help="Run only for max team size N (1–6). Default: loop N=1..6.",
    )
    args = parser.parse_args()
    if args.size is not None and not (1 <= args.size <= 6):
        parser.error("--size must be between 1 and 6")
    _run_cli(args.size)
