# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Union

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.config import ACTION_MAP
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge


class RandPolicy(Policy):
    """
    Example wrapper for training against a random policy.

    To use a base policy, insantiate it inside a wrapper like this,
    and call it from self.compute_actions

    See policies and policy_mapping_fn for how policies are associated
    with agents
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):

        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


class NoOp(Policy):
    """
    No-op policy - stays in place

    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.no_op_index = len(ACTION_MAP) - 1

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):

        return [self.no_op_index], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


def AttackGen(agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], mode: str):

    class AttackPolicy(Policy):
        """
        Creates an attacker policy
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseAttacker(agent_id, env, mode=mode)
            self.action_dict = {}
            # So RLlib includes infos in the inference batch (required for heuristic policies).
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            explore=None,
            timestep=None,
            **kwargs,
        ):

            if info_batch is None:
                raise Warning("Warning: Base policy requires info as well as obs")

            n = len(obs_batch)
            # info_batch from RLlib: list of dicts, numpy array of dicts, or dict of arrays
            if hasattr(info_batch, "items"):
                # dict of lists/arrays
                get_info_i = lambda i: {k: v[i] for k, v in info_batch.items()}
            else:
                # list or ndarray (one dict per batch row)
                get_info_i = lambda i: info_batch[i] if i < len(info_batch) else {}

            # Heuristic expects info[agent_id]["global_state"]; RLlib often passes per-agent info (no agent_id key)
            aid = self.policy.id

            def norm_info(raw):
                if raw is None:
                    return {}
                # Unwrap numpy 0-d array holding a dict
                if hasattr(raw, "item") and callable(raw.item):
                    raw = raw.item()
                if not isinstance(raw, dict):
                    return raw
                # RLlib sometimes passes infos as {0: per_agent_info} (integer-keyed)
                if list(raw.keys()) == [0] and isinstance(raw.get(0), dict):
                    raw = raw[0]
                if aid in raw:
                    return raw
                if "global_state" in raw:
                    return {aid: raw}
                return raw

            actions = []
            for i in range(n):
                raw = get_info_i(i)
                action = self.policy.compute_action(obs_batch[i], norm_info(raw))
                actions.append(action)
            return actions, [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass

    return AttackPolicy


def DefendGen(agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], mode: str):
    class DefendPolicy(Policy):
        """
        Creates a defender policy
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseDefender(agent_id, env, mode=mode)
            self.action_dict = {}

        def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            explore=None,
            timestep=None,
            **kwargs,
        ):

            if info_batch is None:
                raise Warning("Warning: Base policy requires info as well as obs")

            n = len(obs_batch)
            # info_batch from RLlib: list of dicts, numpy array of dicts, or dict of arrays
            if hasattr(info_batch, "items"):
                get_info_i = lambda i: {k: v[i] for k, v in info_batch.items()}
            else:
                get_info_i = lambda i: info_batch[i] if i < len(info_batch) else {}

            aid = self.policy.id

            def norm_info(raw):
                if raw is None:
                    return {}
                if hasattr(raw, "item") and callable(raw.item):
                    raw = raw.item()
                if not isinstance(raw, dict):
                    return raw
                if list(raw.keys()) == [0] and isinstance(raw.get(0), dict):
                    raw = raw[0]
                if aid in raw:
                    return raw
                if "global_state" in raw:
                    return {aid: raw}
                return raw

            actions = []
            for i in range(n):
                raw = get_info_i(i)
                action = self.policy.compute_action(obs_batch[i], norm_info(raw))
                actions.append(action)
            return actions, [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass

    return DefendPolicy


def CombinedGen(agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], mode: str):
    class CombinedPolicy(Policy):
        """
        Creates a combined (attacker and defender) policy
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = Heuristic_CTF_Agent(agent_id, env, mode=mode)
            self.action_dict = {}
            # So RLLib includes infos in the inference batch (required for heuristic)
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            explore=None,
            timestep=None,
            **kwargs,
        ):
            if info_batch is None:
                raise RuntimeError(
                    "Heuristic Red policy requires env infos. Ensure INFOS is included "
                    "in the policy input (view_requirements[INFOS].used_for_compute_actions=True)."
                )
            n = len(obs_batch)
            # info_batch from RLLib: list of dicts, numpy array of dicts, or dict of arrays
            if hasattr(info_batch, "items"):
                # dict of lists/arrays
                get_info_i = lambda i: {k: v[i] for k, v in info_batch.items()}
            else:
                # list or ndarray (one dict per batch row)
                get_info_i = lambda i: info_batch[i] if i < len(info_batch) else {}
            # Heuristic expects info[agent_id]["global_state"]; RLLib often passes per-agent info (no agent_id key)
            agent_id = self.policy.id
            def norm_info(raw):
                if raw is None:
                    return {}
                # Unwrap numpy 0-d array holding a dict
                if hasattr(raw, "item") and callable(raw.item):
                    raw = raw.item()
                if not isinstance(raw, dict):
                    return raw
                # RLLib sometimes passes infos as {0: per_agent_info} (integer-keyed)
                if list(raw.keys()) == [0] and isinstance(raw.get(0), dict):
                    raw = raw[0]
                if agent_id in raw:
                    return raw
                if "global_state" in raw:
                    return {agent_id: raw}
                return raw
            actions = []
            for i in range(n):
                raw = get_info_i(i)
                action = self.policy.compute_action(obs_batch[i], norm_info(raw))
                actions.append(action)
            return actions, [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass

    return CombinedPolicy


def _wrap_info_batch_for_policy(policy_obj, info_batch):
    """Normalize RLlib infos into the shape heuristic policies expect."""
    if info_batch is None:
        raise RuntimeError(
            "Heuristic policy requires env infos. Ensure INFOS is included "
            "in the policy input (view_requirements[INFOS].used_for_compute_actions=True)."
        )
    # info_batch from RLlib: list of dicts, numpy array of dicts, or dict of arrays
    if hasattr(info_batch, "items"):
        get_info_i = lambda i: {k: v[i] for k, v in info_batch.items()}
    else:
        get_info_i = lambda i: info_batch[i] if i < len(info_batch) else {}

    aid = policy_obj.id

    def norm_info(raw):
        if raw is None:
            return {}
        # Unwrap numpy 0-d array holding a dict
        if hasattr(raw, "item") and callable(raw.item):
            raw = raw.item()
        if not isinstance(raw, dict):
            return raw
        # RLlib sometimes passes infos as {0: per_agent_info} (integer-keyed)
        if list(raw.keys()) == [0] and isinstance(raw.get(0), dict):
            raw = raw[0]
        if aid in raw:
            return raw
        if "global_state" in raw:
            return {aid: raw}
        return raw

    return get_info_i, norm_info


def _maybe_apply_random_action(action_space, p_rand: float):
    import numpy as _np
    return action_space.sample() if (_np.random.random() < float(p_rand)) else None


def _remap_discrete_speed(policy_obj, act, desired_speed_frac: float):
    """Take a discrete act index, preserve heading, remap to desired speed under current ACTION_MAP."""
    if act is None:
        return act
    try:
        a = int(act)
    except Exception:
        return act
    # Keep no-op as no-op
    no_op = len(ACTION_MAP) - 1
    if a == no_op:
        return a
    # Preserve the heading of the chosen action, but pick a new index with desired speed.
    try:
        heading = float(ACTION_MAP[a][1])
    except Exception:
        return a
    return int(policy_obj.discrete_action_from_rel_bearing(heading, float(desired_speed_frac)))


def EasyAttackGen(agent_id, env, *args, **kwargs):
    """Easy Attack — BaseAttacker easy + slower speed + 30% random actions."""

    class EasyAttackPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseAttacker(agent_id, env, mode="easy")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                ra = _maybe_apply_random_action(self.action_space, 0.30)
                if ra is not None:
                    actions.append(ra)
                    continue
                act = self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i)))
                # Keep easy attackers slower (esp. under NRL full-speed-only map).
                act = _remap_discrete_speed(self.policy, act, 0.4)
                actions.append(act)
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return EasyAttackPolicy


def EasyDefendGen(agent_id, env, *args, **kwargs):
    """Easy Defend — BaseDefender easy + speed 0.4 + 30% random actions."""

    class EasyDefendPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseDefender(agent_id, env, mode="easy")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                ra = _maybe_apply_random_action(self.action_space, 0.30)
                if ra is not None:
                    actions.append(ra)
                    continue
                act = self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i)))
                act = _remap_discrete_speed(self.policy, act, 0.4)
                actions.append(act)
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return EasyDefendPolicy


def EasyCombinedGen(agent_id, env, *args, **kwargs):
    """Easy Combined — Heuristic_CTF_Agent easy (random+attack-only built in)."""

    class EasyCombinedPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = Heuristic_CTF_Agent(agent_id, env, mode="easy")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                actions.append(self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i))))
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return EasyCombinedPolicy


def MediumAttackGen(agent_id, env, *args, **kwargs):
    """Medium Attack — BaseAttacker medium + speed 0.6 + no randomness."""

    class MediumAttackPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseAttacker(agent_id, env, mode="medium")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                act = self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i)))
                act = _remap_discrete_speed(self.policy, act, 0.6)
                actions.append(act)
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return MediumAttackPolicy


def MediumDefendGen(agent_id, env, *args, **kwargs):
    """Medium Defend — BaseDefender medium + speed 0.6 + no randomness."""

    class MediumDefendPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseDefender(agent_id, env, mode="medium")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                act = self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i)))
                act = _remap_discrete_speed(self.policy, act, 0.6)
                actions.append(act)
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return MediumDefendPolicy


def MediumCombinedGen(agent_id, env, *args, **kwargs):
    """Medium Combined — Heuristic_CTF_Agent medium (70/30 mix built in)."""

    class MediumCombinedPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = Heuristic_CTF_Agent(agent_id, env, mode="medium")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                actions.append(self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i))))
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return MediumCombinedPolicy


def HardAttackGen(agent_id, env, *args, **kwargs):
    """Hard Attack — BaseAttacker hard + speed 1.0."""

    class HardAttackPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseAttacker(agent_id, env, mode="hard")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                act = self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i)))
                act = _remap_discrete_speed(self.policy, act, 1.0)
                actions.append(act)
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return HardAttackPolicy


def HardDefendGen(agent_id, env, *args, **kwargs):
    """Hard Defend — BaseDefender hard + speed 1.0."""

    class HardDefendPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseDefender(agent_id, env, mode="hard")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                act = self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i)))
                act = _remap_discrete_speed(self.policy, act, 1.0)
                actions.append(act)
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return HardDefendPolicy


def HardCombinedGen(agent_id, env, *args, **kwargs):
    """Hard Combined — Heuristic_CTF_Agent hard (intelligent switching built in)."""

    class HardCombinedPolicy(Policy):
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = Heuristic_CTF_Agent(agent_id, env, mode="hard")
            self.action_dict = {}
            if SampleBatch.INFOS in self.view_requirements:
                self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True

        def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                            info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
            get_info_i, norm_info = _wrap_info_batch_for_policy(self.policy, info_batch)
            actions = []
            for i in range(len(obs_batch)):
                actions.append(self.policy.compute_action(obs_batch[i], norm_info(get_info_i(i))))
            return actions, [], {}

        def get_weights(self): return {}
        def learn_on_batch(self, samples): return {}
        def set_weights(self, weights): pass

    return HardCombinedPolicy
