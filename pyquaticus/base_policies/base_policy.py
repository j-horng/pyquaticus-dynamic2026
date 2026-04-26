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

from typing import Any

import numpy as np

from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
# Discrete action layout may be patched at runtime (e.g. apply_nrl_action_map()).
from pyquaticus.config import ACTION_MAP
# from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge


class BaseAgentPolicy:
    """
    Parent class for all base policies.
    """

    def __init__(
        self,
        agent_id: str,
        env: PyQuaticusEnv,
        suppress_numpy_warnings=True,
    ):
        self.id = agent_id
        # Used to approximate "slow speed" when discrete action map has only one nonzero speed
        # (e.g., NRL full-speed-only). This avoids fully random stutter behavior.
        self._discrete_move_accum = 0.0

        if self.id in env.agent_ids_of_team[Team.BLUE_TEAM]:
            self.team = Team.BLUE_TEAM
            self.teammate_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
            self.opponent_ids = env.agent_ids_of_team[Team.RED_TEAM]
        elif self.id in env.agent_ids_of_team[Team.RED_TEAM]:
            self.team = Team.RED_TEAM
            self.teammate_ids = env.agent_ids_of_team[Team.RED_TEAM]
            self.opponent_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
        else:
            raise ValueError(f"{self.id} not on a team")

        if suppress_numpy_warnings:
            np.seterr(all="ignore")

    def compute_action(self, obs, info: dict[str, dict]) -> Any:
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and relative bearing.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        raise NotImplementedError

    @staticmethod
    def _wrap_angle180(deg: float) -> float:
        """Wrap degrees to [-180, 180]."""
        x = float(deg)
        return (x + 180.0) % 360.0 - 180.0

    def discrete_action_from_rel_bearing(self, rel_bearing_deg: float, desired_speed_frac: float) -> int:
        """
        Map a desired relative bearing + (normalized) speed request to a discrete action index.

        This *must* consult the current `ACTION_MAP` because the project supports multiple layouts:
        - legacy: multiple speeds and coarse headings
        - nrl: full-speed only, fine headings in [-10..10] plus a few wider turns
        """
        # Convention across this repo: last entry is no-op.
        no_op = len(ACTION_MAP) - 1
        if desired_speed_frac <= 0.0 or len(ACTION_MAP) <= 1:
            return int(no_op)

        # If the action map only contains a single nonzero speed (e.g. NRL: full-speed only),
        # approximate "slower" speeds by emitting no-op at a *controlled cadence*.
        # This preserves the requested difficulty spread even when discrete speed control is absent,
        # without making the policy look randomly indecisive.
        try:
            speeds = {float(s) for (s, _h) in ACTION_MAP[:-1]}
        except Exception:
            speeds = set()
        if len(speeds) == 1:
            only_spd = next(iter(speeds))
            if only_spd > 0.0 and desired_speed_frac < only_spd:
                p_move = max(0.0, min(1.0, float(desired_speed_frac) / only_spd))
                # Accumulator: add p_move each step; move whenever it crosses 1.0.
                # Example: p_move=0.5 => move every other step; p_move=0.75 => move 3 of 4 steps.
                self._discrete_move_accum = float(self._discrete_move_accum) + float(p_move)
                if self._discrete_move_accum < 1.0:
                    return int(no_op)
                self._discrete_move_accum -= 1.0

        target_h = self._wrap_angle180(rel_bearing_deg)
        best_i = 0
        best_score = float("inf")
        # Exclude no-op from matching.
        for i, (spd, hdg) in enumerate(ACTION_MAP[:-1]):
            dh = abs(self._wrap_angle180(float(hdg) - target_h)) / 180.0
            ds = abs(float(spd) - float(desired_speed_frac))
            # Heading dominates; speed tie-breaker for maps that include multiple speeds.
            score = dh + 0.25 * ds
            if score < best_score:
                best_score = score
                best_i = i
        return int(best_i)
