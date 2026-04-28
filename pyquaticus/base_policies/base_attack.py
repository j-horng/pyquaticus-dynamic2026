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

from typing import Union

import numpy as np

from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.base_policies.utils import (dist_rel_bearing_to_local_rect,
                                            get_avoid_vect,
                                            global_rect_to_abs_bearing,
                                            local_rect_to_rel_bearing,
                                            rel_bearing_to_local_unit_rect)
from pyquaticus.config import ACTION_MAP
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180, closest_point_on_line, dist

MODES = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}
EASY_RANDOM_ACTION_PROB = 0.30


class BaseAttacker(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
        mode: str = "easy",
    ):
        super().__init__(agent_id, env)

        self.set_mode(mode)

        self.continuous = continuous
        self.goal = "SC"

        self.state_normalizer = env.global_state_normalizer
        self.walls = env._walls[self.team.value]
        self.max_speed = env.max_speeds[env.players[self.id].idx]

        if isinstance(env, PyQuaticusMoosBridge) or not env.gps_env:
            self.aquaticus_field_points = env.aquaticus_field_points

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in MODES:
            raise ValueError(f"mode {mode} not in set of valid modes: {MODES}")
        self.mode = mode

    def compute_action(self, obs, info: dict[str, dict]):
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and heading error.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """

        self.update_state(obs, info)

        # Preserve explicit no-op mode.
        if self.mode == "nothing":
            return self.action_from_vector(None, 0)

        # Treat competition_* as hard for this simplified "pure capture" attacker.
        effective_mode = self.mode if self.mode in ("easy", "medium", "hard") else "hard"

        if effective_mode == "easy" and np.random.random() < EASY_RANDOM_ACTION_PROB:
            if self.continuous:
                return (float(np.random.uniform(0.0, self.max_speed)), float(np.random.uniform(-180.0, 180.0)))
            return int(np.random.randint(0, len(ACTION_MAP)))

        # Pure capture logic:
        # - If carrying: always return home (full commitment).
        # - Otherwise: always attack toward opponent flag.
        if effective_mode == "easy":
            # Easy should be clearly beatable: slower and less committed "beeline" chasing.
            spd_flag, spd_home = 0.4, 0.5
        elif effective_mode == "medium":
            spd_flag, spd_home = 0.6, 0.6
        else:
            spd_flag, spd_home = 1.0, 1.0

        if self.has_flag:
            my_action = np.asarray(self.home_loc, dtype=np.float64)
            desired_speed = spd_home
        else:
            # In easy mode, sometimes "cruise" instead of hard-locking onto the flag each step.
            # This makes easy attackers less effective without making them look broken.
            if effective_mode == "easy" and np.random.random() < 0.50:
                my_action = rel_bearing_to_local_unit_rect(float(np.random.uniform(-60.0, 60.0)))
            else:
                my_action = np.asarray(self.opp_flag_loc, dtype=np.float64)
            desired_speed = spd_flag

        # In hard mode, add a small formation offset + teammate separation to reduce clumping
        # when multiple attackers head to the same objective.
        if effective_mode == "hard" and not self.has_flag:
            try:
                idx = int(str(self.id).split("_", 1)[1])
            except Exception:
                idx = 0
            try:
                base_bearing = float(local_rect_to_rel_bearing(my_action))
            except Exception:
                base_bearing = 0.0
            offset_bearing = [-30.0, 0.0, 30.0][idx % 3]
            my_action = rel_bearing_to_local_unit_rect(base_bearing + offset_bearing)
            if getattr(self, "my_team_pos", None):
                my_action = np.asarray(my_action, dtype=np.float64) + get_avoid_vect(self.my_team_pos, avoid_threshold=15.0)

        # Opponent avoidance in medium+hard (per spec).
        if effective_mode in ("medium", "hard"):
            my_action = rel_bearing_to_local_unit_rect(local_rect_to_rel_bearing(my_action)) + get_avoid_vect(
                self.opp_team_pos, avoid_threshold=30.0
            )

        # OOB avoidance for both (exact per-task snippet).
        wall_pos = []
        for wd, wb in zip(self.wall_distances, self.wall_bearings):
            if wd < 8 and (-90 < wb < 90):
                wall_pos.append((wd, wb))
        if wall_pos:
            avoid = get_avoid_vect(wall_pos, avoid_threshold=8.0)
            my_action = my_action + avoid

        return self.action_from_vector(my_action, desired_speed)

    def action_from_vector(self, vector, desired_speed_normalized):
        if desired_speed_normalized == 0 or vector is None:
            if self.continuous:
                return (0, 0)
            else:
                from pyquaticus.config import ACTION_MAP
                return len(ACTION_MAP) - 1
        rel_bearing = local_rect_to_rel_bearing(vector)
        if self.continuous:
            return (desired_speed_normalized * self.max_speed, rel_bearing)
        return self.discrete_action_from_rel_bearing(rel_bearing, desired_speed_normalized)

    def update_state(self, obs, info: dict[str, dict]) -> None:
        """
        Method to convert the gym obs and info into data more relative to the
        agent.

        Note: all rectangular positions are in the ego agent's local coordinate frame.
        Note: all bearings are relative, measured in degrees clockwise from the ego agent's heading.

        Args:
            obs: observation from gym
            info: info from gym
        """

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        my_pos = global_state[(self.id, "pos")]
        my_heading = global_state[(self.id, "heading")]

        self.has_flag = global_state[(self.id, "has_flag")]
        self.is_tagged = global_state[(self.id, "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        team_str = self.team.name.lower().split("_")[0]
        opp_str = "red" if team_str == "blue" else "blue"

        self.opp_flag_distance = dist(my_pos, global_state[opp_str + "_flag_pos"])
        self.opp_flag_bearing = angle180(
            global_rect_to_abs_bearing(global_state[opp_str + "_flag_pos"] - my_pos)
            - my_heading
        )
        self.opp_flag_loc = dist_rel_bearing_to_local_rect(
            self.opp_flag_distance, self.opp_flag_bearing
        )

        home_distance = dist(my_pos, global_state[team_str + "_flag_home"])
        self.home_bearing = angle180(
            global_rect_to_abs_bearing(global_state[team_str + "_flag_home"] - my_pos)
            - my_heading
        )
        self.home_loc = dist_rel_bearing_to_local_rect(home_distance, self.home_bearing)

        self.opp_team_pos = []
        self.my_team_has_flag = False
        self.my_team_pos = []
        for id in self.teammate_ids:
            if float(global_state.get((id, "is_disabled"), 0.0)) > 0.5:
                continue
            if id != self.id:
                self.my_team_has_flag = (
                    self.my_team_has_flag or global_state[(id, "has_flag")]
                )
                distance = dist(my_pos, global_state[(id, "pos")])
                bearing = angle180(
                    global_rect_to_abs_bearing(global_state[(id, "pos")] - my_pos) - my_heading
                )
                self.my_team_pos.append(np.array((distance, bearing)))
        for id in self.opponent_ids:
            if float(global_state.get((id, "is_disabled"), 0.0)) > 0.5:
                continue
            distance = dist(my_pos, global_state[(id, "pos")])
            bearing = angle180(
                global_rect_to_abs_bearing(global_state[(id, "pos")] - my_pos)
                - my_heading
            )
            self.opp_team_pos.append(np.array((distance, bearing)))

        self.wall_distances = []
        self.wall_bearings = []
        for wall in self.walls:
            closest = closest_point_on_line(wall[0], wall[1], my_pos)
            self.wall_distances.append(dist(closest, my_pos))
            bearing = angle180(
                global_rect_to_abs_bearing(closest - my_pos) - my_heading
            )
            self.wall_bearings.append(bearing)
