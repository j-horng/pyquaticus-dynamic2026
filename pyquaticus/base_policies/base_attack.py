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

        # Match the classic (old) attacker behavior, but keep the new actionmap
        # via our current action_from_vector/discrete_action_from_rel_bearing.
        try:
            idx = int(str(self.id).split("_", 1)[1])
        except Exception:
            idx = 0
        spread_angle = (-24.0, -8.0, 8.0, 24.0)[idx % 4]

        if self.mode == "easy":
            if self.has_flag:
                goal_vect = rel_bearing_to_local_unit_rect(self.home_bearing)
                avoid_vect = get_avoid_vect(self.opp_team_pos, avoid_threshold=20.0)
                my_action = goal_vect + 0.5 * avoid_vect
                return self.action_from_vector(my_action, 0.7)
            goal_vect = rel_bearing_to_local_unit_rect(self.opp_flag_bearing + spread_angle)
            avoid_vect = get_avoid_vect(self.opp_team_pos, avoid_threshold=15.0)
            my_action = goal_vect + 0.4 * avoid_vect
            return self.action_from_vector(my_action, 0.7)

        if self.mode == "competition_easy":
            # Keep competition behavior unchanged (MOOS/Aquaticus waypoint logic).
            assert self.aquaticus_field_points is not None

            if self.team == Team.RED_TEAM:
                estimated_position = np.asarray(
                    [
                        self.wall_distances[1],
                        self.wall_distances[0],
                    ]
                )
            else:
                estimated_position = np.asarray(
                    [
                        self.wall_distances[3],
                        self.wall_distances[2],
                    ]
                )

            value = self.goal

            if self.team == Team.BLUE_TEAM:
                if "P" in self.goal:
                    value = "S" + value[1:]
                elif "S" in self.goal:
                    value = "P" + value[1:]
                if "X" not in self.goal and self.goal not in ["SC", "CC", "PC"]:
                    value += "X"
                elif self.goal not in ["SC", "CC", "PC"]:
                    value = value[:-1]

            if self.is_tagged:
                self.goal = "SC"
            if (
                -2.5
                <= dist(estimated_position, self.aquaticus_field_points[value])
                <= 2.5
            ):
                if self.goal == "SC":
                    self.goal = "CFX"
                elif self.goal == "CFX":
                    self.goal = "PC"
                elif self.goal == "PC":
                    self.goal = "CF"
                elif self.goal == "CF":
                    self.goal = "SC"

            if (
                self.goal == "CF"
                and -6
                <= dist(estimated_position, self.aquaticus_field_points[value])
                <= 6
            ):
                self.goal = "SC"
            return self.goal

        if self.mode == "medium":
            if self.has_flag:
                goal_vect = 2.0 * rel_bearing_to_local_unit_rect(self.home_bearing)
                avoid_vect = get_avoid_vect(self.opp_team_pos, avoid_threshold=35.0)
                my_action = goal_vect + 1.2 * avoid_vect
            else:
                goal_vect = 2.0 * rel_bearing_to_local_unit_rect(self.opp_flag_bearing + spread_angle)
                avoid_vect = get_avoid_vect(self.opp_team_pos, avoid_threshold=25.0)
                my_action = goal_vect + 0.8 * avoid_vect
            return self.action_from_vector(my_action, 0.9)

        if self.mode == "competition_medium":
            # Keep competition behavior unchanged (classic wall-as-obstacle logic).
            if self.wall_distances[0] < 10 and (-90 < self.wall_bearings[0] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[0],
                        self.wall_bearings[0],
                    )
                )
            elif self.wall_distances[2] < 10 and (-90 < self.wall_bearings[2] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[2],
                        self.wall_bearings[2],
                    )
                )
            if self.wall_distances[1] < 10 and (-90 < self.wall_bearings[1] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[1],
                        self.wall_bearings[1],
                    )
                )
            elif self.wall_distances[3] < 10 and (-90 < self.wall_bearings[3] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[3],
                        self.wall_bearings[3],
                    )
                )

            avoid_thresh = 60.0

            if self.has_flag:
                goal_vect = 1.25 * rel_bearing_to_local_unit_rect(self.home_bearing)
                avoid_vect = get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                my_action = goal_vect + avoid_vect
            else:
                goal_vect = rel_bearing_to_local_unit_rect(self.opp_flag_bearing + spread_angle)
                avoid_vect = get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                if (not np.any(goal_vect + (avoid_vect))) or (
                    np.allclose(
                        np.abs(np.abs(goal_vect) - np.abs(avoid_vect)),
                        np.zeros(np.array(goal_vect).shape),
                        atol=1e-01,
                        rtol=1e-02,
                    )
                ):
                    top_dist = self.wall_distances[0]
                    bottom_dist = self.wall_distances[2]
                    if top_dist > 1.25 * bottom_dist:
                        my_action = dist_rel_bearing_to_local_rect(
                            top_dist, self.wall_bearings[0]
                        )
                    else:
                        my_action = dist_rel_bearing_to_local_rect(
                            bottom_dist, self.wall_bearings[2]
                        )
                else:
                    my_action = 1.25 * goal_vect + avoid_vect

            return self.action_from_vector(my_action, 1)

        # Hard (and any other remaining modes): classic wall + opponent avoidance.
        # Add nearby walls as obstacles.
        if self.wall_distances[0] < 10 and (-90 < self.wall_bearings[0] < 90):
            self.opp_team_pos.append((self.wall_distances[0], self.wall_bearings[0]))
        elif self.wall_distances[2] < 10 and (-90 < self.wall_bearings[2] < 90):
            self.opp_team_pos.append((self.wall_distances[2], self.wall_bearings[2]))
        if self.wall_distances[1] < 10 and (-90 < self.wall_bearings[1] < 90):
            self.opp_team_pos.append((self.wall_distances[1], self.wall_bearings[1]))
        elif self.wall_distances[3] < 10 and (-90 < self.wall_bearings[3] < 90):
            self.opp_team_pos.append((self.wall_distances[3], self.wall_bearings[3]))

        avoid_thresh = 30.0

        if self.has_flag:
            # I have flag — go home with avoidance
            goal_vect = 1.25 * rel_bearing_to_local_unit_rect(self.home_bearing)
            avoid_vect = get_avoid_vect(self.opp_team_pos, avoid_threshold=avoid_thresh)
            my_action = goal_vect + avoid_vect
        else:
            # Teammate has flag OR no flag — keep attacking opponent flag
            goal_vect = rel_bearing_to_local_unit_rect(self.opp_flag_bearing + spread_angle)
            avoid_vect = get_avoid_vect(self.opp_team_pos, avoid_threshold=avoid_thresh)
            if (not np.any(goal_vect + (avoid_vect))) or (
                np.allclose(
                    np.abs(np.abs(goal_vect) - np.abs(avoid_vect)),
                    np.zeros(np.array(goal_vect).shape),
                    atol=1e-01,
                    rtol=1e-02,
                )
            ):
                top_dist = self.wall_distances[0]
                bottom_dist = self.wall_distances[2]
                if top_dist > 1.25 * bottom_dist:
                    my_action = dist_rel_bearing_to_local_rect(
                        top_dist, self.wall_bearings[0]
                    )
                else:
                    my_action = dist_rel_bearing_to_local_rect(
                        bottom_dist, self.wall_bearings[2]
                    )
            else:
                my_action = np.multiply(1.25, goal_vect) + avoid_vect

        return self.action_from_vector(my_action, 1)

    def action_from_vector(self, vector, desired_speed_normalized):
        if desired_speed_normalized == 0 or vector is None:
            if self.continuous:
                return (0, 0)
            else:
                from pyquaticus.config import ACTION_MAP
                return len(ACTION_MAP) - 1
        vector = np.asarray(vector, dtype=np.float64)
        near_wall = []
        for wd, wb in zip(getattr(self, "wall_distances", []), getattr(self, "wall_bearings", [])):
            if float(wd) < 10.0 and (-100.0 < float(wb) < 100.0):
                near_wall.append((float(wd), float(wb)))
        if near_wall:
            wall_avoid = 1.5 * get_avoid_vect(near_wall, avoid_threshold=10.0)
            candidate = vector + wall_avoid
            if float(np.linalg.norm(candidate)) >= 3.0:
                vector = candidate
            else:
                vector = rel_bearing_to_local_unit_rect(local_rect_to_rel_bearing(wall_avoid))
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
