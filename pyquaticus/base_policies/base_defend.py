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

import hashlib
from typing import Union

import numpy as np

from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.base_policies.utils import (dist_rel_bearing_to_local_rect,
                                            get_avoid_vect,
                                            global_rect_to_abs_bearing,
                                            local_rect_to_rel_bearing,
                                            rel_bearing_to_local_unit_rect,
                                            unit_vect_between_points)
from pyquaticus.config import ACTION_MAP, config_dict_std
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180, closest_point_on_line, dist

MODES = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}
EASY_RANDOM_ACTION_PROB = 0.30
_PATROL_PHASE = {}


class BaseDefender(BaseAgentPolicy):
    """This is a Policy class that contains logic for defending the flag."""

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        flag_keepout: float = config_dict_std["flag_keepout"],
        catch_radius: float = config_dict_std["catch_radius"],
        continuous: bool = False,
        mode: str = "easy",
    ):
        super().__init__(agent_id, env)

        self.set_mode(mode)
        self.continuous = continuous
        self.flag_keepout = flag_keepout
        self.catch_radius = catch_radius
        self.goal = "PM"
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

        if self.mode == "nothing":
            return self.action_from_vector(None, 0)

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        # Restore the classic (old) defender behavior; keep new actionmap through
        # our current action_from_vector/discrete_action_from_rel_bearing.
        if self.mode == "easy":
            # Easy patrol generally in front of the flag on our side.
            my_pos = np.asarray(global_state[(self.id, "pos")], dtype=np.float64)
            my_heading = float(global_state[(self.id, "heading")])
            team_str = self.team.name.lower().split("_")[0]
            flag_pos = np.asarray(global_state[team_str + "_flag_pos"], dtype=np.float64)
            opp_str = "red" if team_str == "blue" else "blue"
            my_flag_home = np.asarray(global_state[team_str + "_flag_home"], dtype=np.float64)
            opp_flag_home = np.asarray(global_state[opp_str + "_flag_home"], dtype=np.float64)
            scrimmage_x = float((my_flag_home[0] + opp_flag_home[0]) / 2.0)
            on_own_side = (my_pos[0] <= scrimmage_x) if team_str == "blue" else (my_pos[0] >= scrimmage_x)

            def _to_local(target_global: np.ndarray) -> np.ndarray:
                delta = np.asarray(target_global, dtype=np.float64) - my_pos
                d = float(np.linalg.norm(delta))
                abs_b = global_rect_to_abs_bearing(delta)
                rel_b = angle180(abs_b - my_heading)
                return np.asarray(dist_rel_bearing_to_local_rect(d, rel_b), dtype=np.float64)

            # Never pursue across scrimmage; if we drift across, return home-side.
            if not on_own_side:
                return self.action_from_vector(_to_local(my_flag_home), 0.7)

            # Chase nearest untagged opponent if close enough.
            nearest_local = None
            nearest_dist = float("inf")
            for enem, pos in getattr(self, "opp_team_pos_dict", {}).items():
                is_tagged = bool(global_state.get((enem, "is_tagged"), False))
                on_our_side = float(global_state.get((enem, "on_side"), 1.0)) == 0.0
                if is_tagged:
                    continue
                if not on_our_side:
                    continue
                d = float(pos[0])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_local = np.asarray(
                        dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64
                    )

            # Priority 1: chase flag carrier only while carrier is still on our side.
            if self.opp_team_has_flag:
                carrier_on_our_side = False
                for enem in getattr(self, "opp_team_pos_dict", {}).keys():
                    if not bool(global_state.get((enem, "has_flag"), False)):
                        continue
                    if float(global_state.get((enem, "on_side"), 1.0)) == 0.0:
                        carrier_on_our_side = True
                        break
                if carrier_on_our_side:
                    return self.action_from_vector(self.my_flag_loc, 0.7)
            # Priority 2: chase nearest enemy if close
            if nearest_local is not None and nearest_dist <= 35.0:
                return self.action_from_vector(nearest_local, 0.7)
            # Otherwise patrol

            team_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16)
            side_sign = 1.0 if team_str == "blue" else -1.0
            side_span = abs(scrimmage_x - float(my_flag_home[0]))
            patrol_x = float(my_flag_home[0] + side_sign * min(0.96 * side_span, 6.0 * float(self.flag_keepout)))
            half_height = float(self.flag_keepout) * 7.0
            lane_scale = 1.0 + 0.18 * float(team_hash % 3)
            patrol_pts = (
                np.asarray([patrol_x, float(flag_pos[1] - lane_scale * half_height)], dtype=np.float64),
                np.asarray([patrol_x, float(flag_pos[1] + 0.25 * (team_hash % 2) * half_height)], dtype=np.float64),
                np.asarray([patrol_x, float(flag_pos[1] + lane_scale * half_height)], dtype=np.float64),
            )
            if team_str == "blue":
                patrol_pts = tuple(
                    np.asarray([min(pt[0], float(scrimmage_x - 3.0)), pt[1]], dtype=np.float64)
                    for pt in patrol_pts
                )
            else:
                patrol_pts = tuple(
                    np.asarray([max(pt[0], float(scrimmage_x + 3.0)), pt[1]], dtype=np.float64)
                    for pt in patrol_pts
                )
            phase = _PATROL_PHASE.get(self.id, 0) % 3
            start_idx = team_hash % 3
            patrol_target_global = patrol_pts[(start_idx + phase) % 3]
            patrol_target_local = _to_local(patrol_target_global)
            if float(np.linalg.norm(patrol_target_local)) < 6.0:
                phase = (phase + 1) % 3
                _PATROL_PHASE[self.id] = phase
                patrol_target_global = patrol_pts[(start_idx + phase) % 3]
                patrol_target_local = _to_local(patrol_target_global)
            else:
                _PATROL_PHASE[self.id] = phase
            my_action = patrol_target_local

            # Reduce overlap/clumping.
            if getattr(self, "my_team_pos", None):
                my_action = np.asarray(my_action, dtype=np.float64) + 0.75 * get_avoid_vect(
                    self.my_team_pos, avoid_threshold=20.0
                )
            # Wall avoidance.
            wall_pos = []
            for wd, wb in zip(self.wall_distances, self.wall_bearings):
                if float(wd) < 8.0 and (-90.0 < float(wb) < 90.0):
                    wall_pos.append((float(wd), float(wb)))
            if wall_pos:
                my_action = np.asarray(my_action, dtype=np.float64) + get_avoid_vect(wall_pos, avoid_threshold=8.0)

            return self.action_from_vector(my_action, 0.7)

        if self.mode == "competition_easy":
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
            if dist(estimated_position, self.aquaticus_field_points[value]) <= 2.5:
                if self.goal == "SM":
                    self.goal = "PM"
                else:
                    self.goal = "SM"

            return self.goal

        if self.mode == "competition_medium":
            # Leave competition medium behavior unchanged.
            assert self.aquaticus_field_points is not None

            my_flag_vec = rel_bearing_to_local_unit_rect(self.my_flag_bearing)

            # Check if opponents are on team's side
            min_enemy_distance = 1000.00
            enemy_dis_dict = {}
            closest_enemy = None
            enemy_loc = None
            for enem, pos in self.opp_team_pos_dict.items():
                enemy_dis_dict[enem] = pos[0]
                if (
                    pos[0] < min_enemy_distance
                    and not global_state[(enem, "is_tagged")]
                    and global_state[(enem, "on_side")] == 0
                ):
                    min_enemy_distance = pos[0]
                    closest_enemy = enem
                    enemy_loc = dist_rel_bearing_to_local_rect(pos[0], pos[1])

            if self.opp_team_has_flag:
                ag_vect = my_flag_vec
            elif closest_enemy is not None:
                ag_vect = enemy_loc
            else:
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
                point = "CH" if self.team == Team.RED_TEAM else "CHX"
                if (
                    dist(
                        estimated_position,
                        self.aquaticus_field_points[point],
                    )
                    <= 2.5
                ):
                    return self.action_from_vector(None, 0)
                return "CH"

            return self.action_from_vector(ag_vect, 1)

        if self.mode == "medium":
            # If opposing team has the flag, chase them (flag pos moves with carrier).
            # Medium patrol generally in front of the flag on our side.
            my_pos = np.asarray(global_state[(self.id, "pos")], dtype=np.float64)
            my_heading = float(global_state[(self.id, "heading")])
            team_str = self.team.name.lower().split("_")[0]
            flag_pos = np.asarray(global_state[team_str + "_flag_pos"], dtype=np.float64)
            opp_str = "red" if team_str == "blue" else "blue"
            my_flag_home = np.asarray(global_state[team_str + "_flag_home"], dtype=np.float64)
            opp_flag_home = np.asarray(global_state[opp_str + "_flag_home"], dtype=np.float64)
            scrimmage_x = float((my_flag_home[0] + opp_flag_home[0]) / 2.0)
            on_own_side = (my_pos[0] <= scrimmage_x) if team_str == "blue" else (my_pos[0] >= scrimmage_x)

            def _to_local(target_global: np.ndarray) -> np.ndarray:
                delta = np.asarray(target_global, dtype=np.float64) - my_pos
                d = float(np.linalg.norm(delta))
                abs_b = global_rect_to_abs_bearing(delta)
                rel_b = angle180(abs_b - my_heading)
                return np.asarray(dist_rel_bearing_to_local_rect(d, rel_b), dtype=np.float64)

            # Never pursue across scrimmage; if we drift across, return home-side.
            if not on_own_side:
                return self.action_from_vector(_to_local(my_flag_home), 0.9)

            if self.opp_team_has_flag:
                carrier_on_our_side = False
                for enem in getattr(self, "opp_team_pos_dict", {}).keys():
                    if not bool(global_state.get((enem, "has_flag"), False)):
                        continue
                    if float(global_state.get((enem, "on_side"), 1.0)) == 0.0:
                        carrier_on_our_side = True
                        break
                if carrier_on_our_side:
                    return self.action_from_vector(self.my_flag_loc, 1)

            nearest_local = None
            nearest_dist = float("inf")
            for enem, pos in getattr(self, "opp_team_pos_dict", {}).items():
                is_tagged = bool(global_state.get((enem, "is_tagged"), False))
                on_our_side = float(global_state.get((enem, "on_side"), 1.0)) == 0.0
                if is_tagged:
                    continue
                if not on_our_side:
                    continue
                d = float(pos[0])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_local = np.asarray(
                        dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64
                    )
            if nearest_local is not None and nearest_dist <= 60.0:
                return self.action_from_vector(nearest_local, 0.9)

            team_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16)
            side_sign = 1.0 if team_str == "blue" else -1.0
            side_span = abs(scrimmage_x - float(my_flag_home[0]))
            patrol_x = float(my_flag_home[0] + side_sign * min(0.90 * side_span, 4.8 * float(self.flag_keepout)))
            half_height = float(self.flag_keepout) * 8.0
            lane_scale = 1.0 + 0.22 * float(team_hash % 3)
            patrol_pts = (
                np.asarray([patrol_x, float(flag_pos[1] - lane_scale * half_height)], dtype=np.float64),
                np.asarray([patrol_x, float(flag_pos[1] + 0.30 * ((team_hash % 3) - 1) * half_height)], dtype=np.float64),
                np.asarray([patrol_x, float(flag_pos[1] + lane_scale * half_height)], dtype=np.float64),
            )
            if team_str == "blue":
                patrol_pts = tuple(
                    np.asarray([min(pt[0], float(scrimmage_x - 3.0)), pt[1]], dtype=np.float64)
                    for pt in patrol_pts
                )
            else:
                patrol_pts = tuple(
                    np.asarray([max(pt[0], float(scrimmage_x + 3.0)), pt[1]], dtype=np.float64)
                    for pt in patrol_pts
                )
            phase = _PATROL_PHASE.get(self.id, 0) % 3
            start_idx = team_hash % 3
            patrol_target_global = patrol_pts[(start_idx + phase) % 3]
            my_action = _to_local(patrol_target_global)
            if float(np.linalg.norm(np.asarray(my_action, dtype=np.float64))) < 7.0:
                phase = (phase + 1) % 3
                _PATROL_PHASE[self.id] = phase
                patrol_target_global = patrol_pts[(start_idx + phase) % 3]
                my_action = _to_local(patrol_target_global)
            else:
                _PATROL_PHASE[self.id] = phase

            # Reduce overlap/clumping.
            if getattr(self, "my_team_pos", None):
                my_action = np.asarray(my_action, dtype=np.float64) + 0.85 * get_avoid_vect(
                    self.my_team_pos, avoid_threshold=22.0
                )
            # Wall avoidance.
            wall_pos = []
            for wd, wb in zip(self.wall_distances, self.wall_bearings):
                if float(wd) < 8.0 and (-90.0 < float(wb) < 90.0):
                    wall_pos.append((float(wd), float(wb)))
            if wall_pos:
                my_action = np.asarray(my_action, dtype=np.float64) + get_avoid_vect(wall_pos, avoid_threshold=8.0)

            return self.action_from_vector(my_action, 0.9)

        # Hard mode: intercept when threatened; otherwise zone patrol on own side.
        team_str = self.team.name.lower().split("_")[0]
        opp_str = "red" if team_str == "blue" else "blue"
        my_pos = np.asarray(global_state[(self.id, "pos")], dtype=np.float64)
        my_heading = float(global_state[(self.id, "heading")])
        my_flag_home = np.asarray(global_state[team_str + "_flag_home"], dtype=np.float64)
        opp_flag_home = np.asarray(global_state[opp_str + "_flag_home"], dtype=np.float64)
        scrimmage_x = float((my_flag_home[0] + opp_flag_home[0]) / 2.0)
        on_own_side = (my_pos[0] <= scrimmage_x) if team_str == "blue" else (my_pos[0] >= scrimmage_x)

        def _to_local(target_global: np.ndarray) -> np.ndarray:
            delta = np.asarray(target_global, dtype=np.float64) - my_pos
            d = float(np.linalg.norm(delta))
            abs_b = global_rect_to_abs_bearing(delta)
            rel_b = angle180(abs_b - my_heading)
            return np.asarray(dist_rel_bearing_to_local_rect(d, rel_b), dtype=np.float64)

        # Prevent crossing the scrimmage line: if we drift to enemy side, redirect back to flag.
        if not on_own_side:
            return self.action_from_vector(np.asarray(self.my_flag_loc, dtype=np.float64), 1)

        wall_pos = []
        if self.wall_distances[0] < 7 and (-90 < self.wall_bearings[0] < 90):
            wall_pos.append((self.wall_distances[0], self.wall_bearings[0]))
        elif self.wall_distances[2] < 7 and (-90 < self.wall_bearings[2] < 90):
            wall_pos.append((self.wall_distances[2], self.wall_bearings[2]))
        if self.wall_distances[1] < 7 and (-90 < self.wall_bearings[1] < 90):
            wall_pos.append((self.wall_distances[1], self.wall_bearings[1]))
        elif self.wall_distances[3] < 7 and (-90 < self.wall_bearings[3] < 90):
            wall_pos.append((self.wall_distances[3], self.wall_bearings[3]))

        min_enemy_distance = 1000.00
        enemy_dis_dict = {}
        closest_enemy = None
        enemy_loc = np.asarray((0.0, 0.0), dtype=np.float64)
        for enem, pos in self.opp_team_pos_dict.items():
            enemy_dis_dict[enem] = pos[0]
            if (
                pos[0] < min_enemy_distance
                and not global_state[(enem, "is_tagged")]
            ):
                min_enemy_distance = pos[0]
                closest_enemy = enem
                enemy_loc = np.asarray(dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64)

        if closest_enemy is None and enemy_dis_dict:
            closest_enemy = min(enemy_dis_dict, key=enemy_dis_dict.__getitem__)
            pos = self.opp_team_pos_dict[closest_enemy]
            enemy_loc = np.asarray(dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64)

        # If their team has our flag, chase carrier only while it remains on our side.
        if self.opp_team_has_flag:
            carrier_on_our_side = False
            for enem in getattr(self, "opp_team_pos_dict", {}).keys():
                if not bool(global_state.get((enem, "has_flag"), False)):
                    continue
                if float(global_state.get((enem, "on_side"), 1.0)) == 0.0:
                    carrier_on_our_side = True
                    break
            if carrier_on_our_side:
                ag_vect = rel_bearing_to_local_unit_rect(self.my_flag_bearing)
            else:
                agent_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16) % 3
                team_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16)
                side_sign = 1.0 if team_str == "blue" else -1.0
                side_span = abs(scrimmage_x - float(my_flag_home[0]))
                patrol_depths = (
                    min(0.65 * side_span, 3.6 * float(self.flag_keepout)),
                    min(0.80 * side_span, 4.8 * float(self.flag_keepout)),
                    min(0.92 * side_span, 6.0 * float(self.flag_keepout)),
                )
                patrol_x = float(my_flag_home[0] + side_sign * patrol_depths[agent_hash])
                if team_str == "blue":
                    patrol_x = min(patrol_x, float(scrimmage_x - 3.0))
                else:
                    patrol_x = max(patrol_x, float(scrimmage_x + 3.0))
                half_height = float((3.8 + 1.0 * agent_hash) * self.flag_keepout)
                patrol_pts = (
                    np.asarray([patrol_x, float(my_flag_home[1] - half_height)], dtype=np.float64),
                    np.asarray([patrol_x, float(my_flag_home[1])], dtype=np.float64),
                    np.asarray([patrol_x, float(my_flag_home[1] + half_height)], dtype=np.float64),
                )
                phase = _PATROL_PHASE.get(self.id, 0) % 3
                start_idx = team_hash % 3
                patrol_choice = patrol_pts[(start_idx + phase) % 3]
                ag_vect = _to_local(patrol_choice)
                if float(np.linalg.norm(np.asarray(ag_vect, dtype=np.float64))) < 8.0:
                    phase = (phase + 1) % 3
                    _PATROL_PHASE[self.id] = phase
                    patrol_choice = patrol_pts[(start_idx + phase) % 3]
                    ag_vect = _to_local(patrol_choice)
                else:
                    _PATROL_PHASE[self.id] = phase
        else:
            # Threat detection: if nearest untagged enemy is close enough to matter, keep intercept logic.
            threat_range = 6.0 * float(self.flag_keepout)
            enemy_dist_2_flag = dist(np.array(self.my_flag_loc), enemy_loc)
            enemy_is_threat = (
                closest_enemy is not None
                and (not bool(global_state.get((closest_enemy, "is_tagged"), False)))
                and (enemy_dist_2_flag <= threat_range)
            )

            if enemy_is_threat:
                # Existing intercept logic (halfway between flag and enemy, clamped by perimeter).
                defense_perim = 5.0 * float(self.flag_keepout)
                unit_flag_enemy = unit_vect_between_points(np.array(self.my_flag_loc), enemy_loc)
                defend_pt = np.asarray(self.my_flag_loc, dtype=np.float64) + (enemy_dist_2_flag / 2.0) * unit_flag_enemy

                defend_pt_flag_dist = dist(defend_pt, np.array(self.my_flag_loc))
                unit_def_flag = unit_vect_between_points(np.array(self.my_flag_loc), defend_pt)

                if enemy_dist_2_flag > defense_perim:
                    if defend_pt_flag_dist > defense_perim:
                        ag_vect = np.asarray(
                            [
                                float(self.my_flag_loc[0]) + (unit_def_flag[0] * defense_perim),
                                float(self.my_flag_loc[1]) + (unit_def_flag[1] * defense_perim),
                            ],
                            dtype=np.float64,
                        )
                    else:
                        ag_vect = np.asarray(defend_pt, dtype=np.float64)
                else:
                    ag_vect = np.asarray(enemy_loc, dtype=np.float64)
            else:
                # Zone patrol generally in front of the flag on our side.
                agent_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16) % 3
                team_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16)
                side_sign = 1.0 if team_str == "blue" else -1.0
                side_span = abs(scrimmage_x - float(my_flag_home[0]))
                patrol_depths = (
                    min(0.65 * side_span, 3.6 * float(self.flag_keepout)),
                    min(0.80 * side_span, 4.8 * float(self.flag_keepout)),
                    min(0.92 * side_span, 6.0 * float(self.flag_keepout)),
                )
                patrol_x = float(my_flag_home[0] + side_sign * patrol_depths[agent_hash])
                if team_str == "blue":
                    patrol_x = min(patrol_x, float(scrimmage_x - 3.0))
                else:
                    patrol_x = max(patrol_x, float(scrimmage_x + 3.0))
                half_height = float((3.8 + 1.0 * agent_hash) * self.flag_keepout)
                lane_scale = 1.0 + 0.25 * float(team_hash % 3)
                patrol_pts = (
                    np.asarray([patrol_x, float(my_flag_home[1] - lane_scale * half_height)], dtype=np.float64),
                    np.asarray([patrol_x, float(my_flag_home[1] + 0.35 * ((team_hash % 3) - 1) * half_height)], dtype=np.float64),
                    np.asarray([patrol_x, float(my_flag_home[1] + lane_scale * half_height)], dtype=np.float64),
                )
                phase = _PATROL_PHASE.get(self.id, 0) % 3
                start_idx = team_hash % 3
                patrol_choice = patrol_pts[(start_idx + phase) % 3]
                ag_vect = _to_local(patrol_choice)
                if float(np.linalg.norm(np.asarray(ag_vect, dtype=np.float64))) < 8.0:
                    phase = (phase + 1) % 3
                    _PATROL_PHASE[self.id] = phase
                    patrol_choice = patrol_pts[(start_idx + phase) % 3]
                    ag_vect = _to_local(patrol_choice)
                else:
                    _PATROL_PHASE[self.id] = phase

        if len(wall_pos) > 0:
            ag_vect = np.asarray(ag_vect, dtype=np.float64) + get_avoid_vect(wall_pos)

        # Reduce overlap/clumping (all hard behaviors).
        if getattr(self, "my_team_pos", None):
            ag_vect = np.asarray(ag_vect, dtype=np.float64) + 0.95 * get_avoid_vect(
                self.my_team_pos, avoid_threshold=24.0
            )

        return self.action_from_vector(ag_vect, 1)

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

        self.is_tagged = global_state[(self.id, "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        team_str = self.team.name.lower().split("_")[0]

        self.my_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[team_str + "_flag_pos"]
        )
        self.my_flag_bearing = angle180(
            global_rect_to_abs_bearing(
                global_state[team_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.my_flag_loc = dist_rel_bearing_to_local_rect(
            self.my_flag_distance, self.my_flag_bearing
        )

        # Copy the polar positions of each agent, separated by team and get their tag status
        self.opp_team_pos = []
        self.my_team_pos = []
        self.opp_team_pos_dict = {}  # for labeling by agent_id
        self.opp_team_tag = []
        self.opp_team_has_flag = False
        for id in self.teammate_ids:
            if float(global_state.get((id, "is_disabled"), 0.0)) > 0.5:
                continue
            if id != self.id:
                distance = dist(
                    global_state[(self.id, "pos")], global_state[(id, "pos")]
                )
                bearing = angle180(
                    global_rect_to_abs_bearing(
                        global_state[(id, "pos")] - global_state[(self.id, "pos")]
                    )
                    - global_state[(self.id, "heading")]
                )
                self.my_team_pos.append(np.array((distance, bearing)))
        for id in self.opponent_ids:
            if float(global_state.get((id, "is_disabled"), 0.0)) > 0.5:
                continue
            distance = dist(global_state[(self.id, "pos")], global_state[(id, "pos")])
            bearing = angle180(
                global_rect_to_abs_bearing(
                    global_state[(id, "pos")] - global_state[(self.id, "pos")]
                )
                - global_state[(self.id, "heading")]
            )
            self.opp_team_pos.append(np.array((distance, bearing)))
            self.opp_team_has_flag = (
                self.opp_team_has_flag or global_state[(id, "has_flag")]
            )
            self.opp_team_tag.append(global_state[(id, "is_tagged")])
            self.opp_team_pos_dict[id] = np.array((distance, bearing))

        self.wall_distances = []
        self.wall_bearings = []
        for wall in self.walls:
            closest = closest_point_on_line(
                wall[0], wall[1], global_state[(self.id, "pos")]
            )
            self.wall_distances.append(dist(closest, global_state[(self.id, "pos")]))
            bearing = angle180(
                global_rect_to_abs_bearing(closest - global_state[(self.id, "pos")])
                - global_state[(self.id, "heading")]
            )
            self.wall_bearings.append(bearing)
