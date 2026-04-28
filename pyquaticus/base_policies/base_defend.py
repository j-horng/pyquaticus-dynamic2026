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

        # Keep competition_* behavior aligned with existing policy handling.
        effective_mode = self.mode if self.mode in ("easy", "medium", "hard") else "hard"

        if effective_mode == "easy" and np.random.random() < EASY_RANDOM_ACTION_PROB:
            if self.continuous:
                return (float(np.random.uniform(0.0, self.max_speed)), float(np.random.uniform(-180.0, 180.0)))
            return int(np.random.randint(0, len(ACTION_MAP)))

        if effective_mode == "easy":
            spd = 0.4
            carrier_chase_spd = 0.6
            react_thresh = 20.0
            carrier_thresh = 60.0
            zone_mod = 1
        elif effective_mode == "medium":
            spd = 0.6
            carrier_chase_spd = 0.85
            react_thresh = 80.0
            carrier_thresh = 100.0
            zone_mod = 2
        else:
            spd = 1.0
            carrier_chase_spd = 1.0
            react_thresh = float("inf")
            carrier_thresh = float("inf")
            zone_mod = 3

        team_str = self.team.name.lower().split("_")[0]
        opp_str = "red" if team_str == "blue" else "blue"
        my_flag_home = np.asarray(global_state[team_str + "_flag_home"], dtype=np.float64)
        opp_flag_home = np.asarray(global_state[opp_str + "_flag_home"], dtype=np.float64)
        scrimmage_x = float((my_flag_home[0] + opp_flag_home[0]) / 2.0)
        my_pos = np.asarray(global_state[(self.id, "pos")], dtype=np.float64)
        on_own_side = (my_pos[0] <= scrimmage_x) if team_str == "blue" else (my_pos[0] >= scrimmage_x)

        def _to_local(target_global: np.ndarray) -> np.ndarray:
            delta = np.asarray(target_global, dtype=np.float64) - my_pos
            d = float(np.linalg.norm(delta))
            abs_b = global_rect_to_abs_bearing(delta)
            rel_b = angle180(abs_b - float(global_state[(self.id, "heading")]))
            return np.asarray(dist_rel_bearing_to_local_rect(d, rel_b), dtype=np.float64)

        # If crossed scrimmage, redirect to own flag immediately.
        if not on_own_side:
            return self.action_from_vector(np.asarray(self.my_flag_loc, dtype=np.float64), spd)

        # Zone-based patrol using agent hash.
        agent_hash = int(hashlib.md5(self.id.encode()).hexdigest(), 16) % 3
        zone = agent_hash % zone_mod

        team_ids = sorted(list(getattr(self, "teammate_ids", [])))
        try:
            slot_idx = team_ids.index(self.id)
        except ValueError:
            slot_idx = 0
        num_slots = max(1, len(team_ids))

        # Lateral spread on own side (10%..90% of field height) to avoid flag clumping.
        lane_count = min(4, max(2, num_slots))
        lane_idx = slot_idx % lane_count
        lane_frac = (lane_idx + 0.5) / lane_count

        my_to_opp_x = float(opp_flag_home[0] - my_flag_home[0])
        x_to_scrim = abs(scrimmage_x - float(my_flag_home[0]))

        # Difficulty-specific depth bands on own side, all constrained before scrimmage.
        if effective_mode == "easy":
            depth_fractions = (0.20, 0.35, 0.50)
        elif effective_mode == "medium":
            depth_fractions = (0.30, 0.55, 0.80)
        else:
            depth_fractions = (0.35, 0.65, 0.92)
        depth_frac = float(depth_fractions[min(zone, len(depth_fractions) - 1)])
        depth = depth_frac * x_to_scrim

        if my_to_opp_x >= 0.0:
            patrol_x = float(my_flag_home[0] + depth)
            patrol_x = min(patrol_x, scrimmage_x - 2.0)
        else:
            patrol_x = float(my_flag_home[0] - depth)
            patrol_x = max(patrol_x, scrimmage_x + 2.0)

        y_coords = [float(my_flag_home[1]), float(opp_flag_home[1])]
        for wall in getattr(self, "walls", []):
            for endpoint in wall:
                y_coords.append(float(endpoint[1]))
        y_min = min(y_coords)
        y_max = max(y_coords)
        patrol_y = y_min + lane_frac * (y_max - y_min)

        scrim_push_margin = 5.0
        if team_str == "blue":
            near_scrimmage_x = float(scrimmage_x - scrim_push_margin)
        else:
            near_scrimmage_x = float(scrimmage_x + scrim_push_margin)

        if zone == 0:
            if effective_mode in ("easy", "medium"):
                # Easy/medium mode: spread in a front-centered arc near our flag lane.
                centered_lane_frac = 0.5 + 0.6 * (lane_frac - 0.5)
                lane_phase = (centered_lane_frac - 0.5) * np.pi  # narrower fan around centerline
                arc_depth = 0.60 * x_to_scrim
                arc_width = 0.30 * (y_max - y_min)
                if my_to_opp_x >= 0.0:
                    arc_x = float(my_flag_home[0] + arc_depth * np.cos(lane_phase))
                    arc_x = min(arc_x, scrimmage_x - 2.0)
                else:
                    arc_x = float(my_flag_home[0] - arc_depth * np.cos(lane_phase))
                    arc_x = max(arc_x, scrimmage_x + 2.0)
                arc_y = float(my_flag_home[1] + arc_width * np.sin(lane_phase))
                arc_y = min(max(arc_y, y_min + 1.0), y_max - 1.0)
                patrol_target = np.asarray([arc_x, arc_y], dtype=np.float64)
            else:
                patrol_target = np.asarray([patrol_x, patrol_y], dtype=np.float64)
        elif zone == 1:
            patrol_target = np.asarray([patrol_x, patrol_y], dtype=np.float64)
        else:
            patrol_target = np.asarray([near_scrimmage_x, patrol_y], dtype=np.float64)

        # Flag carrier pursuit — separate from general enemy chase.
        carrier_local = None
        carrier_dist = float("inf")
        if self.opp_team_has_flag:
            for enem, pos in getattr(self, "opp_team_pos_dict", {}).items():
                if not bool(global_state.get((enem, "has_flag"), False)):
                    continue
                on_our_side = float(global_state.get((enem, "on_side"), 1.0)) == 0.0
                if not on_our_side:
                    continue
                d = float(pos[0])
                if d < carrier_dist:
                    carrier_dist = d
                    carrier_local = np.asarray(
                        dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64
                    )
        desired_speed = spd
        if carrier_local is not None and carrier_dist <= carrier_thresh:
            my_action = carrier_local
            desired_speed = carrier_chase_spd
        else:
            # Enemy detection — only chase enemies on own side, not tagged.
            nearest_local = None
            nearest_dist = float("inf")
            for enem, pos in getattr(self, "opp_team_pos_dict", {}).items():
                on_our_side = float(global_state.get((enem, "on_side"), 1.0)) == 0.0
                is_tagged = bool(global_state.get((enem, "is_tagged"), False))
                if not on_our_side or is_tagged:
                    continue
                d = float(pos[0])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_local = np.asarray(
                        dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64
                    )

            if nearest_local is not None and nearest_dist <= react_thresh:
                my_action = nearest_local
            else:
                my_action = _to_local(patrol_target)

        # Wall avoidance.
        wall_pos = []
        for wd, wb in zip(self.wall_distances, self.wall_bearings):
            if float(wd) < 8 and (-90 < float(wb) < 90):
                wall_pos.append((float(wd), float(wb)))
        if wall_pos:
            my_action = my_action + get_avoid_vect(wall_pos, avoid_threshold=8.0)

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
