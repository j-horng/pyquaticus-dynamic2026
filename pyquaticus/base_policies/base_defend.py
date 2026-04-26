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
                                            rel_bearing_to_local_unit_rect,
                                            unit_vect_between_points)
from pyquaticus.config import config_dict_std
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180, closest_point_on_line, dist

MODES = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}


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

        # Treat competition_* as hard for this simplified "pure defend" defender.
        effective_mode = self.mode if self.mode in ("easy", "medium", "hard") else "hard"

        if effective_mode == "easy":
            spd = 0.4
            react_thresh = 20.0
        elif effective_mode == "medium":
            spd = 0.6
            react_thresh = 40.0
        else:
            spd = 1.0
            react_thresh = float("inf")

        # If opponent has our flag, chase aggressively at full speed regardless of mode.
        if self.opp_team_has_flag:
            # Easy defenders should be clearly beatable: don't instantly full-send after the carrier
            # from anywhere on the map. Only pursue if the carrier is fairly close; otherwise patrol.
            if effective_mode == "easy":
                carrier = None
                carrier_dist = float("inf")
                for enem, pos in getattr(self, "opp_team_pos_dict", {}).items():
                    try:
                        has_flag = bool(global_state.get((enem, "has_flag"), 0.0))
                    except Exception:
                        has_flag = False
                    if not has_flag:
                        continue
                    d = float(pos[0])
                    if d < carrier_dist:
                        carrier_dist = d
                        carrier = np.asarray(dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64)
                # If we can't identify the carrier, fall back to patrolling.
                if carrier is not None and carrier_dist <= 60.0:
                    my_action = carrier
                    desired_speed = spd
                else:
                    my_action = np.asarray(self.my_flag_loc, dtype=np.float64)
                    desired_speed = spd
            else:
                my_action = np.asarray(self.my_flag_loc, dtype=np.float64)
                # Medium should not suddenly become "hard speed" just because the opponent has the flag.
                desired_speed = 1.0 if effective_mode == "hard" else spd
        else:
            # Always prioritize chasing nearest untagged opponent on our side.
            nearest = None
            nearest_dist = float("inf")
            for enem, pos in getattr(self, "opp_team_pos_dict", {}).items():
                try:
                    on_our_side = float(global_state.get((enem, "on_side"), 1.0)) == 0.0
                    is_tagged = bool(global_state.get((enem, "is_tagged"), 0.0))
                except Exception:
                    on_our_side, is_tagged = False, False
                if (not on_our_side) or is_tagged:
                    continue
                d = float(pos[0])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest = np.asarray(dist_rel_bearing_to_local_rect(pos[0], pos[1]), dtype=np.float64)

            if nearest is not None and nearest_dist <= react_thresh:
                my_action = nearest
                desired_speed = spd
            else:
                # Patrol between flag and scrimmage line (approx): hover near flag, then push outward.
                if self.my_flag_distance > (self.flag_keepout + self.catch_radius + 5.0):
                    my_action = np.asarray(self.my_flag_loc, dtype=np.float64)
                else:
                    my_action = np.asarray(-1.0 * self.my_flag_loc, dtype=np.float64)
                desired_speed = spd

        # OOB avoidance for both (exact per-task snippet).
        wall_pos = []
        for wd, wb in zip(getattr(self, "wall_distances", []), getattr(self, "wall_bearings", [])):
            if float(wd) < 8 and (-90 < float(wb) < 90):
                wall_pos.append((float(wd), float(wb)))
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
