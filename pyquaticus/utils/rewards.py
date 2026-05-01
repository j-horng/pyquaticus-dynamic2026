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

"""
#Configureable Rewards
    # -- NOTE --
    #   All headings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard heading format that is counterclockwise
    # by using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    ## Each custom reward function should have the following arguments ##
    Args:
        agent_id (int): ID of the agent we are computing the reward for
        team (Team): team of the agent we are computing the reward for
        agents (list): list of agent ID's (this is used to map agent_id's to agent indices and viceversa)
        agent_inds_of_team (dict): mapping from team to agent indices of that team
        state (dict):
            'agent_position' (array): list of agent positions (indexed in the order of agents list)

                        Ex. Usage: Get agent's current position
                        agent_id = 'agent_1'
                        position = state['agent_position'][agents.index(agent_id)]

            'prev_agent_position' (array): list of agent positions (indexed in the order of agents list) at the previous timestep

                        Ex. Usage: Get agent's previous position
                        agent_id = 'agent_1'
                        prev_position = state['prev_agent_position'][agents.index(agent_id)]

            'agent_speed' (array): list of agent speeds (indexed in the order of agents list)

                        Ex. Usage: Get agent's speed
                        agent_id = 'agent_1'
                        speed = state

            'agent_heading' (array): list of agent headings (indexed in the order of agents list)

                        Ex. Usage: Get agent's heading
                        agent_id = 'agent_1'
                        heading = state['agent_heading'][agents.index(agent_id)]

            'agent_on_sides' (array): list of booleans (indexed in the order of agents list) where True means the
                                      agent is on its own side, and False means the agent is not on its own side

                        Ex. Usage: Check if agent is on its own side
                        agent_id = 'agent_1'
                        on_own_side = state['agent_on_sides'][agents.index(agent_id)]

            'agent_oob' (array): list of booleans (indexed in the order of agents list) where True means the
                                 agent is out-of-bounds (OOB), and False means the agent is not out-of-bounds
                        
                        Ex. Usage: Check if agent is out-of-bounds
                        agent_id = 'agent_1'
                        num_oob = state['agent_oob'][agents.index(agent_id)]
            
            'agent_has_flag' (array): list of booleans (indexed in the order of agents list) where True means the
                                     agent has a flag, and False means the agent does not have a flag

                        Ex. Usage: Check if agent has a flag
                        agent_id = 'agent_1'
                        has_flag = state['agent_has_flag'][agents.index(agent_id)]

            'agent_is_tagged' (array): list of booleans (indexed in the order of agents list) where True means
                                       the agent is tagged, and False means the agent is not tagged

                        Ex. Usage: Check if agent is tagged
                        agent_id = 'agent_1'
                        is_tagged = state['agent_is_tagged'][agents.index(agent_id)]

            'agent_made_tag' (array): list (indexed in the order of agents list) where the value at an entry is the index of a different
                                     agent which the agent at the given index has tagged at the current timestep, otherwise None

                        Ex. Usage: Check if agent has tagged an agent
                        agent_id = 'agent_1'
                        tagged_opponent_idx = state['agent_made_tag'][agents.index(agent_id)]

            'agent_tagging_cooldown' (array): current agent tagging cooldowns (indexed in the order of agents list)
                        Note: agent is able to tag when this value is equal to tagging_cooldown
    
                        Ex. Usage: Get agent's current tagging cooldown
                        agent_id = 'agent_1'
                        cooldown = self.state['agent_tagging_cooldown'][agents.index(agent_id)]

            'dist_bearing_to_obstacles' (dict): For each agent in game list out distances and bearings
                                                to all obstacles in game in order of obstacles list

            'flag_home' (array): list of flag homes (indexed by team number)

            'flag_position' (array): list of flag homes (indexed by team number)

            'flag_taken' (array): list of booleans (indexed by team number) where True means the team's flag
                                  is taken (picked up by an opponent), and False means the flag is not taken 

            'team_has_flag' (array): list of booleans (indexed by team number) where True means an agent of the
                                     team has a flag, and False means that no agents are in possesion of a flag

            'captures' (array): list of total captures made by each team (indexed by team number)

            'tags' (array): list of total tags made by each team (indexed by team number)

            'grabs' (array): list of total flag grabs made by each team (indexed by team number)

            'agent_collisions' (array): list of total agent collisions  for each agent (indexed in the order of agents list)

            'agent_dynamics' (array): list of dictionaries containing agent-specific dynamics information (state attribute of a dynamics class - see dynamics.py)

            ######################################################################################
            ##### The following keys will exist in the state dictionary if lidar_obs is True #####
                'lidar_labels' (dict):

                'lidar_labels' (dict):

                'lidar_labels' (dict):
            ######################################################################################
            
            'obs_hist_buffer' (dict): Observation history buffer where the keys are agent_id's and values are the agents' observations

            'global_state_hist_buffer' (array): Global state history buffer

        prev_state (dict): Contains the state information from the previous step

        env_size (array): field dimensions [horizontal, vertical]

        agent_radii (array): list of agent radii (indexed in the order of agents list)

        catch_radius (float): tag and flag grab radius

        scrimmage_coords (array): endpoints [x,y] of the scrimmage line

        max_speeds (list): list of agent max speeds (indexed in the order of agents list)

        tagging_cooldown (float): tagging cooldown time
"""

import math
import numpy as np

from pyquaticus.config import config_dict_std
from pyquaticus.structs import Team
from pyquaticus.utils.utils import *

# Set to True to print reward events to console during deployment/rendering
REWARD_DEBUG = False
# If True, record per-step reward component breakdown for analysis.
REWARD_PARTS_DEBUG = False

_IDLE_STREAK_STEPS = {}
_DEEP_PUSH_REWARDED = {}
_DEEP_FLEE_75_REWARDED = {}
_DEEP_FLEE_625_REWARDED = {}
_ENEMY_DEEP_PENALIZED = {}

### Example Reward Funtion ###
def example_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    return 0.0

def caps_and_grabs(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    global _DEEP_PUSH_REWARDED, _DEEP_FLEE_75_REWARDED, _DEEP_FLEE_625_REWARDED
    reward = 0.0
    parts = None
    if REWARD_PARTS_DEBUG:
        parts = {"deep_flee_75": 0.0, "deep_flee_625": 0.0, "deep_push": 0.0}

        def _add_part(k: str, v: float):
            parts[k] = float(parts.get(k, 0.0)) + float(v)

    R_DEEP_FLEE_75 = 1.0
    R_DEEP_FLEE_625 = 0.6
    R_DEEP_PUSH_BASE = 0.5
    R_DEEP_PUSH_BONUS = 0.5

    agent_index = agents.index(agent_id)

    # Inactive/disabled agents must not receive shaping or team-event credit (e.g. red_dummy 3v0).
    disabled = state.get("disabled_agents")
    if disabled is not None and len(disabled) > agent_index and bool(disabled[agent_index]):
        return 0.0

    prev_num_oob = prev_state["agent_oob"][agent_index]
    num_oob = state["agent_oob"][agent_index]
    if num_oob > prev_num_oob:
        reward += -2.0
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} OOB: -2.00")

    # Reward for tagging an opponent
    if state["agent_made_tag"][agent_index] is not None:
        reward += 0.25
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} tagged opponent: +0.25")

    # Check if agents lost or gained flag (0/1 or bool from env state).
    prev_has_flag = prev_state["agent_has_flag"][agent_index]
    has_flag = state["agent_has_flag"][agent_index]
    had_flag_before = bool(np.asarray(prev_has_flag).item())
    has_flag_now = bool(np.asarray(has_flag).item())

    # Agent lost opponent flag (carrier -> not carrier).
    if had_flag_before and not has_flag_now:
        # Successful capture clears the carrier and resets the flag in the same env step as
        # captures[team] increments — do not treat that as a bad "lost flag" (-1).
        team_i = int(team)
        captured_now = state["captures"][team_i] > prev_state["captures"][team_i]
        if not captured_now:
            reward += -1.0
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} lost flag: -1.00")

    # Agent grabbed flag individually
    if has_flag_now and not had_flag_before:
        reward += 0.5
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} grabbed flag: +0.50")

    # Grabs and captures are of shape [team_0 (BLUE), team_1 (RED)].
    # Full per-agent credit (no team_size scaling).
    for t in range(len(state["grabs"])):
        prev_num_caps = prev_state["captures"][t]
        num_caps = state["captures"][t]
        if num_caps > prev_num_caps:
            # Capture reward: +0.5 team-wide for the capturing team, plus +1 individual
            # for the agent that was carrying the flag at capture time.
            r_team = 0.5 if t == int(team) else -0.5
            reward += r_team
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} capture team (team {t}): {r_team:+.2f}")

            if t == int(team) and had_flag_before:
                reward += 1.0
                if REWARD_DEBUG:
                    print(f"[REWARD] {agent_id} capture individual: +1.00")

    # ── Deep Flee (one-time per carry, flag carrier retreating toward home) ─
    pos = np.asarray(state["agent_position"][agent_index], dtype=np.float64)
    prev_pos = np.asarray(prev_state["agent_position"][agent_index], dtype=np.float64)
    if has_flag_now:
        field_w = float(env_size[0])
        if int(team) == 0:
            # Blue on left; "fleeing back" means decreasing x
            crossed_75 = float(prev_pos[0]) > 0.75 * field_w >= float(pos[0])
            crossed_625 = float(prev_pos[0]) > 0.625 * field_w >= float(pos[0])
        else:
            crossed_75 = float(prev_pos[0]) < 0.25 * field_w <= float(pos[0])
            crossed_625 = float(prev_pos[0]) < 0.375 * field_w <= float(pos[0])

        if crossed_75 and not _DEEP_FLEE_75_REWARDED.get(agent_id, False):
            reward += R_DEEP_FLEE_75
            if parts is not None:
                _add_part("deep_flee_75", R_DEEP_FLEE_75)
            _DEEP_FLEE_75_REWARDED[agent_id] = True
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} deep flee 3/4: +{R_DEEP_FLEE_75}")
        if crossed_625 and not _DEEP_FLEE_625_REWARDED.get(agent_id, False):
            reward += R_DEEP_FLEE_625
            if parts is not None:
                _add_part("deep_flee_625", R_DEEP_FLEE_625)
            _DEEP_FLEE_625_REWARDED[agent_id] = True
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} deep flee 5/8: +{R_DEEP_FLEE_625}")
    else:
        # Reset flee flags when agent no longer has flag (captured or tagged)
        _DEEP_FLEE_75_REWARDED[agent_id] = False
        _DEEP_FLEE_625_REWARDED[agent_id] = False

    # ── Deep Push (one-time per cooldown window, while on tag cooldown) ────
    cd = float(np.asarray(state["agent_tagging_cooldown"][agent_index]).item())
    on_cooldown = cd < tagging_cooldown
    cooldown_progress = float(np.clip(cd / tagging_cooldown, 0.0, 1.0)) if tagging_cooldown > 0 else 1.0
    is_tagged_scalar = bool(np.asarray(state["agent_is_tagged"][agent_index]).item())

    if on_cooldown and (not has_flag_now) and (not is_tagged_scalar):
        field_w = float(env_size[0])
        threshold = 0.75 * field_w
        crossed = (
            float(prev_pos[0]) < threshold <= float(pos[0])
            if int(team) == 0
            else float(prev_pos[0]) > (field_w - threshold) >= float(pos[0])
        )
        if crossed and not _DEEP_PUSH_REWARDED.get(agent_id, False):
            deep_reward = R_DEEP_PUSH_BASE + R_DEEP_PUSH_BONUS * (1.0 - cooldown_progress)
            reward += deep_reward
            if parts is not None:
                _add_part("deep_push", deep_reward)
            _DEEP_PUSH_REWARDED[agent_id] = True
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} deep push: +{deep_reward:.3f}")
    if not on_cooldown:
        _DEEP_PUSH_REWARDED[agent_id] = False

    if parts is not None:
        setattr(caps_and_grabs, "_last_parts", getattr(caps_and_grabs, "_last_parts", {}))
        caps_and_grabs._last_parts[agent_id] = parts
    return float(reward)



### Add Custom Reward Functions Here ###