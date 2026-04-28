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
TAGGED_PENALTY = -0.75
IDLE_STEP_PENALTY = -0.05
IDLE_DIST_THRESH = 0.15
# With default tau=0.1s and sim_speedup_factor=1, 10 steps ~= 1 second.
IDLE_GRACE_STEPS = 10

_IDLE_STREAK_STEPS = {}
_DEEP_HOLD_REWARDED = {}
_TERMINAL_REWARDED = {}

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
    global _IDLE_STREAK_STEPS, _DEEP_HOLD_REWARDED, _TERMINAL_REWARDED
    reward = 0.0
    
    #----------
    agent_index = agents.index(agent_id)

    # Inactive/disabled agents must not receive shaping or team-event credit (e.g. red_dummy 3v0).
    disabled = state.get("disabled_agents")
    if disabled is not None and len(disabled) > agent_index and bool(disabled[agent_index]):
        return 0.0

    # Only Blue team (team 0) gets trained — Red agents always return 0.0.
    if int(team) != int(Team.BLUE_TEAM):
        return 0.0

    prev_num_oob = prev_state["agent_oob"][agent_index]
    num_oob = state["agent_oob"][agent_index]
    if num_oob > prev_num_oob:
        reward += -3.0
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} OOB: -3.00")

    pos = np.asarray(state["agent_position"][agent_index], dtype=np.float64)
    prev_pos = np.asarray(prev_state["agent_position"][agent_index], dtype=np.float64)
    step_dist = float(np.linalg.norm(pos - prev_pos))
    idle_streak = _IDLE_STREAK_STEPS.get(agent_id, 0)
    if step_dist < IDLE_DIST_THRESH:
        idle_streak += 1
        _IDLE_STREAK_STEPS[agent_id] = idle_streak
    else:
        _IDLE_STREAK_STEPS[agent_id] = 0
    if idle_streak >= IDLE_GRACE_STEPS:
        reward += IDLE_STEP_PENALTY
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} idle step: {IDLE_STEP_PENALTY:+.4f} (streak={idle_streak})")

    prev_is_tagged = bool(np.asarray(prev_state["agent_is_tagged"][agent_index]).item())
    is_tagged = bool(np.asarray(state["agent_is_tagged"][agent_index]).item())

    prev_has_flag = prev_state["agent_has_flag"][agent_index]
    has_flag = state["agent_has_flag"][agent_index]
    had_flag_before = bool(np.asarray(prev_has_flag).item())
    has_flag_now = bool(np.asarray(has_flag).item())

    tagged_idx = state["agent_made_tag"][agent_index]
    tagged_had_flag = (
        tagged_idx is not None and
        bool(np.asarray(prev_state["agent_has_flag"][tagged_idx]).item())
    )

    blue_caps = state["captures"][int(Team.BLUE_TEAM)]
    red_caps = state["captures"][int(Team.RED_TEAM)]
    score_diff = blue_caps - red_caps
    is_tied = score_diff == 0
    is_behind = score_diff < 0
    is_ahead = score_diff > 0
    enemy_team = 1 - int(team)

    team_caps_up = state["captures"][int(team)] > prev_state["captures"][int(team)]
    enemy_caps_up = state["captures"][enemy_team] > prev_state["captures"][enemy_team]
    our_flag_taken_now = bool(state["flag_taken"][int(team)])
    our_flag_taken_prev = bool(prev_state["flag_taken"][int(team)])

    # ── Attack Phase ──────────────────────────────────────────────────────
    if is_tied or is_behind:

        # INDIVIDUAL: grabbed the flag
        if has_flag_now and not had_flag_before:
            reward += 2.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK grab individual: +2.00")

        # TEAM: someone grabbed the flag (includes the grabber)
        blue_grabs_up = state["grabs"][int(team)] > prev_state["grabs"][int(team)]
        if blue_grabs_up:
            reward += 1.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK grab team: +1.00")

        # INDIVIDUAL: I was the carrier who scored
        if team_caps_up and had_flag_before:
            reward += 5.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK capture individual: +5.00")

        # TEAM: a capture happened (includes carrier)
        if team_caps_up:
            reward += 1.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK capture team: +1.00")

        # INDIVIDUAL: tagged someone
        if tagged_idx is not None:
            reward += 0.25
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK tagged opponent: +0.25")

        # INDIVIDUAL: got tagged empty-handed
        if (not prev_is_tagged) and is_tagged and not had_flag_before:
            reward += -0.25
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK got tagged empty: -0.25")

        # INDIVIDUAL: lost the flag without scoring
        if had_flag_before and not has_flag_now and not team_caps_up:
            reward += -1.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK lost flag: -1.00")

        # TEAM: enemy grabbed our flag
        if our_flag_taken_now and not our_flag_taken_prev:
            reward += -0.5
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK enemy grab team: -1.00")

        # TEAM: enemy scored
        if enemy_caps_up:
            reward += -2.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK enemy scored team: -2.00")

        # DEEP PUSH: only during attack phase, only while on cooldown from being tagged
        cd = float(state["agent_tagging_cooldown"][agent_index])
        on_cooldown = cd < tagging_cooldown
        not_carrying = state["agent_has_flag"][agent_index] == 0
        not_tagged = state["agent_is_tagged"][agent_index] == 0
        if tagging_cooldown > 0:
            cooldown_progress = float(np.clip(cd / tagging_cooldown, 0.0, 1.0))
        else:
            cooldown_progress = 1.0

        if on_cooldown and not_carrying and not_tagged:
            field_w = float(env_size[0])
            pos_x = float(pos[0])
            prev_pos_x = float(prev_pos[0])
            threshold_x = 0.75 * field_w

            if int(team) == 0:
                crossed = prev_pos_x < threshold_x <= pos_x
            else:
                crossed = prev_pos_x > (field_w - threshold_x) >= pos_x

            already_rewarded = _DEEP_HOLD_REWARDED.get(agent_id, False)

            if crossed and not already_rewarded:
                depth_reward = 0.5 + 0.5 * (1.0 - cooldown_progress)
                reward += depth_reward
                _DEEP_HOLD_REWARDED[agent_id] = True
                if REWARD_DEBUG: print(f"[REWARD] {agent_id} ATTACK deep push (cd={cooldown_progress:.2f}): +{depth_reward:.4f}")

        if not on_cooldown:
            _DEEP_HOLD_REWARDED[agent_id] = False

    # ── Defense Phase ─────────────────────────────────────────────────────
    if is_ahead:

        # INDIVIDUAL: tagged the flag carrier specifically
        if tagged_had_flag:
            reward += 3.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND tagged carrier: +3.00")

        # INDIVIDUAL: tagged anyone
        if tagged_idx is not None:
            reward += 1.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND tagged opponent: +1.00")

        # INDIVIDUAL: I extended the lead as the carrier
        if team_caps_up and had_flag_before:
            reward += 2.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND extend lead individual: +2.00")

        # TEAM: capture happened (includes carrier)
        if team_caps_up:
            reward += 1.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND extend lead team: +1.00")

        # TEAM: enemy grabbed our flag
        if our_flag_taken_now and not our_flag_taken_prev:
            reward += -2.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND enemy grab team: -2.00")

        # TEAM: our flag returned safely
        if our_flag_taken_prev and not our_flag_taken_now and not enemy_caps_up:
            reward += 2.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND flag recovered team: +2.00")

        # TEAM: enemy scored, lead eroded
        if enemy_caps_up:
            reward += -4.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND enemy scored team: -4.00")

        # INDIVIDUAL: got tagged recklessly
        if (not prev_is_tagged) and is_tagged and not had_flag_before:
            reward += -1.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} DEFEND reckless tag: -1.00")

    # ── Terminal (end-of-game) ─────────────────────────────────────────────
    max_score = int(state.get("max_score", config_dict_std.get("max_score", 20)))
    prev_game_over = bool(prev_state.get("game_done", np.any(np.asarray(prev_state["captures"]) >= max_score)))
    game_over_now = bool(state.get("game_done", np.any(np.asarray(state["captures"]) >= max_score)))

    if game_over_now and not prev_game_over and not _TERMINAL_REWARDED.get(agent_id, False):
        if is_ahead:
            reward += 10.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} TERMINAL win: +10.00")
        elif is_tied:
            reward += -6.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} TERMINAL tie: -6.00")
        else:
            reward += -10.0
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} TERMINAL loss: -10.00")
        _TERMINAL_REWARDED[agent_id] = True

    if not game_over_now:
        _TERMINAL_REWARDED[agent_id] = False

    return reward



### Add Custom Reward Functions Here ###
