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

from pyquaticus.structs import Team
from pyquaticus.utils.utils import *

# Set to True to print reward events to console during deployment/rendering
REWARD_DEBUG = False
TAGGED_PENALTY = -0.75
IDLE_STEP_PENALTY = -0.002
CIRCLE_STEP_PENALTY = -0.003
IDLE_DIST_THRESH = 0.15
CIRCLE_DIST_THRESH = 0.60
CIRCLE_HEADING_DELTA_DEG = 35.0
# With default tau=0.1s and sim_speedup_factor=1, 20 steps ~= 2 seconds.
IDLE_GRACE_STEPS = 20
CIRCLE_GRACE_STEPS = 20

_IDLE_STREAK_STEPS = {}
_CIRCLE_STREAK_STEPS = {}

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
    global _IDLE_STREAK_STEPS, _CIRCLE_STREAK_STEPS
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

    heading_now = float(np.asarray(state["agent_heading"][agent_index]).item())
    heading_prev = float(np.asarray(prev_state["agent_heading"][agent_index]).item())
    heading_delta = abs(((heading_now - heading_prev + 180.0) % 360.0) - 180.0)
    circle_streak = _CIRCLE_STREAK_STEPS.get(agent_id, 0)
    if step_dist < CIRCLE_DIST_THRESH and heading_delta > CIRCLE_HEADING_DELTA_DEG:
        circle_streak += 1
        _CIRCLE_STREAK_STEPS[agent_id] = circle_streak
    else:
        _CIRCLE_STREAK_STEPS[agent_id] = 0
    if circle_streak >= CIRCLE_GRACE_STEPS:
        reward += CIRCLE_STEP_PENALTY
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} circle step: {CIRCLE_STEP_PENALTY:+.4f} (streak={circle_streak})")

    prev_is_tagged = bool(np.asarray(prev_state["agent_is_tagged"][agent_index]).item())
    is_tagged = bool(np.asarray(state["agent_is_tagged"][agent_index]).item())
    if (not prev_is_tagged) and is_tagged:
        reward += TAGGED_PENALTY
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} got tagged: {TAGGED_PENALTY:+.2f}")

    # Reward for tagging an opponent
    if state["agent_made_tag"][agent_index] is not None:
        reward += 0.75
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} tagged opponent: +0.75")

    # Bonus for tagging the flag carrier
    tagged_idx = state["agent_made_tag"][agent_index]
    if tagged_idx is not None and bool(state["agent_has_flag"][tagged_idx]):
        reward += 1.0
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} tagged flag carrier bonus: +1.00")

    # Note: this function now adds a separate bonus for tagging the flag carrier.

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
        reward += 1.0
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} grabbed flag: +1.00")

    # Grabs and captures are of shape [team_0 (BLUE), team_1 (RED)].
    # Full per-agent credit (no team_size scaling).
    for t in range(len(state['grabs'])):
        prev_num_grabs = prev_state['grabs'][t]
        num_grabs = state['grabs'][t]
        # Note: grab reward is individual-only (see has_flag delta above). No team grab reward here.

        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            # Capture reward: +0.5 team-wide for the capturing team, plus +1 individual
            # for the agent that was carrying the flag at capture time.
            r_team = 1.0 if t == int(team) else -1.5
            reward += r_team
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} capture team (team {t}): {r_team:+.2f}")

            if t == int(team) and had_flag_before:
                reward += 1.5
                if REWARD_DEBUG:
                    print(f"[REWARD] {agent_id} capture individual: +1.50")

    cd = float(state["agent_tagging_cooldown"][agent_index])
    if (
        cd < tagging_cooldown
        and state["agent_has_flag"][agent_index] == 0
        and state["agent_is_tagged"][agent_index] == 0
    ):
        # Reward once when agent crosses the 3/4 map threshold into enemy territory while on cooldown.
        # 3/4 of field width means agent is 75% across the map toward the enemy side.
        field_w = float(env_size[0])
        pos_x = float(pos[0])
        prev_pos_x = float(prev_pos[0])
        threshold_x = 0.75 * field_w
        # Blue is on left side (x=0), enemy flag is on right side (x=field_w).
        # Flip for red team (team 1).
        if int(team) == 0:
            crossed = prev_pos_x < threshold_x <= pos_x
        else:
            crossed = prev_pos_x > (field_w - threshold_x) >= pos_x
        if crossed:
            reward += 0.5
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} cooldown deep push: +0.50")

    agent_on_own_side = bool(state["agent_on_sides"][agent_index])
    # Shared evasion: find nearest active opponent distance delta
    _evasion_active = (
        (not agent_on_own_side and not has_flag_now and not bool(state["agent_is_tagged"][agent_index]))
        or
        (has_flag_now and not bool(state["agent_is_tagged"][agent_index]))
    )
    _can_tag_now = cd >= float(tagging_cooldown)
    if _evasion_active and _can_tag_now:
        field_diag = float(np.linalg.norm(env_size))
        min_curr_dist = float("inf")
        min_prev_dist = float("inf")
        for opp_idx in agent_inds_of_team.get(Team.RED_TEAM, []):
            opp_disabled = state.get("disabled_agents")
            if opp_disabled is not None and len(opp_disabled) > opp_idx and bool(opp_disabled[opp_idx]):
                continue
            opp_pos = np.asarray(state["agent_position"][opp_idx], dtype=np.float64)
            opp_prev_pos = np.asarray(prev_state["agent_position"][opp_idx], dtype=np.float64)
            curr_d = np.linalg.norm(pos - opp_pos)
            prev_d = np.linalg.norm(prev_pos - opp_prev_pos)
            if curr_d < min_curr_dist:
                min_curr_dist = curr_d
                min_prev_dist = prev_d
        if min_curr_dist < float("inf"):
            evasion_delta = min_curr_dist - min_prev_dist
            if evasion_delta > 0 and field_diag > 0 and min_curr_dist < 30.0:
                # Higher multiplier when carrying flag (more critical to survive)
                multiplier = 0.3 if has_flag_now else 0.2
                r = multiplier * evasion_delta / field_diag
                reward += r
                if REWARD_DEBUG:
                    label = "carrying flag" if has_flag_now else "enemy side"
                    print(f"[REWARD] {agent_id} evasion ({label}): +{r:.4f}")

    return reward



### Add Custom Reward Functions Here ###
