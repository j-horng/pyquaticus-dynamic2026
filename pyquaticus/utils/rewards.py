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
    reward = 0.0

    agent_index = agents.index(agent_id)
    team_i = int(team)
    opp_team = 1 - team_i

    disabled = state.get("disabled_agents")
    if disabled is not None and len(disabled) > agent_index and bool(disabled[agent_index]):
        return 0.0

    def is_disabled(idx):
        return disabled is not None and len(disabled) > idx and bool(disabled[idx])

    # Resolve own team indices for distinguishing opponents
    my_indices = set()
    for k in [team, team_i]:
        if k in agent_inds_of_team:
            my_indices = set(agent_inds_of_team[k])
            break

    # OOB penalty
    prev_num_oob = prev_state["agent_oob"][agent_index]
    num_oob = state["agent_oob"][agent_index]
    if num_oob > prev_num_oob:
        reward += -1.0
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} OOB: -1.00")

    # Tag reward — reduced to discourage farming; skip tagging disabled/inactive agents
    tagged_idx = state["agent_made_tag"][agent_index]
    if tagged_idx is not None and not is_disabled(tagged_idx):
        reward += 0.05
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} tagged opponent: +0.05")

    # Flag gain/loss
    prev_has_flag = prev_state['agent_has_flag'][agent_index]
    has_flag = state['agent_has_flag'][agent_index]
    if prev_has_flag > has_flag:
        captured_now = state["captures"][team_i] > prev_state["captures"][team_i]
        if not (captured_now and prev_has_flag == 1):
            reward += -2.0
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} lost flag: -2.00")
    if has_flag > prev_has_flag:
        reward += 1.0
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} grabbed flag: +1.00")

    # Extra penalty for being tagged while carrying (on top of the -2.0 flag-loss above)
    if prev_has_flag and not has_flag:
        cap_check = state["captures"][team_i] > prev_state["captures"][team_i]
        if not cap_check and bool(state["agent_is_tagged"][agent_index]):
            reward += -1.0
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} tagged while carrying: -1.00")

    # Captures — increased to make scoring the dominant incentive
    for t in range(len(state['captures'])):
        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            r_team = 1.5 if t == team_i else -1.5
            reward += r_team
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} capture team (team {t}): {r_team:+.2f}")
            if t == team_i and prev_has_flag == 1:
                reward += 3.0
                if REWARD_DEBUG:
                    print(f"[REWARD] {agent_id} capture individual: +3.00")

    pos = np.asarray(state["agent_position"][agent_index], dtype=np.float64)
    prev_pos = np.asarray(prev_state["agent_position"][agent_index], dtype=np.float64)
    moved = float(np.linalg.norm(pos - prev_pos))
    field_diag = float(np.linalg.norm(env_size)) if np.linalg.norm(env_size) > 0 else 1.0

    # Stationary penalty: break the zero-reward equilibrium of doing nothing
    if moved < 1e-3 and not state["agent_is_tagged"][agent_index]:
        reward += -0.002
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} stationary: -0.002")

    # Spinning penalty: penalize large heading changes when barely moving
    curr_heading = float(state["agent_heading"][agent_index])
    prev_heading = float(prev_state["agent_heading"][agent_index])
    heading_diff = abs(((curr_heading - prev_heading + 180.0) % 360.0) - 180.0)
    if heading_diff > 30.0 and moved < catch_radius:
        r = -0.03 * (heading_diff / 180.0)
        reward += r
        if REWARD_DEBUG:
            print(f"[REWARD] {agent_id} spinning penalty: {r:.4f}")

    on_own_side = bool(state["agent_on_sides"][agent_index])

    # Carrier urgency: penalize lingering on enemy side; reward progress toward own flag home
    if has_flag:
        if not on_own_side:
            reward += -0.01
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} carrier on enemy side: -0.01")
        own_home = np.asarray(state["flag_home"][team_i], dtype=np.float64)
        curr_dist_home = float(np.linalg.norm(pos - own_home))
        prev_dist_home = float(np.linalg.norm(prev_pos - own_home))
        delta_home = prev_dist_home - curr_dist_home
        if delta_home > 0 and moved > 1e-3:
            r = 0.25 * delta_home / field_diag
            reward += r
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} carrier moving home: +{r:.4f}")

    # Defense: proactive — respond to any enemy invader on our side
    invaders = [
        i for i in range(len(agents))
        if i not in my_indices
        and not is_disabled(i)
        and not bool(state["agent_on_sides"][i])  # enemy is NOT on their own side = on our side
    ]

    if invaders:
        invader_positions = [np.asarray(state["agent_position"][i], dtype=np.float64) for i in invaders]
        dists_to_invaders = [float(np.linalg.norm(pos - p)) for p in invader_positions]
        closest_i = int(np.argmin(dists_to_invaders))
        closest_dist = dists_to_invaders[closest_i]
        closest_idx = invaders[closest_i]

        # Penalty for being far from the closest invader when on own side
        if on_own_side:
            r = -0.008 * (closest_dist / field_diag)
            reward += r
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} far from invader: {r:.4f}")

        # Reward for closing distance toward the closest invader
        prev_opp_pos = np.asarray(prev_state["agent_position"][closest_idx], dtype=np.float64)
        prev_dist = float(np.linalg.norm(prev_pos - prev_opp_pos))
        delta = prev_dist - closest_dist
        if delta > 0 and moved > 1e-3:
            r = 0.25 * delta / field_diag
            reward += r
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} intercept invader: +{r:.4f}")

    # Defense: flag already taken — extra step penalty + stronger chase reward for the carrier
    if state["flag_taken"][team_i]:
        reward += -0.005
        for opp_idx in range(len(agents)):
            if opp_idx not in my_indices and not is_disabled(opp_idx) and state["agent_has_flag"][opp_idx]:
                opp_pos = np.asarray(state["agent_position"][opp_idx], dtype=np.float64)
                prev_opp_pos = np.asarray(prev_state["agent_position"][opp_idx], dtype=np.float64)
                curr_dist = np.linalg.norm(pos - opp_pos)
                prev_dist = np.linalg.norm(prev_pos - prev_opp_pos)
                delta = prev_dist - curr_dist
                if delta > 0 and moved > 1e-3:
                    r = 0.35 * delta / field_diag
                    reward += r
                    if REWARD_DEBUG:
                        print(f"[REWARD] {agent_id} chase flag carrier: +{r:.4f}")

    # Progress toward enemy flag — always active when not carrying and not tagged
    if has_flag == 0 and state["agent_is_tagged"][agent_index] == 0:
        opp_flag_curr = np.asarray(state["flag_position"][opp_team], dtype=np.float64)
        curr_dist = np.linalg.norm(pos - opp_flag_curr)
        prev_dist = np.linalg.norm(prev_pos - opp_flag_curr)
        delta = prev_dist - curr_dist
        if delta > 0 and moved > 1e-3:
            r = 0.15 * delta / field_diag
            reward += r
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} progress toward flag: +{r:.4f}")

    # Enemy proximity penalty: discourage attackers from running into enemies
    # Only fires when on enemy territory (not on own side), not carrying, not tagged.
    # danger_radius is a multiple of catch_radius so it scales with the game config.
    danger_radius = catch_radius * 3.0
    if has_flag == 0 and state["agent_is_tagged"][agent_index] == 0 and not on_own_side:
        opp_indices = [
            i for i in range(len(agents))
            if i not in my_indices and not is_disabled(i)
        ]
        for opp_i in opp_indices:
            opp_pos = np.asarray(state["agent_position"][opp_i], dtype=np.float64)
            dist_to_opp = float(np.linalg.norm(pos - opp_pos))
            if dist_to_opp < danger_radius:
                proximity_ratio = 1.0 - dist_to_opp / danger_radius  # 0 at edge → 1 at contact
                r = -0.015 * proximity_ratio
                reward += r
                if REWARD_DEBUG:
                    print(f"[REWARD] {agent_id} near enemy {opp_i}: {r:.4f}")

    return reward

### Add Custom Reward Functions Here ###
