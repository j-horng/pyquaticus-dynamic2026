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
    global _DEEP_PUSH_REWARDED
    parts = None
    if REWARD_PARTS_DEBUG:
        parts = {
            "self_tagged": 0.0,
            "close_enemy": 0.0,
            "oob": 0.0,
            "tag_enemy": 0.0,
            "tag_carrier": 0.0,
            "grab_individual": 0.0,
            "grab_team": 0.0,
            "cap_individual": 0.0,
            "cap_team": 0.0,
            "enemy_grab": 0.0,
            "enemy_cap": 0.0,
            "enemy_deep": 0.0,
            "deep_flee_75": 0.0,
            "deep_flee_625": 0.0,
            "deep_push": 0.0,
        }

        def _add_part(k: str, v: float):
            parts[k] = float(parts.get(k, 0.0)) + float(v)

    # ── Reward Values ──────────────────────────────────────────────────────
    R_TAG_ENEMY       =  1.0    # tagged any enemy
    R_TAG_CARRIER     =  3.0    # tagged the enemy flag carrier (stacks with R_TAG_ENEMY)
    R_GRAB_INDIVIDUAL =  2.0    # this agent grabbed the enemy flag
    R_GRAB_TEAM       =  0.3    # any ally grabbed the enemy flag (reduced)
    R_CAP_INDIVIDUAL  =  5.0    # this agent scored
    R_CAP_TEAM        =  1.0    # any ally scored
    R_DEEP_PUSH_BASE  =  0.5    # one-time reward for crossing 75% field depth on cooldown
    R_DEEP_PUSH_BONUS =  0.5    # extra bonus scaled by remaining cooldown fraction
    R_DEEP_FLEE_75    =  1.0    # one-time reward for flag carrier crossing back past 3/4 depth
    R_DEEP_FLEE_625   =  0.6    # one-time reward for flag carrier crossing back past 5/8 depth
    P_ENEMY_DEEP      = -1.0    # penalty when any enemy crosses past 1/4 depth into our side

    P_ENEMY_GRAB      = -1.0    # enemy grabbed our flag
    P_ENEMY_CAP       = -3.0    # enemy scored
    P_SELF_TAGGED     = -1.0    # this agent was newly tagged (individual penalty)
    P_CLOSE_ENEMY     = -0.05   # per-step penalty when too close to any enemy
    P_OOB             = -3.0    # agent went out of bounds
    CLOSE_ENEMY_MULT  =  1.5    # penalize when within this multiple of catch_radius
    # ──────────────────────────────────────────────────────────────────────

    reward = 0.0
    agent_index = agents.index(agent_id)

    # Skip disabled agents
    disabled = state.get("disabled_agents")
    if disabled is not None and len(disabled) > agent_index and bool(disabled[agent_index]):
        return 0.0

    # Only train Blue team
    if int(team) != int(Team.BLUE_TEAM):
        return 0.0

    enemy_team = 1 - int(team)

    # ── Commonly Reused State ──────────────────────────────────────────────
    pos            = np.asarray(state["agent_position"][agent_index], dtype=np.float64)
    prev_pos       = np.asarray(prev_state["agent_position"][agent_index], dtype=np.float64)
    had_flag       = bool(np.asarray(prev_state["agent_has_flag"][agent_index]).item())
    has_flag       = bool(np.asarray(state["agent_has_flag"][agent_index]).item())
    is_tagged      = bool(np.asarray(state["agent_is_tagged"][agent_index]).item())
    tagged_idx     = state["agent_made_tag"][agent_index]
    team_caps_up = state["captures"][int(team)] > prev_state["captures"][int(team)]
    enemy_caps_up = state["captures"][enemy_team] > prev_state["captures"][enemy_team]
    grabs_up = state["grabs"][int(team)] > prev_state["grabs"][int(team)]
    our_flag_now = bool(state["flag_taken"][int(team)])
    our_flag_prev = bool(prev_state["flag_taken"][int(team)])

    # If we're tagged, don't apply any shaping/penalties while returning.
    # Only apply the one-time "got tagged" penalty on the transition into tagged.
    prev_is_tagged = bool(np.asarray(prev_state["agent_is_tagged"][agent_index]).item())
    if is_tagged:
        if (not prev_is_tagged):
            reward += P_SELF_TAGGED
            if parts is not None:
                _add_part("self_tagged", P_SELF_TAGGED)
            if REWARD_DEBUG:
                print(f"[REWARD] {agent_id} got tagged: {P_SELF_TAGGED:+.3f}")
        if parts is not None:
            setattr(caps_and_grabs, "_last_parts", getattr(caps_and_grabs, "_last_parts", {}))
            caps_and_grabs._last_parts[agent_id] = parts
        return reward

    # ── Close to enemy penalty (1.5x tag radius) ───────────────────────────
    # Only apply this when we're in enemy territory; if we're safely on our own side,
    # don't punish defenders just because enemies are near the scrimmage line.
    try:
        agent_on_own_side = bool(np.asarray(state["agent_on_sides"][agent_index]).item())
    except Exception:
        agent_on_own_side = True
    if not agent_on_own_side:
        enemy_team_enum = Team.RED_TEAM if int(team) == int(Team.BLUE_TEAM) else Team.BLUE_TEAM
        enemy_inds = agent_inds_of_team.get(enemy_team_enum, []) if isinstance(agent_inds_of_team, dict) else []
        if isinstance(enemy_inds, np.ndarray):
            enemy_inds = enemy_inds.reshape(-1).tolist()
        if len(enemy_inds) > 0:
            disabled_all = state.get("disabled_agents")
            thresh2 = (float(CLOSE_ENEMY_MULT) * float(catch_radius)) ** 2
            for e_i in enemy_inds:
                ei = int(e_i)
                if disabled_all is not None and ei < len(disabled_all) and bool(disabled_all[ei]):
                    continue

                # Do not penalize proximity to tagged enemies.
                try:
                    e_tagged = bool(np.asarray(state["agent_is_tagged"][ei]).item())
                except Exception:
                    e_tagged = True
                if e_tagged:
                    continue

                # Do not penalize proximity to enemies whose tagging is on cooldown.
                try:
                    e_cd = float(np.asarray(state["agent_tagging_cooldown"][ei]).item())
                except Exception:
                    # Safe default: if we can't read cooldown, assume enemy can't tag.
                    e_cd = 0.0
                if e_cd < tagging_cooldown:
                    continue

                # Do not penalize proximity to enemies that are already on Blue's side.
                # `agent_on_sides` is defined relative to each agent's own team:
                # - For Red agents: False => they are on Blue side.
                try:
                    enemy_on_own_side = bool(np.asarray(state["agent_on_sides"][ei]).item())
                except Exception:
                    enemy_on_own_side = True
                if not enemy_on_own_side:
                    continue

                epos = np.asarray(state["agent_position"][ei], dtype=np.float64)
                if float(np.sum((pos - epos) ** 2)) < thresh2:
                    reward += P_CLOSE_ENEMY
                    if parts is not None:
                        _add_part("close_enemy", P_CLOSE_ENEMY)
                    if REWARD_DEBUG:
                        print(f"[REWARD] {agent_id} close enemy (<{CLOSE_ENEMY_MULT:.1f}x): {P_CLOSE_ENEMY:+.2f}")
                    break

    # ── Out of Bounds ──────────────────────────────────────────────────────
    if state["agent_oob"][agent_index] > prev_state["agent_oob"][agent_index]:
        reward += P_OOB
        if parts is not None:
            _add_part("oob", P_OOB)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} OOB: {P_OOB}")

    # ── Tagging ────────────────────────────────────────────────────────────
    if tagged_idx is not None:
        reward += R_TAG_ENEMY
        if parts is not None:
            _add_part("tag_enemy", R_TAG_ENEMY)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} tagged enemy: +{R_TAG_ENEMY}")

        if bool(np.asarray(prev_state["agent_has_flag"][tagged_idx]).item()):
            reward += R_TAG_CARRIER
            if parts is not None:
                _add_part("tag_carrier", R_TAG_CARRIER)
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} tagged carrier: +{R_TAG_CARRIER}")

    # ── Flag Grab ──────────────────────────────────────────────────────────
    if has_flag and not had_flag:
        reward += R_GRAB_INDIVIDUAL
        if parts is not None:
            _add_part("grab_individual", R_GRAB_INDIVIDUAL)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} grabbed flag: +{R_GRAB_INDIVIDUAL}")
    elif grabs_up:
        reward += R_GRAB_TEAM
        if parts is not None:
            _add_part("grab_team", R_GRAB_TEAM)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} ally grabbed flag: +{R_GRAB_TEAM}")

    # ── Capture ────────────────────────────────────────────────────────────
    if team_caps_up:
        reward += R_CAP_TEAM
        if parts is not None:
            _add_part("cap_team", R_CAP_TEAM)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} team capture: +{R_CAP_TEAM}")
        if had_flag:
            reward += R_CAP_INDIVIDUAL
            if parts is not None:
                _add_part("cap_individual", R_CAP_INDIVIDUAL)
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} individual capture: +{R_CAP_INDIVIDUAL}")

    # ── Enemy Events ───────────────────────────────────────────────────────
    if our_flag_now and not our_flag_prev:
        reward += P_ENEMY_GRAB
        if parts is not None:
            _add_part("enemy_grab", P_ENEMY_GRAB)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} enemy grabbed flag: {P_ENEMY_GRAB}")

    if enemy_caps_up:
        reward += P_ENEMY_CAP
        if parts is not None:
            _add_part("enemy_cap", P_ENEMY_CAP)
        if REWARD_DEBUG: print(f"[REWARD] {agent_id} enemy captured: {P_ENEMY_CAP}")

    # ── Enemy Deep Penetration Penalty ─────────────────────────────────────
    # Penalize once per enemy agent per incursion past 1/4 depth into our side.
    # Reset only if the enemy is tagged OR retreats all the way back past midfield (2/4).
    global _ENEMY_DEEP_PENALIZED
    field_w = float(env_size[0])
    enemy_team_enum = Team.RED_TEAM if int(team) == int(Team.BLUE_TEAM) else Team.BLUE_TEAM
    enemy_inds = (
        agent_inds_of_team.get(enemy_team_enum, [])
        if isinstance(agent_inds_of_team, dict)
        else []
    )
    for e_idx in enemy_inds:
        ei = int(e_idx)
        if ei < 0 or ei >= len(agents):
            continue
        e_id = agents[ei]
        e_pos = np.asarray(state["agent_position"][ei], dtype=np.float64)
        e_prev_pos = np.asarray(prev_state["agent_position"][ei], dtype=np.float64)
        e_tagged = bool(np.asarray(state["agent_is_tagged"][ei]).item())
        if e_tagged:
            _ENEMY_DEEP_PENALIZED[e_id] = False
            continue

        if int(team) == 0:
            # Blue on left; enemy (red) pushes left past 1/4 field width
            crossed = float(e_prev_pos[0]) > 0.25 * field_w >= float(e_pos[0])
        else:
            # Blue on right; enemy pushes right past 3/4 field width
            crossed = float(e_prev_pos[0]) < 0.75 * field_w <= float(e_pos[0])

        if crossed and not _ENEMY_DEEP_PENALIZED.get(e_id, False):
            reward += P_ENEMY_DEEP
            if parts is not None:
                _add_part("enemy_deep", P_ENEMY_DEEP)
            _ENEMY_DEEP_PENALIZED[e_id] = True
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} enemy {e_id} crossed deep: {P_ENEMY_DEEP}")

        if _ENEMY_DEEP_PENALIZED.get(e_id, False):
            # Reset once enemy retreats back to their own side past midfield (2/4).
            if int(team) == 0:
                retreated = float(e_pos[0]) > 0.5 * field_w
            else:
                retreated = float(e_pos[0]) < 0.5 * field_w
            if retreated:
                _ENEMY_DEEP_PENALIZED[e_id] = False

    # ── Deep Flee (one-time per carry, flag carrier retreating toward home) ─
    global _DEEP_FLEE_75_REWARDED, _DEEP_FLEE_625_REWARDED
    if has_flag:
        field_w = float(env_size[0])
        if int(team) == 0:
            # Blue on left; "fleeing back" means decreasing x
            crossed_75  = float(prev_pos[0]) > 0.75 * field_w >= float(pos[0])
            crossed_625 = float(prev_pos[0]) > 0.625 * field_w >= float(pos[0])
        else:
            crossed_75  = float(prev_pos[0]) < 0.25 * field_w <= float(pos[0])
            crossed_625 = float(prev_pos[0]) < 0.375 * field_w <= float(pos[0])

        if crossed_75 and not _DEEP_FLEE_75_REWARDED.get(agent_id, False):
            reward += R_DEEP_FLEE_75
            if parts is not None:
                _add_part("deep_flee_75", R_DEEP_FLEE_75)
            _DEEP_FLEE_75_REWARDED[agent_id] = True
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} deep flee 3/4: +{R_DEEP_FLEE_75}")
        if crossed_625 and not _DEEP_FLEE_625_REWARDED.get(agent_id, False):
            reward += R_DEEP_FLEE_625
            if parts is not None:
                _add_part("deep_flee_625", R_DEEP_FLEE_625)
            _DEEP_FLEE_625_REWARDED[agent_id] = True
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} deep flee 5/8: +{R_DEEP_FLEE_625}")
    else:
        # Reset flee flags when agent no longer has flag (captured or tagged)
        _DEEP_FLEE_75_REWARDED[agent_id] = False
        _DEEP_FLEE_625_REWARDED[agent_id] = False

    # ── Deep Push (one-time per cooldown window, while on tag cooldown) ────
    cd = float(state["agent_tagging_cooldown"][agent_index])
    on_cooldown = cd < tagging_cooldown
    cooldown_progress = float(np.clip(cd / tagging_cooldown, 0.0, 1.0)) if tagging_cooldown > 0 else 1.0

    if on_cooldown and not has_flag and not is_tagged:
        field_w   = float(env_size[0])
        threshold = 0.75 * field_w
        crossed   = (float(prev_pos[0]) < threshold <= float(pos[0])) if int(team) == 0 \
                else (float(prev_pos[0]) > (field_w - threshold) >= float(pos[0]))
        if crossed and not _DEEP_PUSH_REWARDED.get(agent_id, False):
            deep_reward = R_DEEP_PUSH_BASE + R_DEEP_PUSH_BONUS * (1.0 - cooldown_progress)
            reward += deep_reward
            if parts is not None:
                _add_part("deep_push", deep_reward)
            _DEEP_PUSH_REWARDED[agent_id] = True
            if REWARD_DEBUG: print(f"[REWARD] {agent_id} deep push: +{deep_reward:.3f}")
    if not on_cooldown:
        _DEEP_PUSH_REWARDED[agent_id] = False

    if parts is not None:
        setattr(caps_and_grabs, "_last_parts", getattr(caps_and_grabs, "_last_parts", {}))
        caps_and_grabs._last_parts[agent_id] = parts
    return reward



### Add Custom Reward Functions Here ###
