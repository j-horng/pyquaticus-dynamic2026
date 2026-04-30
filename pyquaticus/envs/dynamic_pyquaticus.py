# SPDX-License-Identifier: BSD-3-Clause
"""
Dynamic PyQuaticus Environment

Extends PyQuaticusEnv to support:
- Variable team sizes (1–6 per team) randomized at each reset
- Mid-game agent removal (captured/destroyed) via disabled_agents
- Mid-game agent spawning (reinforcements)
- disabled_agents state for tracking active agents
"""

import random
from typing import Optional, Union

import numpy as np

from pyquaticus.config import ACTION_MAP
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.structs import Team
from pyquaticus.utils.utils import closest_point_on_line, vec_to_mag_heading


class DynamicPyQuaticusEnv(PyQuaticusEnv):
    """
    PyQuaticus environment with dynamic team sizes and mid-game agent changes.

    - team_size_range: (min, max) agents per team at reset (default (1, 6))
    - tag_removes_agent: if True, tagged agents are disabled (removed from game)
    - reinforcement_interval: steps between reinforcement spawns (0 = disabled)
    - reinforcement_prob: probability of spawning a reinforcement when interval hits
    - dynamic_toggle_on: if True, periodically roll random removals (only tagged agents;
      at least one active per team) and random revivals of disabled agents
    - dynamic_toggle_interval: env steps between those rolls (0 = off)
    - dynamic_toggle_remove_prob / dynamic_toggle_add_prob: per-roll probabilities
    - stationary_red_block_anchor: optional "midfield" | "topfield" | "bottomfield" (with stationary_red_mode)
      to place active stationary Reds on two slots of the default-init forward spawn row (not arena center).
    - stationary_red_block_anchor_random: if True, each reset uniformly picks one of those three row pairs.
    """

    def __init__(
        self,
        team_size_range: tuple[int, int] = (1, 6),
        tag_removes_agent: bool = False,
        reinforcement_interval: int = 0,
        reinforcement_prob: float = 0.5,
        dynamic_toggle_on: bool = False,
        dynamic_toggle_interval: int = 200,
        dynamic_toggle_remove_prob: float = 0.5,
        dynamic_toggle_add_prob: float = 0.5,
        stationary_red_attack_easy_slot0: bool = False,
        action_space: Union[str, list[str], dict[str, str]] = "discrete",
        reward_config: dict = None,
        config_dict=None,
        render_mode: Optional[str] = None,
    ):
        max_team_size = team_size_range[1]
        _cfg = config_dict or {}
        super().__init__(
            team_size=max_team_size,
            action_space=action_space,
            reward_config=reward_config,
            config_dict=_cfg,
            render_mode=render_mode,
        )

        self.team_size_range = team_size_range
        self.tag_removes_agent = tag_removes_agent
        self.reinforcement_interval = reinforcement_interval
        self.reinforcement_prob = reinforcement_prob
        self.dynamic_toggle_on = bool(dynamic_toggle_on)
        self.dynamic_toggle_interval = max(0, int(dynamic_toggle_interval))
        self.dynamic_toggle_remove_prob = float(
            max(0.0, min(1.0, dynamic_toggle_remove_prob))
        )
        self.dynamic_toggle_add_prob = float(max(0.0, min(1.0, dynamic_toggle_add_prob)))
        self.num_blue_active = max_team_size
        self.num_red_active = max_team_size
        self._step_count = 0
        self.red_dummy_mode = _cfg.get("red_dummy_mode", False)
        self.stationary_red_mode = _cfg.get("stationary_red_mode", False)
        # Prefer explicit ctor arg; keep config fallback for compatibility.
        self.stationary_red_attack_easy_slot0 = bool(
            stationary_red_attack_easy_slot0 or _cfg.get("stationary_red_attack_easy_slot0", False)
        )
        # Optional: force a fixed number of Red agents active at reset (others disabled).
        self.force_num_red_active = _cfg.get("force_num_red_active", None)
        # When stationary_red_mode: how many Red slots are active (stationary); remainder disabled.
        _sra = int(_cfg.get("stationary_red_active", 2))
        self.stationary_red_active = max(0, min(_sra, max_team_size))
        self.stationary_red_active_random = bool(_cfg.get("stationary_red_active_random", False))
        _srr = _cfg.get("stationary_red_random_range", None)
        if _srr is not None:
            try:
                a, b = int(_srr[0]), int(_srr[1])
                a = max(1, min(a, max_team_size))
                b = max(a, min(b, max_team_size))
                self.stationary_red_random_range = (a, b)
            except (TypeError, IndexError, ValueError):
                self.stationary_red_random_range = None
        else:
            self.stationary_red_random_range = None
        _anchor_raw = _cfg.get("stationary_red_block_anchor")
        if _anchor_raw is None and _cfg.get("stationary_red_midfield_spawn"):
            _anchor_raw = "midfield"
        if _anchor_raw is None and _cfg.get("stationary_red_center_spawn"):
            _anchor_raw = "midfield"
        if isinstance(_anchor_raw, str):
            _a = _anchor_raw.strip().lower()
            self.stationary_red_block_anchor = _a if _a in ("midfield", "topfield", "bottomfield") else None
        else:
            self.stationary_red_block_anchor = None
        self.stationary_red_block_anchor_random = bool(_cfg.get("stationary_red_block_anchor_random", False))
        if self.stationary_red_block_anchor_random:
            self.stationary_red_block_anchor = None
        if self.stationary_red_attack_easy_slot0:
            # Fixed composition for mixed mode: 1 easy attacker + 2 stationary.
            self.stationary_red_active = min(3, max_team_size)
            self.stationary_red_active_random = False
            self.stationary_red_random_range = None
        # Spawn-row block (mid/top/bottom or random among them) only supports two stationary slots.
        self._stationary_red_spawn_row_block = self.stationary_red_block_anchor_random or (
            self.stationary_red_block_anchor is not None
        )
        if self._stationary_red_spawn_row_block:
            block_cap = 3 if self.stationary_red_attack_easy_slot0 else 2
            self.stationary_red_active = min(self.stationary_red_active, block_cap)
            if self.stationary_red_random_range is not None:
                a, b = self.stationary_red_random_range
                b = min(b, block_cap)
                a = max(1, min(a, b))
                self.stationary_red_random_range = (a, b)
        if self.stationary_red_attack_easy_slot0 and self.stationary_red_active <= 0:
            # Ensure the designated attacker slot can be active.
            self.stationary_red_active = 1

        for player in self.players.values():
            if not hasattr(player, "is_disabled"):
                player.is_disabled = False

    def _register_state_elements(self, num_on_team, num_obstacles):
        """Override to add is_disabled to observation space."""
        agent_obs_normalizer, global_state_normalizer = super()._register_state_elements(num_on_team, num_obstacles)
        max_bool, min_bool = [1.0], [0.0]
        agent_obs_normalizer.register("is_disabled", max_bool, min_bool)
        for i in range(num_on_team - 1):
            agent_obs_normalizer.register((f"teammate_{i}", "is_disabled"), max_bool, min_bool)
        for i in range(num_on_team):
            agent_obs_normalizer.register((f"opponent_{i}", "is_disabled"), max_bool, min_bool)
        for player in self.players.values():
            global_state_normalizer.register((player.id, "is_disabled"), max_bool, min_bool)
        return agent_obs_normalizer, global_state_normalizer

    def state_to_obs(self, agent_id, normalize=True):
        """Override to add is_disabled to observations."""
        obs, unnorm = super().state_to_obs(agent_id, normalize=False)
        disabled = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        agent = self.players[agent_id]
        obs["is_disabled"] = float(disabled[agent.idx])
        for team in [agent.team, (Team.RED_TEAM if agent.team == Team.BLUE_TEAM else Team.BLUE_TEAM)]:
            dif_agents = [a for a in self.agents_of_team[team] if a.id != agent.id]
            for i, dif_agent in enumerate(dif_agents):
                entry_name = f"teammate_{i}" if team == agent.team else f"opponent_{i}"
                obs[(entry_name, "is_disabled")] = float(disabled[dif_agent.idx])
        if normalize:
            return self.agent_obs_normalizer.normalized(obs), obs
        return obs, None

    def state_to_global_state(self, normalize=True):
        gs = super().state_to_global_state(normalize=False)
        disabled = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        for i, aid in enumerate(self.agents):
            gs[(aid, "is_disabled")] = float(disabled[i]) if i < len(disabled) else 0.0
        if normalize:
            return self.global_state_normalizer.normalized(gs)
        return gs

    def _set_initial_disabled(self, active_blue_inds: list[int], active_red_inds: list[int]):
        """Disable agents not in the active set.

        Note: agent indices are 0..num_blue-1 for Blue and num_blue..num_agents-1 for Red.
        """
        disabled = np.ones(self.num_agents, dtype=bool)
        for i in active_blue_inds:
            disabled[int(i)] = False
        for i in active_red_inds:
            disabled[int(i)] = False
        self.state["disabled_agents"] = disabled
        for i, player in enumerate(self.players.values()):
            player.is_disabled = bool(disabled[i])

    def _default_init_forward_spawn_slots_for_team(self, team: Team) -> list[np.ndarray]:
        """First-row spawn slots (k=0,1,2) matching PyQuaticusEnv._generate_agent_starts default_init geometry."""
        team_idx = int(team)
        flag_home = np.asarray(self.flags[team_idx].home, dtype=np.float64).reshape(-1)[:2]
        closest_scrim_line_point = closest_point_on_line(*self.scrimmage_coords, flag_home)
        halfway_point = (flag_home + closest_scrim_line_point) / 2.0
        mag, _ = vec_to_mag_heading(halfway_point - flag_home)
        team_inds = self.agent_inds_of_team[team]
        max_team_radius = float(np.max(self.agent_radius[team_inds]))
        if mag < (self.flag_keepout_radius + max_team_radius):
            mid = (np.asarray(self.flags[int(Team.BLUE_TEAM)].home) + np.asarray(self.flags[int(Team.RED_TEAM)].home)) / 2.0
            return [mid.copy(), mid.copy(), mid.copy()]
        spawn_line_env_intersection_1 = self._get_polygon_intersection(halfway_point, self.scrimmage_vec, self.env_corners)[1]
        spawn_line_env_intersection_2 = self._get_polygon_intersection(halfway_point, -self.scrimmage_vec, self.env_corners)[1]
        int1 = np.asarray(spawn_line_env_intersection_1, dtype=np.float64).reshape(-1)[:2]
        int2 = np.asarray(spawn_line_env_intersection_2, dtype=np.float64).reshape(-1)[:2]
        d = int2 - int1
        spawn_line_mag = float(np.linalg.norm(d))
        u = d / max(spawn_line_mag, 1e-9)
        slots = []
        ts = int(self.team_size)
        for k in range(min(3, ts)):
            if ts <= 3:
                frac = (k + 1) / float(ts + 1)
            else:
                frac = (k + 1) / 4.0
            pos = int1 + spawn_line_mag * frac * u
            gi = int(team_inds[k])
            ar = float(self.agent_radius[gi])
            pos[0] = float(np.clip(pos[0], 2.0 * ar, float(self.env_size[0]) - 2.0 * ar))
            pos[1] = float(np.clip(pos[1], 2.0 * ar, float(self.env_size[1]) - 2.0 * ar))
            slots.append(pos.copy())
        while len(slots) < 3:
            slots.append(slots[-1].copy())
        return slots

    def _place_stationary_red_block_arena_fallback(self, active_red_inds: list[int], anchor: str) -> None:
        """Legacy arena-relative cluster (used for gps_env where spawn-row math differs)."""
        blue_flag = np.asarray(self.flags[int(Team.BLUE_TEAM)].home, dtype=np.float64)
        red_flag = np.asarray(self.flags[int(Team.RED_TEAM)].home, dtype=np.float64)
        mid = (blue_flag + red_flag) / 2.0
        delta = red_flag - blue_flag
        n = float(np.linalg.norm(delta))
        if n < 1e-6:
            u = np.array([1.0, 0.0], dtype=np.float64)
        else:
            u = delta / n
        v = np.array([-u[1], u[0]], dtype=np.float64)
        ar_max = float(np.max(self.agent_radius))
        sep = max(12.0, 3.5 * ar_max)
        margin = max(8.0, 2.0 * ar_max)
        ex, ey = float(self.env_size[0]), float(self.env_size[1])
        span = max(n, 1e-6)
        half_sep = min(0.5 * sep, 0.14 * span)
        k = len(active_red_inds)
        if anchor == "midfield":
            spread = v
            base = mid.copy()
        elif anchor == "topfield":
            spread = u
            base = np.array([float(mid[0]), ey - margin], dtype=np.float64)
            base[1] = float(np.clip(base[1] + 0.14 * ey, margin, ey - margin))
        else:
            spread = u
            base = np.array([float(mid[0]), margin], dtype=np.float64)
            base[1] = float(np.clip(base[1] - 0.14 * ey, margin, ey - margin))
        for j, idx in enumerate(sorted(int(i) for i in active_red_inds)):
            offset = (j - (k - 1) / 2.0) * (2.0 * half_sep)
            pos = base + spread * offset
            pos[0] = float(np.clip(pos[0], margin, ex - margin))
            pos[1] = float(np.clip(pos[1], margin, ey - margin))
            self.state["agent_position"][idx] = pos.copy()
            self.state["prev_agent_position"][idx] = pos.copy()
            aid = self.agents[idx]
            self.players[aid].pos = pos.copy()
            self.players[aid].prev_pos = pos.copy()

    def _place_stationary_red_block(self, active_red_inds: list[int], anchor: str) -> None:
        """Place stationary Reds: bottom/top use the lower/higher adjacent-slot centroid, nudged in ±Y, grouped like midfield (c ± u_hat·half_sep)."""
        if not active_red_inds or anchor not in ("midfield", "topfield", "bottomfield"):
            return
        if getattr(self, "gps_env", False):
            self._place_stationary_red_block_arena_fallback(active_red_inds, anchor)
            return
        slots = self._default_init_forward_spawn_slots_for_team(Team.RED_TEAM)
        p0, p1, p2 = slots[0], slots[1], slots[2]
        u_hat = p2 - p0
        nu = float(np.linalg.norm(u_hat))
        u_hat = u_hat / max(nu, 1e-9)
        red_inds = self.agent_inds_of_team[Team.RED_TEAM]
        ar_max = float(np.max(self.agent_radius[red_inds]))
        sep = max(12.0, 3.5 * ar_max)
        row_span = float(np.linalg.norm(p2 - p0))
        # Visual: env_to_screen flips Y so larger world-y is toward the top of the window.
        # bottomfield / topfield = adjacent pair on the spawn row whose midpoint is lower / higher Y.
        m01 = 0.5 * (float(p0[1]) + float(p1[1]))
        m12 = 0.5 * (float(p1[1]) + float(p2[1]))
        y_eps = 1e-3 * max(float(self.env_size[1]), 1.0)

        def _pair_bottom_top_tiebreak(want_bottom: bool) -> tuple[np.ndarray, np.ndarray]:
            if abs(m01 - m12) >= y_eps:
                if want_bottom:
                    return (p0.copy(), p1.copy()) if m01 < m12 else (p1.copy(), p2.copy())
                return (p0.copy(), p1.copy()) if m01 > m12 else (p1.copy(), p2.copy())
            mn0 = min(float(p0[1]), float(p1[1]))
            mn1 = min(float(p1[1]), float(p2[1]))
            mx0 = max(float(p0[1]), float(p1[1]))
            mx1 = max(float(p1[1]), float(p2[1]))
            if want_bottom:
                return (p0.copy(), p1.copy()) if mn0 < mn1 else (p1.copy(), p2.copy())
            return (p0.copy(), p1.copy()) if mx0 > mx1 else (p1.copy(), p2.copy())

        # Same along-row grouping as midfield: two poses at c ± u_hat * half_sep (then clip).
        half_sep = min(0.5 * sep, 0.14 * max(row_span, 1e-6))
        ey_sz = float(self.env_size[1])
        shift_y = min(0.14 * ey_sz, 0.52 * max(row_span, 1e-6))

        if anchor == "bottomfield":
            pa, pb = _pair_bottom_top_tiebreak(True)
            c = 0.5 * (pa + pb) + np.array([0.0, -shift_y], dtype=np.float64)
            a = c - u_hat * half_sep
            b = c + u_hat * half_sep
        elif anchor == "topfield":
            pa, pb = _pair_bottom_top_tiebreak(False)
            c = 0.5 * (pa + pb) + np.array([0.0, shift_y], dtype=np.float64)
            a = c - u_hat * half_sep
            b = c + u_hat * half_sep
        else:
            c = p1.copy()
            a = c - u_hat * half_sep
            b = c + u_hat * half_sep

        pair_pts = [a, b]
        sorted_idx = sorted(int(i) for i in active_red_inds)
        n = len(sorted_idx)
        margin = max(8.0, 2.0 * ar_max)
        ex, ey = float(self.env_size[0]), float(self.env_size[1])
        for j, idx in enumerate(sorted_idx):
            if n <= 1:
                pos = pair_pts[0].copy()
            else:
                t = j / float(n - 1)
                pos = (1.0 - t) * pair_pts[0] + t * pair_pts[1]
            pos[0] = float(np.clip(pos[0], margin, ex - margin))
            pos[1] = float(np.clip(pos[1], margin, ey - margin))
            self.state["agent_position"][idx] = pos.copy()
            self.state["prev_agent_position"][idx] = pos.copy()
            aid = self.agents[idx]
            self.players[aid].pos = pos.copy()
            self.players[aid].prev_pos = pos.copy()

    def _tuck_disabled_reds_behind_flag(self) -> None:
        """Move disabled Red agents off the spawn row (same idea as red_dummy_mode) so active block pair is visible."""
        margin = 5.0
        red_flag_home = np.array(self.flags[int(Team.RED_TEAM)].home, dtype=np.float64)
        side_pos = red_flag_home + np.array([margin, 0.0], dtype=np.float64)
        if side_pos[0] >= float(self.env_size[0]):
            side_pos[0] = float(self.env_size[0]) - margin
        side_pos[1] = np.clip(side_pos[1], margin, float(self.env_size[1]) - margin)
        disabled = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        for red_agent_idx in range(self.num_blue, self.num_agents):
            if not bool(disabled[red_agent_idx]):
                continue
            self.state["agent_position"][red_agent_idx] = side_pos.copy()
            self.state["prev_agent_position"][red_agent_idx] = side_pos.copy()
            aid = self.agents[red_agent_idx]
            self.players[aid].pos = np.array(side_pos, dtype=np.float64)
            self.players[aid].prev_pos = np.array(side_pos, dtype=np.float64)

    def _finalize_dynamic_reset_observations(self) -> None:
        """Recompute on_own_side and observation buffers after dynamic reset mutates poses or disabled_agents."""
        for i, player in enumerate(self.players.values()):
            own_side = self._check_on_sides(player.pos, player.team)
            self.state["agent_on_sides"][i] = bool(np.asarray(own_side).reshape(-1)[0])
        self._set_player_attributes_from_state()
        self._update_dist_bearing_to_obstacles()
        if self.lidar_obs:
            self._update_lidar()
        for agent_id in self.agents:
            reset_obs, reset_unnorm_obs = self.state_to_obs(agent_id, self.normalize_obs)
            self.state["obs_hist_buffer"][agent_id] = np.array(self.obs_hist_buffer_len * [reset_obs])
            if self.normalize_obs:
                self.state["unnorm_obs_hist_buffer"][agent_id] = np.array(self.obs_hist_buffer_len * [reset_unnorm_obs])
        self.state["global_state_hist_buffer"] = np.array(
            self.state_hist_buffer_len * [self.state_to_global_state(self.normalize_state)]
        )

    def reset(self, seed=None, options: Optional[dict] = None):
        """Reset with randomized team sizes."""
        super().reset(seed=seed, options=options)
        self._step_count = 0

        min_size, max_size = self.team_size_range
        if self.red_dummy_mode:
            self.num_blue_active = random.randint(min_size, max_size)  # match team_size_range; Red dummy stays 0
            self.num_red_active = 0  # No Red agents; Blue plays alone (capture the flag only)
        else:
            self.num_blue_active = random.randint(min_size, max_size)
            if self.force_num_red_active is None:
                self.num_red_active = random.randint(min_size, max_size)
            else:
                self.num_red_active = max(0, min(self.num_red, int(self.force_num_red_active)))

        if "disabled_agents" not in self.state:
            self.state["disabled_agents"] = np.zeros(self.num_agents, dtype=bool)

        # Randomize *which* specific agents are active so we don't always activate the lowest indices.
        # This prevents systematic bias like "active agents are always at the top of the list".
        blue_pool = list(range(self.num_blue))
        red_pool = list(range(self.num_blue, self.num_agents))
        active_blue_inds = blue_pool if self.num_blue_active >= self.num_blue else random.sample(blue_pool, k=self.num_blue_active)
        active_red_inds = [] if self.num_red_active <= 0 else (
            red_pool if self.num_red_active >= self.num_red else random.sample(red_pool, k=self.num_red_active)
        )
        self._set_initial_disabled(active_blue_inds, active_red_inds)
        self.state["num_blue_active"] = self.num_blue_active
        self.state["num_red_active"] = self.num_red_active
        self.state["active_blue_inds"] = np.array(active_blue_inds, dtype=np.int64)
        self.state["active_red_inds"] = np.array(active_red_inds, dtype=np.int64)

        # In red_dummy_mode, all Red agents are disabled; place them in-bounds behind the Red flag
        # (to the right of the flag, same side) so they're out of the way
        if self.red_dummy_mode:
            margin = 5.0
            red_flag_home = np.array(self.flags[int(Team.RED_TEAM)].home, dtype=np.float64)
            # Behind the flag = right of the flag (further toward Red's back edge), stay in bounds
            side_pos = red_flag_home + np.array([margin, 0.0], dtype=np.float64)
            if side_pos[0] >= float(self.env_size[0]):
                side_pos[0] = float(self.env_size[0]) - margin
            side_pos[1] = np.clip(side_pos[1], margin, float(self.env_size[1]) - margin)
            for red_agent_idx in range(self.num_blue, self.num_agents):
                self.state["agent_position"][red_agent_idx] = side_pos.copy()
                self.state["prev_agent_position"][red_agent_idx] = side_pos.copy()
                self.players[self.agents[red_agent_idx]].pos = np.array(side_pos, dtype=np.float64)
                self.players[self.agents[red_agent_idx]].prev_pos = np.array(side_pos, dtype=np.float64)

        if self.stationary_red_mode:
            red_inds = list(range(self.num_blue, self.num_agents))
            if self.stationary_red_attack_easy_slot0:
                k = min(3, len(red_inds))
            elif self.stationary_red_active_random:
                if self.stationary_red_random_range is not None:
                    lo = max(1, min(self.stationary_red_random_range[0], len(red_inds)))
                    hi = min(self.stationary_red_random_range[1], len(red_inds))
                else:
                    lo = max(1, int(min_size))
                    hi = min(int(max_size), len(red_inds))
                k = 0 if hi < lo else random.randint(lo, hi)
            else:
                k = max(0, min(self.stationary_red_active, len(red_inds)))
            if self._stationary_red_spawn_row_block:
                block_cap = 3 if self.stationary_red_attack_easy_slot0 else 2
                k = min(k, block_cap, len(red_inds))
            if k <= 0:
                active_red = []
            elif k >= len(red_inds):
                active_red = red_inds
            else:
                if self.stationary_red_attack_easy_slot0 and k > 0 and len(red_inds) > 0:
                    # Force slot-0 red (agent_{num_blue}) into active set so mixed stationary+attacker mode is guaranteed.
                    anchor = int(red_inds[0])
                    remaining_pool = [idx for idx in red_inds if idx != anchor]
                    active_red = [anchor]
                    if k > 1:
                        active_red.extend(random.sample(remaining_pool, k=k - 1))
                else:
                    active_red = random.sample(red_inds, k=k)
            self._set_initial_disabled(active_blue_inds, active_red)
            self.state["num_blue_active"] = int(len(active_blue_inds))
            self.state["num_red_active"] = int(len(active_red))
            self.state["active_blue_inds"] = np.array(active_blue_inds, dtype=np.int64)
            self.state["active_red_inds"] = np.array(active_red, dtype=np.int64)

        block_anchor_episode = None
        if self.stationary_red_block_anchor_random and self.stationary_red_mode:
            active_red_list = [int(i) for i in np.asarray(self.state.get("active_red_inds", [])).reshape(-1)]
            block_anchor_episode = random.choice(["midfield", "topfield", "bottomfield"])
            self._place_stationary_red_block(active_red_list, block_anchor_episode)
        elif self.stationary_red_block_anchor and self.stationary_red_mode:
            active_red_list = [int(i) for i in np.asarray(self.state.get("active_red_inds", [])).reshape(-1)]
            block_anchor_episode = self.stationary_red_block_anchor
            self._place_stationary_red_block(active_red_list, block_anchor_episode)

        if self.stationary_red_mode and (
            self.stationary_red_block_anchor_random or self.stationary_red_block_anchor
        ):
            self._tuck_disabled_reds_behind_flag()

        self.state["blue_oob_count"] = 0
        self.state["red_oob_count"] = 0

        self._finalize_dynamic_reset_observations()

        obs = {aid: self._history_to_obs(aid, "obs_hist_buffer") for aid in self.players}
        global_state = self._history_to_state()
        disabled_agents = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        num_blue_active = int(np.sum(~disabled_agents[: self.num_blue]))
        num_red_active = int(np.sum(~disabled_agents[self.num_blue : self.num_agents]))
        info = {
            aid: {
                "global_state": global_state,
                "num_blue_active": num_blue_active,
                "num_red_active": num_red_active,
                "disabled_agents": disabled_agents,
                **(
                    {"stationary_red_block_anchor": block_anchor_episode}
                    if block_anchor_episode is not None
                    else {}
                ),
            }
            for aid in self.players
        }
        for aid in self.agents:
            if self.normalize_obs:
                info[aid]["unnorm_obs"] = self._history_to_obs(aid, "unnorm_obs_hist_buffer")
        return obs, info

    def step(self, raw_action_dict):
        """Step with no-op for disabled agents, tag removal, and reinforcements."""
        NO_OP_INDEX = len(ACTION_MAP) - 1
        # Patch actions for disabled agents (discrete no-op = last ACTION_MAP entry, continuous = [0,0])
        patched = dict(raw_action_dict)
        for i, player in enumerate(self.players.values()):
            if getattr(player, "is_disabled", False):
                if self.act_space_str.get(player.id, "discrete") == "continuous":
                    patched[player.id] = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    patched[player.id] = NO_OP_INDEX

        if self.stationary_red_mode:
            # Force Red to no-op so they stay in place (even if "active").
            for player in self.players.values():
                if int(player.team) != int(Team.RED_TEAM):
                    continue
                if self.stationary_red_attack_easy_slot0 and int(player.idx) == int(self.num_blue):
                    # Allow one designated Red attacker policy to act.
                    continue
                if self.act_space_str.get(player.id, "discrete") == "continuous":
                    patched[player.id] = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    patched[player.id] = NO_OP_INDEX

        prev_oob = np.asarray(self.state["agent_oob"], dtype=bool).copy()
        disabled_before = np.asarray(
            self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool)), dtype=bool
        )

        obs, rewards, terminated, truncated, info = super().step(patched)
        self._step_count += 1

        new_oob = np.asarray(self.state["agent_oob"], dtype=bool)
        for i in range(self.num_agents):
            if bool(disabled_before[i]):
                continue
            if bool(new_oob[i]) and not bool(prev_oob[i]):
                if i < self.num_blue:
                    self.state["blue_oob_count"] += 1
                else:
                    self.state["red_oob_count"] += 1

        # Tag removes agent (disable)
        if self.tag_removes_agent:
            for i, player in enumerate(self.players.values()):
                if player.is_tagged and not getattr(player, "is_disabled", False):
                    self.state["disabled_agents"][i] = True
                    player.is_disabled = True
                    if i < self.num_blue:
                        self.num_blue_active = max(0, self.num_blue_active - 1)
                    else:
                        self.num_red_active = max(0, self.num_red_active - 1)
                    self.state["num_blue_active"] = self.num_blue_active
                    self.state["num_red_active"] = self.num_red_active

        # Reinforcement spawn
        if self.reinforcement_interval > 0 and self._step_count % self.reinforcement_interval == 0:
            if self._step_count > 0 and random.random() < self.reinforcement_prob:
                self._spawn_reinforcement()

        if (
            self.dynamic_toggle_on
            and self.dynamic_toggle_interval > 0
            and self._step_count % self.dynamic_toggle_interval == 0
            and self._step_count > 0
        ):
            self._dynamic_toggle_random_roster_tick()

        disabled_agents = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        num_blue_active = int(np.sum(~disabled_agents[: self.num_blue]))
        num_red_active = int(np.sum(~disabled_agents[self.num_blue : self.num_agents]))
        for aid in self.agents:
            info[aid]["num_blue_active"] = num_blue_active
            info[aid]["num_red_active"] = num_red_active
            info[aid]["disabled_agents"] = disabled_agents

        return obs, rewards, terminated, truncated, info

    def _check_flag_pickups(self):
        """Override so disabled agents cannot pick up the flag."""
        disabled = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        for i, player in enumerate(self.players.values()):
            if disabled[i]:
                continue
            team_idx = int(player.team)
            other_team_idx = int(not team_idx)
            if not (player.has_flag or self.flags[other_team_idx].taken) and not player.on_own_side and not player.is_tagged:
                flag_pos = self.flags[other_team_idx].pos
                flag_distance = self.get_distance_between_2_points(player.pos, flag_pos)
                if flag_distance < self.catch_radius:
                    player.has_flag = True
                    self.state["agent_has_flag"][i] = 1
                    self.flags[other_team_idx].taken = True
                    self.flags[other_team_idx].pos = np.array(player.pos)
                    self.state["flag_position"][other_team_idx] = self.flags[other_team_idx].pos
                    self.state["flag_taken"][other_team_idx] = 1
                    self.state["grabs"][team_idx] += 1
                    self.game_events[player.team]["grabs"] += 1

    def _check_agent_made_tag(self):
        """Override so disabled agents cannot tag or be tagged."""
        disabled = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        self.state["agent_made_tag"] = [None] * self.num_agents
        for player in self.players.values():
            if disabled[player.idx]:
                continue
            if not (
                player.on_own_side and not player.oob and not player.is_tagged
                and player.tagging_cooldown == self.tagging_cooldown
                and not getattr(player, "is_disabled", False)
            ):
                continue
            for other_player in self.players.values():
                if disabled[other_player.idx]:
                    continue
                if (
                    not other_player.on_own_side and not other_player.is_tagged
                    and other_player.team != player.team
                ):
                    agent_distance = self.get_distance_between_2_points(player.pos, other_player.pos)
                    if agent_distance < self.catch_radius:
                        team_idx = int(player.team)
                        other_team_idx = int(other_player.team)
                        other_player.is_tagged = True
                        self.state["agent_is_tagged"][other_player.idx] = 1
                        self.state["agent_made_tag"][player.idx] = other_player.idx
                        self.state["tags"][team_idx] += 1
                        self.game_events[player.team]["tags"] += 1
                        if other_player.has_flag:
                            other_player.has_flag = False
                            self.state["agent_has_flag"][other_player.idx] = 0
                            self.flags[team_idx].reset()
                            self.state["flag_position"][team_idx] = self.flags[team_idx].pos
                            self.state["flag_taken"][team_idx] = 0
                        player.tagging_cooldown = 0.0
                        self.state["agent_tagging_cooldown"][player.idx] = 0.0
                        break

    @staticmethod
    def _player_eliminated_on_field(player) -> bool:
        """Tagged agents are treated as eliminated on the field (can be roster-removed)."""
        return bool(getattr(player, "is_tagged", False))

    def _try_random_remove_eliminated(self) -> None:
        """Disable one random tagged, non-disabled agent if the team would still have >=1 active."""
        disabled = self.state["disabled_agents"]
        teams = [Team.BLUE_TEAM, Team.RED_TEAM]
        random.shuffle(teams)
        for team in teams:
            if self.red_dummy_mode and team == Team.RED_TEAM:
                continue
            inds = self.agent_inds_of_team[team]
            active = [i for i in inds if not disabled[i]]
            if len(active) <= 1:
                continue
            eliminated = [
                i for i in active if self._player_eliminated_on_field(self.players[self.agents[i]])
            ]
            if not eliminated:
                continue
            idx = random.choice(eliminated)
            self.state["disabled_agents"][idx] = True
            self.players[self.agents[idx]].is_disabled = True
            if idx < self.num_blue:
                self.num_blue_active = max(0, self.num_blue_active - 1)
            else:
                self.num_red_active = max(0, self.num_red_active - 1)
            self.state["num_blue_active"] = self.num_blue_active
            self.state["num_red_active"] = self.num_red_active
            return

    def _revive_one_random_from_team(self, team: Team) -> bool:
        """Re-enable one random disabled agent on ``team``. Returns True if an agent was revived."""
        if self.red_dummy_mode and team == Team.RED_TEAM:
            return False
        disabled = self.state["disabled_agents"]
        inds = self.agent_inds_of_team[team]
        disabled_team = [i for i in inds if disabled[i]]
        if not disabled_team:
            return False
        idx = random.choice(disabled_team)
        self.state["disabled_agents"][idx] = False
        self.players[self.agents[idx]].is_disabled = False
        flag_home = np.array(self.flags[int(team)].home)
        offset = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        new_pos = np.clip(flag_home + offset, [0, 0], self.env_size).astype(np.float64)
        self.state["agent_position"][idx] = new_pos
        p = self.players[self.agents[idx]]
        p.pos = np.array(new_pos, dtype=np.float64)
        p.prev_pos = p.pos.copy()
        p.is_tagged = False
        self.state["agent_is_tagged"][idx] = 0
        if team == Team.BLUE_TEAM:
            self.num_blue_active = min(self.num_blue, self.num_blue_active + 1)
        else:
            self.num_red_active = min(self.num_red, self.num_red_active + 1)
        self.state["num_blue_active"] = self.num_blue_active
        self.state["num_red_active"] = self.num_red_active
        return True

    def _dynamic_toggle_random_roster_tick(self) -> None:
        """Random removal (eliminated only, min one per team) and/or random revival."""
        if random.random() < self.dynamic_toggle_remove_prob:
            self._try_random_remove_eliminated()
        if random.random() < self.dynamic_toggle_add_prob:
            teams = [Team.BLUE_TEAM, Team.RED_TEAM]
            random.shuffle(teams)
            for team in teams:
                if self._revive_one_random_from_team(team):
                    break

    def _spawn_reinforcement(self):
        """Re-enable one disabled agent on the first team (Blue then Red) that passes a prob check."""
        for team in [Team.BLUE_TEAM, Team.RED_TEAM]:
            if self.red_dummy_mode and team == Team.RED_TEAM:
                continue
            inds = self.agent_inds_of_team[team]
            disabled_team = [i for i in inds if self.state["disabled_agents"][i]]
            if disabled_team and random.random() < self.reinforcement_prob:
                self._revive_one_random_from_team(team)
                break
