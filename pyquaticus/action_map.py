"""Action-map helpers for switching discrete control layouts."""

import pyquaticus.config as _pq_config


def nrl_recommended_headings():
    """[-100, -60, -10..10, 60, 100] in degrees."""
    return [-100, -60, *range(-10, 11), 60, 100]


def nrl_action_map_entries():
    """Full-speed steering at each heading plus no-op."""
    out = [[1.0, float(h)] for h in nrl_recommended_headings()]
    out.append([0.0, 0.0])
    return out


def apply_action_map_entries(entries):
    """Patch pyquaticus.config.ACTION_MAP in place for importers."""
    _pq_config.ACTION_MAP.clear()
    _pq_config.ACTION_MAP.extend(entries)


def apply_nrl_action_map():
    """Switch ACTION_MAP to the NRL-compatible discrete layout."""
    apply_action_map_entries(nrl_action_map_entries())


def _legacy_action_map_entries():
    out = []
    for spd in [1.0, 0.5]:
        for hdg in range(180, -180, -45):
            out.append([spd, float(hdg)])
    out.append([0.0, 0.0])
    return out


def apply_legacy_action_map():
    """Restore the prior default (2-speed x 8-heading + no-op)."""
    apply_action_map_entries(_legacy_action_map_entries())
