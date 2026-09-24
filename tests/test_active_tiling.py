"""Hermetic tests for EXP-2026-U: Active-Evidential 3D GW Error-Volume Tiling MDP."""
from __future__ import annotations

import math
import pytest
import numpy as np

from celestrium.active_tiling import (
    angular_distance_deg,
    generate_gw_tiling_field,
    ActiveEvidentialTilingMDP,
    TelescopeSpecs,
)


def test_angular_distance_deg():
    """Verify great-circle angular distance on a sphere."""
    # Distance to self is 0
    assert abs(angular_distance_deg(180.0, 0.0, 180.0, 0.0)) < 1e-5
    # North pole to equator is 90 degrees
    assert abs(angular_distance_deg(0.0, 90.0, 0.0, 0.0) - 90.0) < 1e-4
    # Equator 180 deg separation
    assert abs(angular_distance_deg(0.0, 0.0, 180.0, 0.0) - 180.0) < 1e-4


def test_generate_gw_tiling_field():
    """Verify field generation and probability normalizations."""
    tiles, specs = generate_gw_tiling_field(n_tiles=40, inject_kilonova_tile=5, seed=123)

    assert len(tiles) == 40
    assert specs.name == "DECam_Blanco_4m"
    assert tiles[5].kilonova_present is True
    assert sum(1 for t in tiles if t.kilonova_present) == 1

    # Probabilities must sum to ~1.0
    sum_2d = sum(t.prob_2d for t in tiles)
    sum_3d = sum(t.prob_3d_mass for t in tiles)
    assert abs(sum_2d - 1.0) < 1e-4
    assert abs(sum_3d - 1.0) < 1e-4


def test_tiling_mdp_policies():
    """Verify execution of all three tiling policies under shutter budget constraints."""
    tiles, specs = generate_gw_tiling_field(n_tiles=50, inject_kilonova_tile=10, seed=42)
    mdp = ActiveEvidentialTilingMDP(tiles=tiles, specs=specs, night_shutter_sec=3.0 * 3600.0)

    # 1. Greedy 2D
    res_2d = mdp.run_policy("greedy_2d", seed=42)
    assert res_2d["total_tiles_observed"] > 0
    assert 0.0 <= res_2d["mass_coverage_pct"] <= 100.0
    assert 0.0 < res_2d["shutter_efficiency_pct"] <= 100.0

    # 2. Static 3D
    res_3d = mdp.run_policy("static_3d_mass", seed=42)
    assert res_3d["total_tiles_observed"] > 0
    assert 0.0 <= res_3d["mass_coverage_pct"] <= 100.0

    # 3. Active-Evidential
    res_ev = mdp.run_policy("active_evidential", seed=42)
    assert res_ev["total_tiles_observed"] > 0
    assert 0.0 <= res_ev["mass_coverage_pct"] <= 100.0
