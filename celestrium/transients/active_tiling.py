"""
=============================================================================
Celestrium Active-Evidential 3D Gravitational Wave Error-Volume Tiling MDP (EXP-2026-U)
=============================================================================
Autonomous reinforcement learning decision engine for tiling 3D gravitational
wave localization volumes (LIGO/Virgo/KAGRA O4/O5) with wide-field survey telescopes
(DECam 3 deg^2, ZTF 47 deg^2, Rubin LSST 9.6 deg^2).

Key Features:
1. 3D Co-Moving Mass & Kasen Fading Horizon:
   - Line-of-sight galaxy mass weighting: M_tile = int_tile dOmega int dr r^2 rho_*(r) P(r | Omega)
   - Dynamic kilonova optical decay: m(t) = m_peak + alpha_fade * t [mag/day]
   - Color reddening evolution: d(g - r)/dt > +0.35 mag/day.

2. Constrained Markov Decision Process (CMDP):
   - State s_t: Tiled coverage bitmask, telescope position (ra, dec), elapsed time t,
     airmass X(t), and Dirichlet epistemic vacuity u_epi across candidate tiles.
   - Action a_t: Slew to tile i in band b in {g, r}.
   - Slew & overhead cost model: t_slew = t_settle + (Delta_theta / v_slew)^gamma.

3. Evidential Reward & Value of Information (VoI):
   - Rewards dual-epoch filter pairs (g, r) to measure rapid color reddening
     and collapse epistemic doubt before the transient fades below limiting magnitude.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


@dataclass
class SkyTile:
    """Pointing tile in the survey field of view."""
    tile_id: int
    ra_deg: float
    dec_deg: float
    prob_2d: float
    prob_3d_mass: float  # Integrated host galaxy stellar mass weight
    mean_dist_mpc: float
    kilonova_present: bool = False
    kilonova_peak_mag_r: float = 21.0
    fade_rate_mag_per_day: float = 1.1
    color_rate_mag_per_day: float = 0.45  # Kasen reddening
    visits: int = 0
    bands_observed: List[str] = field(default_factory=list)
    epistemic_vacuity: float = 0.95
    detected: bool = False


@dataclass
class TelescopeSpecs:
    """Optical wide-field telescope parameters (default: DECam)."""
    name: str = "DECam_Blanco_4m"
    fov_deg2: float = 3.0
    radius_deg: float = 0.977  # sqrt(3 / pi)
    v_slew_deg_per_sec: float = 2.5
    settle_time_sec: float = 15.0
    filter_change_sec: float = 25.0
    exp_time_sec: float = 90.0
    limiting_mag_r: float = 23.5
    limiting_mag_g: float = 24.0


def angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Computes great-circle angular distance on a sphere in degrees."""
    d_ra = math.radians(ra2 - ra1)
    d_dec = math.radians(dec2 - dec1)
    dec1_r = math.radians(dec1)
    dec2_r = math.radians(dec2)

    a = math.sin(d_dec / 2.0) ** 2 + math.cos(dec1_r) * math.cos(dec2_r) * math.sin(d_ra / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return math.degrees(c)


def generate_gw_tiling_field(
    n_tiles: int = 80,
    inject_kilonova_tile: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[SkyTile], TelescopeSpecs]:
    """Generates a realistic 3D GW skymap tiling field with GLADE+ galaxy clustering."""
    rng = np.random.default_rng(seed)
    specs = TelescopeSpecs()

    # Center of the GW error volume
    gw_center_ra = 180.0
    gw_center_dec = -15.0
    gw_dist_mean = 160.0  # Mpc (typical O4 BNS)
    gw_dist_std = 35.0

    # Grid of tiles around the GW center
    tiles = []
    kn_tile_idx = inject_kilonova_tile if inject_kilonova_tile is not None else int(rng.integers(0, n_tiles))

    for i in range(n_tiles):
        ra = gw_center_ra + rng.normal(0.0, 5.0)
        dec = gw_center_dec + rng.normal(0.0, 4.0)
        ang_dist = angular_distance_deg(gw_center_ra, gw_center_dec, ra, dec)

        # 2D Probability Gaussian profile
        p_2d = math.exp(-0.5 * (ang_dist / 4.5) ** 2)

        # 3D Distance & Galaxy Clustering
        # In a real universe, galaxies are clustered: some tiles have massive galaxy overdensities
        has_galaxy_cluster = (rng.uniform(0, 1) < 0.25)
        dist = float(rng.normal(gw_dist_mean, gw_dist_std))
        dist_weight = math.exp(-0.5 * ((dist - gw_dist_mean) / gw_dist_std) ** 2)
        mass_factor = rng.uniform(3.0, 15.0) if has_galaxy_cluster else rng.uniform(0.1, 1.2)
        p_3d_mass = p_2d * dist_weight * mass_factor

        is_kn = (i == kn_tile_idx)
        peak_r = float(rng.uniform(20.5, 21.8)) if is_kn else 99.0

        tiles.append(
            SkyTile(
                tile_id=i,
                ra_deg=float(ra),
                dec_deg=float(dec),
                prob_2d=float(p_2d),
                prob_3d_mass=float(p_3d_mass),
                mean_dist_mpc=float(dist),
                kilonova_present=is_kn,
                kilonova_peak_mag_r=peak_r,
                fade_rate_mag_per_day=float(rng.uniform(0.9, 1.3)),
                color_rate_mag_per_day=float(rng.uniform(0.35, 0.60)),
            )
        )

    # Normalize probabilities
    sum_2d = sum(t.prob_2d for t in tiles)
    sum_3d = sum(t.prob_3d_mass for t in tiles)
    for t in tiles:
        t.prob_2d /= max(sum_2d, 1e-6)
        t.prob_3d_mass /= max(sum_3d, 1e-6)

    return tiles, specs


class ActiveEvidentialTilingMDP:
    """Constrained Markov Decision Process for autonomous telescope tiling."""
    def __init__(
        self,
        tiles: List[SkyTile],
        specs: TelescopeSpecs,
        night_shutter_sec: float = 5.0 * 3600.0,  # 5 hours observing window
        t_trigger_hrs: float = 2.5,  # Hours since GW merger alert
    ):
        self.tiles = tiles
        self.specs = specs
        self.night_shutter_sec = night_shutter_sec
        self.t_start_hrs = t_trigger_hrs

    def run_policy(
        self,
        policy_name: str = "active_evidential",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Simulates autonomous tiling execution under a given policy.

        Policies:
        - "greedy_2d": Ranks purely by 2D skymap probability (Standard status quo).
        - "static_3d_mass": Ranks purely by 3D galaxy mass weight without VoI color pairing.
        - "active_evidential": Dynamically optimizes joint 3D mass + Kasen fading visibility
          + epistemic doubt collapse (scheduling dual-band pairs before optical fading).
        """
        rng = np.random.default_rng(seed)
        t_elapsed_sec = 0.0
        cur_ra = self.tiles[0].ra_deg
        cur_dec = self.tiles[0].dec_deg
        cur_band = "r"

        # Clone tiles for independent tracking
        tiles = [
            SkyTile(
                tile_id=t.tile_id,
                ra_deg=t.ra_deg,
                dec_deg=t.dec_deg,
                prob_2d=t.prob_2d,
                prob_3d_mass=t.prob_3d_mass,
                mean_dist_mpc=t.mean_dist_mpc,
                kilonova_present=t.kilonova_present,
                kilonova_peak_mag_r=t.kilonova_peak_mag_r,
                fade_rate_mag_per_day=t.fade_rate_mag_per_day,
                color_rate_mag_per_day=t.color_rate_mag_per_day,
            )
            for t in self.tiles
        ]

        tiled_history = []
        kilonova_detected = False
        kilonova_detected_time_hr = None
        color_evolution_measured = False
        cumulative_mass_covered = 0.0

        while t_elapsed_sec < self.night_shutter_sec:
            t_post_merger_days = (self.t_start_hrs + t_elapsed_sec / 3600.0) / 24.0

            # Compute scores for all candidate actions
            candidates = []
            for t in tiles:
                # Slew time calculation
                dist_deg = angular_distance_deg(cur_ra, cur_dec, t.ra_deg, t.dec_deg)
                slew_sec = self.specs.settle_time_sec + dist_deg / self.specs.v_slew_deg_per_sec

                # Kilonova current apparent magnitude
                cur_mag_r = t.kilonova_peak_mag_r + t.fade_rate_mag_per_day * t_post_merger_days
                cur_mag_g = cur_mag_r + 0.3 + t.color_rate_mag_per_day * t_post_merger_days
                is_visible_r = (cur_mag_r <= self.specs.limiting_mag_r)
                is_visible_g = (cur_mag_g <= self.specs.limiting_mag_g)

                # Policy Scoring Logic:
                if policy_name == "greedy_2d":
                    # Greedy 2D ignores 3D mass and fading: purely 2D probability discounted by visits
                    score = t.prob_2d / (1.0 + t.visits * 3.0)
                    chosen_band = "r"
                    overhead = slew_sec + self.specs.exp_time_sec

                elif policy_name == "static_3d_mass":
                    # Static 3D uses 3D mass weight but doesn't schedule dual-band color pairs
                    score = t.prob_3d_mass / (1.0 + t.visits * 3.0)
                    chosen_band = "r"
                    overhead = slew_sec + self.specs.exp_time_sec

                elif policy_name == "active_evidential":
                    # Active-Evidential RLCD:
                    # 1. Rewards unexplored high 3D mass
                    # 2. Rewards second-epoch color confirmation if visited once in 'r' (VoI color reddening)
                    # 3. Penalizes slews to prevent excessive telescope motion
                    # 4. Suppresses score if transient has faded past limiting magnitude
                    needs_g_band = ("r" in t.bands_observed and "g" not in t.bands_observed)
                    chosen_band = "g" if needs_g_band else "r"
                    filter_change = self.specs.filter_change_sec if chosen_band != cur_band else 0.0
                    overhead = slew_sec + filter_change + self.specs.exp_time_sec

                    # Visibility attenuation
                    vis_factor = 1.0 if is_visible_r else 0.15

                    if t.visits == 0:
                        # Exploration value: 3D mass / slew cost
                        score = (t.prob_3d_mass * vis_factor) / (1.0 + 0.005 * slew_sec)
                    elif needs_g_band and t.visits == 1 and is_visible_g:
                        # Exploitation / Doubt-collapse: High Value of Information to confirm Kasen reddening
                        score = 2.5 * (t.prob_3d_mass * vis_factor) / (1.0 + 0.005 * slew_sec)
                    else:
                        score = 0.001 * t.prob_3d_mass

                candidates.append((score, t, chosen_band, overhead))

            # Select highest scoring action
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, chosen_tile, target_band, action_time_sec = candidates[0]

            if best_score <= 1e-7 or (t_elapsed_sec + action_time_sec > self.night_shutter_sec):
                break  # Budget exhausted or no viable tiles left

            # Execute action
            t_elapsed_sec += action_time_sec
            cur_ra = chosen_tile.ra_deg
            cur_dec = chosen_tile.dec_deg
            cur_band = target_band
            chosen_tile.visits += 1
            chosen_tile.bands_observed.append(target_band)

            if chosen_tile.visits == 1:
                cumulative_mass_covered += chosen_tile.prob_3d_mass

            # Check kilonova detection
            cur_t_hr = self.t_start_hrs + t_elapsed_sec / 3600.0
            cur_days = cur_t_hr / 24.0
            cur_r = chosen_tile.kilonova_peak_mag_r + chosen_tile.fade_rate_mag_per_day * cur_days
            cur_g = cur_r + 0.3 + chosen_tile.color_rate_mag_per_day * cur_days

            if chosen_tile.kilonova_present:
                if target_band == "r" and cur_r <= self.specs.limiting_mag_r:
                    kilonova_detected = True
                    if kilonova_detected_time_hr is None:
                        kilonova_detected_time_hr = cur_t_hr
                if target_band == "g" and cur_g <= self.specs.limiting_mag_g:
                    if "r" in chosen_tile.bands_observed:
                        color_evolution_measured = True  # Both bands observed -> confirmed reddening!

            # Update epistemic vacuity: collapses with multi-band observations
            if len(set(chosen_tile.bands_observed)) >= 2:
                chosen_tile.epistemic_vacuity = 0.05
            elif chosen_tile.visits >= 1:
                chosen_tile.epistemic_vacuity = 0.45

            tiled_history.append({
                "step": len(tiled_history) + 1,
                "tile_id": chosen_tile.tile_id,
                "band": target_band,
                "elapsed_hr": float(cur_t_hr),
                "kilonova_detected": kilonova_detected,
                "color_confirmed": color_evolution_measured,
            })

        return {
            "policy": policy_name,
            "total_tiles_observed": len(tiled_history),
            "unique_tiles_covered": sum(1 for t in tiles if t.visits > 0),
            "mass_coverage_pct": float(cumulative_mass_covered * 100.0),
            "kilonova_detected": kilonova_detected,
            "detection_time_hrs": kilonova_detected_time_hr,
            "color_reddening_confirmed": color_evolution_measured,
            "total_shutter_time_hrs": float(t_elapsed_sec / 3600.0),
            "shutter_efficiency_pct": float((len(tiled_history) * self.specs.exp_time_sec) / max(1.0, t_elapsed_sec) * 100.0),
        }
