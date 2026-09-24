"""
=============================================================================
EXP-2026-U: Active-Evidential 3D GW Error-Volume Tiling MDP Benchmark
=============================================================================
Simulates 100 realistic O4/O5 gravitational wave BNS error volumes (50 - 250 deg^2)
under optical decay and evaluates three autonomous telescope tiling policies:
  1. Greedy 2D Ranking (Standard survey broker status quo)
  2. Static 3D Mass Ranking (Galaxy mass weighting without color pairing)
  3. Active-Evidential RLCD Tiling MDP (Dynamic 3D mass + Kasen fading visibility + VoI color reddening)

Outputs:
  - 4-Panel publication diagnostic figure: docs/research/figures/experiment_u_active_gw_tiling_benchmark.png
  - Quantitative metrics JSON: docs/research/experiment_u_active_gw_tiling_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from celestrium.active_tiling import (
    generate_gw_tiling_field,
    ActiveEvidentialTilingMDP,
    TelescopeSpecs,
)


def run_gw_tiling_monte_carlo(n_simulations: int = 100, seed: int = 42) -> Dict[str, Any]:
    print("=" * 76)
    print(f"EXP-2026-U: ACTIVE-EVIDENTIAL 3D GW TILING BENCHMARK ({n_simulations} REALIZATIONS)")
    print("=" * 76)

    policies = ["greedy_2d", "static_3d_mass", "active_evidential"]
    results = {p: {"detected": 0, "color_confirmed": 0, "mass_coverage": [], "detection_times": [], "efficiency": []} for p in policies}

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    for sim_idx in range(n_simulations):
        # Generate random GW field with 60 tiles (~180 deg^2 footprint)
        # Kilonova randomly located in one of the tiles
        tiles, specs = generate_gw_tiling_field(
            n_tiles=60,
            inject_kilonova_tile=int(rng.integers(0, 60)),
            seed=seed + sim_idx * 13,
        )
        mdp = ActiveEvidentialTilingMDP(
            tiles=tiles,
            specs=specs,
            night_shutter_sec=5.0 * 3600.0,  # 5 hours available shutter
            t_trigger_hrs=float(rng.uniform(1.5, 4.0)),  # Response time 1.5 - 4 hours post-merger
        )

        for p in policies:
            res = mdp.run_policy(p, seed=seed + sim_idx)
            if res["kilonova_detected"]:
                results[p]["detected"] += 1
                if res["detection_time_hrs"] is not None:
                    results[p]["detection_times"].append(res["detection_time_hrs"])
            if res["color_reddening_confirmed"]:
                results[p]["color_confirmed"] += 1
            results[p]["mass_coverage"].append(res["mass_coverage_pct"])
            results[p]["efficiency"].append(res["shutter_efficiency_pct"])

    dt = time.perf_counter() - t0
    print(f"Completed {n_simulations} simulations across 3 policies in {dt:.2f}s ({n_simulations*3/dt:.1f} runs/sec).")

    summary = {
        "experiment_id": "EXP-2026-U",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_simulations": n_simulations,
        "elapsed_seconds": dt,
        "policy_comparison": {},
    }

    for p in policies:
        det_rate = (results[p]["detected"] / n_simulations) * 100.0
        color_rate = (results[p]["color_confirmed"] / n_simulations) * 100.0
        mean_mass = float(np.mean(results[p]["mass_coverage"]))
        mean_eff = float(np.mean(results[p]["efficiency"]))
        mean_t_det = float(np.mean(results[p]["detection_times"])) if results[p]["detection_times"] else None

        summary["policy_comparison"][p] = {
            "kilonova_detection_rate_pct": det_rate,
            "color_reddening_confirmation_rate_pct": color_rate,
            "mean_galaxy_mass_coverage_pct": mean_mass,
            "mean_shutter_efficiency_pct": mean_eff,
            "mean_detection_time_hrs": mean_t_det,
        }
        print(f"\nPolicy: {p.upper()}")
        print(f"  Kilonova Detection Rate:     {det_rate:.1f}%")
        print(f"  Kasen Reddening Confirmed:   {color_rate:.1f}% (Dual-epoch g-r confirmation)")
        print(f"  Galaxy Mass Completeness:    {mean_mass:.1f}%")
        print(f"  Shutter Efficiency:          {mean_eff:.1f}%")
        if mean_t_det:
            print(f"  Mean Time of Discovery:      {mean_t_det:.2f} hrs post-merger")

    # Export JSON results
    out_json = ROOT_DIR / "docs" / "research" / "experiment_u_active_gw_tiling_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_json}")

    # Generate 4-Panel Diagnostic Figure
    out_fig = ROOT_DIR / "docs" / "research" / "figures" / "experiment_u_active_gw_tiling_benchmark.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    p_labels = ["Greedy 2D\n(Status Quo)", "Static 3D Mass\n(No VoI)", "Active-Evidential\n(Celestrium RLCD)"]
    p_keys = ["greedy_2d", "static_3d_mass", "active_evidential"]

    # Panel 1: Kilonova Detection Rate
    ax = axs[0, 0]
    det_vals = [summary["policy_comparison"][k]["kilonova_detection_rate_pct"] for k in p_keys]
    bars = ax.bar(p_labels, det_vals, color=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black", width=0.55)
    for b, v in zip(bars, det_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim([0, 105])
    ax.set_ylabel("Kilonova Detection Rate [%]", fontsize=11, fontweight="bold")
    ax.set_title("O4 Kilonova Recovery Rate (100 Simulated Volumes)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Dual-Band Kasen Color Reddening Confirmation
    ax = axs[0, 1]
    col_vals = [summary["policy_comparison"][k]["color_reddening_confirmation_rate_pct"] for k in p_keys]
    bars = ax.bar(p_labels, col_vals, color=["#7f7f7f", "#1f77b4", "#9467bd"], edgecolor="black", width=0.55)
    for b, v in zip(bars, col_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim([0, 105])
    ax.set_ylabel("Color Confirmation Rate [%]", fontsize=11, fontweight="bold")
    ax.set_title("Kasen Reddening (d(g-r)/dt > +0.35 mag/d) Confirmation", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Galaxy Mass Completeness
    ax = axs[1, 0]
    mass_vals = [summary["policy_comparison"][k]["mean_galaxy_mass_coverage_pct"] for k in p_keys]
    bars = ax.bar(p_labels, mass_vals, color=["#e377c2", "#17becf", "#2ca02c"], edgecolor="black", width=0.55)
    for b, v in zip(bars, mass_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim([0, 105])
    ax.set_ylabel("Enclosed Stellar Mass [%]", fontsize=11, fontweight="bold")
    ax.set_title("Integrated Host Galaxy Stellar Mass Completeness", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Shutter Efficiency
    ax = axs[1, 1]
    eff_vals = [summary["policy_comparison"][k]["mean_shutter_efficiency_pct"] for k in p_keys]
    bars = ax.bar(p_labels, eff_vals, color=["#8c564b", "#bcbd22", "#1f77b4"], edgecolor="black", width=0.55)
    for b, v in zip(bars, eff_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim([60, 95])
    ax.set_ylabel("Shutter Efficiency [% on target]", fontsize=11, fontweight="bold")
    ax.set_title("Observatory Shutter Utilization (Minimizing Slew Overhead)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("EXP-2026-U: Active-Evidential 3D GW Error-Volume Tiling MDP Benchmark\nDECam 3 deg² Field of View | Kasen (2017) Fast Reddening Decay | 100 O4 Error Volumes", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_fig, dpi=200)
    print(f"Saved diagnostic figure to: {out_fig}")
    print("=" * 76)
    print("EXP-2026-U BENCHMARK COMPLETE")
    print("=" * 76)
    return summary


if __name__ == "__main__":
    run_gw_tiling_monte_carlo(n_simulations=100)
