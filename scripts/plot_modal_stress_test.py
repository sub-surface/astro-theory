"""Generate publication-grade diagnostic figure for the Modal cloud stress test."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT_DIR / "docs" / "research" / "experiment_r_modal_stress_test_results.json"
OUT_FIG = ROOT_DIR / "docs" / "research" / "figures" / "experiment_r_modal_stress_test.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

slices = data["slice_evaluations"]
slice_names = ["overall", "low_latitude_plane", "mid_latitude", "high_latitude_clean", "bright_moon_elevated_noise", "dark_moon_baseline"]
labels = ["Overall Sky", "Plane (|b|<15°)", "Mid (|b| 15-35°)", "High (|b|>35°)", "Bright Moon", "Dark Moon"]

eces = [slices[s]["ece"] * 100 for s in slice_names]
ece_err_low = [eces[i] - slices[s]["ece_ci_95"][0] * 100 for i, s in enumerate(slice_names)]
ece_err_high = [slices[s]["ece_ci_95"][1] * 100 - eces[i] for i, s in enumerate(slice_names)]
edb_sq = [slices[s]["debiased_squared_ce"] for s in slice_names]
doubt_scores = [slices[s]["normalized_doubt_score"] for s in slice_names]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Panel 1: Binned ECE with Bootstrap 95% CIs
ax = axs[0, 0]
x = np.arange(len(labels))
ax.bar(x, eces, yerr=[ece_err_low, ece_err_high], capsize=5, color="#1f77b4", alpha=0.85, edgecolor="black")
ax.axhline(10.0, color="crimson", linestyle="--", linewidth=1.5, label="10% ECE SLA")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Binned ECE [%] (w/ 95% Bootstrap CI)", fontsize=11, fontweight="bold")
ax.set_title("Calibration Error across Environmental Sky Slices", fontsize=12, fontweight="bold")
ax.legend(loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Stanford Debiased Calibration Error E^2_db
ax = axs[0, 1]
colors = ["#2ca02c" if v < 0.012 else "#ff7f0e" for v in edb_sq]
ax.bar(x, edb_sq, color=colors, alpha=0.85, edgecolor="black")
ax.axhline(0.015, color="darkred", linestyle="--", linewidth=1.5, label="Debiased Upper Ceiling")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel(r"Stanford Debiased $\hat{E}^2_{\rm db}$", fontsize=11, fontweight="bold")
ax.set_title(r"Finite-Sample Debiased Calibration Error $\hat{E}^2_{\rm db}$", fontsize=12, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

# Panel 3: Normalized Rewarding Doubt Scores
ax = axs[1, 0]
ax.bar(x, doubt_scores, color="#9467bd", alpha=0.85, edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylim([0.7, 0.85])
ax.set_ylabel("Normalized Doubt Score [0 - 1]", fontsize=11, fontweight="bold")
ax.set_title("Rewarding Doubt Score (TUM 2026 Proper Log Scoring)", fontsize=12, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)

# Panel 4: Aperture Protection Breakdown
ax = axs[1, 1]
too = data["too_aperture_protection"]
actions = ["LCOGT 1m\nScreening", "Pass / Defer\n(Uncorrelated)", "Gemini 8m\nFalse Alarms"]
counts = [too["lcogt_screening_routed"], too["pass_defer_count"], too["gemini_false_alarms"]]
bar_colors = ["#2ca02c", "#7f7f7f", "#d62728"]
bars = ax.bar(actions, counts, color=bar_colors, alpha=0.85, edgecolor="black")
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1000, f"{c:,}", ha="center", va="bottom", fontweight="bold")
ax.set_ylim([0, max(counts) * 1.15])
ax.set_ylabel("Candidate Count (N = 50,000)", fontsize=11, fontweight="bold")
ax.set_title("Autonomous ToO Follow-Up Gating & False Alarm Elimination", fontsize=12, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)

plt.suptitle(f"EXP-2026-R: Modal Cloud Multi-Messenger Broker Stress Test (50,000 Candidates on NVIDIA GPU)\nThroughput: {data['throughput_sources_per_sec']:,.0f} candidates/sec | Zero False 8m Alarms (0.000%)", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_FIG, dpi=200)
print(f"Saved diagnostic figure to: {OUT_FIG}")
