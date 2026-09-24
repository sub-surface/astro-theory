"""
=============================================================================
Scaled Evidential AstroJev Analysis & Dirichlet Simplex Evaluation
=============================================================================
Evaluates the H100-trained Evidential AstroJev checkpoint:
  checkpoints/astrojev_evidential_h100_scaled.pt
across:
  1. In-Distribution (ID) Astronomical Test Set (Quasars, Stars, Galaxies, White Dwarfs)
  2. Out-of-Distribution (OOD) Astronomical Anomalies (FBOTs, Lensed Supernovae, Cosmic Rays)

Analyzes:
  - Exact epistemic vacuity (u_epi = K / S) vs aleatoric entropy (u_ale)
  - Dirichlet concentration alpha and evidence tension damping
  - Calibration of axiomatic Epistemic Noul decision gating

Outputs figure:
  docs/figures/evidential_modal_h100_analysis.png
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from celestrium.evidential_astrojev import EvidentialAstroJev
from celestrium.experiments.differentiable_uncertainty_test import build_error_bar_dataset
from celestrium.astrojev import evaluate_calibration, NUM_FEATURES, NUM_CLASSES, CLASSES

FIGURES_DIR = ROOT_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


def synthesize_astronomical_anomalies(n_samples: int = 2000, seed: int = 777):
    """Synthesizes out-of-distribution anomalies and detector glitches."""
    rng = np.random.default_rng(seed)
    n_each = n_samples // 3

    # 1. Fast Blue Optical Transients (FBOTs / AT2018cow analogs)
    # Extreme optical brightness, anomalous rapid blue color BP - RP ~ -0.4, no mid-IR match
    fbot_g = rng.uniform(16.0, 19.5, n_each)
    fbot_bprp = rng.normal(-0.45, 0.15, n_each)
    fbot_gbp = rng.normal(0.15, 0.1, n_each)
    fbot_w1 = rng.uniform(19.0, 21.0, n_each)  # Extremely faint IR (undetected)
    fbot_w1w2 = rng.normal(0.0, 0.4, n_each)
    fbot_pm = np.abs(rng.normal(0.0, 0.5, n_each))
    fbot_pmerr = rng.uniform(0.3, 0.8, n_each)
    fbot_snr = rng.uniform(15.0, 50.0, n_each)
    fbot_g_err = 1.0857 / fbot_snr
    fbot_w1_err = fbot_g_err * 3.0
    fbots = np.column_stack([fbot_g, fbot_bprp, fbot_gbp, fbot_w1, fbot_w1w2, fbot_pm, fbot_pmerr, fbot_g_err, fbot_w1_err, fbot_snr])

    # 2. Gravitationally Lensed Multiplet Supernovae / Oddball Blends
    # Highly non-physical colors due to host contamination + magnification
    lens_g = rng.uniform(18.5, 21.0, n_each)
    lens_bprp = rng.normal(3.2, 0.5, n_each)     # Ultra-red optical
    lens_gbp = rng.normal(-1.5, 0.3, n_each)
    lens_w1 = rng.normal(13.5, 0.8, n_each)     # Hyper-bright infrared
    lens_w1w2 = rng.normal(2.1, 0.3, n_each)    # Super-red infrared excess
    lens_pm = np.abs(rng.normal(0.0, 0.4, n_each))
    lens_pmerr = rng.uniform(0.3, 0.8, n_each)
    lens_snr = rng.uniform(8.0, 30.0, n_each)
    lens_g_err = 1.0857 / lens_snr
    lens_w1_err = lens_g_err * 1.5
    lenses = np.column_stack([lens_g, lens_bprp, lens_gbp, lens_w1, lens_w1w2, lens_pm, lens_pmerr, lens_g_err, lens_w1_err, lens_snr])

    # 3. Cosmic Ray / Stray Light Glitches
    # Single-pixel spike with zero signal-to-noise consistency
    ray_g = rng.uniform(15.0, 18.0, n_each)
    ray_bprp = rng.normal(0.0, 2.5, n_each)
    ray_gbp = rng.normal(0.0, 2.5, n_each)
    ray_w1 = rng.normal(16.0, 3.0, n_each)
    ray_w1w2 = rng.normal(0.0, 2.0, n_each)
    ray_pm = rng.uniform(10.0, 80.0, n_each)
    ray_pmerr = rng.uniform(5.0, 25.0, n_each)
    ray_snr = rng.uniform(1.0, 4.0, n_each)
    ray_g_err = 1.0857 / ray_snr
    ray_w1_err = ray_g_err * 2.0
    rays = np.column_stack([ray_g, ray_bprp, ray_gbp, ray_w1, ray_w1w2, ray_pm, ray_pmerr, ray_g_err, ray_w1_err, ray_snr])

    return np.vstack([fbots, lenses, rays]).astype(np.float32)


def run_evidential_analysis():
    print("=" * 72)
    print("SCALED EVIDENTIAL ASTROJEV EVALUATION & SIMPLEX ANALYSIS")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = CHECKPOINTS_DIR / "astrojev_evidential_h100_scaled.pt"

    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    print(f"Loading Scaled Evidential model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=128, num_classes=NUM_CLASSES, n_iter=5).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. In-Distribution (ID) Test Evaluation
    _, _, X_id, y_id = build_error_bar_dataset(n_samples=10000, seed=888)
    X_ood = synthesize_astronomical_anomalies(n_samples=3000, seed=777)

    with torch.no_grad():
        out_id = model(X_id.to(device))
        out_ood = model(torch.from_numpy(X_ood).to(device))

    # ID Metrics
    probs_id = out_id["probs"].cpu().numpy()
    u_epi_id = out_id["u_epi"].cpu().numpy()
    u_ale_id = out_id["u_ale"].cpu().numpy()
    noul_id = out_id["noul"].cpu().numpy()
    targets_id = y_id.numpy()

    # OOD Metrics
    probs_ood = out_ood["probs"].cpu().numpy()
    u_epi_ood = out_ood["u_epi"].cpu().numpy()
    u_ale_ood = out_ood["u_ale"].cpu().numpy()
    noul_ood = out_ood["noul"].cpu().numpy()

    id_calib = evaluate_calibration(probs_id, targets_id)

    print("\nEVIDENTIAL TEST PERFORMANCE:")
    print(f"  In-Distribution Accuracy:         {id_calib['accuracy']*100:.2f}%")
    print(f"  In-Distribution Brier Score:      {id_calib['brier']:.4f}")
    print(f"  In-Distribution Mean Vacuity:     {np.mean(u_epi_id):.4f}")
    print(f"  In-Distribution Mean Noul:        {np.mean(noul_id):.4f}")
    print(f"  OOD Anomaly Mean Vacuity:         {np.mean(u_epi_ood):.4f}")
    print(f"  OOD Anomaly Mean Noul:            {np.mean(noul_ood):.4f}")

    # OOD Rejection Rate under Noul Threshold > 0.80
    ood_rejected = np.mean(noul_ood < 0.80) * 100.0
    id_accepted = np.mean(noul_id >= 0.80) * 100.0
    print(f"  Autonomous OOD Anomaly Rejection: {ood_rejected:.2f}%")
    print(f"  Autonomous ID Source Acceptance:  {id_accepted:.2f}%")
    print("=" * 72)

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Epistemic Vacuity vs Aleatoric Entropy Disentanglement
    ax_a = axes[0]
    ax_a.scatter(u_ale_id[:600], u_epi_id[:600], color="#2b83ba", s=15, alpha=0.6, label="In-Distribution Sources (Gaia/Quaia)")
    ax_a.scatter(u_ale_ood[:600], u_epi_ood[:600], color="#d7191c", s=15, alpha=0.6, label="OOD Anomalies (FBOTs, Lenses, Cosmic Rays)")

    ax_a.axhline(0.20, color="black", linestyle="--", lw=1.2, label="Epistemic Rejection Threshold ($u_{\\rm epi} > 0.20$)")
    ax_a.set_xlabel("Aleatoric Uncertainty $u_{\\rm ale}$ (Measurement Noise)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Epistemic Vacuity $u_{\\rm epi} = K / S$ (Model Ignorance)", fontsize=11, fontweight="bold")
    ax_a.set_title("(A) Disentanglement: Epistemic Ignorance vs Aleatoric Noise", fontsize=12, fontweight="bold")
    ax_a.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Axiomatic Epistemic Noul Credence CDF
    ax_b = axes[1]
    sorted_id = np.sort(noul_id)
    sorted_ood = np.sort(noul_ood)

    cdf_id = np.linspace(0, 1, len(sorted_id))
    cdf_ood = np.linspace(0, 1, len(sorted_ood))

    ax_b.plot(sorted_id, cdf_id, lw=2.5, color="#2b83ba", label="In-Distribution Astronomical Sources")
    ax_b.plot(sorted_ood, cdf_ood, lw=2.5, color="#d7191c", label="Out-of-Distribution Anomalies & Glitches")
    ax_b.axvline(0.80, color="gray", linestyle=":", lw=1.5, label="Calibrated Action Gating (Noul >= 0.80)")

    ax_b.set_xlabel("Epistemic Noul Credence (Noul = 1 - K/S)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Cumulative Empirical Density", fontsize=11, fontweight="bold")
    ax_b.set_title(f"(B) Autonomous Decision Gating ({ood_rejected:.1f}% Anomalies Rejected)", fontsize=12, fontweight="bold")
    ax_b.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    out_fig = FIGURES_DIR / "evidential_modal_h100_analysis.png"
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, wspace=0.25)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {out_fig}")


if __name__ == "__main__":
    run_evidential_analysis()
