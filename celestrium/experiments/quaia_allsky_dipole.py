"""
=============================================================================
Full All-Sky Quaia Cosmological Dipole Recovery with Scaled AstroJev
=============================================================================
Ingests all 1,299,997 sources from the real Quaia FITS catalog:
  Archive/2026-06-G-dipole/data/quaia/quaia_G20.5.fits
Evaluates the scaled H100 AstroJev model to obtain continuous probabilistic
selection weights:
  w_i = p_i(Quasar)
and computes the all-sky cosmological dipole:
  D = (3 / N_eff) * sum_i w_i * r_hat_i
Comparing:
  1. Standard Hard Cut Selection (w_i = 1 if p_i >= 0.8 else 0)
  2. Continuous Calibrated Probabilistic Selection (w_i = p_i)

Outputs figure:
  docs/figures/quaia_allsky_dipole_recovery.png
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
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from celestrium.astrojev import AstroJev, NUM_FEATURES, NUM_CLASSES

FIGURES_DIR = ROOT_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


def compute_dipole_vector(ra_deg: np.ndarray, dec_deg: np.ndarray, weights: np.ndarray):
    """Computes the 3D dipole vector and spherical coordinates from weighted unit vectors."""
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Unit vectors in equatorial coordinates
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    r_hat = np.column_stack([x, y, z])

    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.zeros(3), 0.0, 0.0, 0.0

    # 3D dipole moment: D = (3 / N_eff) sum_i w_i r_hat_i
    d_vec = (3.0 / w_sum) * np.sum(r_hat * weights[:, None], axis=0)
    amp = np.linalg.norm(d_vec)

    # Spherical direction (RA, Dec in degrees)
    d_unit = d_vec / max(amp, 1e-12)
    dec_dipole = np.degrees(np.arcsin(np.clip(d_unit[2], -1.0, 1.0)))
    ra_dipole = np.degrees(np.arctan2(d_unit[1], d_unit[0])) % 360.0

    return d_vec, amp, ra_dipole, dec_dipole


def run_quaia_dipole_experiment():
    print("=" * 72)
    print("ALL-SKY QUAIA COSMOLOGICAL DIPOLE RECOVERY BENCHMARK")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fits_path = ROOT_DIR / "Archive" / "2026-06-G-dipole" / "data" / "quaia" / "quaia_G20.5.fits"
    ckpt_path = CHECKPOINTS_DIR / "astrojev_h100_scaled.pt"

    print(f"Loading full Quaia catalog: {fits_path}")
    t0 = time.time()
    tab = Table.read(str(fits_path), memmap=True)
    n_sources = len(tab)
    print(f"Ingested {n_sources:,} real sources in {time.time()-t0:.2f}s.")

    # Extract astronomical coordinates and features
    ra = np.array(tab["ra"], dtype=np.float64)
    dec = np.array(tab["dec"], dtype=np.float64)
    gal_l = np.array(tab["l"], dtype=np.float64)
    gal_b = np.array(tab["b"], dtype=np.float64)

    g = np.array(tab["phot_g_mean_mag"], dtype=np.float32)
    bp = np.array(tab["phot_bp_mean_mag"], dtype=np.float32)
    rp = np.array(tab["phot_rp_mean_mag"], dtype=np.float32)
    w1 = np.array(tab["mag_w1_vg"], dtype=np.float32)
    w2 = np.array(tab["mag_w2_vg"], dtype=np.float32)
    pm = np.array(tab["pm"], dtype=np.float32)
    pmerr = np.array(tab["pmra_error"], dtype=np.float32)

    bprp = bp - rp
    gbp = g - bp
    w1w2 = w1 - w2
    g_err = np.clip(0.01 + 0.005 * (g - 17.0)**2, 0.01, 0.25).astype(np.float32)
    w1_err = (g_err * 1.5).astype(np.float32)
    snr = np.clip(1.0857 / g_err, 3.0, 100.0).astype(np.float32)

    X_all = np.column_stack([g, bprp, gbp, w1, w1w2, pm, pmerr, g_err, w1_err, snr]).astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0)

    # Load model
    print(f"Loading Scaled AstroJev model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AstroJev(in_features=NUM_FEATURES, d_model=128, num_classes=NUM_CLASSES, n_iter=5).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    opt_temp = float(ckpt.get("temperature", 0.6227))

    # Inference in batches over all 1.3M sources
    print("Evaluating all-sky probabilistic classification...")
    t_inf = time.time()
    batch_size = 16384
    q_probs_list = []

    with torch.no_grad():
        for i in range(0, n_sources, batch_size):
            bx = torch.from_numpy(X_all[i : i + batch_size]).to(device)
            out = model(bx)
            logits = out["choice_logits"] / opt_temp
            probs = F.softmax(logits, dim=-1)
            q_probs_list.append(probs[:, 0].cpu().numpy())

    q_probs = np.concatenate(q_probs_list)
    dt_inf = time.time() - t_inf
    print(f"Classified {n_sources:,} sources in {dt_inf:.2f}s ({n_sources/dt_inf:,.0f} sources/sec).")

    # 1. Hard Cut Weights (threshold at 0.8)
    w_hard = (q_probs >= 0.80).astype(np.float64)
    # 2. Continuous Probabilistic Weights
    w_prob = q_probs.astype(np.float64)

    # Compute cosmological dipoles
    _, amp_hard, ra_hard, dec_hard = compute_dipole_vector(ra, dec, w_hard)
    _, amp_prob, ra_prob, dec_prob = compute_dipole_vector(ra, dec, w_prob)

    # CMB Dipole reference:
    # Amplitude: D_CMB ~ 0.0070
    # Direction: (l, b) ~ (264 deg, 48 deg) -> (RA, Dec) ~ (168 deg, -7 deg)
    cmb_amp = 0.0070
    cmb_ra = 168.0
    cmb_dec = -7.0

    print("\n" + "=" * 72)
    print("ALL-SKY COSMOLOGICAL DIPOLE RECOVERY RESULTS:")
    print("=" * 72)
    print(f"Reference CMB Dipole:       Amp: {cmb_amp:.4f} | RA: {cmb_ra:.1f}° | Dec: {cmb_dec:.1f}°")
    print(f"Hard Cut (p >= 0.80):       Amp: {amp_hard:.4f} | RA: {ra_hard:.1f}° | Dec: {dec_hard:.1f}° (Bias |D - D_CMB|: {abs(amp_hard - cmb_amp):.4f})")
    print(f"Continuous Probabilistic:   Amp: {amp_prob:.4f} | RA: {ra_prob:.1f}° | Dec: {dec_prob:.1f}° (Bias |D - D_CMB|: {abs(amp_prob - cmb_amp):.4f})")
    bias_reduction = (abs(amp_hard - cmb_amp) - abs(amp_prob - cmb_amp)) / abs(amp_hard - cmb_amp) * 100.0
    print(f"Cosmological Dipole Bias Reduction: {bias_reduction:.2f}%")
    print("=" * 72)

    # Generate Publication Figure
    fig = plt.figure(figsize=(15, 6), dpi=300)

    # Panel 1: Mollweide All-Sky Map of Continuous Probabilistic Density
    ax1 = fig.add_subplot(1, 2, 1, projection="mollweide")
    # Subsample for smooth density visualization
    sub_idx = np.random.choice(n_sources, 100000, replace=False)
    # Convert gal_l, gal_b to radians in [-pi, pi]
    l_rad = np.radians(gal_l[sub_idx])
    l_rad = np.remainder(l_rad + np.pi, 2 * np.pi) - np.pi  # shift to [-pi, pi]
    b_rad = np.radians(gal_b[sub_idx])

    scatter = ax1.scatter(
        l_rad, b_rad,
        c=w_prob[sub_idx],
        cmap="viridis",
        s=0.5,
        alpha=0.6,
        rasterized=True,
    )
    cbar = plt.colorbar(scatter, ax=ax1, orientation="horizontal", pad=0.08, shrink=0.7)
    cbar.set_label("Continuous Calibrated Quasar Credence $p_i$", fontsize=10, fontweight="bold")

    ax1.set_title("(A) All-Sky Continuous Quasar Weight Map\n($1.3\\times 10^6$ Quaia Sources, Gaia DR3 + CatWISE)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Dipole Vector Reconstruction Comparison
    ax2 = fig.add_subplot(1, 2, 2)
    methods = ["CMB Reference\n(True Kinematic)", "Hard Cuts\n(p >= 0.80)", "AstroJev Probabilistic\n(w_i = p_i)"]
    amps = [cmb_amp * 1000, amp_hard * 1000, amp_prob * 1000]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]

    bars = ax2.bar(methods, amps, color=colors, alpha=0.85, width=0.45)
    ax2.set_ylabel("Dipole Amplitude D x 1000", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Cosmological Dipole Recovery & Leakage Suppression", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    for bar, a in zip(bars, amps):
        ax2.annotate(
            f"{a:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.text(
        0.5, 0.85,
        f"Raw Footprint Pseudo-Dipole: D = {amp_prob:.4f}\n"
        f"Continuous weights eliminate threshold edge cuts\n"
        f"Effective N: {np.sum(w_prob):,.0f} clean quasars",
        transform=ax2.transAxes,
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )

    out_path = FIGURES_DIR / "quaia_allsky_dipole_recovery.png"
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, wspace=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {out_path}")


if __name__ == "__main__":
    run_quaia_dipole_experiment()
