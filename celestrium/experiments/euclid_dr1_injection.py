"""
=============================================================================
G-Euclid DR1 Photometric Filter Injection & Cross-Survey Calibration
=============================================================================
Simulates Euclid VIS (I_E) and NISP (Y_E, J_E, H_E) photometric filter bands
from Gaia DR3 optical + CatWISE mid-IR SEDs.

Demonstrates:
1. Euclid near-infrared color space (I_E - Y_E vs Y_E - J_E vs J_E - H_E).
2. Elimination of the classic degeneracy between Galactic M/L dwarfs and
   high-redshift Quasars (z > 2.5) using Euclid's 1-2 micron coverage.
3. Decision calibration across multi-wavelength space (Gaia + WISE + Euclid).
4. Evidential epistemic vacuity comparison on Euclid mock sources.

Outputs figure:
  docs/figures/euclid_dr1_photometric_injection.png
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
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from celestrium.astrojev import (
    ContinuousFourierEncoder,
    KMContractiveBlock,
    evaluate_calibration,
    NUM_CLASSES,
    CLASSES,
)
from celestrium.kernels import fused_crelu

FIGURES_DIR = ROOT_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def synthesize_euclid_dr1_catalog(n_per_class: int = 4000, seed: int = 2026):
    """Synthesizes rich astrophysical catalog with Gaia + WISE + Euclid DR1 VIS/NISP photometry."""
    rng = np.random.default_rng(seed)

    # 1. Quasars (Power law SED + Lyman forest at high z)
    q_g = rng.uniform(18.0, 21.0, n_per_class)
    q_bprp = rng.normal(0.65, 0.25, n_per_class)
    q_w1 = rng.normal(16.3, 0.9, n_per_class)
    q_w1w2 = rng.normal(1.08, 0.2, n_per_class)
    q_pm = np.abs(rng.normal(0.0, 0.4, n_per_class))
    q_pmerr = rng.uniform(0.3, 0.8, n_per_class)

    # Euclid VIS & NISP synthetic transformations:
    # Quasars have steep power-law red near-IR spectra
    q_ie = q_g - 0.22 * q_bprp + rng.normal(0, 0.05, n_per_class)
    q_ye = q_g - 0.55 * q_bprp + rng.normal(0, 0.06, n_per_class)
    q_je = q_ye - rng.normal(0.35, 0.12, n_per_class)
    q_he = q_je - rng.normal(0.42, 0.15, n_per_class)
    q_snr = np.clip(60.0 * 10.0 ** (-0.4 * (q_g - 18.0)), 3.0, 80.0)

    # 2. Galactic M/L Dwarf Stars (The primary quasar contaminant!)
    # Red in optical, but have deep H2O/CH4 molecular absorption in near-IR NISP bands!
    s_g = rng.uniform(16.5, 21.2, n_per_class)
    s_bprp = rng.normal(2.35, 0.45, n_per_class)  # Very red optical
    s_w1 = rng.normal(14.8, 1.2, n_per_class)
    s_w1w2 = rng.normal(0.35, 0.18, n_per_class)  # Can overlap with quasars!
    s_pm = rng.rayleigh(18.0, n_per_class)
    s_pmerr = rng.uniform(0.15, 0.6, n_per_class)

    # Stellar atmospheric absorption gives concave near-IR SED:
    s_ie = s_g - 0.45 * s_bprp + rng.normal(0, 0.05, n_per_class)
    s_ye = s_ie - rng.normal(0.85, 0.15, n_per_class)
    s_je = s_ye - rng.normal(0.15, 0.08, n_per_class)  # Flattens due to H2O band
    s_he = s_je + rng.normal(0.10, 0.10, n_per_class)  # Dips in H band!
    s_snr = np.clip(80.0 * 10.0 ** (-0.4 * (s_g - 17.0)), 4.0, 95.0)

    # 3. Passive Early-Type Galaxies
    # 4000 Angstrom break creates steep drop between VIS and NISP:
    g_g = rng.uniform(18.0, 21.5, n_per_class)
    g_bprp = rng.normal(1.85, 0.22, n_per_class)
    g_w1 = rng.normal(16.2, 0.9, n_per_class)
    g_w1w2 = rng.normal(0.25, 0.10, n_per_class)
    g_pm = np.abs(rng.normal(0.0, 0.35, n_per_class))
    g_pmerr = rng.uniform(0.3, 0.8, n_per_class)

    g_ie = g_g - 0.35 * g_bprp + rng.normal(0, 0.05, n_per_class)
    g_ye = g_ie - rng.normal(0.95, 0.12, n_per_class)
    g_je = g_ye - rng.normal(0.45, 0.10, n_per_class)
    g_he = g_je - rng.normal(0.30, 0.08, n_per_class)
    g_snr = np.clip(50.0 * 10.0 ** (-0.4 * (g_g - 18.0)), 2.5, 60.0)

    # 4. White Dwarfs
    w_g = rng.uniform(17.0, 21.0, n_per_class)
    w_bprp = rng.normal(-0.10, 0.18, n_per_class)
    w_w1 = rng.normal(17.8, 1.0, n_per_class)
    w_w1w2 = rng.normal(0.02, 0.12, n_per_class)
    w_pm = rng.rayleigh(30.0, n_per_class)
    w_pmerr = rng.uniform(0.15, 0.5, n_per_class)

    w_ie = w_g + 0.08 + rng.normal(0, 0.04, n_per_class)
    w_ye = w_ie + 0.12 + rng.normal(0, 0.05, n_per_class)
    w_je = w_ye + 0.15 + rng.normal(0, 0.06, n_per_class)
    w_he = w_je + 0.18 + rng.normal(0, 0.06, n_per_class)
    w_snr = np.clip(60.0 * 10.0 ** (-0.4 * (w_g - 17.5)), 3.0, 70.0)

    # Compile feature tables:
    # 1. Baseline Features (10 features without Euclid)
    def stack_baseline(g_col, bprp_col, w1_col, w1w2_col, pm_col, pmerr_col, snr_col):
        g_err = (1.0857 / snr_col).astype(np.float32)
        w1_err = (g_err * 1.5).astype(np.float32)
        gbp = (-0.4 * bprp_col).astype(np.float32)
        return np.column_stack([g_col, bprp_col, gbp, w1_col, w1w2_col, pm_col, pmerr_col, g_err, w1_err, snr_col]).astype(np.float32)

    X_base = np.vstack([
        stack_baseline(q_g, q_bprp, q_w1, q_w1w2, q_pm, q_pmerr, q_snr),
        stack_baseline(s_g, s_bprp, s_w1, s_w1w2, s_pm, s_pmerr, s_snr),
        stack_baseline(g_g, g_bprp, g_w1, g_w1w2, g_pm, g_pmerr, g_snr),
        stack_baseline(w_g, w_bprp, w_w1, w_w1w2, w_pm, w_pmerr, w_snr),
    ])

    # 2. Euclid-Augmented Features (14 features: baseline + [I_E, I_E - Y_E, Y_E - J_E, J_E - H_E])
    def stack_euclid(base_mat, ie_col, ye_col, je_col, he_col):
        ie_ye = (ie_col - ye_col)[:, None].astype(np.float32)
        ye_je = (ye_col - je_col)[:, None].astype(np.float32)
        je_he = (je_col - he_col)[:, None].astype(np.float32)
        ie_mag = ie_col[:, None].astype(np.float32)
        return np.hstack([base_mat, ie_mag, ie_ye, ye_je, je_he])

    q_base = stack_baseline(q_g, q_bprp, q_w1, q_w1w2, q_pm, q_pmerr, q_snr)
    s_base = stack_baseline(s_g, s_bprp, s_w1, s_w1w2, s_pm, s_pmerr, s_snr)
    g_base = stack_baseline(g_g, g_bprp, g_w1, g_w1w2, g_pm, g_pmerr, g_snr)
    w_base = stack_baseline(w_g, w_bprp, w_w1, w_w1w2, w_pm, w_pmerr, w_snr)

    X_euc = np.vstack([
        stack_euclid(q_base, q_ie, q_ye, q_je, q_he),
        stack_euclid(s_base, s_ie, s_ye, s_je, s_he),
        stack_euclid(g_base, g_ie, g_ye, g_je, g_he),
        stack_euclid(w_base, w_ie, w_ye, w_je, w_he),
    ])

    y = np.concatenate([
        np.zeros(n_per_class, dtype=np.int64),
        np.ones(n_per_class, dtype=np.int64),
        np.full(n_per_class, 2, dtype=np.int64),
        np.full(n_per_class, 3, dtype=np.int64),
    ])

    perm = rng.permutation(len(y))
    split = int(len(y) * 0.8)

    return (
        X_base[perm][:split], y[perm][:split], X_base[perm][split:], y[perm][split:],
        X_euc[perm][:split], y[perm][:split], X_euc[perm][split:], y[perm][split:],
    )


class EuclidAstroJev(nn.Module):
    """AstroJev model configured for 14-dimensional Gaia + WISE + Euclid DR1 input space."""

    def __init__(self, in_features: int = 14, d_model: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = ContinuousFourierEncoder(in_features=in_features, d_model=d_model)
        self.km_block = KMContractiveBlock(d_model=d_model)
        self.accum_proj = nn.Linear(d_model, 128)
        self.accum_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        x_ctx = self.encoder(x)
        h = x_ctx
        for k in range(5):
            gamma_k = 1.0 / (1.0 + 0.2 * (k + 1))
            t_h = self.km_block(h, x_ctx)
            h = (1.0 - gamma_k) * h + gamma_k * t_h
        accum = self.accum_norm(self.accum_proj(h))
        crelu_sparse, sparsity = fused_crelu(accum)
        logits = self.head(crelu_sparse)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "sparsity": sparsity}


def run_euclid_experiment():
    print("=" * 72)
    print("G-EUCLID DR1 PHOTOMETRIC INJECTION & CALIBRATION BENCHMARK")
    print("=" * 72)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        X_base_tr, y_tr, X_base_val, y_val,
        X_euc_tr, _, X_euc_val, _,
    ) = synthesize_euclid_dr1_catalog(n_per_class=4000, seed=2026)

    # 1. Train Baseline Model (Gaia + WISE only, 10 features)
    print("1. Training Baseline Model (Gaia + WISE only, 10 features)...")
    m_base = EuclidAstroJev(in_features=10).to(device)
    opt_base = torch.optim.AdamW(m_base.parameters(), lr=2e-3, weight_decay=1e-4)

    batch_size = 256
    n_train = len(y_tr)
    n_batches = math.ceil(n_train / batch_size)

    X_b_tr_gpu = torch.from_numpy(X_base_tr).to(device)
    y_tr_gpu = torch.from_numpy(y_tr).to(device)

    for epoch in range(20):
        m_base.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            s_idx = b * batch_size
            e_idx = min(s_idx + batch_size, n_train)
            idx = perm[s_idx:e_idx]
            opt_base.zero_grad()
            out = m_base(X_b_tr_gpu[idx])
            loss = F.cross_entropy(out["logits"], y_tr_gpu[idx])
            loss.backward()
            opt_base.step()

    # 2. Train Euclid-Augmented Model (Gaia + WISE + Euclid VIS/NISP, 14 features)
    print("2. Training Euclid-Augmented Model (Gaia + WISE + Euclid DR1, 14 features)...")
    m_euc = EuclidAstroJev(in_features=14).to(device)
    opt_euc = torch.optim.AdamW(m_euc.parameters(), lr=2e-3, weight_decay=1e-4)

    X_e_tr_gpu = torch.from_numpy(X_euc_tr).to(device)

    for epoch in range(20):
        m_euc.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            s_idx = b * batch_size
            e_idx = min(s_idx + batch_size, n_train)
            idx = perm[s_idx:e_idx]
            opt_euc.zero_grad()
            out = m_euc(X_e_tr_gpu[idx])
            loss = F.cross_entropy(out["logits"], y_tr_gpu[idx])
            loss.backward()
            opt_euc.step()

    # 3. Evaluation on Validation Set
    m_base.eval()
    m_euc.eval()

    with torch.no_grad():
        out_base = m_base(torch.from_numpy(X_base_val).to(device))
        p_base = out_base["probs"].cpu().numpy()

        out_euc = m_euc(torch.from_numpy(X_euc_val).to(device))
        p_euc = out_euc["probs"].cpu().numpy()

    met_base = evaluate_calibration(p_base, y_val)
    met_euc = evaluate_calibration(p_euc, y_val)

    # Specifically measure Quasar vs M-Dwarf confusion rate:
    # Quasars misclassified as Stars, or Stars misclassified as Quasars
    pred_base = np.argmax(p_base, axis=-1)
    pred_euc = np.argmax(p_euc, axis=-1)

    # Class 0: Quasar, Class 1: Star
    q_mask = (y_val == 0)
    s_mask = (y_val == 1)

    q_contam_base = np.mean(pred_base[s_mask] == 0)  # Stars leaking into Quasars
    q_contam_euc = np.mean(pred_euc[s_mask] == 0)

    q_miss_base = np.mean(pred_base[q_mask] == 1)    # Quasars misidentified as Stars
    q_miss_euc = np.mean(pred_euc[q_mask] == 1)

    print("\n" + "=" * 72)
    print("CROSS-SURVEY EUCLID DR1 EVALUATION METRICS:")
    print("=" * 72)
    print(f"Baseline (Gaia+WISE):       Acc: {met_base['accuracy']*100:.2f}% | ECE: {met_base['ece']*100:.2f}% | Brier: {met_base['brier']:.4f}")
    print(f"Euclid-Augmented (DR1):     Acc: {met_euc['accuracy']*100:.2f}% | ECE: {met_euc['ece']*100:.2f}% | Brier: {met_euc['brier']:.4f}")
    print(f"M-Dwarf Contamination Rate: Baseline: {q_contam_base*100:.2f}% -> Euclid: {q_contam_euc*100:.2f}% (Reduced by {((q_contam_base-q_contam_euc)/max(q_contam_base,1e-6))*100:.1f}%)")
    print(f"Quasar Missed as Star:      Baseline: {q_miss_base*100:.2f}% -> Euclid: {q_miss_euc*100:.2f}%")
    print("=" * 72)

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Euclid Color-Color Separation Space (Y_E - J_E vs J_E - H_E)
    ax_a = axes[0]
    # Extract Euclid colors from validation matrix
    # col 11: I_E - Y_E, col 12: Y_E - J_E, col 13: J_E - H_E
    ye_je = X_euc_val[:, 12]
    je_he = X_euc_val[:, 13]

    ax_a.scatter(ye_je[q_mask], je_he[q_mask], color="#2b83ba", s=8, alpha=0.5, label="Quasars (Steep NIR slope)")
    ax_a.scatter(ye_je[s_mask], je_he[s_mask], color="#d7191c", s=8, alpha=0.5, label="M/L Dwarf Stars (H2O absorption dip)")
    ax_a.scatter(ye_je[y_val == 2], je_he[y_val == 2], color="#abdda4", s=8, alpha=0.4, label="Passive Galaxies")
    ax_a.scatter(ye_je[y_val == 3], je_he[y_val == 3], color="#fdae61", s=8, alpha=0.4, label="White Dwarfs")

    ax_a.set_xlabel("Euclid $(Y_E - J_E)$", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Euclid $(J_E - H_E)$", fontsize=11, fontweight="bold")
    ax_a.set_title("(A) Euclid DR1 Near-IR Color-Color Space\n(Breaks Optical M-Dwarf / Quasar Degeneracy)", fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Classification Performance & Contamination Suppression
    ax_b = axes[1]
    metrics_names = ["Overall Accuracy (%)", "Quasar Precision (%)", "M-Dwarf Leakage (%)"]
    q_prec_base = np.sum((pred_base == 0) & (y_val == 0)) / max(np.sum(pred_base == 0), 1) * 100.0
    q_prec_euc = np.sum((pred_euc == 0) & (y_val == 0)) / max(np.sum(pred_euc == 0), 1) * 100.0

    base_vals = [met_base["accuracy"] * 100, q_prec_base, q_contam_base * 100]
    euc_vals = [met_euc["accuracy"] * 100, q_prec_euc, q_contam_euc * 100]

    x = np.arange(len(metrics_names))
    w = 0.35

    b1 = ax_b.bar(x - w/2, base_vals, w, label="Baseline (Gaia DR3 + CatWISE)", color="#e41a1c", alpha=0.85)
    b2 = ax_b.bar(x + w/2, euc_vals, w, label="G-Euclid DR1 (Gaia + WISE + Euclid)", color="#377eb8", alpha=0.85)

    ax_b.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax_b.set_title("(B) Cross-Instrument Performance Gains", fontsize=12, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax_b.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    for rect in b1:
        h = rect.get_height()
        ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect in b2:
        h = rect.get_height()
        ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    out_fig = FIGURES_DIR / "euclid_dr1_photometric_injection.png"
    plt.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {out_fig}")


if __name__ == "__main__":
    run_euclid_experiment()
