"""
=============================================================================
EXP-2026-V: Continuous-Flow Foundation AstroJev for Multi-Survey Cross-Calibration
=============================================================================
Runs multi-survey cross-calibration and autonomous fiber allocation benchmark across:
  - Euclid DR1 Wide: VIS (I_E), NISP (Y_E, J_E, H_E)
  - DESI Legacy Surveys DR10: g, r, z
  - Rubin LSST DP0: u, g, r, i, z, y

Underlying Real Data:
  80,000 real cosmic sources streamed from data/real_phenomena_dataset.npz.

Evaluates:
  1. Conditional Flow Matching (CFM) simulation-free straight-path latent alignment.
  2. Cross-survey zero-point offset recovery (|Delta m_pred - Delta m_true|).
  3. Disentangled RLCD calibration (Stanford debiased calibration error E^2_db).
  4. Autonomous focal-plane fiber allocation (DESI 5,000-fiber focal plane) with
     strict Conformal Risk Control (alpha_risk = 0.02, <= 2.0% false allocation).
  5. Brown Dwarf & Contaminant Null Audit.

Outputs:
  - docs/research/experiment_v_flow_calibration_results.json
  - docs/research/figures/experiment_v_flow_cross_calibration.png
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from celestrium.continuous_flow_astrojev import (
    ALLOCATION_CLASSES,
    NUM_ALLOCATION_CLASSES,
    TOTAL_OBSERVABLE_DIM,
    LATENT_SED_DIM,
    MultiSurveyPhotometryGenerator,
    ContinuousFlowFoundationAstroJev,
    train_continuous_flow_astrojev,
    SpectroscopicFiberTriageEngine,
    SpectroscopicTriageDecision,
)
from celestrium.astrojev import evaluate_calibration


def run_benchmark():
    print("=" * 76)
    print("EXP-2026-V: CONTINUOUS-FLOW FOUNDATION ASTROJEV MULTI-SURVEY CROSS-CALIBRATION")
    print("=" * 76)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ContinuousFlow] Computing Device: {device}")

    # 1. Ingest Real Phenomena Dataset
    data_path = ROOT_DIR / "data" / "real_phenomena_dataset.npz"
    print(f"[ContinuousFlow] Loading real multi-phenomena dataset from {data_path}...")
    real_data = np.load(str(data_path))
    raw_mu = real_data["mu"]         # (80000, 10)
    raw_labels = real_data["labels"] # (80000,)
    print(f"  Ingested {len(raw_mu):,} real astronomical sources across 12 phenomena classes.")

    # 2. Generate Multi-Survey Photometry (Euclid x DESI x Rubin)
    print("\n[ContinuousFlow] Generating 13-band panchromatic photometry (Euclid VIS/NISP, DESI, Rubin)...")
    generator = MultiSurveyPhotometryGenerator(rng_seed=42)
    # Use subset of 12,000 sources for highly responsive benchmark
    subset_n = 12000
    indices = np.random.choice(len(raw_mu), subset_n, replace=False)
    base_mu = raw_mu[indices]
    labels_sub = raw_labels[indices]

    phot, errors, masks, targets, zp_true = generator.generate_survey_photometry(
        base_mu=base_mu,
        labels=labels_sub,
        zp_jitter_mag=0.035
    )

    # Pack 39-D input condition vector c = [photometry, errors, masks]
    c_all = np.column_stack([phot, errors, masks]).astype(np.float32)

    # Train / Cal / Test split
    n_train = 8000
    n_cal = 2000
    n_test = 2000

    c_train = torch.tensor(c_all[:n_train], device=device)
    y_train = torch.tensor(targets[:n_train], device=device)
    zp_train = torch.tensor(zp_true[:n_train], device=device)

    c_cal = torch.tensor(c_all[n_train:n_train + n_cal], device=device)
    y_cal = torch.tensor(targets[n_train:n_train + n_cal], device=device)

    c_test = torch.tensor(c_all[n_train + n_cal:], device=device)
    y_test = targets[n_train + n_cal:]
    zp_test = zp_true[n_train + n_cal:]

    print(f"  Split: {n_train:,} Train | {n_cal:,} Calibration | {n_test:,} Test Evaluation")

    # 3. Initialize & Train Continuous-Flow Foundation AstroJev
    print("\n" + "-" * 76)
    print("Stage 1: Training Conditional Flow Matching & Disentangled RLCD...")
    print("-" * 76)
    model = ContinuousFlowFoundationAstroJev(
        z_dim=LATENT_SED_DIM,
        cond_dim=TOTAL_OBSERVABLE_DIM,
        num_classes=NUM_ALLOCATION_CLASSES,
        hidden_dim=128
    ).to(device)

    train_summary = train_continuous_flow_astrojev(
        model=model,
        c_train=c_train,
        y_train=y_train,
        zp_train=zp_train,
        num_epochs=18,
        lr=1e-3,
        verbose=False
    )
    print(f"  * CFM & Evidential Optimization Loss: {train_summary['final_loss']:.4f}")

    # 4. Evaluate Zero-Point Recovery & Calibration on Test Set
    print("\n" + "-" * 76)
    print("Stage 2: Evaluating Zero-Point Disentanglement & Debiased Calibration...")
    print("-" * 76)
    model.eval()
    with torch.no_grad():
        alpha_test, zp_pred_test, z_1_test = model(c_test, num_ode_steps=6)
        probs_test, stds_test, u_epi_test, u_ale_test = model.compute_dirichlet_statistics(alpha_test)

    zp_pred_np = zp_pred_test.cpu().numpy()
    probs_np = probs_test.cpu().numpy()
    stds_np = stds_test.cpu().numpy()
    pred_classes = np.argmax(probs_np, axis=-1)

    # Zero-point recovery error
    zp_residuals = np.abs(zp_pred_np - zp_test)
    mean_zp_err = float(np.mean(zp_residuals))
    rmse_zp = float(np.sqrt(np.mean(zp_residuals ** 2)))
    print(f"  * Zero-Point Recovery RMSE: {rmse_zp:.4f} mag (Mean Absolute Error: {mean_zp_err:.4f} mag)")

    # Stanford Debiased Calibration Error & ECE
    cal_res = evaluate_calibration(probs_np, y_test, n_bins=15)
    ece_val = float(cal_res["ece"])
    edb_val = float(cal_res["debiased_squared_ce"])
    print(f"  * Binned ECE: {ece_val * 100:.2f}%")
    print(f"  * Stanford Debiased Calibration Error E^2_db: {edb_val:.6f}")

    # 5. Calibrate Conformal Gate & Spectroscopic Fiber Triage
    print("\n" + "-" * 76)
    print("Stage 3: Calibrating Spectroscopic Fiber Allocation & Conformal Gate...")
    print("-" * 76)
    triage_engine = SpectroscopicFiberTriageEngine(model, alpha_risk=0.02)
    conf_thresh = triage_engine.calibrate_conformal_gate(c_cal, y_cal)
    print(f"  * Calibrated Conformal Gate Threshold: tau_conformal = {conf_thresh:.4f} (alpha_risk = 2.0%)")

    # Triage test set
    triage_decisions: List[SpectroscopicTriageDecision] = []
    for i in range(len(c_test)):
        dec = triage_engine.triage_target(c_test[i])
        triage_decisions.append(dec)

    allocated_decisions = [d for d in triage_decisions if not d.is_rejected]
    rejected_decisions = [d for d in triage_decisions if d.is_rejected]

    total_allocated_hours = sum(d.fiber_exposure_minutes for d in allocated_decisions) / 60.0
    print(f"  * Total Targets Triaged: {len(triage_decisions):,}")
    print(f"  * Dispatched Fibers: {len(allocated_decisions):,} ({len(allocated_decisions) / len(triage_decisions):.1%}) | Total Exposure: {total_allocated_hours:.1f} fiber-hours")
    print(f"  * Purged Contaminants: {len(rejected_decisions):,} ({len(rejected_decisions) / len(triage_decisions):.1%})")

    # 6. Contaminant Null Audit (True Contaminants vs Allocated Fibers)
    contaminant_indices = np.where(y_test == 4)[0]
    n_contaminants = len(contaminant_indices)
    wasted_fibers = sum(1 for idx in contaminant_indices if not triage_decisions[idx].is_rejected)
    waste_rate = wasted_fibers / max(n_contaminants, 1)

    quasar_indices = np.where(y_test == 0)[0]
    n_quasars = len(quasar_indices)
    quasar_allocated = sum(1 for idx in quasar_indices if not triage_decisions[idx].is_rejected and triage_decisions[idx].target_class == "HIGH_Z_QUASAR_120M")
    quasar_recall = quasar_allocated / max(n_quasars, 1)

    print(f"\n[Spectroscopic Audit]")
    print(f"  * Brown Dwarf & Contaminant Interlopers: {n_contaminants:,}")
    print(f"  * Wasted Fibers on Contaminants: {wasted_fibers} (False Allocation Rate: {waste_rate:.2%}, Target <= 2.0%)")
    print(f"  * High-z Quasar Recall: {quasar_recall:.1%} ({quasar_allocated}/{n_quasars} targets awarded 120-min fibers)")

    # 7. Save Summary Artifact JSON
    results_dir = ROOT_DIR / "docs" / "research"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = results_dir / "experiment_v_flow_calibration_results.json"

    benchmark_summary = {
        "experiment": "EXP-2026-V: Continuous-Flow Foundation AstroJev for Multi-Survey Cross-Calibration",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_size": subset_n,
        "device": str(device),
        "zero_point_recovery": {
            "rmse_mag": rmse_zp,
            "mae_mag": mean_zp_err,
        },
        "calibration_metrics": {
            "binned_ece": ece_val,
            "stanford_debiased_edb2": edb_val,
        },
        "fiber_triage": {
            "tau_conformal": conf_thresh,
            "target_risk_level": 0.02,
            "total_evaluated": len(triage_decisions),
            "allocated_fibers": len(allocated_decisions),
            "purged_contaminants": len(rejected_decisions),
            "total_fiber_hours": total_allocated_hours,
            "wasted_contaminant_fibers": wasted_fibers,
            "contaminant_false_allocation_rate": waste_rate,
            "high_z_quasar_recall": quasar_recall,
        },
        "execution_time_seconds": time.time() - t0,
    }

    with open(results_json_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"\n[Artifact] Summary written to {results_json_path}")

    # 8. Generate Publication Figure
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "experiment_v_flow_cross_calibration.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#0b0f19")
    for ax in axes.flat:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Panel 1: Zero-Point Recovery
    ax1 = axes[0, 0]
    ax1.scatter(zp_test[:, 0], zp_pred_np[:, 0], alpha=0.4, color="#38bdf8", s=15, edgecolors="none", label="Test Sources")
    ax1.plot([-0.10, 0.10], [-0.10, 0.10], color="#f43f5e", linestyle="--", linewidth=2.0, label="Ideal Recovery (1:1)")
    ax1.set_xlim(-0.08, 0.08)
    ax1.set_ylim(-0.08, 0.08)
    ax1.set_title(f"Cross-Survey Zero-Point Recovery (RMSE = {rmse_zp:.4f} mag)", color="#f9fafb", fontsize=12, pad=10)
    ax1.set_xlabel("True Injected Zero-Point Shift Delta m_true [mag]", color="#d1d5db")
    ax1.set_ylabel("Flow-Recovered Shift Delta m_pred [mag]", color="#d1d5db")
    ax1.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.3, color="#4b5563")

    # Panel 2: Continuous Flow Latent Manifold Embedding (t=0 vs t=1)
    ax2 = axes[0, 1]
    with torch.no_grad():
        z_0_sample = torch.randn(25, LATENT_SED_DIM, device=device)
        c_sample = c_test[:25]
        t_steps = np.linspace(0, 1, 8)
        dt = 1.0 / 7
        traj = [z_0_sample.detach().cpu().numpy()[:, 0:2]]
        z_curr = z_0_sample
        for s_idx in range(7):
            t_val = torch.full((25,), s_idx * dt, device=device)
            v = model.velocity_net(z_curr, t_val, c_sample)
            z_curr = z_curr + dt * v
            traj.append(z_curr.detach().cpu().numpy()[:, 0:2])

    traj_arr = np.array(traj)  # (8, 25, 2)
    for p_idx in range(25):
        ax2.plot(traj_arr[:, p_idx, 0], traj_arr[:, p_idx, 1], color="#38bdf8", alpha=0.4, linewidth=1.2)
    ax2.scatter(traj_arr[0, :, 0], traj_arr[0, :, 1], color="#9ca3af", s=30, label="Base Noise z_0 ~ N(0, I)")
    ax2.scatter(traj_arr[-1, :, 0], traj_arr[-1, :, 1], color="#a855f7", s=45, label="Invariant Latent SED z_1")
    ax2.set_title("Conditional Flow ODE Latent Trajectories (z_0 -> z_1)", color="#f9fafb", fontsize=12, pad=10)
    ax2.set_xlabel("Latent Coordinate z_1", color="#d1d5db")
    ax2.set_ylabel("Latent Coordinate z_2", color="#d1d5db")
    ax2.legend(loc="lower right", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.3, color="#4b5563")

    # Panel 3: Calibration Reliability & E^2_db
    ax3 = axes[1, 0]
    confidences = np.max(probs_np, axis=-1)
    y_test_bin = (pred_classes == y_test).astype(int)
    bins = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    emp_accs = []
    for b_idx in range(len(bins) - 1):
        mask_b = (confidences >= bins[b_idx]) & (confidences < bins[b_idx + 1])
        if np.sum(mask_b) > 0:
            emp_accs.append(np.mean(y_test_bin[mask_b]))
        else:
            emp_accs.append(bin_centers[b_idx])

    ax3.plot([0, 1], [0, 1], color="#f43f5e", linestyle="--", linewidth=2.0, label="Perfect Calibration")
    ax3.plot(bin_centers, emp_accs, marker="o", color="#34d399", linewidth=2.5, label=f"Disentangled RLCD (E^2_db = {edb_val:.6f})")
    ax3.fill_between(bin_centers, bin_centers, emp_accs, color="#34d399", alpha=0.15)
    ax3.set_title(f"Reliability Diagram (Binned ECE = {ece_val * 100:.2f}%)", color="#f9fafb", fontsize=12, pad=10)
    ax3.set_xlabel("Predicted Dirichlet Confidence max p_k", color="#d1d5db")
    ax3.set_ylabel("Empirical Allocation Accuracy", color="#d1d5db")
    ax3.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.3, color="#4b5563")

    # Panel 4: Spectroscopic Fiber Allocation Efficiency
    ax4 = axes[1, 1]
    classes = ["Quasar (120m)", "ELG (45m)", "LRG (60m)", "Calibration (15m)", "Purged (0m)"]
    alloc_counts = [
        sum(1 for d in triage_decisions if d.target_class == "HIGH_Z_QUASAR_120M" and not d.is_rejected),
        sum(1 for d in triage_decisions if d.target_class == "ELG_TRACER_45M" and not d.is_rejected),
        sum(1 for d in triage_decisions if d.target_class == "LRG_RULER_60M" and not d.is_rejected),
        sum(1 for d in triage_decisions if d.target_class == "CALIBRATION_STAR_15M" and not d.is_rejected),
        len(rejected_decisions),
    ]
    bar_colors = ["#38bdf8", "#34d399", "#fbbf24", "#a855f7", "#ef4444"]
    bars = ax4.bar(classes, alloc_counts, color=bar_colors, edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax4.annotate(f"{h:,}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom", color="#f3f4f6", fontweight="bold", fontsize=9)
    ax4.set_title(f"DESI 5,000-Fiber Allocation (Contaminant False Rate: {waste_rate:.2%})", color="#f9fafb", fontsize=12, pad=10)
    ax4.set_ylabel("Allocated Focal-Plane Targets", color="#d1d5db")
    ax4.tick_params(axis="x", rotation=15)
    ax4.grid(True, linestyle=":", alpha=0.3, color="#4b5563")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Artifact] Publication figure saved to {fig_path}")
    print(f"[ContinuousFlow] Benchmark completed in {time.time() - t0:.2f}s.\n")


if __name__ == "__main__":
    run_benchmark()
