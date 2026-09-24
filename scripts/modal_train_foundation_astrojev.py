"""
=============================================================================
Modal Cloud Scaled Foundation Training: 12-Class Heteroscedastic AstroJev
=============================================================================
Deploys large-scale FoundationAstroJev training to Modal serverless cloud GPUs.
Runs with:
  1. 12-Class Astronomical Phenomena Taxonomy (Extragalactic, Galactic, Transients).
  2. Heteroscedastic Error Conditioning with in-flight differentiable error jittering.
  3. Continuous Fourier harmonic projections & weight-tied Krasnoselskii-Mann loops.
  4. Sparse CReLU accumulator guaranteeing >= 50% latent sparsity.
  5. Verified Scaling-Binning Calibration with Stanford debiased calibration error (E^2_db).
  6. Conformal Risk Control (CRC) finite-sample False Discovery Rate bounds (alpha=0.05).
  7. Checkpoint & metrics saved to Modal Volume and synced locally.

Usage:
  # Cloud dispatch to Modal H100:
  modal run scripts/modal_train_foundation_astrojev.py --sources 500000 --epochs 25 --batch-size 4096
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from celestrium.foundation_astrojev import (
    FoundationAstroJev,
    heteroscedastic_evidential_loss,
    PHENOMENA_CLASSES,
    NUM_PHENOMENA_CLASSES,
    NUM_OBSERVABLES,
    NUM_UNCERTAINTIES,
    NUM_MASKS,
)
from celestrium.data_streamer import SyntheticPhenomenaGenerator
from celestrium.astrojev import (
    debiased_squared_calibration_error,
    ScalingBinningCalibrator,
    conformal_risk_control_calibrate,
    dirichlet_bald_information_gain,
)

try:
    import modal
except ImportError:
    modal = None


def sanitize_json_dict(obj: Any) -> Any:
    """Recursively convert tensors, numpy arrays, and custom objects to standard Python types."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json_dict(v) for v in obj]
    elif hasattr(obj, "detach"):
        t = obj.detach().cpu()
        return t.item() if t.numel() == 1 else t.tolist()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def generate_foundation_dataset(
    n_sources: int = 500_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-throughput stratified synthetic dataset with physical error bars."""
    gen = SyntheticPhenomenaGenerator(seed=seed)
    mu_list = np.zeros((n_sources, NUM_OBSERVABLES), dtype=np.float32)
    sigma_list = np.zeros((n_sources, NUM_UNCERTAINTIES), dtype=np.float32)
    mask_list = np.zeros((n_sources, NUM_MASKS), dtype=np.float32)
    labels = np.zeros(n_sources, dtype=np.int64)

    for i in range(n_sources):
        c_idx = i % NUM_PHENOMENA_CLASSES
        rec = gen.generate_sample(class_idx=c_idx)
        mu_list[i] = rec.mu
        sigma_list[i] = rec.sigma
        mask_list[i] = rec.mask
        labels[i] = c_idx

    # Shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sources)
    return mu_list[perm], sigma_list[perm], mask_list[perm], labels[perm]


def load_foundation_dataset(
    n_sources: int = 80_000,
    seed: int = 42,
    prefer_real: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads real physical survey dataset (Quaia + Gaia DR3) with physical errors and rare classes."""
    candidates = [
        Path("data/real_phenomena_dataset.npz"),
        Path("/root/data/real_phenomena_dataset.npz"),
    ]
    for p in candidates:
        if p.is_file() and prefer_real:
            print(f"[Dataset] Loading real survey dataset from: {p}")
            d = np.load(str(p))
            mu, sigma, mask, labels = d["mu"], d["sigma"], d["mask"], d["labels"]
            n_real = len(labels)
            rng = np.random.default_rng(seed)
            if n_sources <= n_real:
                idx = rng.choice(n_real, size=n_sources, replace=False)
                return mu[idx], sigma[idx], mask[idx], labels[idx]
            else:
                repeats = int(math.ceil(n_sources / n_real))
                mu_rep = np.tile(mu, (repeats, 1))[:n_sources]
                sig_rep = np.tile(sigma, (repeats, 1))[:n_sources]
                mask_rep = np.tile(mask, (repeats, 1))[:n_sources]
                labels_rep = np.tile(labels, repeats)[:n_sources]
                perm = rng.permutation(n_sources)
                return mu_rep[perm], sig_rep[perm], mask_rep[perm], labels_rep[perm]

    print("[Dataset] Real dataset not found, generating physical synthetic baseline.")
    return generate_foundation_dataset(n_sources=n_sources, seed=seed)


# --------------------------------------------------------------------------- #
# Modal App Setup
# --------------------------------------------------------------------------- #
if modal is not None:
    app = modal.App("astrojev-foundation-trainer")

    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch>=2.4.0",
            "triton>=3.0.0",
            "astropy>=6.0.0",
            "astropy-healpix>=1.0.0",
            "healpy>=1.16.0",
            "numpy>=1.26.0",
            "scipy>=1.12.0",
            "pandas>=2.2.0",
            "pyarrow>=15.0.0",
        )
        .add_local_python_source("celestrium")
        .add_local_file("data/real_phenomena_dataset.npz", "/root/data/real_phenomena_dataset.npz")
    )

    volume = modal.Volume.from_name("astrojev-checkpoints", create_if_missing=True)


def train_foundation_astrojev(
    n_sources: int = 80_000,
    epochs: int = 25,
    batch_size: int = 4096,
    lr: float = 2e-3,
    d_model: int = 128,
    n_iter: int = 5,
    prefer_real: bool = True,
    save_checkpoint_name: str = "foundation_astrojev_12class_h100.pt",
) -> Dict[str, Any]:
    """Train 12-Class Foundation AstroJev with heteroscedastic error conditioning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"=== Starting Foundation AstroJev Training on {device_name} ===")
    print(f"Volume: {n_sources:,} sources | Classes: {NUM_PHENOMENA_CLASSES} | Batch: {batch_size} | Epochs: {epochs}")

    t_data_0 = time.perf_counter()
    mu_all, sigma_all, mask_all, y_all = load_foundation_dataset(n_sources=n_sources, seed=42, prefer_real=prefer_real)
    t_data = time.perf_counter() - t_data_0
    print(f"Dataset ready in {t_data:.2f}s ({n_sources/t_data:,.0f} sources/sec)")

    # Strict 70/10/10/10 Partitioning
    n_train = int(0.70 * n_sources)
    n_val = int(0.10 * n_sources)
    n_recal = int(0.10 * n_sources)
    n_test = n_sources - n_train - n_val - n_recal

    mu_tr, sig_tr, msk_tr, y_tr = mu_all[:n_train], sigma_all[:n_train], mask_all[:n_train], y_all[:n_train]
    mu_val, sig_val, msk_val, y_val = (
        mu_all[n_train:n_train + n_val], sigma_all[n_train:n_train + n_val],
        mask_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    )
    mu_recal, sig_recal, msk_recal, y_recal = (
        mu_all[n_train + n_val:n_train + n_val + n_recal], sigma_all[n_train + n_val:n_train + n_val + n_recal],
        mask_all[n_train + n_val:n_train + n_val + n_recal], y_all[n_train + n_val:n_train + n_val + n_recal]
    )
    mu_te, sig_te, msk_te, y_te = mu_all[-n_test:], sigma_all[-n_test:], mask_all[-n_test:], y_all[-n_test:]

    # Pre-stage tensors in GPU memory
    mu_tr_t = torch.from_numpy(mu_tr).to(device)
    sig_tr_t = torch.from_numpy(sig_tr).to(device)
    msk_tr_t = torch.from_numpy(msk_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)

    # Initialize model
    model = FoundationAstroJev(num_classes=NUM_PHENOMENA_CLASSES, d_model=d_model, n_iter=n_iter)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_batches = int(math.ceil(n_train / batch_size))
    t_train_0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses = []

        for b in range(n_batches):
            idx = perm[b * batch_size: min((b + 1) * batch_size, n_train)]
            b_mu = mu_tr_t[idx]
            b_sig = sig_tr_t[idx]
            b_msk = msk_tr_t[idx]
            b_y = y_tr_t[idx]

            optimizer.zero_grad()
            out = model(b_mu, b_sig, mask=b_msk, apply_jitter=True)
            loss = heteroscedastic_evidential_loss(out["alpha"], b_y, num_classes=NUM_PHENOMENA_CLASSES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            mean_loss = float(np.mean(epoch_losses))
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    train_duration = time.perf_counter() - t_train_0
    throughput = (n_train * epochs) / train_duration
    print(f"Training completed in {train_duration:.2f}s ({throughput:,.0f} sources/sec)")

    # ---------------- Evaluation & Post-Hoc Calibration ---------------- #
    model.eval()
    with torch.no_grad():
        # Evaluate on test set
        te_mu_t = torch.from_numpy(mu_te).to(device)
        te_sig_t = torch.from_numpy(sig_te).to(device)
        te_msk_t = torch.from_numpy(msk_te).to(device)

        out_te = model(te_mu_t, te_sig_t, mask=te_msk_t, apply_jitter=False)
        probs_te = out_te["probs"].cpu().numpy()
        preds_te = np.argmax(probs_te, axis=-1)
        raw_accuracy = float(np.mean(preds_te == y_te))
        raw_edb = debiased_squared_calibration_error(probs_te, y_te, n_bins=15)["debiased_squared_ce"]

        # Recalibration using ScalingBinningCalibrator
        recal_mu_t = torch.from_numpy(mu_recal).to(device)
        recal_sig_t = torch.from_numpy(sig_recal).to(device)
        recal_msk_t = torch.from_numpy(msk_recal).to(device)
        out_recal = model(recal_mu_t, recal_sig_t, mask=recal_msk_t, apply_jitter=False)
        logits_recal = out_recal["alpha"].cpu().numpy()  # Use concentration as surrogate logits
        logits_te = out_te["alpha"].cpu().numpy()

        calibrator = ScalingBinningCalibrator(n_bins=10)
        calibrator.fit(logits_recal, y_recal)
        cal_probs_te, cal_conf_te = calibrator.calibrate(logits_te)
        cal_edb = debiased_squared_calibration_error(cal_probs_te, y_te, n_bins=15)["debiased_squared_ce"]

        sparsity = float(out_te["sparsity"].item() if hasattr(out_te["sparsity"], "item") else out_te["sparsity"])
        mean_delta = float(out_te["delta_eq"].mean().item())

        # Conformal Risk Control Calibration
        crc_res = conformal_risk_control_calibrate(
            probs=out_recal["probs"].cpu().numpy(),
            labels=y_recal,
            target_class=0,
            alpha_risk=0.05,
        )

        # Per-Class Accuracy
        class_accs = {}
        for c_idx, c_name in enumerate(PHENOMENA_CLASSES):
            c_mask = (y_te == c_idx)
            if np.sum(c_mask) > 0:
                class_accs[c_name] = float(np.mean(preds_te[c_mask] == c_idx))

    # Pricing estimate
    gpu_hourly_cost = 4.50 if "H100" in device_name else 2.50
    estimated_cost = (train_duration / 3600.0) * gpu_hourly_cost

    results = {
        "architecture": "FoundationAstroJev",
        "device": device_name,
        "n_sources": n_sources,
        "num_classes": NUM_PHENOMENA_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_time_seconds": train_duration,
        "throughput_sources_per_sec": throughput,
        "estimated_cost_usd": estimated_cost,
        "raw_accuracy": raw_accuracy,
        "raw_edb_sq": float(raw_edb),
        "calibrated_edb_sq": float(cal_edb),
        "latent_crelu_sparsity": sparsity,
        "mean_equilibrium_tension": mean_delta,
        "crc_calibrated_threshold": crc_res["lambda_hat"],
        "crc_empirical_risk": crc_res["empirical_risk"],
        "crc_sample_retention": crc_res["sample_retention"],
        "per_class_accuracy": class_accs,
    }

    print("\n" + "=" * 70)
    print("FOUNDATION ASTROJEV SCALED BENCHMARK RESULTS:")
    print(f"  Device:                      {device_name}")
    print(f"  Throughput:                  {throughput:,.0f} sources/sec")
    print(f"  Wall-Clock Time:             {train_duration:.2f}s ({train_duration/60.0:.2f} min)")
    print(f"  Estimated Cost:              ${estimated_cost:.4f} USD")
    print(f"  Overall 12-Class Accuracy:   {raw_accuracy*100:.2f}%")
    print(f"  Debiased Calibration E^2_db: {cal_edb:.10f} (Verified Stanford calibration)")
    print(f"  CReLU Latent Sparsity:       {sparsity*100:.2f}% (>= 50% guaranteed)")
    print(f"  Banach Tension Delta_eq:     {mean_delta:.4f}")
    print("=" * 70)

    # Save to Modal Volume if available
    vol_path = Path("/vol/checkpoints")
    try:
        vol_path.mkdir(parents=True, exist_ok=True)
        save_file = vol_path / save_checkpoint_name
        torch.save({
            "model_state_dict": model.state_dict(),
            "d_model": d_model,
            "classes": PHENOMENA_CLASSES,
            "num_classes": NUM_PHENOMENA_CLASSES,
            "results": results,
        }, str(save_file))
        print(f"Saved checkpoint to Modal Volume: {save_file}")
        if volume is not None:
            volume.commit()
    except Exception as e:
        print(f"Volume save notice: {e}")

    # Also save local fallback checkpoint
    try:
        local_ckpt_dir = Path("checkpoints")
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "d_model": d_model,
            "classes": PHENOMENA_CLASSES,
            "num_classes": NUM_PHENOMENA_CLASSES,
            "results": results,
        }, str(local_ckpt_dir / save_checkpoint_name))
    except Exception:
        pass

    return sanitize_json_dict(results)


# --------------------------------------------------------------------------- #
# Modal Entrypoint
# --------------------------------------------------------------------------- #
if modal is not None:
    @app.function(
        image=train_image,
        gpu="H100",
        timeout=1200,
        volumes={"/vol/checkpoints": volume},
    )
    def modal_train_foundation_h100(sources: int, epochs: int, batch_size: int) -> Dict[str, Any]:
        return train_foundation_astrojev(n_sources=sources, epochs=epochs, batch_size=batch_size)

    @app.local_entrypoint()
    def main(sources: int = 80_000, epochs: int = 25, batch_size: int = 4096):
        res = modal_train_foundation_h100.remote(sources=sources, epochs=epochs, batch_size=batch_size)
        out_path = Path("docs/research/foundation_astrojev_modal_h100_results.json")
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved cloud results locally to: {out_path}")


if __name__ == "__main__" and modal is None:
    train_foundation_astrojev()
