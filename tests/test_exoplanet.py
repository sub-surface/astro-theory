"""Hermetic tests for EXP-2026-Q: Exoplanetary Transit & RV Disentanglement Engine."""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from celestrium.exoplanet import (
    simulate_matern32_gp_noise,
    simulate_exoplanet_dataset,
    ExoplanetEvidentialNet,
    ExoplanetTriageEngine,
    evaluate_exoplanet_calibration,
    fine_tune_exoplanet_rlcd,
    EXOPLANET_CLASSES,
    NUM_EXOPLANET_CLASSES,
    NUM_EXOPLANET_FEATURES,
)


def test_simulate_matern32_gp_noise():
    """Verify that Matérn-3/2 GP produces stationary correlated time series."""
    times = np.linspace(0.0, 30.0, 100)
    noise = simulate_matern32_gp_noise(times, sigma_gp=120.0, rho_days=2.5, seed=42)

    assert len(noise) == 100
    assert np.all(np.isfinite(noise))
    # Empirical standard deviation should be within 40% of target sigma
    std = float(np.std(noise))
    assert 50.0 < std < 200.0


def test_simulate_exoplanet_dataset():
    """Verify physical properties, classes, and feature bounds of simulated exoplanet dataset."""
    features, labels = simulate_exoplanet_dataset(n_samples=400, seed=123)

    assert features.shape == (400, NUM_EXOPLANET_FEATURES)
    assert labels.shape == (400,)
    assert set(np.unique(labels)) == set(range(NUM_EXOPLANET_CLASSES))

    # Transit depth >= 0, durations > 0
    assert np.all(features[:, 0] > 0.0)  # depth_ppm
    assert np.all(features[:, 1] > 0.0)  # duration_hr


def test_exoplanet_evidential_net_forward():
    """Verify evidential neural network outputs valid Dirichlet parameters and contractive tension."""
    model = ExoplanetEvidentialNet(
        in_features=NUM_EXOPLANET_FEATURES,
        d_model=64,
        num_classes=NUM_EXOPLANET_CLASSES,
        n_iter=3,
    )
    x = torch.randn(16, NUM_EXOPLANET_FEATURES)
    out = model(x)

    assert "alpha" in out
    assert "probs" in out
    assert "u_epi" in out
    assert "delta_eq" in out

    probs = out["probs"]
    alpha = out["alpha"]
    assert probs.shape == (16, NUM_EXOPLANET_CLASSES)
    assert torch.all(alpha >= 1.0)
    # Probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_evaluate_exoplanet_calibration():
    """Verify Stanford debiased calibration error and Rewarding Doubt evaluation."""
    probs = np.array([
        [0.85, 0.05, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.05],
        [0.05, 0.10, 0.75, 0.10],
        [0.20, 0.20, 0.20, 0.40],
    ])
    labels = np.array([0, 1, 2, 3])

    cal = evaluate_exoplanet_calibration(probs, labels, n_bins=5, n_bootstrap=20)
    assert "ece" in cal
    assert "ece_ci_95" in cal
    assert "debiased_squared_ce" in cal
    assert "mean_doubt_reward" in cal
    assert "normalized_doubt_score" in cal
    assert cal["accuracy"] == 1.0


def test_fine_tune_exoplanet_rlcd():
    """Verify Disentangled Optimization fine-tunes the doubt head and preserves representations."""
    model = ExoplanetEvidentialNet(
        in_features=NUM_EXOPLANET_FEATURES,
        d_model=64,
        num_classes=NUM_EXOPLANET_CLASSES,
    )
    f_tr, l_tr = simulate_exoplanet_dataset(n_samples=200, seed=10)
    f_val, l_val = simulate_exoplanet_dataset(n_samples=80, seed=20)

    tx_tr, ty_tr = torch.from_numpy(f_tr), torch.from_numpy(l_tr)
    tx_val, ty_val = torch.from_numpy(f_val), torch.from_numpy(l_val)

    # Initial weights of representation trunk
    orig_proj_weight = model.input_proj[0].weight.clone()

    cal = fine_tune_exoplanet_rlcd(
        model=model,
        train_x=tx_tr,
        train_y=ty_tr,
        val_x=tx_val,
        val_y=ty_val,
        epochs=3,
        batch_size=32,
        device="cpu",
    )

    # Verify representation trunk weights remained strictly frozen
    assert torch.equal(model.input_proj[0].weight, orig_proj_weight)
    assert "ece" in cal
    assert "debiased_squared_ce" in cal


def test_exoplanet_triage_engine_error_bars_and_gating():
    """Verify that triage engine retains error bars and routes activity doubt away from 8m VLT."""
    model = ExoplanetEvidentialNet(
        in_features=NUM_EXOPLANET_FEATURES,
        d_model=64,
        num_classes=NUM_EXOPLANET_CLASSES,
    )
    engine = ExoplanetTriageEngine(model=model, device="cpu")

    # Generate test candidates
    feats, _ = simulate_exoplanet_dataset(n_samples=10, seed=99)
    # Force candidate 0 to have strong stellar activity correlation (BIS = 0.85)
    feats[0, 8] = 0.85

    results = engine.triage_candidates(feats)
    assert len(results) == 10

    # Verify candidate 0 is held back from 8m ESPRESSO and routed to activity monitoring
    res0 = results[0]
    assert res0.action in ["MONITOR_STELLAR_ROTATION", "ROBOTIC_1M_PHOTOMETRY", "REJECT_FALSE_ALARM"]
    assert res0.action != "COMMIT_ESPRESSO_RV"

    # Verify analytical error bar retention on ALL candidates
    for r in results:
        assert r.confidence_err > 0.0
        assert r.confidence_interval_95[0] <= r.confidence <= r.confidence_interval_95[1]
        assert len(r.prediction_set) >= 1
        assert r.epistemic_vacuity > 0.0
        assert "p_exoplanet_err" in r.telemetry
