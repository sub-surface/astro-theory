"""Hermetic unit tests for EXP-2026-S: Cosmic Dawn & Lensed Quasar Evidential Disentanglement."""
from __future__ import annotations

import pytest
import numpy as np
import torch

from celestrium.cosmic_dawn import (
    RealCosmicDawnStreamer,
    CosmicDawnEvidentialNet,
    CosmicDawnTriageEngine,
    evaluate_cosmic_dawn_calibration,
    fine_tune_cosmic_dawn_rlcd,
    DAWN_CLASSES,
    NUM_DAWN_CLASSES,
    NUM_DAWN_FEATURES,
)


def test_real_cosmic_dawn_streamer():
    """Verify data streaming and real survey feature mapping."""
    streamer = RealCosmicDawnStreamer(seed=42)
    features, labels = streamer.stream_batch(batch_size=300)

    assert features.shape == (300, NUM_DAWN_FEATURES)
    assert labels.shape == (300,)
    assert set(np.unique(labels)) == set(range(NUM_DAWN_CLASSES))

    # Lyman break color is positive, SNR is positive
    assert np.all(features[:, 11] > 0.0)  # snr_det


def test_cosmic_dawn_evidential_net_forward():
    """Verify Dirichlet parameter bounds and probability normalization."""
    model = CosmicDawnEvidentialNet(
        in_features=NUM_DAWN_FEATURES,
        d_model=64,
        num_classes=NUM_DAWN_CLASSES,
        n_iter=3,
    )
    x = torch.randn(16, NUM_DAWN_FEATURES)
    out = model(x)

    assert "alpha" in out
    assert "probs" in out
    assert "u_epi" in out
    assert torch.all(out["alpha"] >= 1.0)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(16), atol=1e-5)


def test_fine_tune_cosmic_dawn_rlcd_freezing():
    """Verify representation trunk weights remain frozen during RLCD fine-tuning."""
    streamer = RealCosmicDawnStreamer(seed=101)
    f_tr, l_tr = streamer.stream_batch(batch_size=150)
    f_val, l_val = streamer.stream_batch(batch_size=50)

    model = CosmicDawnEvidentialNet(
        in_features=NUM_DAWN_FEATURES,
        d_model=64,
        num_classes=NUM_DAWN_CLASSES,
    )
    orig_proj_weight = model.input_proj[0].weight.clone()

    cal = fine_tune_cosmic_dawn_rlcd(
        model=model,
        train_x=torch.from_numpy(f_tr),
        train_y=torch.from_numpy(l_tr),
        val_x=torch.from_numpy(f_val),
        val_y=torch.from_numpy(l_val),
        epochs=2,
        batch_size=32,
        device="cpu",
    )

    # Assert representation trunk weights are untouched
    assert torch.equal(model.input_proj[0].weight, orig_proj_weight)
    assert "debiased_squared_ce" in cal


def test_cosmic_dawn_triage_gating_and_brown_dwarf_elimination():
    """Verify that brown dwarfs are purged and error bars are retained on all decisions."""
    model = CosmicDawnEvidentialNet(
        in_features=NUM_DAWN_FEATURES,
        d_model=64,
        num_classes=NUM_DAWN_CLASSES,
    )
    engine = CosmicDawnTriageEngine(model=model, device="cpu")

    streamer = RealCosmicDawnStreamer(seed=999)
    features, _ = streamer.stream_batch(batch_size=15)

    # Candidate 0: Force to be a Galactic Brown Dwarf (point-source r_hl = 0.03'', PM SNR = 4.5)
    features[0, 8] = 0.03
    features[0, 10] = 4.5

    # Candidate 1: Force to be a Quadruply Lensed Quasar (Asymmetry = 0.75, Euclid detection = 25.0)
    features[1, 5] = 25.0
    features[1, 9] = 0.75

    results = engine.triage_candidates(features)
    assert len(results) == 15

    # Candidate 0 must be purged from NIRSpec queue
    assert results[0].action == "REJECT_GALACTIC_CONTAMINANT"

    # All decisions must retain analytical error bars
    for r in results:
        assert r.confidence_err > 0.0
        assert r.confidence_interval_95[0] <= r.confidence <= r.confidence_interval_95[1]
        assert r.epistemic_vacuity > 0.0
        assert len(r.prediction_set) >= 1
        assert "p_cosmic_dawn_err" in r.telemetry
