"""Tests for Real Astronomical Data Streaming and RLCD Decision Calibration."""
from pathlib import Path
import math
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from celestrium.data_streamer import (
    RealAstroDataStreamer,
    RealAstroStreamingDataset,
    RealMultiMessengerStreamer,
    RealMultiMessengerStreamingDataset,
    PhenomenaRecord,
)
from celestrium.multimessenger import (
    MultiMessengerEvidentialNet,
    MultiMessengerTriageEngine,
    evaluate_multimessenger_calibration,
    fine_tune_rlcd_doubt_head,
    NUM_MM_FEATURES,
    NUM_MM_CLASSES,
)


def test_real_astro_data_streamer(tmp_path):
    # The real catalogs under data/ are gitignored, so use a small npz fixture
    # to keep this test hermetic on a fresh clone.
    rng = np.random.default_rng(0)
    n = 50
    npz_path = tmp_path / "phenomena.npz"
    np.savez(
        npz_path,
        mu=rng.normal(size=(n, 10)).astype(np.float32),
        sigma=rng.uniform(0.01, 0.3, (n, 6)).astype(np.float32),
        mask=np.ones((n, 6), np.float32),
        labels=rng.integers(0, 12, n),
    )
    streamer = RealAstroDataStreamer(
        npz_dataset_path=npz_path,
        gaia_cache_path=None,
        quaia_fits_path=None,
        chunk_size=1000,
        sample_noise=True,
        seed=42,
    )
    assert streamer.active_backends == ["npz_dataset"]
    records = list(streamer.stream_records(limit=10))
    assert len(records) == 10

    for rec in records:
        assert isinstance(rec, PhenomenaRecord)
        assert rec.mu.shape == (10,)
        assert rec.sigma.shape == (6,)
        assert rec.mask.shape == (6,)
        assert 0 <= rec.label < 12
        assert np.all(np.isfinite(rec.mu))
        assert np.all(np.isfinite(rec.sigma))
        assert np.all(rec.sigma > 0.0)


def test_real_astro_streaming_batches():
    streamer = RealAstroDataStreamer(chunk_size=500, sample_noise=False, seed=123)
    batches = list(streamer.stream_batches(batch_size=32, limit_samples=100))
    assert len(batches) >= 3

    total_samples = 0
    for mu_b, sig_b, mask_b, y_b in batches:
        assert isinstance(mu_b, torch.Tensor)
        assert mu_b.dtype == torch.float32
        assert y_b.dtype == torch.int64
        assert mu_b.shape[1] == 10
        assert sig_b.shape[1] == 6
        assert mask_b.shape[1] == 6
        total_samples += len(y_b)

    assert total_samples == 100


def test_real_multi_messenger_streamer():
    streamer = RealMultiMessengerStreamer(use_live_network=False, seed=99)
    scenarios = list(streamer.stream_scenarios(n_scenarios=3, candidates_per_scenario=10, inject_kilonova_rate=1.0))
    assert len(scenarios) == 3

    for gw_alert, nu_alert, candidates in scenarios:
        assert gw_alert.superevent_id != ""
        assert gw_alert.distance_mean_mpc > 0.0
        assert nu_alert is not None
        assert len(candidates) == 10

        # Check injected kilonova presence
        kn_cand = next((c for c in candidates if c.true_class == "Kilonova"), None)
        assert kn_cand is not None
        assert kn_cand.rate_color_gr > 0.30  # Kasen rapid reddening signature
        assert kn_cand.host_galaxy is not None


def test_real_multi_messenger_pytorch_dataset():
    streamer = RealMultiMessengerStreamer(use_live_network=False, seed=42)
    ds = RealMultiMessengerStreamingDataset(
        streamer=streamer,
        n_scenarios=2,
        candidates_per_scenario=20,
        batch_size=16,
    )
    loader = DataLoader(ds, batch_size=None)
    n_batches = 0
    total_samples = 0

    for x, y in loader:
        assert x.shape[1] == NUM_MM_FEATURES
        assert y.dtype == torch.int64
        n_batches += 1
        total_samples += len(y)

    assert n_batches > 0
    assert total_samples == 40


def test_evaluate_multimessenger_calibration():
    rng = np.random.default_rng(42)
    n = 500
    labels = rng.integers(0, NUM_MM_CLASSES, n)

    # 1. Perfectly calibrated probabilities
    probs_calib = np.full((n, NUM_MM_CLASSES), 0.1666, dtype=np.float32)
    for i in range(n):
        true_c = labels[i]
        probs_calib[i, true_c] = 0.70
        other_mass = 0.30 / (NUM_MM_CLASSES - 1)
        for k in range(NUM_MM_CLASSES):
            if k != true_c:
                probs_calib[i, k] = other_mass

    res_calib = evaluate_multimessenger_calibration(probs_calib, labels, n_bins=10, n_bootstrap=50)
    assert "ece" in res_calib
    assert "debiased_squared_ce" in res_calib
    assert "ece_ci_95" in res_calib
    assert "mean_doubt_reward" in res_calib
    assert res_calib["accuracy"] == 1.0
    assert res_calib["ece"] < 0.35
    assert res_calib["mean_doubt_reward"] > -0.50

    # 2. Overconfident miscalibrated probabilities
    probs_over = np.full((n, NUM_MM_CLASSES), 0.001, dtype=np.float32)
    for i in range(n):
        # Assign 0.995 to random class (often wrong)
        guess_c = (labels[i] + 1) % NUM_MM_CLASSES if i % 2 == 0 else labels[i]
        probs_over[i, guess_c] = 0.995

    res_over = evaluate_multimessenger_calibration(probs_over, labels, n_bins=10, n_bootstrap=50)
    assert res_over["mean_doubt_reward"] < res_calib["mean_doubt_reward"]
    assert res_over["ece"] > res_calib["ece"]


def test_fine_tune_rlcd_doubt_head():
    torch.manual_seed(42)
    model = MultiMessengerEvidentialNet(in_features=NUM_MM_FEATURES, d_model=32, num_classes=NUM_MM_CLASSES, n_iter=2)

    # Save initial trunk parameters to verify freezing (Disentangled Optimization)
    enc_weight_init = model.input_proj[0].weight.clone()
    contr_weight_init = model.recurrent_cell[1].weight.clone()

    train_x = torch.randn(128, NUM_MM_FEATURES)
    train_y = torch.randint(0, NUM_MM_CLASSES, (128,))
    val_x = torch.randn(64, NUM_MM_FEATURES)
    val_y = torch.randint(0, NUM_MM_CLASSES, (64,))

    res = fine_tune_rlcd_doubt_head(
        model=model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=3,
        batch_size=32,
        lr=1e-3,
    )

    assert "calib_before" in res
    assert "calib_after" in res

    # Verify that the representation trunk was frozen during doubt fine-tuning
    assert torch.equal(model.input_proj[0].weight.cpu(), enc_weight_init.cpu())
    assert torch.equal(model.recurrent_cell[1].weight.cpu(), contr_weight_init.cpu())


def test_triage_decision_retains_explicit_error_bars():
    streamer = RealMultiMessengerStreamer(use_live_network=False, seed=77)
    gw, nu, cands = next(streamer.stream_scenarios(n_scenarios=1, candidates_per_scenario=5))

    engine = MultiMessengerTriageEngine()
    results = engine.triage_candidates(cands, gw, nu)
    assert len(results) == 5

    for r in results:
        # Check all decision error bar attributes
        assert hasattr(r, "confidence_err")
        assert hasattr(r, "confidence_interval_95")
        assert hasattr(r, "prediction_set")
        assert hasattr(r, "prediction_set_risk_bound")
        assert hasattr(r, "doubt_reward")

        assert r.confidence_err > 0.0
        assert r.confidence_interval_95[0] <= r.confidence <= r.confidence_interval_95[1]
        assert 0.0 <= r.confidence_interval_95[0] <= 1.0
        assert 0.0 <= r.confidence_interval_95[1] <= 1.0
        assert len(r.prediction_set) >= 1
        assert np.isfinite(r.doubt_reward)
        assert r.prediction_set_risk_bound == 0.05
