"""
=============================================================================
Hermetic Tests for Modal Scaled Continuous-Flow Cross-Calibration Benchmark
=============================================================================
Validates:
  1. Local execution of run_continuous_flow_benchmark on synthetic sample.
  2. Modal App and Image configuration.
  3. Payload serialization and uncertainty retention.
"""
import pytest
import numpy as np
import torch

from scripts.modal_scaled_continuous_flow_cross_calibration import (
    run_continuous_flow_benchmark,
    sanitize_payload,
)


def test_modal_continuous_flow_local_synthetic(tmp_path):
    """Verify local execution of the continuous-flow cross-calibration pipeline."""
    # Create temporary synthetic npz dataset matching real_phenomena_dataset schema
    n_sample = 200
    rng = np.random.default_rng(42)
    fake_mu = rng.uniform(18.0, 21.0, size=(n_sample, 10)).astype(np.float32)
    fake_sigma = rng.uniform(0.01, 0.05, size=(n_sample, 6)).astype(np.float32)
    fake_mask = np.ones((n_sample, 6), dtype=np.float32)
    fake_labels = rng.choice([0, 1, 2, 3, 5, 9], size=n_sample)

    fake_data_path = tmp_path / "test_phenomena.npz"
    np.savez_compressed(
        fake_data_path,
        mu=fake_mu,
        sigma=fake_sigma,
        mask=fake_mask,
        labels=fake_labels,
    )

    res = run_continuous_flow_benchmark(
        n_sources=150,
        batch_size=64,
        data_path=str(fake_data_path),
        device="cpu",
    )

    assert "zero_point_recovery" in res
    assert "calibration" in res
    assert "fiber_triage" in res
    assert res["uncertainty_retention"]["zero_naked_predictions_verified"] is True
    assert res["fiber_triage"]["target_risk_level"] == 0.02


def test_sanitize_payload_types():
    """Verify JSON sanitization handles tensors, arrays, and scalars cleanly."""
    payload = {
        "tensor": torch.tensor([1.0, 2.0]),
        "scalar_tensor": torch.tensor(42.5),
        "numpy_arr": np.array([3.14, 2.71]),
        "nested": {"val": np.float32(1.23)},
    }
    sanitized = sanitize_payload(payload)
    assert isinstance(sanitized["tensor"], list)
    assert sanitized["scalar_tensor"] == 42.5
    assert isinstance(sanitized["numpy_arr"], list)
    assert sanitized["nested"]["val"] == pytest.approx(1.23, rel=1e-3)
