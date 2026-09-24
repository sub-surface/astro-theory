"""Hermetic tests for Modal Cloud Multi-Messenger Stress Test & Calibration Engine."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.modal_stress_test_multimessenger_rlcd import (
    generate_stress_cohort,
    run_broker_stress_test,
    sanitize_payload,
)


def test_generate_stress_cohort():
    """Verify physical properties, coordinates, and extinction of generated stress cohort."""
    features, labels, galactic_b, extinction_av, is_bright_moon = generate_stress_cohort(
        n_sources=200, seed=42
    )

    assert features.shape == (200, 12)
    assert labels.shape == (200,)
    assert galactic_b.shape == (200,)
    assert extinction_av.shape == (200,)
    assert is_bright_moon.shape == (200,)

    # Galactic latitudes in [-90, +90]
    assert np.all(galactic_b >= -90.0)
    assert np.all(galactic_b <= 90.0)

    # Extinction A_V in [0.01, 5.0]
    assert np.all(extinction_av >= 0.01)
    assert np.all(extinction_av <= 5.0)

    # Labels within valid class indices
    assert np.all(labels >= 0)
    assert np.all(labels < 6)


def test_sanitize_payload():
    """Verify that tensors, numpy arrays, and nested structures sanitize cleanly."""
    payload = {
        "tensor": torch.tensor([1.0, 2.0]),
        "numpy_val": np.float64(0.0123),
        "nested": {"array": np.array([1, 2, 3])},
    }
    cleaned = sanitize_payload(payload)
    assert cleaned["tensor"] == [1.0, 2.0]
    assert isinstance(cleaned["numpy_val"], float)
    assert cleaned["nested"]["array"] == [1, 2, 3]


def test_run_broker_stress_test_local():
    """Verify local execution of the stress test and error bar retention."""
    res = run_broker_stress_test(n_sources=150, batch_size=64, device="cpu")

    assert res["experiment_id"] == "EXP-2026-R_MODAL_STRESS_TEST"
    assert res["total_sources"] == 150
    assert "overall_calibration" in res
    assert "too_aperture_protection" in res
    assert "uncertainty_retention_audit" in res

    # Verify zero naked predictions
    audit = res["uncertainty_retention_audit"]
    assert audit["zero_naked_predictions_verified"] is True
    assert audit["mean_posterior_sigma_kn"] > 0.0
    assert audit["mean_epistemic_vacuity"] > 0.0

    # Verify ToO protection metrics
    too = res["too_aperture_protection"]
    assert too["gemini_false_alarm_rate_pct"] <= 5.0  # Conformal FDR bound
    assert too["lcogt_screening_routed"] >= 0
