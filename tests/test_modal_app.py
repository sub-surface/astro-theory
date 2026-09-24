"""Hermetic unit tests for Celestrium Modal Cloud Application."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from celestrium.modal_app import sanitize_payload, CelestriumDecisionEngine, simulate_pseudo_cl_realization


def test_sanitize_payload():
    """Verify that tensors, numpy arrays, and nested structures sanitize cleanly."""
    payload = {
        "tensor": torch.tensor([1.0, 2.0, 3.0]),
        "scalar_tensor": torch.tensor(42.5),
        "numpy_array": np.array([0.1, 0.2]),
        "numpy_scalar": np.float32(3.14),
        "nested": {
            "inner_list": [torch.tensor(1), np.int64(2), 3],
        }
    }
    cleaned = sanitize_payload(payload)
    assert cleaned["tensor"] == [1.0, 2.0, 3.0]
    assert cleaned["scalar_tensor"] == 42.5
    assert cleaned["numpy_array"] == [0.1, 0.2]
    assert abs(cleaned["numpy_scalar"] - 3.14) < 1e-4
    assert cleaned["nested"]["inner_list"] == [1, 2, 3]


def test_decision_engine_local_classify_and_triage():
    """Verify that CelestriumDecisionEngine runs inference and triage locally."""
    engine = CelestriumDecisionEngine()

    # 1. Test batch classification on 5 sources
    features = [
        [19.5, 0.7, -0.2, 15.0, 0.9, 0.2, 0.1, 120.0, 45.0, 30.0],
        [16.0, 1.2, 0.5, 14.5, 0.1, 15.0, 0.2, 30.0, 10.0, 80.0],
        [18.0, 1.6, -0.6, 16.0, 0.3, 0.3, 0.2, 180.0, 60.0, 40.0],
        [17.5, -0.1, 0.1, 17.0, 0.0, 30.0, 0.2, 90.0, -20.0, 50.0],
        [21.0, 0.5, -0.1, 16.5, 1.2, 0.1, 0.3, 240.0, -50.0, 15.0],
    ]
    res = engine.classify_batch.local(features)
    assert res["n_sources"] == 5
    assert len(res["predicted_classes"]) == 5
    assert len(res["calibrated_probabilities"]) == 5
    assert len(res["epistemic_vacuity"]) == 5
    assert len(res["bald_information_gain"]) == 5

    # 2. Test alert triage
    alert = {
        "source_id": "TEST_ALERT_001",
        "phot_g": 19.2,
        "bp_rp": 0.65,
        "w1_w2": 1.10,
        "pm": 0.1,
        "snr": 45.0,
    }
    triage = engine.triage_alert.local(alert)
    assert triage["source_id"] == "TEST_ALERT_001"
    assert triage["action"] in ["AUTO_CATALOG", "SCHEDULE_FOLLOWUP", "CONTRACTIVE_DELIBERATION", "REJECT"]
    assert "confidence" in triage
    assert "epistemic_vacuity" in triage
    assert "bald_gain_nats" in triage
    assert "recommendation" in triage
