"""Hermetic unit tests for Astronomical Decision Models.

Covers:
1. Dirichlet BALD (Bayesian Active Learning by Disagreement) mutual information.
2. Conformal Risk Control (CRC) finite-sample false discovery rate bounds.
3. TelescopeQueueMDP sequential telescope follow-up scheduler.
"""
import numpy as np
import pytest

from celestrium.astrojev import (
    dirichlet_bald_information_gain,
    conformal_risk_control_calibrate,
)
from celestrium.followup import (
    TelescopeQueueMDP,
    triage_sources,
)


def test_dirichlet_bald_information_gain():
    """Verify that Dirichlet BALD mutual information peaks on high epistemic variance."""
    # Case 1: Confident prediction (high concentration in one class)
    # alpha = [100.0, 1.0, 1.0, 1.0] -> high certainty -> low BALD
    alpha_confident = np.array([[100.0, 1.0, 1.0, 1.0]])
    bald_confident = dirichlet_bald_information_gain(alpha_confident)[0]

    # Case 2: Ambiguous / high epistemic disagreement
    # alpha = [2.0, 2.0, 2.0, 2.0] -> low evidence, high variance -> high BALD
    alpha_ambiguous = np.array([[2.0, 2.0, 2.0, 2.0]])
    bald_ambiguous = dirichlet_bald_information_gain(alpha_ambiguous)[0]

    # Case 3: Infinite evidence uniform
    # alpha = [1000.0, 1000.0, 1000.0, 1000.0] -> aleatoric coin flip, zero epistemic doubt -> low BALD
    alpha_aleatoric = np.array([[1000.0, 1000.0, 1000.0, 1000.0]])
    bald_aleatoric = dirichlet_bald_information_gain(alpha_aleatoric)[0]

    assert bald_confident >= 0.0
    assert bald_ambiguous > bald_confident
    assert bald_ambiguous > bald_aleatoric


def test_conformal_risk_control_calibrate():
    """Verify that Conformal Risk Control finds valid thresholds bounding target risk."""
    rng = np.random.default_rng(42)
    n = 1000
    # True labels: 0 = Quasar, 1 = Star
    labels = rng.binomial(1, 0.4, n)

    # Probabilities for class 0 with some noise
    probs_qso = np.where(labels == 0, rng.beta(5, 2, n), rng.beta(1, 4, n))
    probs = np.column_stack([probs_qso, 1.0 - probs_qso])

    # Calibrate risk control at alpha_risk = 0.10 (at most 10% false discoveries)
    res = conformal_risk_control_calibrate(probs, labels, alpha_risk=0.10, target_class=0)

    assert "lambda_hat" in res
    assert 0.0 <= res["lambda_hat"] <= 1.0
    assert res["empirical_risk"] <= 0.10 + 1e-4
    assert res["sample_retention"] > 0.0


def test_telescope_queue_mdp_scheduler():
    """Verify that TelescopeQueueMDP schedules within budget and orders by efficiency."""
    scheduler = TelescopeQueueMDP(site_lat_deg=-24.627, time_budget_min=60.0)

    candidates = [
        {
            "source_id": "CAND_1",
            "ra": 10.0,
            "dec": -20.0,
            "phot_g": 18.5,
            "bald_info_gain": 0.85,
            "u_epi": 0.70,
            "archive": "MAST",
        },
        {
            "source_id": "CAND_2",
            "ra": 12.0,
            "dec": -22.0,
            "phot_g": 19.0,
            "bald_info_gain": 0.75,
            "u_epi": 0.60,
            "archive": "HEASARC",
        },
        {
            "source_id": "CAND_3",
            "ra": 180.0,
            "dec": 45.0,  # far slew, faint
            "phot_g": 22.0,
            "bald_info_gain": 0.20,
            "u_epi": 0.30,
            "archive": "OPTICAL",
        },
    ]

    res = scheduler.schedule(candidates, initial_pointing=(10.0, -20.0))

    assert "schedule" in res
    assert res["total_time_min"] <= 60.0
    assert res["targets_scheduled"] >= 1
    assert res["total_bald_gain"] > 0.0

    # CAND_1 should be prioritized first due to high BALD and close pointing
    first_target = res["schedule"][0]
    assert first_target["source_id"] == "CAND_1"


def test_triage_sources_with_bald():
    """Verify that triage_sources computes and populates bald_info_gain."""
    rng = np.random.default_rng(99)
    # Generate 10 synthetic candidates
    features = np.column_stack([
        rng.normal(20.0, 0.5, 10),
        rng.normal(0.8, 0.2, 10),
        np.zeros(10),
        rng.normal(15.0, 0.5, 10),
        rng.normal(0.4, 0.2, 10),  # ambiguous color
        rng.normal(1.5, 0.5, 10),
        rng.uniform(1.0, 2.0, 10), # high PM error
        rng.uniform(0, 1, 10),
        rng.uniform(0, 1, 10),
        np.full(10, 20.0),
    ])

    report = triage_sources(features, u_epi_threshold=0.3)
    assert "stats" in report
    followups = report["followup_triggered"]
    if followups:
        assert "bald_info_gain" in followups[0]
        assert followups[0]["bald_info_gain"] >= 0.0
