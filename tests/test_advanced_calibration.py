"""Hermetic unit tests for Advanced Calibration, CARL, and TruthRL.

Tests:
1. Debiased Squared Calibration Error (Kumar, Liang, Ma NeurIPS 2019).
2. Scaling-Binning Calibrator (NeurIPS 2019).
3. Evidential Brier-CARL Loss (MIT ICLR 2026 RLCR + USC/AWS ACL 2026 CARL).
4. TruthRL Ternary Decision Policy (Meta TruthRL 2025).
5. CVaR Risk-Sensitive Dipole Weighting (Koren et al. 2025/2026, UAMDP).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from celestrium.astrojev import (
    debiased_squared_calibration_error,
    evaluate_calibration,
    ScalingBinningCalibrator,
)
from celestrium.evidential_astrojev import (
    evidential_brier_carl_loss,
    evidential_brier_loss,
    EvidentialAstroJev,
    NUM_CLASSES,
)
from celestrium.followup import evaluate_truthrl_policy
from celestrium.experiments.quaia_pseudo_cl import compute_risk_sensitive_weights


def test_debiased_squared_calibration_error_basic():
    """Verify that debiased calibration error computes cleanly and removes positive bias."""
    rng = np.random.default_rng(42)
    n = 1000
    # Synthetic 2-class predictions
    probs_col1 = rng.uniform(0.1, 0.9, n)
    probs = np.column_stack([probs_col1, 1.0 - probs_col1])
    # Generate labels with matching probability (calibrated)
    labels = (rng.random(n) > probs_col1).astype(int)

    res = debiased_squared_calibration_error(probs, labels, n_bins=10)
    assert "debiased_squared_ce" in res
    assert "rmsce_debiased" in res
    assert "rmsce_plugin" in res

    # For finite samples drawn from true distribution, debiased E^2 <= plugin E^2
    assert res["debiased_squared_ce"] <= res["rmsce_plugin"] ** 2 + 1e-6
    assert res["rmsce_debiased"] >= 0.0

    # Also test integration via evaluate_calibration
    eval_res = evaluate_calibration(probs, labels, n_bins=10)
    assert "debiased_squared_ce" in eval_res
    assert "rmsce_debiased" in eval_res


def test_scaling_binning_calibrator():
    """Verify that ScalingBinningCalibrator fits and transforms correctly."""
    rng = np.random.default_rng(123)
    n = 500
    # Overconfident uncalibrated logits
    true_labels = rng.integers(0, 4, n)
    logits = rng.normal(0, 1, (n, 4))
    # Make correct class logit large
    logits[np.arange(n), true_labels] += 3.5

    calibrator = ScalingBinningCalibrator(n_bins=8)
    calibrator.fit(logits, true_labels)

    assert calibrator.temperature > 0.0
    assert calibrator.bin_edges is not None
    assert len(calibrator.bin_values) > 0

    # Calibrate on test set
    test_logits = rng.normal(0, 1, (100, 4))
    cal_probs, cal_confs = calibrator.calibrate(test_logits)

    assert cal_probs.shape == (100, 4)
    assert len(cal_confs) == 100
    # Probabilities should sum to 1.0
    np.testing.assert_allclose(np.sum(cal_probs, axis=1), 1.0, atol=1e-5)
    # Calibrated confidences should match top-1 probability
    top_p = np.max(cal_probs, axis=1)
    np.testing.assert_allclose(top_p, cal_confs, atol=1e-5)


def test_evidential_brier_carl_loss():
    """Verify that CARL penalty pulls Dirichlet parameters toward uniform prior on mistakes."""
    torch.manual_seed(42)
    B = 20
    # Synthetic Dirichlet alphas
    # Case 1: All correct predictions
    labels = torch.randint(0, NUM_CLASSES, (B,))
    evidence = torch.zeros(B, NUM_CLASSES)
    evidence[torch.arange(B), labels] = 10.0  # highly confident and correct
    alpha_correct = evidence + 1.0

    loss_correct, probs_correct, metrics_correct = evidential_brier_carl_loss(
        alpha_correct, labels, carl_weight=0.5
    )
    assert metrics_correct["wrong_fraction"] == 0.0
    assert metrics_correct["carl_penalty"] == 0.0

    # Case 2: All wrong predictions with high confidence (overconfident blunders)
    wrong_labels = (labels + 1) % NUM_CLASSES
    evidence_wrong = torch.zeros(B, NUM_CLASSES)
    evidence_wrong[torch.arange(B), wrong_labels] = 10.0  # confident in WRONG class
    alpha_wrong = evidence_wrong + 1.0

    loss_wrong, probs_wrong, metrics_wrong = evidential_brier_carl_loss(
        alpha_wrong, labels, carl_weight=0.5
    )
    assert metrics_wrong["wrong_fraction"] == 1.0
    assert metrics_wrong["carl_penalty"] > 0.0
    # CARL loss must be strictly higher due to the barycenter penalty
    assert loss_wrong > loss_correct

    # Gradient check: gradients should flow to alpha
    alpha_param = nn.Parameter(alpha_wrong.clone())
    loss, _, _ = evidential_brier_carl_loss(alpha_param, labels, carl_weight=0.5)
    loss.backward()
    assert alpha_param.grad is not None
    assert torch.isfinite(alpha_param.grad).all()


def test_truthrl_policy_evaluation():
    """Verify TruthRL ternary reward scoring (correct vs abstain vs hallucination)."""
    triage_results = {
        "stats": {"total_triaged": 10},
        "auto_cataloged": [
            {"source_id": "SRC_1", "class": "Quasar_AGN"},
            {"source_id": "SRC_2", "class": "Quasar_AGN"},
            {"source_id": "SRC_3", "class": "Galactic_Star"},
        ],
        "followup_triggered": [
            {"source_id": "SRC_4"},
            {"source_id": "SRC_5"},
        ],
        "deliberations": [
            {"source_id": "SRC_6"},
        ],
    }

    ground_truth = {
        "SRC_1": "Quasar_AGN",     # Correct auto-catalog (+1)
        "SRC_2": "Galactic_Star",  # Hallucination (-2)
        "SRC_3": "Galactic_Star",  # Correct auto-catalog (+1)
        "SRC_4": "Quasar_AGN",     # Abstain (0)
        "SRC_5": "Galactic_Star",  # Abstain (0)
        "SRC_6": "Passive_Galaxy", # Abstain (0)
    }

    metrics = evaluate_truthrl_policy(
        triage_results=triage_results,
        ground_truth_labels=ground_truth,
        w_correct=1.0,
        w_abstain=0.0,
        w_hallucination=2.0,
    )

    assert metrics["n_correct"] == 2
    assert metrics["n_hallucination"] == 1
    assert metrics["n_abstain"] == 3
    # Expected reward: 2 * 1.0 + 3 * 0.0 - 1 * 2.0 = 0.0
    assert abs(metrics["truthfulness_score"] - 0.0) < 1e-6
    assert metrics["abstention_rate"] == 0.3


def test_cvar_risk_sensitive_weights():
    """Verify that CVaR risk-sensitive weighting suppresses high-vacuity and unstable sources."""
    p_qso = np.array([0.95, 0.95, 0.95, 0.95])
    u_epi = np.array([0.05, 0.80, 0.05, 0.90])      # source 1 & 3 low vacuity; 2 & 4 high vacuity
    delta_eq = np.array([0.02, 0.03, 0.25, 0.30])   # source 3 & 4 unstable equilibrium

    weights = compute_risk_sensitive_weights(
        p_quasar=p_qso,
        u_epi=u_epi,
        delta_eq=delta_eq,
        gamma_risk=2.0,
        delta_max=0.15,
    )

    # Source 0: clean & stable -> high weight
    assert weights[0] > 0.8
    # Source 1: high vacuity -> suppressed weight
    assert weights[1] < weights[0] * 0.1
    # Source 2: unstable equilibrium (delta_eq > 0.15) -> zeroed out
    assert weights[2] == 0.0
    # Source 3: both high vacuity and unstable -> zeroed out
    assert weights[3] == 0.0
