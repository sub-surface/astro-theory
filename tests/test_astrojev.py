"""Hermetic unit tests for AstroJev architecture and RLCD scoring rules."""
import math
import numpy as np
import pytest
import torch

from celestrium.astrojev import (
    AstroJev,
    ContinuousFourierEncoder,
    KMContractiveBlock,
    brier_score_reward,
    normalized_log_score,
    rewarding_doubt_score,
    rlcd_astronomy_reward,
    evaluate_calibration,
    NUM_FEATURES,
    NUM_CLASSES,
)


def test_fourier_encoder_dimensions():
    encoder = ContinuousFourierEncoder(in_features=NUM_FEATURES, d_model=64)
    x = torch.randn(8, NUM_FEATURES)
    h = encoder(x)
    assert h.shape == (8, 64)
    assert not torch.isnan(h).any()


def test_krasnoselskii_mann_block():
    block = KMContractiveBlock(d_model=64)
    h = torch.randn(4, 64)
    x_ctx = torch.randn(4, 64)
    out = block(h, x_ctx)
    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()


def test_astrojev_crelu_sparsity():
    model = AstroJev(in_features=NUM_FEATURES, d_model=64, num_classes=NUM_CLASSES)
    x = torch.randn(16, NUM_FEATURES)
    out = model(x)

    # CReLU sparse accumulator must have >= 50% guaranteed sparsity
    assert out["sparsity"] >= 0.40  # In random init, usually ~50-60%
    assert out["choice_probs"].shape == (16, NUM_CLASSES)
    assert torch.allclose(out["choice_probs"].sum(dim=-1), torch.ones(16), atol=1e-5)
    assert out["noul_prob"].shape == (16,)
    assert (out["noul_prob"] >= 0.0).all() and (out["noul_prob"] <= 1.0).all()
    assert out["delta_eq"].shape == (16,)


def test_brier_score_reward_bounded():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    targets = torch.tensor([0, 0, 0])
    r = brier_score_reward(probs, targets)

    # Perfect prediction p=[1,0], y=0 -> R = 1.0
    assert math.isclose(r[0].item(), 1.0, abs_tol=1e-5)
    # Worst prediction p=[0,1], y=0 -> R = 0.0
    assert math.isclose(r[1].item(), 0.0, abs_tol=1e-5)
    # Uniform guess p=[0.5, 0.5], y=0 -> R = 0.75
    assert math.isclose(r[2].item(), 0.75, abs_tol=1e-5)


def test_normalized_log_score_properties():
    # 4 classes: log(4)
    probs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    targets = torch.tensor([0, 0])
    r = normalized_log_score(probs, targets)

    # Perfect guess gives 1.0
    assert math.isclose(r[0].item(), 1.0, abs_tol=1e-5)
    # Uniform guess gives 0.0
    assert math.isclose(r[1].item(), 0.0, abs_tol=1e-5)


def test_rewarding_doubt_score():
    # Correct with high confidence
    p_high = torch.tensor([[0.95, 0.05]])
    # Correct with low confidence
    p_low = torch.tensor([[0.55, 0.45]])
    # Incorrect with high confidence
    p_wrong = torch.tensor([[0.05, 0.95]])
    targets = torch.tensor([0])

    r_high = rewarding_doubt_score(p_high, targets).item()
    r_low = rewarding_doubt_score(p_low, targets).item()
    r_wrong = rewarding_doubt_score(p_wrong, targets).item()

    assert r_high > r_low > r_wrong


def test_calibration_metrics_evaluator():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
    ])
    labels = np.array([0, 0, 0, 0])
    metrics = evaluate_calibration(probs, labels, n_bins=2)

    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_tsallis_score_reward_properties():
    from celestrium.astrojev import tsallis_score_reward, normalized_log_score, brier_score_reward

    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    targets = torch.tensor([0, 0, 0])

    # Interpolation test: alpha=1.0 should equal normalized log score
    r_tsallis_1 = tsallis_score_reward(probs, targets, alpha=1.0)
    r_log = normalized_log_score(probs, targets)
    assert torch.allclose(r_tsallis_1, r_log, atol=1e-4)

    # alpha=2.0 should equal Brier score reward
    r_tsallis_2 = tsallis_score_reward(probs, targets, alpha=2.0)
    r_brier = brier_score_reward(probs, targets)
    assert torch.allclose(r_tsallis_2, r_brier, atol=1e-4)

    # alpha=1.5 intermediate scoring
    r_tsallis_15 = tsallis_score_reward(probs, targets, alpha=1.5)
    assert r_tsallis_15[0] > r_tsallis_15[2] > r_tsallis_15[1]


def test_focal_brier_score_reward():
    from celestrium.astrojev import focal_brier_score_reward

    # Highly confident correct prediction
    p_easy = torch.tensor([[0.99, 0.01]])
    # Hard boundary correct prediction
    p_hard = torch.tensor([[0.51, 0.49]])
    # Confident wrong prediction
    p_wrong = torch.tensor([[0.01, 0.99]])
    targets = torch.tensor([0])

    r_easy = focal_brier_score_reward(p_easy, targets, gamma=2.0).item()
    r_hard = focal_brier_score_reward(p_hard, targets, gamma=2.0).item()
    r_wrong = focal_brier_score_reward(p_wrong, targets, gamma=2.0).item()

    assert r_easy > r_hard > r_wrong
    # Easy sample has (1-p)^gamma ~ (0.01)^2 = 0.0001 penalty, so reward is extremely close to 1.0
    assert math.isclose(r_easy, 1.0, abs_tol=1e-3)


def test_conformal_evidential_calibration():
    from celestrium.astrojev import conformal_evidential_calibrate, conformal_predict_sets

    rng = np.random.default_rng(42)
    n = 200
    probs = rng.dirichlet(alpha=[5.0, 1.0, 1.0, 1.0], size=n)
    labels = np.zeros(n, dtype=int)
    # Give some errors
    labels[180:] = 1
    u_epi = rng.uniform(0.05, 0.25, size=n)

    res = conformal_evidential_calibrate(probs, u_epi, labels, alpha_error=0.10, lambda_epi=0.5)

    assert "q_hat" in res
    assert res["target_coverage"] == 0.90
    # Empirical coverage should be close to or exceed target coverage
    assert res["empirical_coverage"] >= 0.85
    assert 1.0 <= res["mean_set_size"] <= 4.0

