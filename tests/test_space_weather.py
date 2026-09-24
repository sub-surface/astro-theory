"""Hermetic tests for EXP-2026-T: Operational Space Weather & Short-Arc NEO Impact Triage."""
from __future__ import annotations

import pytest
import numpy as np
import torch

from celestrium.space_weather import (
    simulate_space_weather_dataset,
    SpaceWeatherEvidentialNet,
    SpaceWeatherTriageEngine,
    compute_skill_scores,
    SPACE_WEATHER_CLASSES,
    NUM_SW_CLASSES,
    NUM_SW_FEATURES,
)


def test_simulate_space_weather_dataset():
    """Verify simulation of active regions and asteroid impact candidates."""
    features, labels = simulate_space_weather_dataset(n_samples=300, seed=123)

    assert features.shape == (300, NUM_SW_FEATURES)
    assert labels.shape == (300,)
    assert set(np.unique(labels)) == set(range(NUM_SW_CLASSES))


def test_space_weather_evidential_net_forward():
    """Verify evidential Dirichlet output constraints."""
    model = SpaceWeatherEvidentialNet(
        in_features=NUM_SW_FEATURES,
        d_model=64,
        num_classes=NUM_SW_CLASSES,
        n_iter=3,
    )
    x = torch.randn(12, NUM_SW_FEATURES)
    out = model(x)

    assert "alpha" in out
    assert "probs" in out
    assert "u_epi" in out
    assert torch.all(out["alpha"] >= 1.0)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(12), atol=1e-5)


def test_compute_skill_scores():
    """Verify True Skill Statistic (TSS) and Heidke Skill Score (HSS) computation."""
    # Perfect predictions
    preds = np.array([2, 2, 0, 0, 1])
    labels = np.array([2, 2, 0, 0, 1])
    scores = compute_skill_scores(preds, labels, target_class=2)

    assert scores["tpr"] == 1.0
    assert scores["fpr"] == 0.0
    assert scores["tss"] == 1.0
    assert scores["hss"] == 1.0


def test_space_weather_triage_engine():
    """Verify operational decision gating, error bars, and radar recovery dispatch."""
    model = SpaceWeatherEvidentialNet(
        in_features=NUM_SW_FEATURES,
        d_model=64,
        num_classes=NUM_SW_CLASSES,
    )
    engine = SpaceWeatherTriageEngine(model=model, device="cpu")

    features, _ = simulate_space_weather_dataset(n_samples=15, seed=42)
    # Inject hazardous short-arc asteroid into candidate 0: P_impact = 0.25, Arc = 1.5h
    features[0, 8] = 0.25
    features[0, 9] = 1.5

    results = engine.triage_events(features)
    assert len(results) == 15

    # Candidate 0 must trigger planetary defense radar
    assert results[0].action == "DISPATCH_PLANETARY_DEFENSE_RADAR"

    # All decisions must retain analytical error bars
    for r in results:
        assert r.confidence_err > 0.0
        assert r.confidence_interval_95[0] <= r.confidence <= r.confidence_interval_95[1]
        assert r.epistemic_vacuity > 0.0
