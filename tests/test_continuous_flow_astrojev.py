"""
=============================================================================
Tests for EXP-2026-V: Continuous-Flow Foundation AstroJev
=============================================================================
Hermetic verification of:
  1. Multi-survey photometric generation across Euclid, DESI, and Rubin.
  2. Conditional Flow Matching velocity network & sinusoidal time embeddings.
  3. Continuous flow ODE integration and Dirichlet concentration computation.
  4. Disentangled RLCD calibration optimization.
  5. Spectroscopic fiber triage engine, error-bar retention, and conformal gate.
"""
import numpy as np
import pytest
import torch

from celestrium.continuous_flow_astrojev import (
    ALLOCATION_CLASSES,
    NUM_ALLOCATION_CLASSES,
    TOTAL_OBSERVABLE_DIM,
    LATENT_SED_DIM,
    MultiSurveyPhotometryGenerator,
    ConditionalFlowVelocityNet,
    ContinuousFlowFoundationAstroJev,
    train_continuous_flow_astrojev,
    SpectroscopicFiberTriageEngine,
    SpectroscopicTriageDecision,
)


def test_multi_survey_photometry_generation():
    """Verify synthesis of 13-band Euclid, DESI, and Rubin photometry."""
    n_samples = 50
    base_mu = np.random.uniform(18.0, 21.0, size=(n_samples, 10)).astype(np.float32)
    # Give some realistic optical/IR colors
    base_mu[:, 1] = base_mu[:, 0] + 0.5   # BP
    base_mu[:, 2] = base_mu[:, 0] - 0.5   # RP
    base_mu[:, 3] = base_mu[:, 0] - 1.5   # W1
    base_mu[:, 4] = base_mu[:, 0] - 2.5   # W2
    labels = np.random.choice([0, 2, 3, 5, 9], size=n_samples)

    generator = MultiSurveyPhotometryGenerator(rng_seed=42)
    phot, errors, masks, targets, zp_true = generator.generate_survey_photometry(base_mu, labels)

    assert phot.shape == (n_samples, 13)
    assert errors.shape == (n_samples, 13)
    assert masks.shape == (n_samples, 13)
    assert targets.shape == (n_samples,)
    assert zp_true.shape == (n_samples, 1)
    # Check that high-z quasars map to target 0 and brown dwarfs to 4
    for i, lab in enumerate(labels):
        if lab == 0:
            assert targets[i] == 0
        elif lab == 9:
            assert targets[i] == 4


def test_velocity_net_and_time_embedding():
    """Verify ConditionalFlowVelocityNet evaluates without NaNs."""
    batch_size = 8
    velocity_net = ConditionalFlowVelocityNet(
        z_dim=LATENT_SED_DIM, cond_dim=TOTAL_OBSERVABLE_DIM, hidden_dim=64
    )

    z_t = torch.randn(batch_size, LATENT_SED_DIM)
    t = torch.rand(batch_size)
    c = torch.randn(batch_size, TOTAL_OBSERVABLE_DIM)

    v_pred = velocity_net(z_t, t, c)
    assert v_pred.shape == (batch_size, LATENT_SED_DIM)
    assert not torch.isnan(v_pred).any()


def test_continuous_flow_model_forward_and_ode():
    """Verify ODE integration and evidential readout."""
    batch_size = 6
    model = ContinuousFlowFoundationAstroJev(
        z_dim=LATENT_SED_DIM, cond_dim=TOTAL_OBSERVABLE_DIM, hidden_dim=64
    )

    c = torch.randn(batch_size, TOTAL_OBSERVABLE_DIM)
    alpha, zp_pred, z_1 = model(c, num_ode_steps=3)

    assert alpha.shape == (batch_size, NUM_ALLOCATION_CLASSES)
    assert (alpha >= 1.0).all()  # Dirichlet concentration must be >= 1
    assert zp_pred.shape == (batch_size, 1)
    assert z_1.shape == (batch_size, LATENT_SED_DIM)

    # Test Dirichlet statistics
    probs, stds, u_epi, u_ale = model.compute_dirichlet_statistics(alpha)
    assert probs.shape == (batch_size, NUM_ALLOCATION_CLASSES)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(batch_size), atol=1e-4)
    assert stds.shape == (batch_size, NUM_ALLOCATION_CLASSES)
    assert (stds >= 0.0).all()
    assert (u_epi > 0.0).all()
    assert (u_ale >= 0.0).all()


def test_train_loop_and_rlcd_optimization():
    """Verify fast training and Disentangled RLCD execution."""
    n_samples = 64
    model = ContinuousFlowFoundationAstroJev(
        z_dim=LATENT_SED_DIM, cond_dim=TOTAL_OBSERVABLE_DIM, hidden_dim=32
    )

    c_train = torch.randn(n_samples, TOTAL_OBSERVABLE_DIM)
    y_train = torch.randint(0, NUM_ALLOCATION_CLASSES, (n_samples,))
    zp_train = torch.randn(n_samples, 1) * 0.03

    train_out = train_continuous_flow_astrojev(
        model=model,
        c_train=c_train,
        y_train=y_train,
        zp_train=zp_train,
        num_epochs=3,
        lr=1e-3,
        verbose=False
    )
    assert "final_loss" in train_out
    assert len(train_out["epoch_losses"]) == 3


def test_spectroscopic_fiber_triage_engine():
    """Verify autonomous triage decisions and error-bar retention."""
    model = ContinuousFlowFoundationAstroJev(
        z_dim=LATENT_SED_DIM, cond_dim=TOTAL_OBSERVABLE_DIM, hidden_dim=32
    )

    # Calibrate conformal gate
    c_cal = torch.randn(40, TOTAL_OBSERVABLE_DIM)
    y_cal = torch.randint(0, NUM_ALLOCATION_CLASSES, (40,))

    engine = SpectroscopicFiberTriageEngine(model, alpha_risk=0.02)
    threshold = engine.calibrate_conformal_gate(c_cal, y_cal)
    assert 0.0 <= threshold <= 1.0

    # Triage a candidate target
    c_test = torch.randn(TOTAL_OBSERVABLE_DIM)
    decision = engine.triage_target(c_test)

    assert isinstance(decision, SpectroscopicTriageDecision)
    assert decision.action.startswith("ALLOCATE_") or decision.action == "PURGE_CONTAMINANT_NO_FIBER"
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.dirichlet_std >= 0.0
    assert len(decision.ci_95) == 2
    assert decision.ci_95[0] <= decision.ci_95[1]
    assert len(decision.conformal_set) >= 1
    assert decision.fiber_exposure_minutes >= 0.0
