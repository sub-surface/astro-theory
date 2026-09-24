"""
=============================================================================
Tests for EXP-2026-W: Unified Multi-Tracer Anisotropy & Cosmic Dipole Co-Inference
=============================================================================
Hermetic verification of:
  1. Coordinate geometry and CMB kinematic baseline.
  2. Multi-tracer dataset loading & mock generation.
  3. Joint Poisson likelihood across H_0, H_1, and H_2 models.
  4. Goodman & Weare (2010) affine-invariant MCMC sampler & Gelman-Rubin R_hat.
  5. Conformal sky residual anomaly filtering.
"""
import math
import numpy as np
import pytest

from celestrium.multi_tracer import (
    SPEED_OF_LIGHT_KMS,
    V_CMB_KMS,
    BETA_CMB_MAGNITUDE,
    CMB_APEX_L_DEG,
    CMB_APEX_B_DEG,
    TRACER_DEFAULTS,
    lb_to_unit_vector,
    unit_vector_to_lb,
    get_cmb_beta_vector,
    TracerSurvey,
    MultiTracerDataset,
    MultiTracerLikelihood,
    MultiTracerMCMCEngine,
    MultiTracerConformalFilter,
)


def test_geometry_and_cmb_kinematics():
    """Verify coordinate conversion roundtrips and CMB velocity vectors."""
    l_orig = 264.021
    b_orig = 48.253
    vec = lb_to_unit_vector(np.array([l_orig]), np.array([b_orig]))[0]
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-7)

    l_rec, b_rec = unit_vector_to_lb(vec)
    assert np.isclose(l_rec, l_orig, atol=1e-4)
    assert np.isclose(b_rec, b_orig, atol=1e-4)

    beta_cmb = get_cmb_beta_vector()
    assert np.isclose(np.linalg.norm(beta_cmb), BETA_CMB_MAGNITUDE, rtol=1e-5)
    assert np.isclose(np.linalg.norm(beta_cmb) * SPEED_OF_LIGHT_KMS, V_CMB_KMS, atol=1e-2)


def test_mock_multi_tracer_dataset():
    """Verify mock dataset synthesis across all three observational tracers."""
    nside = 16  # Fast test resolution
    dataset = MultiTracerDataset(nside=nside, b_cut_deg=30.0)
    galactic_mask = np.ones(12 * nside * nside, dtype=bool)

    for key in ["quaia", "catwise", "nvss"]:
        survey = dataset._create_mock_survey(key, nside, galactic_mask)
        dataset.surveys[key] = survey
        assert survey.name == key
        assert len(survey.counts) == 12 * nside * nside
        assert survey.total_sources > 0
        assert survey.kinematic_factor > 2.0


def test_joint_poisson_likelihood_evaluations():
    """Test log-likelihood evaluation across H_0, H_1, and H_2 models."""
    nside = 16
    dataset = MultiTracerDataset(nside=nside, b_cut_deg=30.0)
    galactic_mask = np.ones(12 * nside * nside, dtype=bool)
    for key in ["quaia", "catwise", "nvss"]:
        dataset.surveys[key] = dataset._create_mock_survey(key, nside, galactic_mask)

    # 1. Unified Model (H1): 9 parameters
    ll_unified = MultiTracerLikelihood(dataset, model_type="unified")
    theta_u = np.array([
        0.0001, 0.0002, 0.0003,  # beta_x, beta_y, beta_z
        5.0, 1.0,               # Quaia ln_N0, gamma
        5.0, 1.0,               # CatWISE ln_N0, gamma
        3.5, 1.0                # NVSS ln_N0, gamma
    ])
    log_l = ll_unified.log_likelihood(theta_u)
    assert np.isfinite(log_l)
    log_p = ll_unified.log_posterior(theta_u)
    assert np.isfinite(log_p)

    # 2. CMB Null Model (H0): 6 parameters
    ll_null = MultiTracerLikelihood(dataset, model_type="cmb_null")
    theta_0 = np.array([
        5.0, 1.0,
        5.0, 1.0,
        3.5, 1.0
    ])
    log_l_0 = ll_null.log_likelihood(theta_0)
    assert np.isfinite(log_l_0)

    # 3. Decoupled Model (H2): 15 parameters
    ll_dec = MultiTracerLikelihood(dataset, model_type="decoupled")
    theta_2 = np.array([
        0.005, 0.0, 0.0, 5.0, 1.0,
        0.005, 0.0, 0.0, 5.0, 1.0,
        0.003, 0.0, 0.0, 3.5, 1.0
    ])
    log_l_2 = ll_dec.log_likelihood(theta_2)
    assert np.isfinite(log_l_2)


def test_mcmc_sampler_and_diagnostics():
    """Verify affine-invariant ensemble MCMC execution, R_hat, and error bars."""
    nside = 8  # Ultra-fast resolution for unit test
    dataset = MultiTracerDataset(nside=nside, b_cut_deg=30.0)
    galactic_mask = np.ones(12 * nside * nside, dtype=bool)
    for key in ["quaia", "catwise", "nvss"]:
        dataset.surveys[key] = dataset._create_mock_survey(key, nside, galactic_mask)

    likelihood = MultiTracerLikelihood(dataset, model_type="unified")
    engine = MultiTracerMCMCEngine(likelihood, n_walkers=16, seed=42)

    # Run short MCMC chain
    results = engine.run_mcmc(n_steps=40, burn_in=10, verbose=False)
    assert results["acceptance_fraction"] > 0.10
    assert np.isfinite(results["max_log_posterior"])
    assert results["aic"] != 0.0
    assert results["bic"] != 0.0
    assert len(results["param_summaries"]) == 9

    # Verify Dirichlet error-bar retention on each parameter
    for p in results["param_summaries"]:
        assert "ci_95" in p
        assert len(p["ci_95"]) == 2
        assert p["ci_95"][0] <= p["ci_95"][1]
        assert p["std"] >= 0.0
        assert p["r_hat"] >= 1.0

    # Verify physical bulk velocity inference
    b_sum = results["bulk_velocity_summary"]
    assert b_sum is not None
    assert b_sum["v_mean_kms"] > 0.0
    assert len(b_sum["v_ci_95"]) == 2


def test_conformal_residual_filtering():
    """Verify conformal non-conformity calibration on HEALPix sky residuals."""
    nside = 16
    dataset = MultiTracerDataset(nside=nside, b_cut_deg=30.0)
    galactic_mask = np.ones(12 * nside * nside, dtype=bool)
    survey = dataset._create_mock_survey("quaia", nside, galactic_mask)

    conformal_filter = MultiTracerConformalFilter(alpha_risk=0.05)
    best_beta = get_cmb_beta_vector()
    out = conformal_filter.compute_conformal_residuals(
        survey=survey,
        best_beta=best_beta,
        ln_n0=float(np.log(np.mean(survey.counts[survey.mask]))),
        gamma=1.0
    )

    assert out["survey"] == "quaia"
    assert out["tau_conformal"] > 0.0
    assert out["n_calibrated_pixels"] > 0
    # Conformal anomaly fraction should be approximately near alpha_risk (0.05)
    assert 0.0 <= out["anomaly_fraction"] <= 0.15
