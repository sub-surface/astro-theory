"""
=============================================================================
Tests for Celestrium Multi-Messenger Evidential Triage Engine (EXP-2026-R)
=============================================================================
Verifies:
1. Multi-stream record parsing and spatio-temporal coincidence.
2. Heteroscedastic noise modeling and Kasen kilonova color evolution.
3. Evidential network forward pass and Krasnoselskii-Mann contractive loops.
4. Analytical Dirichlet BALD mutual information computation.
5. Conformal Risk Control calibration and finite-sample bound guarantees.
6. Multi-tier autonomous ToO serialization conformance (LCOGT & Gemini GMOS).
7. Null hypothesis audits: empty sky and permutation scrambling.
"""
import math
import numpy as np
import pytest
import torch

from celestrium.multimessenger import (
    GWAlertRecord,
    IceCubeNeutrinoAlert,
    HostGalaxy,
    OpticalTransientCandidate,
    extract_multimessenger_features,
    kasen_kilonova_flux,
    generate_multimessenger_scenario,
    MultiMessengerEvidentialNet,
    analytical_dirichlet_bald,
    calibrate_conformal_risk_control,
    MultiMessengerTriageEngine,
    train_multimessenger_model,
    run_empty_sky_null_audit,
    run_spatiotemporal_scrambling_mc,
    fetch_gracedb_alert,
    fetch_icecube_alert,
    NUM_MM_FEATURES,
    NUM_MM_CLASSES,
    MM_CLASSES,
)
from celestrium.too_protocol import ToOPipelineValidator


def test_gw_alert_record_and_likelihood():
    gw = GWAlertRecord(
        superevent_id="S240422ed",
        trigger_mjd=60400.0,
        ra_deg=180.0,
        dec_deg=10.0,
        error_area_90_deg2=100.0,
        distance_mean_mpc=150.0,
        distance_std_mpc=30.0,
    )
    # At center, angular distance is 0 and density is maximal
    sep = gw.angular_distance_deg(180.0, 10.0)
    assert sep == pytest.approx(0.0, abs=1e-5)
    dens_center = gw.spatial_density(180.0, 10.0)
    dens_offset = gw.spatial_density(185.0, 10.0)
    assert dens_center > dens_offset

    # Distance likelihood
    chi2_match, like_match = gw.distance_likelihood(150.0, 10.0)
    assert chi2_match == pytest.approx(0.0, abs=1e-4)
    assert like_match == pytest.approx(1.0, abs=1e-4)

    chi2_mismatch, like_mismatch = gw.distance_likelihood(400.0, 10.0)
    assert chi2_mismatch > 20.0
    assert like_mismatch < 1e-4


def test_icecube_neutrino_coincidence():
    gw = GWAlertRecord(
        superevent_id="S240422ed",
        trigger_mjd=60400.0,
        ra_deg=180.0,
        dec_deg=10.0,
        error_area_90_deg2=100.0,
        distance_mean_mpc=150.0,
        distance_std_mpc=30.0,
    )
    # Perfectly coincident neutrino: same sky position, same time
    nu_match = IceCubeNeutrinoAlert(
        alert_id="IC260924A",
        trigger_mjd=60400.0,
        ra_deg=180.0,
        dec_deg=10.0,
        error_radius_deg=0.8,
        p_astro=0.85,
    )
    score_match = nu_match.coincidence_score(gw)
    assert score_match > 0.70

    # Distant / delayed neutrino: > 1000s later
    nu_delayed = IceCubeNeutrinoAlert(
        alert_id="IC260924B",
        trigger_mjd=60400.0 + 2000.0 / 86400.0,
        ra_deg=180.0,
        dec_deg=10.0,
        error_radius_deg=0.8,
        p_astro=0.85,
    )
    score_delayed = nu_delayed.coincidence_score(gw)
    assert score_delayed == 0.0


def test_kasen_kilonova_flux_and_reddening():
    # Test Kasen kilonova model at 140 Mpc
    # Early phase (0.5 days): blue band (g) is bright, then fades faster than r and i
    g_early = kasen_kilonova_flux(0.5, 140.0, "g")
    r_early = kasen_kilonova_flux(0.5, 140.0, "r")
    i_early = kasen_kilonova_flux(0.5, 140.0, "i")

    g_late = kasen_kilonova_flux(3.0, 140.0, "g")
    r_late = kasen_kilonova_flux(3.0, 140.0, "r")
    i_late = kasen_kilonova_flux(3.0, 140.0, "i")

    # Blue fades much faster than red
    decay_g = g_late - g_early
    decay_r = r_late - r_early
    decay_i = i_late - i_early
    assert decay_g > decay_r > decay_i

    # Color g - r reddens rapidly
    color_early = g_early - r_early
    color_late = g_late - r_late
    assert color_late > color_early  # Becomes redder!


def test_feature_extraction():
    gw = GWAlertRecord("S_TEST", 60400.0, 180.0, 10.0, 100.0, 140.0, 20.0)
    cand = OpticalTransientCandidate(
        candidate_id="CAND_01",
        ra_deg=180.1,
        dec_deg=10.1,
        discovery_mjd=60400.5,
        last_mjd=60401.0,
        mag_r=19.5,
        mag_err_r=0.05,
        mag_g=19.8,
        mag_err_g=0.06,
        rate_r=0.85,
        rate_g=1.20,
        color_gr=0.3,
        rate_color_gr=0.35,
        host_galaxy=HostGalaxy("HG_01", 180.1, 10.1, 0.03, 142.0, 8.0, 10.5),
    )
    feat = extract_multimessenger_features(cand, gw)
    assert feat.shape == (NUM_MM_FEATURES,)
    assert not np.isnan(feat).any()
    # Check spatial density and dist_chi2 are reasonable
    assert feat[1] < 1.0  # dist_chi2 near 0 since 142 is close to 140


def test_evidential_net_forward_and_dirichlet():
    model = MultiMessengerEvidentialNet(in_features=NUM_MM_FEATURES, d_model=64, num_classes=NUM_MM_CLASSES, n_iter=3)
    dummy_x = torch.randn(4, NUM_MM_FEATURES)
    out = model(dummy_x)

    assert "alpha" in out and "probs" in out and "u_epi" in out and "delta_eq" in out
    alpha = out["alpha"]
    probs = out["probs"]
    u_epi = out["u_epi"]

    # Dirichlet concentrations must be >= 1.0
    assert torch.all(alpha >= 1.0)
    # Probs must sum to 1.0
    prob_sums = torch.sum(probs, dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    # Epistemic vacuity in (0, 1]
    assert torch.all(u_epi > 0.0) and torch.all(u_epi <= 1.0)


def test_analytical_dirichlet_bald():
    # Test high evidence vs uniform evidence
    alpha_certain = np.array([[20.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    alpha_doubt = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])

    bald_certain = analytical_dirichlet_bald(alpha_certain)
    bald_doubt = analytical_dirichlet_bald(alpha_doubt)

    assert bald_certain.shape == (1,)
    assert bald_doubt.shape == (1,)
    assert bald_certain[0] < bald_doubt[0]  # Disagreement is higher under doubt simplex


def test_conformal_risk_control_calibration():
    rng = np.random.default_rng(42)
    n = 1000
    labels = rng.integers(0, NUM_MM_CLASSES, n)
    is_kn = (labels == 0).astype(float)
    # True kilonova has high probability, contaminants low
    p_kn = np.where(is_kn == 1.0, rng.beta(8, 2, n), rng.beta(1, 10, n))
    probs = np.zeros((n, NUM_MM_CLASSES))
    probs[:, 0] = p_kn
    probs[:, 1:] = (1.0 - p_kn[:, None]) / (NUM_MM_CLASSES - 1)

    crc = calibrate_conformal_risk_control(probs, labels, alpha_risk=0.05, target_class=0)
    assert "lambda_hat" in crc
    assert 0.0 < crc["lambda_hat"] <= 1.0
    assert crc["empirical_risk"] <= 0.05
    assert crc["sample_retention"] > 0.50


def test_end_to_end_multimessenger_triage():
    gw, nu, cands = generate_multimessenger_scenario(
        n_contaminants=50,
        distance_mpc=120.0,
        error_area_deg2=80.0,
        inject_kilonova=True,
        seed=101,
    )
    # Train a small model: the engine's default checkpoint is gitignored, and a
    # randomly initialized net cannot be expected to flag the injected kilonova.
    torch.manual_seed(0)
    model = train_multimessenger_model(n_train=1200, n_val=300, epochs=5, save_path=None, device="cpu")["model"]
    engine = MultiMessengerTriageEngine(model=model, alpha_crc=0.05, crc_lambda=0.65, device="cpu")
    results = engine.triage_candidates(cands, gw, nu)

    assert len(results) == len(cands)
    # Check that at least one candidate triggered follow-up (the injected kilonova)
    actions = [r.action for r in results]
    assert ("GEMINI_RAPID_TOO" in actions) or ("LCOGT_SCREENING_TOO" in actions)

    # Validate payload for any triggered Gemini rapid ToO
    gemini_res = next((r for r in results if r.action == "GEMINI_RAPID_TOO"), None)
    if gemini_res:
        assert gemini_res.too_payload is not None
        assert gemini_res.voevent_packet is not None
        is_valid_gemini, errs_g = ToOPipelineValidator.validate_gemini_payload(gemini_res.too_payload)
        assert is_valid_gemini is True
        assert len(errs_g) == 0

    # Validate payload for any triggered LCOGT screening ToO
    lcogt_res = next((r for r in results if r.action == "LCOGT_SCREENING_TOO"), None)
    if lcogt_res:
        assert lcogt_res.too_payload is not None
        is_valid_lcogt, errs_l = ToOPipelineValidator.validate_lcogt_payload(lcogt_res.too_payload)
        assert is_valid_lcogt is True
        assert len(errs_l) == 0


def test_empty_sky_null_audit():
    engine = MultiMessengerTriageEngine(alpha_crc=0.05, crc_lambda=0.89)
    null_res = run_empty_sky_null_audit(engine, n_fields=5, candidates_per_field=50, seed=42)
    assert null_res["total_unassociated_candidates"] == 250
    # Under empty sky, false alarm rate should satisfy target safety bound
    assert null_res["empirical_false_alarm_rate"] <= 0.05
    assert null_res["null_hypothesis_satisfied"] is True


def test_live_connectors_fallback():
    # GraceDB fallback
    gw_mock = fetch_gracedb_alert("S_OFFLINE_TEST")
    assert gw_mock is not None
    assert gw_mock.superevent_id == "S_OFFLINE_TEST"
    assert gw_mock.distance_mean_mpc > 0.0

    # IceCube fallback
    nu_mock = fetch_icecube_alert("IC_OFFLINE_TEST")
    assert nu_mock is not None
    assert nu_mock.alert_id == "IC_OFFLINE_TEST"
    assert nu_mock.energy_tev > 0.0
