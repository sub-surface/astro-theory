"""Hermetic unit tests for Fast Radio Burst (FRB) evidential triage & dispersion modeling (EXP-2026-X)."""
from pathlib import Path
import numpy as np
import pytest
import torch

from celestrium.frb import (
    FRBBurstRecord,
    FRBEvidentialNet,
    FRBTriageEngine,
    FRB_CLASSES,
    NUM_FRB_CLASSES,
    NUM_FRB_FEATURES,
    bhat_empirical_scattering_ms,
    macquart_inferred_redshift,
    extract_frb_features,
    compute_stanford_debiased_ece,
    frb_rlcd_doubt_loss,
    calibrate_frb_conformal_risk,
    RealCHIMEFRBStreamer,
    train_frb_model,
)
from celestrium.data_streamer import CHIMEFRBStreamer


def test_bhat_scattering_and_macquart_redshift():
    # Test Bhat empirical scattering
    tau_low = bhat_empirical_scattering_ms(100.0)
    tau_high = bhat_empirical_scattering_ms(1000.0)
    assert tau_low > 0.0
    assert tau_high > tau_low

    # Test Macquart redshift scaling
    z_0 = macquart_inferred_redshift(60.0, dm_halo=60.0)
    z_1 = macquart_inferred_redshift(1010.0, dm_halo=60.0)
    assert z_0 == 0.0
    assert abs(z_1 - 1.0) < 0.05  # 950 pc cm^-3 excess gives ~z=1.0


def test_frb_feature_extraction():
    burst = FRBBurstRecord(
        name="FRB20180916B",
        ra=29.503,
        dec=65.717,
        glon=129.7,
        glat=3.7,
        dm=348.8,
        dm_err=0.2,
        dm_exc_ne2001=150.0,
        dm_exc_ymw16=140.0,
        snr=45.0,
        scat_ms=0.08,
        width_ms=1.5,
        flux_jy=2.5,
        fluence_jyms=4.0,
        is_repeater=True,
    )
    feats = extract_frb_features(burst, perturb_noise=False)
    assert feats.shape == (NUM_FRB_FEATURES,)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))

    # Test noise perturbation
    rng = np.random.default_rng(123)
    perturbed = extract_frb_features(burst, perturb_noise=True, rng=rng)
    assert perturbed.shape == (NUM_FRB_FEATURES,)
    assert not np.allclose(feats, perturbed)


def test_frb_evidential_net_forward():
    model = FRBEvidentialNet(in_features=NUM_FRB_FEATURES, hidden_dim=32, num_classes=NUM_FRB_CLASSES)
    model.eval()

    x = torch.randn(8, NUM_FRB_FEATURES)
    out = model(x)

    probs = out["probs"]
    sigmas = out["sigmas"]
    u_epi = out["u_epi"]

    assert probs.shape == (8, NUM_FRB_CLASSES)
    assert sigmas.shape == (8, NUM_FRB_CLASSES)
    assert u_epi.shape == (8,)

    # Probabilities sum to 1
    prob_sums = torch.sum(probs, dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    # Epistemic vacuity positive
    assert torch.all(u_epi > 0.0)

    # Sigmas match Dirichlet posterior variance formula: sqrt( p (1-p) / (S+1) )
    alpha = out["alpha"]
    s = torch.sum(alpha, dim=-1, keepdim=True)
    expected_sigmas = torch.sqrt(torch.clamp(probs * (1.0 - probs) / (s + 1.0), min=0.0))
    assert torch.allclose(sigmas, expected_sigmas, atol=1e-5)


def test_stanford_debiased_ece_calculation():
    # Perfectly calibrated synthetic probabilities
    rng = np.random.default_rng(42)
    n = 1000
    p = rng.uniform(0.5, 0.95, size=(n, 1))
    probs = np.hstack([p, 1.0 - p])
    # Generate labels matching probabilities
    labels = (rng.uniform(0.0, 1.0, size=n) > p.squeeze()).astype(int)

    metrics = compute_stanford_debiased_ece(probs, labels, n_bins=10)
    assert "ece" in metrics
    assert "debiased_e2" in metrics
    assert "rmsce" in metrics
    assert metrics["debiased_e2"] >= 0.0
    # Debiased E^2 should be smaller than raw ECE^2 due to finite-sample positive variance subtraction
    assert metrics["debiased_e2"] <= (metrics["ece"] ** 2 + 1e-4)


def test_disentangled_rlcd_training_and_gain():
    torch.manual_seed(0)
    # Train on streamer
    train_res = train_frb_model(epochs=4, device="cpu", seed=42)
    model = train_res["model"]
    calib_pre = train_res["calib_pre"]
    calib_post = train_res["calib_post"]
    crc = train_res["crc"]

    assert isinstance(model, FRBEvidentialNet)
    # RLCD must improve or maintain debiased calibration error
    assert calib_post["debiased_e2"] <= calib_pre["debiased_e2"] + 0.02
    assert "lambda_hat" in crc
    assert 0.0 < crc["lambda_hat"] <= 1.0


def test_production_triage_engine_actions_and_payloads():
    torch.manual_seed(42)
    # Build trained engine
    train_res = train_frb_model(epochs=6, device="cpu", seed=42)
    engine = FRBTriageEngine(model=train_res["model"], alpha_crc=0.05, crc_lambda=0.70, device="cpu")

    # Construct synthetic scenarios
    b_cosmo = FRBBurstRecord(
        name="FRB_COSMO_01",
        ra=180.0, dec=45.0, glon=120.0, glat=65.0,  # High galactic latitude
        dm=950.0, dm_err=0.2, dm_exc_ne2001=890.0, dm_exc_ymw16=885.0,
        snr=35.0, scat_ms=0.05, width_ms=1.2, flux_jy=4.0, fluence_jyms=6.0,
        is_repeater=False,
    )
    b_repeater = FRBBurstRecord(
        name="FRB_REP_02",
        ra=30.0, dec=60.0, glon=130.0, glat=25.0,
        dm=550.0, dm_err=0.5, dm_exc_ne2001=450.0, dm_exc_ymw16=440.0,
        snr=25.0, scat_ms=1.5, width_ms=4.0, flux_jy=1.0, fluence_jyms=4.0,
        is_repeater=True, repeater_name="FRB20180916B",
    )
    b_disk = FRBBurstRecord(
        name="FRB_DISK_03",
        ra=85.0, dec=20.0, glon=190.0, glat=2.5,  # In Galactic plane (|b| < 7)
        dm=220.0, dm_err=0.5, dm_exc_ne2001=30.0, dm_exc_ymw16=25.0,
        snr=18.0, scat_ms=12.0, width_ms=5.0, flux_jy=0.8, fluence_jyms=4.0,
        is_repeater=False,
    )

    results = engine.triage_bursts([b_cosmo, b_repeater, b_disk])
    assert len(results) == 3

    # All decisions must retain explicit analytical error bars (Zero Naked Predictions)
    for res in results:
        assert res.confidence > 0.0
        assert res.confidence_err >= 0.0
        assert 0.0 <= res.confidence_interval_95[0] <= res.confidence_interval_95[1] <= 1.0
        assert res.epistemic_vacuity > 0.0
        assert isinstance(res.prediction_set, list)

    # Repeater burst should be routed to radio repetition monitoring
    assert results[1].action == "MONITOR_RADIO_REPETITION"
    assert results[1].too_payload is not None
    assert "ROBOTIC_RADIO_ARRAY_MONITORING" in results[1].too_payload["instrument"]

    # Low latitude burst should be flagged as local plasma contaminant
    assert results[2].action == "FLAG_LOCAL_PLASMA_CONTAMINANT"


def test_chime_streamer_integration():
    streamer = CHIMEFRBStreamer()
    assert len(streamer) > 0
    bursts = list(streamer.stream_bursts())
    assert len(bursts) == len(streamer)
    b0 = bursts[0]
    assert isinstance(b0, FRBBurstRecord)
    assert b0.dm > 0.0
    assert b0.name != ""
