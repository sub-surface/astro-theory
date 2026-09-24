"""Hermetic unit tests for Euclid DR1 cosmic dipole pipeline."""
import math
import numpy as np
import pytest
from astropy_healpix import HEALPix

from celestrium import forecast, mocks, ellis_baldwin
from celestrium.core.kernel import Kernel
from celestrium.core.ledger import Ledger
import celestrium.caps  # noqa: F401


def test_build_dr1_footprint_geometry():
    mask, meta = forecast.build_dr1_footprint(nside=16, area_deg2=1900.0)
    assert len(mask) == 12 * 16 ** 2
    assert meta["nside"] == 16
    assert 1500.0 <= meta["actual_area_deg2"] <= 2300.0
    assert 0.035 <= meta["f_sky"] <= 0.055
    assert len(meta["patches"]) == 3
    assert mask.sum() == meta["surviving_pixels"]


def test_dipole_fisher_full_sky_limit():
    hp = HEALPix(nside=16, order="ring")
    full_mask = np.ones(hp.npix, dtype=bool)
    n_sources = 1000000.0

    cov_D, sigma_shot, cond = forecast.compute_dipole_fisher(full_mask, n_sources, nside=16)
    expected_sigma = math.sqrt(3.0 / n_sources)

    # On full sky, sigma_D_shot should match sqrt(3/N) within ~2% discretization
    assert math.isclose(sigma_shot, expected_sigma, rel_tol=0.03)
    assert cond < 5.0  # Orthonormal design matrix on full sphere


def test_dipole_fisher_partial_sky():
    mask, _ = forecast.build_dr1_footprint(nside=32, area_deg2=1900.0)
    n_sources = 500000.0

    cov_D, sigma_shot, cond = forecast.compute_dipole_fisher(mask, n_sources, nside=32)
    expected_full_sky = math.sqrt(3.0 / n_sources)

    # Partial sky should have geometric penalty > 1
    assert sigma_shot > expected_full_sky
    assert np.all(np.linalg.eigvalsh(cov_D) > 0)
    assert cond > 10.0


def test_harmonic_leakage_scaling():
    mask, _ = forecast.build_dr1_footprint(nside=32, area_deg2=1900.0)

    cov1, sigma1 = forecast.compute_harmonic_leakage(mask, nside=32, c2_clustering=1e-5)
    cov2, sigma2 = forecast.compute_harmonic_leakage(mask, nside=32, c2_clustering=4e-5)

    # Leakage amplitude scales as sqrt(C_2)
    assert math.isclose(sigma2 / sigma1, math.sqrt(4.0), rel_tol=1e-3)
    assert np.all(np.linalg.eigvalsh(cov1) >= 0)


def test_forecast_dr1_verdict():
    # Standard parameters: harmonic leakage dominates -> DRESS REHEARSAL
    res = forecast.forecast_dr1(
        area_deg2=1900.0,
        density_arcmin2=30.0,
        nside=32,
        d_anom=0.012,
        d_kin=0.0047,
        c2_clustering=5e-5,
    )
    assert res["verdict"] == "DRESS REHEARSAL"
    assert res["is_measurement"] is False
    assert 1.0 <= res["snr"] < 3.0

    # Low clustering, huge anomaly -> MEASUREMENT
    res_meas = forecast.forecast_dr1(
        area_deg2=1900.0,
        density_arcmin2=30.0,
        nside=32,
        d_anom=0.030,
        d_kin=0.0047,
        c2_clustering=1e-6,
    )
    assert res_meas["verdict"] == "MEASUREMENT"
    assert res_meas["is_measurement"] is True
    assert res_meas["snr"] >= 3.0


def test_pixel_mock_generation_and_fit():
    mask, _ = forecast.build_dr1_footprint(nside=16, area_deg2=1900.0)
    rng = np.random.default_rng(123)

    counts = mocks.generate_pixel_mock(
        mask=mask,
        n_sources=50000.0,
        amplitude=0.02,
        lon=264.0,
        lat=48.0,
        nside=16,
        frame="galactic",
        rng=rng,
    )
    assert len(counts) == len(mask)
    assert counts[~mask].sum() == 0
    assert counts[mask].sum() > 40000

    fit = mocks.fit_dipole(counts, mask, nside=16, frame="galactic")
    assert fit["amplitude"] > 0
    assert fit["monopole"] > 0
    assert fit["n_pixels"] == mask.sum()


def test_mock_suite_variance_consistency():
    res = mocks.run_mock_suite(
        n_mocks=20,
        area_deg2=1900.0,
        n_sources=20000.0,
        amplitude=0.005,
        nside=16,
        seed=99,
    )
    assert res["n_mocks"] == 20
    assert res["signal"]["mean"] > 0
    assert res["null"]["mean"] > 0
    # Mock standard deviation should be within a factor of 2 of analytic shot noise
    assert 0.5 <= res["ratio_mock_to_analytic_sigma"] <= 2.0


def test_ellis_baldwin_predictions():
    res = ellis_baldwin.predict_all_bands(n_samples=5000, seed=42)
    preds = res["predictions"]

    assert "VIS" in preds
    assert "NISP_Y" in preds
    assert "GALAXY_COMBINED" in preds
    assert "AGN_QUASAR" in preds

    vis = preds["VIS"]
    assert 0.0040 <= vis["d_kin_mean"] <= 0.0055
    assert vis["d_kin_std"] < 0.001

    qso = preds["AGN_QUASAR"]
    # Quasars have steeper slope x ~ 1.6 -> larger D_kin
    assert qso["d_kin_mean"] > vis["d_kin_mean"]

    apex = res["cmb_apex"]
    assert math.isclose(apex["l"], 264.021, abs_tol=1e-3)
    assert math.isclose(apex["b"], 48.253, abs_tol=1e-3)


def test_dr1_capabilities_in_kernel(tmp_path):
    k = Kernel(Ledger(tmp_path / "test.db"))

    # 1. dr1_footprint
    fp = k.run("analysis.dr1_footprint", {"nside": 16, "area_deg2": 1900.0})
    assert fp.kind == "table"
    tab = k.load(fp.id)
    assert "coverage" in tab.colnames
    assert fp.meta["surviving_pixels"] > 0

    # 2. dr1_forecast
    fc_art = k.run("analysis.dr1_forecast", {
        "area_deg2": 1900.0, "density_arcmin2": 30.0, "nside": 16,
        "d_anom": 0.012, "d_kin": 0.0047, "c2_clustering": 5e-5,
    })
    assert fc_art.kind == "data"
    fc_data = k.load(fc_art.id)
    assert fc_data["verdict"] in ("MEASUREMENT", "DRESS REHEARSAL")

    # 3. dr1_mocks
    mc_art = k.run("analysis.dr1_mocks", {
        "n_mocks": 5, "area_deg2": 1900.0, "n_sources": 5000.0,
        "amplitude": 0.0047, "nside": 16, "seed": 42,
    })
    assert mc_art.kind == "data"
    mc_data = k.load(mc_art.id)
    assert "signal" in mc_data and "null" in mc_data

    # 4. dr1_ellis_baldwin
    eb_art = k.run("analysis.dr1_ellis_baldwin", {"n_samples": 2000, "seed": 42})
    assert eb_art.kind == "data"
    eb_data = k.load(eb_art.id)
    assert "predictions" in eb_data
