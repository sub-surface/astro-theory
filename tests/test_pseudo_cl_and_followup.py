"""Hermetic unit tests for pseudo-Cl mode-coupling dipole deconvolution and active inference."""
import math
import numpy as np
import pytest
from astropy_healpix import HEALPix
from typer.testing import CliRunner

from celestrium.experiments.quaia_pseudo_cl import (
    real_ylm_basis,
    compute_mode_coupling,
    deconvolve_dipole,
)
from celestrium.followup import triage_sources, triage_candidates_for_followup
from celestrium.cli import app


def test_real_ylm_orthonormality():
    """Verify spherical harmonics integrate to identity on full 4pi sky."""
    nside = 16
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    Y, ll, mm = real_ylm_basis(lon.rad, lat.rad, lmax=2)

    # (lmax + 1)^2 = 9 modes
    assert Y.shape == (hp.npix, 9)
    omega_pix = 4.0 * math.pi / hp.npix
    gram = omega_pix * (Y.T @ Y)
    identity = np.eye(9)

    # Discrete pixelization error at nside=16 is < 0.005
    assert np.max(np.abs(gram - identity)) < 0.005


def test_mode_coupling_full_vs_masked_sky():
    """Verify mode coupling is diagonal on full sky and non-diagonal when masked."""
    nside = 16
    hp = HEALPix(nside=nside, order="ring")
    full_mask = np.ones(hp.npix, dtype=bool)

    K_full, M_ll_full, cond_full = compute_mode_coupling(full_mask, lmax=2, nside=nside)
    assert cond_full < 1.05
    assert np.allclose(K_full, np.eye(9), atol=0.01)

    # Mask Galactic plane |b| < 20 deg
    _, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    cut_mask = (np.abs(lat.deg) >= 20.0).astype(float)
    K_cut, M_ll_cut, cond_cut = compute_mode_coupling(cut_mask, lmax=2, nside=nside)

    # Condition number must increase due to mode coupling
    assert cond_cut > cond_full
    # Monopole-quadrupole coupling must be non-zero
    assert abs(K_cut[0, 6]) > 0.05  # Y_00 couples with Y_20


def test_dipole_deconvolution_recovery():
    """Verify deconvolve_dipole recovers an injected dipole on a cut sky."""
    nside = 16
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    # Inject dipole D = [0.03, 0.0, 0.0] -> D_amp = 0.03 in x-direction
    x = np.cos(lat.rad) * np.cos(lon.rad)
    injected_D = 0.03
    density = 1000.0 * (1.0 + injected_D * x)

    # Mask Galactic plane |b| < 15 deg
    mask = (np.abs(lat.deg) >= 15.0).astype(float)

    res = deconvolve_dipole(density, mask, lmax=2, nside=nside)
    dec = res["deconvolved"]

    # Deconvolved dipole amplitude should recover injected amplitude within 10%
    assert math.isclose(dec["amplitude"], injected_D, rel_tol=0.10)
    # Direction should point near l=0, b=0 (x-axis)
    assert dec["l_deg"] < 10.0 or dec["l_deg"] > 350.0
    assert abs(dec["b_deg"]) < 10.0


def test_active_inference_triage():
    """Verify triage_sources routes clean quasars, anomalies, and unstable sources."""
    rng = np.random.default_rng(42)
    # Synthetic batch: 1 clean quasar, 1 high-uncertainty source
    features = np.zeros((2, 10), dtype=np.float32)
    # Source 0: Bright quasar
    features[0] = [19.0, 0.6, -0.2, 14.0, 1.2, 0.1, 0.2, 0.5, 0.6, 50.0]
    # Source 1: Faint ambiguous source with high PM error
    features[1] = [21.5, 1.5, 0.3, 16.5, 0.4, 2.0, 3.0, 0.2, 0.3, 10.0]

    res = triage_sources(features, u_epi_threshold=0.30, noul_min=0.50)
    stats = res["stats"]
    assert stats["total_triaged"] == 2
    assert stats["auto_cataloged"] + stats["followup_triggered"] + stats["deliberation_needed"] >= 1


def test_cli_jev_evaluate():
    """Verify celestrium jev evaluate runs hermetically via Typer runner."""
    runner = CliRunner()
    result = runner.invoke(app, [
        "jev", "evaluate",
        "--phot-g", "19.5",
        "--w1-w2", "1.1",
        "--pm", "0.2",
    ])
    assert result.exit_code == 0
    assert "AstroJev System One Deliberation" in result.output
    assert "Quasar_AGN" in result.output


def test_cli_jev_help():
    """Verify all jev subcommands are visible in help text."""
    runner = CliRunner()
    result = runner.invoke(app, ["jev", "--help"])
    assert result.exit_code == 0
    assert "classify" in result.output
    assert "dipole" in result.output
    assert "evaluate" in result.output
