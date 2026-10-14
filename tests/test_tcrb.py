"""Hermetic tests for EXP-2026-Y (T CrB): disk physics, Roche geometry, UVOT photometry, inversion."""
import math

import numpy as np
import pytest

from celestrium.novae import tcrb
from celestrium.novae.tcrb import DAY, FIDUCIAL, MSUN, YR, ViscousDisk, year_to_jd
from celestrium.novae.roche import (ballistic_stream, donor_surface, ellipsoidal_lightcurve, harmonic_amplitudes,
                                   l1_point)


def test_disk_steady_state_matches_analytic():
    d = ViscousDisk(FIDUCIAL, 3.0, 1.0, n_cells=120)
    ss = d.steady_state()
    assert d.mdot_inner(ss) == pytest.approx(1.0, rel=1e-8)
    r = d.grid.r
    inside = r < 0.6 * d.grid.r_out
    ana = 1 / (3 * np.pi * d.nu) * (1 - np.sqrt(d.grid.r_in / r))
    assert np.max(np.abs(ss[inside] / ana[inside] - 1)) < 0.02


def test_disk_conserves_mass():
    d = ViscousDisk(FIDUCIAL, 1.0, 1.0, n_cells=80)
    resp = d.pulse_response(2.0, 6000, 20.0)
    injected = 20.0 * DAY
    accreted = d.mdot_inner(resp).sum() * 2.0 * DAY
    left = (resp[-1] * d.grid.area).sum()
    assert (accreted + left) / injected == pytest.approx(1.0, rel=1e-6)


def test_disk_teff_scales_with_mdot():
    d = ViscousDisk(FIDUCIAL, 3.0, 1.0, n_cells=80)
    ss = d.steady_state()
    t1, t2 = d.teff(ss * 1e17), d.teff(ss * 16e17)
    np.testing.assert_allclose(t2 / np.clip(t1, 1e-9, None), 2.0, rtol=1e-6, atol=1e-6)  # T ∝ Mdot^(1/4)


def test_roche_lobe_volume_matches_eggleton():
    q = 0.93 / 1.35
    s = donor_surface(q, 1.0, 48, 96)
    c = np.array([1.0, 0, 0])
    vol = np.sum(np.einsum("ijk,ijk->ij", s["xyz"] - c, s["normal"]) * s["dA"]) / 3
    r_l = (3 * vol / (4 * np.pi)) ** (1 / 3)
    assert r_l == pytest.approx(FIDUCIAL.roche_lobe_eggleton(q), rel=0.01)
    assert 0.5 < l1_point(q) < 0.6


def test_ellipsoidal_amplitude_grows_with_inclination():
    q = 0.93 / 1.35
    s = donor_surface(q, 1.0, 30, 60)
    amps = []
    for inc in (30, 60, 85):
        ph, m, _ = ellipsoidal_lightcurve(q, inc, 5450.0, surf=s, n_phase=40)
        amps.append(abs(harmonic_amplitudes(ph, m)[2]))
    assert amps[0] < amps[1] < amps[2]
    assert 0.08 < amps[1] < 0.16  # ~0.12 mag P/2 semi-amplitude for a lobe-filling giant at i~60


def test_stream_hits_disk_at_tens_of_degrees():
    q = 0.93 / 1.35
    a = FIDUCIAL.a_orb / tcrb.RSUN
    _, imp, az = ballistic_stream(q, 58.0 / a)
    assert 10 < az < 30
    assert math.hypot(*imp) == pytest.approx(58.0 / a, rel=0.02)


def test_coincidence_correction_monotonic_and_saturates():
    raw = np.linspace(0.1, 95, 200)
    corr, sat = tcrb.coincidence_correct(raw)
    assert np.all(np.diff(corr[~sat]) > 0)
    assert np.all(corr >= raw * 0.999)
    assert sat[-1] and not sat[0]


def test_nightly_bins_handles_discrepant_pairs_and_nans():
    jd = np.array([2460000.6, 2460000.7, 2460001.6, 2460001.7])
    mag = np.array([10.0, 12.0, np.nan, 10.5])
    nb = tcrb.nightly_bins(jd, mag, np.full(4, 0.01), clip=0.5)
    assert len(nb["jd"]) == 2 and np.all(np.isfinite(nb["mag"]))


def test_ephemeris_recovers_injected_period():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(year_to_jd(2005), year_to_jd(2026), 600))
    p_true = 227.55
    ph = 2 * np.pi * (t - 2455828.9) / p_true
    trend = 0.3 * np.sin(2 * np.pi * (t - t[0]) / 3000.0)
    m = 10.2 + 0.12 * np.cos(2 * ph) + 0.02 * np.cos(ph) + trend + rng.normal(0, 0.03, len(t))
    fit = tcrb.fit_ephemeris(t, m, np.full(len(t), 0.03), np.linspace(224, 231, 701))
    assert fit["p_best"] == pytest.approx(p_true, abs=0.3)


def test_recurrence_predictive_is_survival_conditioned():
    r = tcrb.recurrence_predictive([78.5, 79.7, 81.4], 1946.11, 2026.78, n=200_000)
    assert r["median"] > 2026.78
    assert 0 < r["p_survive_to_now"] < 1


def test_tio_temperature_law_boosts_ellipsoidal_amplitude():
    """M-giant V flux is far more temperature-sensitive than a blackbody (TiO), raising the P/2 amplitude."""
    q = 0.93 / 1.35
    s = donor_surface(q, 1.0, 30, 60)
    bb = abs(harmonic_amplitudes(*ellipsoidal_lightcurve(q, 61, 5450.0, surf=s, n_phase=40)[:2])[2])
    tio = abs(harmonic_amplitudes(*ellipsoidal_lightcurve(q, 61, 5450.0, surf=s, n_phase=40, k_teff=18.0)[:2])[2])
    assert tio > 1.2 * bb


def test_literature_tables_are_consistent():
    from celestrium.novae import literature as lit
    for p in lit.PREDICTIONS:
        assert p.lo <= p.hi and p.made < 2027
        if p.center is not None:
            assert p.lo - 1e-9 <= p.center <= p.hi + 1e-9
    assert {s.name for s in lit.FAMILY} == {"T CrB", "RS Oph", "V3890 Sgr", "V745 Sco"}
    assert all(1.0 < m.value < 1.5 for m in lit.WD_MASS)


def test_truncation_and_giant_prior_hooks():
    torch = pytest.importorskip("torch")
    from celestrium.novae.inversion import AccretionInversion, BandData, InversionConfig
    eph = {"t0": 2455828.9, "period": 227.55}
    stub = [BandData("UVM2", np.array([year_to_jd(2010.0)]), np.array([14.0]), np.array([0.05]))]
    base = AccretionInversion(stub, eph, cfg=InversionConfig(t_start_yr=2008.0, t_end_yr=2012.0, bin_d=120.0, dt_d=10.0))
    trunc = AccretionInversion(stub, eph, cfg=InversionConfig(t_start_yr=2008.0, t_end_yr=2012.0, bin_d=120.0,
                                                              dt_d=10.0, truncation=True, giant_v_prior=(10.17, 0.06)))
    P0, P1 = base.init_params(mdot_q0=1e-9), trunc.init_params(mdot_q0=1e-9)
    jd = np.array([year_to_jd(2011.5)])
    # Mdot below Mdot_c -> inner disk removed -> fainter UV
    assert trunc.predict_mag(P1, "UVM2", jd)[0] > base.predict_mag(P0, "UVM2", jd)[0] + 0.3
    assert torch.isfinite(trunc.neg_log_post(P1))


@pytest.mark.slow
def test_inversion_injection_recovery_small():
    torch = pytest.importorskip("torch")
    from celestrium.novae.inversion import AccretionInversion, BandData, InversionConfig, accreted_mass
    eph = {"t0": 2455828.9, "period": 227.55}
    cfg = InversionConfig(t_start_yr=2008.0, t_end_yr=2020.0, n_iter=400, bin_d=120.0, dt_d=10.0)
    rng = np.random.default_rng(0)
    stub = [BandData(b, np.array([year_to_jd(2010.0)]), np.array([10.0]), np.array([0.05])) for b in ("V", "B", "U")]
    gen = AccretionInversion(stub, eph, cfg=cfg)
    yrs = tcrb.jd_to_year(gen.bin_start)
    truth = np.where((yrs > 2014.5), 8e-8, 2e-9)
    g = MSUN / YR
    P = gen.init_params()
    with torch.no_grad():
        P["log_mdot"].copy_(torch.log(torch.tensor(truth * g)))
        P["log_mdot_q"].fill_(float(np.log(2e-9 * g)))
        P["ell"].copy_(torch.tensor([0.0, 0.0, 0.12, 0.0]))
        P["logit_eta"].fill_(-4.0)
    data = []
    for b, n, s in (("V", 400, 0.04), ("B", 300, 0.08), ("U", 120, 0.15)):
        jd = np.sort(rng.uniform(year_to_jd(2008.5), year_to_jd(2019.9), n))
        data.append(BandData(b, jd, gen.predict_mag(P, b, jd) + rng.normal(0, s, n), np.full(n, 0.02)))
    # v2: the giant-brightness prior (here self-consistent with the truth) breaks the giant/disk degeneracy
    from dataclasses import replace
    cfg2 = replace(cfg, giant_v_prior=(float(gen.giant_v_at_max(P)), 0.06))
    inv = AccretionInversion(data, eph, cfg=cfg2)
    Pf, _ = inv.fit()
    w = {"hi": (2015.5, 2019.5)}
    ratio = accreted_mass(inv, Pf, w)["hi"] / accreted_mass(gen, P, w)["hi"]
    assert 0.4 < ratio < 2.5
