"""Hermetic tests for the plate-archive nova detectors (EXP-2026-Z) and DASCH light-curve assembly."""
import numpy as np

from celestrium.archives import dasch
from celestrium.novae import plates as P


def _lc(t, mag, lim, det, sat=None):
    n = len(t)
    return dict(t=np.asarray(t, float), mag=np.asarray(mag, float), err=np.full(n, 0.15), lim=np.asarray(lim, float),
                det=np.asarray(det, bool), good=np.ones(n, bool), near_lim=np.zeros(n, bool),
                sat=np.zeros(n, bool) if sat is None else np.asarray(sat, bool),
                series=np.array(["a"] * n, dtype=object), plate=np.arange(n, dtype=float))


def _faint_star_with_nova(t_erupt=1930.3, peak=9.0, t2=20.0, seed=0):
    """A 19th-mag star on plates reaching ~14; a nova at `t_erupt` detected while brighter than the limit."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(1890, 1990, 2500))
    lim = rng.normal(14.0, 0.7, len(t))
    m_true = np.full(len(t), 19.0)
    dt = (t - t_erupt) * 365.25
    f = P.template_flux(dt, t2)
    m_true = -2.5 * np.log10(10 ** (-0.4 * m_true) + 10 ** (-0.4 * peak) * f)
    det = m_true < lim
    mag = np.where(det, m_true + rng.normal(0, 0.15, len(t)), np.nan)
    return _lc(t, mag, lim, det)


def test_template_two_magnitude_point():
    f = P.template_flux(np.array([0.0, 20.0]), 20.0)
    assert np.isclose(f[0], 1.0)
    assert np.isclose(-2.5 * np.log10(f[1]), 2.0, atol=1e-6)


def test_d1_recovers_injected_nova_and_d0_too():
    lc = _faint_star_with_nova()
    p = P.prepare(lc, 19.0)
    e = P.d1_scan(p)
    assert e.score > 50
    assert abs(e.t0 - 1930.3) < 0.1
    assert e.n_nights >= 2
    assert P.d0_scan(p).score > 0


def test_quiet_faint_star_scores_nothing():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(1890, 1990, 2000))
    lim = rng.normal(14.0, 0.7, len(t))
    p = P.prepare(_lc(t, np.full(len(t), np.nan), lim, np.zeros(len(t), bool)), 19.0)
    assert P.d1_scan(p).score == 0.0
    assert P.d0_scan(p).score == 0.0


def test_single_plate_event_is_not_an_event():
    """A8: a lone bright point (a plate defect, or a nova too fast to catch twice) is never reported, while the same
    point plus a decline on a second night is (the deeper limits that follow make a constant level impossible)."""
    rng = np.random.default_rng(4)
    base = np.sort(rng.uniform(1890, 1990, 2000))
    base = base[np.abs(base - 1950.0) > 60 / 365.25]
    t = np.sort(np.concatenate([base, [1950.0], 1950.0 + np.arange(1, 30) / 365.25]))
    det = t == 1950.0
    assert det.sum() == 1
    e = P.d1_scan(P.prepare(_lc(t, np.where(det, 12.0, np.nan), np.full(len(t), 15.0), det), 19.0))
    assert e.score == 0 and e.n_nights == 0
    # a fast nova caught on two consecutive nights, next plates 10+ days later (by then fainter than the limit)
    t2 = np.sort(np.concatenate([base, 1950.0 + np.array([0.0, 1.0]) / 365.25, 1950.0 + np.arange(10, 30) / 365.25]))
    det2 = t2 < 1950.0 + 1.5 / 365.25
    det2 &= t2 >= 1950.0
    mag2 = np.where(t2 == 1950.0, 12.0, np.where(det2, 12.4, np.nan))
    e2 = P.d1_scan(P.prepare(_lc(t2, mag2, np.full(len(t2), 15.0), det2), 19.0))
    assert e2.score > 0 and e2.n_nights == 2


def test_slow_state_change_is_absorbed_by_box_null():
    """A bright star stepping up 1.6 mag for 3 years (a symbiotic high state) is flagged as a state change."""
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(1890, 1990, 3000))
    m = np.where((t > 1938) & (t < 1941), 9.4, 11.0) + rng.normal(0, 0.15, len(t))
    step = P.d1_scan(P.prepare(_lc(t, m, np.full(len(t), 14.0), np.ones(len(t), bool)), 11.0))
    real = P.d1_scan(P.prepare(_faint_star_with_nova(1938.0, 7.0, 10.0), 19.0))
    # the constant-level null absorbs the plateau; what survives is the end of the high state, which the
    # pre-event excess exposes (the source was already ~1.6 mag bright before t0), unlike a nova
    assert step.score < 0.5 * real.score
    assert step.extra["pre_excess"] > 1.0
    assert real.extra["pre_excess"] < 0.5


def test_drop_windows_removes_rows():
    lc = _faint_star_with_nova()
    out = P.drop_windows(lc, [(1928.0, 1932.0)])
    assert not ((out["t"] >= 1928) & (out["t"] <= 1932)).any()
    assert len(out["t"]) < len(lc["t"])


def _cols(ra, dec, mags, series, plates):
    n = len(mags)
    m = np.asarray(mags, float)
    return {"date_jd": np.linspace(2.42e6, 2.43e6, n), "series": np.array(series, dtype=object),
            "plate_number": np.asarray(plates, float), "solution_number": np.zeros(n),
            "magcal_magdep": m, "ra_deg": np.where(np.isfinite(m), ra, np.nan),
            "dec_deg": np.where(np.isfinite(m), dec, np.nan), "limiting_mag_local": np.full(n, 14.0)}


def test_assemble_reassigns_split_detections():
    ra, dec = 100.0, 20.0
    own = _cols(ra, dec, [np.nan, np.nan, np.nan], ["a", "a", "a"], [1, 2, 3])
    # neighbour 12" away owns a detection measured 2" from the target on plate 2, and its own image on plate 3
    nra = ra + 12 / 3600 / np.cos(np.radians(dec))
    nb = _cols(ra + 2 / 3600, dec, [np.nan, 8.0, np.nan], ["a", "a", "a"], [1, 2, 3])
    nb_own_image = _cols(nra, dec, [np.nan, np.nan, 16.0], ["a", "a", "a"], [1, 2, 3])
    out = dasch.assemble(own, [(nb, nra, dec), (nb_own_image, nra, dec)], ra, dec)
    det = np.isfinite(out["magcal_magdep"])
    assert det.sum() == 1 and out["magcal_magdep"][det][0] == 8.0
    assert len(out["magcal_magdep"]) == 3  # plates 1 and 3 keep the own non-detection limits


def test_parse_csv_lines_numeric_and_blank():
    c = dasch.parse_csv_lines(["a,b,s", "1,,x", "2.5,3,y"])
    assert np.isnan(c["b"][0]) and c["a"][1] == 2.5 and list(c["s"]) == ["x", "y"]


def test_n_nights_clusters_by_gap_not_ut_day():
    """One Boyden night straddles 0h UT (16-02 UT) and must count once; consecutive nights count twice."""
    jy = 1.0 / 365.25
    one_night = 1929.0 + np.array([-0.30, -0.10, 0.05, 0.08]) * jy  # 7 h of plates across the UT day boundary
    assert P.n_nights(one_night) == 1
    assert P.n_nights(np.r_[one_night, one_night + 1.0 * jy]) == 2
    assert P.n_nights([]) == 0
