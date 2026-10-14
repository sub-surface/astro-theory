"""Nova-eruption detectors for century-long plate light curves (EXP-2026-Z).

D0 (naive): points far brighter than a robust baseline, clustered in time.
D1 (physics-informed): censored-likelihood scan with a universal-decline nova template added in flux to a running
quiescent envelope. Detections enter through a Student-t likelihood, non-detections through the probability that
the model is fainter than the plate limit, so a spurious bright point on one plate is penalised by deeper
non-detections on neighbouring nights.

Input is the dict from `celestrium.archives.dasch.tidy` (t in Julian years).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln, log_ndtr

DAY = 1.0 / 365.25
T2_GRID_D = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
T0_OFFSETS = (0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0)  # in units of t2, before the bright point
NU = 3.0
SIG_LIM = 0.3
MISS = 0.02  # probability that a plate misses a star brighter than its limit (defects, edges)
ENV_HALF_D = 500.0
MAX_ANCHORS = 15
MIN_AMP = 1.5  # A2: eruptions must peak >= 1.5 mag above quiescence (smaller = ordinary variability)
MIN_NIGHTS = 2  # A8: an event must be supported by detections on >= 2 nights (single plates are defect-dominated)


@dataclass
class Event:
    score: float = 0.0
    t0: float = np.nan
    t2_d: float = np.nan
    m_peak: float = np.nan
    n_nights: int = 0
    n_plates: int = 0
    n_contra: int = 0
    extra: dict = field(default_factory=dict)


def _good(lc):
    g = lc["good"]
    return {k: (v[g] if hasattr(v, "__len__") and len(v) == len(g) else v) for k, v in lc.items()}


def envelope(t, mag, det, lim, m_faint: float, half_d: float = ENV_HALF_D, exclude=None, at=None):
    """Running median quiescent level within +-half_d days, using every good plate.

    A non-detection enters as max(limit, m_faint): the star was fainter than the limit, and no brighter than
    its catalogue level would suggest. Without this, a faint nova's few (eruption) detections become their own
    baseline. `exclude` leaves rows out (the event window); `at` restricts the output rows.
    """
    val = np.where(det, mag, np.maximum(np.nan_to_num(lim, nan=m_faint), m_faint))
    use = np.isfinite(val)
    if exclude is not None:
        use &= ~exclude
    td, vd = t[use], val[use]
    o = np.argsort(td)
    td, vd = td[o], vd[o]
    tq = t if at is None else t[at]
    env = np.full(len(tq), m_faint, float)
    if len(td) < 10:
        return env
    lo = np.searchsorted(td, tq - half_d * DAY)
    hi = np.searchsorted(td, tq + half_d * DAY)
    for i in range(len(tq)):
        if hi[i] - lo[i] >= 5:
            env[i] = np.median(vd[lo[i]:hi[i]])
    return env


def _mad_sigma(r):
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 0.3
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def _lnt(r, s):
    """Student-t log density, nu = NU, scale s."""
    z2 = (r / s) ** 2
    return (gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * np.log(NU * np.pi) - np.log(s)
            - (NU + 1) / 2 * np.log1p(z2 / NU))


def _lnlim(m_model, lim):
    """log P(non-detection | model mag) with a plate-miss floor."""
    p = np.exp(log_ndtr((m_model - lim) / SIG_LIM))
    return np.log(MISS + (1 - MISS) * p)


def template_flux(dt_d, t2_d):
    """Peak-normalised flux; universal decline F ~ (1 + dt/tau)^-1.75 (Hachisu & Kato 2006), 3-day rise."""
    tau = t2_d / (10 ** (0.8 / 1.75) - 1)
    f = np.zeros_like(dt_d, dtype=float)
    post = dt_d >= 0
    f[post] = (1 + dt_d[post] / tau) ** -1.75
    rise = (dt_d < 0) & (dt_d > -3)
    f[rise] = 10 ** (-0.4 * 6 * (-dt_d[rise] / 3))
    return f


def prepare(lc, m_faint: float):
    g = _good(lc)
    t, mag, det, err, lim = g["t"], g["mag"], g["det"], g["err"], g["lim"]
    env = envelope(t, mag, det, lim, m_faint)
    # intrinsic scatter only from detections where the star is normally visible (envelope set by detections);
    # for a star below the plate limits the few detections are outliers or eruptions, not quiescent scatter
    vis = det & (env < lim)
    r = np.where(vis, mag - env, np.nan)
    s_int = _mad_sigma(r) if vis.sum() >= 20 else 0.3
    s_extra = np.sqrt(max(0.0, s_int ** 2 - np.median(err[vis]) ** 2)) if vis.sum() >= 20 else 0.0
    sig = np.sqrt(err ** 2 + s_extra ** 2)
    sat = g["sat"] & det
    return dict(t=t, mag=mag, det=det, err=err, lim=lim, env=env, sig=sig, s_int=s_int, m_faint=m_faint, sat=sat)


def _lndet(r, s, sat):
    """Detection log-likelihood; saturated points (A3) only say the source is at least this bright.

    r = m_obs - m_model. The model may be brighter than m_obs at no cost, and up to ~0.5 mag fainter at little
    cost; beyond that the saturated point rules it out: log Phi((r + 0.5) / 0.3).
    """
    return np.where(sat, log_ndtr((r + 0.5) / 0.3), _lnt(r, s))


def n_nights(t_yr) -> int:
    """Distinct observing nights: clusters of epochs separated by > 12 h (no dependence on site or UT day)."""
    if len(t_yr) == 0:
        return 0
    d = np.diff(np.sort(np.asarray(t_yr) * 365.25))
    return int(1 + (d > 0.5).sum())


def _masked(t0, masks):
    return any(a <= t0 <= b for a, b in masks)


def _pre_excess(t, mag, det, lim, t0, t_end, span_d=200.0):
    """Level before the event minus level after it (mag; positive = brighter before).

    Medians over [t0 - span_d, t0 - 5 d] and [t_end, t_end + span_d]; non-detections count as their limit.
    A nova returns to its pre-eruption level (~0). The end of a high state, which can resemble a decline,
    leaves the source fainter afterwards (> 0).
    """
    def level(w):
        if w.sum() < 3:
            return float("nan")
        return float(np.median(np.where(det[w], mag[w], lim[w])))
    pre = level((t >= t0 - span_d * DAY) & (t <= t0 - 5 * DAY))
    post = level((t >= t_end) & (t <= t_end + span_d * DAY))
    return post - pre


def drop_windows(lc, masks):
    """Remove every row inside the (t_lo, t_hi) masks (known eruptions) before a discovery scan."""
    keep = np.ones(len(lc["t"]), bool)
    for a, b in masks:
        keep &= ~((lc["t"] >= a) & (lc["t"] <= b))
    return {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in lc.items()}


def d1_scan(p, masks=(), allow=None, min_nights: int = MIN_NIGHTS) -> Event:
    """Best eruption hypothesis outside `masks` (list of (t_lo, t_hi) years) supported on >= `min_nights` nights.

    If `allow` = (t_lo, t_hi) is given, only t0 inside it are considered (recall test on known eruptions).
    """
    t, mag, det, lim, env, sig = p["t"], p["mag"], p["det"], p["lim"], p["env"], p["sig"]
    best = Event()
    excess = np.where(det & (mag < env - 0.75))[0]
    if len(excess) > MAX_ANCHORS:  # the brightest points fix the peak region; more anchors add little
        excess = excess[np.argsort(mag[excess] - env[excess])[:MAX_ANCHORS]]
    # quiescent level for a window: median plate value within +-ENV_HALF_D outside the window (flat across it)
    val = np.where(det, mag, np.maximum(np.nan_to_num(lim, nan=p["m_faint"]), p["m_faint"]))
    o = np.argsort(t)
    ts, vs = t[o], val[o]
    if len(excess) == 0:
        return best
    tried = set()
    for i in excess:
        for t2 in T2_GRID_D:
            for k in T0_OFFSETS:
                t0 = t[i] - k * t2 * DAY
                key = (round(t0 * 365.25), t2)
                if key in tried:
                    continue
                tried.add(key)
                if allow is not None and not (allow[0] <= t0 <= allow[1]):
                    continue
                if allow is None and _masked(t0, masks):
                    continue
                w = (t >= t0 - 3 * DAY) & (t <= t0 + 10 * t2 * DAY)
                idx = np.where(w)[0]
                if len(idx) == 0:
                    continue
                # envelope with the window's own detections excluded
                a0 = np.searchsorted(ts, t0 - 3 * DAY - ENV_HALF_D * DAY)
                a1 = np.searchsorted(ts, t0 - 3 * DAY)
                b0 = np.searchsorted(ts, t0 + 10 * t2 * DAY, side="right")
                b1 = np.searchsorted(ts, t0 + 10 * t2 * DAY + ENV_HALF_D * DAY, side="right")
                side = np.concatenate([vs[a0:a1], vs[b0:b1]])
                env_w = np.full(len(idx), np.median(side) if len(side) >= 5 else p["m_faint"])
                dt = (t[idx] - t0) * 365.25
                f = template_flux(dt, t2)
                dw, mw, lw, sw, satw = det[idx], mag[idx], lim[idx], sig[idx], p["sat"][idx]
                brightest = np.nanmin(np.where(dw, mw, np.inf))
                if not np.isfinite(brightest):
                    continue
                peaks = np.arange(brightest - 2.0, min(brightest + 3.0, np.median(env_w) - MIN_AMP), 0.1)
                if len(peaks) == 0:
                    continue
                fq = 10 ** (-0.4 * env_w)
                fn = 10 ** (-0.4 * peaks)[:, None] * f[None, :]
                m_mod = -2.5 * np.log10(fq[None, :] + fn)
                l1 = np.where(dw[None, :], _lndet(mw[None, :] - m_mod, sw[None, :], satw[None, :]), _lnlim(m_mod, lw[None, :]))
                # A5: the null is the better of quiescence and a slow state change (the source held at a free
                # constant level over the same window), so high/low states of symbiotics and CVs do not score
                l0 = np.where(dw, _lndet(mw - env_w, sw, satw), _lnlim(env_w, lw))
                levels = np.arange(brightest - 0.5, np.median(env_w) + 0.51, 0.1)
                m_box = np.broadcast_to(levels[:, None], (len(levels), len(idx)))
                lb = np.where(dw[None, :], _lndet(mw[None, :] - m_box, sw[None, :], satw[None, :]), _lnlim(m_box, lw[None, :]))
                l_null = max(np.nan_to_num(l0, nan=0.0).sum(), np.nan_to_num(lb, nan=0.0).sum(axis=1).max())
                d = np.nan_to_num(l1, nan=0.0).sum(axis=1) - l_null
                j = int(np.argmax(d))
                sc = 2 * d[j]
                if sc > best.score:
                    mm = m_mod[j]
                    support = dw & (mm < env_w - 0.5) & (np.abs(mw - mm) < 3 * sw + 0.5)
                    n_n = n_nights(t[idx][support])
                    if n_n < min_nights:
                        continue
                    contra = (~dw) & (mm < lw - 0.5)
                    best = Event(score=float(sc), t0=float(t0), t2_d=t2, m_peak=float(peaks[j]),
                                 n_nights=n_n, n_plates=int(support.sum()), n_contra=int(contra.sum()),
                                 extra={"amp": float(np.median(env_w) - peaks[j]),
                                        "pre_excess": _pre_excess(t, mag, det, lim, t0, t0 + 10 * t2 * DAY)})
    return best


def d0_scan(p, masks=(), allow=None, gap_d: float = 200.0) -> Event:
    """Naive detector: >= 2 nights with points >= max(1 mag, 5 sigma_MAD) above the envelope within gap_d."""
    t, mag, det, env = p["t"], p["mag"], p["det"], p["env"]
    thr = max(1.0, 5 * p["s_int"])
    ex = np.where(det & (mag < env - thr))[0]
    best = Event()
    if len(ex) < 2:
        return best
    tt = t[ex]
    # clusters by gap
    cuts = np.where(np.diff(tt) * 365.25 > gap_d)[0]
    for grp in np.split(np.arange(len(ex)), cuts + 1):
        ii = ex[grp]
        t0 = t[ii].min()
        if allow is not None and not (allow[0] <= t0 <= allow[1]):
            continue
        if allow is None and _masked(t0, masks):
            continue
        nights = n_nights(t[ii])
        if nights < 2:
            continue
        exc = np.sort(env[ii] - mag[ii])[::-1]
        sc = float(exc[1])
        if sc > best.score:
            best = Event(score=sc, t0=float(t0), n_nights=nights, n_plates=len(ii),
                         m_peak=float(mag[ii].min()))
    return best


def mean_level(lc, t_lo, t_hi):
    """Mean, SEM and N of good detections in [t_lo, t_hi) (for the G0 gate)."""
    g = _good(lc)
    m = g["det"] & (g["t"] >= t_lo) & (g["t"] < t_hi)
    x = g["mag"][m]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int(len(x))
