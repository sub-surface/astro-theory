r"""
=============================================================================
T Coronae Borealis: Panchromatic Accretion History & Eruption Clock (EXP-2026-Y)
=============================================================================
Empirical, calibrated reconstruction of the accretion history of the symbiotic
recurrent nova T CrB (eruptions 1866, 1946; next one pending) from public data:

  * AAVSO U/B/V/visual photometry (sampled via the v2 API, scripts/fetch_tcrb_aavso.py)
  * Swift/XRT per-ObsID light curves from LSXPS (Evans et al. 2023)
  * Swift/UVOT UV sky images (photometry here, no HEASoft needed)
  * Gaia DR3 epoch photometry (2014.6-2017.2; SAP onset)

Physics:
1. Orbital ephemeris & ellipsoidal light: the M3III donor fills its Roche lobe
   (Munari+ 2025), so its light is modulated at P_orb/2 with unequal minima. We fit
   P, T0 and harmonics jointly with a smooth secular trend (no external ephemeris).
   Phase convention here: phi = 0 at the deeper-light minimum (giant in front).
2. Viscous disk (Lynden-Bell & Pringle 1974) with prescribed nu(R) = nu_out (R/R_out)^n:
       dSigma/dt = (3/R) d/dR [ R^{1/2} d/dR (nu Sigma R^{1/2}) ] + S(R, t)
   Linear in Sigma, hence an LTI system: Mdot_acc(t) = (g * Mdot_tr)(t). We solve it
   implicitly once for a unit pulse and build any transfer history by superposition.
   Local effective temperature (per face): sigma T^4 = (9/8) nu Sigma Omega^2.
3. Boundary layer: L_BL ~ G M Mdot / (2 R_WD). Optically thin (hard X-rays) at low
   Mdot, optically thick (EUV, X-ray suppressed) at high Mdot (Patterson & Raymond
   1985). The XRT light curve is therefore an *independent* check on Mdot_acc(t).
4. Eruption clock: TNR ignites once the accreted envelope reaches M_ign(M_WD, Mdot).

All uncertainties are carried explicitly (bootstrap over data + systematic grid over
distance, extinction, inclination, M_WD); no naked point estimates are reported.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (cgs) and system parameters
# ---------------------------------------------------------------------------
G = 6.674e-8
MSUN = 1.989e33
RSUN = 6.957e10
SIGMA_SB = 5.6704e-5
H_PLANCK = 6.62607e-27
C_LIGHT = 2.99792458e10
K_B = 1.380649e-16
PC = 3.0857e18
YR = 3.15576e7
DAY = 86400.0

RA_DEG, DEC_DEG = 239.875667, 25.920194

ERUPTIONS_JD = {  # optical peaks / discovery
    "1787": 2374081.5,   # 1787-11-28 (Schaefer 2023; Wollaston)
    "1866": 2402733.5,   # 1866-05-12
    "1946": 2431860.5,   # 1946-02-09
}


@dataclass(frozen=True)
class SystemParams:
    """Fiducial T CrB parameters with 1-sigma systematic ranges."""
    m_wd: float = 1.35            # Msun (Munari+ 2025; Schaefer+ 2025: 1.37)
    m_giant: float = 0.93         # Msun
    incl_deg: float = 61.0        # Munari+ 2025
    dist_pc: float = 887.0        # Gaia DR3 geometric (Bailer-Jones+ 2021), 858-905
    ebv: float = 0.07             # adopted; literature spans ~0.05-0.15
    p_orb: float = 227.5528       # d (Munari+ 2025); fitted independently below
    r_out_rsun: float = 58.0      # disk outer radius from hot-spot azimuth (Munari+ 2025)

    @property
    def r_wd(self) -> float:
        """Nauenberg (1972) mass-radius relation (cm)."""
        mch = 1.44
        x = (mch / self.m_wd) ** (2 / 3) - (self.m_wd / mch) ** (2 / 3)
        return 0.0112 * RSUN * math.sqrt(max(x, 1e-6)) * 1.0

    @property
    def a_orb(self) -> float:
        p = self.p_orb * DAY
        return (G * (self.m_wd + self.m_giant) * MSUN * p ** 2 / (4 * math.pi ** 2)) ** (1 / 3)

    def roche_lobe_eggleton(self, q: float) -> float:
        """Eggleton (1983) r_L/a for mass ratio q = M_donor/M_accretor."""
        q23 = q ** (2 / 3)
        return 0.49 * q23 / (0.6 * q23 + math.log1p(q ** (1 / 3)))


FIDUCIAL = SystemParams()

# Cardelli+ (1989) A_lambda / E(B-V) at R_V = 3.1; UVOT values incl. the 2175 A bump.
EXT_COEFF = {"U": 4.75, "B": 4.10, "V": 3.10, "UVW1": 6.50, "UVM2": 8.90, "UVW2": 8.00,
             "G": 2.74, "BP": 3.37, "RP": 2.04}
# Effective wavelengths (A) and Vega zero-point flux densities (erg/s/cm^2/A; Bessell 1998,
# Breeveld+ 2011 for UVOT).
LAMBDA_EFF = {"U": 3600.0, "B": 4380.0, "V": 5450.0, "UVW1": 2600.0, "UVM2": 2246.0, "UVW2": 1928.0}
F_LAMBDA_ZP = {"U": 4.18e-9, "B": 6.32e-9, "V": 3.63e-9}
# UVOT count-rate -> flux density (erg/s/cm^2/A per c/s; Breeveld+ 2011) and Vega ZP mags.
UVOT_FLUX_CONV = {"UVW1": 4.21e-16, "UVM2": 8.45e-16, "UVW2": 5.98e-16}
UVOT_ZP = {"UVW1": 17.44, "UVM2": 16.85, "UVW2": 17.38}
AAVSO_BANDS = {"0": "Vis", "2": "V", "3": "B", "7": "U"}
# Intrinsic M3III colours (Johnson; Bessell & Brett 1988 / Lee 1970).
GIANT_COLORS = {"B-V": 1.60, "U-B": 1.89}


def jd_to_year(jd):
    return 2000.0 + (np.asarray(jd, float) - 2451545.0) / 365.25


def year_to_jd(yr):
    return 2451545.0 + (np.asarray(yr, float) - 2000.0) * 365.25


def mag_to_flam(mag, band: str, ebv: float = 0.0):
    """Vega mag -> dereddened F_lambda (erg/s/cm^2/A)."""
    m0 = np.asarray(mag, float) - EXT_COEFF[band] * ebv
    if band in F_LAMBDA_ZP:
        return F_LAMBDA_ZP[band] * 10 ** (-0.4 * m0)
    return UVOT_FLUX_CONV[band] * 10 ** (-0.4 * (m0 - UVOT_ZP[band]))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_aavso_sampled(path: Path) -> Dict[str, np.ndarray]:
    """Read the sampled AAVSO JSONL, de-duplicate by observation id, drop limits."""
    seen, rows = set(), []
    effort = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        effort.append((rec["band"], rec["start"], rec["end"], rec["count"]))
        for r in rec["results"]:
            if r["id"] in seen or r.get("fainterthan"):
                continue
            seen.add(r["id"])
            try:
                mag = float(r["magnitude"])
            except (TypeError, ValueError):
                continue
            try:
                err = float(r["uncertainty"]) if r["uncertainty"] not in (None, "") else np.nan
            except ValueError:
                err = np.nan
            rows.append((r["jd_dbl"], mag, err, AAVSO_BANDS.get(r["band"], r["band"]), r["obscode"], r["obstype"]))
    if not rows:
        return {k: np.array([]) for k in ("jd", "mag", "err", "band", "obscode", "obstype")}
    jd, mag, err, band, obs, ot = map(np.array, zip(*rows))
    o = np.argsort(jd)
    return {"jd": jd[o].astype(float), "mag": mag[o].astype(float), "err": err[o].astype(float),
            "band": band[o], "obscode": obs[o], "obstype": ot[o], "effort": effort}


def nightly_bins(jd, mag, err, min_err: float = 0.02, clip: float = 0.5) -> Dict[str, np.ndarray]:
    """Robust nightly medians. Error = max(MAD-scatter/sqrt(n), quoted/sqrt(n), floor)."""
    jd, mag, err = np.asarray(jd, float), np.asarray(mag, float), np.asarray(err, float)
    fin = np.isfinite(jd) & np.isfinite(mag)
    jd, mag, err = jd[fin], mag[fin], err[fin]
    night = np.floor(jd).astype(int)  # integer JD boundary = noon UT
    out_t, out_m, out_e, out_n = [], [], [], []
    for n in np.unique(night):
        s = night == n
        m = mag[s]
        med = np.median(m)
        keep = np.abs(m - med) < clip
        if not keep.any():  # e.g. two discrepant points straddling the median
            keep[:] = True
        m = m[keep]
        e = err[s][keep]
        k = len(m)
        mad = 1.4826 * np.median(np.abs(m - np.median(m))) if k > 2 else 0.0
        quoted = np.nanmedian(e) if np.isfinite(e).any() else 0.05
        out_t.append(np.mean(jd[s][keep]))
        out_m.append(np.median(m))
        out_e.append(max(mad / math.sqrt(k), quoted / math.sqrt(k), min_err))
        out_n.append(k)
    return {"jd": np.array(out_t), "mag": np.array(out_m), "err": np.array(out_e), "n": np.array(out_n)}


def load_lsxps(directory: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """LSXPS per-ObsID light curves saved as CSV by scripts/fetch_tcrb_swift_xrt.py."""
    import pandas as pd
    directory = Path(directory)
    out = {}
    for band in ("Total", "Soft", "Medium", "Hard"):
        r = pd.read_csv(directory / f"{band}_rates.csv")
        u_path = directory / f"{band}_UL.csv"
        u = pd.read_csv(u_path) if u_path.exists() else None
        has_u = u is not None and len(u)
        out[band] = {
            "jd": r["Time"].to_numpy() + 2400000.5, "rate": r["Rate"].to_numpy(),
            "err": 0.5 * (r["RatePos"].to_numpy() - r["RateNeg"].to_numpy()),
            "ul_jd": u["Time"].to_numpy() + 2400000.5 if has_u else np.array([]),
            "ul": u["UpperLimit"].to_numpy() if has_u else np.array([]),
        }
    return out


def load_gaia_epoch(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    from astropy.table import Table
    t = Table.read(path)
    out = {}
    for b, tcol, mcol, fcol, ecol in (("G", "g_transit_time", "g_transit_mag", "g_transit_flux", "g_transit_flux_error"),
                                      ("BP", "bp_obs_time", "bp_mag", "bp_flux", "bp_flux_error"),
                                      ("RP", "rp_obs_time", "rp_mag", "rp_flux", "rp_flux_error")):
        tt = np.ma.filled(t[tcol].astype(float), np.nan)
        mm = np.ma.filled(t[mcol].astype(float), np.nan)
        ff = np.ma.filled(t[fcol].astype(float), np.nan)
        ee = np.ma.filled(t[ecol].astype(float), np.nan)
        ok = np.isfinite(tt) & np.isfinite(mm)
        out[b] = {"jd": tt[ok] + 2455197.5, "mag": mm[ok], "err": 1.0857 * ee[ok] / ff[ok]}
    return out


# ---------------------------------------------------------------------------
# Swift/UVOT aperture photometry (no HEASoft)
# ---------------------------------------------------------------------------
UVOT_FRAMETIME_DEFAULT = 0.0110322
COI_POLY = (0.066, -0.091, 0.029, 0.031)  # Poole+ 2008


def coincidence_correct(raw_rate, frametime: float = UVOT_FRAMETIME_DEFAULT, deadtime_frac: float = 0.0158):
    """Poole+ (2008) coincidence-loss correction for a 5'' aperture.

    Returns (corrected rate, saturation flag). Saturated when raw*frametime >= 0.97.
    """
    raw = np.asarray(raw_rate, float)
    x = raw * frametime
    sat = x >= 0.97
    xc = np.clip(x, 0, 0.97)
    alpha = 1 - deadtime_frac
    theory = -np.log1p(-alpha * xc) / (alpha * frametime)
    a1, a2, a3, a4 = COI_POLY
    poly = 1 + a1 * xc + a2 * xc ** 2 + a3 * xc ** 3 + a4 * xc ** 4
    return theory * poly, sat


def uvot_aperture_photometry(path: Path, ra: float = RA_DEG, dec: float = DEC_DEG,
                             r_src: float = 5.0, r_bkg: Tuple[float, float] = (27.5, 45.0),
                             r_wing: Tuple[float, float] = (10.0, 20.0)) -> List[dict]:
    """Photometry of every snapshot extension in a UVOT *_sk.img(.gz) file."""
    from astropy.io import fits
    from astropy.wcs import WCS

    out = []
    with fits.open(path) as hdul:
        filt = None
        for hdu in hdul[1:]:
            h = hdu.header
            if hdu.data is None or h.get("EXPOSURE", 0) <= 0:
                continue
            filt = h.get("FILTER", filt)
            w = WCS(h)
            x0, y0 = w.all_world2pix([[ra, dec]], 0)[0]
            ny, nx = hdu.data.shape
            if not (20 < x0 < nx - 20 and 20 < y0 < ny - 20):
                continue
            pix = abs(h.get("CDELT1", 2.7889e-4)) * 3600.0
            half = int(r_bkg[1] / pix) + 3
            y_lo, x_lo = max(int(y0) - half, 0), max(int(x0) - half, 0)
            img = hdu.data[y_lo:int(y0) + half, x_lo:int(x0) + half].astype(float)
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            r = np.hypot(xx + x_lo - x0, yy + y_lo - y0) * pix
            src = r <= r_src
            bkg = (r >= r_bkg[0]) & (r <= r_bkg[1]) & np.isfinite(img)
            if src.sum() == 0 or bkg.sum() < 20:
                continue
            exp = float(h["EXPOSURE"])
            ft = float(h.get("FRAMTIME", UVOT_FRAMETIME_DEFAULT))
            # dead-time-corrected EXPOSURE already; raw rates per aperture
            b_per_pix = np.nanmedian(img[bkg])
            src_counts = np.nansum(img[src])
            raw_tot = src_counts / exp
            raw_bkg = b_per_pix * src.sum() / exp
            tot_c, sat = coincidence_correct(raw_tot, ft)
            bkg_c, _ = coincidence_correct(raw_bkg, ft)
            net = float(tot_c - bkg_c)
            err = math.sqrt(max(src_counts, 1.0)) / exp * (tot_c / max(raw_tot, 1e-9))
            # PSF-wing annulus: escapes coincidence saturation; calibrated later against
            # unsaturated epochs (wing/core ratio) to recover saturated high-state epochs.
            wing = (r >= r_wing[0]) & (r <= r_wing[1]) & np.isfinite(img)
            wing_rate = (np.nansum(img[wing]) - b_per_pix * wing.sum()) / exp
            wing_err = math.sqrt(max(np.nansum(img[wing]), 1.0)) / exp
            tmid = 0.5 * (h["TSTART"] + h["TSTOP"])
            out.append({"file": Path(path).name, "filter": str(filt).upper(), "met": tmid, "exposure": exp,
                        "raw_rate": raw_tot, "rate": net, "rate_err": err, "saturated": bool(sat),
                        "wing_rate": float(wing_rate), "wing_err": float(wing_err), "frametime": ft})
    return out


def swift_met_to_jd(met):
    """Swift MET -> JD (UTC), ignoring the ~few-s UTCF correction (irrelevant here)."""
    return 2451910.5 + np.asarray(met, float) / DAY


def uvot_rate_to_mag(rate, band: str, year: Optional[np.ndarray] = None, decline_per_yr: float = 0.01):
    """Vega magnitude with linear sensitivity-loss correction (~1%/yr, Breeveld CALDB)."""
    rate = np.asarray(rate, float)
    corr = 1.0
    if year is not None:
        corr = (1 - decline_per_yr) ** (np.asarray(year) - 2005.0)
    return UVOT_ZP[band] - 2.5 * np.log10(np.clip(rate / corr, 1e-6, None))


# ---------------------------------------------------------------------------
# Orbital ephemeris from ellipsoidal light
# ---------------------------------------------------------------------------
def _design(t, period, t_ref, trend_knots, n_harm=2):
    """Linear design matrix: harmonic orbital terms + cubic B-spline secular trend."""
    from scipy.interpolate import BSpline
    phase = 2 * np.pi * (t - t_ref) / period
    cols = []
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * phase), np.sin(k * phase)]
    deg = 3
    kn = np.concatenate([[trend_knots[0]] * deg, trend_knots, [trend_knots[-1]] * deg])
    nb = len(kn) - deg - 1
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        cols.append(BSpline(kn, c, deg, extrapolate=False)(t))
    X = np.nan_to_num(np.vstack(cols).T)
    return X


def fit_ephemeris(t, mag, err, periods, knot_spacing_d: float = 120.0, n_harm: int = 2):
    """Profile chi^2 over trial periods with a free secular spline (weighted LSQ)."""
    t = np.asarray(t, float)
    t_ref = 2455828.9
    knots = np.arange(t.min(), t.max() + knot_spacing_d, knot_spacing_d)
    w = 1.0 / np.asarray(err, float) ** 2
    chi2 = []
    for p in periods:
        X = _design(t, p, t_ref, knots, n_harm)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], mag * sw, rcond=None)
        chi2.append(np.sum(w * (mag - X @ beta) ** 2))
    chi2 = np.array(chi2)
    i = int(np.argmin(chi2))
    p_best = periods[i]
    X = _design(t, p_best, t_ref, knots, n_harm)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], mag * sw, rcond=None)
    resid = mag - X @ beta
    harm = beta[: 2 * n_harm]
    # secular-trend-only model (for detrending)
    trend = X[:, 2 * n_harm:] @ beta[2 * n_harm:]
    return {"periods": np.asarray(periods), "chi2": chi2, "p_best": p_best, "harm": harm, "t_ref": t_ref,
            "resid": resid, "trend": trend, "dof": len(t) - X.shape[1]}


def ellipsoidal_minimum_epoch(harm, period, t_ref):
    """Epoch of the deeper light minimum (max magnitude) of the harmonic model."""
    ph = np.linspace(0, 1, 20001)
    a = 2 * np.pi * ph
    m = harm[0] * np.cos(a) + harm[1] * np.sin(a) + harm[2] * np.cos(2 * a) + harm[3] * np.sin(2 * a)
    return t_ref + ph[np.argmax(m)] * period


# ---------------------------------------------------------------------------
# Time-dependent viscous accretion disk (linear, implicit, LTI superposition)
# ---------------------------------------------------------------------------
@dataclass
class DiskGrid:
    r_in: float
    r_out: float
    n: int = 160

    def __post_init__(self):
        self.edges = np.geomspace(self.r_in, self.r_out, self.n + 1)
        self.r = np.sqrt(self.edges[:-1] * self.edges[1:])
        self.area = np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)  # one face


def viscosity_profile(grid: DiskGrid, t_visc_out_yr: float, n_exp: float):
    """nu(R) = nu_out (R/R_out)^n with nu_out set by the outer viscous time R_out^2/nu_out."""
    nu_out = grid.r_out ** 2 / (t_visc_out_yr * YR)
    return nu_out * (grid.r / grid.r_out) ** n_exp


def _operator(grid: DiskGrid, nu: np.ndarray):
    """Finite-volume matrix L with d(Sigma)/dt = L @ Sigma, and inner-flux row vector.

    Sigma is the surface density summed over both faces; cell mass = Sigma * A_i with
    A_i = pi (R_{i+1/2}^2 - R_{i-1/2}^2). Inward mass flux through interface R_j:
        Mdot_j = 6 pi R_j^{1/2} [ (nu Sigma R^{1/2})_j - (nu Sigma R^{1/2})_{j-1} ] / (R_j - R_{j-1})
    Inner boundary: Sigma = 0 at R_in (zero torque). Outer boundary: Mdot = 0 (tidal truncation).
    dM_j/dt = Mdot_{j+1} - Mdot_j.
    """
    n = grid.n
    r, e, A = grid.r, grid.edges, grid.area
    f = nu * np.sqrt(r)
    flux = np.zeros((n + 1, n))  # Mdot_j = flux[j] @ Sigma, j = 0..n
    for j in range(1, n):
        c = 6 * np.pi * math.sqrt(e[j]) / (r[j] - r[j - 1])
        flux[j, j] = c * f[j]
        flux[j, j - 1] = -c * f[j - 1]
    c0 = 6 * np.pi * math.sqrt(e[0]) / (r[0] - e[0])
    flux[0, 0] = c0 * f[0]
    L = (flux[1:] - flux[:-1]) / A[:, None]
    return L, flux[0]


class ViscousDisk:
    """Implicit (backward-Euler) viscous disk; linear in Sigma, so responses superpose (LTI)."""

    def __init__(self, params: SystemParams = FIDUCIAL, t_visc_out_yr: float = 3.0, n_exp: float = 1.0,
                 r_in: Optional[float] = None, inject_frac: float = 0.85, n_cells: int = 160):
        self.p = params
        self.grid = DiskGrid(r_in or params.r_wd, params.r_out_rsun * RSUN, n_cells)
        self.nu = viscosity_profile(self.grid, t_visc_out_yr, n_exp)
        self.L, self.inner_flux = _operator(self.grid, self.nu)
        r_inj = inject_frac * self.grid.r_out
        w = np.exp(-0.5 * ((np.log(self.grid.r) - math.log(r_inj)) / 0.05) ** 2)
        self.inject = w / np.sum(w * self.grid.area)  # dSigma/dt per (1 g/s) of transfer
        self.omega = np.sqrt(G * params.m_wd * MSUN / self.grid.r ** 3)

    def mdot_inner(self, sigma):
        """Accretion rate onto the WD (g/s); sigma shape (n,) or (T, n)."""
        return np.asarray(sigma) @ self.inner_flux

    def pulse_response(self, dt_d: float, n_steps: int, pulse_d: float):
        """Sigma(t, R) after a boxcar of unit (1 g/s) mass transfer lasting pulse_d days."""
        from scipy.linalg import lu_factor, lu_solve
        dt = dt_d * DAY
        lu = lu_factor(np.eye(self.grid.n) - dt * self.L)
        s = np.zeros(self.grid.n)
        out = np.zeros((n_steps, self.grid.n))
        n_on = max(1, int(round(pulse_d / dt_d)))
        for k in range(n_steps):
            s = lu_solve(lu, s + (dt * self.inject if k < n_on else 0.0))
            out[k] = s
        return out

    def steady_state(self):
        """Sigma for a steady 1 g/s transfer: L Sigma = -inject."""
        return np.linalg.solve(self.L, -self.inject)

    def teff(self, sigma):
        """Viscous effective temperature per face: sigma_SB T^4 = (9/8) nu Sigma Omega^2."""
        d = (9.0 / 8.0) * self.nu * np.clip(sigma, 0, None) * self.omega ** 2
        return (d / SIGMA_SB) ** 0.25

    def band_flux(self, sigma, band_lambda_A: float, tirr4: Optional[np.ndarray] = None):
        """Observed (unreddened) F_lambda in erg/s/cm^2/A of blackbody annuli seen at inclination i."""
        t4 = self.teff(sigma) ** 4
        if tirr4 is not None:
            t4 = t4 + tirr4
        T = np.clip(t4, 1.0, None) ** 0.25
        lam = band_lambda_A * 1e-8
        x = H_PLANCK * C_LIGHT / (lam * K_B * T)
        b = 2 * H_PLANCK * C_LIGHT ** 2 / lam ** 5 / np.expm1(np.clip(x, 1e-6, 700))
        d = self.p.dist_pc * PC
        cosi = math.cos(math.radians(self.p.incl_deg))
        return (b * self.grid.area).sum(axis=-1) * cosi / d ** 2 * 1e-8


# ---------------------------------------------------------------------------
# Recurrence statistics
# ---------------------------------------------------------------------------
def recurrence_predictive(intervals_yr: Sequence[float], t_last: float, t_now: float, n: int = 2_000_000,
                          seed: int = 1) -> Dict[str, float]:
    """Log-normal recurrence with Jeffreys prior -> Student-t predictive, survival-conditioned."""
    rng = np.random.default_rng(seed)
    l = np.log(np.asarray(intervals_yr, float))
    k = len(l)
    m, s = l.mean(), l.std(ddof=1)
    draws = t_last + np.exp(m + s * math.sqrt(1 + 1 / k) * rng.standard_t(k - 1, n))
    alive = draws > t_now
    tc = draws[alive]
    return {"p_survive_to_now": float(alive.mean()), "median": float(np.median(tc)),
            "q16": float(np.percentile(tc, 16)), "q84": float(np.percentile(tc, 84)),
            "samples": tc}


def orbital_phase(jd, t0: float, period: float):
    return np.mod((np.asarray(jd, float) - t0) / period, 1.0)
