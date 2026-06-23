#!/usr/bin/env python3
"""
Milestone 4 — CatWISE2020 quasar dipole through the SAME pipeline as Quaia.

Two estimators, identical to the Quaia analysis:
  (a) raw linear LS dipole on counts (|b|>30)  -> validate vs Secrest+2021
      (D=0.01554, (l,b)=(238.2,28.8), 27.8deg off CMB);
  (b) PRINCIPLED Poisson forward model with CatWISE's "selection function" = the
      ecliptic-latitude density trend (Secrest's documented systematic), fit from OUR
      data as s_p = (a + b|elat_p|)/mean, then mu_p = s_p*N*(1 + A.nhat_p).
Plus the isotropic-mock null (noise floor + significance), exactly as for Quaia.

x measured from our W1 counts; alpha adopted from the WISE literature (Secrest mean 1.26;
2025 kinematic paper 1.07) -> D_kin ~ 0.007. Writes results JSON for the figure script.

Gentle: single-threaded, Nside=64, reads the stripe CSV checkpoints.
"""
import os, glob, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import ascii as asciirw
from astropy.table import vstack
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize

NSIDE = 64
SRC = "data/catwise/"
BCUT = int(os.environ.get("BCUT", "30"))
NMOCK = int(os.environ.get("NMOCK", "100"))
V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
ALPHA_W = 1.26          # Secrest+2021 mean W1 spectral index
rng = np.random.default_rng(11)

hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
cb = np.cos(lat.rad)
XYZ = np.column_stack([cb * np.cos(lon.rad), cb * np.sin(lon.rad), np.sin(lat.rad)])
_gal = SkyCoord(lon, lat, frame="icrs").galactic
PIX_B = _gal.b.deg
PIX_L = _gal.l.deg
CMB = SkyCoord(L_CMB * u.deg, B_CMB * u.deg, frame="galactic")

# Secrest+2021 mask the Magellanic Clouds & Andromeda (their 291 regions); the LMC/SMC
# survive a |b|>30 cut and inject a large spurious southern overdensity. Mask them as
# galactic cones. (l,b,radius_deg)
EXTRA_MASKS = [
    (280.5, -32.9, 9.0),   # LMC
    (302.8, -44.3, 7.0),   # SMC
]


def extra_mask():
    keep = np.ones(npix, bool)
    pg = SkyCoord(PIX_L * u.deg, PIX_B * u.deg, frame="galactic")
    for l0, b0, r in EXTRA_MASKS:
        c = SkyCoord(l0 * u.deg, b0 * u.deg, frame="galactic")
        keep &= pg.separation(c).deg > r
    return keep


def load():
    files = sorted(glob.glob(os.path.join(SRC, "cw_ra*.csv")))
    tabs = [asciirw.read(f, format="csv") for f in files]
    t = vstack(tabs)
    return t


def lstsq_dipole(counts, good):
    X = np.column_stack([np.ones(npix), XYZ])
    coef, *_ = np.linalg.lstsq(X[good], counts[good], rcond=None)
    return np.linalg.norm(coef[1:]) / coef[0], coef[1:] / np.linalg.norm(coef[1:])


def poisson_dipole(counts, s, good):
    n = XYZ[good]; k = counts[good]; sg = s[good]
    N0 = k.sum() / sg.sum()

    def nll(th):
        mu = sg * np.exp(th[0]) * (1.0 + n @ th[1:])
        if np.any(mu <= 0): return 1e12
        return -np.sum(k * np.log(mu) - mu)

    r = minimize(nll, [np.log(N0), 0, 0, 0], method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-4, "maxiter": 20000})
    A = r.x[1:]
    return np.linalg.norm(A), A / max(np.linalg.norm(A), 1e-12)


def direction(dvec):
    dlon = np.degrees(np.arctan2(dvec[1], dvec[0])) % 360
    dlat = np.degrees(np.arcsin(dvec[2]))
    eq = SkyCoord(dlon * u.deg, dlat * u.deg, frame="icrs")
    g = eq.galactic
    return eq.ra.deg, eq.dec.deg, g.l.deg, g.b.deg, g.separation(CMB).deg


def measure_x(w1, lo=15.0, hi=16.3, nbin=14):
    edges = np.linspace(lo, hi, nbin + 1)
    cnt, _ = np.histogram(w1[(w1 >= lo) & (w1 < hi)], bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt > 0
    A = np.column_stack([np.ones(ok.sum()), ctr[ok]])
    w = cnt[ok] * np.log(10) ** 2
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (np.log10(cnt[ok]) * w))
    return 2.5 * coef[1], 2.5 * np.sqrt(cov[1, 1])


def main():
    t = load()
    ra = np.asarray(t["ra"], float); dec = np.asarray(t["dec"], float)
    w1 = np.asarray(t["w1mpro"], float); w2 = np.asarray(t["w2mpro"], float)
    elat = np.asarray(t["elat"], float)
    print(f"loaded {len(ra):,} CatWISE quasar candidates (all-sky, pre-mask)")

    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    counts = np.bincount(pix, minlength=npix).astype(float)
    # mean |ecliptic latitude| per pixel (for the selection model)
    abel = np.abs(elat)
    sum_el = np.bincount(pix, weights=abel, minlength=npix)
    n_el = np.bincount(pix, minlength=npix)
    pix_abel = np.divide(sum_el, n_el, out=np.full(npix, np.nan), where=n_el > 0)

    emask = extra_mask()
    base = (np.abs(PIX_B) >= BCUT) & (counts > 0) & np.isfinite(pix_abel) & emask
    # automatic hot-pixel mask: robust 6-sigma clip on counts removes localized
    # contamination (resolved galaxies, bright-star artifacts) -- reproduces what
    # Secrest+2021 did with 291 hand-drawn masks. Iterate to convergence.
    good = base.copy()
    for _ in range(6):
        med = np.median(counts[good]); mad = np.median(np.abs(counts[good] - med)) * 1.4826
        new = good & (counts <= med + 6 * mad)
        if new.sum() == good.sum():
            break
        good = new
    n_lmc = counts[(np.abs(PIX_B) >= BCUT) & ~emask].sum()
    n_hot = counts[base & ~good].sum()
    print(f"|b|>{BCUT} + LMC/SMC + hot-pixel(6sig) masks: {good.sum()} pixels, "
          f"{counts[good].sum():,.0f} sources, fsky={good.mean():.2f}")
    print(f"  removed {n_lmc:,.0f} in Magellanic cones, {n_hot:,.0f} in {(base&~good).sum()} hot pixels")

    # --- x and D_kin ---
    # measure x near the flux limit (von Hausegger 2024: dipole dominated by limit sources)
    x, xerr = measure_x(w1, lo=15.5, hi=16.3)
    Dk_meas = (2 + x * (1 + ALPHA_W)) * BETA
    # literature kinematic expectation (2025 kinematic paper: x=1.90, alpha=1.07; Secrest ~0.007)
    Dk = (2 + 1.90 * (1 + 1.07)) * BETA
    print(f"x(W1, near-limit)={x:.3f}+/-{xerr:.3f} -> D_kin(our x,a={ALPHA_W})={Dk_meas:.4f}; "
          f"D_kin(lit x=1.90,a=1.07)={Dk:.4f} [used]")

    # --- (a) raw LS dipole (validate vs Secrest) ---
    amp_r, dv_r = lstsq_dipole(counts, good)
    ra0, dec0, l0, b0, sep0 = direction(dv_r)
    print(f"\n[raw-LS]   D={amp_r:.5f}  (l,b)=({l0:.0f},{b0:+.0f})  offCMB={sep0:.1f}  "
          f"[Secrest: 0.01554 @ (238,29), 27.8]")

    # --- CatWISE selection model: density vs |ecliptic latitude| (Secrest systematic) ---
    A2 = np.column_stack([np.ones(good.sum()), pix_abel[good]])
    cfit, *_ = np.linalg.lstsq(A2, counts[good], rcond=None)
    trend = cfit[0] + cfit[1] * pix_abel
    s = np.clip(trend / np.nanmean(trend[good]), 1e-3, None)   # normalised selection fn
    print(f"ecliptic trend: density = {cfit[0]:.2f} + {cfit[1]:+.4f}*|elat|  "
          f"(Secrest: 68.89 -0.051*|elat|, different pixelisation)")

    # --- (b) principled Poisson forward model ---
    amp_p, dv_p = poisson_dipole(counts, s, good)
    ra1, dec1, l1, b1, sep1 = direction(dv_p)
    print(f"[poisson]  D={amp_p:.5f}  (l,b)=({l1:.0f},{b1:+.0f})  offCMB={sep1:.1f}  "
          f"D/Dkin={amp_p/Dk:.2f}")

    # --- isotropic-mock null (noise floor + significance) ---
    Nbar = counts[good].sum() / s[good].sum()
    exp = s * Nbar
    Dn = np.empty(NMOCK)
    for i in range(NMOCK):
        c = np.zeros(npix); c[good] = rng.poisson(exp[good])
        Dn[i], _ = poisson_dipole(c, s, good)
    sig = (amp_p - Dn.mean()) / Dn.std()
    print(f"[null]     <D>={Dn.mean():.5f}+/-{Dn.std():.5f}  ->  D_data is {sig:.1f}sigma "
          f"above the isotropic floor")

    out = dict(catalog="CatWISE2020", n=int(counts[good].sum()), bcut=BCUT,
               x=float(x), xerr=float(xerr), alpha=ALPHA_W,
               D_kin=float(Dk), D_kin_measured=float(Dk_meas),
               raw_D=float(amp_r), raw_l=float(l0), raw_b=float(b0), raw_offCMB=float(sep0),
               D=float(amp_p), l=float(l1), b=float(b1), offCMB=float(sep1),
               null_mean=float(Dn.mean()), null_std=float(Dn.std()), sigma=float(sig),
               D_over_Dkin=float(amp_p / Dk))
    with open("data/catwise/cw_dipole_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote data/catwise/cw_dipole_result.json")


if __name__ == "__main__":
    main()
