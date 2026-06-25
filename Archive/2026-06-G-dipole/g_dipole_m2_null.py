#!/usr/bin/env python3
"""
M2 uncertainty / significance — isotropic-mock null for the Poisson dipole estimator.

For a given magnitude cut (GMAX) and galactic mask (BCUT), this:
  1. measures the data dipole D_data with the principled Poisson forward model;
  2. builds the NULL distribution of recovered D by Poisson-sampling NMOCK isotropic
     skies whose expected counts are s_p * Nbar (the SAME selection function + mask,
     ZERO injected dipole) and re-fitting each -> gives the cut-sky estimator's bias
     floor and scatter (closes M1 next-step #2);
  3. optionally injects a CMB-amplitude/direction dipole to confirm recovery & direction.

Significance of the measured dipole = (D_data - <D_null>) / std(D_null), and the
"is it CMB-consistent?" question is answered by the direction offset vs the null
direction scatter. Gentle: single-threaded, Nside=64, 4-param fits.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize

NSIDE = 64
CAT = "data/quaia/quaia_G20.5.fits"
SF_FLOOR = 0.05
GMAX = float(os.environ.get("GMAX", "20.0"))
BCUT = int(os.environ.get("BCUT", "40"))
NMOCK = int(os.environ.get("NMOCK", "200"))
SELF = "data/quaia/selfunc_G20.0_nside64.fits" if GMAX <= 20.0 else "data/quaia/selfunc_G20.5_nside64.fits"

V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
X_SLOPE, ALPHA = 1.7, 1.0
D_KIN = (2 + X_SLOPE * (1 + ALPHA)) * BETA

rng = np.random.default_rng(7)
hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
cb = np.cos(lat.rad)
XYZ = np.column_stack([cb * np.cos(lon.rad), cb * np.sin(lon.rad), np.sin(lat.rad)])


def read_map(p):
    d = fits.open(p)[1].data
    return np.asarray(d[d.columns.names[0]]).reshape(-1)[:npix].astype(float)


def fit(counts, s, good):
    n = XYZ[good]; k = counts[good]; sg = s[good]
    N0 = k.sum() / sg.sum()

    def nll(t):
        mu = sg * np.exp(t[0]) * (1.0 + n @ t[1:])
        if np.any(mu <= 0): return 1e12
        return -np.sum(k * np.log(mu) - mu)

    r = minimize(nll, [np.log(N0), 0, 0, 0], method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-4, "maxiter": 20000})
    A = r.x[1:]
    return np.linalg.norm(A), A / max(np.linalg.norm(A), 1e-12)


def vec(l_deg, b_deg):
    c = SkyCoord(l_deg * u.deg, b_deg * u.deg, frame="galactic").icrs
    return XYZ[hp.lonlat_to_healpix(c.ra, c.dec)]  # unit vec in selfunc(ICRS) frame


def main():
    s = read_map(SELF)
    with fits.open(CAT, memmap=True) as f:
        ra = np.asarray(f[1].data["ra"], float); dec = np.asarray(f[1].data["dec"], float)
        g = np.asarray(f[1].data["phot_g_mean_mag"], float)
    m = g < GMAX; ra, dec = ra[m], dec[m]
    counts = np.bincount(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg), minlength=npix).astype(float)

    pix_b = SkyCoord(lon, lat, frame="icrs").galactic.b.deg
    good = (np.abs(pix_b) >= BCUT) & (s > SF_FLOOR) & (counts > 0)
    cmb = SkyCoord(L_CMB * u.deg, B_CMB * u.deg, frame="galactic")
    dhat_cmb = vec(L_CMB, B_CMB)

    print(f"GMAX={GMAX}  |b|>{BCUT}  selfunc={os.path.basename(SELF)}  "
          f"pix={good.sum()}  N={m.sum():,}  NMOCK={NMOCK}")

    D_data, dv = fit(counts, s, good)
    eq = SkyCoord(np.degrees(np.arctan2(dv[1], dv[0])) % 360 * u.deg,
                  np.degrees(np.arcsin(dv[2])) * u.deg, frame="icrs")
    sep_data = eq.galactic.separation(cmb).deg
    print(f"DATA   D={D_data:.4f} (D/Dkin={D_data/D_KIN:.2f})  "
          f"(l,b)=({eq.galactic.l.deg:.0f},{eq.galactic.b.deg:+.0f})  offCMB={sep_data:.1f}")

    # --- isotropic null: expected = s*Nbar, zero dipole ---
    Nbar = counts[good].sum() / s[good].sum()
    exp = s * Nbar
    Dn = np.empty(NMOCK)
    seps = np.empty(NMOCK)
    for i in range(NMOCK):
        c = np.zeros(npix); c[good] = rng.poisson(exp[good])
        D_i, dvi = fit(c, s, good)
        Dn[i] = D_i
        ei = np.degrees(np.arctan2(dvi[1], dvi[0])) % 360, np.degrees(np.arcsin(dvi[2]))
        seps[i] = SkyCoord(ei[0] * u.deg, ei[1] * u.deg, frame="icrs").galactic.separation(cmb).deg
    sig = (D_data - Dn.mean()) / Dn.std()
    print(f"NULL   <D>={Dn.mean():.4f} +/- {Dn.std():.4f}  (isotropic floor on this cut sky)")
    print(f"       D_data is {sig:.1f}sigma above the isotropic null amplitude floor")
    print(f"       null direction offset from CMB: median={np.median(seps):.0f} "
          f"(random dirs span the masked sky; data offCMB={sep_data:.0f})")

    # --- CMB-amplitude injection: can the pipeline recover a true CMB dipole here? ---
    expc = s * Nbar * (1.0 + D_KIN * (XYZ @ dhat_cmb))
    Dc = np.empty(NMOCK); sc = np.empty(NMOCK)
    for i in range(NMOCK):
        c = np.zeros(npix); c[good] = rng.poisson(np.clip(expc[good], 0, None))
        D_i, dvi = fit(c, s, good)
        Dc[i] = D_i
        ei = np.degrees(np.arctan2(dvi[1], dvi[0])) % 360, np.degrees(np.arcsin(dvi[2]))
        sc[i] = SkyCoord(ei[0] * u.deg, ei[1] * u.deg, frame="icrs").galactic.separation(cmb).deg
    print(f"INJECT CMB dipole D_kin={D_KIN:.4f}: recovered <D>={Dc.mean():.4f}+/-{Dc.std():.4f}, "
          f"dir offCMB med={np.median(sc):.0f} (recovery check)")


if __name__ == "__main__":
    main()
