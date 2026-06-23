#!/usr/bin/env python3
"""
NVSS (1.4 GHz radio) number-count dipole through the same pipeline -- a THIRD, fully
independent catalogue (radio systematics share nothing with optical Gaia/Quaia or IR WISE).
If radio aligns with CatWISE, the excess is cross-wavelength; if not, it is systematic.

Footprint: NVSS covers Dec > -40 deg. We fit on |b|>30 AND Dec > -37 (inside the boundary),
isotropic mocks use the SAME footprint so the cut-sky floor is correct. EB inputs (2025
kinematic paper): x=0.74, alpha=0.75 -> D_kin~0.0041. Flux cut S1.4>15 mJy (homogenisation).
Gentle: single-threaded, Nside=64.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import ascii as asciirw
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize

NSIDE = 64
NMOCK = int(os.environ.get("NMOCK", "100"))
V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
X_NVSS, A_NVSS = 0.74, 0.75
D_KIN = (2 + X_NVSS * (1 + A_NVSS)) * BETA
rng = np.random.default_rng(31)
hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
cb = np.cos(lat.rad)
XYZ = np.column_stack([cb * np.cos(lon.rad), cb * np.sin(lon.rad), np.sin(lat.rad)])
PIX_B = SkyCoord(lon, lat, frame="icrs").galactic.b.deg
PIX_DEC = lat.deg
CMB = SkyCoord(L_CMB * u.deg, B_CMB * u.deg, frame="galactic")


def lstsq_dipole(counts, good):
    X = np.column_stack([np.ones(npix), XYZ])
    co, *_ = np.linalg.lstsq(X[good], counts[good], rcond=None)
    return np.linalg.norm(co[1:]) / co[0], co[1:] / np.linalg.norm(co[1:])


def pois(counts, s, good):
    n = XYZ[good]; k = counts[good]; sg = s[good]
    N0 = k.sum() / sg.sum()
    def nll(th):
        mu = sg * np.exp(th[0]) * (1 + n @ th[1:])
        if np.any(mu <= 0): return 1e12
        return -np.sum(k * np.log(mu) - mu)
    r = minimize(nll, [np.log(N0), 0, 0, 0], method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-4, "maxiter": 20000})
    A = r.x[1:]; nrm = np.linalg.norm(A)
    return nrm, A / max(nrm, 1e-12)


def direction(dv):
    eq = SkyCoord((np.degrees(np.arctan2(dv[1], dv[0])) % 360) * u.deg,
                  np.degrees(np.arcsin(dv[2])) * u.deg, frame="icrs")
    g = eq.galactic
    return g.l.deg, g.b.deg, g.separation(CMB).deg


def main():
    t = asciirw.read("data/nvss/nvss_S15.csv", format="csv")
    ra = np.asarray(t["RAJ2000"], float); dec = np.asarray(t["DEJ2000"], float)
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    counts = np.bincount(pix, minlength=npix).astype(float)
    print(f"NVSS S>15mJy: {len(ra):,} sources; D_kin={D_KIN:.4f} (x={X_NVSS},a={A_NVSS})")

    # declination-detrend (radio flux-scale systematic analog), fit density vs sin(dec)
    for bc in (30, 40):
        good = (np.abs(PIX_B) >= bc) & (PIX_DEC > -37) & (counts > 0)
        # selection model: smooth density vs declination (captures the NVSS dec systematic)
        A2 = np.column_stack([np.ones(good.sum()), PIX_DEC[good], PIX_DEC[good] ** 2])
        cf, *_ = np.linalg.lstsq(A2, counts[good], rcond=None)
        trend = cf[0] + cf[1] * PIX_DEC + cf[2] * PIX_DEC ** 2
        s = np.clip(trend / np.mean(trend[good]), 1e-3, None)

        Dr, dvr = lstsq_dipole(counts, good); lr, br, sr = direction(dvr)
        Dp, dvp = pois(counts, s, good); lp, bp, sp = direction(dvp)
        Nbar = counts[good].sum() / s[good].sum(); exp = s * Nbar
        Dn = np.empty(NMOCK)
        for i in range(NMOCK):
            c = np.zeros(npix); c[good] = rng.poisson(exp[good]); Dn[i], _ = pois(c, s, good)
        sig = (Dp - Dn.mean()) / Dn.std()
        print(f"\n|b|>{bc}, Dec>-37  (pix={good.sum()}, N={counts[good].sum():,.0f}, fsky={good.mean():.2f})")
        print(f"  raw-LS : D={Dr:.4f} (l,b)=({lr:.0f},{br:+.0f}) offCMB={sr:.0f}")
        print(f"  poisson: D={Dp:.4f} (l,b)=({lp:.0f},{bp:+.0f}) offCMB={sp:.0f}  D/Dkin={Dp/D_KIN:.2f}")
        print(f"  null   : <D>={Dn.mean():.4f}+/-{Dn.std():.4f}  -> {sig:.1f}sigma above floor")


if __name__ == "__main__":
    main()
