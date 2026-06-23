#!/usr/bin/env python3
"""
Collect Quaia dipole results (principled Poisson estimator) across masks WITH the
isotropic-mock noise floor, and dump JSON for the figure script. Uses the matched
selection function per magnitude cut and the M3-measured D_kin.
Gentle: single-threaded, Nside=64, NMOCK small batches.
"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize

NSIDE = 64
NMOCK = int(os.environ.get("NMOCK", "80"))
SF_FLOOR = 0.05
V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
# M3-measured kinematic expectations (g_xalpha.py)
D_KIN = {20.0: 0.0079, 20.5: 0.0065}
D_KIN_CI = {20.0: (0.0071, 0.0088), 20.5: (0.0057, 0.0073)}
rng = np.random.default_rng(19)

hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
cb = np.cos(lat.rad)
XYZ = np.column_stack([cb * np.cos(lon.rad), cb * np.sin(lon.rad), np.sin(lat.rad)])
PIX_B = SkyCoord(lon, lat, frame="icrs").galactic.b.deg
CMB = SkyCoord(L_CMB * u.deg, B_CMB * u.deg, frame="galactic")


def read_map(p):
    d = fits.open(p)[1].data
    return np.asarray(d[d.columns.names[0]]).reshape(-1)[:npix].astype(float)


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


def run(gmax):
    self = "data/quaia/selfunc_G20.0_nside64.fits" if gmax <= 20.0 else "data/quaia/selfunc_G20.5_nside64.fits"
    s = read_map(self)
    with fits.open("data/quaia/quaia_G20.5.fits", memmap=True) as f:
        ra = np.asarray(f[1].data["ra"], float); dec = np.asarray(f[1].data["dec"], float)
        g = np.asarray(f[1].data["phot_g_mean_mag"], float)
    m = g < gmax
    counts = np.bincount(hp.lonlat_to_healpix(ra[m] * u.deg, dec[m] * u.deg), minlength=npix).astype(float)
    res = []
    for bcut in (30, 40, 50):
        good = (np.abs(PIX_B) >= bcut) & (s > SF_FLOOR) & (counts > 0)
        D, dv = pois(counts, s, good)
        l, b, sep = direction(dv)
        Nbar = counts[good].sum() / s[good].sum(); exp = s * Nbar
        Dn = np.empty(NMOCK)
        for i in range(NMOCK):
            c = np.zeros(npix); c[good] = rng.poisson(exp[good]); Dn[i], _ = pois(c, s, good)
        sig = (D - Dn.mean()) / Dn.std()
        res.append(dict(bcut=bcut, D=float(D), l=float(l), b=float(b), offCMB=float(sep),
                        null_mean=float(Dn.mean()), null_std=float(Dn.std()), sigma=float(sig)))
        print(f"  G<{gmax} |b|>{bcut}: D={D:.4f} (l,b)=({l:.0f},{b:+.0f}) off={sep:.0f} "
              f"null={Dn.mean():.4f}+/-{Dn.std():.4f} {sig:.1f}sig")
    return dict(gmax=gmax, D_kin=D_KIN[gmax], D_kin_ci=D_KIN_CI[gmax], masks=res)


def main():
    out = {"low": run(20.0), "high": run(20.5)}
    with open("data/quaia/quaia_dipole_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote data/quaia/quaia_dipole_results.json")


if __name__ == "__main__":
    main()
