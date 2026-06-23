#!/usr/bin/env python3
"""
Redshift tomography of the Quaia dipole -- a test ONLY Quaia enables (it ships per-source
photo-z's; CatWISE has none).

Logic: the KINEMATIC dipole is a Doppler/aberration boost -> its DIRECTION is the CMB apex in
every redshift slice, and its amplitude is ~z-independent. A dipole from LOCAL STRUCTURE
(clustering) is strongest at low z and decays with depth, and need not point at the CMB. So:
  - direction stable at the CMB across z, flat amplitude  => kinematic
  - amplitude/direction drifting with z (esp. low-z excess) => structure/systematic
Splits Quaia (G<20.5) into redshift bins, runs the principled Poisson dipole (full selfunc as
the forward model -- approximate, since the per-bin selection differs; the DIRECTION trend is
the robust observable) at |b|>40, with the isotropic-mock floor per bin.
Gentle: single-threaded, Nside=64.
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
NMOCK = int(os.environ.get("NMOCK", "80"))
BCUT = 40
V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
rng = np.random.default_rng(37)
hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
cb = np.cos(lat.rad)
XYZ = np.column_stack([cb * np.cos(lon.rad), cb * np.sin(lon.rad), np.sin(lat.rad)])
PIX_B = SkyCoord(lon, lat, frame="icrs").galactic.b.deg
CMB = SkyCoord(L_CMB * u.deg, B_CMB * u.deg, frame="galactic")


def read_self():
    d = fits.open("data/quaia/selfunc_G20.5_nside64.fits")[1].data
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


def main():
    s = read_self()
    with fits.open("data/quaia/quaia_G20.5.fits", memmap=True) as f:
        ra = np.asarray(f[1].data["ra"], float); dec = np.asarray(f[1].data["dec"], float)
        z = np.asarray(f[1].data["redshift_quaia"], float)
    good = (np.abs(PIX_B) >= BCUT) & (s > 0.05)
    # redshift quartiles
    zq = np.nanpercentile(z, [25, 50, 75])
    edges = [0.0, zq[0], zq[1], zq[2], 9.9]
    print(f"Quaia |b|>{BCUT} redshift tomography (quartiles z-edges={np.round(edges,3)})")
    print(f"kinematic prediction: direction ~CMB (l,b={L_CMB:.0f},{B_CMB:.0f}) in EVERY bin, flat amplitude")
    print(f"{'z-bin':>14} {'N':>9} {'zmed':>5} {'D':>7} {'l':>5} {'b':>5} {'offCMB':>6} {'sigma':>6}")
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (z >= lo) & (z < hi) & np.isfinite(z)
        counts = np.bincount(hp.lonlat_to_healpix(ra[sel] * u.deg, dec[sel] * u.deg),
                             minlength=npix).astype(float)
        g2 = good & (counts > 0)
        D, dv = pois(counts, s, g2); l, b, sep = direction(dv)
        Nbar = counts[g2].sum() / s[g2].sum(); exp = s * Nbar
        Dn = np.empty(NMOCK)
        for j in range(NMOCK):
            c = np.zeros(npix); c[g2] = rng.poisson(exp[g2]); Dn[j], _ = pois(c, s, g2)
        sig = (D - Dn.mean()) / Dn.std()
        print(f"  [{lo:.2f},{hi:4.2f}) {sel.sum():>9,} {np.median(z[sel]):>5.2f} "
              f"{D:>7.4f} {l:>5.0f} {b:>+5.0f} {sep:>6.0f} {sig:>6.1f}")
        rows.append((np.median(z[sel]), D, sep, sig, Dn.std()))
    print("\n(direction wandering off the CMB at low z, or a low-z amplitude spike, "
          "would indicate a structure/selection dipole rather than a clean kinematic one)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    zs = [r[0] for r in rows]; Ds = [r[1] for r in rows]; offs = [r[2] for r in rows]
    errs = [r[4] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].errorbar(zs, Ds, yerr=errs, marker="o", lw=2, capsize=4, color="#1f77b4")
    ax[0].axhline(0.0079, ls="--", color="0.4"); ax[0].text(2.0, 0.0082, "D$_{kin}$~0.008", color="0.4")
    ax[0].set_xlabel("median redshift"); ax[0].set_ylabel("dipole amplitude D"); ax[0].grid(alpha=0.3)
    ax[0].set_title("Quaia dipole amplitude vs redshift (|b|>40)")
    ax[1].plot(zs, offs, "-o", lw=2, color="#d62728")
    ax[1].axhline(0, ls="--", color="0.4"); ax[1].text(1.5, 3, "aligned with CMB apex", color="0.4")
    ax[1].set_xlabel("median redshift"); ax[1].set_ylabel("dipole offset from CMB (deg)")
    ax[1].set_ylim(0, 100); ax[1].grid(alpha=0.3)
    ax[1].set_title("direction offset from CMB vs redshift\nhigh-z (cleanest kinematic probe) -> closest to CMB")
    fig.tight_layout(); fig.savefig("figures/fig4_tomography.png", dpi=140)
    print("wrote figures/fig4_tomography.png")


if __name__ == "__main__":
    main()
