#!/usr/bin/env python3
"""
Milestone 1/2 — measure the number-count dipole of the Quaia quasar sample and
cross-reference to the literature (CMB kinematic expectation; CatWISE matter dipole).

Gentle by design: single-threaded, column-subset read, one HEALPix map in memory.
Empirically determines the selection-function map frame (no assumption).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

NSIDE = 64
SELFUNC_MIN = 0.5            # use pixels with completeness above this
CAT = "data/quaia/quaia_G20.5.fits"
SELF = "data/quaia/selfunc_G20.5_nside64.fits"

# CMB kinematic dipole (Planck 2018)
V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
# Ellis-Baldwin expected kinematic amplitude D = [2 + x(1+alpha)] beta
X_SLOPE, ALPHA = 1.7, 1.0   # PLACEHOLDER inputs (Milestone 3 re-measures these)
D_KIN = (2 + X_SLOPE*(1+ALPHA)) * BETA

hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix


def read_map(path):
    d = fits.open(path)[1].data
    return np.asarray(d[d.columns.names[0]]).reshape(-1)[:npix].astype(float)


def load_radec(path):
    with fits.open(path, memmap=True) as f:
        cols = [c.lower() for c in f[1].columns.names]
        print("catalog columns:", f[1].columns.names[:12], "...")
        def pick(*opts):
            for o in opts:
                if o in cols: return f[1].columns.names[cols.index(o)]
            raise KeyError(opts)
        ra = np.asarray(f[1].data[pick("ra")], float)
        dec = np.asarray(f[1].data[pick("dec")], float)
    return ra, dec


def bin_counts(lon_deg, lat_deg):
    pix = hp.lonlat_to_healpix(lon_deg*u.deg, lat_deg*u.deg)
    return np.bincount(pix, minlength=npix).astype(float)


def _xyz():
    lon, lat = hp.healpix_to_lonlat(np.arange(npix))
    cb = np.cos(lat.rad)
    return np.column_stack([cb*np.cos(lon.rad), cb*np.sin(lon.rad), np.sin(lat.rad)])


def fit_dipole(counts, good):
    """divide-estimator: n = A0 + A.(x,y,z) on selection-corrected density."""
    xyz = _xyz()
    X = np.column_stack([np.ones(npix), xyz])
    coef, *_ = np.linalg.lstsq(X[good], counts[good], rcond=None)
    A0, A = coef[0], coef[1:]
    return np.linalg.norm(A)/A0, A/np.linalg.norm(A), A0


def fit_dipole_fwd(counts, selfunc, good):
    """forward-model estimator: counts_p = s_p*(B0 + B.(x,y,z)).
    Design columns scaled by selfunc -> low-completeness pixels self-downweight,
    no noise-inflating division. This is the more correct linear estimator."""
    xyz = _xyz()
    s = selfunc[good][:, None]
    X = np.column_stack([selfunc[good], selfunc[good][:, None]*xyz[good]])
    coef, *_ = np.linalg.lstsq(X, counts[good], rcond=None)
    B0, B = coef[0], coef[1:]
    return np.linalg.norm(B)/B0, B/np.linalg.norm(B), B0


def main():
    selfunc = read_map(SELF)
    ra, dec = load_radec(CAT)
    print(f"N sources = {len(ra):,}")
    sc = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    # --- empirically determine the selection-function map frame ---
    frames = {"icrs": (ra, dec),
              "galactic": (sc.galactic.l.deg, sc.galactic.b.deg)}
    best = None
    for name, (lon, lat) in frames.items():
        c = bin_counts(lon, lat)
        m = selfunc > SELFUNC_MIN
        r = np.corrcoef(c[m], selfunc[m])[0, 1]
        print(f"  frame={name:9s} corr(counts, selfunc)={r:+.3f}")
        if best is None or r > best[1]:
            best = (name, r, c, lon, lat)
    frame_name, _, counts, lon, lat = best
    print(f"--> selection-function frame = {frame_name}")

    # galactic latitude per pixel (for an explicit plane cut)
    plon, plat = hp.healpix_to_lonlat(np.arange(npix))
    f = Galactic if frame_name == "galactic" else "icrs"
    pix_b = SkyCoord(plon, plat, frame=f).galactic.b.deg
    cmb = SkyCoord(L_CMB*u.deg, B_CMB*u.deg, frame="galactic")

    print("\n=== Quaia number-count dipole — masking sweep ===")
    print(f"CMB kinematic expectation D_kin={D_KIN:.4f} (x={X_SLOPE},a={ALPHA}; PLACEHOLDER); "
          f"CatWISE ref D~0.0155 @~28deg (Secrest 2021)")
    print(f"  ref: published Quaia |b|>30 dipole D=0.033+-0.005 @ RA~181,Dec~+20 (~30deg off CMB)")
    print(f"{'est':>7} {'sf_min':>6} {'|b|>':>5} {'fsky':>5} {'D':>8} {'D/Dkin':>7} "
          f"{'RA':>5} {'Dec':>5} {'(l,b)':>13} {'offCMB':>7}")
    def report(tag, amp, dvec):
        dlat = np.degrees(np.arcsin(dvec[2])); dlon = np.degrees(np.arctan2(dvec[1], dvec[0])) % 360
        eq = SkyCoord(dlon*u.deg, dlat*u.deg, frame=f).icrs
        gd = SkyCoord(dlon*u.deg, dlat*u.deg, frame=f).galactic
        sep = gd.separation(cmb).deg
        print(f"{tag:>7} {smin:>6.1f} {bcut:>5d} {good.mean():>5.2f} "
              f"{amp:>8.4f} {amp/D_KIN:>7.2f} {eq.ra.deg:>5.0f} {eq.dec.deg:>+5.0f} "
              f"({gd.l.deg:>4.0f},{gd.b.deg:>4.0f}) {sep:>7.1f}")
    for smin in (0.0, 0.5, 0.9):
        for bcut in (0, 30):
            base = (np.abs(pix_b) >= bcut)
            # raw: no selection handling, just dipole of counts on masked sky
            good = base & (counts > 0)
            report("raw", *fit_dipole(counts, good)[:2])
            # selection-corrected (divide) and forward-model, on selfunc>smin
            good = base & (selfunc > max(smin, 1e-6))
            if good.sum() < 100: continue
            dens = np.zeros(npix); dens[good] = counts[good]/selfunc[good]
            report("divide", *fit_dipole(dens, good)[:2])
            report("fwd", *fit_dipole_fwd(counts, selfunc, good)[:2])


if __name__ == "__main__":
    main()
