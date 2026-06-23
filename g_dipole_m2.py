#!/usr/bin/env python3
"""
Milestone 2 — PRINCIPLED selection correction for the Quaia number-count dipole.

M1 showed naive division by the completeness map inflates noisy low-completeness
pixels and rotates the dipole ~40deg (an estimator artifact, not signal). The field-
standard fix is NOT to divide: it is to FORWARD-MODEL the selection function as a
multiplicative term in a Poisson likelihood (Dam+2023, Oayda+2024 "Poissonian
likelihood"). The random catalogs Quaia ships are just a Monte-Carlo realisation of
exactly this N*selfunc model, so using the published selfunc map directly is the same
forward model with no MC noise and nothing to download.

Model (per pixel p, in the selection-function frame = ICRS, established in M1):
    mu_p = s_p * N * (1 + A . nhat_p)            A = D * dhat  (dipole vector)
Maximise Poisson log-lik  sum_p [ k_p log mu_p - mu_p ]  over (logN, Ax, Ay, Az)
on the masked sky. D = |A|, direction = A/|A|.

Test (Oayda 2024 prediction for Quaia HIGH, G<20.5 = our sample): under a principled
correction + aggressive galactic mask the dipole should NOT reconcile with the CMB but
drift toward (l,b)~(330,60); Quaia LOW (G<20.0) instead collapses to CMB-consistent.

Gentle: single-threaded, column-subset read, one Nside=64 map, 4-param fit (~ms).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
from scipy.optimize import minimize

NSIDE = 64
CAT = "data/quaia/quaia_G20.5.fits"
SF_FLOOR = 0.05                       # ignore pixels with model completeness below this
GMAX = float(os.environ.get("GMAX", "20.5"))   # 20.5 = Quaia high, 20.0 = Quaia low
# use the selection-function map MATCHED to the magnitude cut
SELF = "data/quaia/selfunc_G20.0_nside64.fits" if GMAX <= 20.0 else "data/quaia/selfunc_G20.5_nside64.fits"

V_CMB = 369.82; BETA = V_CMB / 299792.458
L_CMB, B_CMB = 264.021, 48.253
# Ellis-Baldwin kinematic expectation. Oayda find D_kin ~ 0.0068 (Quaia high), 0.0080 (low).
X_SLOPE, ALPHA = 1.7, 1.0
D_KIN = (2 + X_SLOPE * (1 + ALPHA)) * BETA

hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix


def read_map(path):
    d = fits.open(path)[1].data
    return np.asarray(d[d.columns.names[0]]).reshape(-1)[:npix].astype(float)


def load_cat(path):
    with fits.open(path, memmap=True) as f:
        ra = np.asarray(f[1].data["ra"], float)
        dec = np.asarray(f[1].data["dec"], float)
        g = np.asarray(f[1].data["phot_g_mean_mag"], float)
    return ra, dec, g


def pix_unit_vectors():
    lon, lat = hp.healpix_to_lonlat(np.arange(npix))
    cb = np.cos(lat.rad)
    return np.column_stack([cb * np.cos(lon.rad), cb * np.sin(lon.rad), np.sin(lat.rad)]), lon, lat


def lstsq_dipole(counts, good):
    xyz, _, _ = pix_unit_vectors()
    X = np.column_stack([np.ones(npix), xyz])
    coef, *_ = np.linalg.lstsq(X[good], counts[good], rcond=None)
    return np.linalg.norm(coef[1:]) / coef[0], coef[1:] / np.linalg.norm(coef[1:])


def poisson_dipole(counts, selfunc, good):
    """ML fit of mu = s*N*(1 + A.nhat) on masked pixels. Returns D, dvec, success."""
    xyz, _, _ = pix_unit_vectors()
    s = selfunc[good]
    n = xyz[good]
    k = counts[good]
    # init from a crude divide-LS so the optimiser starts near the basin
    N0 = k.sum() / s.sum()

    def nll(theta):
        logN, A = theta[0], theta[1:]
        proj = n @ A
        mu = s * np.exp(logN) * (1.0 + proj)
        if np.any(mu <= 0):
            return 1e12
        return -np.sum(k * np.log(mu) - mu)

    x0 = np.array([np.log(N0), 0.0, 0.0, 0.0])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-4, "maxiter": 20000})
    A = res.x[1:]
    return np.linalg.norm(A), A / np.linalg.norm(A), res.success


def main():
    selfunc = read_map(SELF)
    ra, dec, g = load_cat(CAT)
    sel = g < GMAX
    ra, dec = ra[sel], dec[sel]
    print(f"GMAX={GMAX}  N sources={sel.sum():,}  selfunc={os.path.basename(SELF)}")

    # bin in ICRS (selection-function frame, established M1)
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    counts = np.bincount(pix, minlength=npix).astype(float)

    _, plon, plat = pix_unit_vectors()
    pix_b = SkyCoord(plon, plat, frame="icrs").galactic.b.deg
    cmb = SkyCoord(L_CMB * u.deg, B_CMB * u.deg, frame="galactic")

    def show(tag, bcut, amp, dvec, ok=""):
        dlat = np.degrees(np.arcsin(dvec[2]))
        dlon = np.degrees(np.arctan2(dvec[1], dvec[0])) % 360
        eq = SkyCoord(dlon * u.deg, dlat * u.deg, frame="icrs")
        gd = eq.galactic
        sep = gd.separation(cmb).deg
        print(f"{tag:>9} |b|>{bcut:>2d}  D={amp:>7.4f}  D/Dkin={amp/D_KIN:>5.2f}  "
              f"RA={eq.ra.deg:>5.0f} Dec={eq.dec.deg:>+4.0f}  "
              f"(l,b)=({gd.l.deg:>4.0f},{gd.b.deg:>+4.0f})  offCMB={sep:>5.1f}  {ok}")

    print(f"\n=== Quaia dipole, principled (Poisson forward-model) selection correction ===")
    print(f"D_kin~{D_KIN:.4f} (EB placeholder); Oayda D_kin~0.0068(high)/0.0080(low); "
          f"CMB at (l,b)=({L_CMB:.0f},{B_CMB:.0f})")
    print(f"ref: Oayda+2024 -> Quaia-high drifts to (l,b)~(330,60), NOT CMB-consistent")
    for bcut in (0, 30, 40, 50):
        good = (np.abs(pix_b) >= bcut) & (selfunc > SF_FLOOR) & (counts > 0)
        if good.sum() < 200:
            continue
        amp_r, dv_r = lstsq_dipole(counts, (np.abs(pix_b) >= bcut) & (counts > 0))
        show("raw-LS", bcut, amp_r, dv_r)
        amp_p, dv_p, ok = poisson_dipole(counts, selfunc, good)
        show("poisson", bcut, amp_p, dv_p, "ok" if ok else "FIT?")


if __name__ == "__main__":
    main()
