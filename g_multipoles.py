#!/usr/bin/env python3
"""
Multipole diagnostic: is the number-count signal a clean dipole (l=1), or is l=1 just one
of several comparable modes (quadrupole contamination -> Oayda's "double dipole")?

Fits real spherical harmonics l=0..LMAX on the masked sky with the SAME forward model as the
dipole (counts_p = s_p * sum_lm c_lm Y_lm), then reports per-l amplitude
    A_l = sqrt(sum_m c_lm^2) / c_00      (A_1 == the usual dipole amplitude D).
Isotropic Poisson mocks through the identical fit give each l its shot-noise floor and the
significance of the measured A_l. Crucially, l>=2 carries NO kinematic signal, so the spread of
A_2..A_LMAX is a data-driven estimate of the (clustering + systematic + shot) fluctuation per
mode -> a cheap clustering-aware sanity check on whether the dipole is anomalous vs the field.

Caveat: the cut sky couples multipoles; joint least-squares is the linear ML estimate and the
mocks pass through the SAME coupling, so the significances are apples-to-apples.
Gentle: single-threaded, Nside=64, ~50 mocks/sample.
"""
import os, json, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits, ascii as asciirw
from astropy.table import vstack
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.special import lpmv
import glob

NSIDE = 64
# LMAX=2 (dipole+quadrupole) is well-conditioned on a half-sky mask (cond~7); LMAX>=3 lets
# half-sky mode-coupling corrupt the dipole estimate (verified: cond 15->285). The science
# question is dipole-vs-quadrupole, which LMAX=2 answers cleanly. D_dipole = A_1 * sqrt(3).
LMAX = 2
SQRT3 = np.sqrt(3.0)
NMOCK = int(os.environ.get("NMOCK", "60"))
rng = np.random.default_rng(23)
hp = HEALPix(nside=NSIDE, order="ring")
npix = hp.npix
lon, lat = hp.healpix_to_lonlat(np.arange(npix))
THETA = (np.pi / 2 - lat.rad)        # colatitude
PHI = lon.rad
PIX_B = SkyCoord(lon, lat, frame="icrs").galactic.b.deg


def real_ylm():
    """real spherical harmonics for l=0..LMAX, returns (cols [npix x Nlm], llist)."""
    cols, ll = [], []
    x = np.cos(THETA)
    for l in range(LMAX + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) *
                           math.factorial(l - am) / math.factorial(l + am))
            P = lpmv(am, l, x)
            if m > 0:
                y = np.sqrt(2) * norm * P * np.cos(m * PHI)
            elif m < 0:
                y = np.sqrt(2) * norm * P * np.sin(am * PHI)
            else:
                y = norm * P
            cols.append(y); ll.append(l)
    return np.column_stack(cols), np.array(ll)


YLM, LL = real_ylm()


def fit_Al(counts, s, good):
    """least-squares counts = s * (Ylm @ c); return A_l for l=0..LMAX."""
    X = s[good, None] * YLM[good]
    c, *_ = np.linalg.lstsq(X, counts[good], rcond=None)
    c00 = c[0]
    A = np.array([np.sqrt(np.sum(c[LL == l] ** 2)) / c00 for l in range(LMAX + 1)])
    return A


def read_selfmap(p):
    d = fits.open(p)[1].data
    return np.asarray(d[d.columns.names[0]]).reshape(-1)[:npix].astype(float)


def analyze(name, counts, s, bcut):
    good = (np.abs(PIX_B) >= bcut) & (counts > 0) & (s > 0.05)
    A = fit_Al(counts, s, good)
    Nbar = counts[good].sum() / s[good].sum()
    exp = s * Nbar
    Anull = np.empty((NMOCK, LMAX + 1))
    for i in range(NMOCK):
        c = np.zeros(npix); c[good] = rng.poisson(exp[good])
        Anull[i] = fit_Al(c, s, good)
    mu = Anull.mean(0); sd = Anull.std(0)
    sig = (A - mu) / sd
    print(f"\n=== {name}  |b|>{bcut}  (pix={good.sum()}, N={counts[good].sum():,.0f}) ===")
    print(f"{'l':>2} {'A_l':>8} {'null':>8} {'+/-':>7} {'sigma':>6}  {'D-equiv':>8}")
    for l in range(LMAX + 1):
        tag = "  <- DIPOLE (kinematic)" if l == 1 else ("  <- quadrupole" if l == 2 else "")
        deq = A[l] * SQRT3 if l == 1 else (A[l] if l else 1.0)
        print(f"{l:>2} {A[l]:>8.4f} {mu[l]:>8.4f} {sd[l]:>7.4f} {sig[l]:>6.1f}  {deq:>8.4f}{tag}")
    print(f"  dipole D = A_1*sqrt(3) = {A[1]*SQRT3:.4f} ({sig[1]:.1f}sigma);  "
          f"quadrupole A_2 = {A[2]:.4f} ({sig[2]:.1f}sigma);  A_2/A_1 = {A[2]/A[1]:.2f}")
    return dict(name=name, bcut=bcut, A=A.tolist(), null_mean=mu.tolist(),
                null_std=sd.tolist(), sigma=sig.tolist(),
                D_dipole=float(A[1] * SQRT3), quad_sigma=float(sig[2]))


def main():
    out = []
    # Quaia
    qf = fits.open("data/quaia/quaia_G20.5.fits", memmap=True)[1].data
    ra = np.asarray(qf["ra"], float); dec = np.asarray(qf["dec"], float)
    g = np.asarray(qf["phot_g_mean_mag"], float)
    for gmax, sfile, lab in [(20.0, "selfunc_G20.0_nside64.fits", "Quaia low"),
                             (20.5, "selfunc_G20.5_nside64.fits", "Quaia high")]:
        s = read_selfmap("data/quaia/" + sfile)
        m = g < gmax
        c = np.bincount(hp.lonlat_to_healpix(ra[m] * u.deg, dec[m] * u.deg), minlength=npix).astype(float)
        out.append(analyze(lab, c, s, 40))
    # CatWISE (uniform selection s=1; |b|>30 + the same Magellanic+hot-pixel masking is applied
    # by zeroing those pixels via counts>0 & a coarse cut here for the diagnostic)
    files = sorted(glob.glob("data/catwise/cw_ra*.csv"))
    if files:
        t = vstack([asciirw.read(f, format="csv") for f in files])
        ra = np.asarray(t["ra"], float); dec = np.asarray(t["dec"], float)
        elat = np.abs(np.asarray(t["elat"], float))
        pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
        c = np.bincount(pix, minlength=npix).astype(float)
        pix_abel = np.divide(np.bincount(pix, weights=elat, minlength=npix),
                             np.maximum(np.bincount(pix, minlength=npix), 1),
                             out=np.full(npix, np.nan), where=np.bincount(pix, minlength=npix) > 0)
        gg = SkyCoord(lon, lat, frame="icrs").galactic
        pg = SkyCoord(gg.l.deg * u.deg, gg.b.deg * u.deg, frame="galactic")
        lmc = pg.separation(SkyCoord(280.5 * u.deg, -32.9 * u.deg, frame="galactic")).deg
        smc = pg.separation(SkyCoord(302.8 * u.deg, -44.3 * u.deg, frame="galactic")).deg
        keep = (lmc > 9) & (smc > 7)
        base = (np.abs(PIX_B) >= 30) & (c > 0) & keep & np.isfinite(pix_abel)
        gd = base.copy()
        for _ in range(6):
            med = np.median(c[gd]); mad = np.median(np.abs(c[gd] - med)) * 1.4826
            new = gd & (c <= med + 6 * mad)
            if new.sum() == gd.sum(): break
            gd = new
        # two versions: uniform selection, and ecliptic-trend forward-modelled (M4 correction)
        s_uni = np.where(gd, 1.0, 0.0)
        cc = np.where(gd, c, 0.0)
        out.append(analyze("CatWISE (uniform s)", cc, s_uni, 30))
        A2 = np.column_stack([np.ones(gd.sum()), pix_abel[gd]])
        cfit, *_ = np.linalg.lstsq(A2, c[gd], rcond=None)
        trend = cfit[0] + cfit[1] * pix_abel
        s_ecl = np.where(gd, np.clip(trend / np.nanmean(trend[gd]), 1e-3, None), 0.0)
        out.append(analyze("CatWISE (ecliptic-corrected)", cc, s_ecl, 30))
    with open("data/multipoles_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote data/multipoles_result.json")


if __name__ == "__main__":
    main()
