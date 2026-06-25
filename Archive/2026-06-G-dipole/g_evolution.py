#!/usr/bin/env python3
"""
Milestone 5 — evolution/relativistic correction to the kinematic expectation (Guandalin 2023;
Dalang & Bonvin 2022; Maartens 2018), computed with Quaia's MEASURED redshift distribution
(where Guandalin had to model a QLF -> their result was QLF-model-dependent by up to ~3sigma).

Relativistic 2D dipole factor, projected over f(z)=normalised dN/dz (Guandalin eqs 13-17):
    D = D_cosmo + D_mag + D_evol
    D_cosmo = INT f(z) [ 2 + 2/(r*Hc) + Hdot/Hc^2 ]
    D_mag   = -2 INT f(z) x/(r*Hc)
    D_evol  = - INT f(z) b_e(z)
with conformal Hubble Hc = a H = H(z)/(1+z), comoving distance r, and (derived here)
    Hdot/Hc^2 = 1 - (1+z) H'(z)/H(z),     r*Hc = D_C(z) H(z) / [(1+z) c].
b_e(z) = -(1+z) dln n / dz is the evolution bias (n = comoving number density).

We compare D_relativistic to the Ellis-Baldwin value D_EB = 2 + x(1+alpha) used in M1-M4,
and propagate to D_kin = D * beta. The cosmo+mag terms are exact; b_e is the model-dependent
piece (von Hausegger 2024 argues it is already captured by EB for redshift-INTEGRATED samples
when x is measured at the limit) -- we bracket it: conserved (b_e=0), empirical from Quaia n(z),
and a Guandalin-typical range.
Gentle: single-threaded, reads only the redshift column.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cos
import astropy.units as u

C_KMS = 299792.458
V_CMB = 369.82; BETA = V_CMB / C_KMS
X_EB, ALPHA = 0.96, 2.44     # Quaia high, from M3 (g_xalpha.py)


def Hz(z):
    return cos.H(z).to_value(u.km / u.s / u.Mpc)


def main():
    with fits.open("data/quaia/quaia_G20.5.fits", memmap=True) as f:
        z = np.asarray(f[1].data["redshift_quaia"], float)
        g = np.asarray(f[1].data["phot_g_mean_mag"], float)
    z = z[(g < 20.5) & np.isfinite(z) & (z > 0.05) & (z < 5)]
    print(f"Quaia high: {z.size:,} sources, z median={np.median(z):.2f}")

    # f(z) on a grid
    zg = np.linspace(0.1, 4.0, 200)
    dz = zg[1] - zg[0]
    hist, edges = np.histogram(z, bins=np.append(zg - dz / 2, zg[-1] + dz / 2), density=True)
    fz = hist / np.trapezoid(hist, zg)

    H = Hz(zg)
    Hp = np.gradient(H, zg)                     # H'(z)
    Dc = cos.comoving_distance(zg).to_value(u.Mpc)
    rHc = Dc * H / ((1 + zg) * C_KMS)           # r * conformal-Hubble (dimensionless, c=1)
    Hdot_over_Hc2 = 1 - (1 + zg) * Hp / H       # derived identity

    cosmo = 2 + 2 / rHc + Hdot_over_Hc2
    mag = -2 * X_EB / rHc
    # empirical evolution bias from the observed counts: n(z) ~ N(z)(1+z)Hc/r^2 (invert eq 11)
    Nz = fz                                     # dN/dz (shape)
    Hc = H / (1 + zg)
    nz = Nz * (1 + zg) * Hc / Dc ** 2
    good = (zg > 0.5) & (zg < 3.5) & (nz > 0)
    lnn = np.log(nz)
    be = -(1 + zg) * np.gradient(lnn, zg)
    be = np.clip(be, -10, 10)

    def proj(integrand):
        return np.trapezoid(fz * integrand, zg)

    D_cosmo = proj(cosmo); D_mag = proj(mag)
    D_eb = 2 + X_EB * (1 + ALPHA)
    be_reconcile = (D_cosmo + D_mag - D_eb)      # b_e (const) that makes D == D_EB
    print(f"\nEllis-Baldwin (M1-M4):  D_EB = 2 + x(1+a) = {D_eb:.3f}  "
          f"(x={X_EB}, a={ALPHA}) -> D_kin={D_eb*BETA:.4f}")
    print(f"\nRelativistic projection over Quaia f(z) (cosmo+mag terms are EXACT):")
    print(f"  D_cosmo = {D_cosmo:.3f}   [2 + 2/(rHc) + Hdot/Hc^2; <2/(rHc)>={proj(2/rHc):.2f}, "
          f"<Hdot/Hc^2>={proj(Hdot_over_Hc2):.2f}]  <- the O(1) cosmo terms are NOT negligible vs '2'")
    print(f"  D_mag   = {D_mag:.3f}   (magnification, x={X_EB}; largely cancels +2/(rHc))")
    print(f"  D_cosmo+D_mag (b_e=0) = {D_cosmo+D_mag:.3f}  -> D_kin={(D_cosmo+D_mag)*BETA:.4f}")
    print(f"\n  The EVOLUTION term D_evol=-<b_e> is the decisive, QLF-model-dependent piece:")
    for tag, beval in [("conserved b_e=0", 0.0),
                       ("Guandalin-typ b_e=+1", 1.0),
                       ("Guandalin-typ b_e=-1", -1.0),
                       (f"b_e to MATCH EB ({be_reconcile:+.1f})", be_reconcile)]:
        D = D_cosmo + D_mag - beval
        print(f"    b_e={beval:+5.2f} ({tag:24s}): D={D:6.3f}  D_kin={D*BETA:.4f}  D/D_EB={D/D_eb:.2f}")
    print(f"\n  -> Matching EB (D=5.3) requires b_e ~ {be_reconcile:.1f} (a strongly rising population);")
    print("     a naive empirical b_e from N(z) is VOLUME-CONTAMINATED (1/Dc^2 term dominates) and")
    print("     unreliable -- the proper b_e needs a QLF fit at fixed M_c (Guandalin's method, the")
    print("     source of their ~3sigma model spread). KEY POINT: the cosmo terms alone are O(1) and")
    print("     b_e can swing D_kin by >100%, so D_kin carries a real theoretical systematic that")
    print("     propagates into every 'anomaly' significance -- including our CatWISE 2.3x.")


if __name__ == "__main__":
    main()
