#!/usr/bin/env python3
"""
Milestone 3 — re-measure the Ellis-Baldwin inputs x (number-count slope) and alpha
(spectral index) DIRECTLY from Quaia, replacing the placeholder D_kin used in M1/M2.

D_kin = [2 + x(1 + alpha)] * beta,   beta = v_CMB/c = 369.82/299792.458

x : integral source counts dN/dOmega(>S) ~ S^-x. In magnitude space N(<m) ~ 10^(s m)
    with x = 2.5 * s. Fit s = d log10 (dN/dG) / dG from differential counts in a clean
    window BELOW the completeness rollover near the limit (von Hausegger 2024: measure x
    at the flux limit, where the dipole effect is dominated).

alpha : per-source from the G-BP colour using Gaia DR3 Vega zero-points + mean wavelengths
    (Oayda+2024 method). S_nu ~ nu^-alpha, m_G - m_BP = 2.5*alpha*log10(nu_G/nu_BP) + k,
    k = ZP_G - ZP_BP.  =>  alpha = (k - (m_G - m_BP)) / (2.5*log10(lambda_BP/lambda_G)).

Then Monte-Carlo D_kin (like Oayda): sample alpha from the empirical near-limit
distribution and x from N(x, sigma_x), 50000x -> mean & 16/84 percentiles. Cross-check
against Oayda's D_kin ~ 0.0080 (low) / 0.0068 (high).
Gentle: single column-subset read, no maps.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from astropy.io import fits

CAT = "data/quaia/quaia_G20.5.fits"
V_CMB = 369.82; BETA = V_CMB / 299792.458

# Gaia DR3 Vega zero-points & mean wavelengths (Riello+2021, as used by Oayda+2024)
ZP_G, ZP_BP = 25.6873688, 25.3385422
LAM_G, LAM_BP = 621.8, 511.0      # nm (mean wavelengths)
K = ZP_G - ZP_BP
# Oayda+2024 eq(5): alpha = (k - m_{G-BP}) / (2.5 log10(nu_BP/nu_G)); nu_BP/nu_G = lam_G/lam_BP
DEN = 2.5 * np.log10(LAM_G / LAM_BP)


def alpha_from_colour(mg, mbp):
    return (K - (mg - mbp)) / DEN


def measure_x(g, gmax, lo, hi, nbin=14):
    """slope s of log10(dN/dG) over [lo,hi] (below the rollover); x = 2.5 s."""
    sub = g[(g >= lo) & (g < hi)]
    edges = np.linspace(lo, hi, nbin + 1)
    cnt, _ = np.histogram(sub, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt > 0
    y = np.log10(cnt[ok]); xx = ctr[ok]
    # weighted LS (Poisson: var(log10 N) ~ (1/ln10)^2 / N)
    w = cnt[ok] * (np.log(10) ** 2)
    A = np.column_stack([np.ones(xx.size), xx])
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ (A.T @ W @ y)
    s, ss = coef[1], np.sqrt(cov[1, 1])
    return 2.5 * s, 2.5 * ss, ctr, cnt


def report(tag, g, gmax, xlo, xhi, mg, mbp, rng):
    x, xerr, ctr, cnt = measure_x(g, gmax, xlo, xhi)
    # alpha near the flux limit (within 0.5 mag of the cut)
    near = (g > gmax - 0.5) & (g <= gmax) & np.isfinite(mg) & np.isfinite(mbp)
    a = alpha_from_colour(mg[near], mbp[near])
    a = a[np.isfinite(a)]
    a = a[(a > -2) & (a < 6)]               # drop pathological colours
    amed, amean, astd = np.median(a), a.mean(), a.std()
    # MC D_kin: sample alpha empirically, x ~ N(x,xerr)
    n = 50000
    xs = rng.normal(x, xerr, n)
    als = rng.choice(a, n, replace=True)
    Dk = (2 + xs * (1 + als)) * BETA
    print(f"\n--- {tag} (G<{gmax}) ---  fit window G in [{xlo},{xhi}], N_fit={cnt.sum():,}")
    print(f"  x  = {x:.3f} +/- {xerr:.3f}   (slope s={x/2.5:.4f}/mag)")
    print(f"  alpha (within 0.5 mag of limit, N={a.size:,}): "
          f"median={amed:+.3f} mean={amean:+.3f} std={astd:.3f}")
    print(f"  D_kin = {Dk.mean():.4f}  [16,84]=({np.percentile(Dk,16):.4f},{np.percentile(Dk,84):.4f})  "
          f"D/Dkin_ref: Oayda {'0.0080' if gmax<=20.0 else '0.0068'}")
    return dict(x=x, xerr=xerr, alpha_med=float(amed), alpha_mean=float(amean),
                Dk=float(Dk.mean()), Dk16=float(np.percentile(Dk, 16)),
                Dk84=float(np.percentile(Dk, 84)))


def main():
    rng = np.random.default_rng(3)
    with fits.open(CAT, memmap=True) as f:
        g = np.asarray(f[1].data["phot_g_mean_mag"], float)
        bp = np.asarray(f[1].data["phot_bp_mean_mag"], float)
    print(f"loaded {g.size:,} Quaia sources; beta={BETA:.5f}")
    print("EB: D_kin = [2 + x(1+alpha)] beta")
    r_high = report("Quaia HIGH", g, 20.5, 19.5, 20.3, g, bp, rng)
    r_low = report("Quaia LOW", g[g < 20.0], 20.0, 19.0, 19.8,
                   g[g < 20.0], bp[g < 20.0], rng)
    print("\n(placeholder used in M1/M2 was x=1.7, alpha=1.0 -> D_kin=0.0067)")


if __name__ == "__main__":
    main()
