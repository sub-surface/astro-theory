r"""
Roche geometry & synthetic ellipsoidal light curves for T CrB (EXP-2026-Y).

Units: separation a = 1, total mass = 1, synchronous rotation. Accretor (WD, mass
fraction mu1) at x=0, donor (M giant, q = M_d / M_wd) at x=1. Dimensionless potential
    Phi(r) = -mu_wd/|r - r_wd| - mu_d/|r - r_d| - 0.5 [(x - x_cm)^2 + y^2]
The donor fills its Roche lobe (Munari+ 2025), so its surface is Phi = Phi(L1).
Local T from gravity darkening T ∝ g^beta (Lucy 1967, beta = 0.08 convective envelope);
linear limb darkening; blackbody intensities at band wavelengths.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq

H, C, KB = 6.62607e-27, 2.99792458e10, 1.380649e-16


def _mu(q: float) -> Tuple[float, float]:
    """(mu_wd, mu_donor) mass fractions for q = M_donor / M_wd."""
    return 1.0 / (1.0 + q), q / (1.0 + q)


def potential(x, y, z, q: float):
    mu1, mu2 = _mu(q)
    xcm = mu2  # centre of mass at x = mu2 (WD at 0, donor at 1)
    r1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1) ** 2 + y ** 2 + z ** 2)
    return -mu1 / r1 - mu2 / r2 - 0.5 * ((x - xcm) ** 2 + y ** 2)


def grad_potential(x, y, z, q: float):
    mu1, mu2 = _mu(q)
    xcm = mu2
    r1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1) ** 2 + y ** 2 + z ** 2)
    gx = mu1 * x / r1 ** 3 + mu2 * (x - 1) / r2 ** 3 - (x - xcm)
    gy = mu1 * y / r1 ** 3 + mu2 * y / r2 ** 3 - y
    gz = mu1 * z / r1 ** 3 + mu2 * z / r2 ** 3
    return gx, gy, gz


def l1_point(q: float) -> float:
    """x of L1 between WD (0) and donor (1)."""
    f = lambda x: grad_potential(np.array(x), 0.0, 0.0, q)[0]  # noqa: E731
    return brentq(f, 1e-4, 1 - 1e-4)


def donor_surface(q: float, fill: float = 1.0, n_theta: int = 72, n_phi: int = 144):
    """Points on the donor equipotential (fill = 1 -> Roche-lobe filling).

    Spherical grid centred on the donor; polar axis along -x (towards L1).
    Returns xyz (n_theta, n_phi, 3), outward unit normals, |g|, and area elements.
    """
    xl1 = l1_point(q)
    phi_l1 = float(potential(xl1, 0.0, 0.0, q))
    # fill < 1: equipotential with the same polar radius fraction of the lobe
    th = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    dx, dy, dz = -np.cos(TH), np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH)
    r = np.empty_like(TH)
    rmax = 1 - xl1
    target = phi_l1
    if fill < 1.0:
        # potential at fraction `fill` of the polar (z-direction) lobe radius
        rp = brentq(lambda s: potential(1.0, 0.0, s, q) - phi_l1, 1e-3, 1.0)
        target = float(potential(1.0, 0.0, fill * rp, q))
    for i in range(n_theta):
        for j in range(n_phi):
            f = lambda s: potential(1 + s * dx[i, j], s * dy[i, j], s * dz[i, j], q) - target  # noqa: E731
            hi = rmax * (0.9999 if (fill >= 1.0 and i == 0) else 1.0)
            hi = min(hi, 1.5)
            try:
                r[i, j] = brentq(f, 1e-3, hi if f(hi) > 0 else 1.2)
            except ValueError:
                r[i, j] = hi
    xyz = np.stack([1 + r * dx, r * dy, r * dz], -1)
    gx, gy, gz = grad_potential(xyz[..., 0], xyz[..., 1], xyz[..., 2], q)
    g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    n = np.stack([gx, gy, gz], -1) / g[..., None]  # grad Phi points outward (Phi increases outward)
    # area elements from finite differences of the parametrisation
    dth = th[1] - th[0]
    dph = ph[1] - ph[0]
    d_th = np.gradient(xyz, dth, axis=0)
    d_ph = np.gradient(xyz, dph, axis=1)
    if True:
        d_ph = (np.roll(xyz, -1, axis=1) - np.roll(xyz, 1, axis=1)) / (2 * dph)
    dA = np.linalg.norm(np.cross(d_th, d_ph), axis=-1) * dth * dph
    return {"xyz": xyz, "normal": n, "g": g, "dA": dA, "xl1": xl1}


def planck_lambda(lam_A: float, T):
    lam = lam_A * 1e-8
    return 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * KB * np.asarray(T)))


def ellipsoidal_lightcurve(q: float, incl_deg: float, lam_A: float = 5450.0, t_pole: float = 3600.0,
                           beta: float = 0.08, u_ld: float = 0.75, n_phase: int = 100, fill: float = 1.0,
                           surf: Dict = None, k_teff: float = None):
    """Synthetic donor light curve. Phase 0 = donor in front (inferior conjunction of the giant).

    k_teff: if given, local intensity I ∝ T^k_teff instead of a blackbody. M giants in V are
    dominated by TiO bands whose depth grows steeply as T falls: from M3III -> M4III colours
    (Teff ~3650 -> 3550 K, Delta V ~ +0.5 mag) d ln F_V / d ln T ~ 15-20, versus ~7 for a
    blackbody at 5450 A and 3600 K. This boosts the gravity-darkening part of the amplitude.
    """
    s = surf or donor_surface(q, fill)
    g = s["g"]
    T = t_pole * (g / g.max()) ** beta
    I0 = planck_lambda(lam_A, T) if k_teff is None else (T / t_pole) ** k_teff
    i = math.radians(incl_deg)
    phases = np.linspace(0, 1, n_phase, endpoint=False)
    flux = np.empty(n_phase)
    for k, p in enumerate(phases):
        # observer direction in the co-rotating frame. Phase 0: donor between observer and WD
        # => observer on the +x side (donor at x=1 is nearer the observer).
        a = 2 * math.pi * p
        obs = np.array([math.cos(a) * math.sin(i), math.sin(a) * math.sin(i), math.cos(i)])
        mu = s["normal"] @ obs
        vis = mu > 0
        flux[k] = np.sum((I0 * (1 - u_ld * (1 - mu)) * mu * s["dA"])[vis])
    mag = -2.5 * np.log10(flux / flux.mean())
    return phases, mag, s


def harmonic_amplitudes(phases, mag):
    """Fourier coefficients (a1, b1, a2, b2) of mag(phase)."""
    a = 2 * np.pi * phases
    X = np.vstack([np.ones_like(a), np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)]).T
    beta, *_ = np.linalg.lstsq(X, mag, rcond=None)
    return beta[1:]


def ballistic_stream(q: float, r_disk: float, v0: float = 0.02, t_max: float = 3.0, n: int = 3000):
    """Restricted-3-body trajectory from L1 (rotating frame, Omega = 1).

    Returns (xy path up to impact with a disk of radius r_disk around the WD, impact point,
    impact azimuth measured from the WD-donor line in the orbital direction, degrees).
    """
    from scipy.integrate import solve_ivp
    xl1 = l1_point(q)

    def rhs(t, s):
        x, y, vx, vy = s
        gx, gy, _ = grad_potential(np.array(x), np.array(y), np.array(0.0), q)
        return [vx, vy, -gx + 2 * vy, -gy - 2 * vx]

    def hit(t, s):
        return math.hypot(s[0], s[1]) - r_disk
    hit.terminal, hit.direction = True, -1
    sol = solve_ivp(rhs, (0, t_max), [xl1 - 1e-3, 0.0, -v0, 0.0], events=hit, max_step=t_max / n, rtol=1e-9, atol=1e-11)
    path = sol.y[:2].T
    imp = path[-1]
    az = math.degrees(math.atan2(imp[1], imp[0]))
    return path, imp, az


def circularization_radius(q: float) -> float:
    """Hessman & Hopp (1990) fit: R_circ / a for q_wd = M_wd/M_d."""
    qq = 1.0 / q
    return 0.0883 - 0.04858 * math.log10(qq) + 0.11489 * math.log10(qq) ** 2 - 0.020475 * math.log10(qq) ** 3
