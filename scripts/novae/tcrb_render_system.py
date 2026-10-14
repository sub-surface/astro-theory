"""Render T CrB to scale: Roche-filling M giant, L1 stream, viscous disk, white dwarf (EXP-2026-Y).

Outputs (docs/figures/tcrb/):
  * tcrb_roche_map.png      orbital-plane Roche potential, lobes, L-points, stream, disk
  * tcrb_system_states.png  inclined (i = 61 deg) blackbody-colour views: quiescence vs high state
  * tcrb_orbit.gif          one orbit at i = 61 deg with the measured quiescent V fold (and the Roche model)
Disk temperatures come from the fitted accretion history when data/tcrb/inversion_states.npz
exists (written by scripts/tcrb_analysis.py); otherwise from steady-state disks at fiducial Mdot.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celestrium.novae.tcrb import FIDUCIAL, MSUN, RSUN, YR, ViscousDisk  # noqa: E402
from celestrium.novae.roche import (ballistic_stream, donor_surface, ellipsoidal_lightcurve, grad_potential,  # noqa: E402
                                   l1_point, potential)

FIG = ROOT / "docs" / "figures" / "tcrb"
FIG.mkdir(parents=True, exist_ok=True)
Q = FIDUCIAL.m_giant / FIDUCIAL.m_wd
A_RSUN = FIDUCIAL.a_orb / RSUN
INC = FIDUCIAL.incl_deg


def blackbody_rgb(T):
    """Approximate sRGB colour of a blackbody (Tanner Helland fit), T in K -> (..., 3) in [0, 1]."""
    t = np.clip(np.asarray(T, float), 1000, 40000) / 100.0
    t = np.where(np.isfinite(t), t, 10.0)
    r = np.where(t <= 66, 255.0, 329.698727446 * (t - 60) ** -0.1332047592)
    g = np.where(t <= 66, 99.4708025861 * np.log(t) - 161.1195681661, 288.1221695283 * (t - 60) ** -0.0755148492)
    b = np.where(t >= 66, 255.0, np.where(t <= 19, 0.0, 138.5177312231 * np.log(np.clip(t - 10, 1e-9, None)) - 305.0447927307))
    return np.clip(np.stack([r, g, b], -1) / 255.0, 0, 1)


def _glow(T):
    """Display brightness on a log-temperature scale (true surface brightness ~T^4 would hide the disk)."""
    return float(np.clip((math.log10(max(T, 300.0)) - 2.6) / (4.4 - 2.6), 0.05, 1.0))


def disk_temperature(mdot_msun_yr: float):
    d = ViscousDisk(FIDUCIAL, 3.0, 1.0, n_cells=120)
    sig = d.steady_state() * mdot_msun_yr * MSUN / YR
    return d.grid.r / FIDUCIAL.a_orb, d.teff(sig)


def roche_map(states):
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="black")
    ax.set_facecolor("black")
    x = np.linspace(-0.75, 1.75, 900)
    y = np.linspace(-0.9, 0.9, 650)
    X, Y = np.meshgrid(x, y)
    P = potential(X, Y, 0 * X, Q)
    xl1 = l1_point(Q)
    pl1 = potential(xl1, 0, 0, Q)
    lv = np.sort(pl1 + np.array([-0.6, -0.3, -0.15, 0.0, 0.05, 0.1, 0.2]))
    ax.contour(X, Y, P, levels=lv, colors="#3a5a8a", linewidths=0.6)
    ax.contour(X, Y, P, levels=[pl1], colors="#9fc3ff", linewidths=1.4)
    s = donor_surface(Q, 1.0, 60, 120)
    T = 3600 * (s["g"] / s["g"].max()) ** 0.08
    z0 = np.abs(s["xyz"][..., 2]) < 0.02
    # filled donor cross-section
    th = np.linspace(0, 2 * np.pi, 400)
    rr = []
    for t in th:
        from scipy.optimize import brentq
        f = lambda r: potential(1 + r * math.cos(t), r * math.sin(t), 0.0, Q) - pl1  # noqa: E731
        # never cross the L1 plane x = xl1 into the white dwarf's lobe
        hi = 0.9 if math.cos(t) >= 0 else min(0.9, (1 - xl1) / abs(math.cos(t)) - 1e-6)
        try:
            rr.append(brentq(f, 1e-3, hi))
        except ValueError:
            rr.append(1 - xl1)
    rr = np.array(rr)
    ax.fill(1 + rr * np.cos(th), rr * np.sin(th), color=blackbody_rgb(3500), alpha=0.85, zorder=2)
    # disk (high state colours)
    rq, Tq = states["high"]
    for i in range(len(rq) - 1, 0, -1):
        ax.add_patch(plt.Circle((0, 0), rq[i], color=blackbody_rgb(max(Tq[i], 1500)) * _glow(Tq[i]), zorder=3))
    ax.plot(0, 0, "o", color="white", ms=3, zorder=5)
    path, imp, az = ballistic_stream(Q, 58 / A_RSUN)
    ax.plot(path[:, 0], path[:, 1], color="#ffcc66", lw=1.5, zorder=4)
    ax.plot(*imp, "*", color="#ffee88", ms=12, zorder=6)
    # Lagrange points
    from scipy.optimize import brentq
    gx = lambda xx: grad_potential(np.array(xx), 0.0, 0.0, Q)[0]  # noqa: E731
    l2 = brentq(gx, 1.0001, 2.0)
    l3 = brentq(gx, -1.5, -1e-3)
    for lx, nm in ((xl1, "L1"), (l2, "L2"), (l3, "L3")):
        ax.plot(lx, 0, "+", color="#9fc3ff")
        ax.text(lx, -0.06, nm, color="#9fc3ff", ha="center", fontsize=9)
    ax.plot(1 / (1 + 1 / Q), 0, "x", color="#888888")
    ax.text(0, -0.36, "white dwarf 1.35 M$_\\odot$\n+ disk (R = 58 R$_\\odot$)", color="white", ha="center", fontsize=9)
    ax.text(1, -0.45, "M3III giant 0.93 M$_\\odot$\nfills Roche lobe (R$_L$ = %.0f R$_\\odot$)" % (0.3472 * A_RSUN), color="white", ha="center", fontsize=9)
    ax.text(imp[0] + 0.04, imp[1] + 0.03, "hot spot\nazimuth %.0f$^\\circ$" % az, color="#ffee88", fontsize=8)
    sc = A_RSUN
    ax.plot([1.2, 1.2 + 50 / sc], [0.78, 0.78], color="white")
    ax.text(1.2 + 25 / sc, 0.80, "50 R$_\\odot$ (0.23 au)", color="white", ha="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(-0.9, 0.9)
    ax.axis("off")
    ax.set_title("T CrB to scale in the co-rotating frame (a = %.0f R$_\\odot$, P = 227.6 d)" % A_RSUN, color="white")
    fig.savefig(FIG / "tcrb_roche_map.png", dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def project(xyz, phase, inc_deg):
    """Sky-plane coordinates (u, v) and depth w (towards observer) for orbital phase."""
    i = math.radians(inc_deg)
    a = 2 * math.pi * phase
    obs = np.array([math.cos(a) * math.sin(i), math.sin(a) * math.sin(i), math.cos(i)])
    zhat = np.array([0, 0, 1.0])
    e1 = np.cross(zhat, obs)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(obs, e1)
    return xyz @ e1, xyz @ e2, xyz @ obs, obs


def draw_view(ax, phase, disk_state, surf, title=None):
    ax.set_facecolor("black")
    rq, Tq = disk_state
    polys, cols, depth = [], [], []
    # donor facets
    xyz = surf["xyz"]
    T = 3600 * (surf["g"] / surf["g"].max()) ** 0.08
    u, v, w, obs = project(xyz.reshape(-1, 3), phase, INC)
    u, v, w = u.reshape(xyz.shape[:2]), v.reshape(xyz.shape[:2]), w.reshape(xyz.shape[:2])
    mu = surf["normal"] @ obs
    nt, npp = xyz.shape[:2]
    for i in range(nt - 1):
        for j in range(npp):
            jj = (j + 1) % npp
            m = mu[i, j]
            if m <= 0:
                continue
            polys.append([(u[i, j], v[i, j]), (u[i + 1, j], v[i + 1, j]), (u[i + 1, jj], v[i + 1, jj]), (u[i, jj], v[i, jj])])
            ld = 1 - 0.75 * (1 - m)
            cols.append(np.clip(blackbody_rgb(T[i, j]) * (0.25 + 0.75 * ld), 0, 1))
            depth.append(w[i, j])
    # disk annuli as facets (thin, flared slightly)
    phs = np.linspace(0, 2 * np.pi, 90)
    for k in range(0, len(rq) - 1, 2):
        r0, r1 = rq[k], rq[min(k + 2, len(rq) - 1)]
        if r1 < 0.004:
            continue
        for j in range(len(phs) - 1):
            pts = np.array([[r0 * np.cos(phs[j]), r0 * np.sin(phs[j]), 0], [r1 * np.cos(phs[j]), r1 * np.sin(phs[j]), 0],
                            [r1 * np.cos(phs[j + 1]), r1 * np.sin(phs[j + 1]), 0], [r0 * np.cos(phs[j + 1]), r0 * np.sin(phs[j + 1]), 0]])
            pu, pv, pw, _ = project(pts, phase, INC)
            polys.append(list(zip(pu, pv)))
            Tk = max(Tq[k], 800)
            cols.append(np.clip(blackbody_rgb(Tk) * _glow(Tk), 0, 1))
            depth.append(pw.mean())
    o = np.argsort(depth)
    pc = PolyCollection([polys[i] for i in o], facecolors=[cols[i] for i in o], edgecolors="none", antialiased=False)
    ax.add_collection(pc)
    wu, wv, ww, _ = project(np.zeros((1, 3)), phase, INC)
    ax.plot(wu, wv, "o", color="white", ms=2.5)
    cu, cv, _, _ = project(np.array([[Q / (1 + Q), 0, 0]]), phase, INC)
    ax.set_xlim(cu[0] - 1.0, cu[0] + 1.0)
    ax.set_ylim(cv[0] - 0.62, cv[0] + 0.62)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=10)


def load_states():
    f = ROOT / "data" / "tcrb" / "inversion_states.npz"
    if f.exists():
        d = np.load(f)
        return {"quiet": (d["r_a"], d["T_quiet"]), "high": (d["r_a"], d["T_high"]), "now": (d["r_a"], d["T_now"]),
                "labels": {k: str(d[f"label_{k}"]) for k in ("quiet", "high", "now")}}
    return {"quiet": disk_temperature(2e-9), "high": disk_temperature(1e-7), "now": disk_temperature(2e-8),
            "labels": {"quiet": "steady disk, 2e-9 Msun/yr", "high": "steady disk, 1e-7 Msun/yr", "now": "steady disk, 2e-8 Msun/yr"}}


def quiescent_fold(n_bins: int = 20):
    """AAVSO V 2006-2014.5 (cold disk, giant-dominated), secular trend removed, folded on our ephemeris.

    Phase 0 = deeper V minimum, which falls at phase 0.99 of the literature light-minimum (giant in front) ephemeris.
    Returns bin centres, median offsets (mean-subtracted) and their errors, plus the raw fold.
    """
    import json
    sys.path.insert(0, str(ROOT / "scripts" / "novae"))
    from tcrb_analysis import load_aavso  # noqa: E402
    from celestrium.novae import tcrb
    from celestrium.novae.tcrb import year_to_jd
    eph = json.loads((ROOT / "docs" / "results" / "experiment_y_tcrb_results.json").read_text())["ephemeris"]
    V = load_aavso()[0]["V"]
    t, m, e = V["jd"], V["mag"], np.hypot(V["err"], 0.04)
    q = (t > year_to_jd(2006.0)) & (t < year_to_jd(2014.5))
    fq = tcrb.fit_ephemeris(t[q], m[q], e[q], [eph["period_d"]], knot_spacing_d=150.0)
    orb = m[q] - fq["trend"]
    orb -= np.median(orb)
    ph = tcrb.orbital_phase(t[q], eph["t_deeper_min_jd"], eph["period_d"])
    edges = np.linspace(0, 1, n_bins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(orb[(ph >= a) & (ph < b)]) for a, b in zip(edges[:-1], edges[1:])])
    err = np.array([1.253 * np.std(orb[(ph >= a) & (ph < b)]) / np.sqrt(((ph >= a) & (ph < b)).sum())
                    for a, b in zip(edges[:-1], edges[1:])])
    off = med.mean()
    return mid, med - off, err, ph, orb - off


def main():
    states = load_states()
    roche_map(states)
    surf = donor_surface(Q, 1.0, 40, 80)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="black")
    for ax, k in zip(axs, ("quiet", "high", "now")):
        draw_view(ax, 0.30, states[k], surf, f"{k}: {states['labels'][k]}")
    fig.suptitle("T CrB at i = 61$^\\circ$, orbital phase 0.30 - blackbody colours (display brightness on a log-T scale)", color="white")
    fig.savefig(FIG / "tcrb_system_states.png", dpi=140, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    # orbit animation with synthetic ellipsoidal light curve
    import imageio.v2 as imageio
    ph_lc, m_lc, _ = ellipsoidal_lightcurve(Q, INC, 5450, surf=donor_surface(Q, 1.0, 48, 96))
    m_lc = np.asarray(m_lc) - np.mean(m_lc)
    f_mid, f_med, f_err, f_ph, f_orb = quiescent_fold()
    frames = []
    tmp = FIG / "_frame.png"
    for k, p in enumerate(np.linspace(0, 1, 48, endpoint=False)):
        fig = plt.figure(figsize=(8, 6), facecolor="black")
        ax = fig.add_axes([0, 0.28, 1, 0.7])
        draw_view(ax, p, states["high"], surf)
        bx = fig.add_axes([0.12, 0.07, 0.8, 0.18], facecolor="black")
        bx.plot(f_ph, f_orb, ".", ms=1.2, color="#555555")
        bx.plot(ph_lc, m_lc, color="#8899aa", lw=0.9, ls="--", label="Roche model, blackbody, i = 61°")
        bx.errorbar(f_mid, f_med, f_err, fmt="o", ms=3, lw=0.8, color="#ffcc66", label="AAVSO V 2006-14, folded")
        bx.axvline(p, color="white", lw=0.8)
        bx.set_ylim(0.32, -0.32)
        bx.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, labelcolor="white")
        bx.tick_params(colors="white", labelsize=8)
        for sp in bx.spines.values():
            sp.set_color("#555555")
        bx.set_xlabel("orbital phase (0 = giant in front)", color="white", fontsize=8)
        bx.set_ylabel("$\\Delta V$ (mag)", color="white", fontsize=8)
        fig.text(0.02, 0.95, "T CrB  i = 61$^\\circ$  P = 227.6 d", color="white", fontsize=10)
        fig.savefig(tmp, dpi=80, facecolor="black")
        plt.close(fig)
        frames.append(imageio.imread(tmp))
    tmp.unlink()
    imageio.mimsave(FIG / "tcrb_orbit.gif", frames, duration=0.08, loop=0)
    print("wrote", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
