"""Face-on movie of T CrB's disk through the fitted accretion history, 2005 -> now (EXP-2026-Y).

Reads data/tcrb/inversion_history.npz (written by `tcrb_analysis.py inversion`): surface density,
effective temperature and truncation radius on the disk grid every few time steps. Each frame shows
the disk face-on in blackbody colours of the computed T_eff (brightness on a log-T display scale),
the white dwarf, the inner truncation hole (dashed), and a timeline of the inner-edge radius.
The inner-edge history is robust across inversion variants; the absolute Mdot is not (see EXP-2026-Y).
Output: docs/figures/tcrb/tcrb_accretion_movie.gif
Usage: python scripts/novae/tcrb_accretion_movie.py [n_frames]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "novae"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from celestrium.novae.tcrb import MSUN, RSUN, YR  # noqa: E402
from tcrb_render_system import blackbody_rgb  # noqa: E402

FIG = ROOT / "docs" / "figures" / "tcrb"
EVENTS = [(2015.1, "high state begins"), (2023.33, "2023 dip"), (2026.74, "Sep 2026 UV collapse")]


def disk_image(r, teff, r_tr, r_view, npx=320):
    """Face-on RGB image of the inner disk out to r_view (cm); axes in units of r_view."""
    x = np.linspace(-1.0, 1.0, npx)
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y) * r_view
    T = np.interp(np.log(np.clip(R, r[0], None)), np.log(r), teff, left=np.nan, right=np.nan)
    T = np.where((R < r_tr) | (R > r[-1]), np.nan, T)
    rgb = blackbody_rgb(np.nan_to_num(T, nan=1000.0))
    glow = np.clip((np.log10(np.clip(T, 300, None)) - 3.0) / (4.6 - 3.0), 0.0, 1.0)
    glow = np.where(np.isfinite(T), np.maximum(glow, 0.06), 0.0)
    return rgb * glow[..., None]


def main(n_frames: int = 90):
    import imageio.v2 as imageio
    h = np.load(ROOT / "data" / "tcrb" / "inversion_history.npz")
    t, r, teff, macc, rtr = h["t_yr"], h["r"], h["teff"], h["mdot_acc"], h["r_trunc"]
    mdot = macc  # stored in Msun/yr
    r_view = 8.0 * RSUN
    sel = np.unique(np.linspace(0, len(t) - 1, n_frames).round().astype(int))
    frames, tmp = [], FIG / "_movie_frame.png"
    for k in sel:
        fig = plt.figure(figsize=(8, 6.4), facecolor="black")
        ax = fig.add_axes([0.18, 0.30, 0.64, 0.66])
        ax.imshow(disk_image(r, teff[k], rtr[k], r_view), extent=[-1, 1, -1, 1], origin="lower")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rtr[k] / r_view * np.cos(th), rtr[k] / r_view * np.sin(th), color="#9ecbff", lw=0.6, ls="--")
        ax.plot(0, 0, "o", color="#cfe8ff", ms=3)
        ax.set_axis_off()
        ax.plot([0.5, 0.5 + 2 * RSUN / r_view], [-0.95, -0.95], color="white", lw=1.5)
        ax.text(0.5 + RSUN / r_view, -0.9, r"2 R$_\odot$", color="white", fontsize=8, ha="center")
        tmax = np.nanmax(teff[k][r >= rtr[k]]) if np.any(r >= rtr[k]) else np.nan
        fig.text(0.03, 0.95, f"T CrB inner disk, face-on   {t[k]:.2f}", color="white", fontsize=12)
        fig.text(0.03, 0.91, f"model Mdot onto WD = {mdot[k]:.1e} Msun/yr (absolute scale model-dependent)", color="#ffcc66", fontsize=9)
        fig.text(0.03, 0.875, f"inner edge = {rtr[k] / RSUN:.2f} Rsun   T_max = {tmax:,.0f} K", color="#9ecbff",
                 fontsize=9)
        bx = fig.add_axes([0.1, 0.07, 0.84, 0.18], facecolor="black")
        bx.plot(t, rtr / RSUN, color="#9ecbff", lw=1.2)
        bx.invert_yaxis()
        bx.axvline(t[k], color="white", lw=0.8)
        for te, lab in EVENTS:
            if t[0] < te < t[-1]:
                bx.axvline(te, color="#777777", lw=0.6, ls=":")
                bx.text(te, (rtr / RSUN).min() * 0.9, lab, color="#aaaaaa", fontsize=6.5, ha="center")
        bx.tick_params(colors="white", labelsize=8)
        for sp in bx.spines.values():
            sp.set_color("#555555")
        bx.set_ylabel("inner edge (Rsun)", color="white", fontsize=8)
        fig.savefig(tmp, dpi=80, facecolor="black")
        plt.close(fig)
        frames.append(imageio.imread(tmp))
    tmp.unlink()
    imageio.mimsave(FIG / "tcrb_accretion_movie.gif", frames, duration=0.09, loop=0)
    print("wrote tcrb_accretion_movie.gif", len(frames), "frames")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 90)
