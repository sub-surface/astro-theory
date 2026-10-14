"""EXP-2026-Z: vetting montages for candidate events (plate stamps + light curve around the event).

For each candidate: the position-assembled light curve +-3 yr around t0 (detections, limits, the D1 model), and
DASCH plate cutouts (+-90") from up to 4 event plates (different series first) and 2 earlier reference plates.
What to look for: a stellar image at the marked position on the event plates that is absent on the references,
with a PSF like its neighbours'; defects, emulsion flaws, ghosts, and asteroids (ecliptic) are the usual impostors.
Usage: python scripts/novae/rn_vet.py T0123 [T0456 ...]   (reads data/rn/candidates.csv for t0/t2)
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "novae"))
from celestrium.archives import dasch  # noqa: E402
from celestrium.novae import plates as P  # noqa: E402
from rn_search import load  # noqa: E402

RN = ROOT / "data" / "rn"
OUT = RN / "vet"


def stamp(ax, series, plate, sol, ra, dec, title, half_arcsec=90.0):
    try:
        b = dasch.cutout_fits(series, plate, sol, ra, dec)
        h = fits.open(io.BytesIO(b))[0]
        w = WCS(h.header)
        x, y = w.world_to_pixel_values(ra, dec)
        scale = abs(w.pixel_scale_matrix[0, 0]) * 3600 or 1.0
        r = int(half_arcsec / scale)
        x, y = int(round(float(x))), int(round(float(y)))
        im = h.data[max(0, y - r):y + r, max(0, x - r):x + r]
        lo, hi = np.nanpercentile(im, [5, 99.5])
        ax.imshow(im, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        cy, cx = min(r, y), min(r, x)
        ax.add_patch(plt.Circle((cx, cy), 10 / scale, fill=False, color="tab:red", lw=0.8))
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"no cutout\n{str(e)[:40]}", ha="center", va="center", fontsize=6, transform=ax.transAxes)
    ax.set_title(title, fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])


def vet(tid, row, T, I, NB):
    tr = T.loc[tid]
    ix = I.loc[tid]
    nbr = NB[NB.tid == tid].to_dict("records")
    lc = load(tid, "t", [x for x in nbr if x["role"] == "t"], (tr.ra, tr.dec))
    own = dasch.parse_csv_lines(dasch.cached(RN / "lc" / f"{tid}_t.csv.gz", None))
    t0, t2 = float(row.d1_t0), float(row.d1_t2)
    t_end = t0 + 10 * t2 / 365.25
    g = lc["good"]
    ev = g & lc["det"] & (lc["t"] >= t0 - 3 / 365.25) & (lc["t"] <= t_end)
    fig = plt.figure(figsize=(8.0, 5.2))
    ax = fig.add_axes([0.07, 0.56, 0.9, 0.38])
    w = g & (np.abs(lc["t"] - t0) < 3)
    d = w & lc["det"]
    ax.scatter(lc["t"][w & ~lc["det"]], lc["lim"][w & ~lc["det"]], marker="v", s=6, c="0.75", lw=0, label="limit")
    ax.scatter(lc["t"][d], lc["mag"][d], s=8, c="k", lw=0, label="detection")
    ax.scatter(lc["t"][ev], lc["mag"][ev], s=14, facecolors="none", edgecolors="tab:red", lw=0.7, label="event plates")
    tt = np.linspace(t0 - 0.02, t_end, 300)
    ax.plot(tt, -2.5 * np.log10(10 ** (-0.4 * row.d1_mpeak) * P.template_flux((tt - t0) * 365.25, t2)), "tab:red", lw=0.8)
    ax.invert_yaxis()
    ax.set_title(f"{tid} {tr['name']} ({tr.cls})  D1={row.d1:.0f}  t0={t0:.3f}  t2={t2:g} d  peak={row.d1_mpeak:.1f}  "
                 f"nights={row.d1_nights}  known={tr.known_eruptions}", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.set_xlabel("year")
    # choose plates: event detections from different series first, then 2 deep references 30-400 d before
    oi = {}
    ot = 2000 + (own["date_jd"] - 2451545) / 365.25 if own else np.array([])
    picks = []
    seen = set()
    for i in np.where(ev)[0][np.argsort(lc["mag"][ev])]:
        s = lc["series"][i]
        if s in seen and len(picks) < 2:
            continue
        seen.add(s)
        picks.append(("event", lc["series"][i], int(lc["plate"][i]), lc["t"][i], lc["mag"][i]))
        if len(picks) >= 4:
            break
    ref = g & ~lc["det"] & (lc["t"] < t0 - 30 / 365.25) & (lc["t"] > t0 - 400 / 365.25)
    for i in np.where(ref)[0][np.argsort(-lc["lim"][ref])][:2]:
        picks.append(("ref", lc["series"][i], int(lc["plate"][i]), lc["t"][i], lc["lim"][i]))
    for k, (kind, s, pl, tp, mg) in enumerate(picks[:6]):
        a = fig.add_axes([0.02 + k * 0.163, 0.04, 0.15, 0.4])
        sol = 0
        if own:
            m = (own["series"] == s) & (own["plate_number"] == pl)
            if m.any():
                sol = int(np.nan_to_num(own["solution_number"][m][0]))
        stamp(a, s, pl, sol, tr.ra, tr.dec, f"{kind} {s}{pl:05d}\n{tp:.3f}  {'m' if kind == 'event' else 'lim'}={mg:.1f}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{tid}.png", dpi=140)
    plt.close(fig)
    return OUT / f"{tid}.png"


def main():
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    I = pd.read_csv(RN / "fetch_index.csv").drop_duplicates("tid", keep="last").set_index("tid", drop=False)
    NB = pd.read_csv(RN / "neighbours.csv").drop_duplicates()
    C = pd.read_csv(RN / "candidates.csv").set_index("tid", drop=False)
    for tid in sys.argv[1:]:
        print(vet(tid, C.loc[tid], T, I, NB))


if __name__ == "__main__":
    main()
