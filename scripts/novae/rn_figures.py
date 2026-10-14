"""EXP-2026-Z figures: G0 (T CrB on the Harvard plates), score distributions and recall, completeness, DE Cir.

Usage: python scripts/novae/rn_figures.py [g0|scores|completeness|decir|v347nor|v347nor_onset|all]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from celestrium.archives import dasch  # noqa: E402
from celestrium.novae import plates as P  # noqa: E402,F401

FIG = ROOT / "docs" / "figures" / "rn"
RN = ROOT / "data" / "rn"
RES = ROOT / "docs" / "results" / "experiment_z_rn_results.json"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _yearly(t, m, lo, hi, step=0.5):
    edges = np.arange(lo, hi + step, step)
    c, med, e = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        w = (t >= a) & (t < b)
        if w.sum() >= 5:
            c.append((a + b) / 2)
            med.append(np.median(m[w]))
            e.append(1.253 * np.std(m[w]) / np.sqrt(w.sum()))
    return np.array(c), np.array(med), np.array(e)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("wrote", FIG / name)


def g0():
    lc = dasch.tidy(dasch.parse_csv_lines(dasch.cached(RN / "lc" / "tcrb_atlas.csv", None)))
    g = lc["good"] & lc["det"]
    t, m, sat = lc["t"][g], lc["mag"][g], lc["sat"][g]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={"height_ratios": [1, 1.2]})
    a = ax[0]
    a.scatter(t[~sat], m[~sat], s=2, c="0.55", lw=0, label="plate detections (good)")
    a.scatter(t[sat], m[sat], s=4, c="tab:orange", lw=0, label="saturated (lower limit on brightness)")
    c, med, e = _yearly(t[~sat], m[~sat], 1890, 1990, 1.0)
    a.plot(c, med, "k-", lw=1, label="yearly median")
    a.axvline(1946.11, color="tab:red", lw=0.8)
    a.text(1946.6, 5.5, "eruption\n1946 Feb 9", color="tab:red", fontsize=8)
    a.set_ylim(13, 4.5)
    a.set_xlim(1888, 1992)
    a.set_ylabel("B-like mag (DASCH magcal_magdep)")
    a.set_title("T CrB on 4,500 Harvard plate measurements, 1890-1990 (DASCH DR7)")
    a.legend(loc="upper right", fontsize=7, frameon=False, markerscale=3)
    a = ax[1]
    w = (t > 1934) & (t < 1948) & ~sat
    a.scatter(t[w], m[w], s=4, c="0.6", lw=0)
    c, med, e = _yearly(t[w], m[w], 1934, 1948, 0.25)
    a.errorbar(c, med, e, fmt="o-", ms=3, color="k", lw=1, capsize=0, label="quarter-year median ± s.e.")
    for (lo, hi, col, lab) in [(1938, 1945, "tab:blue", "high state 1938-45: 0.37 ± 0.04 mag brighter than 1950s"),
                               (1945.3, 1946.0, "tab:purple", "pre-eruption dip: 0.86 ± 0.06 mag below 1943-45")]:
        a.axvspan(lo, hi, color=col, alpha=0.08, lw=0)
        a.plot([], [], color=col, lw=6, alpha=0.3, label=lab)
    a.axvline(1946.11, color="tab:red", lw=0.8)
    a.set_ylim(12.0, 8.8)
    a.set_xlim(1934, 1948)
    a.set_xlabel("year")
    a.set_ylabel("B-like mag")
    a.legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "tcrb_dasch_g0.png")


def scores():
    """Discovery-score distributions (targets vs controls) and recall of catalogued eruptions vs plate depth."""
    S = pd.read_csv(RN / "scores.csv")
    R = pd.read_csv(RN / "recall.csv")
    res = json.loads(RES.read_text(encoding="utf-8"))
    tau = res["thresholds"]["d1"]["0.01"]["tau"]
    cmax = res["R2"]["control_max_d1"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a = ax[0]
    bins = np.r_[0, np.logspace(0, np.log10(max(S.d1.max(), 10) * 1.2), 40)]
    for role, col, lab in (("t", "tab:red", "nova / symbiotic positions"), ("c", "0.4", "matched field stars")):
        x = S[S.role.str.startswith(role)].d1.clip(lower=0.5)
        a.hist(x, bins=bins, histtype="step", color=col, density=True, lw=1.2, label=f"{lab} (n={len(x)})")
    a.axvline(tau, color="k", ls="--", lw=0.8, label="1% control FAR")
    a.axvline(cmax, color="k", ls=":", lw=0.8, label="control maximum")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("D1 discovery score (Δ ln L)")
    a.set_ylabel("density")
    a.legend(fontsize=7, frameon=False)
    a = ax[1]
    Rc = R[R.n_plates_win >= 3].copy()
    Rc["depth"] = Rc.med_lim_win - Rc.vpeak
    edges = np.array([-4, 0, 2, 4, 6, 10])
    for col, lab, c in (("d1", "D1 physics-informed", "tab:blue"), ("d0", "D0 naive", "0.5")):
        th = res["thresholds"][col]["0.01"]["tau"]
        k, n, mid = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (Rc.depth >= lo) & (Rc.depth < hi)
            if m.sum():
                k.append((Rc[m][col] > th).sum())
                n.append(m.sum())
                mid.append((lo + hi) / 2)
        k, n, mid = map(np.array, (k, n, mid))
        p = k / n
        e = np.sqrt(p * (1 - p) / n + 1 / (4 * n * n))
        off = 0.12 if col == "d0" else -0.12
        a.errorbar(mid + off, p, e, fmt="o-" if col == "d1" else "s--", color=c, ms=4, capsize=0, label=lab)
        if col == "d1":
            for x, kk, nn in zip(mid, k, n):
                a.text(x, -0.1, f"{kk}/{nn}", ha="center", fontsize=6, color="0.3")
    a.set_ylim(-0.15, 1.05)
    a.set_xlabel("median plate limit − catalogued peak (mag)")
    a.set_ylabel("recall at 1% control FAR")
    a.set_title("catalogued eruptions on the Harvard plates", fontsize=9)
    a.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "rn_scores_recall.png")


def completeness():
    """Per-nova completeness for hidden recurrence vs recurrence time (R3 injection-recovery)."""
    C = pd.read_csv(RN / "completeness.csv")
    res = json.loads(RES.read_text(encoding="utf-8"))
    tau = res["R3"]["threshold"]
    C["rec"] = (C.score > tau) & (C.nights >= 2)
    fig, a = plt.subplots(figsize=(4.6, 3.2))
    for cls, c, lab in (("typical", "tab:blue", "typical, t₂ 5–100 d"), ("fast", "tab:orange", "fast, t₂ = 3 d")):
        g = C[C.cls == cls].groupby("T").rec
        p, n = g.mean(), g.size()
        a.errorbar(p.index, p.values, np.sqrt(p * (1 - p) / n).values, fmt="o-", ms=3, color=c, capsize=0, label=lab)
    a.set_xscale("log")
    a.set_xticks([10, 20, 30, 50, 100])
    a.set_xticklabels(["10", "20", "30", "50", "100"])
    a.set_ylim(0, 1)
    a.set_xlabel("assumed recurrence time T (yr)")
    a.set_ylabel("P(earlier eruption recovered)")
    a.set_title(f"{C.tid.nunique()} Schaefer (2022) novae, own cadence & peak", fontsize=9)
    a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "rn_completeness.png")


def decir():
    """DE Cir (nova 2003) on the Harvard plates: the century, and the 1929 transient."""
    sys.path.insert(0, str(ROOT / "scripts" / "novae"))
    from rn_search import load
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    NB = pd.read_csv(RN / "neighbours.csv").drop_duplicates()
    row = T.loc["T0072"]
    lc = load("T0072", "t", NB[(NB.tid == "T0072") & (NB.role == "t")].to_dict("records"), (row.ra, row.dec))
    g = lc["good"]
    t, det, mag, lim = lc["t"][g], lc["det"][g], lc["mag"][g], lc["lim"][g]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), gridspec_kw={"height_ratios": [1, 1.1]})
    for a, (lo, hi), s in zip(ax, ((1888, 1992), (1928.9, 1929.75)), (5, 16)):
        w = (t >= lo) & (t <= hi)
        a.scatter(t[w & ~det], lim[w & ~det], marker="v", s=s, c="0.75", lw=0, label="plate limit, not detected")
        a.scatter(t[w & det], mag[w & det], s=s, c="k", lw=0, label="detection at DE Cir")
        a.set_xlim(lo, hi)
        a.set_ylim(17.5, 11.5)
        a.set_ylabel("B-like mag")
    ax[0].axvspan(1929.2, 1929.45, color="tab:red", alpha=0.15, lw=0)
    ax[0].annotate("1929", (1929.3, 12.4), (1940, 12.2), fontsize=8, color="tab:red",
                   arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.6))
    ax[0].set_title("DE Cir (nova 2003) on the Harvard plates (DASCH, position-assembled)", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")
    ax[1].set_xlabel("year (1929)")
    ax[1].axhline(16.0, color="0.5", lw=0.6, ls=":")
    ax[1].text(1929.6, 15.85, "quiescent blend ≈ 16", fontsize=7, color="0.4")
    fig.tight_layout()
    _save(fig, "decir_1929.png")


def v347nor():
    """V347 Nor (D-type symbiotic) on the Harvard plates: the unrecorded 1943 nova eruption."""
    sys.path.insert(0, str(ROOT / "scripts" / "novae"))
    from rn_search import load
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    NB = pd.read_csv(RN / "neighbours.csv").drop_duplicates()
    row = T.loc["T1274"]
    lc = load("T1274", "t", NB[(NB.tid == "T1274") & (NB.role == "t")].to_dict("records"), (row.ra, row.dec))
    g = lc["good"]
    t, det, mag, lim = lc["t"][g], lc["det"][g], lc["mag"][g], lc["lim"][g]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), gridspec_kw={"height_ratios": [1, 1.1]})
    for a, (lo, hi), s in zip(ax, ((1888, 1992), (1942.5, 1947.0)), (4, 12)):
        w = (t >= lo) & (t <= hi)
        a.scatter(t[w & ~det], lim[w & ~det], marker="v", s=s, c="0.78", lw=0, label="plate limit, not detected")
        a.scatter(t[w & det], mag[w & det], s=s, c="k", lw=0, label="detection at V347 Nor")
        a.set_xlim(lo, hi)
        a.set_ylim(17.5, 7.0)
        a.set_ylabel("B-like mag")
    S = pd.read_csv(RN / "v347nor" / "stamps_S.csv")
    thr = json.loads((ROOT / "docs" / "results" / "v347nor_stamps.json").read_text())["control_S_p99"]
    S = S[(S.S > thr) & S.set.isin(["eruption", "pre"])]
    ts = 2000 + (pd.to_datetime(S.date_utc).map(lambda d: d.to_julian_date()).values - 2451545.0) / 365.25
    for a, (lo, hi) in zip(ax, ((1888, 1992), (1942.5, 1947.0))):
        w = (ts >= lo) & (ts <= hi)
        a.scatter(ts[w], np.full(w.sum(), 7.35), marker="|", s=25, c="tab:red", lw=0.7,
                  label="star in cutout, missed by the catalogue")
    ax[0].set_title("V347 Nor (Hen 2-147) on the Harvard plates: an unrecorded eruption in 1943", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")
    a = ax[1]
    a.annotate("1943 Feb 9 UT: absent to 12.9 (cutout)", (1943.107, 12.93), (1942.55, 15.6), fontsize=7,
               arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    a.annotate("1943 Feb 26 UT: B = 8.0", (1943.153, 7.99), (1943.55, 7.9), fontsize=7,
               arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    a.text(1945.6, 8.2, "red ticks: plates whose cutouts show the star\nthough the catalogue logs a non-detection",
           fontsize=6.5, color="0.35")
    a.set_xlabel("year")
    fig.tight_layout()
    _save(fig, "v347nor_1943.png")


def v347nor_onset():
    """Cutouts of V347 Nor across the onset (Feb 9-26, 1943) and two quiet-year plates of the same series."""
    from astropy.io import fits
    from astropy.wcs import WCS
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    ra, dec = float(T.loc["T1274"].ra), float(T.loc["T1274"].dec)
    S = pd.read_csv(RN / "v347nor" / "stamps_S.csv").set_index("plate_id")
    names = ["bi00717", "bi00722", "am22958", "rb11996", "bi00741", "am22977", "rb12036", "rb11685"]
    fig, axs = plt.subplots(2, 4, figsize=(9.6, 6.0))
    for a, n in zip(axs.flat, names):
        h = fits.open(next((RN / "v347nor" / "stamps").glob(f"{n}_*.fits")))[0]
        x, y = (int(v) for v in WCS(h.header).world_to_pixel_values(ra, dec))
        d = h.data[y - 200:y + 200, x - 200:x + 200]
        lo, hi = np.nanpercentile(d, [20, 99.9])
        a.imshow(d, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        a.add_patch(plt.Circle((200, 200), 14, fill=False, color="tab:red", lw=0.8))
        r = S.loc[n]
        a.set_title(f"{n}  {r.date_utc[:10]}\nlim {r.lim:.1f}  S = {r.S:.1f}", fontsize=7.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("V347 Nor: absent Feb 9-10, 1943; present Feb 11 on two cameras (the catalogue logs all of these as "
                 "non-detections except rb12036).\nLast panel: a 1942 plate. Cutouts 9.6' wide, circle 20\" radius; "
                 "S = core vs 100-200\" annulus (quiet-plate 99th pct 1.9)", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "v347nor_onset.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("g0", g0), ("scores", scores), ("completeness", completeness), ("decir", decir),
                     ("v347nor", v347nor), ("v347nor_onset", v347nor_onset)):
        if what in (name, "all"):
            fn()
