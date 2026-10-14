"""T CrB literature synthesis figures (EXP-2026-Y).

  bibliometrics  ADS corpus: papers per year (focused vs full-text mentions), annotated with the system's history
  scoreboard     every published eruption-date forecast vs when it was made, against the 'no eruption yet' line
  forest         white-dwarf mass & inclination estimates through time

Inputs: data/tcrb/literature/ads_tcrb.json, ads_tcrb_fulltext.json (harvested via the ADS API),
celestrium/novae/literature.py (curated numbers), docs/results/experiment_y_tcrb_results.json.
Usage: python scripts/novae/tcrb_literature.py [stage ...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celestrium.novae import literature as lit  # noqa: E402

LIT = ROOT / "data" / "tcrb" / "literature"
FIG = ROOT / "docs" / "figures" / "tcrb"
RESULTS = ROOT / "docs" / "results" / "experiment_y_tcrb_results.json"
TODAY = 2026.78
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def stage_bibliometrics():
    foc = json.loads((LIT / "ads_tcrb.json").read_text(encoding="utf-8"))
    full = json.loads((LIT / "ads_tcrb_fulltext.json").read_text(encoding="utf-8"))
    yrs = np.arange(1860, 2027)

    def count(docs, pred=lambda d: True):
        c = collections.Counter(int(d["year"]) for d in docs if pred(d))
        return np.array([c.get(y, 0) for y in yrs])
    ref = count(foc, lambda d: "REFEREED" in d.get("property", []))
    tel = count(foc, lambda d: d.get("doctype") == "circular")
    oth = count(foc) - ref - tel
    allm = count(full)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(yrs, ref, color="#3949ab", label=f"refereed, T CrB in title/abstract (n={ref.sum()})")
    ax.bar(yrs, tel, bottom=ref, color="#e53935", label=f"ATels / circulars (n={tel.sum()})")
    ax.bar(yrs, oth, bottom=ref + tel, color="#9fa8da", label=f"other (proceedings, abstracts) (n={oth.sum()})")
    ax.step(yrs, allm, where="mid", color="k", lw=0.9, label=f"any full-text mention (n={allm.sum()})")
    ev = [(1866.37, "1866 eruption"), (1946.11, "1946 eruption"), (1958, "Kraft: binary"), (1992, "IUE: WD accretor"),
          (2015.0, "super-active phase"), (2023.3, "2023 dip")]
    for x, t in ev:
        ax.axvline(x, color="0.4", lw=0.7, ls=":")
        ax.text(x + 0.4, 0.97, t, rotation=90, va="top", fontsize=7.5, transform=ax.get_xaxis_transform())
    ax.set_xlim(1860, 2027)
    ax.set_ylabel("papers per year (ADS)")
    ax.set_title("160 years of T CrB in the literature: the field wakes up with the star")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.04, 0.62))
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_literature_bibliometrics.png", dpi=140)
    plt.close(fig)
    top = sorted(foc, key=lambda d: -(d.get("citation_count") or 0))[:10]
    out = {"n_focused": len(foc), "n_fulltext": len(full), "n_refereed": int(ref.sum()), "n_circulars": int(tel.sum()),
           "papers_2023_2026": int(count(foc, lambda d: int(d["year"]) >= 2023).sum()),
           "top_cited": [{"bibcode": d["bibcode"], "title": d["title"][0], "cites": d.get("citation_count")} for d in top]}
    _save({"literature": out})
    print("bibliometrics:", {k: v for k, v in out.items() if k != "top_cited"})


def stage_scoreboard():
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    preds = sorted(lit.PREDICTIONS, key=lambda p: p.made)
    preds = [p for p in preds if p.made > 2000]  # the 1946 forecast is quoted in the caption
    fc = res.get("forecast", {}).get("recurrence", {})
    ours = []
    if fc:
        lo, hi = fc.get("q16", 2027.12), fc.get("q84", 2030.90)
        ours.append(("This work: survival-conditioned recurrence", lo, hi, fc.get("median", 2028.13)))
    ov = res.get("dip_overlay", {})
    if ov:
        ours.append(("This work: 1945-dip analogue (Sep 2026 UV collapse)", ov["implied_eruption_alignment_B"] - 0.15,
                     ov["implied_eruption_alignment_B"] + 0.15, ov["implied_eruption_alignment_B"]))
    n = len(preds) + len(ours)
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * n + 1.6))
    for i, p in enumerate(preds):
        y = n - 1 - i
        lapsed = p.hi < TODAY
        col = "#9e9e9e" if lapsed else "#1e88e5"
        ax.plot([p.lo, p.hi], [y, y], color=col, lw=6, alpha=0.55, solid_capstyle="butt")
        if p.center is not None:
            ax.plot(p.center, y, "o", color="k" if not lapsed else "0.45", ms=4)
        for dd in p.discrete:
            ax.plot(dd, y, "D", color="#6a1b9a" if dd > TODAY else "0.55", ms=4.5)
        ax.plot(p.made, y, "|", color="#e65100", ms=11, mew=1.6)
        ax.text(2022.95, y, f"{p.who}  ", ha="right", va="center", fontsize=7.8, color="0.15" if not lapsed else "0.45")
    for j, (lab, lo, hi, c) in enumerate(ours):
        y = len(ours) - 1 - j
        ax.plot([lo, hi], [y, y], color="#d81b60", lw=6, alpha=0.7, solid_capstyle="butt")
        ax.plot(c, y, "o", color="k", ms=4)
        ax.text(2022.95, y, f"{lab}  ", ha="right", va="center", fontsize=7.8, color="#ad1457", weight="bold")
    ax.axvline(TODAY, color="#e53935", lw=1.4)
    ax.text(TODAY + 0.05, n - 0.4, "today\n2026 Oct 13", color="#e53935", fontsize=7.5, va="top")
    ax.axvspan(2023.0, TODAY, color="0.85", alpha=0.35, zorder=0)
    ax.set_yticks([])
    ax.set_xlim(2023.0, 2032.2)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("predicted eruption date")
    lapsed_n = sum(p.hi < TODAY for p in preds)
    ax.set_title(f"The T CrB eruption forecast scoreboard: {lapsed_n} of {len(preds)} published windows have closed "
                 "without an eruption", fontsize=10)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#1e88e5", lw=6, alpha=0.55, label="open forecast window"),
         Line2D([], [], color="#9e9e9e", lw=6, alpha=0.55, label="lapsed window"),
         Line2D([], [], color="#e65100", marker="|", ls="", ms=11, mew=1.6, label="date forecast was published"),
         Line2D([], [], color="#6a1b9a", marker="D", ls="", label="orbital-phase candidate dates")]
    ax.legend(handles=h, fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.93, bottom=0.08)
    # labels sit left of the axes: extend the canvas leftwards
    ax.set_xlim(2023.0, 2032.2)
    fig.savefig(FIG / "tcrb_forecast_scoreboard.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    _save({"forecast_scoreboard": {"n_published": len(preds), "n_lapsed": lapsed_n,
                                   "first_forecast": "Knight & Lindley 1946: 2026-27 (Shears 2024)"}})
    print("scoreboard:", lapsed_n, "/", len(preds), "lapsed")


def stage_forest():
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, data, lab, ch in ((axs[0], lit.WD_MASS, r"$M_{\rm WD}$ ($M_\odot$)", 1.38),
                              (axs[1], lit.INCLINATION, "inclination (deg)", None)):
        for k, m in enumerate(sorted(data, key=lambda m: m.year)):
            approx = m.note in ("~",)
            ax.errorbar(m.value, k, xerr=[[m.err_lo], [m.err_hi]], fmt="o" if not approx else "s",
                        color="#3949ab" if not approx else "#9fa8da", capsize=3, ms=5)
            ax.annotate(f"{m.who} ({m.method})", (m.value, k), xytext=(6, 5), textcoords="offset points", fontsize=7)
        ax.set_yticks([])
        ax.set_xlabel(lab)
        if ch:
            ax.axvline(ch, color="#e53935", lw=1, ls="--")
            ax.text(ch, 0.02, " Chandrasekhar\n (non-rotating)", color="#e53935", fontsize=7,
                    transform=ax.get_xaxis_transform())
    eph = res.get("ephemeris", {})
    if "incl_from_ellipsoidal" in eph:
        e = eph["incl_from_ellipsoidal"]
        axs[1].axvspan(e["lo"], e["hi"], color="#43a047", alpha=0.2, label="this work: ellipsoidal amplitude")
        axs[1].legend(fontsize=7, loc="lower left")
    axs[0].set_xlim(1.1, 1.75)
    axs[1].set_xlim(45, 85)
    axs[0].set_title("White-dwarf mass: converged on 1.29-1.37 Msun")
    axs[1].set_title("Inclination: 55-70 deg, method-dependent")
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_parameter_forest.png", dpi=140)
    plt.close(fig)
    print("forest done")


def _save(update):
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res.update(update)
    RESULTS.write_text(json.dumps(res, indent=2, default=float))


STAGES = {"bibliometrics": stage_bibliometrics, "scoreboard": stage_scoreboard, "forest": stage_forest}

if __name__ == "__main__":
    for st in (sys.argv[1:] or list(STAGES)):
        STAGES[st]()
