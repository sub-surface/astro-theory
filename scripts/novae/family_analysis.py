"""The four Galactic symbiotic recurrent novae side by side (EXP-2026-Y follow-on).

Swift/UVOT UVM2 (no red leak) and XRT 2-10 keV (LSXPS) for T CrB, RS Oph, V3890 Sgr, V745 Sco:
  * cycle   dereddened UVM2 absolute magnitude vs fraction of the recurrence cycle elapsed
  * plane   hard X-ray luminosity proxy (rate x d^2) vs UVM2 luminosity, quiescent epochs only
Eruption windows are excluded (t_erupt - 0.02 yr .. t_erupt + 2.5 yr) so nova-shell emission
does not masquerade as accretion. Distances / reddenings are the adopted literature values in
celestrium.novae.literature.FAMILY; their uncertainties (factor ~1.5 in d for the bulge systems)
shift whole clouds diagonally, which the figure shows as an arrow.
Usage: python scripts/novae/family_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "novae"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from celestrium.novae import tcrb  # noqa: E402
from celestrium.novae.literature import FAMILY  # noqa: E402
from celestrium.novae.tcrb import EXT_COEFF, jd_to_year  # noqa: E402

FIG = ROOT / "docs" / "figures" / "tcrb"
RESULTS = ROOT / "docs" / "results" / "experiment_y_tcrb_results.json"
SLUG = {"RS Oph": "rs_oph", "V3890 Sgr": "v3890_sgr", "V745 Sco": "v745_sco"}
COL = {"T CrB": "#d81b60", "RS Oph": "#1e88e5", "V3890 Sgr": "#43a047", "V745 Sco": "#fb8c00"}
T_REC = {"T CrB": 80.0, "RS Oph": 18.0, "V3890 Sgr": 28.5, "V745 Sco": 25.0}


def in_eruption(yr, eruptions):
    return np.any([(yr > e - 0.02) & (yr < e + 2.5) for e in eruptions], axis=0)


def cycle_phase(yr, eruptions):
    last = np.array([max([e for e in eruptions if e <= y], default=np.nan) for y in yr])
    return last, yr - last


def sibling_uv(name):
    d = pd.read_csv(ROOT / "data" / "srn" / SLUG[name] / "uvot_photometry.csv")
    d = d[(d["filter"] == "UVM2") & (d.exposure >= 20) & (~d.saturated) & (d.rate > 0)]
    d["jd"] = tcrb.swift_met_to_jd(d.met)
    d["night"] = np.floor(d.jd)
    g = d.groupby("night").agg(jd=("jd", "mean"), rate=("rate", "median"), err=("rate_err", "median"),
                               n=("rate", "size")).reset_index()
    yr = jd_to_year(g.jd.values)
    mag = tcrb.uvot_rate_to_mag(g.rate.values, "UVM2", yr)
    return g.jd.values, mag


def tcrb_uv():
    import tcrb_analysis as A
    uv = A.load_uvot()["UVM2"]
    return uv["jd"], uv["mag"]


def xray(name):
    d = ROOT / "data" / ("tcrb/lsxps" if name == "T CrB" else f"srn/{SLUG[name]}/lsxps")
    x = tcrb.load_lsxps(d)["Hard"]
    return x["jd"], x["rate"], x["err"]


# adopted 1-sigma systematics: E(B-V) and fractional distance (bulge systems poorly constrained)
EBV_SD = {"T CrB": 0.02, "RS Oph": 0.10, "V3890 Sgr": 0.10, "V745 Sco": 0.20}
D_FRAC = {"T CrB": 0.03, "RS Oph": 0.25, "V3890 Sgr": 0.35, "V745 Sco": 0.35}


def sys_err(name):
    """1-sigma systematic on a dereddened absolute UVM2 magnitude (extinction (+) distance)."""
    return math.hypot(EXT_COEFF["UVM2"] * EBV_SD[name], 5 / math.log(10) * D_FRAC[name])


def main():
    out, Mq = {}, {}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for sys_ in FAMILY:
        name = sys_.name
        try:
            jd, mag = tcrb_uv() if name == "T CrB" else sibling_uv(name)
        except FileNotFoundError:
            continue
        yr = jd_to_year(jd)
        q = ~in_eruption(yr, sys_.eruptions)
        M = mag - EXT_COEFF["UVM2"] * sys_.ebv - 5 * math.log10(sys_.dist_kpc * 100.0)
        _, since = cycle_phase(yr, sys_.eruptions)
        frac = since / T_REC[name]
        se = sys_err(name)
        axs[0].errorbar(frac[q], M[q], yerr=None if name == "T CrB" else se, fmt="o", ms=3 if name == "T CrB" else 5,
                        color=COL[name], alpha=0.55 if name == "T CrB" else 0.9, elinewidth=0.8, capsize=2,
                        label=f"{name} (d={sys_.dist_kpc} kpc, E(B-V)={sys_.ebv}; sys. +/-{se:.1f} mag)")
        Mq[name] = (M[q], se, yr[q])
        xj, xr, _ = xray(name)
        xq = ~in_eruption(jd_to_year(xj), sys_.eruptions)
        out[name] = {"n_uv_quiescent": int(q.sum()), "uvm2_abs_median_quiescent": float(np.median(M[q])) if q.any() else None,
                     "uvm2_abs_sys_mag": se,
                     "cycle_fraction_range": [float(np.nanmin(frac[q])), float(np.nanmax(frac[q]))] if q.any() else None,
                     "xrt_hard_rate_median_quiescent": float(np.median(xr[xq])) if xq.any() else None,
                     "xrt_hard_lum_proxy_median": float(np.median(xr[xq]) * sys_.dist_kpc ** 2) if xq.any() else None}
    axs[0].invert_yaxis()
    axs[0].set_xlabel("fraction of the recurrence cycle elapsed since the last eruption")
    axs[0].set_ylabel("UVM2 absolute mag (dereddened, Swift/UVOT)")
    axs[0].set_title("Four symbiotic recurrent novae through their cycles (eruption years excluded)")
    axs[0].legend(fontsize=7, loc="lower left")
    axs[0].axvline(1.0, color="0.4", ls=":")
    Mt, set_, yt = Mq["T CrB"]
    states = {"T CrB quiescence 2005-14": yt < 2014.8, "T CrB high state 2016-22": (yt > 2016) & (yt < 2022.9),
              "T CrB 2026 Sep 29": yt > 2026.7}
    ref = float(np.median(Mt[states["T CrB quiescence 2005-14"]]))
    rows = [(lab, float(np.median(Mt[sel])), set_, COL["T CrB"]) for lab, sel in states.items() if sel.any()]
    for nm in ("RS Oph", "V3890 Sgr", "V745 Sco"):
        if nm in Mq and len(Mq[nm][0]):
            rows.append((f"{nm} quiescence", float(np.median(Mq[nm][0])), Mq[nm][1], COL[nm]))
    comp = {}
    for k, (lab, m_, e_, c_) in enumerate(rows):
        lr = -0.4 * (m_ - ref)
        axs[1].errorbar(lr, k, xerr=0.4 * e_, fmt="o", color=c_, capsize=3, ms=7)
        axs[1].text(lr, k - 0.25, f"{lab}  (x{10 ** lr:,.0f})", fontsize=7.5, ha="center")
        comp[lab] = {"log10_ratio": lr, "sys_dex": 0.4 * e_, "abs_mag": m_}
    out["uv_luminosity_vs_tcrb_quiescence"] = comp
    axs[1].set_yticks([])
    axs[1].set_ylim(len(rows) - 0.4, -0.8)
    axs[1].axvline(0, color="0.5", lw=0.8, ls=":")
    axs[1].set_xlabel(r"log$_{10}$ UVM2 luminosity relative to T CrB in quiescence")
    axs[1].set_title("T CrB is the UV-faint member of the family")
    fig.tight_layout()
    fig.savefig(FIG / "srn_family_uv_xray.png", dpi=140)
    plt.close(fig)
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["srn_family"] = out
    RESULTS.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
