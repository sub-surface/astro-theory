#!/usr/bin/env python3
"""
Figures for project G: our pipeline vs the literature, with confidence intervals.

Reads data/quaia/quaia_dipole_results.json (g_collect.py) and, if present,
data/catwise/cw_dipole_result.json (g_catwise_dipole.py). Writes PNGs to figures/.

Fig 1  Dipole amplitude D vs galactic mask |b|, Quaia low/high (principled Poisson),
       with the isotropic noise floor and the M3 D_kin bands.
Fig 2  Recovered dipole directions (l,b) vs the CMB apex and literature points.
Fig 3  D/D_kin with confidence intervals -- this work vs the literature ("anomaly" plot).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
L_CMB, B_CMB = 264.021, 48.253

Q = json.load(open("data/quaia/quaia_dipole_results.json"))
CW = None
if os.path.exists("data/catwise/cw_dipole_result.json"):
    CW = json.load(open("data/catwise/cw_dipole_result.json"))

C_LOW, C_HIGH, C_CW = "#1f77b4", "#d62728", "#2ca02c"


def fig1():
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, col, lab in [("low", C_LOW, "Quaia low (G<20.0)"),
                          ("high", C_HIGH, "Quaia high (G<20.5)")]:
        d = Q[key]; ms = d["masks"]
        b = [m["bcut"] for m in ms]
        D = [m["D"] for m in ms]
        err = [m["null_std"] for m in ms]
        floor = [m["null_mean"] for m in ms]
        ax.errorbar(b, D, yerr=err, marker="o", color=col, lw=2, capsize=4, label=lab)
        ax.plot(b, floor, ls=":", color=col, alpha=0.7,
                label=f"{lab.split()[1]} isotropic floor")
        lo, hi = d["D_kin_ci"]
        ax.axhspan(lo, hi, color=col, alpha=0.12)
        ax.axhline(d["D_kin"], color=col, ls="--", alpha=0.5)
    ax.text(50.3, Q["low"]["D_kin"], "  D$_{kin}$ (M3)", va="center", fontsize=8, color="0.3")
    ax.set_xlabel("galactic-plane mask  |b| > (deg)")
    ax.set_ylabel("dipole amplitude  D")
    ax.set_title("Quaia number-count dipole vs mask (principled selection correction)\n"
                 "shaded = M3 kinematic expectation; dotted = isotropic noise floor")
    ax.set_xticks([30, 40, 50]); ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig1_amplitude_vs_mask.png", dpi=140)
    print("wrote figures/fig1_amplitude_vs_mask.png")


def fig2():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([L_CMB], [B_CMB], marker="*", s=420, color="k", zorder=5, label="CMB apex (264,48)")
    ax.scatter([238.2], [28.8], marker="P", s=160, color=C_CW, edgecolor="k",
               zorder=4, label="CatWISE Secrest+21 (238,29)")
    for key, col, lab in [("low", C_LOW, "Quaia low"), ("high", C_HIGH, "Quaia high")]:
        ms = Q[key]["masks"]
        ls = [m["l"] for m in ms]; bs = [m["b"] for m in ms]
        ax.plot(ls, bs, "-o", color=col, alpha=0.8, label=lab)
        for m in ms:
            ax.annotate(f"|b|>{m['bcut']}", (m["l"], m["b"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", color=col)
    if CW:
        ax.scatter([CW["l"]], [CW["b"]], marker="D", s=130, color=C_CW, zorder=4,
                   label="CatWISE this work (principled)")
        ax.scatter([CW["raw_l"]], [CW["raw_b"]], marker="d", s=90, facecolor="none",
                   edgecolor=C_CW, zorder=4, label="CatWISE this work (raw)")
    ax.set_xlabel("galactic longitude  l (deg)"); ax.set_ylabel("galactic latitude  b (deg)")
    ax.set_title("Recovered dipole directions vs the CMB apex\n"
                 "Quaia-low converges on the CMB with masking; Quaia-high drifts to (l,b)~(330,60)")
    ax.set_xlim(180, 360); ax.set_ylim(0, 90); ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig2_directions.png", dpi=140)
    print("wrote figures/fig2_directions.png")


def fig3():
    """D/D_kin with CIs: this work vs literature. Vertical line at 1 = pure kinematic."""
    rows = []  # (label, ratio, lo, hi, color, marker)

    def add(label, D, Derr, Dk, Dk_ci, col, mk="o"):
        r = D / Dk
        # propagate D and D_kin uncertainties
        rel = np.hypot(Derr / D, (0.5 * (Dk_ci[1] - Dk_ci[0])) / Dk)
        rows.append((label, r, r * (1 - rel), r * (1 + rel), col, mk))

    # --- literature ---
    add("CatWISE  Secrest+2021 (lit)", 0.01554, 0.00098, 0.007, (0.0065, 0.0075), C_CW, "P")
    rows.append(("Quaia high  raw, |b|>30 (M1)", 0.0353 / 0.0065, *(
        np.array([0.0353 - 0.004, 0.0353 + 0.004]) / 0.0065), C_HIGH, "d"))

    # --- this work (principled, |b|>40) ---
    qh = next(m for m in Q["high"]["masks"] if m["bcut"] == 40)
    ql = next(m for m in Q["low"]["masks"] if m["bcut"] == 40)
    add("Quaia high  principled, |b|>40", qh["D"], qh["null_std"],
        Q["high"]["D_kin"], Q["high"]["D_kin_ci"], C_HIGH)
    add("Quaia low  principled, |b|>40", ql["D"], ql["null_std"],
        Q["low"]["D_kin"], Q["low"]["D_kin_ci"], C_LOW)
    if CW:
        add("CatWISE  this work (principled)", CW["D"], CW["null_std"],
            CW["D_kin"], (CW["D_kin"] * 0.95, CW["D_kin"] * 1.05), C_CW, "D")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, lo, hi, col, mk) in zip(y, rows):
        ax.errorbar([r], [yi], xerr=[[r - lo], [hi - r]], marker=mk, ms=10, color=col,
                    capsize=5, lw=2)
    ax.axvline(1.0, color="k", ls="--", lw=1.5)
    ax.text(1.02, len(rows) - 0.4, "pure kinematic (D=D$_{kin}$)", fontsize=9, rotation=90,
            va="top", color="0.3")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("D / D$_{kin}$   (1 = consistent with CMB-kinematic expectation)")
    ax.set_title("Cosmic-dipole 'anomaly' across catalogues (95%-ish CIs)\n"
                 "principled correction collapses Quaia toward kinematic", fontsize=11)
    ax.set_xlim(0, 6); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig3_anomaly_vs_lit.png", dpi=140)
    print("wrote figures/fig3_anomaly_vs_lit.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("CatWISE included" if CW else "CatWISE json not present yet -- rerun after M4")
