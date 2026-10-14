"""EXP-2026-Y: T CrB panchromatic accretion history & eruption clock.

Stages (each writes figures to docs/figures/tcrb/ and merges numbers into
docs/results/experiment_y_tcrb_results.json):
  lightcurve  panchromatic 2005-2026 light curve (AAVSO U/B/V, Gaia, UVOT UV, XRT)
  ephemeris   orbital period & phase from the ellipsoidal modulation (no external ephemeris)
  inversion   light curves -> Mdot_transfer(t) -> Mdot_acc(t) with Laplace posterior,
              calibrated by the injection ensemble (scripts/tcrb_injection_calibration.py)
  xray        boundary-layer test: XRT hard/soft vs inferred Mdot_acc (held-out data)
  dips        1945 vs 2023-2026 pre-eruption dip comparison (superposed epochs)
  forecast    eruption-time forecast: recurrence + accreted-mass budget + orbital windows

Usage: python scripts/novae/tcrb_analysis.py [stage ...]   (default: all)
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celestrium.novae import tcrb  # noqa: E402
from celestrium.novae.tcrb import (ERUPTIONS_JD, EXT_COEFF, FIDUCIAL, jd_to_year, year_to_jd)  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data" / "tcrb"
FIG = ROOT / "docs" / "figures" / "tcrb"
RESULTS = ROOT / "docs" / "results" / "experiment_y_tcrb_results.json"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
BAND_COL = {"U": "#7b3fbf", "B": "#2a6fdb", "V": "#2a9d4b", "Vis": "#888888", "UVM2": "#c2185b",
            "UVW2": "#8e24aa", "UVW1": "#e64a19", "G": "#555555", "BP": "#3f51b5", "RP": "#d32f2f"}
UVOT_WING_RATIO = {}  # filled by load_uvot()


def save_results(update: dict):
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res.update(update)
    RESULTS.write_text(json.dumps(res, indent=2, default=float))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_munari():
    """Munari+ 2025 (A&A 701, A176; VizieR J/A+A/701/A176) ANS Collaboration UBVRI, 2005-2025."""
    f = DATA / "munari2025_ubvri.ecsv"
    if not f.exists():
        return {}
    from astropy.table import Table
    t = Table.read(f)
    out = {}
    for b in ("U", "B", "V", "R", "I"):
        m = np.ma.filled(t[f"{b}mag"].astype(float), np.nan)
        e = np.ma.filled(t[f"e_{b}mag"].astype(float), np.nan)
        ok = np.isfinite(m)
        out[b] = {"jd": np.asarray(t["HJD"], float)[ok], "mag": m[ok], "err": np.where(np.isfinite(e[ok]), e[ok], 0.02)}
    return out


def load_aavso(with_munari: bool = True):
    """AAVSO nightly bins per band, merged (when available) with the Munari+ 2025 ANS nights.

    ANS and AAVSO CCD photometry are both on the Johnson system; nights present in both are
    kept twice (independent measurements), each with the 0.02-mag floor of nightly_bins.
    """
    d = tcrb.load_aavso_sampled(DATA / "aavso_sampled.jsonl")
    ans = load_munari() if with_munari else {}
    out = {}
    for b in ("V", "B", "U", "Vis"):
        s = d["band"] == b
        jd, mag, err = d["jd"][s], d["mag"][s], d["err"][s]
        if b in ans:
            jd = np.concatenate([jd, ans[b]["jd"]])
            mag = np.concatenate([mag, ans[b]["mag"]])
            err = np.concatenate([err, ans[b]["err"]])
        if len(jd) == 0:
            continue
        nb = tcrb.nightly_bins(jd, mag, err, min_err=0.02 if b != "Vis" else 0.1)
        out[b] = nb
    return out, d


def load_uvot():
    d = pd.read_csv(DATA / "uvot_photometry.csv")
    d["jd"] = tcrb.swift_met_to_jd(d.met)
    d["yr"] = jd_to_year(d.jd)
    out = {}
    for f, g in d.groupby("filter"):
        x = g.raw_rate * g.frametime
        cal = g[(~g.saturated) & (x > 0.15) & (x < 0.6)]
        ratio = float((cal.wing_rate / cal.rate).median())
        r_sd = float(1.4826 * np.median(np.abs(cal.wing_rate / cal.rate - ratio)))
        UVOT_WING_RATIO[f] = (ratio, r_sd, len(cal))
        g = g[g.exposure >= 20].copy()
        x = g.raw_rate * g.frametime
        # extreme saturation hollows out the PSF core (raw core rate collapses while the
        # wings stay bright): flag whenever the wing predicts > 2.5x the measured core rate
        hollow = (g.wing_rate / ratio) > 2.5 * np.clip(g.rate, 1e-3, None)
        use_wing = g.saturated | (x >= 0.8) | hollow
        rate = np.where(use_wing, g.wing_rate / ratio, g.rate)
        frac = np.where(use_wing, np.hypot(g.wing_err / np.clip(g.wing_rate, 1e-3, None), r_sd / ratio),
                        g.rate_err / np.clip(g.rate, 1e-3, None))
        ok = rate > 0
        mag = tcrb.uvot_rate_to_mag(rate[ok], f, g.yr.values[ok])
        # one point per snapshot -> nightly medians (flicker averaged)
        nb = tcrb.nightly_bins(g.jd.values[ok], mag, 1.0857 * frac[ok], min_err=0.03, clip=1.5)
        nb["wing"] = np.array([use_wing.values[ok][np.floor(g.jd.values[ok]).astype(int) == int(np.floor(t))].mean()
                               for t in nb["jd"]])
        out[f] = nb
    return out


def load_xrt():
    return tcrb.load_lsxps(DATA / "lsxps")


def load_gaia():
    return tcrb.load_gaia_epoch(DATA / "gaia_dr3_epoch.ecsv")


# ---------------------------------------------------------------------------
# Stage: panchromatic light curve
# ---------------------------------------------------------------------------
def stage_lightcurve():
    av, raw = load_aavso()
    uv = load_uvot()
    xr = load_xrt()
    ga = load_gaia()
    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    ax = axs[0]
    for b in ("V", "B", "U"):
        if b in av:
            ax.errorbar(jd_to_year(av[b]["jd"]), av[b]["mag"], av[b]["err"], fmt=".", ms=2, lw=0.4,
                        color=BAND_COL[b], label=f"AAVSO {b} (n={len(av[b]['jd'])} nights)")
    for b in ("G", "BP"):
        ax.plot(jd_to_year(ga[b]["jd"]), ga[b]["mag"], "x", ms=3, color=BAND_COL[b], label=f"Gaia DR3 {b}")
    ax.invert_yaxis()
    ax.set_ylabel("mag")
    ax.legend(ncol=3, fontsize=7, loc="lower left")
    ax.set_title("T CrB 2005-2026: optical (AAVSO, Gaia), UV (Swift/UVOT), X-ray (Swift/XRT)")
    ax = axs[1]
    for f in ("UVW2", "UVM2", "UVW1"):
        if f in uv:
            w = uv[f]["wing"] > 0.5
            ax.errorbar(jd_to_year(uv[f]["jd"][~w]), uv[f]["mag"][~w], uv[f]["err"][~w], fmt="o", ms=2, lw=0.4,
                        color=BAND_COL[f], label=f"{f}")
            ax.errorbar(jd_to_year(uv[f]["jd"][w]), uv[f]["mag"][w], uv[f]["err"][w], fmt="s", ms=2.5, lw=0.4, mfc="none",
                        color=BAND_COL[f], label=f"{f} (PSF-wing, saturated core)")
    ax.invert_yaxis()
    ax.set_ylabel("UVOT Vega mag")
    ax.legend(ncol=3, fontsize=7)
    ax = axs[2]
    for b, c in (("Hard", "#c62828"), ("Soft", "#1565c0"), ("Medium", "#ef6c00")):
        x = xr[b]
        ax.errorbar(jd_to_year(x["jd"]), x["rate"], x["err"], fmt=".", ms=2, lw=0.4, color=c, label=f"XRT {b}")
        if len(x["ul"]):
            ax.plot(jd_to_year(x["ul_jd"]), x["ul"], "v", ms=2, color=c, alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylabel("XRT count/s")
    ax.legend(ncol=3, fontsize=7)
    ax = axs[3]
    # optical colour (B-V) and (U-B) from nights with both
    if "B" in av and "V" in av:
        for (b1, b2), c in ((("B", "V"), "#2a6fdb"), (("U", "B"), "#7b3fbf")):
            if b1 not in av or b2 not in av:
                continue
            n1 = np.round(av[b1]["jd"]).astype(int)
            n2 = np.round(av[b2]["jd"]).astype(int)
            com, i1, i2 = np.intersect1d(n1, n2, return_indices=True)
            ax.plot(jd_to_year(av[b1]["jd"][i1]), av[b1]["mag"][i1] - av[b2]["mag"][i2], ".", ms=3, color=c, label=f"{b1}-{b2}")
    ax.axhline(tcrb.GIANT_COLORS["B-V"] + FIDUCIAL.ebv, color="#2a6fdb", ls=":", lw=0.8)
    ax.text(2005.2, tcrb.GIANT_COLORS["B-V"] + FIDUCIAL.ebv + 0.03, "M3III giant B-V (reddened)", fontsize=7, color="#2a6fdb")
    ax.invert_yaxis()
    ax.set_ylabel("colour")
    ax.legend(fontsize=7)
    for a in axs:
        for t, lab in ((2015.1, "SAP onset"), (2023.3, "SAP end / dip"), (2024.4, "re-brightening")):
            a.axvline(t, color="k", lw=0.5, ls="--", alpha=0.4)
    axs[0].text(2015.15, axs[0].get_ylim()[1] + 0.1, "SAP onset", fontsize=7)
    axs[-1].set_xlabel("year")
    axs[-1].set_xlim(2004.8, 2026.95)
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_panchromatic_lightcurve.png", dpi=140)
    plt.close(fig)
    save_results({"data_summary": {
        "aavso_nights": {b: int(len(v["jd"])) for b, v in av.items()},
        "aavso_raw_obs": int(len(raw["jd"])),
        "uvot_nights": {f: int(len(v["jd"])) for f, v in uv.items()},
        "uvot_wing_ratio": {f: {"ratio": r, "sd": s, "n_cal": n} for f, (r, s, n) in UVOT_WING_RATIO.items()},
        "xrt_obs": {b: int(len(v["jd"])) for b, v in xr.items()},
        "gaia_epochs": {b: int(len(v["jd"])) for b, v in ga.items()},
    }})
    print("lightcurve: done")


# ---------------------------------------------------------------------------
# Stage: ephemeris
# ---------------------------------------------------------------------------
def stage_ephemeris(n_boot: int = 200):
    av, _ = load_aavso()
    V = av["V"]
    t, m, e = V["jd"], V["mag"], np.hypot(V["err"], 0.04)  # 0.04 mag flicker floor
    periods = np.linspace(222.0, 233.0, 1101)
    fit = tcrb.fit_ephemeris(t, m, e, periods)
    p_best = fit["p_best"]
    # bootstrap over nights (resample residual-based) for period uncertainty
    rng = np.random.default_rng(0)
    fine = np.linspace(p_best - 1.5, p_best + 1.5, 301)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        fb = tcrb.fit_ephemeris(t[idx], m[idx], e[idx], fine)
        boots.append(fb["p_best"])
    boots = np.array(boots)
    t_min = tcrb.ellipsoidal_minimum_epoch(fit["harm"], p_best, fit["t_ref"])
    # bring t_min close to 2026
    k = round((year_to_jd(2026.0) - t_min) / p_best)
    t_min26 = t_min + k * p_best
    harm = fit["harm"]
    a2 = math.hypot(harm[2], harm[3])
    a1 = math.hypot(harm[0], harm[1])
    # literature: Zamanov/Shore light-minimum (giant in front) JD 2455828.9 + 227.55 E
    lit_t0, lit_p = 2455828.9, 227.55
    ph_lit = ((t_min - lit_t0) / lit_p) % 1.0
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(periods, fit["chi2"] - fit["chi2"].min(), color="k", lw=0.8)
    axs[0].axvline(227.5528, color="C3", ls=":", label="Munari+25 227.553")
    axs[0].axvline(227.5687, color="C0", ls=":", label="Fekel+00 227.569")
    axs[0].set_xlabel("trial period (d)")
    axs[0].set_ylabel(r"$\Delta\chi^2$")
    axs[0].legend(fontsize=7)
    axs[0].set_title("AAVSO V, secular trend profiled out")
    axs[1].hist(boots, bins=30, color="0.5")
    axs[1].axvline(np.median(boots), color="k")
    axs[1].set_xlabel("bootstrap period (d)")
    axs[1].set_title(f"P = {np.median(boots):.2f} ± {np.std(boots):.2f} d")
    ph = tcrb.orbital_phase(t, t_min, p_best)
    det = fit["resid"] + (fit["resid"] * 0)  # residual after full model
    orb = m - fit["trend"]
    axs[2].plot(ph, orb, ".", ms=1.5, color="0.6")
    axs[2].plot(ph + 1, orb, ".", ms=1.5, color="0.6")
    bins = np.linspace(0, 1, 21)
    mid = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(orb[(ph >= bins[i]) & (ph < bins[i + 1])]) for i in range(20)]
    axs[2].plot(np.r_[mid, mid + 1], np.r_[med, med], "o-", color="k", ms=3)
    axs[2].invert_yaxis()
    axs[2].set_xlabel("phase (0 = deeper V minimum)")
    axs[2].set_ylabel("V - trend")
    axs[2].set_title(f"ellipsoidal: A(P/2)={a2:.3f}, A(P)={a1:.3f} mag")
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_ephemeris.png", dpi=140)
    plt.close(fig)
    # Inclination from the quiescent ellipsoidal amplitude (2006-2014.5: cold disk, giant-dominated V).
    # Any residual disk/hot-spot light dilutes the amplitude, so this is formally a LOWER limit on i.
    from celestrium.novae.roche import donor_surface, ellipsoidal_lightcurve, harmonic_amplitudes
    q_s = (t > year_to_jd(2006.0)) & (t < year_to_jd(2014.5))
    fq = tcrb.fit_ephemeris(t[q_s], m[q_s], e[q_s], [p_best], knot_spacing_d=150.0)
    a2q = math.hypot(fq["harm"][2], fq["harm"][3])
    a2_boot = []
    for _ in range(200):
        idx = rng.integers(0, q_s.sum(), q_s.sum())
        fb = tcrb.fit_ephemeris(t[q_s][idx], m[q_s][idx], e[q_s][idx], [p_best], knot_spacing_d=150.0)
        a2_boot.append(math.hypot(fb["harm"][2], fb["harm"][3]))
    a2_sd = float(np.std(a2_boot))
    incl_grid = np.arange(40, 90.1, 2.5)
    out_incl = {}
    # Intensity law: blackbody (reference) and T^k with k = 15, 18, 21 (M-giant TiO sensitivity in V)
    for qname, q in (("Munari+25 q=0.689", 0.93 / 1.35), ("Hinkle+25 q=0.504", 0.69 / 1.37)):
        surf = donor_surface(q, 1.0, 36, 72)
        for kname, k in (("blackbody", None), ("k15", 15.0), ("k18", 18.0), ("k21", 21.0)):
            amps = np.array([abs(harmonic_amplitudes(*ellipsoidal_lightcurve(q, i_, 5450.0, surf=surf, n_phase=48,
                                                                             k_teff=k)[:2])[2]) for i_ in incl_grid])
            ok = amps.max() >= a2q
            inv_i = lambda a: float(np.interp(a, amps, incl_grid)) if a <= amps.max() else float("nan")  # noqa: E731
            out_incl[f"{qname} {kname}"] = {"i_best": inv_i(a2q), "lo": inv_i(a2q - a2_sd), "hi": inv_i(a2q + a2_sd),
                                            "model_amp_at_61": float(np.interp(61, incl_grid, amps)),
                                            "model_amp_max": float(amps.max()), "reachable": bool(ok)}
    res_incl = {"quiescent_amp_half_period_mag": a2q, "quiescent_amp_sd": a2_sd, "n_quiescent_nights": int(q_s.sum()),
                "by_model": out_incl}
    # summary band: q=0.689 with k in 15-21 (lower limit on i: disk light dilutes the amplitude)
    vals = [out_incl[f"Munari+25 q=0.689 {k}"] for k in ("k15", "k18", "k21")]
    lo = [v["lo"] for v in vals if np.isfinite(v["lo"])]
    hi = [v["hi"] for v in vals if np.isfinite(v["hi"])]
    res_incl.update({"lo": float(min(lo)) if lo else float("nan"), "hi": float(max(hi)) if hi else 90.0})
    res = {"ephemeris": {"incl_from_ellipsoidal": res_incl,
                         "period_d": float(np.median(boots)), "period_sd_d": float(np.std(boots)),
                         "period_map_d": float(p_best), "t_deeper_min_jd": float(t_min26),
                         "amp_half_period_mag": a2, "amp_period_mag": a1,
                         "phase_of_deeper_min_in_lit_light_min_ephem": float(ph_lit),
                         "n_nights": int(len(t)), "rms_resid": float(np.std(fit["resid"]))}}
    save_results(res)
    print("ephemeris:", res)
    return res["ephemeris"]


# ---------------------------------------------------------------------------
# Stage: inversion
# ---------------------------------------------------------------------------
def build_band_data(av, uv, t_min_yr=2005.0):
    from celestrium.novae.inversion import BandData
    data = []
    for b in ("V", "B", "U"):
        if b in av:
            s = jd_to_year(av[b]["jd"]) >= t_min_yr
            data.append(BandData(b, av[b]["jd"][s], av[b]["mag"][s], av[b]["err"][s]))
    if "UVM2" in uv:  # no red leak: the cleanest UV tracer of the disk
        data.append(BandData("UVM2", uv["UVM2"]["jd"], uv["UVM2"]["mag"], uv["UVM2"]["err"]))
    return data


# v2 forward model: giant brightness prior (Zamanov+ 2024) + inner-disk truncation (see inversion.py)
MODEL_V2 = dict(giant_v_prior=(10.17, 0.06), truncation=True, physical_priors=True)  # v3 priors (see inversion.py)


def stage_inversion(n_samples: int = 300, grid=((0.75, 1.0), (1.5, 1.0), (3.0, 1.0), (1.5, 0.5))):
    import torch
    from celestrium.novae.inversion import (AccretionInversion, InversionConfig, accreted_mass, laplace_samples,
                                           mdot_acc_curve)
    av, _ = load_aavso()
    uv = load_uvot()
    res = json.loads(RESULTS.read_text())
    if "ephemeris" in res:
        eph = {"t0": res["ephemeris"]["t_deeper_min_jd"], "period": res["ephemeris"]["period_d"]}
    else:  # literature light-minimum ephemeris (giant in front), used until our own fit exists
        eph = {"t0": 2455828.9, "period": 227.55}
    data = build_band_data(av, uv)
    runs = []
    for tv, nexp in grid:
        cfg = InversionConfig(t_visc_out_yr=tv, n_exp=nexp, n_iter=900, **MODEL_V2)
        inv = AccretionInversion(data, eph, FIDUCIAL, cfg)
        P, L = inv.fit()
        runs.append({"t_visc": tv, "n_exp": nexp, "nlp": L, "inv": inv, "P": P})
        print(f"  t_visc={tv} n={nexp}: -logpost={L:.1f}")
    best = min(runs, key=lambda r: r["nlp"])
    inv, P = best["inv"], best["P"]
    samples = laplace_samples(inv, P, n_samples=n_samples)
    # calibration inflation from the injection ensemble (if available)
    infl = 1.0
    calf = DATA / "injection_calibration.json"
    if calf.exists():
        recs = json.loads(calf.read_text())
        z = np.array([r["z"]["total_2005_2026.7"] for r in recs])
        infl = max(1.0, float(np.sqrt(np.mean(z ** 2))))
    curves = np.array([mdot_acc_curve(inv, s) for s in samples])
    mmap = mdot_acc_curve(inv, P)
    t = jd_to_year(inv.t_grid)
    lc = np.log10(np.clip(curves, 1e-13, None))
    mu, sd = np.log10(np.clip(mmap, 1e-13, None)), lc.std(0) * infl
    masses = [accreted_mass(inv, s) for s in samples]
    mass_map = accreted_mass(inv, P)
    summ = {}
    for k in mass_map:
        lv = np.log10(np.clip([m[k] for m in masses], 1e-14, None))
        sdk = lv.std() * infl
        summ[k] = {"map": mass_map[k], "lo68": 10 ** (np.log10(mass_map[k]) - sdk), "hi68": 10 ** (np.log10(mass_map[k]) + sdk),
                   "sd_dex": sdk}
    # Empirically calibrated intervals: the injection ensemble gives the distribution of
    # MAP/true ratios (Laplace widths were shown to be overconfident), so true = MAP / ratio.
    if calf.exists():
        recs = json.loads(calf.read_text())
        for k in summ:
            if k in recs[0]["ratio_map"]:
                ratios = np.array([r["ratio_map"][k] for r in recs])
                q = np.percentile(mass_map[k] / ratios, [2.5, 16, 50, 84, 97.5])
                summ[k]["calibrated"] = {"q2.5": q[0], "q16": q[1], "q50": q[2], "q84": q[3], "q97.5": q[4],
                                         "n_injections": int(len(ratios))}
    # X-ray-anchored variant at the preferred viscosity
    cfg_a = InversionConfig(t_visc_out_yr=best["t_visc"], n_exp=best["n_exp"], n_iter=900,
                            xray_anchor=(2016.5, 2019.0, math.log10(5e-8), 0.3), **MODEL_V2)
    inv_a = AccretionInversion(data, eph, FIDUCIAL, cfg_a)
    P_a, L_a = inv_a.fit()
    anchored = accreted_mass(inv_a, P_a)
    mdot_anch = mdot_acc_curve(inv_a, P_a)
    # systematic spread across the disk-viscosity grid (model uncertainty)
    sys_spread = {}
    for k in mass_map:
        vals = [accreted_mass(r["inv"], r["P"])[k] for r in runs]
        sys_spread[k] = {"min": float(min(vals)), "max": float(max(vals))}
    S = inv.summarize(P)
    # figure: data + model, Mdot_tr/Mdot_acc, disk mass
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
    ax = axs[0]
    tt = np.linspace(year_to_jd(2005), year_to_jd(2026.8), 4000)
    for d in data:
        ax.errorbar(jd_to_year(d.jd), d.mag, d.err, fmt=".", ms=2, lw=0.3, color=BAND_COL[d.band], alpha=0.6)
        ax.plot(jd_to_year(tt), inv.predict_mag(P, d.band, tt), color=BAND_COL[d.band], lw=0.8,
                label=f"{d.band}: flicker {S['jitter'][d.band]:.2f} mag")
    ax.invert_yaxis()
    ax.set_ylabel("mag")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Viscous-disk inversion (t_visc,out = {best['t_visc']} yr, n = {best['n_exp']}); giant V = {S['giant_v']:.2f}, eta = {S['eta']:.3f}")
    ax = axs[1]
    ax.fill_between(t, 10 ** (mu - 2 * sd), 10 ** (mu + 2 * sd), color="C1", alpha=0.2, lw=0)
    ax.fill_between(t, 10 ** (mu - sd), 10 ** (mu + sd), color="C1", alpha=0.4, lw=0, label=r"$\dot M_{\rm acc}$ onto WD (68/95%, calibrated)")
    ax.step(S["bin_yr"], S["mdot_tr"], where="post", color="k", lw=0.8, label=r"$\dot M_{\rm tr}$ from giant (MAP)")
    ax.plot(t, mdot_anch, color="C3", lw=1.2, label=r"$\dot M_{\rm acc}$, X-ray-anchored (Luna+18 BL)")
    for r in runs:
        if r is not best:
            ax.plot(t, mdot_acc_curve(r["inv"], r["P"]), lw=0.6, alpha=0.6, ls="--",
                    label=f"alt. viscosity t={r['t_visc']}, n={r['n_exp']}")
    ax.set_yscale("log")
    ax.set_ylabel(r"M$_\odot$ yr$^{-1}$")
    ax.legend(fontsize=6.5, ncol=2)
    ax = axs[2]
    ax.plot(S["t_yr"], S["m_disk"] * 1e6, color="C2")
    ax.set_ylabel(r"disk mass [$10^{-6}$ M$_\odot$]")
    ax.set_xlabel("year")
    ax.set_xlim(2005, 2026.9)
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_accretion_inversion.png", dpi=140)
    plt.close(fig)
    # states for the renderer
    def T_at(yr=None, i=None):
        i = int(np.argmin(np.abs(S["t_yr"] - yr))) if i is None else i
        T = inv.disk.teff(S["sigma"][i])
        if S["r_trunc_cm"] is not None:  # truncated inner disk does not emit (hot flow instead)
            T = np.where(inv.disk.grid.r > S["r_trunc_cm"][i], T, 0.0)
        return T
    i_hi = int(np.argmax(mmap))
    np.savez(DATA / "inversion_states.npz", r_a=inv.disk.grid.r / FIDUCIAL.a_orb,
             T_quiet=T_at(2010.0), T_high=T_at(i=i_hi), T_now=T_at(2026.75),
             label_quiet="2010 (quiescence)", label_high=f"{S['t_yr'][i_hi]:.1f} (peak accretion)",
             label_now="2026.75 (now)")
    np.savez(DATA / "inversion_curves.npz", t=t, mu=mu, sd=sd, bin_yr=S["bin_yr"], mdot_tr=S["mdot_tr"],
             m_disk=S["m_disk"])
    # full disk history for the accretion movie (every 4th time step)
    k = slice(None, None, 4)
    np.savez(DATA / "inversion_history.npz", t_yr=S["t_yr"][k], sigma=S["sigma"][k], r=inv.disk.grid.r,
             teff=np.array([inv.disk.teff(sg) for sg in S["sigma"][k]]), mdot_acc=S["mdot_acc"][k],
             r_trunc=(S["r_trunc_cm"][k] if S["r_trunc_cm"] is not None else np.full(len(S["t_yr"][k]), inv.disk.grid.r[0])),
             a_orb=FIDUCIAL.a_orb)
    out = {"inversion": {"best_viscosity": {"t_visc_out_yr": best["t_visc"], "n_exp": best["n_exp"]},
                         "grid_neg_log_post": {f"{r['t_visc']}_{r['n_exp']}": r["nlp"] for r in runs},
                         "accreted_mass_msun": summ, "viscosity_systematic": sys_spread,
                         "xray_anchored": {"accreted_mass_msun": anchored, "neg_log_post": L_a,
                                           "mdot_acc_now": float(mdot_anch[-1]),
                                           "mdot_acc_peak": float(mdot_anch.max())},
                         "calibration_inflation": infl, "jitter_mag": S["jitter"], "eta": S["eta"],
                         "giant_v": S["giant_v"], "giant_v_at_max_app": S["giant_v_at_max_app"],
                         "mdot_c_truncation": S.get("mdot_c"), "truncation_index_a": S.get("tr_a"),
                         "model": "v3: giant prior + truncation + physical priors (a, eta)",
                         "mdot_q_msun_yr": S["mdot_q"],
                         "mdot_acc_peak": float(mmap.max()), "mdot_acc_peak_year": float(t[np.argmax(mmap)]),
                         "mdot_acc_now": float(mmap[-1]),
                         "mdot_acc_now_68": [float(10 ** (mu[-1] - sd[-1])), float(10 ** (mu[-1] + sd[-1]))]}}
    save_results(out)
    print(json.dumps(out, indent=1, default=float)[:3000])


# ---------------------------------------------------------------------------
# Stage: X-ray boundary-layer test (held-out)
# ---------------------------------------------------------------------------
def stage_xray():
    xr = load_xrt()
    c = np.load(DATA / "inversion_curves.npz")
    t, mu, sd = c["t"], c["mu"], c["sd"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    out = {}
    for b, col in (("Hard", "#c62828"), ("Soft", "#1565c0")):
        x = xr[b]
        ty = jd_to_year(x["jd"])
        s = (ty > t.min()) & (ty < t.max())
        lm = np.interp(ty[s], t, mu)
        axs[0].errorbar(10 ** lm, x["rate"][s], x["err"][s], fmt=".", ms=3, lw=0.4, color=col, label=f"XRT {b}")
        if b == "Hard":
            from scipy.stats import spearmanr
            rho, p = spearmanr(lm, x["rate"][s])
            out["spearman_hard_vs_mdot"] = {"rho": float(rho), "p": float(p), "n": int(s.sum())}
            # optically thick/thin transition: Mdot where hard rate falls to half its low-Mdot plateau
            order = np.argsort(lm)
            lm_s, r_s = lm[order], x["rate"][s][order]
            plateau = np.median(r_s[: max(5, len(r_s) // 10)])
            win = 15
            run = np.array([np.median(r_s[max(0, i - win):i + win]) for i in range(len(r_s))])
            below = np.where(run < 0.5 * plateau)[0]
            if len(below):
                out["bl_transition_mdot_msun_yr"] = float(10 ** lm_s[below[0]])
            axs[0].plot(10 ** lm_s, run, color="k", lw=1)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"inferred $\dot M_{\rm acc}$ (M$_\odot$/yr), optical/UV only")
    axs[0].set_ylabel("XRT count/s")
    if "bl_transition_mdot_msun_yr" in out:
        axs[0].axvline(out["bl_transition_mdot_msun_yr"], color="k", ls=":", lw=0.8)
        axs[0].text(out["bl_transition_mdot_msun_yr"] * 1.1, axs[0].get_ylim()[1] * 0.5, "BL thick", fontsize=8)
    axs[0].legend(fontsize=8)
    axs[0].set_title("Held-out test: boundary layer turns optically thick at high $\\dot M$")
    ax = axs[1]
    ax2 = ax.twinx()
    ax.fill_between(t, 10 ** (mu - sd), 10 ** (mu + sd), color="C1", alpha=0.4)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\dot M_{\rm acc}$ (M$_\odot$/yr)", color="C1")
    x = xr["Hard"]
    ax2.plot(jd_to_year(x["jd"]), x["rate"], ".", ms=2, color="#c62828")
    ax2.set_yscale("log")
    ax2.set_ylabel("XRT hard c/s", color="#c62828")
    ax.set_xlim(2005, 2026.9)
    ax.set_xlabel("year")
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_xray_boundary_layer.png", dpi=140)
    plt.close(fig)
    save_results({"xray": out})
    print("xray:", out)


# ---------------------------------------------------------------------------
# Stage: dips (1945 vs now)
# ---------------------------------------------------------------------------
def stage_dips():
    av, raw = load_aavso()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    out = {}
    e46 = jd_to_year(ERUPTIONS_JD["1946"])
    if "Vis" in av:
        v = av["Vis"]
        ty = jd_to_year(v["jd"])
        s = (ty > 1935) & (ty < e46 - 0.5 / 365.25)
        # T CrB cannot have been brighter than ~9.0 between eruptions: such visual estimates
        # (6.3-8.0 mag in 1941-44) are misidentifications and are excluded, and counted.
        bad = s & (v["mag"] < 9.0)
        s = s & ~bad
        out["1935_46_visual_nights_used"] = int(s.sum())
        out["1935_46_visual_nights_rejected_too_bright"] = int(bad.sum())
        bins = np.arange(1935, e46 + 0.01, 90 / 365.25)
        idx = np.digitize(ty[s], bins)
        bm = np.array([[bins[i - 1] + 45 / 365.25, np.median(v["mag"][s][idx == i]), (idx == i).sum()]
                       for i in np.unique(idx) if (idx == i).sum() >= 2])
        axs[0].plot(ty[s] - e46, v["mag"][s], ".", ms=3, color="0.55", label="AAVSO visual (mostly one observer)")
        axs[0].plot(bm[:, 0] - e46, bm[:, 1], "-o", ms=3, color="k", lw=1, label="90-d median")
        base_sel = (bm[:, 0] > 1942.0) & (bm[:, 0] < 1944.8)
        base = float(np.median(bm[base_sel, 1]))
        late = bm[:, 0] > 1944.5
        on = bm[late & (bm[:, 1] >= base + 0.3), 0]
        if len(on):
            dip = late & (bm[:, 0] < e46)
            imin = np.argmax(np.where(dip, bm[:, 1], -99))
            out["1945_baseline_vis_mag"] = base
            out["1945_dip_onset_yr"] = float(on[0])
            out["1945_dip_onset_years_before_eruption"] = float(e46 - on[0])
            on15 = bm[late & (bm[:, 1] >= base + 0.15), 0]
            out["1945_dip_onset_0.15mag_years_before_eruption"] = float(e46 - on15[0])
            out["1945_dip_bin_halfwidth_yr"] = 45 / 365.25
            out["1945_dip_depth_vis_mag"] = float(bm[imin, 1] - base)
            out["1945_dip_minimum_years_before_eruption"] = float(e46 - bm[imin, 0])
            out["1945_last_bin_before_eruption_mag"] = float(bm[dip, 1][-1])
            axs[0].axhline(base, color="k", ls=":", lw=0.7)
            axs[0].axvspan(on[0] - e46 - 45 / 365.25, 0, color="C3", alpha=0.08, label="1945 dip")
        axs[0].set_xlim(-11.2, 0.15)
    for b, c in (("V", BAND_COL["V"]), ("B", BAND_COL["B"])):
        if b in av:
            ty = jd_to_year(av[b]["jd"])
            s = ty > 2019
            axs[1].plot(ty[s], av[b]["mag"][s], ".", ms=2, color=c, alpha=0.5, label=f"AAVSO {b}")
            bins = np.arange(2019, 2026.9, 30 / 365.25)
            idx = np.digitize(ty[s], bins)
            bm = np.array([[bins[i - 1] + 15 / 365.25, np.median(av[b]["mag"][s][idx == i])] for i in np.unique(idx)])
            axs[1].plot(bm[:, 0], bm[:, 1], "-", color=c, lw=1.2)
            hi = (bm[:, 0] > 2021) & (bm[:, 0] < 2023.0)
            d23 = (bm[:, 0] > 2023.2) & (bm[:, 0] < 2024.3)
            d26 = bm[:, 0] > 2026.0
            if hi.any() and d23.any():
                out[f"2023_dip_depth_{b}"] = float(np.max(bm[d23, 1]) - np.median(bm[hi, 1]))
            if hi.any() and d26.any():
                out[f"2026_level_vs_high_state_{b}"] = float(np.median(bm[d26, 1]) - np.median(bm[hi, 1]))
    for a in axs:
        a.invert_yaxis()
        a.legend(fontsize=7)
    axs[0].set_xlabel("years relative to 1946 Feb 9 eruption")
    axs[0].set_ylabel("visual mag")
    axs[0].set_title("Previous cycle: high state and the 1945 pre-eruption dip")
    axs[1].set_xlabel("year")
    axs[1].set_title("This cycle: end of SAP (2023), re-brightening (2024), 2026 decline")
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_dips_1946_vs_now.png", dpi=140)
    plt.close(fig)
    save_results({"dips": out})
    print("dips:", out)


def _binned(ty, mag, lo, hi, width_d):
    bins = np.arange(lo, hi + 1e-6, width_d / 365.25)
    idx = np.digitize(ty, bins)
    rows = []
    for i in np.unique(idx):
        sel = idx == i
        if sel.sum() < 2 or i == 0 or i >= len(bins):
            continue
        m = mag[sel]
        rows.append((np.median(ty[sel]), np.median(m), 1.2533 * 1.4826 * np.median(np.abs(m - np.median(m))) /
                     np.sqrt(sel.sum()) + 0.01, sel.sum()))
    return np.array(rows)


def stage_overlay():
    """Superpose the 1945 pre-eruption dip on the 2023-2026 light curve under two alignments.

    Visual (1935-46) and CCD V (2015-26) are put on a common scale as Delta-mag relative to each
    cycle's own late high-state level (1942.0-1944.8 vs 2021.0-2023.0), which removes the
    visual-vs-V colour term for this very red star. Alignments map the 1945 dip onset onto
    (A) the 2023 dip onset and (B) the 2026 Sep UV collapse; each transfers the 1946 eruption
    epoch to an implied date for this cycle.
    """
    av, _ = load_aavso()
    uv = load_uvot()
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    e46 = jd_to_year(ERUPTIONS_JD["1946"])
    v = av["Vis"]
    ty = jd_to_year(v["jd"])
    s = (ty > 1935) & (ty < e46 - 0.5 / 365.25) & (v["mag"] >= 9.0)
    old = _binned(ty[s], v["mag"][s], 1935.0, e46, 60)
    base_old = np.median(old[(old[:, 0] > 1942.0) & (old[:, 0] < 1944.8), 1])
    on_old = res.get("dips", {}).get("1945_dip_onset_yr", 1945.47)
    tv = jd_to_year(av["V"]["jd"])
    sv = tv > 2014.5
    # remove the ellipsoidal modulation (0.27 mag p-p at P/2 ~ 114 d) so it does not alias into
    # the binned curve; the 1940s visual estimates are quantised at 0.1 mag and are left as is
    jv, mv, ev = av["V"]["jd"][sv], av["V"]["mag"][sv], av["V"]["err"][sv]
    fe = tcrb.fit_ephemeris(jv, mv, np.hypot(ev, 0.05), [FIDUCIAL.p_orb], knot_spacing_d=90.0)
    harm = tcrb._design(jv, FIDUCIAL.p_orb, fe["t_ref"], np.array([jv.min(), jv.max()]))[:, :4] @ fe["harm"]
    new = _binned(tv[sv], mv - harm, 2014.5, 2026.9, 30)
    base_new = np.median(new[(new[:, 0] > 2021.0) & (new[:, 0] < 2023.0), 1])
    on23 = res.get("uv_dips", {}).get("onset_2023_yr", 2023.60)
    on26 = res.get("uv_dips", {}).get("onset_2026_yr", 2026.742)
    # V onset of the 2023 optical dip: first 30-d bin after 2023.0 fading >= 0.15 mag below the high state
    cand = new[(new[:, 0] > 2023.0) & (new[:, 1] >= base_new + 0.15), 0]
    on23v = float(cand[0]) if len(cand) else on23
    lead = e46 - on_old  # 1945: eruption came this long after the dip onset
    tu = jd_to_year(uv["UVM2"]["jd"])
    mu = uv["UVM2"]["mag"]
    hi_uv = np.median(mu[(tu > 2021.0) & (tu < 2023.0)])

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.0, 1.0], hspace=0.38, wspace=0.12)
    ax0 = fig.add_subplot(gs[0, :])
    c_old, c_new = "#5d4037", BAND_COL["V"]
    # full-cycle overlay: years before eruption (old) vs years relative to an eruption at
    # (2026 dip onset + 1945 lead); top axis shows calendar years for this cycle
    shift_b = on26 - on_old
    ax0.plot(new[:, 0] - shift_b - e46, new[:, 1] - base_new, "-", color="#c62828", lw=1.4, alpha=0.9,
             label="2014-26 V (30-d medians, ellipsoidal removed), shifted so the 2026 UV collapse = 1945 dip onset")
    ax0.axvspan(2023.24 - shift_b - e46, 2024.3 - shift_b - e46, color="#1565c0", alpha=0.07)
    ax0.text(2023.3 - shift_b - e46, 0.97, "2023 dip", color="#1565c0", fontsize=8, va="top",
             transform=ax0.get_xaxis_transform())
    sec = ax0.secondary_xaxis("top", functions=(lambda x: x + e46 + shift_b, lambda y: y - e46 - shift_b))
    sec.set_xlabel("calendar year (modern curve)", fontsize=8)
    ax0.errorbar(old[:, 0] - e46, old[:, 1] - base_old, old[:, 2], fmt="o", ms=3.5, color=c_old, lw=0.8,
                 label="1935-46 visual (60-d medians)", zorder=5)
    ax0.axvline(0, color="k", lw=1.0)
    ax0.text(0.03, 0.98, "1946 eruption", rotation=90, va="top", ha="left", fontsize=8, transform=ax0.get_xaxis_transform())
    ax0.axhline(0, color="0.4", ls=":", lw=0.7)
    ax0.set_xlim(-11.5, 1.6)
    ax0.invert_yaxis()
    ax0.set_ylabel(r"$\Delta$mag vs late high state")
    ax0.set_xlabel("years relative to the 1946 eruption (modern curve shifted by alignment B)")
    ax0.set_title("1935-46 vs 2014-26 on one axis", fontsize=10)
    ax0.legend(fontsize=7.5, loc="lower left")

    out = {"baseline_visual_1942_44.8": float(base_old), "baseline_V_2021_23": float(base_new),
           "dip_onset_2023_V_yr": on23v, "uv_collapse_2026_yr": float(on26), "lead_1945_yr": float(lead)}
    for k, (lab, on_now, col) in enumerate((("A: 1945 dip onset = 2023 optical dip onset", on23v, "#1565c0"),
                                            ("B: 1945 dip onset = 2026 Sep UV collapse", on26, "#c62828"))):
        ax = fig.add_subplot(gs[1, k])
        x_old = old[:, 0] - on_old
        sel = (x_old > -2.2) & (x_old < 1.5)
        ax.errorbar(x_old[sel], old[sel, 1] - base_old, old[sel, 2], fmt="o-", ms=3.5, lw=0.9, color=c_old,
                    label="1944-46 visual")
        x_new = new[:, 0] - on_now
        seln = (x_new > -2.2) & (x_new < 1.5)
        ax.errorbar(x_new[seln], new[seln, 1] - base_new, new[seln, 2], fmt="s-", ms=3, lw=0.9, color=col,
                    label="V (30-d medians)")
        xu = tu - on_now
        su = (xu > -2.2) & (xu < 1.5)
        ax.plot(xu[su], (mu[su] - hi_uv) / 4.0, ".", ms=2.5, color=BAND_COL["UVM2"], alpha=0.6,
                label=r"UVM2 $\Delta$mag / 4")
        ax.axvline(lead, color="k", lw=1.0)
        implied = on_now + lead
        ax.text(lead + 0.03, 0.02, f"1946-equivalent\neruption: {implied:.2f}", fontsize=7.5,
                transform=ax.get_xaxis_transform(), va="bottom")
        ax.axvline(jd_to_year(year_to_jd(2026.78)) - on_now, color="0.3", ls="--", lw=0.8)
        ax.text(2026.78 - on_now, 0.98, " today", fontsize=7, va="top", transform=ax.get_xaxis_transform())
        ax.axhline(0, color="0.4", ls=":", lw=0.7)
        ax.set_xlim(-2.2, 1.5)
        ax.set_ylim(1.15, -0.45)
        ax.set_title(lab, fontsize=9, color=col)
        ax.set_xlabel("years since aligned dip onset")
        if k == 0:
            ax.set_ylabel(r"$\Delta$mag")
        ax.legend(fontsize=7, loc="lower left")
        out[["implied_eruption_alignment_A", "implied_eruption_alignment_B"][k]] = float(implied)
    # verdict strip: the falsified alignment
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    txt = (f"Alignment A predicted the eruption at {out['implied_eruption_alignment_A']:.2f}: falsified (no eruption by 2026.78). "
           f"Alignment B places it at {out['implied_eruption_alignment_B']:.2f}.\n"
           "The 1945 dip was ~0.7 mag deep (visual), reached minimum within ~0.1 yr of onset, and was recovering when the nova "
           "went off. The 2023 dip ended in re-brightening, so the 1945 dip had no clear analogue until Sep 2026 -- if any.\n"
           "Caveats: 1935-46 curve is dominated by one visual observer; visual and V respond differently to a red giant; "
           "the 2026 optical response is still being sampled. Visual Delta-mag of 0.0 = the 9.8-mag comparison-star level (0.1-mag quantisation).")
    ax3.text(0.0, 0.95, txt, fontsize=8.3, va="top", family="monospace", wrap=True)
    fig.savefig(FIG / "tcrb_dips_overlay.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    save_results({"dip_overlay": out})
    r3 = lambda a, k=3: [round(float(x), k) for x in a]  # noqa: E731
    sel_u = tu > 2014.5
    (DATA / "overlay_curves.json").write_text(json.dumps({
        "e46": float(e46), "on_old": float(on_old), "lead": float(lead),
        "old": {"t": r3(old[:, 0] - e46), "d": r3(old[:, 1] - base_old), "e": r3(old[:, 2])},
        "new": {"t": r3(new[:, 0]), "d": r3(new[:, 1] - base_new), "e": r3(new[:, 2])},
        "uv": {"t": r3(tu[sel_u]), "d": r3((mu[sel_u] - hi_uv) / 4.0)}}, separators=(",", ":")))
    print("overlay:", out)





# ---------------------------------------------------------------------------
# Stage: forecast
# ---------------------------------------------------------------------------
def stage_forecast():
    res = json.loads(RESULTS.read_text())
    t_now = jd_to_year(2461326.5)  # 2026-10-13
    yr = 365.25
    ints = [(ERUPTIONS_JD["1866"] - ERUPTIONS_JD["1787"]) / yr, (ERUPTIONS_JD["1946"] - ERUPTIONS_JD["1866"]) / yr,
            (1787.91 - 1217.8) / 7]
    rec = tcrb.recurrence_predictive(ints, jd_to_year(ERUPTIONS_JD["1946"]), t_now)
    eph = res.get("ephemeris", {})
    out = {"recurrence": {k: v for k, v in rec.items() if k != "samples"},
           "intervals_yr": ints}
    for lim in (2027.0, t_now + 1, 2030.0):
        out["recurrence"][f"p_before_{lim:.2f}"] = float((rec["samples"] < lim).mean())
    # orbital-phase windows: 1866/1946 eruption phases with our ephemeris
    if eph:
        P, t0 = eph["period_d"], eph["t_deeper_min_jd"]
        phs = {k: float(tcrb.orbital_phase(v, t0, P)) for k, v in ERUPTIONS_JD.items()}
        # phase uncertainty from period error propagated back to each eruption
        dph = {k: float(abs((t0 - v) / P) * eph["period_sd_d"] / P) for k, v in ERUPTIONS_JD.items()}
        out["eruption_phases"] = {k: {"phase": phs[k], "sd": dph[k]} for k in phs}
        nxt = {}
        for k in ("1866", "1946"):
            n0 = math.ceil((year_to_jd(t_now) - ERUPTIONS_JD[k]) / P)
            nxt[k] = [float(jd_to_year(ERUPTIONS_JD[k] + (n0 + j) * P)) for j in range(4)]
        out["same_phase_dates_yr"] = nxt
    # accreted-mass budget
    inv = res.get("inversion")
    if inv:
        mm = inv["accreted_mass_msun"]
        out["mass_budget"] = {
            "accreted_2005_2026.7": mm["total_2005_2026.7"],
            "quiescent_rate": inv["mdot_q_msun_yr"],
            "accreted_1946_2005_if_quiescent": inv["mdot_q_msun_yr"] * (2005.0 - 1946.11),
            "current_mdot_acc": inv["mdot_acc_now"],
            "current_mdot_acc_68": inv["mdot_acc_now_68"]}
    fig, ax = plt.subplots(figsize=(9, 4))
    s = rec["samples"]
    ax.hist(s[s < 2036], bins=200, density=True, color="0.6", label="recurrence (log-normal, survival-conditioned)")
    if "same_phase_dates_yr" in out:
        for k, c in (("1866", "C0"), ("1946", "C3")):
            for j, d in enumerate(out["same_phase_dates_yr"][k]):
                ax.axvline(d, color=c, ls=":", lw=1, label=f"{k}-phase return" if j == 0 else None)
    ax.axvline(t_now, color="k", lw=1)
    ax.text(t_now + 0.05, ax.get_ylim()[1] * 0.9, "today", fontsize=8)
    ax.set_xlim(t_now - 0.2, 2033)
    ax.set_xlabel("year")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("When? recurrence statistics conditioned on no eruption by 2026-10-13")
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_forecast.png", dpi=140)
    plt.close(fig)
    save_results({"forecast": out})
    print("forecast:", json.dumps(out, indent=1, default=float)[:2000])


STAGES = {"lightcurve": stage_lightcurve, "ephemeris": stage_ephemeris, "inversion": stage_inversion,
          "xray": stage_xray, "dips": stage_dips, "forecast": stage_forecast}



# ---------------------------------------------------------------------------
# Stage: UV dip onsets (2023 vs 2026), superposed on the UV collapse date
# ---------------------------------------------------------------------------
def stage_uvdips():
    uv = load_uvot()
    xr = load_xrt()
    f = "UVM2"
    t = jd_to_year(uv[f]["jd"])
    m = uv[f]["mag"]
    out = {}
    # onset = first night after the high state with UVM2 fainter than the threshold
    thr = 13.0
    on23 = t[(t > 2023.0) & (t < 2024.0) & (m > thr)].min()
    late = (t > 2026.0) & (m > thr)
    on26 = t[late].min() if late.any() else np.nan
    pre26 = t[(t > 2026.0) & (t < on26)].max() if np.isfinite(on26) else np.nan
    out["uvm2_threshold_mag"] = thr
    out["onset_2023_yr"] = float(on23)
    out["onset_2026_yr"] = float(on26)
    out["last_bright_2026_yr"] = float(pre26)
    out["uvm2_2026_last_bright_mag"] = float(m[np.argmin(np.abs(t - pre26))])
    out["uvm2_2026_latest_mag"] = float(m[np.argmax(t)])
    out["uvm2_2026_latest_date_yr"] = float(t.max())
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for on, c, lab in ((on23, "C0", "2023 dip"), (on26, "C3", "2026")):
        if not np.isfinite(on):
            continue
        for ff, mk in (("UVM2", "o"), ("UVW2", "s")):
            tt = jd_to_year(uv[ff]["jd"])
            s = (tt > on - 1.0) & (tt < on + 1.6)
            axs[0].errorbar((tt[s] - on) * 365.25, uv[ff]["mag"][s], uv[ff]["err"][s], fmt=mk, ms=3, lw=0.5, color=c,
                            mfc=c if ff == "UVM2" else "none", label=f"{lab} {ff}")
        x = xr["Hard"]
        tx = jd_to_year(x["jd"])
        s = (tx > on - 1.0) & (tx < on + 1.6)
        axs[1].errorbar((tx[s] - on) * 365.25, x["rate"][s], x["err"][s], fmt=".", ms=3, lw=0.4, color=c, label=f"{lab} XRT hard")
    axs[0].invert_yaxis()
    axs[0].set_ylabel("UVOT Vega mag")
    axs[0].legend(fontsize=7, ncol=2)
    axs[0].set_title(f"UV collapse onsets aligned: 2023 ({on23:.3f}) vs 2026 ({on26:.3f}); UVM2 threshold {thr} mag")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("XRT hard (2-10 keV) c/s")
    axs[1].set_xlabel("days since UV collapse onset")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_uv_dip_onsets.png", dpi=140)
    plt.close(fig)
    save_results({"uv_dips": out})
    print("uvdips:", out)


STAGES["uvdips"] = stage_uvdips


# ---------------------------------------------------------------------------
# Stage: simultaneous UV-X-ray track (model-free boundary-layer diagnostic)
# ---------------------------------------------------------------------------
def stage_uvx():
    uv = load_uvot()
    xr = load_xrt()
    f = "UVM2"
    tu, mu, eu = uv[f]["jd"], uv[f]["mag"], uv[f]["err"]
    x = xr["Hard"]
    pairs = []
    for j, t in enumerate(x["jd"]):
        k = np.argmin(np.abs(tu - t))
        if abs(tu[k] - t) < 0.5:
            pairs.append((t, mu[k], eu[k], x["rate"][j], x["err"][j]))
    pairs = np.array(pairs)
    flux = tcrb.mag_to_flam(pairs[:, 1], f, FIDUCIAL.ebv)  # dereddened F_lambda
    yrs = jd_to_year(pairs[:, 0])
    from scipy.stats import spearmanr
    rho, p = spearmanr(flux, pairs[:, 3])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(flux, pairs[:, 3], c=yrs, cmap="viridis", s=14, zorder=3)
    ax.errorbar(flux, pairs[:, 3], yerr=pairs[:, 4], fmt="none", ecolor="0.7", lw=0.5, zorder=1)
    latest = yrs > 2026.70
    ax.scatter(flux[latest], pairs[latest, 3], s=120, facecolors="none", edgecolors="red", lw=1.5, zorder=4,
               label="2026 Sep 29 / Oct 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"UVM2 dereddened $F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)  [inner-disk proxy]")
    ax.set_ylabel("XRT hard 2-10 keV count/s  [optically thin boundary layer]")
    cb = fig.colorbar(sc)
    cb.set_label("year")
    ax.legend()
    ax.set_title(f"Same-day Swift UV vs hard X-ray (n = {len(pairs)}): Spearman rho = {rho:.2f} (p = {p:.1e})")
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_uv_xray_track.png", dpi=140)
    plt.close(fig)
    out = {"n_pairs": int(len(pairs)), "spearman_rho": float(rho), "spearman_p": float(p),
           "latest": [{"yr": float(y), "uvm2_flux": float(fl), "hard_rate": float(r)} for y, fl, r in
                      zip(yrs[latest], flux[latest], pairs[latest, 3])]}
    # where does the latest point sit relative to the 2023 dip vs the 2025 high branch?
    dip23 = (yrs > 2023.6) & (yrs < 2024.3)
    hi25 = (yrs > 2024.8) & (yrs < 2026.6)
    out["median_2023_dip"] = {"uvm2_flux": float(np.median(flux[dip23])), "hard_rate": float(np.median(pairs[dip23, 3]))}
    out["median_2025_26_high"] = {"uvm2_flux": float(np.median(flux[hi25])), "hard_rate": float(np.median(pairs[hi25, 3]))}
    save_results({"uv_xray_track": out})
    print("uvx:", out)


STAGES["uvx"] = stage_uvx


# ---------------------------------------------------------------------------
# Stage: fastest possible UV fade of a viscous disk after a complete supply shut-off
# ---------------------------------------------------------------------------
def stage_drain(mdot0: float = 2e-8):
    from scipy.linalg import lu_factor, lu_solve
    from celestrium.novae.tcrb import DAY, LAMBDA_EFF, MSUN, YR, ViscousDisk
    rows = []
    for tv in (0.75, 1.5, 3.0, 6.0):
        for n in (0.5, 1.0, 1.5):
            d = ViscousDisk(FIDUCIAL, tv, n, n_cells=160)
            s = d.steady_state() * mdot0 * MSUN / YR
            f0, m0 = d.band_flux(s, LAMBDA_EFF["UVM2"]), d.mdot_inner(s)
            lu = lu_factor(np.eye(d.grid.n) - DAY * d.L)
            t_uv = t_m = None
            curve = []
            for k in range(1, 3650):
                s = lu_solve(lu, s)
                dm = -2.5 * np.log10(d.band_flux(s, LAMBDA_EFF["UVM2"]) / f0)
                curve.append(dm)
                if t_uv is None and dm >= 2.0:
                    t_uv = k
                if t_m is None and d.mdot_inner(s) <= 0.1 * m0:
                    t_m = k
                if t_uv and t_m and k > 400:
                    break
            rows.append({"t_visc_out_yr": tv, "n_exp": n, "days_uvm2_fade_2mag": t_uv, "days_mdot_drop_10x": t_m,
                         "uvm2_fade_after_30d_mag": float(curve[29])})
    obs = json.loads(RESULTS.read_text()).get("uv_dips", {})
    dt_obs = (obs.get("onset_2026_yr", np.nan) - obs.get("last_bright_2026_yr", np.nan)) * 365.25
    dm_obs = obs.get("uvm2_2026_latest_mag", np.nan) - obs.get("uvm2_2026_last_bright_mag", np.nan)
    fastest = min(r["days_uvm2_fade_2mag"] for r in rows if r["days_uvm2_fade_2mag"])
    out = {"supply_shutoff_from_mdot": mdot0, "grid": rows, "fastest_2mag_fade_days": fastest,
           "max_fade_in_30d_mag": max(r["uvm2_fade_after_30d_mag"] for r in rows),
           "observed_2026_fade_mag": float(dm_obs), "observed_2026_interval_days": float(dt_obs)}
    save_results({"viscous_drain_test": out})
    print("drain:", {k: v for k, v in out.items() if k != "grid"})


STAGES["drain"] = stage_drain


# ---------------------------------------------------------------------------
# Stage: export compact data for the interactive page
# ---------------------------------------------------------------------------
def stage_export():
    uv = load_uvot()
    xr = load_xrt()
    av, _ = load_aavso()
    r3 = lambda a, n=3: [round(float(v), n) for v in a]  # noqa: E731
    out = {"uvm2": {"t": r3(jd_to_year(uv["UVM2"]["jd"]), 4), "m": r3(uv["UVM2"]["mag"], 2), "e": r3(uv["UVM2"]["err"], 2),
                    "w": [int(v > 0.5) for v in uv["UVM2"]["wing"]]},
           "hard": {"t": r3(jd_to_year(xr["Hard"]["jd"]), 4), "r": r3(xr["Hard"]["rate"], 4), "e": r3(xr["Hard"]["err"], 4)}}
    for b in ("V", "B", "U", "Vis"):
        if b in av:
            out[b] = {"t": r3(jd_to_year(av[b]["jd"]), 4), "m": r3(av[b]["mag"], 2)}
    curves = DATA / "inversion_curves.npz"
    if curves.exists():
        c = np.load(curves)
        k = slice(None, None, 4)
        out["mdot"] = {"t": r3(c["t"][k], 3), "mu": r3(c["mu"][k], 3), "sd": r3(c["sd"][k], 3)}
    ov = DATA / "overlay_curves.json"
    if ov.exists():
        out["overlay"] = json.loads(ov.read_text())
    out["results"] = json.loads(RESULTS.read_text())
    (DATA / "artifact_data.json").write_text(json.dumps(out, separators=(",", ":"), default=float))
    print("export:", (DATA / "artifact_data.json").stat().st_size // 1024, "kB")


STAGES["export"] = stage_export
STAGES["overlay"] = stage_overlay


if __name__ == "__main__":
    for st in (sys.argv[1:] or list(STAGES)):
        STAGES[st]()
