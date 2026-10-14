"""EXP-2026-Z: score the pre-registered result conditions from data/rn/scores.csv and recall.csv.

R1  recall of catalogued eruptions (unconditional and coverage-conditional, Wilson intervals)
H_PI D1 vs D0 recall at control false-alarm rates of 1% and 0.1% per light curve, paired bootstrap over targets
R2  discovery candidates: target events above the control maximum, with >= 2 nights
Writes docs/results/experiment_z_rn_results.json and data/rn/candidates.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RN = ROOT / "data" / "rn"
OUT = ROOT / "docs" / "results" / "experiment_z_rn_results.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def thresholds(ctrl, col, alphas=(0.01, 0.001)):
    x = np.sort(ctrl[col].values)
    out = {}
    for a in alphas:
        tau = float(np.quantile(x, 1 - a, method="higher"))
        out[a] = (tau, float((x > tau).mean()))  # achieved FAR (ties make it <= a)
    return out


def recall_at(rec, col, tau):
    return (rec[col].values > tau)


def h_ml(S, R, a=0.01, nboot=2000):
    """H_ML: D2 vs D1 recall at control FAR a, both thresholded on the same held-out (test-half) controls."""
    f = RN / "d2_eval.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    if len(z["pos"]) != len(R):
        return {"error": "d2_eval.npz is stale (recall.csv changed); rerun rn_d2.py"}
    c2 = pd.DataFrame({"tid": z["ctrl_tid"], "role": z["ctrl_role"], "d2": z["ctrl"]})
    c1 = S[S.role != "t"][["tid", "role", "d1"]]
    C = c2.merge(c1, on=["tid", "role"])
    rc = R.assign(d2=z["pos"])
    rc = rc[(rc.n_plates_win >= 3) & np.isfinite(rc.d2)]
    def rec(cc, rr):
        t1 = np.quantile(cc.d1.values, 1 - a, method="higher")
        t2 = np.quantile(cc.d2.values, 1 - a, method="higher")
        return (rr.d1.values > t1).mean(), (rr.d2.values > t2).mean()
    r1, r2 = rec(C, rc)
    rng = np.random.default_rng(3)
    gc = {t: np.where(C.tid.values == t)[0] for t in C.tid.unique()}
    gr = {t: np.where(rc.tid.values == t)[0] for t in rc.tid.unique()}
    kc, kr = list(gc), list(gr)
    d = []
    for _ in range(nboot):
        ic = np.concatenate([gc[t] for t in rng.choice(kc, len(kc))])
        ir = np.concatenate([gr[t] for t in rng.choice(kr, len(kr))])
        x1, x2 = rec(C.iloc[ic], rc.iloc[ir])
        d.append(x2 - x1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta = r2 - r1
    verdict = ("learned denoising adds information" if delta >= 0.05 and lo > 0 else
               "sim-to-real gap dominates" if delta <= -0.05 and hi < 0 else "physics prior suffices")
    info = json.loads((RN / "d2_info.json").read_text()) if (RN / "d2_info.json").exists() else {}
    return {"far": a, "n_test_controls": int(len(C)), "n_eruptions": int(len(rc)), "recall_d1": float(r1),
            "recall_d2": float(r2), "delta": float(delta), "delta_ci95": [float(lo), float(hi)], "verdict": verdict,
            "train": info}


def main():
    S = pd.read_csv(RN / "scores.csv")
    R = pd.read_csv(RN / "recall.csv")
    T = pd.read_csv(RN / "targets.csv").set_index("tid")
    ctrl = S[S.role != "t"].copy()
    targ = S[S.role == "t"].copy()
    R["covered"] = R.n_plates_win >= 3
    res = {"n_targets_scored": int(len(targ)), "n_controls": int(len(ctrl)), "n_eruptions": int(len(R)),
           "n_eruptions_covered": int(R.covered.sum())}

    # thresholds and H_PI
    th = {d: thresholds(ctrl, d) for d in ("d0", "d1")}
    res["thresholds"] = {d: {str(a): {"tau": v[0], "far_achieved": v[1]} for a, v in th[d].items()} for d in th}
    rng = np.random.default_rng(1)
    hp = {}
    for a in (0.01, 0.001):
        rc = R[R.covered]
        r0 = recall_at(rc, "d0", th["d0"][a][0])
        r1 = recall_at(rc, "d1", th["d1"][a][0])
        # paired bootstrap: resample eruption rows by target and control rows by target, recomputing thresholds
        tids_r = rc.tid.unique()
        tids_c = ctrl.tid.unique()
        gr = {t: np.where(rc.tid.values == t)[0] for t in tids_r}
        gc = {t: np.where(ctrl.tid.values == t)[0] for t in tids_c}
        diffs = []
        for _ in range(2000):
            ir = np.concatenate([gr[t] for t in rng.choice(tids_r, len(tids_r))])
            ic = np.concatenate([gc[t] for t in rng.choice(tids_c, len(tids_c))])
            c = ctrl.iloc[ic]
            t0_ = np.quantile(c.d0.values, 1 - a, method="higher")
            t1_ = np.quantile(c.d1.values, 1 - a, method="higher")
            diffs.append((rc.d1.values[ir] > t1_).mean() - (rc.d0.values[ir] > t0_).mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        hp[str(a)] = {"recall_d0": float(r0.mean()), "recall_d1": float(r1.mean()),
                      "ci_d0": wilson(r0.sum(), len(r0)), "ci_d1": wilson(r1.sum(), len(r1)),
                      "delta": float(r1.mean() - r0.mean()), "delta_ci95": [float(lo), float(hi)], "n": int(len(rc))}
    res["H_PI"] = hp
    d = hp["0.01"]
    res["H_PI_verdict"] = ("SUPPORTED" if d["delta"] >= 0.10 and d["delta_ci95"][0] > 0 else "NOT SUPPORTED")

    # R1 recall at the 1% threshold, by coverage and peak brightness relative to plate depth
    tau1 = th["d1"][0.01][0]
    R["rec_d1"] = R.d1 > tau1
    R["depth"] = R.med_lim_win - R.vpeak
    r1 = {"unconditional": {"k": int(R.rec_d1.sum()), "n": int(len(R)), "ci": wilson(R.rec_d1.sum(), len(R))},
          "covered": {"k": int(R[R.covered].rec_d1.sum()), "n": int(R.covered.sum()),
                      "ci": wilson(R[R.covered].rec_d1.sum(), R.covered.sum())}}
    bins = [(-99, 2), (2, 4), (4, 6), (6, 99)]
    r1["by_depth_covered"] = []
    for lo, hi in bins:
        m = R.covered & (R.depth >= lo) & (R.depth < hi)
        r1["by_depth_covered"].append({"depth_mag": [lo, hi], "k": int(R[m].rec_d1.sum()), "n": int(m.sum()),
                                        "ci": wilson(R[m].rec_d1.sum(), m.sum())})
    res["R1"] = r1

    # R2 candidates
    cmax = float(ctrl.d1.max())
    # GPD-free empirical tail: expected false events in the target set at threshold x = N_t * P_ctrl(score > x)
    cand = targ[(targ.d1 > cmax) & (targ.d1_nights >= 2)].copy()
    # A7: a dwarf-nova-like repeater has >= 2 further events above the 1% threshold within 5 yr of the main one
    tau01 = th["d1"][0.01][0]
    nrep = np.zeros(len(cand), int)
    for j in (2, 3):
        if f"d1_{j}" in cand:
            nrep += ((cand[f"d1_{j}"] > tau01) & (np.abs(cand[f"d1_{j}_t0"] - cand.d1_t0) < 5)).fillna(False).values
    cand["repeater"] = nrep >= 2
    cand = cand.join(T[["name", "cls", "ra", "dec", "known_eruptions", "vpeak", "src"]], on="tid")
    cand = cand.sort_values("d1", ascending=False)
    cand.to_csv(RN / "candidates.csv", index=False)
    res["R2"] = {"control_max_d1": cmax, "n_candidates_pre_vetting": int(len(cand)),
                 "n_repeaters_vetoed_A7": int(cand.repeater.sum()),
                 "candidates": cand[["tid", "name", "cls", "d1", "d1_t0", "d1_t2", "d1_mpeak", "d1_amp", "d1_nights",
                                     "d1_contra", "repeater", "known_eruptions"]].head(40).to_dict("records")}
    vf = RN / "vetted.csv"
    if vf.exists():
        V = pd.read_csv(vf).set_index("tid")
        res["R2"]["vetting"] = {k: int(v) for k, v in V.verdict.value_counts().items()}
        res["R2"]["genuine"] = V[V.verdict == "genuine"].reset_index().to_dict("records")
    res["H_ML"] = h_ml(S, R)
    # fraction of targets vs controls above thresholds (excess = signal)
    res["excess"] = {str(a): {"targets_above": float((targ.d1 > th["d1"][a][0]).mean()),
                              "controls_above": th["d1"][a][1]} for a in (0.01, 0.001)}
    # R3 population: hidden-RN fraction among Schaefer 2022 classical novae, given completeness
    cf = RN / "completeness.csv"
    if cf.exists():
        C = pd.read_csv(cf)
        tau = th["d1"][0.01][0] if len(sys.argv) < 2 else float(sys.argv[1])
        tau = max(tau, cmax)  # discovery threshold: above every control
        C["rec"] = (C.score > tau) & (C.nights >= 2)
        sch_cn = set(C.tid)
        vetted = set()
        vf = RN / "vetted.csv"  # tid, verdict (genuine / artifact / known) written after image vetting
        if vf.exists():
            V = pd.read_csv(vf)
            vetted = set(V[V.verdict == "genuine"].tid)
        k = len(vetted & sch_cn)
        fgrid = np.linspace(0, 1, 2001)
        r3 = {"threshold": tau, "n_novae": len(sch_cn), "k_detected_genuine": k}
        for speed in ("typical", "fast"):
            cbar = C[C.cls == speed].groupby("tid").rec.mean()  # averaged over the log-spaced T grid
            lam1 = float(cbar.sum())
            post = np.exp(-fgrid * lam1) * (fgrid * lam1) ** k
            post /= np.trapezoid(post, fgrid)
            cdf = np.cumsum(post) * (fgrid[1] - fgrid[0])
            by_T = C[C.cls == speed].groupby("T").rec.mean()
            r3[speed] = {"sum_completeness": lam1, "mean_completeness": float(cbar.mean()),
                         "f_upper95": float(fgrid[np.searchsorted(cdf, 0.95)]),
                         "f_median": float(fgrid[np.searchsorted(cdf, 0.5)]),
                         "completeness_by_T": {int(t): float(v) for t, v in by_T.items()}}
        r3["comparison"] = "Pagnotta & Schaefer 2014: 24 +- 4% (methods span 12-35%)"
        res["R3"] = r3
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "R2"}, indent=1, default=float))
    print("candidates:", len(cand))
    print(cand[["tid", "name", "cls", "d1", "d1_t0", "d1_t2", "d1_mpeak", "d1_amp", "d1_nights", "d1_contra",
                "repeater", "known_eruptions"]].head(40).to_string())


if __name__ == "__main__":
    main()
