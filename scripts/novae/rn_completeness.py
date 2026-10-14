"""EXP-2026-Z R3: per-nova completeness for hidden recurrence on the Harvard plates (injection-recovery).

For each Schaefer 2022 classical nova (known RNe excluded) with a position-assembled light curve: assume it is
an RN with recurrence time T (grid 10-100 yr, each interval jittered by 10%), place every implied eruption
Y +- kT (Y = catalogued eruption) inside 1890-1990 and outside the dropped +-3 yr around Y, at that nova's own
catalogued peak (B_peak = V_peak + 0.23 + E(B-V); (B-V)_0 ~ +0.23 at peak, van den Bergh & Younger 1987), into its
real plate cadence and limits, and run D1 on +-3 yr around each injected eruption. A plate detects the injected
nova with probability (1 - MISS) * Phi((lim - m)/0.3). Speeds: t2 log-uniform in [5, 100] d ("typical") and
t2 = 3 d ("fast", RN-like). Per realisation we keep the best (score, nights, |t0 - t_inj|) over its eruptions;
the threshold is applied in rn_analyse.py.
Usage: python scripts/novae/rn_completeness.py [--n 6] [--procs 10] [--resume]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "novae"))
from celestrium.novae import plates as P  # noqa: E402
from rn_search import eruptions, load  # noqa: E402

RN = ROOT / "data" / "rn"


def inject(lc, m_q, t_inj, b_peak, t2, rng):
    out = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in lc.items()}
    g = out["good"]
    dt = (out["t"] - t_inj) * 365.25
    f = P.template_flux(dt, t2)
    m_inj = -2.5 * np.log10(10 ** (-0.4 * m_q) + 10 ** (-0.4 * b_peak) * f)
    act = g & (f > 1e-3) & np.isfinite(out["lim"])
    p_det = (1 - P.MISS) * ndtr((out["lim"] - m_inj) / 0.3)
    hit = act & (rng.random(len(m_inj)) < p_det) & (m_inj < np.where(out["det"], out["mag"], 99))
    out["mag"] = np.where(hit, m_inj + rng.normal(0, 0.2, len(m_inj)), out["mag"])
    out["err"] = np.where(hit, 0.2, out["err"])
    out["det"] = out["det"] | hit
    return out


T_GRID = (10, 15, 20, 30, 40, 50, 65, 80, 100)


def run_target(args):
    row, ix, nbr, n = args
    rng = np.random.default_rng(int(row.tid[1:]) + 7)
    lc = load(row.tid, "t", [x for x in nbr if x["role"] == "t"], (row.ra, row.dec))
    if lc is None or len(lc["t"]) == 0:
        return []
    er = eruptions(row)
    masks = [(lo - 2.0, hi + 2.0) for _, lo, hi in er]
    lc = P.drop_windows(lc, masks)
    y = er[0][0]
    ebv = row.ebv if np.isfinite(row.ebv) else 0.6
    b_peak = row.vpeak + 0.23 + ebv
    m_q = float(ix.t_mag)
    out = []
    for cls in ("typical", "fast"):
        for T in T_GRID:
            for k in range(n):
                # eruption epochs implied by recurrence T with 10% interval jitter, both directions from Y
                eps, e = [], y
                while e > 1890:
                    e -= T * (1 + 0.1 * rng.standard_normal())
                    eps.append(e)
                e = y
                while e < 1990:
                    e += T * (1 + 0.1 * rng.standard_normal())
                    eps.append(e)
                eps = [x for x in eps if 1890 <= x <= 1990 and not any(a <= x <= b for a, b in masks)]
                best = dict(score=0.0, nights=0, dt0=np.nan)
                if eps:
                    t2s = [float(np.exp(rng.uniform(np.log(5), np.log(100)))) if cls == "typical" else 3.0 for _ in eps]
                    inj = lc
                    for x, t2 in zip(eps, t2s):
                        inj = inject(inj, m_q, x, b_peak, t2, rng)
                    for x in eps:
                        w = np.abs(inj["t"] - x) < 3
                        sub = {kk: (v[w] if hasattr(v, "__len__") and len(v) == len(w) else v) for kk, v in inj.items()}
                        if w.sum() == 0:
                            continue
                        ev = P.d1_scan(P.prepare(sub, m_q))
                        ok = np.isfinite(ev.t0) and abs(ev.t0 - x) < 0.5
                        if ok and (ev.n_nights >= 2) and ev.score > best["score"]:
                            best = dict(score=ev.score, nights=ev.n_nights, dt0=ev.t0 - x)
                out.append(dict(tid=row.tid, cls=cls, T=T, n_eps=len(eps), **best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--tids", default="")
    ap.add_argument("--resume", action="store_true", help="keep completeness.csv rows and run only missing novae")
    ap.add_argument("--out", default="completeness.csv")
    ap.add_argument("--exclude-file", default="", help="skip tids listed (one per line), e.g. a run in progress")
    a = ap.parse_args()
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    I = pd.read_csv(RN / "fetch_index.csv").drop_duplicates("tid", keep="last")
    I = I[I.status == "ok"].set_index("tid", drop=False)
    done = set((RN / "neighbours_done.txt").read_text().split())
    NB = pd.read_csv(RN / "neighbours.csv").drop_duplicates()
    nbg = {t: g.to_dict("records") for t, g in NB.groupby("tid")}
    sch = T[T.src.str.startswith("Schaefer22") & np.isfinite(T.vpeak) & (T.cls != "rn")]
    tids = a.tids.split(",") if a.tids else [t for t in sch.index if t in I.index and t in done]
    rows = []
    out = RN / a.out
    if a.exclude_file:
        skip = set(Path(a.exclude_file).read_text().split())
        tids = [t for t in tids if t not in skip]
    if a.resume and out.exists():
        old = pd.read_csv(out)
        rows = old.to_dict("records")
        tids = [t for t in tids if t not in set(old.tid)]
    jobs = [(T.loc[t], I.loc[t], nbg.get(t, []), a.n) for t in tids]
    with Pool(a.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(run_target, jobs, chunksize=1)):
            rows += r
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", len(rows), "injections;", len(jobs), "novae run now")


if __name__ == "__main__":
    main()
