"""EXP-2026-Z: run D0 / D1 over every fetched target and control light curve.

Writes data/rn/scores.csv (discovery scan: best event outside +-2 yr of every catalogued eruption) and
data/rn/recall.csv (each catalogued eruption in 1890-1991, scanned only inside its own date window).
Usage: python scripts/novae/rn_search.py [--procs 10]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from celestrium.archives import dasch  # noqa: E402
from celestrium.novae import plates as P  # noqa: E402

LC = ROOT / "data" / "rn" / "lc"
DASCH_SPAN = (1885.0, 1992.0)


def eruptions(row):
    """[(epoch, lo, hi)] for catalogued eruptions; exact epochs for known RNe, year-level otherwise."""
    out = []
    ks = str(row.known_eruptions) if isinstance(row.known_eruptions, str) else ""
    exact = row.cls == "rn"
    for k in ks.split(";"):
        if not k:
            continue
        y = float(k)
        out.append((y, y - 0.3, y + 0.3) if exact else (y, y - 1.0, y + 1.0))
    return out


def load(tid, tag, nbrows, pos, ref=None):
    """Position-based light curve (amendment A4): own refcat source + reassigned neighbour detections."""
    f = LC / (f"{tid}_t.csv.gz" if tag == "t" else f"{tid}_{tag}_{int(ref)}.csv.gz")
    if not f.exists():
        return None
    own = dasch.parse_csv_lines(dasch.cached(f, None))
    if not own:
        return None
    nb = []
    for x in nbrows:
        g = LC / "nb" / f"{x['refcat']}_{x['gsc']}_{x['ref']}.csv.gz"
        if g.exists():
            nb.append((dasch.parse_csv_lines(dasch.cached(g, None)), x["nb_ra"], x["nb_dec"]))
    return dasch.tidy(dasch.assemble(own, nb, pos[0], pos[1]))


def run_one(args):
    row, ix, nbr = args
    er = eruptions(row)
    masks = [(lo - 2.0, hi + 2.0) for _, lo, hi in er]
    scores, recall = [], []
    for tag, mf in (("t", ix.t_mag), ("c0", ix.c0_mag), ("c1", ix.c1_mag)):
        if not np.isfinite(mf):
            continue
        pos = (row.ra, row.dec) if tag == "t" else (ix[f"{tag}_ra"], ix[f"{tag}_dec"])
        lc = load(row.tid, tag, [x for x in nbr if x["role"] == tag], pos, None if tag == "t" else ix[f"{tag}_ref"])
        if lc is None or len(lc["t"]) == 0:
            continue
        p = P.prepare(lc, float(mf))
        pdisc = P.prepare(P.drop_windows(lc, masks), float(mf)) if masks else p
        e1 = P.d1_scan(pdisc)
        e0 = P.d0_scan(pdisc)
        scores.append(dict(tid=row.tid, role=tag, n_good=int(len(p["t"])), n_det=int(p["det"].sum()),
                           m_faint=float(mf), s_int=float(p["s_int"]), d0=e0.score, d0_t0=e0.t0, d0_nights=e0.n_nights,
                           d1=e1.score, d1_t0=e1.t0, d1_t2=e1.t2_d, d1_mpeak=e1.m_peak, d1_amp=e1.extra.get("amp", np.nan),
                           d1_nights=e1.n_nights, d1_plates=e1.n_plates, d1_contra=e1.n_contra))
        if tag == "t" and np.isfinite(e1.t0):
            # A7 episode count: drop +-0.5 yr around each event found so far and rescan (dwarf-nova repeaters)
            mk, ev = list(masks), e1
            for j in (2, 3):
                mk.append((ev.t0 - 0.5, ev.t0 + 0.5))
                ev = P.d1_scan(P.prepare(P.drop_windows(lc, mk), float(mf)))
                scores[-1].update({f"d1_{j}": ev.score, f"d1_{j}_t0": ev.t0, f"d1_{j}_nights": ev.n_nights})
                if not np.isfinite(ev.t0):
                    break
        if tag != "t":
            continue
        for ep, lo, hi in er:
            if not (DASCH_SPAN[0] + 5 <= ep <= DASCH_SPAN[1] - 0.5):
                continue
            w = (p["t"] >= lo) & (p["t"] <= hi)
            a1 = P.d1_scan(p, allow=(lo, hi))
            a0 = P.d0_scan(p, allow=(lo, hi))
            recall.append(dict(tid=row.tid, epoch=ep, lo=lo, hi=hi, n_plates_win=int(w.sum()),
                               med_lim_win=float(np.nanmedian(p["lim"][w])) if w.any() else np.nan,
                               vpeak=row.vpeak, d0=a0.score, d0_nights=a0.n_nights, d1=a1.score, d1_t0=a1.t0,
                               d1_nights=a1.n_nights, d1_amp=a1.extra.get("amp", np.nan)))
    return scores, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--tids", default="")
    a = ap.parse_args()
    T = pd.read_csv(ROOT / "data" / "rn" / "targets.csv").set_index("tid", drop=False)
    I = pd.read_csv(ROOT / "data" / "rn" / "fetch_index.csv").drop_duplicates("tid", keep="last")
    I = I[I.status == "ok"].set_index("tid", drop=False)
    done = set((ROOT / "data" / "rn" / "neighbours_done.txt").read_text().split())
    NB = pd.read_csv(ROOT / "data" / "rn" / "neighbours.csv").drop_duplicates()
    nbg = {t: g.to_dict("records") for t, g in NB.groupby("tid")}
    tids = a.tids.split(",") if a.tids else list(I.index)
    jobs = [(T.loc[t], I.loc[t], nbg.get(t, [])) for t in tids if t in I.index and t in done]
    S, R = [], []
    with Pool(a.procs) as pool:
        for n, (s, r) in enumerate(pool.imap_unordered(run_one, jobs, chunksize=2)):
            S += s
            R += r
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
    pd.DataFrame(S).to_csv(ROOT / "data" / "rn" / "scores.csv", index=False)
    pd.DataFrame(R).to_csv(ROOT / "data" / "rn" / "recall.csv", index=False)
    print("wrote", len(S), "score rows,", len(R), "recall rows")


if __name__ == "__main__":
    main()
