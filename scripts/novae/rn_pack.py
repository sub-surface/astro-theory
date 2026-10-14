"""EXP-2026-Z H_ML: pack every position-assembled light curve (targets + controls) into data/rn/packed.npz.

Arrays are concatenated rows (t, val, det, err, lim, sat) with per-light-curve offsets, plus metadata
(tid, role, m_faint, split). The split (train/test by target ID, 50/50, seeded) is fixed here so D1 and D2 are
evaluated against the same held-out controls. Only control light curves are ever used for training; target
light curves (which carry the real eruptions) are used only for evaluation.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "novae"))
from rn_search import load  # noqa: E402

RN = ROOT / "data" / "rn"


def one(args):
    row, ix, nbr = args
    out = []
    for tag in ("t", "c0", "c1"):
        mf = ix.t_mag if tag == "t" else ix.get(f"{tag}_mag", np.nan)
        if not np.isfinite(mf):
            continue
        pos = (row.ra, row.dec) if tag == "t" else (ix[f"{tag}_ra"], ix[f"{tag}_dec"])
        lc = load(row.tid, tag, [x for x in nbr if x["role"] == tag], pos, None if tag == "t" else ix[f"{tag}_ref"])
        if lc is None or len(lc["t"]) == 0:
            continue
        g = lc["good"]
        val = np.where(lc["det"], lc["mag"], lc["lim"])
        arr = np.stack([lc["t"][g], val[g], lc["det"][g].astype(float), lc["err"][g], lc["lim"][g],
                        lc["sat"][g].astype(float)], 1).astype(np.float32)
        arr = arr[np.isfinite(arr).all(1)]
        out.append((row.tid, tag, float(mf), arr))
    return out


def main():
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    I = pd.read_csv(RN / "fetch_index.csv").drop_duplicates("tid", keep="last")
    I = I[I.status == "ok"].set_index("tid", drop=False)
    done = set((RN / "neighbours_done.txt").read_text().split())
    NB = pd.read_csv(RN / "neighbours.csv").drop_duplicates()
    nbg = {t: g.to_dict("records") for t, g in NB.groupby("tid")}
    jobs = [(T.loc[t], I.loc[t], nbg.get(t, [])) for t in I.index if t in done]
    res = []
    with Pool(10) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=4):
            res += r
    res.sort(key=lambda x: (x[0], x[1]))
    tids = sorted({r[0] for r in res})
    rng = np.random.default_rng(2026)
    train_tids = set(rng.choice(tids, len(tids) // 2, replace=False))
    rows = np.concatenate([r[3] for r in res])
    off = np.cumsum([0] + [len(r[3]) for r in res])
    np.savez_compressed(RN / "packed.npz", rows=rows, off=off, tid=np.array([r[0] for r in res]),
                        role=np.array([r[1] for r in res]), m_faint=np.array([r[2] for r in res], np.float32),
                        train=np.array([r[0] in train_tids for r in res]))
    print("packed", len(res), "light curves,", len(rows), "rows;", (RN / "packed.npz").stat().st_size // 2**20, "MB")


if __name__ == "__main__":
    main()
