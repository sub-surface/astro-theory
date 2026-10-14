"""EXP-2026-Z amendment A4: fetch the DASCH light curves of every refcat neighbour within 20" of each target and
control, so detections DASCH filed under a neighbour ("source splitting") can be reassigned by measured position.

Neighbour light curves are cached once under data/rn/lc/nb/; data/rn/neighbours.csv lists (tid, role, ra, dec,
refcat, gsc, ref, sep). Resumable. Usage: python scripts/novae/rn_fetch_neighbours.py [--workers 6]
"""
import argparse
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from celestrium.archives import dasch  # noqa: E402

LC = ROOT / "data" / "rn" / "lc"
NB = LC / "nb"
OUT = ROOT / "data" / "rn" / "neighbours.csv"
DONE = ROOT / "data" / "rn" / "neighbours_done.txt"
R_NB = 20.0
_lock = threading.Lock()


def _pos_from_lc(path):
    c = dasch.parse_csv_lines(dasch.cached(path, None))
    if not c:
        return None
    m = np.isfinite(c.get("magcal_magdep", np.array([])))
    if m.sum() < 3:
        return None
    return float(np.nanmedian(c["ra_deg"][m])), float(np.nanmedian(c["dec_deg"][m]))


def one(tid, role, ra, dec, refcat, own_ref):
    cat = dasch.parse_csv_lines(dasch.querycat(ra, dec, R_NB, refcat))
    rows = []
    if not cat:
        return rows
    sep = np.hypot((cat["ra_deg"] - ra) * np.cos(np.radians(dec)), cat["dec_deg"] - dec) * 3600
    for j in range(len(sep)):
        ref, gsc = int(cat["ref_number"][j]), int(cat["gsc_bin_index"][j])
        if ref == own_ref or not (np.nan_to_num(cat["num_matches"][j]) > 0):
            continue
        dasch.cached(NB / f"{refcat}_{gsc}_{ref}.csv.gz", lambda g=gsc, r=ref: dasch.lightcurve(refcat, g, r),
                     reduce=True)
        rows.append(dict(tid=tid, role=role, ra=ra, dec=dec, refcat=refcat, gsc=gsc, ref=ref,
                         sep=round(float(sep[j]), 2), nb_ra=float(cat["ra_deg"][j]), nb_dec=float(cat["dec_deg"][j])))
    return rows


def job(row, ix):
    out = []
    for role, refcol in (("t", "t_ref"), ("c0", "c0_ref"), ("c1", "c1_ref")):
        if not np.isfinite(ix[refcol]):
            continue
        if role == "t":
            ra, dec = row.ra, row.dec
        else:
            ra, dec = float(ix[f"{role}_ra"]), float(ix[f"{role}_dec"])
        out += one(row.tid, role, ra, dec, ix.refcat, int(ix[refcol]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tids", default="")
    a = ap.parse_args()
    T = pd.read_csv(ROOT / "data" / "rn" / "targets.csv").set_index("tid", drop=False)
    I = pd.read_csv(ROOT / "data" / "rn" / "fetch_index.csv").drop_duplicates("tid", keep="last")
    I = I[I.status == "ok"].set_index("tid", drop=False)
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    tids = [t for t in (a.tids.split(",") if a.tids else I.index) if t not in done and t in I.index]
    new = not OUT.exists()
    f = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, ["tid", "role", "ra", "dec", "refcat", "gsc", "ref", "sep", "nb_ra", "nb_dec"])
    if new:
        w.writeheader()
    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, T.loc[t], I.loc[t]): t for t in tids}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                rows = fu.result()
            except Exception as e:
                print("error", t, str(e)[:100], flush=True)
                continue
            with _lock:
                for r in rows:
                    w.writerow(r)
                f.flush()
                with open(DONE, "a") as d:
                    d.write(t + "\n")
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(tids)}", flush=True)
    f.close()
    print("done", n)


if __name__ == "__main__":
    main()
