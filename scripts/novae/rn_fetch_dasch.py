"""EXP-2026-Z: fetch DASCH DR7 light curves for every target and two matched field-star controls.

For each target: one refcat cone (ATLAS, APASS fallback) gives the target match (nearest within 5") and the
controls (2 stars 0.4-1x the cone radius away, |dmag| < 0.5, not flagged variable, similar plate count).
Raw API CSV is cached under data/rn/lc/<tid>_{t,c0,c1}.csv; data/rn/fetch_index.csv records the matches.
Resumable: cached files are not refetched. Usage: python scripts/novae/rn_fetch_dasch.py [--workers 4] [--limit N]
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
INDEX = ROOT / "data" / "rn" / "fetch_index.csv"
FIELDS = ["tid", "refcat", "radius", "t_ref", "t_gsc", "t_sep", "t_mag", "t_nmatch", "n_within5",
          "c0_ref", "c0_gsc", "c0_mag", "c0_ra", "c0_dec", "c1_ref", "c1_gsc", "c1_mag", "c1_ra", "c1_dec", "status"]
_lock = threading.Lock()


def _cone(ra, dec):
    for rc in ("atlas", "apass"):
        for rad in (300.0, 120.0, 60.0):
            try:
                cat = dasch.parse_csv_lines(dasch.querycat(ra, dec, rad, rc))
            except RuntimeError:
                continue
            if cat and len(cat.get("ra_deg", [])) and np.nanmin(_sep(cat, ra, dec)) <= 5:
                return rc, rad, cat
            break  # cone worked but no match: try the other refcat
    return None, None, None


def _sep(cat, ra, dec):
    d = np.radians
    c = np.sin(d(cat["dec_deg"])) * np.sin(d(dec)) + np.cos(d(cat["dec_deg"])) * np.cos(d(dec)) * np.cos(
        d(cat["ra_deg"] - ra))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def _controls(row, rc, rad, cat, sep, i):
    """Two field stars like the target: |dmag| < 0.5 (then 1.0), not flagged variable, 0.4-1x the cone radius away,
    and with similar plate visibility. A target DASCH rarely detects (num_matches NaN or < 20, i.e. below most
    plate limits) gets equally invisible controls; a detected target gets controls with >= half its matches."""
    def pick_from(cat, sep, i, inner):
        nm = np.nan_to_num(cat["num_matches"])
        vflag = np.nan_to_num(cat.get("v_flag", np.zeros_like(sep)))
        nm_t = nm[i]
        vis = (nm < 20) if nm_t < 20 else (nm >= 0.5 * nm_t)
        for dm in (0.5, 1.0):
            ok = (sep > inner) & (sep <= rad_now) & (np.abs(cat["stdmag"] - cat["stdmag"][i]) < dm) & (vflag == 0) & vis
            cand = np.where(ok)[0]
            if len(cand) >= 2:
                return cand
        return cand
    rad_now = rad
    cand = pick_from(cat, sep, i, 0.4 * rad)
    if len(cand) < 2 and rad >= 300:  # sparse (bright) case: widen to a 900" annulus
        try:
            wide = dasch.parse_csv_lines(dasch.querycat(row.ra, row.dec, 900.0, rc))
        except RuntimeError:
            wide = None
        if wide:
            cat, sep = wide, _sep(wide, row.ra, row.dec)
            i = int(np.nanargmin(sep))
            rad_now = 900.0
            cand = pick_from(cat, sep, i, 120.0)
    rng = np.random.default_rng(int(row.tid[1:]))
    pick = rng.choice(cand, size=min(2, len(cand)), replace=False) if len(cand) else []
    return pick, cat, sep


def one(row):
    tid = row.tid
    out = {"tid": tid}
    rc, rad, cat = _cone(row.ra, row.dec)
    if cat is None:
        out["status"] = "no_refcat_match"
        return out
    sep = _sep(cat, row.ra, row.dec)
    i = int(np.nanargmin(sep))
    nm = cat["num_matches"]
    out.update(refcat=rc, radius=rad, t_ref=int(cat["ref_number"][i]), t_gsc=int(cat["gsc_bin_index"][i]),
               t_sep=round(float(sep[i]), 2), t_mag=cat["stdmag"][i], t_nmatch=nm[i],
               n_within5=int((sep <= 5).sum()))
    pick, cat_c, sep_c = _controls(row, rc, rad, cat, sep, i)
    for k, j in enumerate(pick):
        out[f"c{k}_ref"], out[f"c{k}_gsc"], out[f"c{k}_mag"] = (int(cat_c["ref_number"][j]),
                                                                int(cat_c["gsc_bin_index"][j]), cat_c["stdmag"][j])
        out[f"c{k}_ra"], out[f"c{k}_dec"] = float(cat_c["ra_deg"][j]), float(cat_c["dec_deg"][j])
    dasch.cached(LC / f"{tid}_t.csv.gz", lambda: dasch.lightcurve(rc, out["t_gsc"], out["t_ref"]), reduce=True)
    for k in range(len(pick)):
        g, r = out[f"c{k}_gsc"], out[f"c{k}_ref"]
        dasch.cached(LC / f"{tid}_c{k}_{r}.csv.gz", lambda g=g, r=r: dasch.lightcurve(rc, g, r), reduce=True)
    out["status"] = "ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tids", default="")
    a = ap.parse_args()
    t = pd.read_csv(ROOT / "data" / "rn" / "targets.csv")
    done = set(pd.read_csv(INDEX)["tid"]) if INDEX.exists() else set()
    todo = t[~t.tid.isin(done)]
    if a.tids:
        todo = todo[todo.tid.isin(a.tids.split(","))]
    if a.limit:
        todo = todo.head(a.limit)
    new = not INDEX.exists()
    f = open(INDEX, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, r): r.tid for r in todo.itertuples()}
        for fu in as_completed(futs):
            try:
                res = fu.result()
            except Exception as e:
                res = {"tid": futs[fu], "status": f"error:{str(e)[:80]}"}
            with _lock:
                w.writerow(res)
                f.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    f.close()
    print("done", n)


if __name__ == "__main__":
    main()
