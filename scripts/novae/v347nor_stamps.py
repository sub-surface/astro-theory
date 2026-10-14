"""V347 Nor: is the star on the plates DASCH logs as non-detections during the 1943-46 eruption?

For every plate in the eruption window with no catalogued detection at V347 Nor, fetch the DASCH cutout and measure
the median density within 10" of the Gaia position above the 100-200" annulus, in units of the annulus scatter
(S). Saturated images on the patrol plates are ~70" across, so a tighter annulus sits inside the star.
Controls are the same measurement on plates of the same position in quiescence (1938-42, 1947-52), where the star
(B ~ 17-20) is far below every plate limit; their S distribution includes the 6.5" neighbour and any blending.
A plate counts as showing the star if S exceeds the 99th percentile of the controls. Per-plate S is written out.
Usage: python scripts/novae/v347nor_stamps.py
"""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from celestrium.archives import dasch  # noqa: E402

RN = ROOT / "data" / "rn"
CACHE = RN / "v347nor" / "stamps"
OUT = ROOT / "docs" / "results" / "v347nor_stamps.json"
TID = "T1274"
WINDOWS = {"eruption": (1943.15, 1946.6), "pre": (1942.9, 1943.15), "control": [(1938.0, 1942.9), (1947.0, 1952.0)]}


def stamp(series, plate, sol, ra, dec):
    f = CACHE / f"{series}{int(plate):05d}_{int(sol)}.fits"
    if not f.exists():
        b = dasch.cutout_fits(series, plate, sol, ra, dec)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(b)
        tmp.replace(f)
    return fits.open(io.BytesIO(f.read_bytes()))[0]


def measure(h, ra, dec):
    w = WCS(h.header)
    x, y = (float(v) for v in w.world_to_pixel_values(ra, dec))
    scale = abs(w.pixel_scale_matrix[0, 0]) * 3600
    yy, xx = np.indices(h.data.shape)
    r = np.hypot(xx - x, yy - y) * scale
    d = h.data.astype(float)
    ann = d[(r > 100) & (r < 200) & np.isfinite(d)]
    if ann.size < 500 or not (r < 10).any():
        return np.nan
    sd = 1.4826 * np.median(np.abs(ann - np.median(ann)))
    return float((np.nanmedian(d[r < 10]) - np.median(ann)) / sd) if sd > 0 else np.nan


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    ra, dec = float(T.loc[TID].ra), float(T.loc[TID].dec)
    c = dasch.parse_csv_lines(dasch.cached(RN / "lc" / f"{TID}_t.csv.gz", None))
    yr = 2000 + (c["date_jd"] - 2451545.0) / 365.25
    nondet = ~np.isfinite(c["magcal_magdep"])
    rows = []
    for i in np.where(nondet)[0]:
        lab = ("eruption" if WINDOWS["eruption"][0] <= yr[i] <= WINDOWS["eruption"][1] else
               "pre" if WINDOWS["pre"][0] <= yr[i] < WINDOWS["pre"][1] else
               "control" if any(a <= yr[i] <= b for a, b in WINDOWS["control"]) else None)
        if lab:
            rows.append((lab, str(c["series"][i]), int(c["plate_number"][i]), int(c["solution_number"][i]),
                         float(c["date_jd"][i]), float(c["limiting_mag_local"][i])))
    D = pd.DataFrame(rows, columns=["set", "series", "plate", "sol", "jd", "lim"]).drop_duplicates(["series", "plate"])

    def one(r):
        try:
            return measure(stamp(r.series, r.plate, r.sol, ra, dec), ra, dec)
        except Exception as e:  # noqa: BLE001
            print(f"{r.series}{r.plate:05d}: {str(e)[:80]}", file=sys.stderr)
            return np.nan

    with ThreadPoolExecutor(4) as ex:
        D["S"] = list(ex.map(one, D.itertuples()))
    D["plate_id"] = [f"{s}{p:05d}" for s, p in zip(D.series, D.plate)]
    D["date_utc"] = Time(D.jd.values, format="jd").iso
    ctl = D[(D.set == "control") & np.isfinite(D.S)]
    thr = float(np.percentile(ctl.S, 99))
    er = D[(D.set == "eruption") & np.isfinite(D.S)].sort_values("jd")
    er_1943 = er[er.jd < Time("1944-01-01").jd]
    out = {"n_control": len(ctl), "control_S_median": round(float(ctl.S.median()), 1),
           "control_S_p99": round(thr, 1), "control_S_max": round(float(ctl.S.max()), 1),
           "n_eruption_nondetections_measured": len(er), "n_above_p99": int((er.S > thr).sum()),
           "n_1943": len(er_1943), "n_1943_above_p99": int((er_1943.S > thr).sum()),
           "first_months": er[er.jd < Time("1943-05-03").jd][["plate_id", "date_utc", "lim", "S"]]
           .round(2).to_dict("records"),
           "by_year": {str(y): [int(len(g)), int((g.S > thr).sum())] for y, g in er.groupby(er.date_utc.str[:4])},
           "pre_eruption": D[(D.set == "pre") & np.isfinite(D.S)].sort_values("jd")[["plate_id", "date_utc", "lim", "S"]]
           .round(2).to_dict("records"),
           "failed": int(D.S.isna().sum())}
    (RN / "v347nor").mkdir(exist_ok=True)
    D.drop(columns=["jd"]).round(3).to_csv(RN / "v347nor" / "stamps_S.csv", index=False)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
