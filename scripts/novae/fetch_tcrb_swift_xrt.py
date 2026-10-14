"""Fetch Swift/XRT per-ObsID light curves of T CrB from LSXPS (source 2012) as CSV (EXP-2026-Y)."""
from pathlib import Path

import swifttools.ukssdc.data.SXPS as uds

OUT = Path(__file__).resolve().parents[2] / "data" / "tcrb" / "lsxps"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lcs = uds.getLightCurves(sourceID=2012, cat="LSXPS", binning="obsid", timeFormat="MJD",
                             bands="all", returnData=True, saveData=False)
    for k, v in lcs.items():
        if hasattr(v, "to_csv"):
            v.to_csv(OUT / f"{k}.csv", index=False)
    print(sorted(p.name for p in OUT.iterdir()))
