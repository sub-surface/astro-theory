"""Fetch Swift/XRT (LSXPS per-ObsID light curves) and UVOT UV sky images for the other three
Galactic symbiotic recurrent novae (RS Oph, V3890 Sgr, V745 Sco), then run aperture photometry.

Output: data/srn/<slug>/lsxps/*.csv, data/srn/<slug>/uvot/*.img.gz, data/srn/<slug>/uvot_photometry.csv
Usage: python scripts/novae/fetch_family_swift.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celestrium.novae.tcrb import uvot_aperture_photometry  # noqa: E402

BASE = "https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs"
TARGETS = {  # slug: (SIMBAD ra, dec, LSXPS source id)
    "rs_oph": (267.5548298, -6.7079113, 1239688),
    "v3890_sgr": (277.6803657, -24.0191513, 227919),
    "v745_sco": (268.8426087, -33.2495977, 518593),
}


def obslist(ra, dec):
    import pyvo
    s = pyvo.dal.TAPService("https://heasarc.gsfc.nasa.gov/xamin/vo/tap")
    q = (f"SELECT obsid,start_time,xrt_exposure,uvot_exposure FROM swiftmastr WHERE "
         f"CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},0.15))=1 ORDER BY start_time")
    return s.search(q).to_table().to_pandas()


def main():
    import swifttools.ukssdc.data.SXPS as uds
    from astropy.time import Time
    sess = requests.Session()
    for slug, (ra, dec, sid) in TARGETS.items():
        out = ROOT / "data" / "srn" / slug
        (out / "lsxps").mkdir(parents=True, exist_ok=True)
        (out / "uvot").mkdir(parents=True, exist_ok=True)
        lcs = uds.getLightCurves(sourceID=sid, cat="LSXPS", binning="obsid", timeFormat="MJD", bands="all",
                                 returnData=True, saveData=False)
        for k, v in lcs.items():
            if hasattr(v, "to_csv"):
                v.to_csv(out / "lsxps" / f"{k}.csv", index=False)
        obs = obslist(ra, dec)
        obs = obs[obs.uvot_exposure > 0]
        obs.to_csv(out / "obslist.csv", index=False)
        rows = []
        for _, r in obs.iterrows():
            oid = str(r.obsid).zfill(11)
            ym = Time(float(r.start_time), format="mjd").strftime("%Y_%m")
            url = f"{BASE}/{ym}/{oid}/uvot/image/"
            try:
                listing = sess.get(url, timeout=60).text
            except Exception as exc:  # network hiccup: skip this pointing
                print("list fail", slug, oid, exc, flush=True)
                continue
            for name in sorted(set(re.findall(r"sw\d{11}(?:uw2|um2|uw1)_sk\.img\.gz", listing))):
                dest = out / "uvot" / name
                if not (dest.exists() and dest.stat().st_size > 0):
                    for _ in range(3):
                        try:
                            rr = sess.get(url + name, timeout=120)
                            rr.raise_for_status()
                            dest.write_bytes(rr.content)
                            break
                        except Exception as exc:
                            print("retry", name, exc, flush=True)
                            time.sleep(5)
                try:
                    for ph in uvot_aperture_photometry(dest, ra=ra, dec=dec):
                        ph["file"] = name
                        rows.append(ph)
                except Exception as exc:
                    print("phot fail", name, exc, flush=True)
        pd.DataFrame(rows).to_csv(out / "uvot_photometry.csv", index=False)
        print(slug, "obs", len(obs), "snapshots", len(rows), flush=True)


if __name__ == "__main__":
    main()
