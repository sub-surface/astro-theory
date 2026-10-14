"""Fetch Swift/UVOT UV sky images (uvw2, uvm2, uvw1) for every T CrB pointing (EXP-2026-Y).

Only the *_sk.img.gz sky images are pulled (~1.5 MB each); photometry is done in
celestrium.novae.tcrb.uvot_photometry without HEASoft. Resumable; output data/tcrb/uvot/.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "tcrb" / "uvot"
BASE = "https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs"
FILTERS = ("uw2", "um2", "uw1")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obs = Table.read(ROOT / "data" / "tcrb" / "swift_obslist.ecsv")
    s = requests.Session()
    for row in obs:
        if float(row["uvot_exposure"]) <= 0:
            continue
        oid, ym = str(row["obsid"]), str(row["date"])
        url = f"{BASE}/{ym}/{oid}/uvot/image/"
        try:
            listing = s.get(url, timeout=60).text
        except Exception as exc:
            print("list fail", oid, exc, flush=True)
            continue
        names = sorted(set(re.findall(r"sw\d{11}(?:uw2|um2|uw1)_sk\.img\.gz", listing)))
        for name in names:
            dest = OUT / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            for attempt in range(3):
                try:
                    r = s.get(url + name, timeout=120)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    break
                except Exception as exc:
                    print("retry", name, exc, flush=True)
                    time.sleep(10)
        print(oid, ym, names, flush=True)


if __name__ == "__main__":
    main()
