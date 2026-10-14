"""Sky positions for the website's hero map (site/src/data/sky.json), in Galactic coordinates.

Layers: objects with pages or active work, the dipole directions P1 tests against, and the 1,211 targets of the
Harvard-plate search (EXP-2026-Z, data/rn/targets.csv + scores.csv). Coordinates go through astropy's ICRS ->
Galactic transform (CLAUDE.md I.1). Rerun when a layer changes: python scripts/infra/export_sky.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "src" / "data" / "sky.json"
RN = ROOT / "data" / "rn"

# name, RA, Dec (deg, from data/rn/targets.csv), link, one-line note
OBJECTS = [
    ("T CrB", 239.875680, 25.920170, "/tcrb/", "recurrent nova; erupted 1866 and 1946, due again"),
    ("V347 Nor", 243.504583, -56.991111, "/v347nor/", "symbiotic nova; unrecorded eruption Feb 1943 (candidate)"),
    ("DE Cir", 229.469540, -61.954520, "/v347nor/#decir", "nova 2003; a 1929 brightening on the plates (exploratory)"),
    ("RS Oph", 267.554830, -6.707910, "/tcrb/#family", "symbiotic recurrent nova; 2006, 2021"),
    ("V3890 Sgr", 277.680370, -24.019150, "/tcrb/#family", "symbiotic recurrent nova; 1962, 1990, 2019"),
    ("V745 Sco", 268.842610, -33.249600, "/tcrb/#family", "symbiotic recurrent nova; 1937, 1989, 2014"),
]
# Galactic directions (l, b in deg)
DIRECTIONS = [
    ("CMB dipole", 264.021, 48.253, "Planck 2018: our motion against the CMB"),
    ("CatWISE quasar dipole", 238.2, 28.8, "Secrest+ 2021: the matter dipole that disagrees in amplitude"),
]


def gal(ra, dec):
    g = SkyCoord(np.atleast_1d(ra), np.atleast_1d(dec), unit="deg", frame="icrs").galactic
    return g.l.deg, g.b.deg


def main():
    out = {"objects": [], "directions": [], "search": []}
    for name, ra, dec, link, note in OBJECTS:
        l, b = gal(ra, dec)
        out["objects"].append({"name": name, "ra": ra, "dec": dec, "l": round(float(l[0]), 3),
                               "b": round(float(b[0]), 3), "link": link, "note": note})
    for name, l, b, note in DIRECTIONS:
        out["directions"].append({"name": name, "l": l, "b": b, "note": note})
    T = pd.read_csv(RN / "targets.csv")
    S = pd.read_csv(RN / "scores.csv")
    searched = set(S[S.role == "t"].tid)
    T = T[T.tid.isin(searched)]
    l, b = gal(T.ra.values, T.dec.values)
    yr = [("" if pd.isna(v) else str(int(v))) for v in T.year]
    out["search"] = [[round(float(x), 2), round(float(y), 2), n, c, y_]
                     for x, y, n, c, y_ in zip(l, b, T.name, T.cls.fillna(""), yr)]
    out["search_note"] = f"{len(T)} novae and symbiotic stars searched on the Harvard plates (EXP-2026-Z)"
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({len(out['search'])} search targets)")


if __name__ == "__main__":
    main()
