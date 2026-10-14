"""EXP-2026-Z: build the hidden-recurrent-nova target list (data/rn/targets.csv).

Sources (VizieR): Schaefer 2022 Galactic novae (J/MNRAS/517/6150, 402 novae with eruption year, V_peak, E(B-V));
VSX types N*/NR* and ZAND* (B/vsx/vsx); Akras+ 2019 symbiotic census (J/ApJS/240/21).
Extragalactic entries are dropped by position (M31, M33, LMC, SMC). Duplicates within 5" are merged, keeping
the Schaefer row's eruption metadata. Known recurrent-nova eruptions (Schaefer 2010 and later) are attached so
the search can mask them.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "rn" / "targets.csv"

# Known eruptions of the Galactic RNe (Schaefer 2010 ApJS 187, 275; plus V745 Sco 1897 (Schaefer 2024 CBET 5368),
# KT Eri 2009, V3890 Sgr 2019, U Sco 2010/2022, RS Oph 2006/2021, V745 Sco 2014, T CrB 1866/1946).
KNOWN_RN = {
    "T CrB": [1866.36, 1946.11],
    "RS Oph": [1898.4, 1933.6, 1958.6, 1967.8, 1985.1, 2006.1, 2021.6],
    "U Sco": [1863.4, 1906.4, 1917.4, 1936.5, 1945.5, 1969.5, 1979.5, 1987.4, 1999.2, 2010.1, 2022.4],
    "V745 Sco": [1897.5, 1937.3, 1989.6, 2014.1],
    "V3890 Sgr": [1962.4, 1990.7, 2019.7],
    "T Pyx": [1890.9, 1902.4, 1920.4, 1944.9, 1966.9, 2011.3],
    "CI Aql": [1917.4, 1941.4, 2000.3],
    "IM Nor": [1920.3, 2002.0],
    "V394 CrA": [1949.6, 1987.6],
    "V2487 Oph": [1900.4, 1998.5],
    "KT Eri": [2009.9],
}
EXGAL = [(10.685, 41.269, 3.0), (23.462, 30.660, 1.5), (80.894, -69.756, 10.0), (13.187, -72.829, 5.0)]


def _not_exgal(ra, dec):
    c = SkyCoord(ra * u.deg, dec * u.deg)
    keep = np.ones(len(ra), bool)
    for r0, d0, rad in EXGAL:
        keep &= c.separation(SkyCoord(r0 * u.deg, d0 * u.deg)).deg > rad
    return keep


def _year_from_name(s):
    s = str(s)
    m = re.match(r"(?:Gaia|ZTF|PGIR|ASASSN-|ATLAS|MASTER OT J|PNV J|TCP J)?(?:Gaia|ZTF|PGIR|ASASSN-|ATLAS)(\d\d)", s)
    if m:
        return 2000.0 + float(m.group(1)) + 0.5
    m = re.search(r"(18[5-9]\d|19\d\d|20[0-2]\d)", s)
    return float(m.group(1)) + 0.5 if m else np.nan


def main():
    V = Vizier(columns=["*", "_RA", "_DE"], row_limit=-1, timeout=300)
    s6 = V.get_catalogs("J/MNRAS/517/6150/table6")[0].to_pandas()
    s3 = V.get_catalogs("J/MNRAS/517/6150/table3")[0].to_pandas()[["Nova", "E(B-V)"]]
    sch = s6.merge(s3, on="Nova", how="left")
    sch = pd.DataFrame({"name": sch["Nova"].str.strip(), "ra": sch["_RA"], "dec": sch["_DE"], "cls": "nova",
                        "year": sch["Year"].astype(float) + 0.5, "vpeak": sch["Vpeak"], "ebv": sch["E(B-V)"],
                        "lc": sch["LC"], "porb_d": sch["P"], "src": "Schaefer22"})

    vsx_rows = []
    for ty, cls in (("N*", "nova"), ("NR*", "nova"), ("ZAND*", "symbiotic")):
        t = Vizier(columns=["Name", "Type", "max", "min", "RAJ2000", "DEJ2000", "Epoch", "Period"], row_limit=-1,
                   timeout=300).query_constraints(catalog="B/vsx/vsx", Type=ty)[0].to_pandas()
        t = t[~t["Type"].str.startswith("NL")] if cls == "nova" else t  # novalikes are not novae
        ep = pd.to_numeric(t["Epoch"], errors="coerce")
        yr = np.where(ep > 2.3e6, 2000 + (ep - 2451545) / 365.25, [_year_from_name(n) for n in t["Name"]])
        vsx_rows.append(pd.DataFrame({"name": t["Name"].str.strip(), "ra": t["RAJ2000"], "dec": t["DEJ2000"],
                                      "cls": np.where(t["Type"].str.startswith("NR"), "rn", cls),
                                      "year": yr if cls == "nova" else np.nan, "vpeak": np.nan, "ebv": np.nan,
                                      "lc": "", "porb_d": t["Period"], "src": "VSX:" + t["Type"].astype(str)}))
    ak = V.get_catalogs("J/ApJS/240/21/systars")[0].to_pandas()
    akr = SkyCoord(ak["RAJ2000"].astype(str), ak["DEJ2000"].astype(str), unit=(u.hourangle, u.deg)) \
        if ak["RAJ2000"].dtype == object else SkyCoord(ak["RAJ2000"] * u.deg, ak["DEJ2000"] * u.deg)
    ak = pd.DataFrame({"name": ak["Name"].str.strip(), "ra": akr.ra.deg, "dec": akr.dec.deg, "cls": "symbiotic",
                       "year": np.nan, "vpeak": np.nan, "ebv": np.nan, "lc": "", "porb_d": np.nan,
                       "src": "Akras19"})

    allt = pd.concat([sch] + vsx_rows + [ak], ignore_index=True)
    allt = allt[np.isfinite(allt.ra) & np.isfinite(allt.dec)]
    allt = allt[_not_exgal(allt.ra.values, allt.dec.values)].reset_index(drop=True)

    # merge duplicates within 5" (priority: Schaefer22 > VSX > Akras)
    pri = allt["src"].map(lambda s: 0 if s == "Schaefer22" else (1 if s.startswith("VSX") else 2))
    allt = allt.assign(pri=pri).sort_values("pri").reset_index(drop=True)
    c = SkyCoord(allt.ra.values * u.deg, allt.dec.values * u.deg)
    idx, d2d, _ = c.match_to_catalog_sky(c, nthneighbor=1)
    keep = np.ones(len(allt), bool)
    groups = {}
    i1, i2, _, _ = c.search_around_sky(c, 5 * u.arcsec)
    for a, b in zip(i1, i2):
        if a < b:
            groups.setdefault(a, []).append(b)
    srcs = allt["src"].astype(str).values.copy()
    cls = allt["cls"].values.copy()
    for a in range(len(allt)):
        if not keep[a]:
            continue
        for b in groups.get(a, []):
            if keep[b]:
                keep[b] = False
                srcs[a] = srcs[a] + "|" + srcs[b]
                if cls[b] == "rn" or (cls[b] == "symbiotic" and cls[a] == "nova"):
                    cls[a] = "rn" if cls[b] == "rn" else "nova+symbiotic"
    allt["src"], allt["cls"] = srcs, cls
    allt = allt[keep].drop(columns="pri").reset_index(drop=True)

    # attach known eruption epochs; mark the known RNe
    def known(row):
        eps = [row.year] if np.isfinite(row.year) else []
        for k, v in KNOWN_RN.items():
            if re.sub(r"\s+", " ", str(row["name"])).upper() == k.upper():
                return ";".join(f"{x:.2f}" for x in v)
        return ";".join(f"{x:.2f}" for x in eps)
    allt["known_eruptions"] = allt.apply(known, axis=1)
    allt.loc[allt["name"].str.upper().isin([k.upper() for k in KNOWN_RN]), "cls"] = "rn"
    allt.insert(0, "tid", [f"T{i:04d}" for i in range(len(allt))])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allt.to_csv(OUT, index=False)
    print(allt["cls"].value_counts().to_string())
    print(f"wrote {OUT} ({len(allt)} targets); known RNe found:",
          sorted(allt.loc[allt.cls == "rn", "name"].tolist())[:20])


if __name__ == "__main__":
    main()
