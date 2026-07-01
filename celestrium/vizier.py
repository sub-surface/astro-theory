#!/usr/bin/env python
"""VizieR (CDS) — any *published* catalogue via astroquery.vizier.

The universal fallback: thousands of catalogues from papers, each with a CDS id.
NVSS = 'VIII/65/nvss' (radio leg of the dipole anomaly, already in our archive);
search find_catalogs() for others. Great when a result table isn't in a big archive.
Docs: https://astroquery.readthedocs.io/en/latest/vizier/vizier.html
"""
from astroquery.vizier import Vizier


def find(keyword: str):
    """Map keyword -> {CDS id: description}; pick the id you want."""
    return {k: v.description for k, v in Vizier.find_catalogs(keyword).items()}


def query(catalog_id: str, columns=("*",), row_limit=20, **constraints):
    """Pull rows from one CDS catalogue. constraints e.g. S1_4='>1000' (mJy)."""
    v = Vizier(columns=list(columns), row_limit=row_limit)
    res = v.query_constraints(catalog=catalog_id, **constraints)
    return res[0] if res else None


def cone_pull(ra_deg: float, dec_deg: float, radius_arcmin: float = 1.0, catalog: str = "I/355/gaiadr3"):
    """Query a specific catalogue around a coordinate."""
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    pos = SkyCoord(ra_deg, dec_deg, unit=u.deg)
    v = Vizier(columns=["**"], row_limit=500)
    tables = v.query_region(pos, radius=radius_arcmin * u.arcmin, catalog=catalog)
    return tables[0] if tables else None


if __name__ == "__main__":
    # NVSS bright sources — connectivity smoke test.
    tab = query("VIII/65/nvss", columns=["RAJ2000", "DEJ2000", "S1.4"],
                row_limit=5, **{"S1.4": ">2000"})
    print(tab)
