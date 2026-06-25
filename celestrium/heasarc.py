#!/usr/bin/env python
"""NASA HEASARC — high-energy archive via astroquery.heasarc.

X-ray / gamma-ray: Chandra, XMM-Newton, NuSTAR, Swift, Fermi, eROSITA-DR1
(western Galactic hemisphere), ROSAT. Useful for AGN/cluster cross-matches and
an X-ray source-count dipole as an independent isotropy front.
API was rewritten recently: query_region / query_tap return astropy Tables directly.
Docs: https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html
"""
from astroquery.heasarc import Heasarc
from astropy.coordinates import SkyCoord
from astropy import units as u


def query_region(ra_deg, dec_deg, catalog="chanmaster", radius_deg=1.0):
    pos = SkyCoord(ra_deg, dec_deg, unit=u.deg)
    return Heasarc.query_region(pos, catalog=catalog, radius=radius_deg * u.deg)


def query(adql: str):
    """Custom ADQL across HEASARC tables -> astropy Table."""
    return Heasarc.query_tap(adql).to_table()


def discover(substr: str = ""):
    cats = Heasarc.list_catalogs()
    return cats if not substr else cats[[substr.lower() in str(n).lower()
                                         for n in cats["name"]]]


if __name__ == "__main__":
    # Chandra master observations near the Galactic Center — smoke test.
    tab = query_region(266.4, -29.0, catalog="chanmaster", radius_deg=0.5)
    print(tab[:5])
