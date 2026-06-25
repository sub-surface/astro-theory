#!/usr/bin/env python
"""CDS X-Match — cross-match our table against any VizieR/SIMBAD catalogue.

The cross-catalogue-audit primitive: take a local table of positions and match
it to a big catalogue by sky separation, server-side. This is how we ask "do
CatWISE / Quaia / Euclid agree on the same sources under a common match radius."
See ../docs/toolbox.md S3.
Docs: https://astroquery.readthedocs.io/en/latest/xmatch/xmatch.html
"""
from astroquery.xmatch import XMatch
from astropy.table import Table
from astropy import units as u


def match(local: Table, cat2="vizier:VIII/65/nvss",
          ra="ra", dec="dec", radius_arcsec=5.0):
    """Match a local table (with RA/Dec deg columns) to a VizieR catalogue."""
    return XMatch.query(cat1=local, cat2=cat2,
                        max_distance=radius_arcsec * u.arcsec,
                        colRA1=ra, colDec1=dec)


if __name__ == "__main__":
    # Two positions near bright NVSS sources -> match against NVSS.
    local = Table({"id": [1, 2],
                   "ra": [0.8417, 1.5578],
                   "dec": [-17.4532, -6.3931]})
    out = match(local, cat2="vizier:VIII/65/nvss", radius_arcsec=30.0)
    print(out["id", "angDist", "RAJ2000", "DEJ2000", "S1.4"])
