#!/usr/bin/env python
"""NASA/IPAC IRSA — TAP/ADQL via astroquery.ipac.irsa.

Infrared + survey catalogues: AllWISE, CatWISE2020, NEOWISE, 2MASS, Spitzer, and
SPHEREx (all-sky NIR spectro-photometry, releasing through 2026 — our all-sky
dipole sibling probe). CatWISE2020 already underpins the G dipole work.
Docs: https://astroquery.readthedocs.io/en/latest/ipac/irsa/irsa.html
"""
from . import vo_generic

client = vo_generic.TAPClient(vo_generic.ENDPOINTS["irsa"])


def query(adql: str):
    """Run ADQL against NASA/IPAC IRSA TAP."""
    try:
        from astroquery.ipac.irsa import Irsa
        return Irsa.query_tap(adql).to_table()
    except Exception:
        return client.query(adql)


def discover(substr: str = ""):
    """List catalogue (table) names, optionally filtered by substring."""
    try:
        from astroquery.ipac.irsa import Irsa
        cats = Irsa.list_catalogs()
        names = cats.keys() if hasattr(cats, "keys") else cats
        return [n for n in names if substr.lower() in str(n).lower()]
    except Exception:
        return client.discover(substr)


if __name__ == "__main__":
    # AllWISE point sources near a position — connectivity smoke test.
    # (CatWISE2020 table is 'catwise_2020'; confirm via discover('wise').)
    tab = query(
        "SELECT TOP 5 designation, ra, dec, w1mpro, w2mpro "
        "FROM allwise_p3as_psd "
        "WHERE CONTAINS(POINT('ICRS', ra, dec), "
        "CIRCLE('ICRS', 150.0, 2.0, 0.05))=1"
    )
    print(tab)
