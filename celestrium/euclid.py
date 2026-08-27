#!/usr/bin/env python
"""ESA Euclid archive — TAP/ADQL via astroquery.esa.euclid.

Q1 (deep fields) and Q2 (Galactic Bulge Survey) are public; DR1 (~1900 deg2 wide)
lands 21 Oct 2026 — that's the one we're prepping for (see ../docs/research/euclid-dr1-prep.md).
For images/cutouts, run on ESA Datalabs instead of downloading locally.
Docs: https://astroquery.readthedocs.io/en/latest/esa/euclid/euclid.html
Tutorials: https://github.com/ESA-Datalabs/Euclid-Q1
"""
from . import vo_generic

client = vo_generic.TAPClient(vo_generic.ENDPOINTS["euclid"])


def query(adql: str, async_: bool = False):
    """Run ADQL against ESA Euclid TAP."""
    try:
        from astroquery.esa.euclid import Euclid
        job = Euclid.launch_job_async(adql) if async_ else Euclid.launch_job(adql)
        return job.get_results()
    except Exception:
        return client.query(adql)


def discover():
    """List available tables (schema names drift between Q1/Q2/DR1 — check live)."""
    try:
        from astroquery.esa.euclid import Euclid
        return [t.name for t in Euclid.load_tables(only_names=True)]
    except Exception:
        return client.discover()


if __name__ == "__main__":
    # MER catalogue smoke test. Confirm table/column names against discover()
    # before trusting them — they change across releases.
    tab = query(
        "SELECT TOP 5 object_id, right_ascension, declination "
        "FROM catalogue.mer_catalogue"
    )
    print(tab)
