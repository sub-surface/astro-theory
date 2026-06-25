#!/usr/bin/env python
"""ESA Gaia archive — TAP/ADQL via astroquery.gaia.

Anonymous access works; query results are auto-purged after 72h (fine for us —
we keep the *output* locally). DR3 is live; DR4 expected ~late 2026.
Docs: https://astroquery.readthedocs.io/en/latest/gaia/gaia.html
"""
from astroquery.gaia import Gaia


def query(adql: str):
    """Run ADQL, return an astropy Table. Async handles large results cleanly."""
    job = Gaia.launch_job_async(adql)
    return job.get_results()


def discover():
    """List available tables (run once when exploring a new analysis)."""
    return [t.name for t in Gaia.load_tables(only_names=True)]


if __name__ == "__main__":
    # 5 bright sources with parallaxes — connectivity smoke test.
    tab = query(
        "SELECT TOP 5 source_id, ra, dec, parallax, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source "
        "WHERE phot_g_mean_mag < 8 AND parallax > 10 "
        "ORDER BY phot_g_mean_mag"
    )
    print(tab)
    # Cone-search form (uncomment): CONTAINS for a position-based pull.
    # tab = query(
    #     "SELECT source_id, ra, dec FROM gaiadr3.gaia_source "
    #     "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
    #     "CIRCLE('ICRS', 266.4, -29.0, 0.1))"
    # )
