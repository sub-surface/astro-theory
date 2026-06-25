#!/usr/bin/env python
"""NOIRLab Astro Data Lab — DESI / Legacy Surveys / DES / SDSS via server-side SQL.

Run queries *next to* the data (no pixel download). DESI DR1 = 18.7M spectra
(redshifts, QSO/ELG/LRG targets, peculiar-velocity survey -> bulk-flow sibling
probe); Legacy Surveys = deep optical imaging catalogues (sweeps/tractor).
Install: pip install astro-datalab
Docs: https://datalab.noirlab.edu/  ·  https://datalab.noirlab.edu/data/desi
"""
from dl import queryClient as qc


def query(sql: str, fmt: str = "pandas"):
    """Run ADQL/SQL server-side. fmt='pandas' -> DataFrame, 'table' -> astropy."""
    return qc.query(sql=sql, fmt=fmt)


def schema(table: str = ""):
    """Browse available schemas/tables/columns (names differ per release)."""
    return qc.schema(table) if table else qc.schema()


if __name__ == "__main__":
    # DESI DR1 redshifts smoke test. Confirm table name via schema('desi_dr1').
    df = query(
        "SELECT ra, dec, z, spectype "
        "FROM desi_dr1.zpix "
        "WHERE z BETWEEN 1.0 AND 2.0 AND spectype = 'QSO' "
        "LIMIT 5"
    )
    print(df)
    # Legacy Surveys imaging example:
    #   "SELECT ra, dec, mag_g, mag_r FROM ls_dr10.tractor LIMIT 5"
