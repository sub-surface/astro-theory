#!/usr/bin/env python
"""Generic IVOA TAP via pyvo — the escape hatch for any VO service.

When an archive has no astroquery wrapper, point pyvo at its TAP endpoint and send
ADQL. Examples: CASDA (ASKAP/RACS), LOFAR/LoTSS, VizieR TAP, Gaia mirrors.
The radio dipole legs (RACS, LoTSS) live here.
Docs: https://pyvo.readthedocs.io/en/latest/dal/
"""
import pyvo

# A few useful endpoints (verify URLs — services move):
ENDPOINTS = {
    "casda":  "https://casda.csiro.au/casda_vo_tools/tap",   # ASKAP / RACS
    "vizier": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "gaia":   "https://gea.esac.esa.int/tap-server/tap",
}


def query(endpoint_url: str, adql: str):
    """Send ADQL to any TAP service; returns astropy Table via .to_table()."""
    service = pyvo.dal.TAPService(endpoint_url)
    return service.search(adql).to_table()


def list_tables(endpoint_url: str, substr: str = ""):
    service = pyvo.dal.TAPService(endpoint_url)
    return [t.name for t in service.tables if substr.lower() in t.name.lower()]


if __name__ == "__main__":
    # VizieR-over-TAP smoke test: 5 NVSS rows via generic TAP.
    tab = query(
        ENDPOINTS["vizier"],
        'SELECT TOP 5 "RAJ2000", "DEJ2000", "S1.4" FROM "VIII/65/nvss" '
        'WHERE "S1.4" > 2000',
    )
    print(tab)
    # RACS via CASDA (uncomment; confirm current table name, e.g. 'AS110.racs_...'):
    # print(list_tables(ENDPOINTS["casda"], substr="racs"))
