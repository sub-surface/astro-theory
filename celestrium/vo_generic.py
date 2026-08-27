#!/usr/bin/env python
"""Generic IVOA TAP via pyvo — the unified client for all TAP services.

When an archive has no astroquery wrapper, point pyvo at its TAP endpoint and send
ADQL. Examples: CASDA (ASKAP/RACS), LOFAR/LoTSS, VizieR TAP, Gaia mirrors.
"""
from __future__ import annotations

from typing import Optional
import pyvo

# Canonical TAP Endpoints across astronomy archives
ENDPOINTS = {
    "gaia":       "https://gea.esac.esa.int/tap-server/tap",
    "irsa":       "https://irsa.ipac.caltech.edu/TAP",
    "euclid":     "https://eas.esac.esa.int/tap-server/tap",
    "heasarc":    "https://heasarc.gsfc.nasa.gov/xamin/vo/tap",
    "casda":      "https://casda.csiro.au/casda_vo_tools/tap",   # ASKAP / RACS
    "vizier":     "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "desi":       "https://datalab.noirlab.edu/tap",
}


class TAPClient:
    """A clean, reusable client for any IVOA TAP service."""

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self._service: Optional[pyvo.dal.TAPService] = None

    @property
    def service(self) -> pyvo.dal.TAPService:
        if self._service is None:
            self._service = pyvo.dal.TAPService(self.endpoint_url)
        return self._service

    def query(self, adql: str):
        """Submit an ADQL query and return an astropy Table."""
        return self.service.search(adql).to_table()

    def discover(self, substr: str = "") -> list[str]:
        """List table names matching an optional substring filter."""
        tables = [t.name for t in self.service.tables]
        if not substr:
            return tables
        sub = substr.lower()
        return [t for t in tables if sub in t.lower()]


def query(endpoint_url: str, adql: str):
    """Send ADQL to any TAP service; returns astropy Table via .to_table()."""
    return TAPClient(endpoint_url).query(adql)


def list_tables(endpoint_url: str, substr: str = ""):
    return TAPClient(endpoint_url).discover(substr)
