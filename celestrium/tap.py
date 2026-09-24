"""Unified TAP (Table Access Protocol) client for all astronomical archives.

Replaces fragmented per-archive shims with a single robust interface over PyVO
and specialized astroquery drivers with transparent fallback.
"""
from __future__ import annotations

from typing import Optional, List
import pyvo

ENDPOINTS = {
    "gaia":       "https://gea.esac.esa.int/tap-server/tap",
    "irsa":       "https://irsa.ipac.caltech.edu/TAP",
    "euclid":     "https://eas.esac.esa.int/tap-server/tap",
    "heasarc":    "https://heasarc.gsfc.nasa.gov/xamin/vo/tap",
    "casda":      "https://casda.csiro.au/casda_vo_tools/tap",
    "vizier":     "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "desi":       "https://datalab.noirlab.edu/tap",
}

AVAILABILITY = {
    "gaia":       "https://gea.esac.esa.int/tap-server/tap/availability?archive=gaia",
    "irsa":       "https://irsa.ipac.caltech.edu/TAP/availability",
    "euclid":     "https://eas.esac.esa.int/tap-server/tap/availability",
    "heasarc":    "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/availability",
    "casda":      "https://casda.csiro.au/casda_vo_tools/tap/availability",
    "vizier":     "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/availability",
    "desi":       "https://datalab.noirlab.edu/tap/availability",
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

    def discover(self, substr: str = "") -> List[str]:
        """List table names matching an optional substring filter."""
        tables = [t.name for t in self.service.tables]
        if not substr:
            return tables
        sub = substr.lower()
        return [t for t in tables if sub in t.lower()]


def _query_driver(archive: str, adql: str):
    """Attempt specialized astroquery driver if available, else raise."""
    if archive == "gaia":
        from astroquery.gaia import Gaia
        job = Gaia.launch_job_async(adql)
        return job.get_results()
    elif archive == "irsa":
        from astroquery.ipac.irsa import Irsa
        return Irsa.query_tap(adql).to_table()
    elif archive == "euclid":
        from astroquery.esa.euclid import Euclid
        job = Euclid.launch_job(adql)
        return job.get_results()
    elif archive == "heasarc":
        from astroquery.heasarc import Heasarc
        # HEASARC TAP usually via generic VO
        raise NotImplementedError
    raise NotImplementedError


def query(archive_or_url: str, adql: str):
    """Send ADQL to any registered archive or direct TAP URL.
    
    Tries driver first where appropriate, seamlessly falling back to generic TAP.
    """
    key = archive_or_url.lower().replace("-tap", "")
    url = ENDPOINTS.get(key, archive_or_url)
    client = TAPClient(url)

    if key in ("gaia", "irsa", "euclid"):
        try:
            return _query_driver(key, adql)
        except Exception:
            return client.query(adql)

    return client.query(adql)


def discover(archive_or_url: str, substr: str = "") -> List[str]:
    """List available tables for an archive or direct TAP URL."""
    key = archive_or_url.lower().replace("-tap", "")
    url = ENDPOINTS.get(key, archive_or_url)
    client = TAPClient(url)

    if key == "gaia":
        try:
            from astroquery.gaia import Gaia
            return [t.name for t in Gaia.load_tables(only_names=True)]
        except Exception:
            return client.discover(substr)
    elif key == "irsa":
        try:
            from astroquery.ipac.irsa import Irsa
            cats = Irsa.list_catalogs()
            names = cats.keys() if hasattr(cats, "keys") else cats
            return [n for n in names if substr.lower() in str(n).lower()]
        except Exception:
            return client.discover(substr)

    return client.discover(substr)
