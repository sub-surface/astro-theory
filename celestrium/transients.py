"""Transient alerts via ALeRCE's public ZTF broker API (anonymous, no auth).

The practical "Rubin watch" scoped in docs/data-atlas.md: the full Rubin
Science Platform archive is data-rights-gated, but public broker alert APIs
are not — ALeRCE (and Fink) already ingest ZTF alerts today and are the
documented path for LSST/Rubin alerts once that stream goes public. This
module is the first real implementation of that scoping: a global "what's
new" feed and a target-aware cone search, both through ALeRCE.

Verified against the live OpenAPI spec at
https://api.alerce.online/ztf/v1/swagger.json (2026-07-10) — GET /objects/
takes ra/dec/radius(arcsec) for a conesearch and firstmjd/lastmjd (as
[min, max] ranges) for a date window, returning a paginated `items` list.
Fink was the plan's other candidate broker, but this sandbox could not
reach fink-portal.org to verify its schema live (persistent connection
timeout, not a DNS or server issue) — ALeRCE alone satisfies the "broker
alert API" requirement; Fink remains a natural second source to add once
its API can be verified.
"""
from __future__ import annotations

from typing import Optional

from astropy.table import Table
from astropy.time import Time

from .net import DEFAULT_TIMEOUT

API = "https://api.alerce.online/ztf/v1/objects/"

# ALeRCE's "Object List Item" fields we keep, renamed to celestrium's
# convention (main_id/ra/dec) so results compose with the rest of the
# instrument: crossmatch's RA/Dec column detection, the TUI's ambient
# sky-scatter scene, and the highlighted-row poster action all look for
# exactly these names.
_FIELDS = ("oid", "meanra", "meandec", "class", "classifier", "probability",
          "ndet", "firstmjd", "lastmjd")
_RENAME = {"oid": "main_id", "meanra": "ra", "meandec": "dec"}
_NUMERIC = {"meanra", "meandec", "probability", "ndet", "firstmjd", "lastmjd"}


def _get(params: dict, timeout: float = DEFAULT_TIMEOUT, fetcher=None) -> dict:
    """GET /objects/; `fetcher` is injectable (returns parsed JSON) for tests."""
    if fetcher is not None:
        return fetcher(params)
    import requests
    r = requests.get(API, params=params, timeout=timeout,
                     headers={"User-Agent": "Celestrium/1.0"})
    r.raise_for_status()
    return r.json()


def _clean(item: dict, field: str):
    value = item.get(field)
    if value is None:
        return float("nan") if field in _NUMERIC else ""
    return value


def _to_table(items: list) -> Table:
    if not items:
        return Table()
    rows = [tuple(_clean(it, f) for f in _FIELDS) for it in items]
    tab = Table(rows=rows, names=_FIELDS)
    for old, new in _RENAME.items():
        tab.rename_column(old, new)
    return tab


def fetch_latest_transients(days: float = 3.0, limit: int = 50,
                            fetcher=None) -> Optional[Table]:
    """Recent ALeRCE ZTF alerts, most-recent first — the global "what's new" feed."""
    now = Time.now().mjd
    try:
        data = _get({"firstmjd": [now - days, now], "page_size": limit,
                    "order_by": "lastmjd", "order_mode": "DESC"}, fetcher=fetcher)
    except Exception as e:
        raise RuntimeError(f"ALeRCE fetch failed: {e}")
    items = data.get("items") or []
    return _to_table(items) if items else None


def fetch_transients_near(ra: float, dec: float, radius_arcmin: float = 2.0,
                          days: Optional[float] = None,
                          fetcher=None) -> Optional[Table]:
    """ALeRCE alerts within radius_arcmin of a position — the target/field-scoped
    "show alerts near this object" action scoped in data-atlas.md."""
    params = {"ra": ra, "dec": dec, "radius": radius_arcmin * 60.0, "page_size": 50}
    if days is not None:
        now = Time.now().mjd
        params["firstmjd"] = [now - days, now]
    try:
        data = _get(params, fetcher=fetcher)
    except Exception as e:
        raise RuntimeError(f"ALeRCE cone search failed: {e}")
    items = data.get("items") or []
    return _to_table(items) if items else None


if __name__ == "__main__":
    tab = fetch_latest_transients(days=1.0, limit=5)
    print(tab if tab is not None else "no recent ALeRCE alerts")
