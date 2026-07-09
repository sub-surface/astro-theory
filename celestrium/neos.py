"""Near-Earth Object tracking via JPL CNEOS."""

from astropy.table import Table
import urllib.request
import json

from .net import DEFAULT_TIMEOUT

def fetch_close_approaches() -> Table | None:
    """Fetch recent and upcoming Near-Earth Object close approaches."""
    url = "https://ssd-api.jpl.nasa.gov/cad.api?dist-max=0.05"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Celestrium/1.0'})
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as response:
            data = json.loads(response.read())
            if "data" not in data or not data["data"]:
                return None
            fields = data["fields"]
            rows = data["data"]
            tab = Table(rows=rows, names=fields)
            return tab
    except Exception as e:
        raise RuntimeError(f"CNEOS fetch failed: {e}")
