"""Artificial Satellite tracking via Celestrak."""

from astropy.table import Table
import urllib.request

from .net import DEFAULT_TIMEOUT

def fetch_visible_satellites() -> Table | None:
    """Fetch TLEs for visible artificial satellites (ISS, etc)."""
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Celestrium/1.0'})
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
            if len(lines) < 3:
                return None

            names = []
            tle_line1 = []
            tle_line2 = []

            for i in range(0, len(lines), 3):
                if i + 2 < len(lines):
                    names.append(lines[i].strip())
                    tle_line1.append(lines[i+1].strip())
                    tle_line2.append(lines[i+2].strip())

            tab = Table()
            tab["Satellite"] = names
            tab["TLE_1"] = tle_line1
            tab["TLE_2"] = tle_line2
            return tab
    except Exception as e:
        raise RuntimeError(f"Celestrak TLE fetch failed: {e}")
