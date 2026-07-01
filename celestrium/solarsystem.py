"""Executors for Solar System objects (ephemerides, planetary views)."""

from astropy.table import Table

def fetch_ephemeris(target_name: str) -> Table | None:
    """Fetch a 30-day ephemeris track for the target using JPL Horizons."""
    from astroquery.jplhorizons import Horizons
    from celestrium.resolvers import _SOLAR_SYSTEM_IDS
    import datetime

    target_id = _SOLAR_SYSTEM_IDS.get(target_name.lower().strip(), target_name.strip())

    start = datetime.datetime.utcnow()
    stop = start + datetime.timedelta(days=30)

    try:
        obj = Horizons(id=target_id, epochs={'start': start.strftime("%Y-%m-%d"),
                                             'stop': stop.strftime("%Y-%m-%d"),
                                             'step': '1d'})
        eph = obj.ephemerides()
        if eph is None or len(eph) == 0:
            return None
        return eph
    except Exception as e:
        raise RuntimeError(f"Horizons ephemeris fetch failed: {e}")
