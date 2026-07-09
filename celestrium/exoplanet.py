#!/usr/bin/env python
"""NASA Exoplanet Archive via astroquery.ipac.nexsci.nasa_exoplanet_archive.

Queries the Planetary Systems Composite Parameters (pscomppars) table for
exoplanets and host star metadata.
"""
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from .net import set_timeout

set_timeout(NasaExoplanetArchive)

def query_target(target_name: str):
    """Fetch planetary system parameters for a target.

    Returns an astropy Table of planets in the system, or None if not found.
    """
    try:
        # pscomppars provides a single composite row per planet with the most precise values
        tab = NasaExoplanetArchive.query_object(target_name, table="pscomppars")
        if tab is not None and len(tab) > 0:
            return tab
        return None
    except Exception:
        # NasaExoplanetArchive raises exceptions for unknown objects
        return None

def predict_transits(target_name: str):
    """Predict the next transits for planets in the system.

    Returns a Table with predicted Next Transit UTC, Duration, and Depth.
    """
    import numpy as np
    from astropy.table import Table, Column
    from astropy.time import Time

    tab = query_target(target_name)
    if tab is None:
        return None

    out = Table()
    out["Planet"] = tab["pl_name"]

    now_jd = Time.now().jd

    next_transits = []
    durations = []
    depths = []

    for row in tab:
        try:
            val_t0 = row["pl_tranmid"]
            val_per = row["pl_orbper"]

            # Extract underlying value if it's a MaskedQuantity or Quantity
            if hasattr(val_t0, 'mask') and val_t0.mask:
                t0 = np.nan
            else:
                t0 = float(val_t0.value) if hasattr(val_t0, 'value') else float(val_t0)

            if hasattr(val_per, 'mask') and val_per.mask:
                per = np.nan
            else:
                per = float(val_per.value) if hasattr(val_per, 'value') else float(val_per)

            if np.isnan(t0) or np.isnan(per):
                next_transits.append("No ephemeris")
                durations.append("-")
                depths.append("-")
                continue

            # Compute number of orbits since T0
            n_orbits = np.ceil((now_jd - t0) / per)
            next_t = t0 + (n_orbits * per)

            # Convert JD to UTC string
            next_utc = Time(next_t, format='jd').iso
            next_transits.append(next_utc[:16]) # 'YYYY-MM-DD HH:MM'

            if "pl_trandur" in row.colnames:
                v_dur = row["pl_trandur"]
                dur = float(v_dur.value) if hasattr(v_dur, 'value') else float(v_dur)
            else:
                dur = np.nan
            durations.append(f"{dur:.2f} h" if not np.isnan(dur) else "Unknown")

            if "pl_trandep" in row.colnames:
                v_dep = row["pl_trandep"]
                dep = float(v_dep.value) if hasattr(v_dep, 'value') else float(v_dep)
            else:
                dep = np.nan
            depths.append(f"{dep:.2f} %" if not np.isnan(dep) else "Unknown")
        except Exception:
            next_transits.append("Error")
            durations.append("-")
            depths.append("-")

    out["Next_Transit_UTC"] = next_transits
    out["Duration"] = durations
    out["Depth"] = depths
    return out
