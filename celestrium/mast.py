#!/usr/bin/env python
"""NASA MAST (STScI) — via astroquery.mast.

JWST, HST, TESS, Kepler/K2, Pan-STARRS, GALEX. MAST is observation-metadata
centric (find observations -> get data products), not a single flat catalogue;
for TESS/Kepler light curves, `lightkurve` is the friendlier front end.
Docs: https://astroquery.readthedocs.io/en/latest/mast/mast.html
"""
from astroquery.mast import Observations


def query_region(ra_deg: float, dec_deg: float, radius_deg: float = 0.02):
    """Observations overlapping a cone, as an astropy Table."""
    return Observations.query_region(f"{ra_deg} {dec_deg}", radius=f"{radius_deg} deg")


def query_criteria(**criteria):
    """Metadata search, e.g. obs_collection='JWST', instrument_name='NIRCam'."""
    return Observations.query_criteria(**criteria)


def fetch_lightcurves(target_name: str):
    """Fetch TESS / Kepler / K2 lightcurve metadata from MAST."""
    try:
        obs_table = Observations.query_criteria(
            target_name=target_name,
            dataproduct_type=["timeseries"],
        )
        if obs_table is None or len(obs_table) == 0:
            return None
        return obs_table
    except Exception as e:
        raise RuntimeError(f"MAST lightcurve fetch failed: {e}")


if __name__ == "__main__":
    # JWST observations in a small field — connectivity smoke test.
    obs = query_criteria(
        obs_collection="JWST", dataproduct_type="image",
        s_ra=[150.0, 150.2], s_dec=[2.0, 2.2],
    )
    print(f"{len(obs)} JWST image observations")
    print(obs["obs_id", "instrument_name", "filters", "t_exptime"][:5])
    # products = Observations.get_product_list(obs[0])  # then filter + download
