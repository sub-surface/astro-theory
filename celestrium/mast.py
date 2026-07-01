#!/usr/bin/env python
"""NASA MAST (STScI) — via astroquery.mast.

JWST, HST, TESS, Kepler/K2, Pan-STARRS, GALEX. MAST is observation-metadata
centric (find observations -> get data products), not a single flat catalogue;
for TESS/Kepler light curves, `lightkurve` is the friendlier front end.
Docs: https://astroquery.readthedocs.io/en/latest/mast/mast.html
"""
from astroquery.mast import Observations


UV_OPTICAL_COLLECTIONS = ("GALEX", "HST", "HLA", "SWIFTUVOT")
LIGHTCURVE_COLLECTIONS = ("TESS", "Kepler", "K2")


def _filter_table(tab, *, collections=None, products=None):
    if tab is None or len(tab) == 0:
        return tab
    keep = []
    collections_l = {c.lower() for c in collections or ()}
    products_l = {p.lower() for p in products or ()}
    for row in tab:
        ok = True
        if collections_l and "obs_collection" in tab.colnames:
            ok = str(row["obs_collection"]).lower() in collections_l
        if ok and products_l and "dataproduct_type" in tab.colnames:
            ok = str(row["dataproduct_type"]).lower() in products_l
        keep.append(ok)
    return tab[keep]


def query_region(ra_deg: float, dec_deg: float, radius_deg: float = 0.02):
    """Observations overlapping a cone, as an astropy Table."""
    return Observations.query_region(f"{ra_deg} {dec_deg}", radius=f"{radius_deg} deg")


def query_criteria(**criteria):
    """Metadata search, e.g. obs_collection='JWST', instrument_name='NIRCam'."""
    return Observations.query_criteria(**criteria)


def fetch_uv_optical_observations(ra_deg: float, dec_deg: float,
                                  radius_deg: float = 0.02):
    """Return UV/optical MAST observation metadata around a coordinate."""
    try:
        obs_table = query_region(ra_deg, dec_deg, radius_deg=radius_deg)
        if obs_table is None or len(obs_table) == 0:
            return None
        return _filter_table(
            obs_table,
            collections=UV_OPTICAL_COLLECTIONS,
            products=("image", "spectrum"),
        )
    except Exception as e:
        raise RuntimeError(f"MAST UV/optical observation fetch failed: {e}")


def fetch_lightcurves(target_name: str, missions=LIGHTCURVE_COLLECTIONS):
    """Fetch TESS / Kepler / K2 lightcurve metadata from MAST."""
    try:
        obs_table = Observations.query_criteria(
            target_name=target_name,
            dataproduct_type="timeseries",
        )
        if obs_table is None or len(obs_table) == 0:
            return None
        return _filter_table(obs_table, collections=missions, products=("timeseries",))
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
