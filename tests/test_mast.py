from astropy.table import Table

from celestrium import mast


def test_fetch_uv_optical_observations_filters_mast_collections(monkeypatch):
    tab = Table({
        "obs_collection": ["GALEX", "HST", "TESS"],
        "dataproduct_type": ["image", "spectrum", "timeseries"],
        "obs_id": ["g", "h", "t"],
    })
    monkeypatch.setattr(mast.Observations, "query_region", lambda *a, **k: tab)

    out = mast.fetch_uv_optical_observations(10.0, 20.0)

    assert list(out["obs_id"]) == ["g", "h"]


def test_fetch_lightcurves_filters_tess_kepler_missions(monkeypatch):
    tab = Table({
        "obs_collection": ["TESS", "Kepler", "HST"],
        "dataproduct_type": ["timeseries", "timeseries", "image"],
        "obs_id": ["t", "k", "h"],
    })
    monkeypatch.setattr(mast.Observations, "query_criteria", lambda **kw: tab)

    out = mast.fetch_lightcurves("TOI-700")

    assert list(out["obs_id"]) == ["t", "k"]
