import numpy as np
from astropy.table import Table

from celestrium import sdss


def test_fetch_spectrum_renders_first_sdss_table(monkeypatch, tmp_path):
    class Hdu:
        def __init__(self, data):
            self.data = data

    hdul = [Hdu(None), Hdu(Table({"loglam": [1.0, 2.0], "flux": [3.0, 4.0]}))]
    out = tmp_path / "spec.png"

    monkeypatch.setattr(sdss.SDSS, "get_spectra", lambda *a, **k: [hdul])

    result = sdss.fetch_spectrum("3C 273", 187.2, 2.0, out=out)

    assert result is not None
    assert result.source == "SDSS"
    assert result.path == out
    assert out.exists()


def test_fetch_optical_image_renders_first_frame(monkeypatch, tmp_path):
    class Hdu:
        data = np.arange(16, dtype=float).reshape(4, 4)

    out = tmp_path / "sdss.png"
    monkeypatch.setattr(sdss.SDSS, "get_images", lambda *a, **k: [[Hdu()]])

    path = sdss.fetch_optical_image(187.2, 2.0, out=out)

    assert path == out
    assert out.exists()


def test_fetch_photometry_calls_query_region_not_query_photoobj(monkeypatch):
    """query_photoobj is keyed by run/rerun/camcol/field, not a coordinate
    cone search — it doesn't accept coordinates/radius kwargs at all and
    raised a TypeError in production (found via a live sweep smoke test).
    query_region is the actual coordinate-based photo-object lookup."""
    seen = {}

    def fake_query_region(coordinates, *, radius=None, spectro=None, **kw):
        seen.update(radius=radius, spectro=spectro)
        return Table({"objid": [1], "ra": [187.2], "dec": [2.0]})

    monkeypatch.setattr(sdss.SDSS, "query_region", fake_query_region)
    monkeypatch.setattr(sdss.SDSS, "query_photoobj",
                        lambda *a, **k: (_ for _ in ()).throw(
                            TypeError("query_photoobj must not be called")))

    tab = sdss.fetch_photometry(187.2, 2.0, radius_arcsec=5.0)

    assert len(tab) == 1
    assert seen["spectro"] is False
    from astropy import units as u
    assert seen["radius"] == 5.0 * u.arcsec
