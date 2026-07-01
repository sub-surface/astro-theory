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
