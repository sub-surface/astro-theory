import pytest
from astropy.table import Table

from celestrium import spectra


def test_render_spectrum_table_writes_artifact(tmp_path):
    tab = Table({"wavelength": [4000, 5000, 6000], "flux": [1.0, 3.0, 2.0]})
    result = spectra.render_spectrum_table(
        tab, target="Demo", source="fake", out=tmp_path / "demo.png")
    assert result.kind == "spectrum"
    assert result.path.exists()
    assert result.source == "fake"
    assert result.columns == ("wavelength", "flux")


def test_render_spectrum_table_rejects_missing_columns(tmp_path):
    tab = Table({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(ValueError, match="spectral columns"):
        spectra.render_spectrum_table(
            tab, target="Bad", source="fake", out=tmp_path / "bad.png")


def test_fetch_ned_spectrum_returns_none_when_no_spectra(monkeypatch):
    class FakeNed:
        @staticmethod
        def get_spectra(target):
            return []

    monkeypatch.setattr(spectra, "_ned_client", lambda: FakeNed)
    assert spectra.fetch_ned_spectrum("Nope") is None


def test_fetch_ned_spectrum_renders_first_table(monkeypatch, tmp_path):
    from astropy.io import fits

    table = Table({"wavelength": [1, 2, 3], "flux": [2.0, 3.0, 4.0]})
    hdu = fits.BinTableHDU(table)

    class FakeNed:
        @staticmethod
        def get_spectra(target):
            return [[hdu]]

    monkeypatch.setattr(spectra, "_ned_client", lambda: FakeNed)
    result = spectra.fetch_ned_spectrum("3C 273", out=tmp_path / "ned.png")
    assert result is not None
    assert result.path.exists()
    assert result.source == "NED"
