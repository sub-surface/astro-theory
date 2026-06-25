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
