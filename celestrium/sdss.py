"""SDSS spectra, photometry, and quick-look optical image helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.visualization import simple_norm
from astroquery.sdss import SDSS

from . import spectra
from .net import set_timeout

set_timeout(SDSS)

OUT = Path(__file__).resolve().parent.parent / "data" / "sdss"


def _coord(ra_deg: float, dec_deg: float) -> SkyCoord:
    return SkyCoord(ra_deg, dec_deg, unit=u.deg)


def _table_from_hdul(hdul) -> Optional[Table]:
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is None:
            continue
        try:
            tab = Table(data)
        except Exception:
            continue
        if tab.colnames:
            return tab
    return None


def query_region(ra_deg: float, dec_deg: float, radius_arcsec: float = 3.0,
                 *, spectro: bool = False):
    """Return SDSS rows around a coordinate."""
    return SDSS.query_region(
        _coord(ra_deg, dec_deg),
        radius=radius_arcsec * u.arcsec,
        spectro=spectro,
    )


def fetch_photometry(ra_deg: float, dec_deg: float, radius_arcsec: float = 3.0):
    """Return SDSS photo-object rows around a coordinate.

    `query_photoobj` is keyed by run/rerun/camcol/field (SDSS imaging-run
    identifiers), not a coordinate + radius cone search — `query_region`
    (spectro=False) is the actual coordinate-based photo-object lookup.
    """
    return SDSS.query_region(
        _coord(ra_deg, dec_deg),
        radius=radius_arcsec * u.arcsec,
        spectro=False,
    )


def fetch_spectrum(target_name: str, ra_deg: float, dec_deg: float,
                   radius_arcsec: float = 3.0,
                   out: Optional[Path] = None) -> Optional[spectra.SpectrumResult]:
    """Fetch and render the first SDSS spectrum around a coordinate."""
    try:
        hduls = SDSS.get_spectra(
            coordinates=_coord(ra_deg, dec_deg),
            radius=radius_arcsec * u.arcsec,
        )
    except Exception:
        return None
    for hdul in hduls or []:
        tab = _table_from_hdul(hdul)
        if tab is None:
            continue
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            out_path = out or OUT / f"spectrum_{target_name.replace(' ', '_')}.png"
            return spectra.render_spectrum_table(
                tab,
                target=target_name,
                source="SDSS",
                out=out_path,
                provenance={
                    "adapter": "SDSS",
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "radius_arcsec": radius_arcsec,
                },
            )
        except ValueError:
            continue
    return None


def fetch_optical_image(ra_deg: float, dec_deg: float, fov_arcmin: float = 3.0,
                        out: Optional[Path] = None) -> Optional[Path]:
    """Fetch and render the first SDSS optical image around a coordinate."""
    try:
        hduls = SDSS.get_images(
            coordinates=_coord(ra_deg, dec_deg),
            radius=fov_arcmin * u.arcmin,
        )
    except Exception:
        return None
    for hdul in hduls or []:
        data = getattr(hdul[0], "data", None) if len(hdul) else None
        if data is None:
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        out_path = Path(out) if out else OUT / f"image_{ra_deg:.4f}_{dec_deg:+.4f}.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, origin="lower", cmap="gray",
                  norm=simple_norm(data, "asinh", percent=99.5))
        ax.set_title(f"SDSS optical  RA={ra_deg:.5f} Dec={dec_deg:+.5f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path
    return None
