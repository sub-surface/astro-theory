#!/usr/bin/env python
"""Multi-wavelength image cutouts from one set of coordinates — for fun & sanity.

Give a position (RA, Dec in degrees) and get back a panel of the same patch of
sky across the spectrum: UV -> optical -> near-IR -> mid-IR -> radio, each from a
different public survey, all co-registered by SkyView. Plus an optional true-ish
colour image via CDS HiPS. A quick "what's actually here?" before any analysis.

    python access/cutouts.py 213.6905918 -12.5801013

Backends: astroquery.skyview (FITS per survey), astroquery.hips2fits (colour).
See ../toolbox.md S4. Outputs land in data/cutouts/ (git-ignored).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
from astropy.visualization import simple_norm
from astroquery.skyview import SkyView
from astroquery.hips2fits import hips2fits

OUT = Path(__file__).resolve().parent.parent / "data" / "cutouts"

# Survey per wavelength regime. SkyView identifiers must be exact.
BANDS = {
    "GALEX Near UV": "UV",
    "DSS2 Red": "optical",
    "2MASS-J": "near-IR",
    "WISE 3.4": "mid-IR",
    "NVSS": "radio (1.4 GHz)",
}


def fetch(ra, dec, fov_arcmin=5.0, surveys=tuple(BANDS)):
    """Return {survey: HDUList} of co-registered cutouts (skips any that fail)."""
    pos = SkyCoord(ra, dec, unit=u.deg)
    out = {}
    for s in surveys:
        try:
            hduls = SkyView.get_images(position=pos, survey=[s],
                                       radius=fov_arcmin * u.arcmin)
            if hduls:
                out[s] = hduls[0]
        except Exception as e:
            print(f"  [skip] {s}: {type(e).__name__}")
    return out


def panel(ra, dec, fov_arcmin=5.0, out=None):
    """Render a multi-wavelength panel PNG; returns the saved path."""
    cuts = fetch(ra, dec, fov_arcmin)
    if not cuts:
        raise RuntimeError("no cutouts returned for this position")
    OUT.mkdir(parents=True, exist_ok=True)
    out = out or OUT / f"panel_{ra:.4f}_{dec:+.4f}.png"
    fig, axes = plt.subplots(1, len(cuts), figsize=(3 * len(cuts), 3.2))
    axes = [axes] if len(cuts) == 1 else axes
    for ax, (name, hdul) in zip(axes, cuts.items()):
        data = hdul[0].data
        ax.imshow(data, origin="lower", cmap="gray",
                  norm=simple_norm(data, "asinh", percent=99.5))
        ax.set_title(f"{name}\n{BANDS.get(name, '')}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"RA={ra:.5f}  Dec={dec:+.5f}  ({fov_arcmin}' FOV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def color(ra, dec, fov_arcmin=5.0, hips="CDS/P/DSS2/color", pix=512, out=None):
    """Save a colour image from a CDS HiPS survey (e.g. DSS2/PanSTARRS colour)."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = out or OUT / f"color_{ra:.4f}_{dec:+.4f}.jpg"
    img = hips2fits.query(hips=hips, width=pix, height=pix,
                          ra=Longitude(ra * u.deg), dec=Latitude(dec * u.deg),
                          fov=Angle(fov_arcmin * u.arcmin),
                          projection="TAN", format="jpg")
    plt.imsave(out, img)
    return Path(out)


if __name__ == "__main__":
    ra, dec = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 \
        else (213.6905918, -12.5801013)
    print(f"position: RA={ra} Dec={dec}")
    print("panel ->", panel(ra, dec))
    try:
        print("color ->", color(ra, dec))
    except Exception as e:
        print(f"color skipped ({type(e).__name__})")
