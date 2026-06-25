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
from astroquery.simbad import Simbad

OUT = Path(__file__).resolve().parent.parent / "data" / "cutouts"

# Deep optical colour HiPS by declination footprint, best first. The whole point:
# don't hand back low-res all-sky DSS2 when a deep survey covers this patch. See
# ../imaging-guide.md for the full decision logic (object type, resolution, λ).
COLOR_FOOTPRINTS = [
    # (dec_min, dec_max, HiPS id, label)
    (-68, +84, "CDS/P/DESI-Legacy-Surveys/DR10/color", "Legacy Surveys DR10 (deep)"),
    (-30, +90, "CDS/P/PanSTARRS/DR1/color-z-zg-g", "Pan-STARRS DR1"),
    (-90, +5,  "CDS/P/DES-DR2/ColorIRG", "DES DR2"),
    (-90, +90, "CDS/P/DSS2/color", "DSS2 colour (all-sky fallback)"),
]

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


def best_color_hips(dec):
    """Pick the deepest colour HiPS that covers this declination (id, label)."""
    for dmin, dmax, hips, label in COLOR_FOOTPRINTS:
        if dmin <= dec <= dmax:
            return hips, label
    return COLOR_FOOTPRINTS[-1][2], COLOR_FOOTPRINTS[-1][3]


def identify_field(ra, dec, radius_arcmin=2.0):
    """Nearest SIMBAD object to a position: (main_id, otype, sep_arcsec) or None.
    Lets us caption/scale an image by what's actually there before rendering."""
    try:
        s = Simbad()
        s.add_votable_fields("otype")
        res = s.query_region(SkyCoord(ra, dec, unit=u.deg),
                             radius=radius_arcmin * u.arcmin)
        if res is None or len(res) == 0:
            return None
        here = SkyCoord(ra, dec, unit=u.deg)
        objs = SkyCoord(res["ra"], res["dec"], unit=u.deg)
        i = here.separation(objs).arcsec.argmin()
        return (str(res["main_id"][i]), str(res["otype"][i]),
                float(here.separation(objs[i]).arcsec))
    except Exception:
        return None


def color(ra, dec, fov_arcmin=5.0, hips=None, pix=512, out=None):
    """Save a colour image from a CDS HiPS survey.
    hips=None auto-selects the deepest survey covering this declination."""
    if hips is None:
        hips, _ = best_color_hips(dec)
    OUT.mkdir(parents=True, exist_ok=True)
    out = out or OUT / f"color_{ra:.4f}_{dec:+.4f}.jpg"
    img = hips2fits.query(hips=hips, width=pix, height=pix,
                          ra=Longitude(ra * u.deg), dec=Latitude(dec * u.deg),
                          fov=Angle(fov_arcmin * u.arcmin),
                          projection="TAN", format="jpg")
    plt.imsave(out, img)
    return Path(out)


def smart(ra, dec, fov_arcmin=None):
    """Resolve the field, choose the best survey + FOV, render colour + panel.
    Prints a short summary of what's there and why a survey was picked."""
    obj = identify_field(ra, dec)
    if obj:
        name, otype, sep = obj
        print(f"field: {name}  ({otype})  {sep:.1f}\" from centre")
        # extended types want a wider FOV; point-like want a tight one
        if fov_arcmin is None:
            extended = any(k in otype for k in ("G", "Cl", "Neb", "SNR"))
            fov_arcmin = 8.0 if extended else 3.0
    else:
        print("field: no catalogued SIMBAD object within 2' (blank/uncharted)")
        fov_arcmin = fov_arcmin or 5.0
    hips, label = best_color_hips(dec)
    print(f"colour survey: {label}   (FOV {fov_arcmin}')")
    print("color ->", color(ra, dec, fov_arcmin=fov_arcmin, hips=hips))
    print("panel ->", panel(ra, dec, fov_arcmin=fov_arcmin))


if __name__ == "__main__":
    ra, dec = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 \
        else (213.6905918, -12.5801013)
    print(f"position: RA={ra} Dec={dec}")
    smart(ra, dec)
