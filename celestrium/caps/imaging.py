"""The imaging wing — colour cutouts, panels, posters, contact sheets.

Every renderer writes straight to `ctx.out_path()`, so the artifact's payload
location is its content address and reruns are free.
"""
from __future__ import annotations

from ..core.capability import Param, capability
from .objects import TARGET, resolve

STYLES = ("clean", "label", "science")


def _fov(found, requested: float) -> float:
    if requested and requested > 0:
        return float(requested)
    return 8.0 if "G" in (found.otype or "").upper() else 3.0


@capability(
    name="imaging.cutout", kind="image", wing="imaging", cost="network",
    params={"target": TARGET, "fov": Param("float", 0.0, help="arcmin; 0 = auto"),
            "pix": Param("int", 0, help="0 = auto from FOV"),
            "survey": Param("str", "auto")},
    summary="Colour cutout with survey fallback (blank tiles are skipped).",
    tags=("target",),
)
def imaging_cutout(ctx, target, fov, pix, survey):
    from .. import cutouts
    found = resolve(ctx, target)
    fov_arcmin = _fov(found, fov)
    chosen = survey if survey in cutouts.COLOR_SURVEYS else None
    ctx.progress(f"colour cutout {found.display_name} @ {fov_arcmin:.1f}'")
    path, label = cutouts.color_auto(found.ra, found.dec, fov_arcmin=fov_arcmin,
                                     pix=(pix or None), survey=chosen,
                                     out=ctx.out_path(".jpg"))
    ctx.note(f"survey: {label}")
    return ctx.image(path, label=f"{found.display_name} ({label})",
                     survey=label, fov=fov_arcmin, ra=found.ra, dec=found.dec)


@capability(
    name="imaging.panel", kind="figure", wing="imaging", cost="network",
    params={"target": TARGET, "fov": Param("float", 5.0)},
    summary="Multi-wavelength panel (UV → radio) for a position.",
    tags=("target",),
)
def imaging_panel(ctx, target, fov):
    from .. import cutouts
    found = resolve(ctx, target)
    ctx.progress(f"multi-wavelength panel {found.display_name}")
    path = cutouts.panel(found.ra, found.dec, fov_arcmin=fov,
                         out=ctx.out_path(".png"))
    return ctx.figure(path, label=f"panel {found.display_name}",
                      fov=fov, ra=found.ra, dec=found.dec)


@capability(
    name="imaging.poster", kind="image", wing="imaging", cost="network",
    params={"target": TARGET, "fov": Param("float", 8.0, help="arcmin; 0 = auto"),
            "width": Param("int", 1920), "height": Param("int", 1080),
            "style": Param("enum", "clean", STYLES)},
    summary="Wallpaper-grade render of a target.",
    tags=("target",),
)
def imaging_poster(ctx, target, fov, width, height, style):
    from .. import cutouts
    found = resolve(ctx, target)
    fov = _fov(found, fov)
    ctx.progress(f"poster {found.display_name} {width}×{height} ({style})")
    path = cutouts.poster(found.ra, found.dec, fov_arcmin=fov, width=width,
                          height=height, style=style,
                          label=found.display_name, out=ctx.out_path(".jpg"))
    return ctx.image(path, label=f"poster {found.display_name}",
                     style=style, width=width, height=height, fov=fov)


@capability(
    name="imaging.atlas", kind="figure", wing="imaging", cost="network",
    params={"limit": Param("int", 6), "fov": Param("float", 0.0)},
    summary="Contact sheet of the curated atlas targets.",
)
def imaging_atlas(ctx, limit, fov):
    from .. import cutouts, config
    targets = config.ATLAS_TARGETS[:max(1, limit)]
    rendered = []
    for entry in targets:
        ctx.check_cancel()
        ctx.progress(f"atlas: {entry['name']}")
        try:
            child = ctx.child("imaging.cutout", target=entry["name"],
                              fov=(fov or entry.get("fov", 8.0)))
            rendered.append(child.full_path)
        except Exception as exc:                       # one bad tile ≠ no sheet
            ctx.note(f"{entry['name']}: {type(exc).__name__}: {exc}")
    if not rendered:
        raise RuntimeError("no atlas tiles rendered")
    path = cutouts.contact_sheet(rendered, ctx.out_path(".png"),
                                 f"Celestrium atlas — {len(rendered)} targets")
    return ctx.figure(path, label=f"atlas sheet ({len(rendered)})",
                      tiles=len(rendered))


@capability(
    name="imaging.shootout", kind="figure", wing="imaging", cost="network",
    params={"target": TARGET, "fov": Param("float", 8.0)},
    summary="Same field across every covering survey — see the choice, don't trust it.",
    tags=("target",),
)
def imaging_shootout(ctx, target, fov):
    """A diagnostic for the imaging guide's own footprint logic: tile one field
    across Legacy/Pan-STARRS/DES/DSS2 rather than showing only the winner."""
    from .. import cutouts
    found = resolve(ctx, target)
    rendered = []
    for key in cutouts.COLOR_SURVEYS:
        ctx.check_cancel()
        ctx.progress(f"shootout: {key}")
        try:
            child = ctx.child("imaging.cutout", target=target, fov=fov, survey=key)
            rendered.append(child.full_path)
        except Exception as exc:
            ctx.note(f"{key}: {type(exc).__name__}: {exc}")
    if not rendered:
        raise RuntimeError("no survey returned an image for this field")
    path = cutouts.contact_sheet(rendered, ctx.out_path(".png"),
                                 f"{found.display_name} — survey shootout")
    return ctx.figure(path, label=f"shootout {found.display_name}",
                      surveys=len(rendered))
