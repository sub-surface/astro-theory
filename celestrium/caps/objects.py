"""Target-scoped capabilities: identity, then everything you can ask *about* it.

`target` accepts a name or a position — `resolvers.resolve_target` already parses
decimal degrees and sexagesimal, so one parameter covers both and no capability
needs a coordinate variant. Position-taking capabilities resolve via
`ctx.child("object.resolve")`, which means the resolution itself is cached and
recorded as lineage rather than repeated silently.
"""
from __future__ import annotations

from ..core.capability import Param, capability

TARGET = Param("str", help="object name, 'RA Dec', or sexagesimal position")


def resolve(ctx, target):
    """Shared front door: resolve `target` through the cached child capability."""
    from .. import resolvers
    art = ctx.child("object.resolve", target=target)
    payload = ctx.load(art.id)
    if not payload or payload.get("ra") is None:
        raise LookupError(f"could not resolve target {target!r}")
    return resolvers.ResolvedTarget(
        display_name=payload["display_name"], aliases=tuple(payload.get("aliases", ())),
        ra=float(payload["ra"]), dec=float(payload["dec"]),
        otype=payload.get("otype", ""), object_class=payload.get("object_class", "unknown"),
        confidence=float(payload.get("confidence", 0.0)),
        match_kind=payload.get("match_kind", "exact"),
    )


@capability(
    name="object.resolve", kind="data", wing="theory", cost="network",
    params={"target": TARGET},
    summary="Resolve a target to identity + position (SIMBAD/Horizons, layered).",
)
def object_resolve(ctx, target):
    from .. import resolvers
    ctx.progress(f"resolving {target}")
    found = resolvers.resolve_target(target)
    if found is None:
        raise LookupError(f"nothing resolves for {target!r}")
    payload = found.to_dict()
    ctx.progress(f"{found.display_name} ({found.object_class}, {found.match_kind})")
    return ctx.data(payload, label=found.display_name,
                    object_class=found.object_class, otype=found.otype,
                    ra=found.ra, dec=found.dec, match_kind=found.match_kind)


@capability(
    name="object.products", kind="data", wing="theory", cost="network",
    params={"target": TARGET},
    summary="Rank the capabilities that apply to this target.",
)
def object_products(ctx, target):
    from ..core import capability as capmod
    found = resolve(ctx, target)
    ranked = [{"capability": c.name, "kind": c.kind, "cost": c.cost,
               "wing": c.wing, "summary": c.summary}
              for c in capmod.for_target(found)]
    return ctx.data({"target": found.to_dict(), "products": ranked},
                    label=f"{found.display_name}: {len(ranked)} products")


@capability(
    name="object.dossier", kind="document", wing="theory", cost="network",
    params={"target": TARGET, "rows": Param("int", 6), "fov": Param("float", 0.0),
            "images": Param("bool", True), "ned": Param("bool", False)},
    summary="Object packet: identity + images + bibliography → Markdown.",
    tags=("target",),
)
def object_dossier(ctx, target, rows, fov, images, ned):
    ctx.progress(f"building dossier for {target}")
    return ctx.document(f"# Dossier: {target}\n(mocked)", label=f"dossier {target}",
                        target=target)


@capability(
    name="object.field", kind="document", wing="validation", cost="network",
    params={"target": TARGET, "fov": Param("float", 5.0),
            "images": Param("bool", True)},
    summary="Blank-field packet for a sky position → Markdown.",
    tags=("target",),
)
def object_field(ctx, target, fov, images):
    found = resolve(ctx, target)
    return ctx.document(f"# Field: {found.ra} {found.dec}\n(mocked)",
                        label=f"field {found.ra:.4f} {found.dec:+.4f}",
                        ra=found.ra, dec=found.dec, fov=fov)


@capability(
    name="object.sweep", kind="table", wing="validation", cost="network",
    params={"target": TARGET},
    summary="Ask every table-shaped archive product what it knows about a target.",
    tags=("target",),
)
def object_sweep(ctx, target):
    found = resolve(ctx, target)
    ctx.progress(f"sweeping archives for {found.display_name}")
    entries = []
    from astropy.table import Table
    table = Table(rows=[[e.get("product", ""), e.get("status", ""),
                         e.get("rows", 0), e.get("detail", "")] for e in entries] or None,
                  names=("product", "status", "rows", "detail"))
    return ctx.table(table, label=f"sweep {found.display_name}",
                     target=found.display_name)


@capability(
    name="object.spectrum", kind="spectrum", wing="validation", cost="network",
    params={"target": TARGET},
    summary="NED 1-D spectrum for an extragalactic target.",
    applies_to=("galaxy_agn", "cluster"), tags=("target",),
)
def object_spectrum(ctx, target):
    from .. import spectra
    ctx.progress(f"NED spectrum for {target}")
    result = spectra.fetch_ned_spectrum(target, out=ctx.out_path(".png"))
    if result is None:
        raise LookupError(f"no NED spectrum for {target!r}")
    return ctx.spectrum(result.path, label=f"spectrum {target}",
                        source=getattr(result, "source", "NED"),
                        summary=getattr(result, "summary", ""))


@capability(
    name="object.lightcurve", kind="table", wing="validation", cost="network",
    params={"target": TARGET},
    summary="MAST TESS/Kepler/K2 lightcurve observation metadata.",
    applies_to=("star", "exoplanet"), tags=("target",),
)
def object_lightcurve(ctx, target):
    from .. import mast
    table = mast.fetch_lightcurves(target)
    if table is None or len(table) == 0:
        raise LookupError(f"no MAST time-series observations for {target!r}")
    return ctx.table(table, label=f"lightcurves {target}")


@capability(
    name="object.exoplanets", kind="table", wing="validation", cost="network",
    params={"target": TARGET},
    summary="NASA Exoplanet Archive rows for a host star.",
    applies_to=("star", "exoplanet"), tags=("target",),
)
def object_exoplanets(ctx, target):
    from .. import exoplanet
    table = exoplanet.query_target(target)
    if table is None or len(table) == 0:
        raise LookupError(f"no exoplanet-archive rows for {target!r}")
    return ctx.table(table, label=f"exoplanets {target}")


@capability(
    name="object.transits", kind="table", wing="validation", cost="network",
    params={"target": TARGET},
    summary="Next transit windows from published ephemerides.",
    applies_to=("star", "exoplanet"), tags=("target",),
)
def object_transits(ctx, target):
    from .. import exoplanet
    table = exoplanet.predict_transits(target)
    if table is None or len(table) == 0:
        raise LookupError(f"no transit ephemeris for {target!r}")
    return ctx.table(table, label=f"transits {target}")


@capability(
    name="object.ephemeris", kind="table", wing="validation", cost="network",
    params={"target": TARGET},
    summary="JPL Horizons 30-day ephemeris track for a solar-system body.",
    applies_to=("solar_system",), tags=("target",),
)
def object_ephemeris(ctx, target):
    from .. import solarsystem
    table = solarsystem.fetch_ephemeris(target)
    if table is None or len(table) == 0:
        raise LookupError(f"no Horizons ephemeris for {target!r}")
    return ctx.table(table, label=f"ephemeris {target}")


@capability(
    name="object.highenergy", kind="table", wing="validation", cost="network",
    params={"target": TARGET, "catalog": Param("str", "chanmaster"),
            "radius_deg": Param("float", 0.2)},
    summary="HEASARC high-energy observations near a target.",
    tags=("target",),
)
def object_highenergy(ctx, target, catalog, radius_deg):
    from .. import heasarc
    found = resolve(ctx, target)
    table = heasarc.query_region(found.ra, found.dec, catalog=catalog,
                                 radius_deg=radius_deg)
    if table is None or len(table) == 0:
        raise LookupError(f"no {catalog} rows near {found.display_name}")
    return ctx.table(table, label=f"{catalog} near {found.display_name}")


@capability(
    name="object.catalogue", kind="table", wing="validation", cost="network",
    params={"target": TARGET, "catalog": Param("str", "I/355/gaiadr3"),
            "radius_arcmin": Param("float", 1.0)},
    summary="VizieR cone pull around a target.",
    tags=("target",),
)
def object_catalogue(ctx, target, catalog, radius_arcmin):
    from .. import vizier
    found = resolve(ctx, target)
    table = vizier.cone_pull(found.ra, found.dec, radius_arcmin=radius_arcmin,
                             catalog=catalog)
    if table is None or len(table) == 0:
        raise LookupError(f"no {catalog} rows near {found.display_name}")
    return ctx.table(table, label=f"{catalog} near {found.display_name}")


@capability(
    name="object.watch", kind="table", wing="validation", cost="network",
    params={"target": Param("str", ""), "candidate_list": Param("str", ""),
            "radius_arcmin": Param("float", 2.0), "days": Param("float", 7.0),
            "row_limit": Param("int", 20)},
    summary="ALeRCE transient alerts near a target or across a candidate list.",
    tags=("target",),
)
def object_watch(ctx, target, candidate_list, radius_arcmin, days, row_limit):
    from astropy.table import Table
    if not target and not candidate_list:
        raise ValueError("object.watch needs a target or a candidate_list")
    entries = []
    rows = [[e.get("name", ""), e.get("ra", float("nan")), e.get("dec", float("nan")),
             e.get("alerts", 0), e.get("detail", "")] for e in entries]
    table = Table(rows=rows or None, names=("name", "ra", "dec", "alerts", "detail"))
    return ctx.table(table, label=f"watch {target or candidate_list}",
                     days=days, radius_arcmin=radius_arcmin)
