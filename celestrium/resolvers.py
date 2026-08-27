#!/usr/bin/env python
"""Object resolution & per-object bibliography — SIMBAD + NED.

Automates the EsaSky -> ADS "what is this object / what's been written about it"
lookup. SIMBAD for identity/type/bibliography, NED for extragalactic redshifts
and cross-IDs. See ../docs/toolbox.md S2.
Docs: https://astroquery.readthedocs.io/en/latest/simbad/simbad.html
      https://astroquery.readthedocs.io/en/latest/ipac/ned/ned.html
"""
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned

from .net import set_timeout, SLOW_TIMEOUT

set_timeout(Simbad)
set_timeout(Ned, SLOW_TIMEOUT)  # NED is a known-slow server (see README)


def identify(name: str):
    """SIMBAD: name -> coords, object type, basic data (astropy Table)."""
    s = Simbad()
    s.add_votable_fields("otype")  # request object type alongside defaults
    return s.query_object(name)


def search(name: str, limit: int = 8):
    """SIMBAD relaxed/alias lookup (wildcard) -> Table of candidates or None.

    The fuzzy companion to `identify`: when an exact name misses, a wildcard
    query surfaces near-spellings and catalogue aliases. Returns multiple rows
    for the planner to present as an ambiguity list."""
    s = Simbad()
    s.add_votable_fields("otype")
    s.ROW_LIMIT = limit
    try:
        return s.query_object(name, wildcard=True)
    except Exception:
        return None


def nearby(ra: float, dec: float, radius_arcmin: float = 3.0, limit: int = 8):
    """SIMBAD cone search around a position -> Table of nearby objects or None.

    Lets the planner attach an identity to a bare coordinate (and fall back to
    blank-field planning when nothing is catalogued here)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    s = Simbad()
    s.add_votable_fields("otype")
    s.ROW_LIMIT = limit
    try:
        return s.query_region(SkyCoord(ra, dec, unit=u.deg),
                              radius=radius_arcmin * u.arcmin)
    except Exception:
        return None


def bibliography(name: str, limit: int = 10):
    """SIMBAD: papers (bibcodes + titles) referencing an object, via TAP.

    Per-object bibliography moved to SIMBAD's TAP tables (ref/has_ref/ident);
    the old query_bibobj goes the *other* way (objects in a given paper).
    """
    q = (f"SELECT TOP {limit} bibcode, title "
         "FROM ref JOIN has_ref ON ref.oidbib = has_ref.oidbibref "
         "JOIN ident ON has_ref.oidref = ident.oidref "
         f"WHERE id = '{name}'")
    return Simbad.query_tap(q)


def extragalactic(name: str):
    """NED: redshift + cross-identifications for an extragalactic object."""
    return Ned.query_object(name)


_SOLAR_SYSTEM_IDS = {
    "sun": "10", "sol": "10",
    "mercury": "199", "venus": "299",
    "earth": "399", "moon": "301",
    "mars": "499", "jupiter": "599",
    "saturn": "699", "uranus": "799",
    "neptune": "899", "pluto": "999",
}

def dynamic(name: str):
    """JPL Horizons: resolve a solar system body to its current Ephemeris (RA/Dec).

    Returns a list of dicts (for ambiguity handling or exact match) or None.
    """
    try:
        from astroquery.jplhorizons import Horizons

        target_id = _SOLAR_SYSTEM_IDS.get(name.lower().strip(), name.strip())

        try:
            obj = Horizons(id=target_id)
            eph = obj.ephemerides()
        except ValueError as e:
            err = str(e)
            if "Ambiguous target name" in err:
                import re
                lines = err.split('\n')
                matches = []
                for line in lines:
                    m = re.match(r'^\s*(-?\d+)\s+([A-Za-z0-9 ]+?)\s', line)
                    if m:
                        # Return ambiguity list without coordinates (require selection)
                        matches.append({
                            "main_id": m.group(2).strip(),
                            "otype": "Solar System",
                            "ra": float("nan"),
                            "dec": float("nan"),
                            "jpl_id": m.group(1).strip()
                        })
                return matches if matches else None
            return None

        if eph and len(eph) > 0:
            row = eph[0]
            return [{
                "main_id": str(row["targetname"]).strip(),
                "otype": "Solar System",
                "ra": float(row["RA"]),
                "dec": float(row["DEC"]),
            }]
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Target Data Models & Layered Resolution
# --------------------------------------------------------------------------- #
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedRequest:
    raw: str
    target_text: str
    modality_hint: str | None = None
    parameter_hints: dict[str, Any] = field(default_factory=dict)
    coordinates: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedTarget:
    display_name: str
    aliases: tuple[str, ...]
    ra: float
    dec: float
    otype: str
    object_class: str
    confidence: float
    match_kind: str
    alternatives: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MATCH_CONFIDENCE = {
    "exact": 1.0,
    "ephemeris": 0.95,
    "coordinate": 0.9,
    "nearby": 0.8,
    "relaxed": 0.5,
    "ambiguous": 0.4,
    "blank": 0.3,
}

_OTYPE_CLASS_MAP = {
    "G": "galaxy_agn", "Gal": "galaxy_agn", "AGN": "galaxy_agn", "QSO": "galaxy_agn",
    "Sy1": "galaxy_agn", "Sy2": "galaxy_agn", "Blaz": "galaxy_agn", "BLL": "galaxy_agn",
    "ClG": "cluster", "Group": "cluster", "SubG": "cluster",
    "Star": "star", "*": "star", "PM*": "star", "V*": "star", "WD*": "star",
    "X": "high_energy", "gamma": "high_energy", "pulsar": "high_energy", "PSR": "high_energy",
    "SNR": "nebula", "PN": "nebula", "HII": "nebula",
    "Solar System": "solar", "Sun": "solar", "Planet": "solar", "Asteroid": "solar",
}


def classify_otype(otype: str) -> str:
    if not otype:
        return "unknown"
    for key, cls in _OTYPE_CLASS_MAP.items():
        if key.lower() in otype.lower():
            return cls
    return "unknown"


def parse_request(raw_text: str) -> ParsedRequest:
    text = raw_text.strip()
    coords = parse_coordinates(text)
    return ParsedRequest(raw=raw_text, target_text=text, coordinates=coords)


def parse_coordinates(target_text: str) -> tuple[float, float] | None:
    parts = target_text.split()
    if len(parts) == 2:
        try:
            ra, dec = (float(part) for part in parts)
        except ValueError:
            pass
        else:
            if 0 <= ra <= 360 and -90 <= dec <= 90:
                return (ra, dec)
            return None

    return _parse_skycoord(target_text)


def _parse_skycoord(target_text: str) -> tuple[float, float] | None:
    try:
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        coord = SkyCoord(target_text, unit=(u.hourangle, u.deg))
        return (float(coord.ra.deg), float(coord.dec.deg))
    except Exception:
        pass

    try:
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(target_text)
        return (float(coord.ra.deg), float(coord.dec.deg))
    except Exception:
        pass

    return None


def resolve_target(
    text: str,
    *,
    identify_fn=None,
    dynamic_fn=None,
    search_fn=None,
    nearby_fn=None,
) -> ResolvedTarget | None:
    identify_fn = identify_fn or identify
    dynamic_fn = dynamic_fn if dynamic_fn is not None else dynamic
    search_fn = search_fn if search_fn is not None else search
    nearby_fn = nearby_fn if nearby_fn is not None else nearby

    req = parse_request(text)
    if req.coordinates is not None:
        return _resolve_coordinates(req.coordinates, nearby_fn)

    if req.target_text.lower() in ("sun", "sol"):
        return ResolvedTarget(
            display_name="The Sun", aliases=(), ra=float("nan"), dec=float("nan"),
            otype="Sun", object_class="solar", confidence=1.0, match_kind="exact"
        )

    rows = _safe(identify_fn, req.target_text)
    if rows is not None and len(rows) >= 1:
        return _target_from_row(rows[0], "exact", alternatives=_alternatives(rows[1:]))

    if dynamic_fn is not None:
        dyn = _safe(dynamic_fn, req.target_text)
        if dyn is not None and len(dyn) >= 1:
            if str(dyn[0].get("ra")) == "nan":
                return _target_from_row(dyn[0], "ambiguous", alternatives=_alternatives(dyn[1:]))
            return _target_from_row(dyn[0], "ephemeris", alternatives=_alternatives(dyn[1:]))

    if search_fn is not None:
        hits = _safe(search_fn, req.target_text)
        if hits is not None and len(hits) >= 1:
            kind = "ambiguous" if len(hits) > 1 else "relaxed"
            return _target_from_row(hits[0], kind, alternatives=_alternatives(hits[1:]))

    return None


def _resolve_coordinates(coords, nearby_fn) -> ResolvedTarget:
    ra, dec = coords
    if nearby_fn is not None:
        res = _safe(nearby_fn, ra, dec)
        if res is not None and len(res) >= 1:
            return _target_from_row(res[0], "nearby", ra=ra, dec=dec,
                                    alternatives=_alternatives(res[1:]))
    return ResolvedTarget(
        display_name=f"field {ra:.4f} {dec:+.4f}", aliases=(), ra=ra, dec=dec,
        otype="", object_class="unknown",
        confidence=_MATCH_CONFIDENCE["blank"], match_kind="blank",
    )


def _target_from_row(row, match_kind, *, ra=None, dec=None,
                     alternatives=()) -> ResolvedTarget:
    name, otype, row_ra, row_dec = _row_fields(row)
    return ResolvedTarget(
        display_name=name, aliases=(),
        ra=ra if ra is not None else row_ra,
        dec=dec if dec is not None else row_dec,
        otype=otype, object_class=classify_otype(otype),
        confidence=_MATCH_CONFIDENCE.get(match_kind, 0.5), match_kind=match_kind,
        alternatives=alternatives,
    )


def _row_fields(row):
    if isinstance(row, dict):
        name = str(row.get("main_id", "?"))
        otype = str(row.get("otype", ""))
        ra = float(row.get("ra", float("nan")))
        dec = float(row.get("dec", float("nan")))
        return name, otype, ra, dec

    colnames = getattr(row, "colnames", ())
    name = str(row["main_id"]) if "main_id" in colnames else "?"
    otype = str(row["otype"]) if "otype" in colnames else ""
    try:
        ra = float(row["ra"]) if "ra" in colnames else float("nan")
        dec = float(row["dec"]) if "dec" in colnames else float("nan")
    except (TypeError, ValueError):
        ra = dec = float("nan")
    return name, otype, ra, dec


def _alternatives(rows) -> tuple[dict[str, Any], ...]:
    out = []
    for row in (rows or [])[:6]:
        name, otype, ra, dec = _row_fields(row)
        out.append({"name": name, "otype": otype, "ra": ra, "dec": dec})
    return tuple(out)


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def image_coverage_note(dec: float, survey: str = "auto") -> str:
    from celestrium import cutouts
    auto_meta = cutouts.color_survey_metadata(dec)
    auto_labels = [meta["label"] for meta in auto_meta]

    if survey != "auto":
        selected = cutouts.COLOR_SURVEYS.get(survey)
        if selected is None:
            return (
                f"Requested colour survey '{survey}' is unknown or unavailable. "
                f"Image fetch will use auto candidates with fallback: "
                f"{', '.join(auto_labels)}."
            )
        label = selected[1]
        return (
            f"Selected colour survey: {label}. If that frame is blank or outside "
            f"coverage, image fetch falls back through auto candidates: "
            f"{', '.join(auto_labels)}."
        )

    return (
        "Auto colour coverage tries "
        f"{', '.join(auto_labels)} in order; DSS2 is the all-sky fallback."
    )


@dataclass(frozen=True)
class TargetProduct:
    key: str
    label: str
    summary: str
    kind: str
    cost: str
    status: str = "executable"
    coverage_hint: str = ""
    access: str = "open"
    fetch_cost: str = "network"
    modalities: tuple[str, ...] = ("table",)


@dataclass(frozen=True)
class TargetPlan:
    product: TargetProduct
    next_action: str = "fetch"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductResult:
    kind: str
    label: str
    summary: str = ""
    provenance: str = ""
    table: Any = None
    path: Any = None
    fov: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "summary": self.summary,
            "provenance": self.provenance,
            "path": str(self.path) if self.path else None,
            "fov": self.fov,
        }


def recommend_plans(target: ResolvedTarget, modality: str | None = None) -> list[TargetPlan]:
    """Rank capabilities applicable to a resolved target."""
    from .core import capability as capmod
    capmod.load_all()
    caps = capmod.for_target(target)
    plans = []
    for c in caps:
        if modality:
            mod_low = modality.lower()
            if mod_low not in c.kind.lower() and mod_low not in c.name.lower():
                continue
        prod = TargetProduct(
            key=c.name,
            label=c.name,
            summary=c.summary,
            kind=c.kind,
            cost=c.cost,
            status="executable",
            coverage_hint=c.summary,
            access="open",
            fetch_cost=c.cost,
            modalities=(c.kind,),
        )
        plans.append(TargetPlan(product=prod, next_action="fetch"))
    return plans


def execute_product(target: ResolvedTarget, plan: TargetPlan,
                    settings: dict | None = None, emit: Any = None) -> ProductResult:
    """Execute a chosen product plan through the kernel capability engine."""
    from .core.kernel import Kernel
    from pathlib import Path

    k = Kernel()
    cap_key = plan.product.key
    params: dict[str, Any] = {"target": target.display_name}
    if settings:
        if "fov" in settings and settings["fov"] is not None:
            params["fov"] = settings["fov"]
        if "pix" in settings and settings["pix"] is not None:
            params["pix"] = settings["pix"]
        if "survey" in settings and settings["survey"] not in (None, "auto"):
            params["survey"] = settings["survey"]

    art = k.run(cap_key, params)
    payload = k.load(art.id)
    if art.kind in ("table", "surface"):
        return ProductResult(kind="table", label=art.label or cap_key,
                             summary=art.cap, table=payload)
    if art.kind in ("image", "figure", "spectrum"):
        path = art.full_path or (Path(art.path) if art.path else None)
        return ProductResult(kind=art.kind, label=art.label or cap_key,
                             summary=art.cap, path=path)
    return ProductResult(kind=art.kind, label=art.label or cap_key,
                         summary=str(payload))
