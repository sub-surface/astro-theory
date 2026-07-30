"""Request parsing and lightweight object classification for observation plans."""

from __future__ import annotations

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


@dataclass(frozen=True)
class ProductSourceCapability:
    key: str
    label: str
    modalities: tuple[str, ...]
    object_classes: tuple[str, ...]
    access: str
    coverage_hint: str
    fetch_cost: str
    status: str
    service_module: str | None = None
    scope: str = "target"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationPlan:
    target: ResolvedTarget
    product: ProductSourceCapability
    parameters: dict[str, Any]
    recommendation: str
    warnings: tuple[str, ...] = ()
    next_action: str = "fetch"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SOURCE_CAPABILITIES: tuple[ProductSourceCapability, ...] = (
    ProductSourceCapability(
        key="colour-image",
        label="Colour cutout (HiPS)",
        modalities=("colour_image", "image"),
        object_classes=("*",),
        access="hips",
        coverage_hint="Deep optical colour by declination; DSS2 all-sky fallback.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.cutouts",
        scope="field",
    ),
    ProductSourceCapability(
        key="multi-panel",
        label="Multi-wavelength panel (UV->radio)",
        modalities=("multi_panel", "image"),
        object_classes=("*",),
        access="skyview",
        coverage_hint="SkyView UV/optical/near-IR/mid-IR/radio co-registered cutouts.",
        fetch_cost="medium",
        status="executable",
        service_module="celestrium.cutouts",
        scope="field",
    ),
    ProductSourceCapability(
        key="ned",
        label="NED spectra and cross-identifications",
        modalities=("spectrum", "metadata", "redshift"),
        object_classes=("galaxy_agn", "cluster"),
        access="api",
        coverage_hint="Extragalactic objects with literature metadata.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.spectra",
        scope="target",
    ),
    ProductSourceCapability(
        key="sdss",
        label="SDSS spectra and optical imaging",
        modalities=("spectrum", "colour_image", "photometry"),
        object_classes=("galaxy_agn", "star"),
        access="api",
        coverage_hint="Optical footprint where SDSS imaging or spectra exist.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.sdss",
        scope="target",
    ),
    ProductSourceCapability(
        key="mast",
        label="MAST ultraviolet and optical observations",
        modalities=("image", "spectrum", "photometry"),
        object_classes=("galaxy_agn", "nebula", "star"),
        access="api",
        coverage_hint="HST, GALEX, TESS, and other MAST-hosted missions.",
        fetch_cost="medium",
        status="executable",
        service_module="celestrium.mast",
        scope="target",
    ),
    ProductSourceCapability(
        key="exoplanet-archive",
        label="NASA Exoplanet Archive planet and host-star tables",
        modalities=("exoplanet", "metadata"),
        object_classes=("star", "exoplanet"),
        access="api",
        coverage_hint="Confirmed and candidate exoplanet systems.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.exoplanet",
        scope="target",
    ),
    ProductSourceCapability(
        key="lightcurve",
        label="MAST TESS/Kepler photometric lightcurves",
        modalities=("lightcurve", "photometry"),
        object_classes=("star", "exoplanet"),
        access="mast",
        coverage_hint="Time-series photometry from transit missions.",
        fetch_cost="medium",
        status="executable",
        service_module="celestrium.mast",
        scope="target",
    ),
    ProductSourceCapability(
        key="transit",
        label="Exoplanet next-transit predictor",
        modalities=("transit", "metadata"),
        object_classes=("star", "exoplanet"),
        access="api",
        coverage_hint="Calculates the next UTC transit window using exoplanet ephemerides.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.exoplanet",
        scope="target",
    ),
    ProductSourceCapability(
        key="ephemeris",
        label="JPL Horizons 30-day ephemeris track",
        modalities=("ephemeris", "metadata"),
        object_classes=("solar_system",),
        access="api",
        coverage_hint="Calculated orbital path for solar system bodies.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.solarsystem",
        scope="target",
    ),
    ProductSourceCapability(
        key="solar-image",
        label="SDO Solar Dynamics near-real-time imagery",
        modalities=("image", "solar"),
        object_classes=("solar",),
        access="api",
        coverage_hint="Extreme ultraviolet tracking of the Sun.",
        fetch_cost="low",
        status="planned",
        service_module="celestrium.solarsystem",
        scope="target",
    ),
    ProductSourceCapability(
        key="heasarc",
        label="HEASARC high-energy observations",
        modalities=("xray", "gamma", "high_energy", "metadata"),
        object_classes=("galaxy_agn", "cluster", "nebula", "star"),
        access="api",
        coverage_hint="X-ray and gamma-ray mission catalogues and observations.",
        fetch_cost="medium",
        status="executable",
        service_module="celestrium.heasarc",
        scope="target",
    ),
    ProductSourceCapability(
        key="vizier",
        label="VizieR catalogue photometry and measurements",
        modalities=("photometry", "catalogue", "metadata"),
        object_classes=("galaxy_agn", "cluster", "nebula", "star", "unknown"),
        access="api",
        coverage_hint="Broad catalogue coverage across object classes.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.vizier",
        scope="field",
    ),
    ProductSourceCapability(
        key="neo",
        label="JPL CNEOS Close Approaches",
        modalities=("neo", "metadata"),
        object_classes=("solar_system",),
        access="api",
        coverage_hint="Global close-approach feed; not a target/field product.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.neos",
        scope="global",
    ),
    ProductSourceCapability(
        key="satellite",
        label="Celestrak Visible Satellites (TLEs)",
        modalities=("satellite", "metadata"),
        object_classes=("satellite",),
        access="api",
        coverage_hint="Global visible-satellite TLE feed; not a target/field product.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.satellites",
        scope="global",
    ),
    ProductSourceCapability(
        key="transient",
        label="Transient Alerts (ALeRCE/ZTF)",
        modalities=("transient", "metadata"),
        object_classes=("transient",),
        access="api",
        coverage_hint="Global feed of recent ALeRCE ZTF alerts; a Rubin-era broker "
                      "will extend this once its alert stream is public.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.transients",
        scope="global",
    ),
    ProductSourceCapability(
        key="transient-alerts",
        label="Transient alerts near this target (ALeRCE)",
        modalities=("transient", "metadata"),
        object_classes=("*",),
        access="api",
        coverage_hint="ALeRCE ZTF-alert cone search around the resolved position — "
                      "the 'show alerts near this object/field' watch action.",
        fetch_cost="low",
        status="executable",
        service_module="celestrium.transients",
        scope="target",
    ),
)


def parse_request(text: str) -> ParsedRequest:
    raw = text
    target_text = text.strip()
    modality_hint = None

    prefixes = (
        ("spectrum of ", "spectrum"),
        ("spectra of ", "spectrum"),
        ("panel of ", "multi_panel"),
        ("multi-wavelength of ", "multi_panel"),
        ("wise image of ", "colour_image"),
        ("optical image of ", "colour_image"),
        ("image of ", "colour_image"),
        ("exoplanets around ", "exoplanet"),
        ("planets around ", "exoplanet"),
        ("lightcurve of ", "lightcurve"),
        ("light curve of ", "lightcurve"),
        ("ephemeris of ", "ephemeris"),
        ("orbit of ", "ephemeris"),
        ("transits of ", "transit"),
        ("transit of ", "transit"),
        ("neos near ", "neo"),
        ("satellites near ", "satellite"),
        ("transients in ", "transient"),
        ("supernovae in ", "transient"),
    )
    lowered = target_text.lower()
    for prefix, modality in prefixes:
        if lowered.startswith(prefix):
            modality_hint = modality
            target_text = target_text[len(prefix) :].strip()
            break

    coordinates = _parse_coordinates(target_text)
    return ParsedRequest(
        raw=raw,
        target_text=target_text,
        modality_hint=modality_hint,
        coordinates=coordinates,
    )


def classify_otype(otype: str | None) -> str:
    if not otype:
        return "unknown"

    if otype == "Solar System":
        return "solar_system"
    if otype == "Sun":
        return "solar"

    groups = {
        "galaxy_agn": {"QSO", "AGN", "Rad", "G", "GiG", "BLL"},
        "cluster": {"Cl", "ClG", "Cluster"},
        "nebula": {"Neb", "SNR"},
        "star": {"*", "Star", "PM"},
        # SIMBAD's real planet code is "Pl" — the literal string "exoplanet"
        # here previously never matched anything (no SIMBAD otype is spelled
        # that way), so a resolved exoplanet fell through to "unknown" and
        # missed the exoplanet-archive/lightcurve/transit capabilities that
        # explicitly list object_classes=("star", "exoplanet").
        "exoplanet": {"Pl"},
    }
    for label, values in groups.items():
        if otype in values:
            return label
    return "unknown"


# Lower is better: actionable products lead, metadata next, planned last.
_STATUS_RANK = {"executable": 0, "metadata": 1, "planned": 2}


# Confidence by how the target was found (exact name beats a relaxed alias).
_MATCH_CONFIDENCE = {
    "exact": 1.0,
    "ephemeris": 0.95,
    "coordinate": 0.9,
    "nearby": 0.8,
    "relaxed": 0.5,
    "ambiguous": 0.4,
    "blank": 0.3,
}


def resolve_target(
    text: str,
    *,
    identify=None,
    dynamic=None,
    search=None,
    nearby=None,
) -> ResolvedTarget | None:
    """Layered resolution: coordinate → exact → dynamic → relaxed/alias → ambiguity list.

    Returns a `ResolvedTarget` whose `match_kind`/`confidence`/`alternatives`
    record *how* it was found, or ``None`` when a name resolves to nothing. The
    three backends (`identify`, `search`, `nearby`) default to `resolvers.*` but
    are injectable so this stays unit-testable without the network.
    """
    from . import resolvers
    identify = identify or resolvers.identify
    dynamic = dynamic if dynamic is not None else getattr(resolvers, "dynamic", None)
    search = search if search is not None else getattr(resolvers, "search", None)
    nearby = nearby if nearby is not None else getattr(resolvers, "nearby", None)

    req = parse_request(text)
    if req.coordinates is not None:
        return _resolve_coordinates(req.coordinates, nearby)

    # Sun is a special case: handled explicitly because SIMBAD fails or returns odd things
    if req.target_text.lower() in ("sun", "sol"):
        return ResolvedTarget(
            display_name="The Sun", aliases=(), ra=float("nan"), dec=float("nan"),
            otype="Sun", object_class="solar", confidence=1.0, match_kind="exact"
        )

    rows = _safe(identify, req.target_text)
    if rows is not None and len(rows) >= 1:
        return _target_from_row(rows[0], "exact", alternatives=_alternatives(rows[1:]))

    if dynamic is not None:
        dyn = _safe(dynamic, req.target_text)
        if dyn is not None and len(dyn) >= 1:
            if str(dyn[0].get("ra")) == "nan":
                return _target_from_row(dyn[0], "ambiguous", alternatives=_alternatives(dyn[1:]))
            return _target_from_row(dyn[0], "ephemeris", alternatives=_alternatives(dyn[1:]))

    if search is not None:
        hits = _safe(search, req.target_text)
        if hits is not None and len(hits) >= 1:
            kind = "ambiguous" if len(hits) > 1 else "relaxed"
            return _target_from_row(hits[0], kind, alternatives=_alternatives(hits[1:]))

    return None


def _resolve_coordinates(coords, nearby) -> ResolvedTarget:
    ra, dec = coords
    if nearby is not None:
        res = _safe(nearby, ra, dec)
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


def recommend_plans(
    target: ResolvedTarget,
    modality: str | None = None,
    include_global: bool = False,
) -> list[ObservationPlan]:
    plans = []
    for capability in SOURCE_CAPABILITIES:
        if capability.scope == "global" and not include_global:
            continue
        target_match = _matches_target(target, capability)
        unknown_fallback = _unknown_modality_fallback(target, capability, modality)
        if not target_match and not unknown_fallback:
            continue
        if modality is not None and modality not in capability.modalities:
            continue

        plans.append(
            ObservationPlan(
                target=target,
                product=capability,
                parameters={
                    "ra": target.ra,
                    "dec": target.dec,
                    "modalities": capability.modalities,
                },
                recommendation=(
                    f"Use {capability.label} for {target.display_name} "
                    f"({target.object_class})."
                ),
                next_action=(
                    "inspect" if unknown_fallback
                    else "fetch" if capability.status == "executable"
                    else "inspect"
                ),
            )
        )
    # Stable sort keeps SOURCE_CAPABILITIES order within each status band.
    plans.sort(key=lambda p: _STATUS_RANK.get(p.product.status, 3))
    return plans


def image_coverage_note(dec: float, survey: str = "auto") -> str:
    """Describe optical colour survey coverage and blank-tile fallback behavior."""
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


def _matches_target(target: ResolvedTarget, capability: ProductSourceCapability) -> bool:
    if "*" in capability.object_classes or target.object_class in capability.object_classes:
        return True
    return False


def _unknown_modality_fallback(
    target: ResolvedTarget,
    capability: ProductSourceCapability,
    modality: str | None,
) -> bool:
    return (
        target.object_class == "unknown"
        and modality is not None
        and capability.scope != "global"
        and modality in capability.modalities
    )


def _parse_coordinates(target_text: str) -> tuple[float, float] | None:
    """Parse a position: bare decimal degrees fast path, then anything
    astropy's SkyCoord understands (sexagesimal '12:30:49.4 +12:23:28',
    '12h30m49s +12d23m28s', …). Most papers quote sexagesimal — accepting it
    here fixes the paper cut once, for every surface."""
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
    # Cheap pre-filter: SkyCoord parsing is expensive and would happily eat
    # plain object names; require digits plus a separator that looks positional.
    if not any(ch.isdigit() for ch in target_text):
        return None
    lowered = target_text.lower()
    if not (":" in target_text
            or any(u in lowered for u in ("h", "d")) and any(u in lowered for u in ("m", "s"))):
        return None
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(target_text, unit=(u.hourangle, u.deg))
        return (float(coord.ra.deg), float(coord.dec.deg))
    except Exception:
        return None
