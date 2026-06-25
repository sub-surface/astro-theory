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
        key="ned",
        label="NED spectra and cross-identifications",
        modalities=("spectrum", "metadata", "redshift"),
        object_classes=("galaxy_agn", "cluster"),
        access="api",
        coverage_hint="Extragalactic objects with literature metadata.",
        fetch_cost="low",
        status="metadata",
        service_module=None,
    ),
    ProductSourceCapability(
        key="sdss",
        label="SDSS spectra and optical imaging",
        modalities=("spectrum", "colour_image", "photometry"),
        object_classes=("galaxy_agn", "star"),
        access="api",
        coverage_hint="Optical footprint where SDSS imaging or spectra exist.",
        fetch_cost="low",
        status="planned",
        service_module=None,
    ),
    ProductSourceCapability(
        key="mast",
        label="MAST ultraviolet and optical observations",
        modalities=("image", "spectrum", "photometry"),
        object_classes=("galaxy_agn", "nebula", "star"),
        access="api",
        coverage_hint="HST, GALEX, TESS, and other MAST-hosted missions.",
        fetch_cost="medium",
        status="planned",
        service_module="celestrium.mast",
    ),
    ProductSourceCapability(
        key="exoplanet-archive",
        label="NASA Exoplanet Archive planet and host-star tables",
        modalities=("exoplanet", "metadata"),
        object_classes=("star",),
        access="api",
        coverage_hint="Confirmed and candidate exoplanet systems.",
        fetch_cost="low",
        status="planned",
        service_module=None,
    ),
    ProductSourceCapability(
        key="heasarc",
        label="HEASARC high-energy observations",
        modalities=("xray", "gamma", "high_energy", "metadata"),
        object_classes=("galaxy_agn", "cluster", "nebula", "star"),
        access="api",
        coverage_hint="X-ray and gamma-ray mission catalogues and observations.",
        fetch_cost="medium",
        status="planned",
        service_module="celestrium.heasarc",
    ),
    ProductSourceCapability(
        key="vizier",
        label="VizieR catalogue photometry and measurements",
        modalities=("photometry", "catalogue", "metadata"),
        object_classes=("galaxy_agn", "cluster", "nebula", "star", "unknown"),
        access="api",
        coverage_hint="Broad catalogue coverage across object classes.",
        fetch_cost="low",
        status="metadata",
        service_module="celestrium.vizier",
    ),
)


def parse_request(text: str) -> ParsedRequest:
    raw = text
    target_text = text.strip()
    modality_hint = None

    prefixes = (
        ("spectrum of ", "spectrum"),
        ("spectra of ", "spectrum"),
        ("wise image of ", "colour_image"),
        ("optical image of ", "colour_image"),
        ("image of ", "colour_image"),
        ("exoplanets around ", "exoplanet"),
        ("planets around ", "exoplanet"),
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

    groups = {
        "galaxy_agn": {"QSO", "AGN", "Rad", "G", "GiG"},
        "cluster": {"Cl", "Cluster"},
        "nebula": {"Neb", "SNR"},
        "star": {"*", "Star", "PM"},
    }
    for label, values in groups.items():
        if otype in values:
            return label
    return "unknown"


def recommend_plans(
    target: ResolvedTarget,
    modality: str | None = None,
) -> list[ObservationPlan]:
    plans = []
    for capability in SOURCE_CAPABILITIES:
        if not _matches_target_or_unknown_fallback(target, capability, modality):
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
                next_action="fetch"
                if capability.status == "executable"
                else "inspect",
            )
        )
    return plans


def image_coverage_note(dec: float, survey: str = "auto") -> str:
    """Describe optical colour survey coverage and blank-tile fallback behavior."""
    from celestrium import cutouts

    auto_meta = cutouts.color_survey_metadata(dec)
    auto_labels = [meta["label"] for meta in auto_meta]

    if survey != "auto":
        selected = cutouts.COLOR_SURVEY_META.get(survey)
        label = selected["label"] if selected else survey
        return (
            f"Selected colour survey: {label}. If that frame is blank or outside "
            f"coverage, image fetch falls back through auto candidates: "
            f"{', '.join(auto_labels)}."
        )

    return (
        "Auto colour coverage tries "
        f"{', '.join(auto_labels)} in order; DSS2 is the all-sky fallback."
    )


def _matches_target_or_unknown_fallback(
    target: ResolvedTarget,
    capability: ProductSourceCapability,
    modality: str | None,
) -> bool:
    if target.object_class in capability.object_classes:
        return True
    if target.object_class != "unknown" or modality is None:
        return False
    return capability.status in {"metadata", "planned"} and modality in capability.modalities


def _parse_coordinates(target_text: str) -> tuple[float, float] | None:
    parts = target_text.split()
    if len(parts) != 2:
        return None

    try:
        ra, dec = (float(part) for part in parts)
    except ValueError:
        return None

    if 0 <= ra <= 360 and -90 <= dec <= 90:
        return (ra, dec)
    return None
