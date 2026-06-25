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
