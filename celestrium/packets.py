#!/usr/bin/env python
"""packets.py — the hub's service layer: build research packets as data.

Every builder here returns a dataclass that carries the *data* plus `.to_dict()`
(for JSON / the TUI) and `.to_markdown()` (for reports). The CLI (`hub.py`) and the
future Textual TUI (`tui/`) both call these — neither re-implements the logic.

Design rule (see docs/roadmap.md #1): real work lives in `celestrium/`; surfaces present.
These builders are deliberately thin over `cutouts`/`resolvers`/`ads`/`cache` and do
no file IO themselves (so a TUI can render without writing), except `run_runbook`,
which takes explicit output dirs + a write callback so its caller owns the disk.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from . import cache, cutouts, resolvers, registry

_EXTENDED_TYPES = ("G", "Cl", "Neb", "SNR")  # want a wider default FOV


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return s or "untitled"


def write_report(reports_dir: Path, kind: str, stem: str, body: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{kind}-{slug(stem)}.md"
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path


def paper_lines(docs) -> List[str]:
    if not docs:
        return ["- No ADS records returned."]
    return [
        f"- {d.get('bibcode', '?')} ({d.get('year', '?')}): {title_of(d)}"
        for d in docs
    ]


def title_of(doc) -> str:
    """First title of an ADS doc, robust to a missing/null/empty `title` field."""
    titles = doc.get("title") or ["(untitled)"]
    return str(titles[0]) if titles else "(untitled)"


def resolve_target(name: str) -> dict:
    """SIMBAD-resolve a name to {name, otype, ra, dec}; raise if no match."""
    info = resolvers.identify(name)
    if info is None or len(info) == 0:
        raise RuntimeError(f"No SIMBAD match for {name!r}")
    row = info[0]
    return {
        "name": str(row["main_id"]),
        "otype": str(row.get("otype", "?")),
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
    }


def default_fov(otype: str, extended: float, point: float) -> float:
    return extended if any(k in otype for k in _EXTENDED_TYPES) else point


# --------------------------------------------------------------------------- #
# Paper set
# --------------------------------------------------------------------------- #
@dataclass
class PaperSet:
    query: str
    docs: list

    def to_dict(self) -> dict:
        return {"query": self.query, "n": len(self.docs), "docs": self.docs}

    def to_markdown(self) -> str:
        return "\n".join([f"# ADS paper set: {self.query}", "", *paper_lines(self.docs)])


def build_paper_set(query_text: str, rows: int = 8) -> PaperSet:
    from . import ads  # lazy: keeps the rest token-free
    return PaperSet(query_text, ads.search(query_text, rows=rows))


# --------------------------------------------------------------------------- #
# Object packet (dossier)
# --------------------------------------------------------------------------- #
@dataclass
class ObjectPacket:
    name: str
    otype: str
    ra: float
    dec: float
    survey: str
    fov: float
    color_path: Optional[Path] = None
    panel_path: Optional[Path] = None
    docs: list = field(default_factory=list)
    redshift: Optional[float] = None  # from NED, if requested + available

    def to_dict(self) -> dict:
        return {
            "name": self.name, "otype": self.otype, "ra": self.ra, "dec": self.dec,
            "survey": self.survey, "fov": self.fov, "redshift": self.redshift,
            "color_path": str(self.color_path) if self.color_path else None,
            "panel_path": str(self.panel_path) if self.panel_path else None,
            "docs": self.docs,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Dossier: {self.name}",
            "",
            f"- Type: {self.otype}",
            f"- Position: RA={self.ra:.5f}, Dec={self.dec:+.5f}",
        ]
        if self.redshift is not None:
            lines.append(f"- Redshift (NED): {self.redshift}")
        lines += [
            f"- Colour survey: {self.survey}",
            f"- FOV: {self.fov} arcmin",
            f"- Colour image: {self.color_path or 'not rendered'}",
            f"- Multi-wavelength panel: {self.panel_path or 'not rendered'}",
            "",
            "## ADS",
            *paper_lines(self.docs),
        ]
        return "\n".join(lines)


def _ned_redshift(name: str) -> Optional[float]:
    """Best-effort NED redshift; None on any failure (NED can be slow/unreachable)."""
    try:
        res = resolvers.extragalactic(name)
        if res is None or len(res) == 0:
            return None
        z = res[0]["Redshift"]
        return None if z is None else float(z)
    except Exception:
        return None


def build_object_packet(target: str, rows: int = 6, fov: Optional[float] = None,
                        images: bool = True, ned: bool = False,
                        on_error: Optional[Callable[[str], None]] = None) -> ObjectPacket:
    obj = resolve_target(target)
    fov_arcmin = fov or default_fov(obj["otype"], 8.0, 3.0)
    hips, survey = cutouts.best_color_hips(obj["dec"])
    color_path = panel_path = None
    if images:
        try:
            color_path = cutouts.color(obj["ra"], obj["dec"], fov_arcmin=fov_arcmin, hips=hips)
            panel_path = cutouts.panel(obj["ra"], obj["dec"], fov_arcmin=fov_arcmin)
        except Exception as e:  # imaging is optional; don't sink the whole packet
            if on_error:
                on_error(f"imaging unavailable ({type(e).__name__})")
    redshift = _ned_redshift(obj["name"]) if ned else None
    try:
        from . import ads
        docs = ads.search(f'object:"{obj["name"]}"', rows=rows)
    except Exception:
        docs = []
    return ObjectPacket(obj["name"], obj["otype"], obj["ra"], obj["dec"], survey,
                        fov_arcmin, color_path, panel_path, docs, redshift)


# --------------------------------------------------------------------------- #
# Field packet (blank/coordinate position)
# --------------------------------------------------------------------------- #
@dataclass
class FieldPacket:
    ra: float
    dec: float
    fov: float
    survey: str
    nearest: Optional[tuple]  # (main_id, otype, sep_arcsec) or None
    color_path: Optional[Path] = None
    panel_path: Optional[Path] = None

    @property
    def nearest_text(self) -> str:
        if self.nearest:
            return f"{self.nearest[0]} ({self.nearest[1]}), {self.nearest[2]:.1f} arcsec away"
        return "No SIMBAD object within 2 arcmin"

    def to_dict(self) -> dict:
        return {
            "ra": self.ra, "dec": self.dec, "fov": self.fov, "survey": self.survey,
            "nearest": self.nearest,
            "color_path": str(self.color_path) if self.color_path else None,
            "panel_path": str(self.panel_path) if self.panel_path else None,
        }

    def to_markdown(self) -> str:
        return "\n".join([
            f"# Field: RA={self.ra:.5f}, Dec={self.dec:+.5f}",
            "",
            f"- Nearest object: {self.nearest_text}",
            f"- Colour survey: {self.survey}",
            f"- FOV: {self.fov} arcmin",
            f"- Colour image: {self.color_path or 'not rendered'}",
            f"- Multi-wavelength panel: {self.panel_path or 'not rendered'}",
        ])


def build_field_packet(ra: float, dec: float, fov: float = 5.0, images: bool = True,
                       on_error: Optional[Callable[[str], None]] = None) -> FieldPacket:
    nearest = cutouts.identify_field(ra, dec)
    hips, survey = cutouts.best_color_hips(dec)
    color_path = panel_path = None
    if images:
        try:
            color_path = cutouts.color(ra, dec, fov_arcmin=fov, hips=hips)
            panel_path = cutouts.panel(ra, dec, fov_arcmin=fov)
        except Exception as e:
            if on_error:
                on_error(f"imaging unavailable ({type(e).__name__})")
    return FieldPacket(ra, dec, fov, survey, nearest, color_path, panel_path)


# --------------------------------------------------------------------------- #
# Runbook (data-driven, multi-step). Caller owns disk: pass dirs + contact_sheet.
# --------------------------------------------------------------------------- #
@dataclass
class RunbookResult:
    name: str
    description: str
    notes: list
    created: list  # list of (kind, Path)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "notes": self.notes,
                "created": [(k, str(p)) for k, p in self.created]}

    def to_markdown(self) -> str:
        return "\n".join([
            f"# Runbook: {self.name}",
            "",
            f"Purpose: {self.description}",
            "",
            "## Outputs",
            *self.notes,
        ])


def run_runbook(name: str, *, reports_dir: Path, atlas_dir: Path, posters_dir: Path,
                contact_sheet: Callable, limit: int = 4,
                images: bool = True, posters: bool = True,
                samples: bool = True) -> RunbookResult:
    """Execute a registry runbook spec, writing artifacts under the given dirs."""
    spec = registry.RUNBOOKS.get(name)
    if spec is None:
        raise KeyError(name)
    notes: list = []
    created: list = []
    saw_sample = False

    for step in spec.steps:
        kind = step["kind"]
        if kind == "papers":
            qtext = f'abs:"{step["phrase"]}" {step["query"]}'.strip()
            try:
                ps = build_paper_set(qtext, rows=limit)
                path = write_report(reports_dir, "papers", qtext, ps.to_markdown())
                created.append(("papers", path))
                notes.append(f"- Papers `{step['phrase']}`: {len(ps.docs)} ADS rows -> `{path}`")
            except Exception as e:
                notes.append(f"- Papers `{step['phrase']}`: unavailable ({type(e).__name__}: {e})")
        elif kind == "field":
            try:
                fp = build_field_packet(step["ra"], step["dec"],
                                        fov=step.get("fov", 5.0), images=images)
                path = write_report(reports_dir, "field",
                                    f'{step["ra"]:.5f}_{step["dec"]:+.5f}', fp.to_markdown())
                created.append(("field", path))
                notes.append(f"- {step.get('note', 'Field anchor')} -> `{path}`")
            except Exception as e:
                notes.append(f"- {step.get('note', 'Field anchor')}: unavailable ({type(e).__name__}: {e})")
        elif kind == "sample":
            saw_sample = True
            if not samples:
                continue
            recipe = step["recipe"]
            rec = registry.SAMPLE_RECIPES[recipe]
            try:
                source = registry.resolve_query_source(rec.archive)
                tab = cache.cached_query(f"sample:{recipe}", rec.adql,
                                         lambda: source.query(rec.adql))
                notes.append(f"- Sample `{recipe}`: {len(tab)} rows cached.")
            except Exception as e:
                notes.append(f"- Sample `{recipe}`: unavailable ({type(e).__name__}: {e})")
        elif kind == "atlas":
            if not images:
                notes.append("- Atlas images skipped.")
                continue
            try:
                atlas_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for target in registry.ATLAS_TARGETS[:limit]:
                    out = atlas_dir / f"{slug(target['name'])}.jpg"
                    paths.append(cutouts.color(target["ra"], target["dec"],
                                               fov_arcmin=target["fov"], pix=768, out=out))
                sheet = contact_sheet(paths, atlas_dir / "atlas-targets.png", "atlas targets")
                created.append(("atlas", sheet))
                notes.append(f"- Atlas contact sheet -> `{sheet}`")
            except Exception as e:
                notes.append(f"- Atlas contact sheet: unavailable ({type(e).__name__}: {e})")
        elif kind == "poster":
            if not posters:
                notes.append("- Posters skipped.")
                continue
            target = step["target"]
            try:
                obj = resolve_target(target)
                posters_dir.mkdir(parents=True, exist_ok=True)
                out = posters_dir / f"{slug(obj['name'])}-1080p-label.jpg"
                path = cutouts.poster(obj["ra"], obj["dec"], width=1920, height=1080,
                                      fov_arcmin=step.get("fov", 6.0), out=out,
                                      label=obj["name"], style="label")
                created.append(("poster", path))
                notes.append(f"- Poster `{target}` -> `{path}`")
            except Exception as e:
                notes.append(f"- Poster `{target}`: unavailable ({type(e).__name__}: {e})")

    if saw_sample and not samples:
        notes.append("- Samples skipped.")
    return RunbookResult(name, spec.description, notes, created)
