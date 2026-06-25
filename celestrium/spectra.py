"""Spectrum discovery and rendering helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "data" / "spectra"


@dataclass(frozen=True)
class SpectrumResult:
    kind: str
    path: Path
    source: str
    columns: tuple[str, str]
    provenance: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "source": self.source,
            "columns": list(self.columns),
            "provenance": dict(self.provenance),
            "summary": self.summary,
        }


_X_CANDIDATES = ("wavelength", "lambda", "wave", "loglam", "frequency", "channel")
_Y_CANDIDATES = ("flux", "flam", "fnu", "intensity", "ivar")


def _pick_columns(tab) -> tuple[str, str]:
    names = {c.lower(): c for c in tab.colnames}
    xcol = next((names[c] for c in _X_CANDIDATES if c in names), None)
    ycol = next((names[c] for c in _Y_CANDIDATES if c in names), None)
    if not xcol or not ycol:
        raise ValueError("could not identify spectral columns")
    return xcol, ycol


def render_spectrum_table(tab, *, target: str, source: str,
                          out: Optional[Path] = None,
                          provenance: Optional[dict[str, Any]] = None) -> SpectrumResult:
    xcol, ycol = _pick_columns(tab)
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(out) if out else OUT / f"spectrum_{target.replace(' ', '_')}_{source}.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tab[xcol], tab[ycol], lw=1.2)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{target} - {source}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return SpectrumResult("spectrum", out, source, (xcol, ycol), provenance or {},
                          f"{target}: {source} spectrum")
