# Observation Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 4's planner-backed Resolve foundation with source capabilities, image coverage recommendations, spectra artifact rendering, one guarded spectra adapter, and TUI planner controls.

**Architecture:** Add small service modules under `celestrium/` and keep the TUI as a presenter. `planner.py` owns pure request parsing, target modeling, source capability metadata, and recommendation ranking. `spectra.py` owns spectral table interpretation, artifact rendering, and live adapter wrappers. `cutouts.py` remains the image backend but exposes structured survey metadata for the planner.

**Tech Stack:** Python dataclasses, astropy tables/FITS where already installed, matplotlib for artifacts, Textual for TUI, pytest for hermetic tests. Optional network adapters stay lazy and are not exercised by default tests.

---

## File Map

- Create `celestrium/planner.py`: pure dataclasses, source capabilities, request parsing, object-class ranking, image coverage helpers.
- Create `celestrium/spectra.py`: spectra result dataclass, spectral table interpretation, PNG rendering, guarded NED or SDSS adapter.
- Modify `celestrium/cutouts.py`: expose image survey metadata without changing existing render paths.
- Modify `celestrium/tui/app.py`: replace image-only config with planner state and product-aware controls, still using existing `color_auto` for colour images.
- Modify `celestrium/tui/__init__.py`: update feature description if needed.
- Create `tests/test_planner.py`: hermetic planner/source/coverage tests.
- Create `tests/test_spectra.py`: hermetic fake-table spectra rendering tests.
- Modify `tests/test_tui.py`: planner-control smoke tests.
- Optionally modify `celestrium/requirements.txt`: only if a new dependency is truly required. Prefer no new dependency in this phase.

---

### Task 1: Planner Models and Request Parsing

**Files:**
- Create: `celestrium/planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write failing planner parsing/model tests**

Add to `tests/test_planner.py`:

```python
from celestrium import planner


def test_parse_request_extracts_modality_and_target():
    req = planner.parse_request("spectrum of 3C 273")
    assert req.raw == "spectrum of 3C 273"
    assert req.target_text == "3C 273"
    assert req.modality_hint == "spectrum"


def test_parse_request_accepts_coordinates():
    req = planner.parse_request("187.7059 +12.3911")
    assert req.target_text == "187.7059 +12.3911"
    assert req.modality_hint is None
    assert req.coordinates == (187.7059, 12.3911)


def test_classify_object_type_groups_agn_and_blank():
    assert planner.classify_otype("QSO") == "galaxy_agn"
    assert planner.classify_otype("Rad") == "galaxy_agn"
    assert planner.classify_otype("G") == "galaxy_agn"
    assert planner.classify_otype("") == "unknown"
    assert planner.classify_otype(None) == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_planner.py -q`

Expected: FAIL with `ImportError` or `AttributeError` because `celestrium.planner` does not exist yet.

- [ ] **Step 3: Implement minimal planner dataclasses and parsing**

Create `celestrium/planner.py`:

```python
"""Observation planning primitives for the Celestrium TUI and CLI.

This module is intentionally mostly pure: request parsing, source capabilities,
and recommendation ranking should be testable without network access.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ParsedRequest:
    raw: str
    target_text: str
    modality_hint: Optional[str] = None
    parameter_hints: dict[str, Any] = field(default_factory=dict)
    coordinates: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "target_text": self.target_text,
            "modality_hint": self.modality_hint,
            "parameter_hints": dict(self.parameter_hints),
            "coordinates": self.coordinates,
        }


_MODALITY_PREFIXES = {
    "spectrum": ("spectrum of ", "spectra of "),
    "colour_image": ("image of ", "wise image of ", "optical image of "),
    "exoplanet": ("exoplanets around ", "planets around "),
}


def _parse_coordinates(text: str) -> Optional[tuple[float, float]]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        ra, dec = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if 0 <= ra <= 360 and -90 <= dec <= 90:
        return ra, dec
    return None


def parse_request(text: str) -> ParsedRequest:
    raw = text.strip()
    lowered = raw.lower()
    modality = None
    target = raw
    hints: dict[str, Any] = {}
    for key, prefixes in _MODALITY_PREFIXES.items():
        for prefix in prefixes:
            if lowered.startswith(prefix):
                modality = key
                target = raw[len(prefix):].strip()
                if "wise" in prefix:
                    hints["wavelength"] = "mid_ir"
                break
        if modality:
            break
    coords = _parse_coordinates(target)
    return ParsedRequest(raw=raw, target_text=target, modality_hint=modality,
                         parameter_hints=hints, coordinates=coords)


def classify_otype(otype: Optional[str]) -> str:
    text = (otype or "").strip()
    if not text:
        return "unknown"
    if any(k in text for k in ("QSO", "AGN", "Rad", "G", "GiG")):
        return "galaxy_agn"
    if any(k in text for k in ("Cl", "Cluster")):
        return "cluster"
    if any(k in text for k in ("Neb", "SNR")):
        return "nebula"
    if any(k in text for k in ("*", "Star", "PM")):
        return "star"
    return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_planner.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/planner.py tests/test_planner.py
git commit -m "feat: add observation planner request model"
```

---

### Task 2: Source Capability Registry and Recommendations

**Files:**
- Modify: `celestrium/planner.py`
- Modify: `tests/test_planner.py`

- [ ] **Step 1: Write failing source capability tests**

Append to `tests/test_planner.py`:

```python
def test_source_capabilities_include_first_wave_sources():
    keys = {src.key for src in planner.SOURCE_CAPABILITIES}
    assert {"ned", "sdss", "mast", "exoplanet-archive", "heasarc", "vizier"} <= keys


def test_recommendations_for_agn_include_spectra_and_high_energy():
    target = planner.ResolvedTarget(
        display_name="3C 273",
        aliases=(),
        ra=187.2779,
        dec=2.0524,
        otype="QSO",
        object_class="galaxy_agn",
        confidence=1.0,
        match_kind="exact",
    )
    plans = planner.recommend_plans(target)
    labels = [p.product.label for p in plans]
    assert any("NED spectra" in label or "SDSS spectra" in label for label in labels)
    assert any("HEASARC" in label for label in labels)
    assert all(p.product.status in {"executable", "metadata", "planned"} for p in plans)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_planner.py -q`

Expected: FAIL with `AttributeError` for missing `SOURCE_CAPABILITIES`, `ResolvedTarget`, or `recommend_plans`.

- [ ] **Step 3: Implement source capability dataclasses and ranking**

Add to `celestrium/planner.py`:

```python
@dataclass(frozen=True)
class ResolvedTarget:
    display_name: str
    aliases: tuple[str, ...]
    ra: Optional[float]
    dec: Optional[float]
    otype: str
    object_class: str
    confidence: float
    match_kind: str
    alternatives: tuple["ResolvedTarget", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "ra": self.ra,
            "dec": self.dec,
            "otype": self.otype,
            "object_class": self.object_class,
            "confidence": self.confidence,
            "match_kind": self.match_kind,
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


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
    service_module: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "modalities": list(self.modalities),
            "object_classes": list(self.object_classes),
            "access": self.access,
            "coverage_hint": self.coverage_hint,
            "fetch_cost": self.fetch_cost,
            "status": self.status,
            "service_module": self.service_module,
        }


@dataclass(frozen=True)
class ObservationPlan:
    target: ResolvedTarget
    product: ProductSourceCapability
    parameters: dict[str, Any]
    recommendation: str
    warnings: tuple[str, ...] = ()
    next_action: str = "fetch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "product": self.product.to_dict(),
            "parameters": dict(self.parameters),
            "recommendation": self.recommendation,
            "warnings": list(self.warnings),
            "next_action": self.next_action,
        }


SOURCE_CAPABILITIES = (
    ProductSourceCapability("ned", "NED spectra and extragalactic context",
                            ("spectrum", "catalogue", "bibliography"),
                            ("galaxy_agn", "cluster"), "astroquery",
                            "Best for named extragalactic objects; spectra may be sparse.",
                            "artifact", "executable", "celestrium.spectra"),
    ProductSourceCapability("sdss", "SDSS spectra",
                            ("spectrum", "catalogue"), ("star", "galaxy_agn"),
                            "astroquery", "Optical spectra inside the SDSS footprint.",
                            "artifact", "executable", "celestrium.spectra"),
    ProductSourceCapability("mast", "MAST observation metadata",
                            ("observation-metadata", "lightcurve", "image", "spectrum"),
                            ("star", "exoplanet_host", "galaxy_agn", "unknown"),
                            "astroquery", "HST/JWST/TESS/Kepler discovery; downloads are deliberate.",
                            "metadata", "metadata", None),
    ProductSourceCapability("exoplanet-archive", "NASA Exoplanet Archive TAP",
                            ("exoplanet", "catalogue"), ("star", "exoplanet_host"),
                            "TAP", "Host and planet summaries via compact ADQL queries.",
                            "row query", "metadata", None),
    ProductSourceCapability("heasarc", "HEASARC high-energy observations",
                            ("high-energy", "observation-metadata", "catalogue"),
                            ("galaxy_agn", "cluster", "star", "unknown"),
                            "astroquery", "X-ray/gamma observation context and product links.",
                            "metadata", "metadata", "celestrium.heasarc"),
    ProductSourceCapability("vizier", "VizieR catalogue context",
                            ("catalogue",), ("star", "galaxy_agn", "cluster", "nebula", "unknown"),
                            "astroquery/VO", "Broad published-catalogue fallback.",
                            "row query", "metadata", "celestrium.vizier"),
)


def recommend_plans(target: ResolvedTarget,
                    modality: Optional[str] = None) -> list[ObservationPlan]:
    plans: list[ObservationPlan] = []
    for cap in SOURCE_CAPABILITIES:
        if target.object_class not in cap.object_classes and "unknown" not in cap.object_classes:
            continue
        if modality and modality not in cap.modalities:
            continue
        plans.append(ObservationPlan(
            target=target,
            product=cap,
            parameters={},
            recommendation=cap.coverage_hint,
            next_action="fetch" if cap.status == "executable" else "inspect",
        ))
    return plans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_planner.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/planner.py tests/test_planner.py
git commit -m "feat: add planner source capabilities"
```

---

### Task 3: Image Survey Metadata and Coverage Recommendations

**Files:**
- Modify: `celestrium/cutouts.py`
- Modify: `celestrium/planner.py`
- Modify: `tests/test_planner.py`
- Modify: `tests/test_cutouts_fallback.py`

- [ ] **Step 1: Write failing coverage metadata tests**

Append to `tests/test_cutouts_fallback.py`:

```python
def test_color_survey_metadata_matches_candidates():
    meta = cutouts.color_survey_metadata(41.27)
    keys = [m["key"] for m in meta]
    assert keys[0] == "legacy"
    assert "dss2" in keys
    assert all("coverage" in m for m in meta)
```

Append to `tests/test_planner.py`:

```python
def test_coverage_note_mentions_forced_survey_and_fallback():
    note = planner.image_coverage_note(dec=41.27, survey="panstarrs")
    assert "Pan-STARRS" in note
    assert "fallback" in note.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cutouts_fallback.py tests/test_planner.py -q`

Expected: FAIL with missing `color_survey_metadata` and `image_coverage_note`.

- [ ] **Step 3: Implement survey metadata helpers**

Modify `celestrium/cutouts.py` by adding:

```python
COLOR_SURVEY_META = {
    "legacy": {
        "label": "Legacy Surveys DR10 (deep)",
        "hips": "CDS/P/DESI-Legacy-Surveys/DR10/color",
        "wavelength": "optical",
        "coverage": "Dec -68 to +84; deep optical, may still be blank in local gaps.",
    },
    "panstarrs": {
        "label": "Pan-STARRS DR1",
        "hips": "CDS/P/PanSTARRS/DR1/color-z-zg-g",
        "wavelength": "optical",
        "coverage": "Dec > -30; strong northern optical fallback.",
    },
    "des": {
        "label": "DES DR2",
        "hips": "CDS/P/DES-DR2/ColorIRG",
        "wavelength": "optical/near-IR",
        "coverage": "Southern footprint, roughly Dec < +5.",
    },
    "dss2": {
        "label": "DSS2 colour (all-sky)",
        "hips": "CDS/P/DSS2/color",
        "wavelength": "optical",
        "coverage": "All-sky fallback; shallow and lower resolution.",
    },
}


def color_survey_metadata(dec):
    """Return ordered colour-survey metadata for a declination."""
    hips_to_key = {v["hips"]: k for k, v in COLOR_SURVEY_META.items()}
    out = []
    for hips, label in color_hips_candidates(dec):
        key = hips_to_key.get(hips, label.lower().split()[0])
        meta = dict(COLOR_SURVEY_META.get(key, {}))
        meta.setdefault("label", label)
        meta.setdefault("hips", hips)
        meta["key"] = key
        out.append(meta)
    return out
```

Modify `celestrium/planner.py` by adding:

```python
def image_coverage_note(dec: float, survey: str = "auto") -> str:
    from . import cutouts
    meta = cutouts.color_survey_metadata(dec)
    if survey and survey != "auto":
        selected = next((m for m in meta if m["key"] == survey), None)
        label = selected["label"] if selected else survey
        return f"{label} will be tried first; blank/no-coverage frames can fallback through auto candidates."
    labels = ", ".join(m["label"] for m in meta[:3])
    return f"Auto colour coverage candidates: {labels}. DSS2 remains the all-sky fallback."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cutouts_fallback.py tests/test_planner.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/cutouts.py celestrium/planner.py tests/test_cutouts_fallback.py tests/test_planner.py
git commit -m "feat: add planner image coverage metadata"
```

---

### Task 4: Spectra Rendering Service

**Files:**
- Create: `celestrium/spectra.py`
- Create: `tests/test_spectra.py`

- [ ] **Step 1: Write failing spectra rendering tests**

Create `tests/test_spectra.py`:

```python
import pytest
from astropy.table import Table

from celestrium import spectra


def test_render_spectrum_table_writes_artifact(tmp_path):
    tab = Table({"wavelength": [4000, 5000, 6000], "flux": [1.0, 3.0, 2.0]})
    result = spectra.render_spectrum_table(tab, target="Demo", source="fake", out=tmp_path / "demo.png")
    assert result.kind == "spectrum"
    assert result.path.exists()
    assert result.source == "fake"
    assert result.columns == ("wavelength", "flux")


def test_render_spectrum_table_rejects_missing_columns(tmp_path):
    tab = Table({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(ValueError, match="spectral columns"):
        spectra.render_spectrum_table(tab, target="Bad", source="fake", out=tmp_path / "bad.png")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spectra.py -q`

Expected: FAIL with `ImportError` because `celestrium.spectra` does not exist.

- [ ] **Step 3: Implement spectra dataclass and renderer**

Create `celestrium/spectra.py`:

```python
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
    x = next((names[c] for c in _X_CANDIDATES if c in names), None)
    y = next((names[c] for c in _Y_CANDIDATES if c in names), None)
    if not x or not y:
        raise ValueError("could not identify spectral columns")
    return x, y


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
    return SpectrumResult("spectrum", out, source, (xcol, ycol),
                          provenance or {}, f"{target}: {source} spectrum")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spectra.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/spectra.py tests/test_spectra.py
git commit -m "feat: render spectra artifacts"
```

---

### Task 5: Guarded Live Spectra Adapter

**Files:**
- Modify: `celestrium/spectra.py`
- Modify: `tests/test_spectra.py`

- [ ] **Step 1: Write failing adapter tests using fake backends**

Append to `tests/test_spectra.py`:

```python
def test_fetch_ned_spectrum_returns_none_when_no_spectra(monkeypatch):
    class FakeNed:
        @staticmethod
        def get_spectra(target):
            return []

    monkeypatch.setattr(spectra, "_ned_client", lambda: FakeNed)
    assert spectra.fetch_ned_spectrum("Nope") is None


def test_fetch_ned_spectrum_renders_first_table(monkeypatch, tmp_path):
    from astropy.io import fits

    table = Table({"wavelength": [1, 2, 3], "flux": [2.0, 3.0, 4.0]})
    hdu = fits.BinTableHDU(table)

    class FakeNed:
        @staticmethod
        def get_spectra(target):
            return [[hdu]]

    monkeypatch.setattr(spectra, "_ned_client", lambda: FakeNed)
    result = spectra.fetch_ned_spectrum("3C 273", out=tmp_path / "ned.png")
    assert result is not None
    assert result.path.exists()
    assert result.source == "NED"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spectra.py -q`

Expected: FAIL with missing `fetch_ned_spectrum` and `_ned_client`.

- [ ] **Step 3: Implement guarded NED adapter**

Add to `celestrium/spectra.py`:

```python
def _ned_client():
    from astroquery.ipac.ned import Ned
    return Ned


def _table_from_hdul(hdul):
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is not None and hasattr(data, "names"):
            from astropy.table import Table
            return Table(data)
    return None


def fetch_ned_spectrum(target: str, out: Optional[Path] = None) -> Optional[SpectrumResult]:
    """Fetch and render the first NED spectrum for a target, or None if unavailable."""
    try:
        spectra = _ned_client().get_spectra(target)
    except Exception:
        return None
    for hdul in spectra or []:
        tab = _table_from_hdul(hdul)
        if tab is None:
            continue
        try:
            return render_spectrum_table(
                tab, target=target, source="NED", out=out,
                provenance={"adapter": "NED", "target": target},
            )
        except ValueError:
            continue
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spectra.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/spectra.py tests/test_spectra.py
git commit -m "feat: add guarded NED spectra adapter"
```

---

### Task 6: Planner-Backed Resolve TUI State

**Files:**
- Modify: `celestrium/tui/app.py`
- Modify: `tests/test_tui.py`

- [ ] **Step 1: Write failing TUI planner state tests**

Append to `tests/test_tui.py`:

```python
def test_planner_settings_can_select_spectrum_network_free():
    from celestrium.tui.app import ImageSettingsScreen

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.action_image_settings()
            await pilot.pause()
            assert isinstance(app.screen, ImageSettingsScreen)
            app.screen.query_one("#img-product").value = "spectrum"
            app.screen.query_one("#img-ok").press()
            await pilot.pause()
            assert app.image_cfg["product"] == "spectrum"

    asyncio.run(scenario())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tui.py::test_planner_settings_can_select_spectrum_network_free -q`

Expected: FAIL because `#img-product` does not exist.

- [ ] **Step 3: Add product/wavelength controls without changing fetch behavior**

Modify `ImageSettingsScreen` in `celestrium/tui/app.py`:

```python
    PRODUCTS = ["colour_image", "multi_panel", "spectrum", "metadata", "exoplanet"]
    WAVELENGTHS = ["auto", "UV", "optical", "near-IR", "mid-IR", "radio", "X-ray"]
```

In `compose`, before FOV:

```python
            yield Label("Product", classes="dim")
            yield Select([(p, p) for p in self.PRODUCTS],
                         value=self._s.get("product", "colour_image"),
                         id="img-product", allow_blank=False)
            yield Label("Wavelength", classes="dim")
            yield Select([(w, w) for w in self.WAVELENGTHS],
                         value=self._s.get("wavelength", "auto"),
                         id="img-wavelength", allow_blank=False)
```

In `on_button_pressed`, extend dismissed config:

```python
        self.dismiss({"fov": fov, "pix": pix,
                      "survey": self.query_one("#img-survey", Select).value,
                      "product": self.query_one("#img-product", Select).value,
                      "wavelength": self.query_one("#img-wavelength", Select).value})
```

In `CelestriumApp.__init__`, change default:

```python
        self.image_cfg = {
            "fov": "auto", "pix": 512, "survey": "auto",
            "product": "colour_image", "wavelength": "auto",
        }
```

- [ ] **Step 4: Run targeted TUI tests**

Run: `python -m pytest tests/test_tui.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/tui/app.py tests/test_tui.py
git commit -m "feat: add planner product controls to TUI"
```

---

### Task 7: TUI Fetch Actions for Colour Image and Spectrum

**Files:**
- Modify: `celestrium/tui/app.py`
- Modify: `tests/test_tui.py`

- [ ] **Step 1: Write failing test for spectrum fetch path with fake backend**

Append to `tests/test_tui.py`:

```python
def test_render_preview_uses_spectrum_backend_for_spectrum_product(tmp_path, monkeypatch):
    from celestrium import spectra

    class Result:
        path = tmp_path / "spec.png"
        summary = "fake spectrum"

    seen = {}

    def fake_fetch(name):
        seen["name"] = name
        Result.path.write_text("fake", encoding="utf-8")
        return Result()

    monkeypatch.setattr(spectra, "fetch_ned_spectrum", fake_fetch)
    app = CelestriumApp(active_line="test-line")
    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    app.image_cfg = {"fov": "auto", "pix": 512, "survey": "auto",
                     "product": "spectrum", "wavelength": "auto"}
    row = {"otype": "QSO"}
    app._render_preview_image("3C 273", row, 187.2, 2.0, "detail")
    assert seen["name"] == "3C 273"
    assert app.last_image == Result.path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tui.py::test_render_preview_uses_spectrum_backend_for_spectrum_product -q`

Expected: FAIL because `_render_preview_image` always calls `cutouts.color_auto`.

- [ ] **Step 3: Route spectrum product to spectra backend**

Modify imports in `celestrium/tui/app.py`:

```python
from .. import cache, candidates, cutouts, packets, planner, registry, spectra, xmatch
```

Modify `_render_preview_image` near the top:

```python
        product = cfg.get("product", "colour_image")
        if product == "spectrum":
            self.call_from_thread(self._dbg, f"spectrum {name}: source=NED")
            result = spectra.fetch_ned_spectrum(name)
            if result is None:
                self.call_from_thread(
                    self._set_detail,
                    f"{detail}\n\n[yellow]No NED spectrum found for {name}.[/]\n"
                    "Try image or metadata products from the planner.")
                return
            self.last_image = result.path
            self.call_from_thread(
                self._set_detail,
                f"{detail}\n\n[b]spectrum[/] {result.source}\n{result.summary}\n"
                f"[dim]{result.path}[/]\n[b green]Ctrl+O[/] open")
            return
```

Keep the existing colour image branch below it.

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_tui.py tests/test_spectra.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/tui/app.py tests/test_tui.py
git commit -m "feat: route TUI spectrum preview through planner backend"
```

---

### Task 8: Resolve Detail Recommendations

**Files:**
- Modify: `celestrium/tui/app.py`
- Modify: `tests/test_tui.py`

- [ ] **Step 1: Write failing pure helper test**

Append to `tests/test_tui.py`:

```python
def test_resolve_detail_includes_planner_recommendations():
    app = CelestriumApp(active_line="test-line")
    detail = app._planner_detail("3C 273", "QSO", 187.2779, 2.0524, "base")
    assert "recommended" in detail.lower()
    assert "NED" in detail or "SDSS" in detail
    assert "HEASARC" in detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tui.py::test_resolve_detail_includes_planner_recommendations -q`

Expected: FAIL because `_planner_detail` does not exist.

- [ ] **Step 3: Implement planner recommendation detail helper**

Add to `CelestriumApp`:

```python
    def _planner_detail(self, name: str, otype: str, ra: float, dec: float, base: str) -> str:
        object_class = planner.classify_otype(otype)
        target = planner.ResolvedTarget(
            display_name=name, aliases=(), ra=ra, dec=dec, otype=otype,
            object_class=object_class, confidence=1.0, match_kind="exact",
        )
        plans = planner.recommend_plans(target)[:5]
        lines = [base, "", "[b]recommended products[/]"]
        for p in plans:
            lines.append(f"- {p.product.label} [{p.product.status}]")
        lines.append("")
        lines.append(planner.image_coverage_note(dec, self.image_cfg.get("survey", "auto")))
        return "\n".join(lines)
```

In `do_resolve`, replace the first detail construction call:

```python
            detail = self._planner_detail(
                str(row["main_id"]), str(row.get("otype", "?")), ra, dec,
                f"[b cyan]{row['main_id']}[/]\n[yellow]{row.get('otype', '?')}[/]\n"
                f"RA {ra:.5f}  Dec {dec:+.5f}")
```

- [ ] **Step 4: Run TUI tests**

Run: `python -m pytest tests/test_tui.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add celestrium/tui/app.py tests/test_tui.py
git commit -m "feat: show planner recommendations in resolve detail"
```

---

### Task 9: Full Verification and Documentation Check

**Files:**
- Modify docs only if behavior differs from the spec.

- [ ] **Step 1: Run full hermetic test suite**

Run: `python -m pytest tests/ -q`

Expected: all tests pass. The existing Windows pytest temp cleanup warning may appear after the test result; note it separately if it recurs.

- [ ] **Step 2: Smoke import the new modules**

Run: `python -c "from celestrium import planner, spectra; print(len(planner.SOURCE_CAPABILITIES))"`

Expected: prints an integer at least `6`.

- [ ] **Step 3: Check git status**

Run: `git status --short --branch`

Expected: clean except unpushed commits.

- [ ] **Step 4: Commit any remaining documentation updates**

If docs changed:

```bash
git add docs celestrium tests
git commit -m "docs: align phase 4 planner notes"
```

- [ ] **Step 5: Present checkpoint summary**

Report:

- commits created
- tests run and results
- live smoke status, if run
- any adapter/source that is metadata-only
- whether the branch is ahead of origin

Do not claim completion without the verification output from Step 1.

---

## Risk Flags

- The branch currently has documentation commits ahead of `origin/main`; decide whether to push before or after implementation.
- NED and SDSS live spectra coverage is target-dependent. Treat no-spectrum as a normal planner result.
- The TUI is already large. Keep new UI state narrow and avoid turning `app.py` into the planner service.
- Do not add heavy dependencies for this phase. If `lightkurve` or other packages become tempting, record them as second-wave.
- Preserve the existing image fallback behavior and test it after adding metadata helpers.
