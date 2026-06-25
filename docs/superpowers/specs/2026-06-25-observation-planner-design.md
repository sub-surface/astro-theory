# Phase 4 Design: Resolve Intelligence and Observation Planner

## Purpose

Phase 4 turns Resolve into Celestrium's front door for object intelligence. The
current TUI can resolve a known object, render a colour cutout, and show related
papers. The next version should accept a broader range of requests, handle fuzzy
or obscure object input, classify the target, and recommend useful evidence
products before fetching data.

The first implementation should advance two tracks in parallel:

1. A planner-backed Resolve experience that understands object identity,
   ambiguity, object class, coverage, and recommended next actions.
2. One complete non-image modality slice: spectra discovery/fetch/render, with
   external artifact opening like image previews.

This is not a full astronomical workbench rewrite. It is a careful extension of
the existing "one brain, two surfaces" architecture.

## User Experience

Resolve remains the main entry point. A user may type:

- A common name: `M87`, `TRAPPIST-1`, `3C 273`.
- An obscure alias or partial identifier.
- Coordinates: `187.7059 +12.3911`.
- A natural-ish object request: `spectrum of M87`, `exoplanets around trappist`,
  `WISE image of 3C 273`.

The TUI should respond with a richer Resolve panel:

- Best identity match, coordinates, object type, and confidence.
- Ambiguity list when the request is not clearly unique.
- Suggested evidence products, grouped by modality.
- Coverage notes before data is fetched.
- Current selected plan: product type, wavelength family, telescope/survey,
  FOV or spectral range where relevant, and output artifact target.
- Deliberate fetch action. Planning should not automatically pull expensive or
  fragile data just because an object was highlighted.

Image artifacts continue to render externally. Spectral artifacts should follow
the same pattern: render to `data/` as a PNG or HTML artifact, show path and
provenance in the TUI, and use the existing external-open behavior.

## Architecture

### New Planner Service

Add a service module such as `celestrium/planner.py` or
`celestrium/observations.py`. This module owns pure planning logic:

- Parse request text into target terms, optional modality hints, and optional
  parameter hints.
- Normalize target descriptions into a `ResolvedTarget` model.
- Represent ambiguity and confidence without hiding uncertainty.
- Classify objects into broad classes:
  - star
  - exoplanet host
  - galaxy or AGN
  - nebula or remnant
  - cluster
  - solar system object, if supported later
  - blank coordinate field
  - unknown
- Return ranked `ObservationPlan` items for available modalities.

The planner should be mostly pure and easy to test. It should not perform heavy
network fetches directly. It may call existing lightweight resolver functions
through injected functions or thin wrappers, but ranking and recommendation text
should be testable with fake resolver results.

### Image Backend

Keep image execution in `celestrium/cutouts.py`.

Changes should be additive:

- Expose structured survey metadata for planner use: key, label, wavelength,
  approximate footprint, depth/resolution note, and product type.
- Reuse `color_auto` for colour cutouts.
- Reuse `panel` for multi-wavelength panels.
- Keep blank/no-coverage fallback behavior intact.

The planner may recommend HST/JWST/MAST for archival observation searches, but
the first implementation should not pretend those are casual HiPS cutouts.

### Spectra Backend

Add `celestrium/spectra.py` for spectrum-related work.

First implementation goals:

- Represent spectral availability as metadata where possible.
- Fetch or accept a spectral table through a small service API.
- Render a simple spectrum artifact:
  - x-axis: wavelength, frequency, or channel, based on available columns.
  - y-axis: flux or intensity.
  - title includes target name and archive/source.
  - output path under `data/spectra/`.
- Return a small result object with path, source, columns used, and provenance.

The first spectra slice should be robust with fake tables and graceful with live
archive failures. It should not attempt redshift fitting, line identification,
continuum modeling, or publication-grade spectral analysis.

The first implementation should include the renderer plus one live adapter behind
the same interface. Prefer an archive already reachable through installed
dependencies, such as an `astroquery` SDSS or VO/VizieR path, and make "no
spectrum available for this target" a normal result rather than an exception.
Hermetic tests must use fake spectral tables; live smoke tests can validate the
chosen adapter on a known target with available spectra.

### TUI Changes

The existing `ImageSettingsScreen` should evolve into a planner-style Resolve
panel or modal. It should support:

- Product type:
  - colour image
  - multi-wavelength panel
  - spectrum
  - archive observations or metadata
  - catalogue rows
  - bibliography
  - exoplanet context
- Wavelength family:
  - UV
  - optical
  - near-IR
  - mid-IR
  - radio
  - X-ray or gamma, if represented as metadata first
- Telescope/survey selection where relevant.
- FOV/pixels for image products.
- Spectral source/range fields only when spectrum is selected.
- Coverage recommendation text.

The UI should avoid an oversized form that is always visible. Prefer progressive
detail: show the recommended plan and the controls that matter for the selected
product. The detail panel can show identity, ambiguity, and suggested actions;
a modal or subpanel can edit the selected plan.

### CLI Compatibility

No full CLI redesign is required in the first pass, but the service layer should
leave clear hooks for later CLI commands:

- `where` can show planner coverage notes.
- `image` can accept structured survey/wavelength options.
- `spectrum TARGET` can call `celestrium/spectra.py`.

Do not import `hub.py` from the TUI or planner. Preserve the existing boundary.

## Data Models

Suggested dataclasses:

- `ParsedRequest`
  - `raw`
  - `target_text`
  - `modality_hint`
  - `parameter_hints`
- `ResolvedTarget`
  - `display_name`
  - `aliases`
  - `ra`
  - `dec`
  - `otype`
  - `object_class`
  - `confidence`
  - `match_kind`
  - `alternatives`
- `ProductCapability`
  - `key`
  - `label`
  - `modality`
  - `wavelength`
  - `source`
  - `coverage_note`
  - `cost`
  - `available`
- `ObservationPlan`
  - `target`
  - `product`
  - `parameters`
  - `recommendation`
  - `warnings`
  - `next_action`
- `ArtifactResult`
  - `kind`
  - `path`
  - `source`
  - `provenance`
  - `summary`

These models should be lightweight dataclasses with `.to_dict()` when useful for
TUI/CLI display.

## Fuzzy Resolution

Resolution should be layered:

1. Exact object resolve through existing SIMBAD/Sesame path.
2. Coordinate parse, if the input resembles coordinates.
3. Alias or relaxed search, where supported by existing resolver tools.
4. Nearby-object search around coordinates.
5. Ambiguous result list when multiple plausible matches exist.
6. Blank-field plan when coordinates are valid but no nearby object is found.

Failure should be informative. For example, "No exact SIMBAD match; try broader
archive search" is better than a generic error. Ambiguity should not be silently
collapsed to the first row unless confidence is high.

## Exoplanet and Other Object Context

Exoplanets should be modeled as a first-class planning category, but not fully
implemented in the first spectra slice.

First-pass behavior:

- Detect likely exoplanet-host objects from object type, known names, or resolver
  metadata where available.
- Recommend exoplanet context as a product type.
- Show that exoplanet-specific pulls are a follow-on action unless a lightweight
  archive helper already exists.

Follow-on helpers may target NASA Exoplanet Archive, TAP/VO tables, or curated
host catalogues. They should follow the same pattern: plan first, fetch
deliberately, cache/provenance where data rows are pulled.

## Error Handling

- Planning failures should return user-facing status objects, not crash the TUI.
- Fetch failures should leave the current plan visible and report the failed
  archive/source.
- Optional dependencies and network services must remain lazy.
- If a product is recommended but not executable yet, the UI should say so
  plainly and offer the nearest executable product.
- Spectra rendering should validate required columns and produce a clear error
  if a table cannot be interpreted.

## Testing

Tests must remain hermetic.

Planner tests:

- Exact object result produces appropriate recommendations.
- Coordinate input produces a blank-field or nearby-object plan.
- Ambiguous/fuzzy results remain visible as alternatives.
- Object classes change modality ranking, for example AGN favors radio/mid-IR
  and spectra; galaxy favors optical/near-IR; blank field favors panel.

Image planner tests:

- Coverage suggestions include the ordered colour survey candidates.
- Forced survey selection remains first but fallback is still possible.
- M31-style blank fallback is not regressed.

Spectra tests:

- Fake spectral table renders a local artifact path.
- Missing spectral columns yields a clear non-crashing error.
- Result object includes source/provenance summary.

TUI tests:

- App mounts with the planner controls.
- Planner state can be updated without network.
- Selecting a spectrum product changes visible controls or state.
- Fetch action can be tested with a fake backend and returns an artifact path.

Manual validation:

- Launch `python -m celestrium.tui`.
- Try `M87`, `3C 273`, `TRAPPIST-1`, M31, and a coordinate-only field.
- Check that image and spectrum artifact paths open externally.

## Scope Boundaries

In scope for first implementation:

- Planner service and models.
- Resolve panel sophistication.
- Fuzzy/ambiguous input handling at a practical first-pass level.
- Image planning with current cutout backend.
- One spectra artifact path with hermetic tests.
- Coverage suggestions before fetch.

Out of scope for first implementation:

- Full spectral science analysis.
- Line fitting, redshift estimation, and model overlays.
- Internal TUI plotting widgets.
- Bulk archive mining.
- Full NASA Exoplanet Archive workflow.
- HST/JWST pixel download workflows.
- Replacing the existing CLI command set.

## Implementation Order

1. Add pure planner models and tests.
2. Expose image capability metadata and coverage recommendation helpers.
3. Add spectra rendering service with fake-table tests.
4. Wire the TUI Resolve panel to planner state without changing fetch behavior.
5. Add deliberate fetch actions for colour cutout and the first spectra artifact.
6. Run full hermetic test suite.
7. Manually smoke-test the TUI in a terminal.

This order validates the shared planning spine before increasing UI complexity,
while still proving that non-image products fit the architecture.
