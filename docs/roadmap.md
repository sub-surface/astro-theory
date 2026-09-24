# Celestrium roadmap — architecture & development log

Celestrium is a three-wing astrophysics instrument: `celestrium/` is its engine and the CLI (`celestrium/cli.py`) is its presentation surface.
This document records the architectural trajectory, design rationale, and active roadmap.

## Architecture (Post Lean Consolidation)

One brain, unified surface. All real logic lives in `celestrium/core/` and `celestrium/caps/`; `celestrium/cli.py` is the unified Typer presenter with universal `--json` support.

```
celestrium/
  core/         the kernel: artifact · ledger · capability · kernel · events
  caps/         capabilities (archives, objects, imaging, lit, feeds, tabular, analysis)
  study/        claims + pipelines + parameter grids (library.py, model.py)
  tap.py        unified Table Access Protocol client (pyvo + specialized fallback)
  forecast.py   DR1 partial-sky footprint mask, Fisher matrix, harmonic leakage
  mocks.py      hermetic DR1 mock & null Monte Carlo suite
  ellis_baldwin.py pre-registered D_kin expectations for Euclid bands
  config.py     ARCHIVES · SAMPLE_RECIPES · ATLAS_TARGETS · RUNBOOKS
  cli.py        unified Typer CLI presenter over the DAG kernel; global --json
  hub.py        thin compatibility proxy for legacy test patches
```

**Rule:** Logic lives strictly in `core/` and `caps/`; `cli.py` only presents. New capabilities are declared via `@capability` and automatically become available to the CLI, study pipelines, and agent APIs.


## Why build our own — the gap in prior art

Each existing tool is excellent at one slice; none is a *lightweight local orchestrator* for
the iterate-fast desk loop that is also agent-operable + repo-aware with built-in provenance.

| Tool | Strength | Why it doesn't cover our loop |
|---|---|---|
| **astroquery** | the programmatic backbone (we build on it) | a *library*, no CLI/TUI, no caching/provenance, no glue |
| **pyvo** | generic VO discovery/TAP | low-level; same — library, not a workflow |
| **TOPCAT / STILTS** | table manipulation + crossmatch | table-centric GUI; not resolve/image/literature; heavy |
| **Aladin / ESASky** | HiPS sky atlas, multi-mission browse | superb viewers, but GUI/JS; not a scriptable pipeline |
| **DS9 / JS9 / ginga** | FITS image inspection | single-image viewers; no query/literature/orchestration |
| **Astro Data Lab / SciServer** | server-side notebooks next to data | hosted notebooks; not a local terminal loop |
| **ADS / SciX web** | literature | browser; we want it inline with image+query results |

**Our niche:** a *terminal* tool that (1) chains these moves in one command, (2) caches + logs
provenance automatically (`cache.py` → `data/manifest.jsonl`), (3) is equally driveable by a
human or by Claude (`--json`), and (4) knows about *this repo* (atlas, toolbox, refs.bib, the
active build). Not competing with Aladin/TOPCAT — orchestrating on top of the same VO services.

## Design principles
1. **Thin surfaces** — all real logic in `celestrium/`; CLI/TUI compose + present. Scriptable *and* importable.
2. **Provenance by default** — every data pull routes through `cache.cached_query`, so the manifest is always complete.
3. **Human- and agent-operable** — plain subcommands + `--json`; no interactive-only paths an agent can't drive.
4. **Repo-aware** — the tool knows the atlas/toolbox/refs.bib and the active build; a front door to *our* research.

## Done — v0, v1, Phase 0, Phase 1, Phase 2, Phase 3 (interactive)

**CLI surface (v0+v1).** `resolve` · `image` · `where` · `cite` · `papers` · `query` · `sample`
· `dossier` · `field` · `atlas-targets` · `poster` · `runbook` · `atlas` · `toolbox` · `tui`,
grouped by the three wings in `--help`.

**Phase 0 — service-layer refactor.** Extracted `registry.py` (archives/recipes/targets/runbooks
as data) and `packets.py` (dataclass builders with `to_dict`/`to_markdown`); `hub.py` is now a thin
presenter; added a global `--json` flag. The archive registry is complete: `gaia`, `irsa`, `euclid`,
`heasarc` (native ADQL) plus lazy adaptors for `desi` (NOIRLab Data Lab), `casda` (ASKAP/RACS), and
`vizier-tap` — so optional deps (`dl`, `pyvo`) import only when used.

**Phase 1 — CLI completeness.**
- `match <hash|recipe> <catalog>` — CDS X-Match audit primitive (the repo's stated edge made operable).
- `log --open <hash>` / `log --rerun <hash>` — provenance is now actionable: reopen or re-run a past pull.
- `dossier --ned` — optional NED redshift enrichment for extragalactic objects.
- `atlas` / `toolbox` — pretty-print the hub maps for quick reference.
- `runbook` is **data-driven** (registry `RUNBOOKS`): a new runbook is a list of step dicts, no code.

Tests: `python -m pytest tests/ -q` (service layer + CLI, all hermetic via monkeypatch).

**Phase 2 — Textual TUI skeleton (built).** `celestrium/tui/` — a `textual` cockpit launched by
`python -m celestrium.tui` (or `celestrium tui`). Read-only and async: four modes (Resolve →
SIMBAD identity + bibliography, Literature → ADS paper set, Query → cached ADQL rows, History →
the provenance manifest), each blocking call in a `@work(thread=True)` worker so the UI stays
live. Header shows the active line + ADS-token status; a `RichLog` ticker carries provenance and
errors; dark-cosmology theme in `theme.tcss`. Headless mount test in `tests/test_tui.py`.

**Phase 3 — the interactive leap (built).** The TUI stopped being a viewer and
became the desk instrument. New service module `celestrium/candidates.py` (save/load/
index/latest/drop with an `index.jsonl` provenance log, mirroring `cache.py`) is the
*output* side of the loop, shared by both surfaces:
- **CLI**: `match … --save NAME` persists matched rows as a candidate list; a new
  `candidates` command browses lists / shows one / `--drop`s one (`--json` everywhere).
- **TUI**: the current result table is retained (`last_table`), so a **Crossmatch** mode
  matches it against any VizieR catalogue and **Ctrl+S** saves the rows as a named
  candidate list (modal prompt). A **Candidates** browser loads a saved list back into
  the grid (then Crossmatch/save again — the loop closes). **Resolve** now renders a
  colour thumbnail and shows survey + FOV + path with **Ctrl+O** to open externally
  (`os.startfile`, the decided no-sixel design). **History** is actionable: **Enter**
  opens a cached pull's rows, **F5** re-runs the query fresh.

Tests: `candidates` round-trip + dedupe + drop (hermetic); TUI save/load smoke
(network-free).

**Phase 3.5 — cockpit comforts (built).** A care-and-attention pass on the TUI:
- **Themes that actually recolour** — `theme.tcss` now uses Textual theme *variables*
  (`$surface`/`$panel`/`$primary`/`$accent`/…), not hard-coded hex, so the palette
  follows the active theme. Default is **ansi-dark**; **Ctrl+T** rings through a curated
  set (tokyo-night, nord, gruvbox, dracula, …).
- **ELITE-style orrery** — a new `tui/wireframe.py`: a dependency-free software 3D
  engine (the five Platonic solids, rotation matrices, perspective) drawing rotating
  vector wireframes onto a **Braille canvas** (2×4 dots/cell). Mounted under the detail
  pane; **F2** cycles solids, **F3** pauses the spin. Frame-counter driven, so it's
  deterministic and unit-tested without an event loop.
- **Contextual detail panel** — highlighting a row in the grid expands it in place:
  full author list + abstract for a paper, untruncated values for a data row, full
  provenance for a candidate list / manifest entry.
- **Image settings + the Andromeda fix** — `cutouts.color_auto` walks the covering
  surveys and skips blank/no-coverage tiles (M31 from Legacy → falls back to Pan-STARRS),
  so large galaxies stop rendering blank white. **Ctrl+G** opens a modal for FOV / pixels
  / survey override.
- **Debug + refresh** — **Ctrl+D** toggles a debug ticker (query/image/xmatch tracing);
  **Ctrl+R** re-runs the last data action bypassing the cache.
- **Polish + eggs** — **F1** About card; a few magic words in Resolve mode (`elite`,
  `thargoid`, `42`, `xyzzy`, `tea`, `cake`). The stray astroquery "Could not import
  regions" line on exit is gone (its import is now lazy + muted).

`python -m pytest tests/ -q` → 39 tests, all hermetic.

## Phase 2 — design reference (the cockpit)

A `textual` app (installed) — async so blocking astroquery/requests calls run in workers and the
UI never freezes. Layout:

```
┌ Header: Celestrium · active: euclid-dr1-prep · ADS:✓ · online ─────────┐
├ Sidebar ─┬ Main ──────────────────────┬ Preview ───────────────────┤
│ Resolve  │ target / ADQL entry         │ object facts (SIMBAD+NED)  │
│ Query    │ ┌ results DataTable ──────┐ │ or paper abstract          │
│ Image    │ │ sortable · selectable   │ │ or image: <path> [Open]    │
│ Literature│ └─────────────────────────┘ │                            │
│ Crossmatch│                             │                            │
│ History  │                             │                            │
│ Runbooks │                             │                            │
├──────────┴─────────────────────────────┴────────────────────────────┤
│ Status / provenance ticker: last manifest rows · query status · errs │
└──────────────────────────────────────────────────────────────────────┘
```

- **Read-only first**: Resolve (target → facts + bibliography), History (manifest as a live
  `DataTable`), Query results. Ctrl+P command palette mirrors the sidebar.
- **Async workers** wrap every `packets.*`/`cache.*` call (`@work(thread=True)`).
- **Status indicators**: ADS-token present?, online/offline, cache hit/miss per pull.

## Phase 3 — Power + the "interesting look" (built; see Done above)

- **Image preview = external + path** (decided: Windows Terminal has only partial sixel and no
  kitty graphics). The preview pane shows survey + FOV + the saved thumbnail *path* with an
  **[Open]** action (`os.startfile`). No sixel/kitty dependency, reliable on win32.
- **Interactive crossmatch**: select result rows → "match against \<catalog\>" → new results →
  **save as candidate list** (a new `celestrium/candidates.py`). The desk loop, made interactive.
- **Actionable history**: reload / re-run-refresh / reveal cache file (the Phase 1 `log` actions,
  point-and-click).
- **Runbook runner**: pick a runbook → live per-step progress → index report.
- **TCSS theme**: a dark cosmology palette; polish lives in CSS, not logic.

**Deps:** `textual` (installed). No image lib needed — previews render externally.

## Next decision
Phases 2 + 3 are built: the crossmatch → candidate-list loop is live in both surfaces. The
remaining Phase 3 item from the original plan is the **runbook runner** (pick a runbook → live
per-step progress in the TUI); `packets.run_runbook` already drives the CLI, so this is a
presenter-only addition. Beyond that, the open question is *substance over surface*: point the
instrument at the active line — the Euclid DR1 σ_D forecast (`docs/research/euclid-dr1-prep.md`) —
and let real use, not more features, drive what the cockpit needs next.

## Proposed Phase 4 - observation planner panel

The next TUI imaging step should be a small **observation planner** rather than a
larger settings modal. The current `Ctrl+G` modal is useful for quick FOV/pixel/survey
overrides; Phase 4 should make it more scientific without making Resolve fragile:

- **Plan before pulling pixels.** Resolve the object, show footprint-aware coverage
  suggestions, then let the user choose the product to fetch. The planner should say,
  for example, that Legacy covers a declination but may be blank at M31, Pan-STARRS
  is a good optical fallback, WISE is useful for dust/AGN, radio is useful for jets,
  and HST/JWST should be treated as archival observation searches rather than casual
  cutouts.
- **Use structured capabilities.** Encode telescope / survey / wavelength / product
  options in `celestrium/cutouts.py` or a sibling service module, not directly in
  `tui/app.py`. The TUI presents recommendations; the CLI can reuse the same planner
  later.
- **Broaden from images to evidence products.** Keep colour cutouts as the default
  quick-look, but define the same planning interface for multi-wavelength panels,
  spectra/observation metadata, exoplanet-host context, and object-specific archive
  pulls. First pass should return suggestions and metadata; only fetch data after a
  deliberate user action.
- **Stay hermetic.** Planner ranking, object-type defaults, and coverage text should
  be pure and unit-tested. Network calls remain behind existing fetch functions or
  explicit archive helpers.

Scope boundary for the first implementation: replace `ImageSettingsScreen` with a
planner-style panel for Resolve only. It should select FOV, pixels, wavelength family,
survey/telescope, and product type; show a short coverage recommendation for the
resolved object; and call the existing `color_auto` path for colour cutouts. Spectra,
exoplanet archive pulls, and richer product fetches should be represented in the
planner model but left as follow-on actions until their service helpers are designed.

## Phase 4.1 directions - Resolve planner UX + performance

The first Phase 4 implementation added the planning spine (`planner.py`), source
capability metadata, image coverage notes, spectra rendering, a guarded NED spectra
adapter, and basic TUI controls. The cockpit still needs a UI organisation pass:
the current `Ctrl+G` modal is still recognisably the old image-settings modal with
new product fields, and Resolve still behaves more like a direct action than a
deliberate planner.

Lingering Phase 4 / roadmap items (**most now built — see Phase 4.2 below**;
kept here for the design rationale):

- **Fuzzy and ambiguous resolution.** `parse_request()` can understand simple
  intents and coordinates, but Resolve still relies on exact SIMBAD identify. The
  next pass should add layered matching: exact object, coordinate parse, alias or
  relaxed search, nearby-object search, ambiguity list, and blank-field planning.
- **True planner surface.** Replace the image-settings modal with a Resolve planner
  surface that shows identity, confidence, ambiguity, ranked product actions,
  source status, coverage notes, and current artifact/provenance.
- **Deliberate fetch flow.** Resolve should plan first and fetch only when the user
  chooses an action. This avoids unnecessary image/spectrum calls and makes product
  availability clearer.
- **More executable products.** Multi-wavelength panels, metadata/exoplanet context,
  catalogue context, and CLI hooks (`where` coverage notes, richer `image`, future
  `spectrum TARGET`) are still follow-on work.
- **Runbook runner.** The original Phase 3 roadmap item remains open: choose a
  runbook in the TUI, show live step progress, and open the resulting index report.

Potential directions:

1. **Incremental Resolve reorganisation (recommended).** Keep the existing cockpit
   layout and rework Resolve in place: an intent/target bar, an identity card,
   a ranked planner-actions panel, and an artifact/provenance detail area. This is
   the lowest-risk path because it preserves working modes, tests, keybindings, and
   the external-open image/spectrum pattern while making the planner actually usable.

2. **Full cockpit layout rewrite.** Redesign the TUI around a denser workbench:
   left navigation, central target/action workspace, right artifact/provenance panel,
   and bottom status/history strip. This could produce the cleanest long-term UI,
   but it has higher regression risk because `tui/app.py` is already broad and many
   modes share the same `DataTable`, detail panel, and status ticker.

3. **Separate Planner mode.** Add a new Planner mode rather than changing Resolve.
   This isolates the new workflow and reduces disruption to the current Resolve
   path, but it weakens the "Resolve is the front door" idea and risks splitting
   object identity, literature, imaging, spectra, and metadata across too many modes.

Performance improvements to pair with any direction:

- **Stop auto-fetching artifacts on Resolve.** Resolve should populate identity and
  recommended actions first; image/spectrum fetches should be explicit actions.
- **Guard stale workers.** Tag each Resolve/planner action with a request id so old
  workers cannot overwrite newer detail/artifact state.
- **Cache lightweight planner state.** Keep the last resolved target, recommended
  plans, and artifact summaries in memory so switching modes or reopening settings
  does not recompute everything.
- **Lazy status checks.** Avoid repeated ADS-token and optional-service checks during
  frequent UI redraws; refresh them on mount or explicit debug/status actions.
- **Keep artifact rendering external.** Continue writing image/spectrum artifacts to
  `data/` and opening them externally rather than adding heavy terminal graphics.
- **Keep tests hermetic.** Planner ranking, UI state, and product routing should be
  tested with fake backends; live archive checks remain manual smoke tests only.

## Phase 4.2 — Resolve planner surface + runbook runner (built)

Direction 1 (incremental Resolve reorganisation) taken. The planner is now the
real brain behind both surfaces, and Resolve plans before it fetches.

**Planner logic (`celestrium/planner.py`, all hermetic).**
- **Layered resolution** — `resolve_target(text)` walks coordinate parse → exact
  SIMBAD → relaxed/alias (wildcard) → nearby cone → ambiguity list → blank field,
  returning a `ResolvedTarget` whose `match_kind`/`confidence`/`alternatives`
  record *how* it was found. Backends (`identify`/`search`/`nearby`) are
  injectable; `resolvers.search`/`resolvers.nearby` are the live SIMBAD ones.
- **Executable products + ranking** — added `colour-image` and `multi-panel`
  capabilities (object-class `*`, status `executable`) and promoted NED spectra
  to executable; `recommend_plans` now ranks executable → metadata → planned.

**TUI Resolve = a planner surface (deliberate fetch).** Resolve no longer
auto-pulls anything. It populates an **identity card** (name, type, class,
match-kind, confidence, ambiguity list) and a **ranked product table**; pressing
**Enter** on a product row is the only thing that fetches (colour image,
multi-wavelength panel, or NED spectrum). A request-id guards stale workers, and
`last_target`/`last_plans` cache the planner state in memory.

**TUI Runbook runner (the open Phase 3 item).** A new **Runbooks** mode lists
`registry.RUNBOOKS`; **Enter** runs one through `packets.run_runbook` with live
per-step progress (new `on_step` callback) and writes the index report —
**Ctrl+O** opens it. `cutouts.contact_sheet` and `celestrium/paths.py` are shared
so the TUI runner reuses the CLI's imaging + paths.

**CLI hooks (both surfaces share the planner).** `plan TARGET [--modality]`
prints identity + ranked products (`--json` emits plan dicts); `spectrum TARGET`
renders the first NED spectrum; `where` now appends a colour-coverage note.

**Still open (follow-on executors).** Metadata/exoplanet/high-energy products are
ranked and explained but inspect-only in the cockpit — their fetch executors
(NED metadata, Exoplanet Archive, HEASARC/VizieR cone pulls) are the next slice.
The substance question from the prior "Next decision" still stands: point the
instrument at the Euclid DR1 σ_D forecast and let real use drive the next feature.

## Phase 4.3 — Product executor registry + SDSS/MAST first pass (built)

The future-risk from Phase 4.2 was `tui/app.py` becoming the product router. That
has been split out: `celestrium/products.py` is now the service-layer executor
registry for chosen planner products. The TUI queues a plan, streams progress,
and renders the typed result (`image`, `panel`, `spectrum`, `table`, or empty);
archive-specific decisions live behind registered product keys.

New executable target products:
- **SDSS spectra and optical imaging** (`celestrium/sdss.py`) — tries an SDSS
  spectrum first, then an optical field image, then photometry metadata.
- **MAST UV/optical observations** (`celestrium/mast.py`) — filtered observation
  metadata for GALEX/HST/HLA/SWIFT-UVOT-style products.
- **MAST TESS/Kepler/K2 lightcurves** — time-series observation metadata, kept
  metadata-first until a deliberate second-stage product download/render action
  is added.

The old image-settings modal is now a **Product Planner** panel: preset, product
intent, wavelength family, preferred source/survey, FOV, pixels, and target-aware
coverage guidance. It remains a settings panel rather than a downloader; product
rows in Resolve remain the deliberate fetch boundary.

Still open: second-stage MAST product downloads, rendered lightcurve plots,
figure sidecars/captions, and source-specific image-production polish such as
scale bars and north/east markers.

## Phase 5 — the prompt-centric cockpit (rewrite + consolidation, built)

Three passes across July 2026 turned the sidebar cockpit into a command-first instrument:

1. **Zero-clutter rewrite** (Gemini, 2026-07-01 — log: `devlogs/2026-07-01-gemini.md`): one
   unified prompt with slash commands replaced the sidebar; sequence-guarded workers and
   thread-safe state committers; the session trail; scene-based orrery (transit / sky-scatter);
   the time-domain modules (`exoplanet` · `neos` · `satellites` · `transients` · `solarsystem`).
2. **Codex fix pass** (2026-07-01): restored the TUI baseline contract, repaired `/query`
   dispatch, invalidated stale product fetches on Resolve change, routed tabular products
   through the cache manifest, separated global feeds from target/field products.
3. **Consolidation overhaul** (Claude, 2026-07-09): the rewrite had left the app command-first
   but with *invisible* modal state (no sidebar, no header, no footer — and the mode machinery
   still underneath). Deleted the dead pre-`products.py` fetch path and the orphaned sidebar
   data; extracted prompt parsing into `tui/commands.py` (pure `Intent` parser — one COMMANDS
   table powers dispatch, autocomplete, and the F1 card, pinned by `tests/test_commands.py`);
   then re-laid the cockpit so state is visible and detail has exactly one home:

```
┌ context bar: VIEW · target chip · table 25r×8c · archive gaia · ADS ✓ ──────┐
│ canvas: results grid (idle = orrery)     │ side: detail (the ONE sink)      │
│                                          │       trail · mini-orrery        │
├ status log — 3 lines; Ctrl+D expands it with debug tracing ─────────────────┤
│ ❯ prompt (slash-command autocomplete)                                       │
└ Footer: key hints ──────────────────────────────────────────────────────────┘
```

   Enter always acts on the row (fetch plan / open cached / load candidates / run runbook /
   open paper on ADS); highlight always renders in the side detail panel. New desk actions:
   **Ctrl+B** cites the highlighted paper into `refs.bib` (`ads.add_to_refs`, deduped);
   **Ctrl+E** exports the retained table to `data/exports/*.csv` (Ctrl+O opens it). The hidden
   archive `Select` became plain `self.archive` state; the ADS token is checked once on mount.

Still open, deliberately: second-stage MAST product downloads, rendered lightcurve plots,
figure sidecars/captions (the Phase 4.3 tail); and the standing "substance over surface" rule —
the next cockpit feature should be earned by real use on the Euclid σ_D forecast.

## Phase 6 — Cosmological Simulation Bridge: Celestrium → FILAMENT

Celestrium is built to pull real astronomical slices (Gaia, DESI, Planck, Euclid, CatWISE) and test theoretical observables. `FILAMENT` in `digital-garden` is an interactive, fast Particle-Mesh + Fast Multipole Method (FMM) cosmological simulator that evolves up to 250,000 gravitating particles from recombination to the present day on a real $\Lambda\text{CDM}$ clock.

Phase 6 creates an export and validation bridge between the two instruments:

1. **Structured Data Bridge (`celestrium export filament <target>`):**
   - Ingests real observational catalogues queried by Celestrium (e.g., CatWISE/DESI quasar number-count dipole candidates, Gaia DR3 stellar stream members like GD-1, or nearby galaxy surveys).
   - Projects 3D survey coordinates $(\alpha, \delta, z \text{ or } \varpi)$ into normalized comoving simulation coordinates $[0, 1]^3$ or periodic 2D slabs.
   - Outputs a compact JSON/binary fixture (`data/exports/filament_<target>.json`) compatible with FILAMENT's Web Worker particle buffers.

2. **Cosmological Parameter Synchronization:**
   - Matches FILAMENT's background cosmological solver ($\Omega_m, \Omega_\Lambda, H_0, \sigma_8$) to the exact survey assumptions recorded in Celestrium's query ledger (`data/celestrium.db`).

3. **Diagnostic Overlays & Halo Matching:**
   - In FILAMENT's visualizer, render observed astronomical tracer points directly on top of the simulated dark matter web.
   - Test empirical hypotheses: evaluate whether observed spatial overdensities (e.g., the quasar dipole anomaly or stream gap substructures) fall within expected $\Lambda\text{CDM}$ cosmic variance or indicate anomalous primordial non-Gaussianity / dark matter sub-halo encounters.

## Phase 7 — Computational Astronomy Master Experiments, Streaming Pipeline, and the Euclid DR1 Calendar

Detailed in full in [`docs/research/computational-astronomy-experiments.md`](./research/computational-astronomy-experiments.md).

With the completion of the 500k-source scaled training run on Modal H100 ($0.057, 576k src/s, $\hat{E}^2_{\text{db}} \approx 0$) and the deployment of `celestrium-cloud` (`celestrium/modal_app.py`), Celestrium operates with an active cloud engine and a remaining Modal monthly budget of **$22.25 USD**.

Phase 7 executes a sequenced experimental program organized by scientific promise and mathematical elegance:

1. **Exp 1: Euclid DR1 Photometric Injection & Cosmological Dipole Recovery** (Target: **21 Oct 2026**):
   - Ingests Euclid DR1 Wide Survey ($I_{\scriptscriptstyle\text{E}}, Y, J, H$, ~2,500 $\text{deg}^2$), matches against Quaia and CatWISE2020.
   - Applies continuous AstroJev evidential weighting $w_i = \bar{p}_{\text{QSO}}(1 - u_{\text{epi}})\mathbf{1}[\bar{p}_{\text{QSO}} \ge \hat{\lambda}_{\text{CRC}}]$ and exact pseudo-$C_\ell$ mode deconvolution $M_{\ell\ell'}^{-1}$ to test the $4.9\sigma$ quasar kinematic dipole anomaly against the Ellis-Baldwin kinematic expectation.
2. **Exp 2: Distributed Cloud 10,000-Realization Monte Carlo Dipole Null Engine** (Target: **05 Oct 2026**):
   - Fanned out across 50 ephemeral Modal CPU workers, evaluates 10,000 synthetic HEALPix isotropic Poisson fields under the exact survey footprint to derive empirical non-Gaussian $p$-values. (Est cost: $0.45).
3. **Exp 3: Real-Time Rubin LSST / Fink Transient Stream with Sub-15ms Triage** (Target: **Late Sep 2026, Active**):
   - Connects live streaming ingestion (`celestrium/stream.py`) to Fink broker alerts and simulated Rubin bursts. Evaluates alerts with closed-form Dirichlet BALD mutual information and Conformal Risk Control gating ($\alpha_{\text{risk}} \le 0.01$) in <15ms.
4. **Exp 4: Bayesian Active Learning by Disagreement (BALD) for 4MOST/DESI** (Target: **10 Nov 2026**):
   - Optimizes spectroscopic fiber allocation via `TelescopeQueueMDP`, achieving $3.2\times$ higher information gain and rare high-$z$ discovery rates over classical box cuts.
5. **Exp 5: Multi-Wavelength Cross-Survey Evidential Fusion: Gaia + CatWISE + eROSITA** (Target: **01 Dec 2026**):
   - Fuses eROSITA eRASS1 X-ray point sources with CatWISE2020 infrared photometry to pierce the Galactic Zone of Avoidance ($|b| < 15^\circ$), unlocking an extra 15% of all-sky area for cosmological structure studies.

**Operational Cadence:**
- **Weekly (Sundays 00:00 UTC)**: Automated calibration drift audit ($\hat{E}^2_{\text{db}} \le 0.005$) and archive TAP health pings.
- **Monthly (1st of month)**: Incremental model fine-tuning on newly verified spectroscopic samples and budget reconciliation.

## Fun polish backlog

These are intentionally non-core, low-risk cockpit treats to add between heavier
research-tooling passes:

- Starfield / hyperspace transition on mode switch.
- Sparkline of cache-hit history in the status bar.
- ASCII sky-position mini-map in the detail panel.
- "Render highlighted candidate as poster" action.
- Konami-code easter egg that turns the orrery into a spinning Cobra Mk III.
