# CLAUDE.md — onboarding for Celestrium (astro-theory)

Read this first when picking up work here. It's the fast path to oriented.
(Parent context: this is one project inside the `Psychograph/` hub — see `../AGENTS.md`.)

## What this is

**Celestrium** — a three-wing astrophysics instrument for desk-scale research. The premise:
from a desk the binding constraint isn't photons, it's ideas + analysis. Public archives
(Gaia, Euclid, WISE, DESI, Planck) are richer than the community can exploit, so Celestrium
works the idea side across three wings:

1. **Theory workshop** — literature in, ideas assessed (ADS/SciX, object bibliographies, paper
   sets, dossiers, the field-effort census).
2. **Experimental validation** — pull a small data slice, test the empirical claim (ADQL via a
   provenance cache, cross-match audits, row-capped samples).
3. **Imaging & recreation** — the quiet third wing: survey-aware scientific panels, colour
   cutouts, and wallpaper-grade renders (diagrams *and* desktop backgrounds).

Roles: **Leon** = physical judgment (what's worth predicting, which cuts are defensible).
**Claude** = literature throughput, turning signatures into ADQL/Python, the candidate DB and
literature census.

## Repo layout

```
celestrium/   the instrument (Python package — the app's namespace)
  core/       the kernel: artifact · ledger · capability · kernel · events
  caps/       every capability, one module per family
  study/      claims + pipelines + parameter grids (library.py, model.py)
  tap.py      unified Table Access Protocol client (pyvo + driver fallback)
  forecast.py DR1 partial-sky footprint mask, Fisher matrix, harmonic leakage
  mocks.py    hermetic DR1 mock & null Monte Carlo suite
  ellis_baldwin.py pre-registered D_kin expectations for Euclid bands
  cli.py      unified CLI presenter over the DAG kernel; global --json
  hub.py      thin compatibility proxy for legacy test patches
docs/         data-atlas.md · toolbox.md · imaging-guide.md · roadmap.md
  research/   directions · field-map · candidates · theoretical-threads · euclid-dr1-prep
tests/        hermetic CLI + service-layer + kernel tests
Archive/      shelved completed lines (e.g. 2026-06-G-dipole)
data/         git-ignored: celestrium.db (the ledger), artifacts/, cache, reports, posters
refs.bib      bibliography (committed; ads.py appends here)
README.md · CLAUDE.md · AGENTS.md
```

## The kernel (`celestrium/core/`) — read this first

Everything the instrument does is one shape: **take inputs, run a named capability with
parameters, produce a durable artifact, record that it happened.** Five objects:

- `artifact.py` — one typed record for everything produced. **Content-addressed**:
  `id = blake2b(capability, params, inputs)`. So the cache *is* the ledger, `inputs` make it
  a DAG, and the id is the recipe (`celestrium repro <id>`).
- `ledger.py` — `data/celestrium.db`: artifacts · lineage edges · runs · studies. Payloads
  stay on disk under `data/artifacts/`.
- `capability.py` — `@capability(name=…, kind=…, params={…})`. **One declaration.** The CLI,
  methods generator, and agent API all *enumerate the registry*, so a capability cannot exist
  on one surface and be missing from another. Parity is structural.
- `kernel.py` — dedupe → run → persist → record → emit. Retries network capabilities,
  streams events, and generates a methods paragraph from lineage (`celestrium methods <id>`).
- `events.py` — one event model; the CLI renders progress lines.

**Adding a capability is adding one decorated function in `celestrium/caps/`.** Nothing else.
A `Param("artifact")` is special: the kernel folds it into that run's `inputs`, so lineage is
recorded without any capability thinking about it.

```bash
python -m celestrium caps --detail          # the single source of truth
python -m celestrium run analysis.dipole_fit density=<id>
python -m celestrium ledger --lineage <id>  # provenance chain
python -m celestrium study run euclid-dr1-dipole-forecast
```

## The CLI surface (`celestrium/cli.py`)

All real logic lives in `celestrium/`; `celestrium/cli.py` is the unified entry point.
The instrument is 100% headless, fast, scriptable, and agent-driveable via Typer and `--json`.
The bloated Textual TUI and external Obsidian vault sync have been eviscerated.

- `config.py` — data: `ARCHIVES` (gaia/irsa/euclid/heasarc native ADQL + lazy adaptors for
  desi/casda/vizier-tap), `SAMPLE_RECIPES`, `ATLAS_TARGETS`, `RUNBOOKS`.
- `tap.py` — unified Table Access Protocol query and discovery engine.
- `cutouts.py` · `resolvers.py` · `ads.py` · `xmatch.py` · `spectra.py` — thin primitives.
- `cli.py` — Typer CLI, thin presenter; commands grouped by wing in `--help`. Global `--json`.
- `hub.py` — legacy proxy forwarding to `cli.py` to preserve monkeypatching in existing test suites.

```bash
python -m pytest tests/ -q          # hermetic (no network); 115+ tests in ~40s
python -m celestrium --help         # wing-grouped commands
python -m celestrium forecast       # Euclid DR1 gate decision & error budget
python -m celestrium ellis-baldwin  # pre-registered kinematic expectations
python -m celestrium match gaia-bright-nearby vizier:VIII/65/nvss   # X-match audit primitive
```

## Studies (`study/`)

A **study** is a claim, a pipeline, and the grid of cuts you vary against it — the thing that
lets you *"report the amplitude as a function of the cuts, not as a number"*. Pipelines live in
`study/library.py`. `study.run` is itself a capability, so a result surface is a normal artifact
whose lineage children are the individual runs.

Two integrity features: **preregistration** (the pipeline+grid+metric hash is stored on first run
and never silently replaced — changing the analysis after seeing results is flagged in the artifact),
and **cost estimation** (`study estimate` says what will run vs what is already cached, before you commit).

## Active line

**G-Euclid DR1 prep** (`docs/research/euclid-dr1-prep.md`). Euclid **DR1 (~1900 deg² wide)
lands 21 Oct 2026** — first deep optical/NIR sample with a selection function independent of
WISE/Gaia, hence the cleanest cosmic-dipole (isotropy) test.

**Gate decision answered by `celestrium forecast`**:
At ~1,900 deg² (f_sky ~ 4.4% in 3 disjoint patches), **partial-sky harmonic leakage
(σ_leak ≈ 0.0040) dominates over Poisson shot noise (σ_shot ≈ 0.0004)** by an order of magnitude.
Combined σ_total ≈ 0.0040, giving **~1.8σ distinguishability** between CatWISE excess
(D ≈ 0.012) and kinematic expectation (D_kin ≈ 0.0047).
Therefore, **Euclid DR1 is definitively a METHODS DRESS-REHEARSAL & cross-catalogue audit**,
while DR2 provides the full-sky definitive measurement. Pre-DR1 tooling (`forecast.py`,
`mocks.py`, `ellis_baldwin.py`) is fully built and operational.


## Conventions

- **Pull rows, not pixels.** Start `TOP 5`, widen deliberately. Confirm table/column names
  against the live schema (releases drift) — most clients have a `discover()`/`schema()` helper.
- **Provenance by default.** Route data pulls through `cache.cached_query`; the manifest stays complete.
- `data/` is git-ignored. Literature → `refs.bib` (committed).
- **ADS token** (free) enables live `ads.py`/`hub cite`/`papers`: `~/.ads/dev_key` or `$ADS_DEV_KEY`.
  Rubin DP is data-rights-gated — not anonymously queryable.
- Windows + PowerShell host; the Bash tool is available for POSIX scripts.

## Collaboration

Three agents work this repo — Claude, Codex (GPT), and occasionally Gemini — and **all commits
carry the author `Sub-Surface`** (Leon's GitHub account, this machine's git identity), so the
author field does not identify the agent. Check `git log` and the working tree before editing;
session logs live in `docs/devlogs/` (e.g. the 2026-07-01 Gemini TUI rewrite + Codex fix pass).
As of July 2026 Codex is active again (artifact-fallback hardening, Rubin-watch scoping); the
Celestrium reorg, the `tui/commands.py` parser extraction, and the Phase 5 cockpit overhaul are
Claude's.

## Where to go deeper
- `docs/roadmap.md` — the instrument's architecture + the Textual-TUI plan (Phase 2/3).
- `celestrium/README.md` — the query cookbook (one pattern, many dialects) + command reference.
- `docs/imaging-guide.md` — survey-aware cutout decision logic.
- `Archive/2026-06-G-dipole/` — the completed dipole pipeline (M1–M5) we resume against Euclid.
