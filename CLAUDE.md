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
  study/      claims + pipelines + parameter grids
  vault/      the Obsidian bridge (Psychograph-Vault/Astronomy/)
docs/         data-atlas.md · toolbox.md · imaging-guide.md · roadmap.md · rebuild-2026-07.md
  research/   directions · field-map · candidates · theoretical-threads · euclid-dr1-prep  (theory wing)
  devlogs/    per-session agent logs (e.g. 2026-07-01-gemini.md)
scripts/      arxiv_tally.py · fetch_papers.sh   (field-cartography utilities)
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
  the palette, the planner ranking and the JSON/agent API all *enumerate the registry*, so a
  capability cannot exist on one surface and be missing from another. Parity is structural.
- `kernel.py` — dedupe → run → persist → record → emit. Retries network capabilities,
  streams events, and generates a methods paragraph from lineage (`celestrium methods <id>`).
- `events.py` — one event model; the CLI renders progress lines, a TUI would render cards.

**Adding a capability is adding one decorated function in `celestrium/caps/`.** Nothing else.
A `Param("artifact")` is special: the kernel folds it into that run's `inputs`, so lineage is
recorded without any capability thinking about it.

```bash
python -m celestrium caps --detail          # the single source of truth
python -m celestrium run analysis.dipole_fit density=<id>
python -m celestrium ledger --lineage <id>  # provenance chain
python -m celestrium study run euclid-dr1-dipole-forecast
python -m celestrium vault sync <study>     # results → the Observables note
```

## The older surfaces (`hub.py`, `tui/`)

**One brain, two surfaces:** all real logic lives in `celestrium/`; the CLI and TUI
only present. These predate the kernel and still run on `cache.py`/`packets.py`/`products.py`;
they work, and the TUI rebuild onto the kernel is the next phase (`docs/rebuild-2026-07.md`).

- `registry.py` — data: `ARCHIVES` (gaia/irsa/euclid/heasarc native ADQL + lazy adaptors for
  desi/casda/vizier-tap), `SAMPLE_RECIPES`, `ATLAS_TARGETS`, `RUNBOOKS`.
- `packets.py` — builders returning dataclasses with `.to_dict()` (JSON/TUI) + `.to_markdown()` (reports).
- `cache.py` — `cached_query` + `data/manifest.jsonl` provenance + `load_cached`/`find_record` (by hash).
- `cutouts.py` · `resolvers.py` · `ads.py` · `xmatch.py` · `<archive>.py` — thin primitives.
- `hub.py` — Typer CLI, a thin presenter; commands grouped by wing in `--help`. Global `--json`.
- `tui/` — Textual cockpit (`python -m celestrium.tui`), **prompt-centric** (Phase 5): one REPL
  prompt with slash-command autocomplete; `tui/commands.py` parses text → `Intent` (pure,
  unit-tested — a single COMMANDS table powers dispatch, autocomplete, and the F1 card). A
  context bar shows view · target · retained table · archive · ADS. A bare name resolves;
  `gaia: SELECT…` queries; `match <cat>` crossmatches the retained table; `run <rb>` executes a
  runbook; `/global` browses live feeds (neos/satellites/transients). **Resolve is a planner
  surface**: `planner.py` ranks products, `products.py` executes the chosen one (colour image,
  multi-panel, spectra, SDSS/MAST, exoplanet, HEASARC, ephemeris…) — fetch happens only when you
  Enter a product row. Highlight always renders in the side detail panel; Enter always acts on
  the row (open cached / load candidates / run runbook / open paper on ADS). Desk actions:
  Ctrl+S save candidates · Ctrl+B cite highlighted paper → refs.bib · Ctrl+E export table → CSV.

**Rule:** `hub.py` and `tui/` import `celestrium/*`, never each other. New behaviour goes in
`registry.py`/`packets.py` so both surfaces get it. The CLI tests monkeypatch names ON the
`hub` module (`hub.QUERY_ARCHIVES`, `hub.cutouts`, `hub._contact_sheet`, `hub.REPORTS_DIR`…);
`hub.py` re-exports registry objects and packets call modules by attribute so patches flow —
keep that (don't bind functions by value at import). Tests must stay hermetic (no network).

```bash
python -m pytest tests/ -q          # hermetic (no network); 257 tests as of 2026-07-30
python -m celestrium --help         # wing-grouped commands
python -m celestrium runbook list
python -m celestrium match gaia-bright-nearby vizier:VIII/65/nvss   # X-match audit primitive
```

## Studies + the vault (`study/`, `vault/`)

A **study** is a claim, a pipeline, and the grid of cuts you vary against it — the thing that
lets you *"report the amplitude as a function of the cuts, not as a number"*. Claims live in
`Psychograph-Vault/Astronomy/Observables/*.md`; pipelines live in `study/library.py`; **they
bind by `id`** and neither duplicates the other. `study.run` is itself a capability, so a
result surface is a normal artifact whose lineage children are the individual runs.

Two integrity features worth knowing about: **preregistration** (the pipeline+grid+metric hash
is stored on first run and never silently replaced — changing the analysis after seeing results
is flagged in the artifact), and **cost estimation** (`study estimate` says what will run vs
what is already cached, before you commit).

Vault writes are surgical: known frontmatter keys plus `celestrium_*`, and body content only
inside `<!-- celestrium:begin/end -->`. **Prose is never touched.** Generated notes go to
`Log/runs/`, `Objects/generated/`, `Fields/generated/`, `_attachments/` — all gitignored in the
vault (`celestrium vault init`), so machine output never lands in a public commit.

## Active line

**G-Euclid DR1 prep** (`docs/research/euclid-dr1-prep.md`). Euclid **DR1 (~1900 deg² wide)
lands 21 Oct 2026** — first deep optical/NIR sample with a selection function independent of
WISE/Gaia, hence the cleanest cosmic-dipole (isotropy) test. The earlier dipole work (project
G, milestones M1–M5 across Quaia/CatWISE/NVSS) is shelved in `Archive/2026-06-G-dipole/`; its
maths now lives as capabilities in `caps/analysis.py`.

**First action answered (2026-07-30): measure, don't rehearse.**
`celestrium study run euclid-dr1-dipole-forecast` injects a known dipole on a ~4.6%-sky,
|b|>25° DR1-like footprint and measures it back. On 10⁶ sources, **σ_D ≈ 0.0017** — against a
kinematic prediction of 0.0061 (`analysis.kinematic_dipole`) and a claimed ~2× excess of
0.0122, which recovers at **7.4σ** (5.6σ at 5×10⁵). Caveat: the forecast assumes a uniform
selection function within the footprint, which is exactly the systematic the real analysis has
to fight — so treat σ_D as a floor, not a promise.

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
