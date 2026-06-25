# CLAUDE.md — onboarding for astro-theory

Read this first when picking up work in this repo. It's the fast path to oriented.
(Parent context: this is one project inside the `Psychograph/` hub — see `../AGENTS.md`.)

## What this repo is

**Theory-driven candidate-generation for desk cosmology & astronomy.** The premise: from a
desk the binding constraint isn't photons, it's ideas + analysis. Public archives (Gaia, WISE,
Euclid, DESI, Planck) are richer than the community can exploit, so we work the literature side:
turn a theoretical signature → selection cuts → a *small* local pull (a few thousand rows via
TAP/ADQL, not a petabyte) → ranked candidate list → triage against follow-up papers → iterate.

Roles: **Leon** = physical judgment (what's worth predicting, which cuts are defensible).
**Claude** = literature throughput, turning signatures into ADQL/Python, bookkeeping, the
candidate DB and literature census.

## The four standing pillars (read in this order)

1. **`data-atlas.md`** — *what data exists*: every publicly queryable archive, ranked by quality
   and by leverage, + the 2026–27 release calendar.
2. **`access/`** — *how to pull it* (a Python package). See `access/README.md`.
3. **`toolbox.md`** — *everything around the query*: ADS/SciX, CDS X-Match, MOCpy/healpy, dustmaps.
4. **`directions.md`** + **`field-map.md`** + **`euclid-dr1-prep.md`** — *what to do with it*: the
   leverage thesis (effort × tractability), the data-backed field map, and the live build.

When Leon asks "what could we work on / what data can we touch", consult `data-atlas.md` +
`directions.md`/`field-map.md` first. Our edge: cross-catalogue consistency audits,
selection-function re-analysis of live anomalies, living meta-analyses — not new photons.

## Active line

**G-Euclid DR1 prep** (`euclid-dr1-prep.md`). Euclid **DR1 (~1900 deg² wide) lands 21 Oct 2026** —
the first deep optical/NIR sample with a selection function independent of WISE/Gaia, hence the
cleanest available cosmic-dipole (isotropy) test. The earlier dipole work (project G, milestones
M1–M5 across Quaia/CatWISE/NVSS) is shelved in `Archive/2026-06-G-dipole/` — not abandoned; it's
the pipeline we resume against Euclid. First action: forecast σ_D on a realistic DR1 footprint to
decide "measure vs. rehearse".

## The hub tool (`access/`)

A data-access layer + an operable front-end (**astro-hub**). **One brain, two surfaces:** all real
logic lives in `access/`; the CLI and the future TUI only present.

- `registry.py` — data: `ARCHIVES` (gaia/irsa/euclid/heasarc native ADQL + lazy adaptors for
  desi/casda/vizier-tap), `SAMPLE_RECIPES`, `ATLAS_TARGETS`, `RUNBOOKS`.
- `packets.py` — builders returning dataclasses with `.to_dict()` (JSON/TUI) + `.to_markdown()` (reports).
- `cache.py` — `cached_query` + `data/manifest.jsonl` provenance + `load_cached`/`find_record` (by hash).
- `cutouts.py` · `resolvers.py` · `ads.py` · `xmatch.py` · `<archive>.py` — thin primitives.
- `hub.py` — Typer CLI, a thin presenter. `python -m access.hub --help`. Add `--json` for machine output.
- `tui/` — Textual cockpit, **not built yet** (Phase 2/3). Full plan in `roadmap.md`.

**Rule:** `hub.py` and `tui/` import `access/*`, never each other. New behaviour goes in
`registry.py`/`packets.py` so both surfaces get it.

```bash
python -m pytest tests/ -q          # 23 tests, hermetic (monkeypatched, no network)
python -m access.hub runbook list   # data-driven runbooks
python -m access.hub match gaia-bright-nearby vizier:VIII/65/nvss   # X-match audit primitive
```

## Conventions

- **Pull rows, not pixels.** Start `TOP 5`, widen deliberately. Confirm table/column names against
  the live schema (releases drift) — most clients have a `discover()`/`schema()` helper.
- **Provenance by default.** Route data pulls through `cache.cached_query`; the manifest stays complete.
- `data/` is git-ignored (cache, reports, atlas, posters, candidate DBs). Literature → `refs.bib` (committed).
- **ADS token** (free) enables live `ads.py`/`hub cite`/`papers`: paste into `~/.ads/dev_key` or
  `$env:ADS_DEV_KEY`. Rubin DP is data-rights-gated — not anonymously queryable.
- Tests must stay hermetic: monkeypatch `hub.*`/module attrs; never hit the network in a unit test.
- Windows + PowerShell host; the Bash tool is available for POSIX scripts.

## Collaboration

Codex (GPT) has also committed here (author `Sub-Surface`, `feat:` prefix). Two agents touch this
repo — check `git log`/working tree before editing so you don't clobber in-flight work. As of late
June 2026 Codex was on a reheating-in-inflation thread (project **F**, the Copeland GW scorecard).

## Where to go deeper

- `roadmap.md` — the hub tool's architecture + the Textual-TUI plan (Phase 2/3).
- `access/README.md` — the query cookbook (one pattern, many dialects) + hub command reference.
- `imaging-guide.md` — survey-aware cutout decision logic.
- `Archive/2026-06-G-dipole/` — the completed dipole pipeline (M1–M5) we resume against Euclid.
