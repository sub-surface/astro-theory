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
docs/         data-atlas.md · toolbox.md · imaging-guide.md · roadmap.md
  research/   directions.md · field-map.md · candidates.md · euclid-dr1-prep.md  (theory wing)
scripts/      arxiv_tally.py · fetch_papers.sh   (field-cartography utilities)
tests/        hermetic CLI + service-layer tests
Archive/      shelved completed lines (e.g. 2026-06-G-dipole)
data/         git-ignored: cache, manifest, reports, atlas, posters
refs.bib      bibliography (committed; ads.py appends here)
README.md · CLAUDE.md · AGENT.md
```

## The instrument (`celestrium/`)

**One brain, two surfaces:** all real logic lives in `celestrium/`; the CLI and future TUI
only present.

- `registry.py` — data: `ARCHIVES` (gaia/irsa/euclid/heasarc native ADQL + lazy adaptors for
  desi/casda/vizier-tap), `SAMPLE_RECIPES`, `ATLAS_TARGETS`, `RUNBOOKS`.
- `packets.py` — builders returning dataclasses with `.to_dict()` (JSON/TUI) + `.to_markdown()` (reports).
- `cache.py` — `cached_query` + `data/manifest.jsonl` provenance + `load_cached`/`find_record` (by hash).
- `cutouts.py` · `resolvers.py` · `ads.py` · `xmatch.py` · `<archive>.py` — thin primitives.
- `hub.py` — Typer CLI, a thin presenter; commands grouped by wing in `--help`. Global `--json`.
- `tui/` — Textual cockpit (`python -m celestrium.tui`). **Phase 2 skeleton built**: read-only +
  async, modes Resolve/Literature/Query/History. Phase 3 (interactive crossmatch, image preview,
  runbook runner) in `docs/roadmap.md`.

**Rule:** `hub.py` and `tui/` import `celestrium/*`, never each other. New behaviour goes in
`registry.py`/`packets.py` so both surfaces get it. The CLI tests monkeypatch names ON the
`hub` module (`hub.QUERY_ARCHIVES`, `hub.cutouts`, `hub._contact_sheet`, `hub.REPORTS_DIR`…);
`hub.py` re-exports registry objects and packets call modules by attribute so patches flow —
keep that (don't bind functions by value at import). Tests must stay hermetic (no network).

```bash
python -m pytest tests/ -q          # 23 tests, hermetic
python -m celestrium --help         # wing-grouped commands
python -m celestrium runbook list
python -m celestrium match gaia-bright-nearby vizier:VIII/65/nvss   # X-match audit primitive
```

## Active line

**G-Euclid DR1 prep** (`docs/research/euclid-dr1-prep.md`). Euclid **DR1 (~1900 deg² wide)
lands 21 Oct 2026** — first deep optical/NIR sample with a selection function independent of
WISE/Gaia, hence the cleanest cosmic-dipole (isotropy) test. The earlier dipole work (project
G, milestones M1–M5 across Quaia/CatWISE/NVSS) is shelved in `Archive/2026-06-G-dipole/` — the
pipeline we resume against Euclid. First action: forecast σ_D on a realistic DR1 footprint to
decide "measure vs. rehearse". The fuller backlog (A–H, leverage thesis, effort map) is in
`docs/research/`.

## Conventions

- **Pull rows, not pixels.** Start `TOP 5`, widen deliberately. Confirm table/column names
  against the live schema (releases drift) — most clients have a `discover()`/`schema()` helper.
- **Provenance by default.** Route data pulls through `cache.cached_query`; the manifest stays complete.
- `data/` is git-ignored. Literature → `refs.bib` (committed).
- **ADS token** (free) enables live `ads.py`/`hub cite`/`papers`: `~/.ads/dev_key` or `$ADS_DEV_KEY`.
  Rubin DP is data-rights-gated — not anonymously queryable.
- Windows + PowerShell host; the Bash tool is available for POSIX scripts.

## Collaboration

Codex (GPT) has also committed here (author `Sub-Surface`, `feat:` prefix). Two agents touch
this repo — check `git log`/working tree before editing. As of late June 2026 Codex was on a
reheating-in-inflation thread (project **F**, the Copeland GW scorecard) but stopped; the hub
refactor + Celestrium reorg are mine.

## Where to go deeper
- `docs/roadmap.md` — the instrument's architecture + the Textual-TUI plan (Phase 2/3).
- `celestrium/README.md` — the query cookbook (one pattern, many dialects) + command reference.
- `docs/imaging-guide.md` — survey-aware cutout decision logic.
- `Archive/2026-06-G-dipole/` — the completed dipole pipeline (M1–M5) we resume against Euclid.
