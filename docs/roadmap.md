# Celestrium roadmap — architecture & the Textual-TUI leap

Celestrium is a three-wing astrophysics instrument; `celestrium/` is its engine and the CLI/TUI are its surfaces.
This is the single planning doc for that tool (it replaces the old `tui-scope.md`): the design
rationale, what's built, and the Textual-TUI leap still ahead.

## Architecture (post Phase 0 refactor)

One brain, two surfaces. All real logic lives in `celestrium/`; the CLI and the future TUI only
*present*.

```
celestrium/
  registry.py   data: ARCHIVES · SAMPLE_RECIPES · ATLAS_TARGETS · RUNBOOKS
  packets.py    builders: PaperSet · ObjectPacket · FieldPacket · RunbookResult
                (each .to_dict() for JSON/TUI, .to_markdown() for reports)
  cache.py      cached_query + manifest provenance + load_cached/find_record (by hash)
  cutouts · resolvers · ads · xmatch · <archive>.py   ← thin primitives
  hub.py        Typer CLI: a thin presenter over registry + packets (--json everywhere)
  tui/          Textual app (Phase 2/3) — a thin presenter over the SAME registry + packets
```

**Rule:** `hub.py` and `tui/` may import `celestrium/*` but never each other. A behaviour worth
having lives in `registry.py`/`packets.py`, and both surfaces get it for free.

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

## Done — v0, v1, Phase 0, Phase 1, Phase 2 (skeleton)

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

## Phase 3 — Power + the "interesting look"

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
Build Phase 2 read-only first and live with it before adding Phase 3 interactivity. The crossmatch
loop (Phase 3) is where the TUI stops being a viewer and becomes the desk instrument — but it should
ride on a TUI skeleton that already feels right.
