# Hub CLI / TUI — scope, prior art, roadmap

A thin, scriptable, *documented* terminal tool that orchestrates the repo's
`access/` layer (query → cache → cross-match → resolve → image → cite) so both
Leon and Claude can drive the everyday research loop from one place. v0 exists:
[`access/hub.py`](./access/hub.py) (`python -m access.hub --help`).

## Why build our own — the gap in prior art

Each existing tool is excellent at one slice; none is a *lightweight local
orchestrator* for the iterate-fast desk loop, and none is agent-operable + repo-aware
with built-in provenance.

| Tool | Strength | Why it doesn't cover our loop |
|---|---|---|
| **astroquery** | the programmatic backbone (we build on it) | a *library*, no CLI/TUI, no caching/provenance, no glue |
| **pyvo** | generic VO discovery/TAP | low-level; same — library, not a workflow |
| **TOPCAT / STILTS** | table manipulation + crossmatch (STILTS scriptable) | table-centric GUI; not resolve/image/literature; heavy |
| **Aladin Desktop / Lite** | HiPS sky atlas, footprint/catalogue overlay | superb viewer, but GUI/JS; not a scriptable pipeline |
| **ESASky** | multi-mission image + catalogue browser | web GUI; great for *looking*, not for repeatable scripts |
| **DS9 / JS9 / ginga** | FITS image inspection | single-image viewers; no query/literature/orchestration |
| **Astro Data Lab / SciServer** | server-side notebooks next to data | hosted notebooks; not a local terminal loop |
| **ADS / SciX web** | literature | browser; we want it inline with image+query results |

**Our niche:** a *terminal* tool that (1) chains these moves in one command, (2) caches
+ logs provenance automatically (`access/cache.py` → `data/manifest.jsonl`), (3) is
equally driveable by a human or by Claude, and (4) knows about *this repo* (atlas,
toolbox, refs.bib, the active build). Not competing with Aladin/TOPCAT — orchestrating
on top of the same VO services they use.

## Features worth borrowing (improve, don't reinvent)
- **Aladin** — progressive HiPS rendering (we already use HiPS via `cutouts.color`);
  footprint/MOC overlays (pairs with MOCpy in `toolbox.md`).
- **TOPCAT** — its crossmatch ergonomics → wrap `access/xmatch.py` as a `hub match`.
- **ESASky** — multi-mission overlay → our multi-λ `panel` is the terminal analogue.
- **ADS libraries** — saved paper sets → our `refs.bib` is the file-based version.

## Roadmap

**v0 — CLI (done).** typer + rich. `resolve` (SIMBAD id + bibliography), `image`
(smart multi-λ + colour cutout), `where` (footprint + nearest object), `cite` (ADS
search → refs.bib).

**v1 — CLI breadth.** Implemented:
- `query <archive> "<ADQL>"` — run against supported ADQL archives (`gaia`, `irsa`,
  `euclid`, `heasarc`), **auto-cached** via `cache.cached_query`, result as a rich table.
- `log` — browse `data/manifest.jsonl` (what we've pulled, when) as a table.
- `papers <ADS query>` — literature packet with exact-phrase helper, optional BibTeX append,
  and Markdown report output.
- `dossier <target>` — object packet: SIMBAD resolve, survey-aware image/panel, ADS trail,
  Markdown report.
- `field <ra> <dec>` — blank-field or coordinate packet: nearest object, best survey,
  image/panel, Markdown report.
- `sample <recipe>` — row-capped science pulls through the cache.
- `atlas-targets` — curated visual contact sheet of astronomy targets.
- `poster <target>` — wallpaper-style HiPS render at 1080p, 2k, or 4k.

Still to add:
- `match <table> <catalog>` — CDS X-Match wrapper (the audit primitive).
- `atlas` / `toolbox` — pretty-print the hub maps for quick reference.

**v2 — Textual TUI.** A `textual` app (needs `pip install textual`) with panes:
target entry · results table · image preview (open externally, or sixel/kitty in
capable terminals) · literature · query history from the manifest. Live, exploratory,
still thin over `access/`.

## Design principles
1. **Thin wrappers** — all real logic stays in `access/` modules; the CLI/TUI only
   composes + presents. Keeps it scriptable *and* importable.
2. **Provenance by default** — every data pull routes through `cache.cached_query`,
   so the manifest is always complete without thinking about it.
3. **Human- and agent-operable** — plain subcommands + structured output; no
   interactive-only paths that an agent can't drive.
4. **Repo-aware** — the tool knows the atlas/toolbox/refs.bib and the active build,
   so it's a front door to *our* research, not a generic client.

## Next decision
Finish v1 with `match` plus `atlas` / `toolbox` quick-reference commands. The Textual
TUI is the fun leap once the CLI surface has settled — defer until v1 commands feel
right in daily use.
