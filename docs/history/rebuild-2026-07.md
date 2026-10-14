# The kernel rebuild (2026-07-30)

Why the instrument was rebuilt around a kernel, what landed, and what's left.
Companion to `feature-review-2026-07.md`, which catalogued *what was missing*; this
argues those were symptoms of one structural gap, and records the fix.

---

## 1. The diagnosis

**The instrument could not compute.** `Archive/2026-06-G-dipole/` held 1,586 lines of real
science — HEALPix maps, selection masks, dipole fits, null tests — and imported `celestrium`
**zero times**. When the actual research happened, it happened outside the tool. Celestrium
was an excellent acquisition layer with no analysis layer; `plots.py` was presentation, not
inference.

The vault had already specified the missing feature. `Observables/The quasar number-count
dipole.md`: *"vary, one at a time: flux limit, mask width… Report the dipole amplitude as a
**function of the cuts**, not as a number."* That sentence is a parameter grid over a
pipeline with a result surface — and there was no pipeline object, no parameter object, and
no result surface.

**Every result was a special case.** `tui/app.py`: 2,106 lines, one class, 143 methods, 18
`_accept_*` handlers, 75 `call_from_thread` calls, and a hand-rolled `_seq` race guard. That
guard existed because one grid and one detail sink meant every result competed for the same
slot. Adding one capability cost edits in ~7 places across 9 files — which is why parity kept
breaking.

**Five registries wanted to be one**: `ARCHIVES` / `SAMPLE_RECIPES` / `RUNBOOKS` /
`GLOBAL_FEED_EXECUTORS` (registry.py) + `SOURCE_CAPABILITIES` (planner.py) +
`PRODUCT_EXECUTORS` (products.py) — the last two being declaration and implementation of the
same list, kept in sync by hand.

**Provenance was a log, not a graph.** `cache.py` recorded table pulls only; images, spectra,
posters, reports, plots and candidate lists were absent, and nothing recorded that *this*
candidate list came from *that* crossmatch. `manifest.jsonl` held 36 records — for a tool
whose stated identity is "provenance by default", a number that says the system wasn't useful
enough to consult.

## 2. First principles

Every operation reduces to the same shape: **take inputs, run a named thing with parameters,
produce a durable object, record that it happened.** So the instrument is a
provenance-first execution kernel with pluggable capabilities and thin renderers — not a TUI
with a backend.

```
Intent ──► Plan ──► Run ──► Artifact(s) ──► Ledger ──► View
                     └── events ──────────────┴──► every surface
```

All six concepts already existed; each was implemented ad hoc at every call site instead of
once. Naming them was most of the work.

## 3. What landed

**`celestrium/core/` — the spine.**

| | |
|---|---|
| `artifact.py` | one typed record; `id = blake2b(cap, params, inputs)`. The cache *is* the ledger, `inputs` make it a DAG, the id is the recipe. Floats hash via `repr` so 0.1 never drifts. |
| `ledger.py` | `data/celestrium.db` — artifacts · edges · runs · studies · tags. Topologically-ordered lineage; preregistrations are write-once. |
| `capability.py` | `@capability(name, kind, params, cost, wing)`. Unknown parameters are an **error**, not a silent no-op — the worst outcome for a content-addressed cache is a typo that changes nothing. |
| `kernel.py` | dedupe → run → persist → record → emit. Retries `cost="network"`, generates methods paragraphs from lineage, `repro()` rebuilds a chain. |
| `events.py` | one event model, rendered three ways. |

**`celestrium/caps/` — 53 capabilities**, one module per family: `archives · objects ·
imaging · lit · feeds · tabular · analysis · studies · notebook`. Every existing module
(`cutouts`, `ads`, `spectra`, `sdss`, `mast`, `exoplanet`, `heasarc`, `vizier`, `xmatch`,
`plots`, the archive shims) was kept and wrapped rather than rewritten.

**The analysis wing — the part that didn't exist.** `caps/analysis.py`:
`synthetic_sky · sky_density · mask · dipole_fit · multipoles · bootstrap · null_shuffle ·
kinematic_dipole · compare`. The shelved dipole pipeline, generalised, cached and
provenance-linked, plus the two controls that decide whether a result is real.

**`celestrium/study/`** — `Study` (claim + pipeline + grid + metric), grid expansion, the
preregistration hash, and `leverage_score` (the vault's own leverage-per-row ranking, made
computable). `study.run` is itself a capability, so a result surface is a normal artifact
whose lineage children are the individual runs.

**`celestrium/vault/`** — the Obsidian bridge. `Observables/*.md` frontmatter *is* the study
list; results write back to known frontmatter keys plus `celestrium_*` and one fenced block.
Prose is never touched. Generated notes are gitignored in the vault.

**CLI**: `caps · run · ledger · ledger-import · repro · methods`, plus `study` and `vault`
groups — **generated from the registry**, so a new capability appears in `--help` with no CLI
edit.

**Tests**: `tests/test_core.py`, 48 hermetic tests. Suite total 257, all passing.

### Two features worth naming
- **Preregistration.** The pipeline+grid+metric hash is stored on first run and never
  silently replaced; changing the analysis after results exist is flagged in the artifact.
  Nothing else in the toolchain records this, and it's what the vault's refutation rule is
  reaching for.
- **Methods paragraphs from lineage.** `celestrium methods <id>` emits the procedure that
  produced an artifact, in dependency order. Free, given the DAG — and it's the caption a
  figure should carry when it leaves the instrument.

### One real bug found and fixed by the new tests
`analysis.sky_density` initially marked every pixel as covered. On a 4.6%-sky footprint that
fed ~11,700 unobserved pixels to the dipole fit as genuine zero-density measurements,
manufacturing an enormous footprint-shaped dipole that swamped any injected signal (both
injected 0.000 and 0.014 returned D = 0.2497). Empty pixels are now `coverage = 0` by
default; `empty="observed"` remains available for genuinely all-sky data and warns.
Regression-tested.

## 4. What this bought

| Before | After |
|---|---|
| `cache.py` + `manifest.jsonl` + `candidates` index | `core/ledger.py` |
| 5 parallel registries | one capability registry |
| 7 edits to add a capability | 1 decorated function |
| provenance for table pulls only | every artifact, with lineage |
| no analysis layer | 9 analysis capabilities |
| no experiment layer | studies, grids, result surfaces, preregistration |
| no vault integration | two-way, guardrailed |

## 5. What's left

**Phase E — the TUI rebuild** (deferred deliberately; it doesn't serve the DR1 deadline).
The old cockpit still runs on `cache.py`/`packets.py`/`products.py` and works. The rebuild:

1. **Cards, not slots.** Each run is a card in a scrollback — fold, scroll back, pin. Nothing
   overwrites anything, which deletes the stale-result bug class along with `_seq`, `_current`,
   `_bump` and the 18 acceptors. The transcript *is* the history, so the trail goes too.
2. **Fork a card** — re-run with edited params in a form **generated from the param schema**,
   replacing bespoke modals like `ImageSettingsScreen`.
3. **Textual's built-in command palette**, populated from the registry; delete
   `tui/commands.py`'s parallel command language, keep a small fast path for muscle memory.
4. **Panes, not modes.** `self.mode` dies. Transcript · Sky · Ledger · Study · Feeds as Screens.
5. **The Sky pane is where `wireframe.py` earns its keep** — all-sky Aitoff on the Braille
   canvas with the retained table, target, galactic/ecliptic planes and mask boundaries. That
   is *the* diagnostic view for the dipole systematics.
6. **Cancel** (`esc` on a running card), **queue visibility**, and **activation actions**
   (TOPCAT-style: send any row to any capability whose input type it satisfies).

Steal from: k9s (header + hotkey bar), lazygit (contextual panes), marimo/Jupyter (transcript
of cells; reactive DAG — it maps 1:1 onto the artifact DAG), TOPCAT (activation actions),
Aladin/ESASky (all-sky + footprints), dbt (lineage, selective re-runs).

**Smaller, unblocked:**
- Port the *real* CatWISE/Quaia pull for `quasar-number-count-dipole` (currently a row-capped
  IRSA query; a genuine dipole needs a bulk download, not TAP).
- `pyproject.toml` + console script + CI (still absent).
- An MCP server generated from the registry — the honest version of "agent-operable".
- Retire `cache.py`/`products.py`/`planner.SOURCE_CAPABILITIES` once the TUI is off them.

## 6. Open questions

1. **Should there be an `Observables/` note for the Euclid forecast?** Celestrium deliberately
   won't author one — the vault owns claims, code owns pipelines. Currently the forecast has a
   pipeline and no claim note, so `study list` shows it without a refutation line.
2. **Real CatWISE access.** TAP row caps make the quasar study a rehearsal too. Bulk download
   + a local ingest capability, or keep it capped and honest?
3. **When does the TUI rebuild happen** — before or after DR1 lands (21 Oct 2026)?

---

## 7. Architectural Teardown & Anti-Patterns ("The Linus Review")

If Linus Torvalds reviewed this codebase, he would point out three layers of "stacked compensations" — places where clever abstractions were built to work around earlier structural design flaws rather than deleting them:

### 1. Dual Systems & Ghost Architecture
* **The Stack:** 5 legacy parallel registries (`ARCHIVES`, `SAMPLE_RECIPES`, `RUNBOOKS`, `GLOBAL_FEED_EXECUTORS` in [`celestrium/registry.py`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/registry.py#L56-L88), `SOURCE_CAPABILITIES` in `planner.py`, `PRODUCT_EXECUTORS` in `products.py`) + a 2,106-line TUI class with 18 `_accept_*` handlers and hand-rolled `_seq` race guards.
* **The Rebuild:** The new DAG Kernel ([`celestrium/core/kernel.py`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/core/kernel.py)) and `@capability` registry wrapped the 53 existing modules instead of replacing them.
* **The Compensation:** CLI tests monkeypatch dictionary attributes on `hub.py` (`hub.QUERY_ARCHIVES`), and `hub.py` re-exports registry objects so patches flow by reference. Meanwhile, `cache.py` maintains its own `manifest.jsonl` ECSV cache alongside SQLite [`celestrium/core/ledger.py`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/core/ledger.py).

### 2. The Digital Garden Sync Hack ([`celestrium/vault/sync.py`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/vault/sync.py))
* **The Stack:** Using Obsidian Markdown files as the database for research claims.
* **The Compensation:** To avoid PyYAML stripping human comments when updating frontmatter, [`set_frontmatter()`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/vault/sync.py#L99-L135) avoids YAML serializers and parses lines with custom regex (`^([A-Za-z0-9_\-]+):(.*)$`). [`upsert_block()`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/vault/sync.py#L137-L145) partitions text inside HTML comment fences (`<!-- celestrium:begin/end -->`). Finally, [`ensure_gitignore()`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/celestrium/vault/sync.py#L324-L337) auto-patches the target vault's `.gitignore` to hide machine-generated output.

### 3. Cryptographic Governance Over Broken Physics
* **The Paradox:** The system was built with `blake2b` DAG hashing, leverage score metrics, and write-once preregistration hashes to prevent p-hacking.
* **The Reality:** The underlying math in `analysis.sky_density` was treating missing sky pixels as observed zeroes, manufacturing a massive false $D = 0.25$ dipole artifact underneath a cryptographically locked pipeline.

### The Remedy ("The Right Way")
1. **Single Source of Truth:** Delete `cache.py`, `packets.py`, `products.py`, and `planner.SOURCE_CAPABILITIES`. Every execution route goes through `Kernel -> Artifact -> SQLite Ledger`.
2. **One-Way Vault Export:** Treat Obsidian notes strictly as read-only inputs or one-way Markdown exports. Never attempt surgical line-by-line regex mutation of Markdown text files.
3. **Data Integrity Over Ceremonies:** Validate spatial coverage masks (HEALPix `seen` masks vs unobserved masks) before wrapping pipelines in cryptographic DAG locks.

