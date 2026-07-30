# Codex Review - CLI and Backend Stress Test (2026-07-09)

Scope: stress-test the Celestrium CLI and backend after the Wave 1-3 feature pass, with emphasis on the repository rule that `celestrium/` owns the instrument logic while `hub.py` and `tui/` present it. This review used local hermetic tests plus bounded live probes. Network calls from this environment are partially blocked, so live archive failures below are separated from code defects where possible.

Commit under review: `0c352748f4c3c7f72237c9c2e72440d22a331bdd` (`feat: add validation plots sweep and census`)

## Executive Summary

The backend is materially healthier than before the feature pass: the full hermetic suite is green, command discovery includes the new Wave 1-3 commands, JSON error handling works for the newly exercised failure paths, sexagesimal coordinate parsing works through the planner, and the new plot/sweep/census surface is test-covered.

The stress pass also found several user-facing defects that should be fixed before calling the pass polished:

- High: several CLI paths crash on Windows default console encoding (`cp1252`) because help text and docs maps emit Unicode glyphs.
- High: cold CLI startup is too slow for help and simple JSON errors, currently about 5-10 seconds per process.
- Medium: the transient global feed presents as executable and cacheable even though the backend is still a planned stub.
- Medium: the new TUI plot/sweep workers can still commit stale results after the active table/target changes.
- Medium: `python -m celestrium.tui --help` launches the full-screen app instead of behaving like a normal help request.
- Low: documentation still lags the expanded command surface, and one optional module fails direct import without its optional dependency.

## Verification Run

Core checks:

```powershell
python -m compileall celestrium tests scripts
python -m pytest tests\ -q
git diff --check
python -m celestrium --help
python -m celestrium --json plan "12:30:49.4 +12:23:28"
```

Results:

- `compileall`: pass.
- `pytest`: `165 passed in 27.51s`.
- `git diff --check`: pass.
- `python -m celestrium --help`: pass, but cold start took about 7.3 seconds.
- `--json plan "12:30:49.4 +12:23:28"`: pass; planner returned a blank-field target at RA `187.70583333333332`, Dec `12.39111111111111`.
- Pytest emitted an ignored cleanup exception for a temporary `pytest-current` path on Windows after reporting success. This did not change the test result.

## Findings

### High - Windows Unicode crashes in CLI help and docs commands

Several commands fail under the default Windows console encoding:

```powershell
python -m celestrium plan --help
python -m celestrium fetch --help
python -m celestrium atlas
python -m celestrium toolbox
```

Observed failures:

- `plan --help`: exits 1 with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'`.
- `fetch --help`: exits 1 with the same arrow encoding failure.
- `atlas`: exits 1 while printing Markdown, failing on characters such as `\u2260`.
- `toolbox`: exits 1 while printing Markdown, failing on characters such as `\u2605`.

Likely root causes:

- `celestrium/hub.py:205-212`: `plan` help/doc text includes Unicode arrows and punctuation.
- `celestrium/hub.py:249-254`: `fetch` help/doc text includes Unicode arrows and punctuation.
- `celestrium/hub.py:795-798`: `_print_map` prints Markdown directly through Rich.
- `celestrium/hub.py:831-839`: `atlas` and `toolbox` route through `_print_map`.

Impact: command discovery is brittle on the user's primary platform, and docs-map commands can hard-fail even when no network is involved.

Recommended fix: either force UTF-8 console output at CLI entry, configure Rich with an encoding-safe output path, or keep CLI doc/help strings ASCII-only. Because this repo runs heavily on Windows, prefer an explicit CLI output compatibility policy and test at least representative help commands under a cp1252 stdout simulation.

### High - CLI cold start is slow for help and trivial failures

Simple process startup is consistently slow:

- `python -m celestrium --help`: about 7.3 seconds.
- `python -m celestrium --json plot definitely-missing-ref ra dec`: about 5.3 seconds.
- `python -m celestrium --json feed quasars`: about 5.9 seconds.
- Direct `import celestrium.hub`: measured about 10.4 seconds in one probe.
- Importing `celestrium.tui.commands`: measured about 14 seconds.

The earlier Wave 3 review already identified the main contributor: `net.set_timeout` calls `hasattr(client, "timeout")` on astroquery-style clients, and that property access can trigger status-message retrieval or other import-time network-adjacent work. The current stress run reproduced the delay.

Relevant code:

- `celestrium/net.py:17-20`: `set_timeout` uses `hasattr` and assignment against `TIMEOUT`/`timeout`.
- `celestrium/cutouts.py:21-29` and `celestrium/resolvers.py:10-16`: apply timeout configuration at module import.
- `celestrium/hub.py`: imports many heavy backend modules eagerly.

Impact: even help text and local validation errors feel like network commands. This also makes automated CLI smoke sweeps unnecessarily slow.

Recommended fix: make timeout setting class-dict based where possible instead of property probing, and move heavy archive clients behind command-local imports. A good target is sub-second startup for `--help`, local `--json` validation errors, and command help.

### Medium - Transient feed is exposed as executable while the backend is still a stub

The transient feed is currently callable and cacheable:

```powershell
python -m celestrium --json feed transient --refresh
python -m celestrium --json feed transient
```

Observed result:

```json
{"feed": "transient", "status": "empty", "table": {"columns": [], "rows": []}}
```

The live refresh wrote an empty `global-transient` cache entry. Static inspection shows `celestrium/transients.py` still deliberately returns `None` until a real broker implementation exists, while registry/global-feed plumbing now allows the CLI to run it.

Relevant code:

- `celestrium/transients.py`: planned/stub implementation returns no rows.
- `celestrium/registry.py`: global feed executor includes `transient`.
- `celestrium/hub.py`: `feed` treats the empty result as a successful empty feed.
- `celestrium/planner.py`: transient global feed is still marked `planned`, which conflicts with the executable CLI path.

Impact: users can interpret a successful empty transient feed as "no transient alerts" rather than "feature not implemented here yet." It also pollutes cache/manifest history with a planned feed.

Recommended fix: until Wave 4 lands, make `feed transient` return a clear planned/unavailable status without caching an empty scientific result, or hide it from executable feeds while keeping the roadmap entry visible.

### Medium - New TUI plot/sweep workers still have stale-result risk

The Wave 2 sequence-token guard generalized several worker classes, but the Wave 3 TUI additions use local sequence counters that are not invalidated when the underlying active target or last table changes.

Relevant code:

- `celestrium/tui/app.py:1424-1437`: plot worker path.
- `celestrium/tui/app.py:1439-1465`: sweep worker path.

Impact:

- A slow `/plot` can render against an old `last_table` after a newer query/match replaces it.
- A slow `/sweep` can append details for a target that is no longer the active target.

Recommended fix: route these workers through the same `_bump`/`_current` guard family used for query/crossmatch/literature/global feed, keyed by table or target identity as appropriate.

### Medium - `python -m celestrium.tui --help` launches the app

The CLI help path works through:

```powershell
python -m celestrium tui --help
```

But the module entry point:

```powershell
python -m celestrium.tui --help
```

launched the full-screen TUI and timed out after 120 seconds. No stray Python process remained afterward.

Impact: this is a smaller ergonomics issue than the main CLI help failures, but it violates normal Python module expectations and complicates smoke testing.

Recommended fix: have the TUI module entry parse `--help`/`-h` before launching Textual, or document that only the root CLI owns help.

### Low - Global `--json` is not a universal contract yet

The new structured error helper works on the exercised validation/fetch paths:

- `--json plot definitely-missing-ref ra dec`: structured `KeyError`.
- `--json feed quasars`: structured `ValueError`.
- `--json fetch 10.0 -5.0 --product definitely-missing-product`: structured `ValueError`.
- `--json fetch 10.0 -5.0 --product vizier`: structured `ConnectionError`.

But commands like `atlas` and `toolbox` still route through direct Rich/Markdown printing and do not honor JSON semantics. Static inspection shows many direct `console.print` paths remain in `hub.py`, though not all are error paths or intended machine interfaces.

Impact: `--json` is improving but should not yet be advertised as universal for every CLI command.

Recommended fix: define the intended JSON contract per command group, then sweep only those commands. For commands that are intentionally human-only, reject `--json` explicitly or emit a small JSON envelope with the document path/content metadata.

### Low - Optional module direct import fails without optional dependency

A backend import sweep over `celestrium.*` modules found one direct import failure:

```text
celestrium.datalab_desi -> ModuleNotFoundError: No module named 'dl'
```

Normal CLI imports did not fail because DESI access is lazy enough through the registry adaptor path. This is still worth noting because `celestrium/README.md` presents backend modules as useful building blocks.

Impact: developers who bulk-import backend modules or inspect them directly hit an optional dependency error.

Recommended fix: guard the optional `dl` import inside the DESI executor path and raise a user-facing dependency message only when DESI Data Lab functionality is requested.

### Low - Docs lag the command surface

The root README and `celestrium/README.md` still describe an older CLI surface and omit several now-present commands, including `fetch`, `feed`, `export`, `plot`, `sweep`, `census`, and `doctor`.

Also, `docs/feature-review-2026-07.md` is still untracked in the worktree. The plan already calls for committing it and appending an outcome section at the end of the full pass.

Impact: discoverability depends too heavily on the CLI itself, which currently has Windows help failures.

Recommended fix: update command-surface docs after Wave 3/4 stabilizes, or do a smaller interim docs patch that marks the new commands experimental.

## Command Sweep

Help sweep:

| Command | Result |
|---|---|
| `--help` | pass |
| `resolve --help` | pass |
| `image --help` | pass |
| `where --help` | pass |
| `spectrum --help` | pass |
| `cite --help` | pass |
| `papers --help` | pass |
| `query --help` | pass |
| `sample --help` | pass |
| `match --help` | pass |
| `candidates --help` | pass |
| `log --help` | pass |
| `feed --help` | pass |
| `export --help` | pass |
| `plot --help` | pass |
| `sweep --help` | pass |
| `census --help` | pass |
| `dossier --help` | pass |
| `field --help` | pass |
| `atlas-targets --help` | pass |
| `poster --help` | pass |
| `runbook --help` | pass |
| `doctor --help` | pass |
| `atlas --help` | pass |
| `toolbox --help` | pass |
| `tui --help` | pass |
| `plan --help` | fail: UnicodeEncodeError |
| `fetch --help` | fail: UnicodeEncodeError |

No-network/local command sweep:

| Command | Result |
|---|---|
| `sample list` | pass |
| `runbook list` | pass |
| `candidates` | pass |
| `log --open definitely-missing-hash` | pass as expected failure |
| `--json doctor --no-ping` | pass |
| `--json plan 10.0 -5.0` | pass |
| `--json plan 12h30m49s +12d23m28s` | pass |
| `--json plot definitely-missing-ref ra dec` | pass as structured error |
| `--json feed quasars` | pass as structured error |
| `--json fetch 10.0 -5.0 --product definitely-missing-product` | pass as structured error |
| `atlas` | fail: UnicodeEncodeError |
| `toolbox` | fail: UnicodeEncodeError |

Bounded live/network probes:

| Command | Result |
|---|---|
| `--json doctor --timeout 2` | pass; all 7 archive pings reported down in this environment |
| `feed neo --refresh --show 3` | expected environment failure: WinError 10013 |
| `feed satellite --refresh --show 3` | expected environment failure: WinError 10013 |
| `--json feed transient --refresh` | pass but misleading empty stub result |
| `where 187.7059 12.3911` | pass |
| `sample gaia-bright-nearby --show 2` | pass; produced/cached a 25-row table |
| `--json fetch 10.0 -5.0 --product vizier` | structured environment/network failure |
| `papers cosmic --rows 1` | expected environment failure: WinError 10013 |

Doctor details with live pings:

- ADS token: present.
- Cache files: 13.
- Manifest rows: 19.
- Candidate lists: 1.
- Archive pings: 0/7 up from this environment.
- Failed endpoints: Gaia, IRSA, Euclid, HEASARC, DESI, CASDA, VizieR TAP, all surfaced as connection errors rather than crashes.

## Backend Import Sweep

Direct module import sweep:

- Modules ok: 32.
- Modules failed: 1 (`celestrium.datalab_desi`, missing optional `dl`).
- Slow imports: `celestrium.__main__` around 11.8 seconds, `celestrium.exoplanet` around 0.38 seconds, `celestrium.euclid` around 0.27 seconds.

Registry/product smoke:

- Archives registered: 7.
- Sample recipes registered: 4.
- Runbooks registered: 1.
- Source capabilities registered: 15.
- Executable product keys observed: 11.

TUI command registry smoke:

- Command idioms loaded: 17.
- Slash aliases include `/resolve`, `/query`, `/match`, `/run`, `/feed`.
- New Wave 3 intents parse: `plot`, `sweep`.
- Import remains slow because it reaches heavy backend imports.

## Plan/Spec Completeness Notes

Wave 3 status from this stress pass:

- `plots.py`: present, tested, and CLI/TUI command paths exist.
- Flexible coordinate parsing: works for colon sexagesimal and `12h30m49s +12d23m28s` forms through the planner.
- Sweep packet/command: present and test-covered; live archive coverage could not be validated because this environment blocks relevant endpoints.
- Census module/command: present and test-covered.
- Legacy `scripts/arxiv_tally.py` shim: present.

Remaining spec gaps exposed by stress testing:

- Wave 2's structured error goal is not uniformly true for every command under global `--json`.
- Wave 2's trust/race work needs to be extended to the new Wave 3 TUI workers.
- Wave 4 transient work should either land soon or the transient feed should be marked unavailable at runtime.
- Docs/bookkeeping from the plan remain outstanding and are now more important because CLI help has Windows failures.

## Recommended Next Fix Order

1. Fix Windows console Unicode crashes in command help and Markdown-map commands.
2. Remove import-time astroquery/status-message delays from `net.set_timeout` and defer heavy imports in `hub.py`.
3. Make `feed transient` unavailable/non-cacheable until real broker code lands.
4. Add stale-result guards for TUI plot/sweep workers.
5. Decide and document the real `--json` command contract, then sweep remaining direct print/error paths.
6. Guard optional DESI Data Lab imports.
7. Update README/CLAUDE command-surface docs after the fixes above.

