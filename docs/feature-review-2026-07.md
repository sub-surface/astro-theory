# Feature & completeness review (2026-07) — handoff doc

**Purpose of this file:** a critical, evidence-based pass over the CLI (`hub.py`) and TUI
(`tui/app.py`) for completeness, harmony, performance, and fun — written to be handed to
another engineer/model cold. Every claim below is grounded in a specific file/line, not
vibes; if you're picking this up, you can verify anything here in under a minute. Where I'm
speculating or proposing rather than reporting, I've said so explicitly.

**How to use this doc:** it's a punch list plus a set of open invitations, not a spec. Section
9 has a suggested sequence, but use your judgment — if you find something more interesting or
higher-leverage while in the code, chase it and note what you found. The goal is a more
complete, more coherent, more *fun* instrument, not mechanical completion of a checklist.

**Scope:** I read `hub.py`, `registry.py`, `planner.py`, `products.py`, `cache.py`,
`candidates.py`, `paths.py`, `ads.py`, `cutouts.py` (header), `spectra.py`, all five
time-domain modules (`exoplanet/neos/satellites/transients/solarsystem.py`), the full TUI
(`tui/app.py` 1698 lines, `tui/commands.py`, `tui/wireframe.py`, `theme.tcss`), `tests/test_tui.py`,
`docs/roadmap.md`, `docs/imaging-guide.md`, `celestrium/README.md`, root `README.md`, and repo
config/packaging state. I did not run the live network paths (per the hermetic-test convention)
— findings about archive reliability are inferred from code + the repo's own status table, not
freshly verified against live servers.

---

## 1. The one big finding: the two surfaces have diverged, not converged

CLAUDE.md states the rule plainly: *"hub.py and tui/ import celestrium/\*, never each other...
new behaviour goes in registry.py/packets.py so both surfaces get it."* That rule held through
Phase 4.2. Phase 4.3 added `celestrium/products.py` (11 executable product types) and wired it
into the TUI only. Meanwhile the CLI's imaging/report commands (`dossier`, `field`, `poster`,
`atlas-targets`) were never wired into the TUI. The result: **each surface has grown a private
wing the other doesn't have**, and nobody has been keeping the parity matrix.

| Capability | CLI | TUI | Notes |
|---|:--:|:--:|---|
| colour-image | ✅ `image` | ✅ Resolve→Enter | both, via different code paths |
| multi-wavelength panel | ❌ | ✅ | |
| NED spectrum | ✅ `spectrum` | ✅ | |
| SDSS spectrum/image/photometry | ❌ | ✅ | `products.py:_sdss_package` |
| MAST UV/optical observations | ❌ | ✅ | |
| MAST TESS/Kepler/K2 lightcurve metadata | ❌ | ✅ | |
| VizieR cone pull | ❌ | ✅ | |
| HEASARC region query | ❌ | ✅ | |
| NASA Exoplanet Archive query | ❌ | ✅ | |
| Transit prediction | ❌ | ✅ | `exoplanet.py:predict_transits` — real ephemeris math, CLI-invisible |
| JPL Horizons ephemeris | ❌ | ✅ | |
| `dossier` (object packet + Markdown report) | ✅ | ❌ | no TUI equivalent action at all |
| `field` (blank-field packet + report) | ✅ | ❌ | |
| `poster` (wallpaper-grade render) | ✅ | ❌ | only reachable inside the one hardcoded `euclid-q1` runbook |
| `atlas-targets` (contact sheet) | ✅ | ❌ | ditto |
| table export | ❌ | ✅ (CSV only, Ctrl+E) | CLI has no export command at all |
| global feeds (neo/satellite/transient) | ❌ | ✅ (transient is a stub) | |

**Why this matters more than any individual missing command:** an agent (Claude, driving via
`--json`) can currently do things a human at the keyboard can't (pull an exoplanet ephemeris),
and a human at the keyboard can do things an agent can't (`--json dossier`). That's backwards
for a tool whose stated edge is "equally driveable by a human or by Claude." Closing this gap —
in *either* direction, but ideally both — is the single highest-leverage thing to do here,
because almost none of it is new science: it's wiring.

**Concrete next step:** a `celestrium fetch TARGET --product KEY` CLI command
(`planner.resolve_target` + `products.execute_product`, printing whichever `ProductResult.kind`
comes back — table/image/spectrum) would close 9 of the 11 missing CLI rows in one command. A
TUI "Imaging" or "Reports" mode wrapping `packets.build_object_packet` / `build_field_packet` /
`run_runbook`'s poster/atlas steps would close the other direction. Do this before adding
anything net-new — it's the cheapest, most consequential fix in the whole review.

---

## 2. CLI completeness (beyond the parity gap above)

- **No plotting of numeric data.** You can query, sample, and cross-match rich catalogues, but
  there's no way to *see the numbers* — no scatter plot, no CMD/HR diagram, no histogram of a
  candidate-list column. `matplotlib` is already a dependency (`spectra.py:8`, used only for 1D
  spectra). For the "experimental validation" wing specifically — whose whole pitch is testing
  empirical claims — this is a real hole: right now the only way to check if a colour cut worked
  is squinting at a Rich table of floats. A `celestrium plot <table-ref> X Y [--kind scatter|hist]`
  command (table-ref = a manifest hash, a candidate-list name, or a sample recipe) pointed at
  `cache.load_cached`/`candidates.load` would be cheap and high-value.
- **Coordinates only parse as bare decimal degrees.** `planner._parse_coordinates` (planner.py:531)
  requires exactly two floats. No sexagesimal (`12:30:49.4 +12:23:28`), no galactic input, no
  epoch handling — despite `astropy.coordinates.SkyCoord` already being a transitive dependency
  and capable of parsing all of this for free. Most papers quote sexagesimal; this is a paper cut
  that will recur forever until fixed once.
- **No cross-archive sweep.** `where` reports the nearest SIMBAD object and the best colour
  survey, but never asks "does Gaia/WISE/an X-ray catalogue actually have a source here?" A
  "what does every archive know about this position" command would make good on the "eight
  public archives" pitch in a way `where` currently doesn't.
- **No batch action over a candidate list.** You can save 40 crossmatch hits as a candidate list,
  but there's no "dossier every row" / "poster every row" — `packets.build_object_packet` already
  exists; this is a loop, not new science.
- **The field-effort census isn't a hub command.** `scripts/arxiv_tally.py` is *exactly* the
  "living census of where a field spends its effort" the theory wing claims as its job
  (CLAUDE.md), but it's a standalone script outside `python -m celestrium`, with its own
  hardcoded topic list disconnected from `registry.py`. Same story for `fetch_papers.sh`. Both
  are core to the stated mission and both are second-class citizens right now.
- **No cache/manifest maintenance.** `data/cache` and `data/manifest.jsonl` only grow, forever.
  No `celestrium cache size` / `--prune` / `--clear`. Fine today; not fine after a year of daily
  use.
- **No instrument health command.** ADS-token presence, cache size, manifest row count, and
  archive reachability are each checked ad hoc (or not at all). Worth noting: `celestrium/README.md`'s
  own archive status table marks Euclid, MAST, HEASARC, and NOIRLab Data Lab as `·` ("API verified
  against docs, not yet run here") rather than `✓` — the repo *itself* doesn't know if half its
  archives currently work. A `celestrium doctor` that live-pings each registered archive's
  TAP/capabilities endpoint (no data pull, just "are you up") would turn that uncertainty into an
  actual answer, and would be genuinely useful before every session.
- **Shell completion is explicitly disabled**: `typer.Typer(add_completion=False, ...)`
  (hub.py:48). For a CLI meant to be typed by hand daily, this is a near-zero-cost QoL loss —
  flip it on.
- **`--json` mode doesn't emit structured errors.** Every command's error path is
  `console.print(f"[red]{e}[/]"); raise typer.Exit(1)` — this always renders Rich markup to the
  console, *even when `--json` was passed*. An agent scripting the CLI via `--json` gets
  unparsable text on any failure, which undermines the "agent-operable" design goal exactly where
  it matters most (error handling). Worth a small `_emit_error(payload)` helper that JSON-encodes
  `{"error": type(e).__name__, "message": str(e)}` under `--json`.
- **No watchlist for named objects** — only for crossmatch-derived tables. `ATLAS_TARGETS`
  (registry.py:142) is hardcoded and has no CLI-facing add/remove.

---

## 3. TUI review — harmony and completeness

The TUI is the more ambitious, more polished surface (a hand-rolled Braille-canvas 3D wireframe
engine is not a small thing to have built), but a close read turns up real cracks:

### 3.1 Discoverability: two undiscoverable command classes
`tui/commands.py`'s `COMMANDS` table marks each entry `slash: bool`. Only `slash=True` entries
(`/resolve`, `/query`, `/global`, `/history`, `/candidates`, `/runbooks`, `/help`, `/clear`) feed
the prompt's autocomplete (`SLASH_NAMES`, commands.py:53). But `match`, `run <runbook>`,
`feed <x>`, and bare `papers` are real, working idioms (commands.py:41-50) that **never appear in
autocomplete** — a new user who does the natural thing (type `/` and see what's offered) will
never discover crossmatch, runbooks, or feeds exist as prompt commands. They're only documented
in the `/help` card. Either promote these to `/match`, `/run`, `/feed` slash forms (keeping the
bare aliases for muscle memory), or make autocomplete suggest on partial bare-word input too —
right now the interface teaches "type / for commands" and then maintains a second, hidden
command class that contradicts the lesson.

### 3.2 Race-guarding is inconsistent across the five workers
`do_resolve`/`execute_plan` are guarded by manual sequence tokens (`_resolve_seq`, `_fetch_seq`,
checked in `_accept_resolve_result` and `_fetch_is_current`) specifically because Textual's
`@work(thread=True, exclusive=True)` cancels *future* calls to the same method but doesn't stop
an in-flight thread from finishing and calling back stale — the code clearly understands this
(see the "Thread scheduling race check" comment at app.py:1341). **But `do_query`,
`do_crossmatch`, `do_literature`, and `do_global_feed` have no equivalent guard.** Fire a query
against a slow archive, then immediately fire a second query against a fast one, and if the slow
one's HTTP round-trip completes after the fast one, it will silently overwrite `last_table` and
the detail panel with stale data — the exact bug class Resolve was already fixed for. This is
untested too: `test_stale_product_fetch_cannot_overwrite_new_target` (test_tui.py:296) only
covers the product-fetch path. Recommend generalizing the sequence-token pattern into one small
helper (e.g. a `_guarded(method_name)` decorator or a shared `_seq` dict keyed by worker name)
so every `@work` method gets the same protection, rather than re-deriving it by hand each time a
race gets noticed.

### 3.3 The ambient orrery is a beautiful asset that's mostly decorative
`wireframe.py` is a genuine piece of craft — a dependency-free rotation-matrix/perspective engine
drawing onto a Braille canvas at 12 fps (`wireframe.py:307-315`). But its data-awareness stops at
"is this target an exoplanet?" (→ `TransitScene`) vs. everything else (→ a rotating Platonic
solid) and "are these Query results RA/Dec-shaped?" (→ `SkyScatterScene`). Meanwhile
`solarsystem.fetch_ephemeris` already computes real orbital positions elsewhere in the app and
never feeds the orrery. The most technically interesting thing in the codebase is stuck as
ambient wallpaper. Real opportunity: when a Resolve target is a solar-system body, drive the
`TransitScene`/a new `OrbitScene` from the actual Horizons ephemeris rather than a generic
placeholder animation — same rendering engine, real data. This is squarely a "fun *and*
substantive" upgrade, not just polish.

### 3.4 Two mirrored widget instances, kept in sync by hand
There are two `Wireframe` instances (`#orrery-idle`, the big idle canvas, and `#orrery`, the
mini one in the side panel — app.py:385,393). `action_next_solid`, `action_toggle_spin`, and
`_show_egg` all update both, each wrapped in its own `try/except Exception: pass` in case the
other doesn't exist yet (app.py:757-771, 662-670). This is duplicated state kept consistent by
convention, not by structure — every future feature that touches the orrery has to remember
there are two of them. Worth collapsing to one canonical scene-state object that both widgets
read from, rather than two widgets each other code must remember to update.

### 3.5 Imaging & recreation is entirely absent from the TUI
As covered in §1's table: no Resolve/Field mode produces a `dossier`, a `field` packet, a
`poster`, or an `atlas-targets` contact sheet directly — the only way to reach any of that from
inside the cockpit is running the one hardcoded `euclid-q1` runbook, whose poster/atlas steps
are baked into `registry.RUNBOOKS`, not general actions. Given imaging is one of the three named
wings, and the roadmap's own "fun polish backlog" item is *"render highlighted candidate as
poster"* — this is a stated intent that's simply never been built.

### 3.6 Smaller harmony/robustness notes
- Theme choice doesn't persist across launches — `on_mount` hardcodes `self.theme = "ansi-dark"`
  every time (app.py:405); Ctrl+T cycling is forgotten on restart.
- `_show_astropy` hard-caps display at 8 columns / 50 rows (app.py:482) with no way to see more
  short of exporting — no pagination, no "show more" action.
- `#side` panel width is a fixed `44` cells (theme.tcss) with no responsive behaviour for narrow
  terminals; worth a quick check at 80-column width.
- `on_key` hand-rolls Tab-based archive cycling and Escape/`i`//`/` focus shortcuts alongside the
  declarative `BINDINGS` list (app.py:694-731) — workable, but it means "all the keys this app
  responds to" now lives in two different mechanisms, which is exactly the kind of thing that
  drifts out of sync with the `/help` card over time. (Right now it hasn't — `AboutScreen` does
  mention Tab — but it's a structural risk worth flagging before it does drift.)
- The pervasive `try: self.query_one(...) \n except Exception: pass` pattern (dozens of call
  sites) is a reasonable defence against shutdown-order races, but it's broad enough that a typo'd
  widget ID would fail silently rather than loudly during development. Consider narrowing to
  `except NoMatches` (Textual's actual "widget not found" exception) so real bugs still surface.

---

## 4. Infrastructure, performance, and reliability

These are the ones I'd bet an implementing engineer will thank you for catching before they bite
in production use, rather than being asked to build a feature around them.

- **`textual` isn't declared anywhere.** `celestrium/requirements.txt` lists astroquery, astropy,
  pyvo, matplotlib, requests, typer, rich — no `textual`. `pip install -r celestrium/requirements.txt`
  does not get you a working TUI; `hub.py`'s own `tui` command has to catch `ImportError` and tell
  you to `pip install textual` separately (hub.py:583-587). Same story for `pytest` — it's the
  documented test runner (CLAUDE.md) but isn't declared as a dependency anywhere, dev or
  otherwise. Fix: split `requirements.txt` into base + `requirements-dev.txt` (or move to a real
  `pyproject.toml` with `[project.optional-dependencies]`), and declare `textual` + `pytest` in it.
- **No packaging at all** — no `pyproject.toml`, no `setup.py`, no console-script entry point.
  You can only run this from inside the repo via `python -m celestrium`. For a tool meant to be
  used daily (possibly from other working directories), `pip install -e .` plus a `celestrium`
  console-script entry point would remove real friction.
- **No CI.** No `.github/workflows` at all — 120+ hermetic tests exist and are *not* run
  automatically anywhere. This is low-cost to fix (a single GitHub Actions job running
  `pytest tests/ -q` on push) and would catch regressions before they're noticed by hand.
- **Missing/inconsistent network timeouts.** Of everything that makes an HTTP call, only `ads.py`
  (`timeout=30`, twice) and one call in `cutouts.py` (`timeout=15`) set an explicit timeout.
  `neos.py` and `satellites.py` call `urllib.request.urlopen(req)` **with no timeout at all** — a
  stalled connection hangs that call forever. Everything going through astroquery clients (Gaia,
  IRSA, Euclid, HEASARC, MAST, SDSS, SIMBAD/NED via `resolvers.py`, Horizons via `solarsystem.py`,
  NexSci via `exoplanet.py`) inherits whatever default timeout astroquery ships with — usually
  none, or a very long one. The repo's own docs already flag NED as "a frequently-slow server"
  (`celestrium/README.md`) — this isn't a hypothetical risk, it's a known-slow dependency with no
  timeout guard anywhere in the calling code. In the CLI a hang just means a stuck terminal you
  Ctrl+C out of; in the TUI it means a `@work(thread=True)` worker that never returns, silently
  eating a thread and leaving the UI in "queued/fetching" state forever with no way to cancel from
  the keyboard. Worth a blanket pass adding sane timeouts (10-30s depending on archive) everywhere
  a request goes out, plus — in the TUI — a visible "cancel" action and/or a watchdog that turns a
  stuck worker into a visible error after N seconds instead of an invisible hang.
- **No retry/backoff anywhere.** A transient 500 or a rate-limit response from any archive is just
  a raw exception, once. For daily/weekly use against public archives that do occasionally hiccup,
  even a minimal "retry twice with backoff" wrapper around `cache.cached_query`'s fetch callable
  would remove a lot of avoidable "just try it again" friction — and since caching already sits at
  that exact seam, it's a natural place to add it.
- **The ambient orrery runs unconditionally at 12 fps, in duplicate, whenever the app is open**
  (`wireframe.py:315`, two `Wireframe` instances per §3.4). Textual widgets don't auto-pause
  `set_interval` timers when `display = False`, so the hidden `#orrery-idle` instance almost
  certainly keeps ticking after `switch_to_mode` hides it (app.py:505). For a tool explicitly
  meant to sit open on a desk for hours, two permanently-running 12Hz render loops is a real,
  measurable (if probably small) constant CPU draw for zero visible benefit once you've left the
  idle screen. Worth verifying with a profiler and pausing the timer when `display=False`.
- **`app.py` is 1698 lines and one class.** It owns layout, all session state, all five network
  workers, all result-acceptance/rendering logic, and dispatch. The team already recognized this
  exact risk once before — `products.py` was split out in Phase 4.3 specifically so `tui/app.py`
  wouldn't become "a long list of service-specific branches" (products.py:1-6) — but that
  discipline wasn't applied to the App class as a whole, which has kept growing anyway. Before the
  next big feature lands in here, consider splitting by mode (a controller/mixin per
  Resolve/Query/Crossmatch/etc., or Textual `Screen`s) rather than adding a 20th method to one
  class.
- **Credential/config story doesn't generalize.** The ADS token has a bespoke discovery function
  (`ads._token()`, env var or `~/.ads/dev_key`). As more archives need credentials (a TNS/broker
  key for the "rubin watch" feature below, a NOIRLab Data Lab login, a MAST token), a single
  `~/.celestrium/config.toml` (or similar) would avoid reinventing token discovery per-archive
  and gives you one place to add a `celestrium config` CLI command later.
- **No dependency pinning** — everything in `requirements.txt` is `>=`, no lockfile. Minor, but
  worth a one-line note given how much this project's identity rests on "provenance" —
  reproducing a run from six months ago should include reproducing the environment that ran it.

---

## 5. Already-scoped-but-unbuilt features (do these before inventing new ones)

Two things the project has already decided to build, on record, that simply haven't happened yet
— these deserve priority over anything net-new in this doc:

- **Live alert intelligence ("rubin watch"), commit `bcda8a5`.** The doc explicitly scopes:
  broker alert APIs, MPC `X05` submissions, and scheduler/live-status pages as public inputs;
  concrete target actions are *"show alerts near this object/field," "watch this candidate list
  for new transients," "summarize last-night activity in these atlas fields."* Right now
  `celestrium/transients.py` unconditionally returns `None` — "deliberately... until a real
  TNS/ZTF-backed implementation is wired in." TNS, Fink, Lasair, and ALeRCE all have free/anonymous
  query APIs and would make this real. This is the natural next wing of the tool and it's already
  designed — someone just needs to write `transients.py` for real and give it a CLI/TUI surface
  (a `watch` command/mode, per the doc's own framing).
- **The roadmap's "Fun polish backlog"** (bottom of `docs/roadmap.md`) has sat unbuilt since at
  least Phase 3.5: starfield/hyperspace mode-switch transition, a cache-hit sparkline in the
  status bar, an ASCII sky-position mini-map in the detail panel, "render highlighted candidate as
  poster," and the Konami-code→spinning-Cobra-Mk-III easter egg. Either commit to a couple of these
  (they're each small, and §3.3/§3.5 above show real hooks to hang them on) or prune the list so it
  stops reading as active intent when it isn't.

---

## 6. Feature ideas — data & science (bold/speculative, not yet scoped by the team)

These are proposals, not commitments — flagged as such. Pick what's interesting; discard what
isn't; the point is to widen the net, not to freeze scope.

- **A visibility/observability layer.** Nothing today computes airmass, moon separation/
  illumination, or twilight windows — "is this actually observable, and when" is absent even
  though the ephemeris and transit-prediction machinery already exist (`solarsystem.py`,
  `exoplanet.py:predict_transits`). `astroplan` (built on astropy, already a transitive dependency
  family) would make "is TOI-700's next transit visible tonight from here" a real, answerable
  question instead of just a UTC timestamp. Given Celestrium is desk/archive-oriented rather than
  live-observing, this is lower priority than the parity fixes above — but it's the most obvious
  "real feature the tool doesn't have yet" once you notice the ephemeris math is already done.
- **A generic table-plotting command** (§2) — arguably the highest-value net-new CLI feature,
  since it directly serves the stated "test the empirical claim" mission of the validation wing.
- **Sexagesimal/flexible coordinate parsing** via `SkyCoord` (§2) — small, permanent paper cut fix.
- **A cross-archive positional sweep** ("what does every archive know about this position") — a
  natural generalization of `where`, and a good showcase for the "eight archives" pitch.
- **Batch operations over a candidate list** (dossier/poster-per-row) — thin wrapper over
  existing builders.
- **`celestrium doctor`** — live capability-ping every registered archive, resolving the repo's
  own "·  not yet run here" uncertainty (§2).
- **Field-effort census as a first-class command**, replacing the standalone `scripts/arxiv_tally.py`
  with a `registry`-driven topic table and a `celestrium census` command — closes the gap between
  what the theory wing claims to do and what's actually reachable from the tool.

---

## 7. Feature ideas — imaging & recreation (posters and images specifically)

Per direction: skip anything about setting the desktop background (not wanted — other images are
already in use there) and focus on making the *imaging* side itself richer. Ideas, roughly ordered
by how cheap they'd be given the existing `cutouts.py`/`packets.py` machinery:

- **Provenance sidecar for every poster/atlas image** — a `.json`/`.txt` alongside the JPG with
  survey, FOV, coordinates, and pull date. Trivial given `cache`'s manifest pattern already exists;
  turns a poster into a self-documenting, shareable artifact rather than an anonymous JPG.
- **Style presets beyond `clean`/`label`/`science`** — the render pipeline already supports a
  `style` axis (`poster` command, hub.py:499-527); a `noir` (desaturated/high-contrast), `film`
  (grain + vignette), or `deep-field-cinematic` (heavy stretch, deep-field aesthetic à la HUDF
  press images) preset would be genuinely fun and cheap to add as new style branches in
  `cutouts.poster`.
- **"Survey shootout" panel** — same field, same FOV, tiled across every survey that covers it
  (Legacy vs. Pan-STARRS vs. DES vs. DSS2 side by side) rather than the current single
  best-survey-wins `color_auto` choice. This would double as a genuinely useful *diagnostic* for
  the imaging-guide's own footprint logic (`docs/imaging-guide.md`) — you'd be able to see the
  survey selection reasoning, not just trust it.
- **Themed poster series / batch runs** — "all Messier ellipticals," "the whole candidates list,"
  "this week's atlas targets at 4K" — a batch mode over `ATLAS_TARGETS`/candidate lists producing
  a dated folder of renders. Combines naturally with the batch-operations idea in §2/§6.
- **A "tonight's sky" quick render** — naked-eye planets + Moon phase for right now, as a fast,
  fun, low-stakes poster that doesn't need a resolved deep-sky target at all. Nice complement to
  a visibility layer if that gets built (§6), but works as a standalone novelty even without it.
- **Multi-epoch comparison** — if a survey has multiple public data releases of the same field
  (Legacy DR9 vs DR10, say), a side-by-side or blink-comparison render would be a distinctive,
  nobody-else-does-this-easily feature that fits the "desk-tractable, not new photons" ethos in
  the README exactly.

---

## 8. Fun & delight — synthesis

Merge of the roadmap's existing (stale) backlog + what this review turned up:

- Wire the orrery to real ephemeris data for solar-system Resolve targets (§3.3) — the single
  best "fun meets substance" opportunity in the codebase; almost everything needed already exists.
- Ship at least one of the roadmap's long-stalled items (§5) rather than letting the list keep
  growing unimplemented — an ASCII sky-position mini-map or the cache-hit sparkline are both small
  and would prove the backlog isn't just aspirational.
- `celestrium surprise` — dossier/poster of a random pick from `ATLAS_TARGETS` (or a curated
  rotating list), zero new plumbing, purely delightful.
- Style presets and the "survey shootout" panel (§7) are both fun *and* useful — rare combination,
  worth prioritizing over pure eye-candy for that reason.
- The Konami-code easter egg is still just sitting there as an idea — it's small and the wireframe
  engine already exists to receive it (`_show_egg`, app.py:661).

---

## 9. Suggested sequencing (a starting point, not a mandate)

1. **Close the CLI/TUI parity gap (§1).** Highest leverage, no new science, fixes the tool's most
   basic internal contradiction.
2. **Fix the network-timeout and race-guard gaps (§3.2, §4).** Silent hangs and silent stale-data
   overwrites are the kind of bug that erodes trust in a tool used for actual analysis — worth
   fixing before adding more surface area that can hang or race.
3. **Table plotting + flexible coordinate parsing (§2).** Cheap, high-value, directly serves the
   stated mission of the validation wing.
4. **Pick up the already-scoped "rubin watch" / live-alert feature (§5)** — it's designed, just
   not built, and it's the natural next wing of the instrument.
5. **Everything else** — imaging extensions (§7), fun polish (§8), infra cleanup (packaging, CI,
   §4) — roughly in whatever order is most enjoyable to build, honestly. None of it is
   load-bearing; all of it makes the tool nicer to live with.

---

## 10. Explicit invitation to disagree and go further

This review is one pass by one reader over a few hours, not ground truth. If you find something
that seems more interesting, more broken, or more valuable than what's listed here — chase it,
and don't feel obligated to justify the deviation at length. A few prompts if you want them:
*What would make Leon actually open this tool every day rather than reaching for TOPCAT/Aladin
out of habit? What's the smallest change that would make the TUI feel like one coherent
instrument rather than eight modes bolted together? Where else in the codebase has a "we split
this out specifically to avoid X" comment that quietly stopped being true?* (§4's `app.py` finding
came from exactly that last question — there may be others.)
