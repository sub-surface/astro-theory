<h1 align="center">✦ Celestrium ✦</h1>
<p align="center"><em>A three-wing astrophysics instrument for the desk.</em></p>

---

Celestrium turns a laptop into a working astrophysics bench. The premise: from a desk the
binding constraint isn't photons — it's *ideas and analysis*. The public archives (Gaia,
Euclid, WISE, DESI, Planck) are far richer than the community can exploit, so Celestrium is
built to work the *idea* side: bring in the literature, turn a theoretical signature into
concrete selection cuts, pull a **small** slice of real data, test the empirical claim, and
make something beautiful out of the sky while you're at it.

## The three wings

### 🔭 Theory workshop — *literature in, ideas assessed*
Search ADS/SciX, resolve objects to their bibliographies, assemble paper sets and object
dossiers, and maintain a living census of where a field spends its effort. The job: turn the
literature into a ranked, defensible set of *things worth predicting*.
→ `celestrium papers · cite · resolve · dossier`

### 🧪 Experimental validation — *pull a data slice, test the claim*
Run ADQL against eight public archives through a provenance-logging cache, cross-match a pull
against any catalogue (the audit primitive), and inspect what's where — all row-capped, all
reproducible. The job: take a theoretical implication and *check it against real data*.
→ `celestrium query · sample · match · log · where · field`

### 🎨 Imaging & recreation — *the quiet third wing*
Multi-wavelength scientific panels, true-colour cutouts, and wallpaper-grade renders of any
object or blank field — survey-aware, so you get Legacy DR10 depth where it exists, not blurry
all-sky DSS. Diagrams for a paper; desktop backgrounds for the soul.
→ `celestrium image · poster · atlas-targets`

> One tool, one brain: every wing runs through the same `celestrium/` engine, so a literature
> hit, a data pull, and an image of the same object are three commands apart.

## Quickstart

```bash
pip install -e .                             # install editable package with CLI console script
celestrium --help                            # grouped by the three wings
celestrium resolve M87                       # identity + recent papers
celestrium query gaia "SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source"
celestrium match gaia-bright-nearby vizier:VIII/65/nvss   # cross-catalogue audit
celestrium poster M87 --resolution 4k --style label       # a wallpaper
celestrium --json dossier M87                # machine-readable for agents
```

A free [ADS/SciX token](https://ui.adsabs.harvard.edu/user/settings/token) (in `~/.ads/dev_key`
or `$ADS_DEV_KEY`) unlocks the literature commands. Pulls are cached in SQLite ledger (`data/celestrium.db`)
with content-addressed `blake2b` hashes, so every figure traces back to the exact query that made it.

## How it's built

```
celestrium/   the instrument (Python package)
  core/       kernel · ledger · capability · artifact · events (execution spine)
  config.py   single source of truth for archives, recipes, targets, runbooks, feeds
  hub.py · tui/  the thin CLI + Textual cockpit presenters
  caps/       capabilities (archives, objects, imaging, lit, feeds, tabular, analysis)
  cutouts · resolvers · ads · xmatch · <archive>.py   boring, direct primitives
docs/         data-atlas · toolbox · imaging-guide · roadmap · research/
tests/        hermetic unit test suite (pytest tests/)
```

The architecture rule — *all logic in `celestrium/`; the CLI and TUI only present* — is what
lets a human and an AI agent drive the exact same tool. Full design + the function-first specification:
[`docs/Celestrium_ Rewrite the Astronomy Research Instrument.md`](./docs/Celestrium_%20Rewrite%20the%20Astronomy%20Research%20Instrument.md).

## Roles
- **Leon** — physical judgment: what's worth predicting, which cuts are defensible.
- **Claude** — literature throughput, turning signatures into ADQL/Python, the candidate DB
  and literature census.

## What we're working on

The active build is **G-Euclid DR1 prep** — a cosmic-dipole (isotropy) test ready to run on
Euclid DR1 the day it lands (~1900 deg², 21 Oct 2026), the first deep optical/NIR sample with
a selection function independent of WISE/Gaia. The research backlog (candidate projects A–H,
the leverage thesis, the field-effort map, and the theory-side idea ledger) lives in
[`docs/research/`](./docs/research/): [candidates](./docs/research/candidates.md) ·
[directions](./docs/research/directions.md) · [field-map](./docs/research/field-map.md) ·
[theoretical-threads](./docs/research/theoretical-threads.md) ·
[euclid-dr1-prep](./docs/research/euclid-dr1-prep.md).

Our edge is the neglected, desk-tractable middle: cross-catalogue consistency audits,
selection-function re-analysis of live anomalies, and living meta-analyses — not new photons.

---
<p align="center"><sub>Part of the <a href="../">Psychograph</a> hub · agent onboarding in <a href="./CLAUDE.md">CLAUDE.md</a></sub></p>
