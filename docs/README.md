# docs/ — map

| Where | What | Kind |
|---|---|---|
| [`ledger.md`](./ledger.md) | Every experiment (EXP-2026-A…Y, 01–05) with data, artifacts, findings and the 2026-10-14 audit status | **source of truth** |
| [`programmes/`](./programmes/) | One page per research programme: question, what stands, rerun queue | living |
| ↳ [`dipole.md`](./programmes/dipole.md) | P1 cosmic dipole / Euclid DR1 | |
| ↳ [`novae.md`](./programmes/novae.md) | P2 accreting white dwarfs & recurrent novae (T CrB) | |
| ↳ [`transients.md`](./programmes/transients.md) | P3 calibrated triage engines on real data | |
| ↳ [`theory.md`](./programmes/theory.md) | P4 theory threads / idea ledger | |
| [`results/`](./results/) | Machine-readable results JSON written by the scripts | generated |
| [`figures/`](./figures/) | All figures (`figures/tcrb/` for P2) | generated |
| [`reference/`](./reference/) | Stable reference: data atlas, imaging guide, literature reviews, local papers (`papers/`, git-ignored) | reference |
| [`engineering/roadmap.md`](./engineering/roadmap.md) | Instrument (CLI/kernel/site) architecture and phase log | living |
| [`manuscripts/`](./manuscripts/) | Paper prospectus and drafts (claims predate the audit; check the ledger) | drafts |
| [`history/`](./history/) | Devlogs, superseded plans and overview docs, kept for the record (not maintained; may overclaim) | archive |

Dated observables and data releases: `python -m celestrium calendar` (source: `celestrium/watchlist.py`).
