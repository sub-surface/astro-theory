# G-Euclid — DR1-ready cosmic-dipole pipeline

Opened **2026-06-25**, the day after Euclid Q2. Successor line to project **G**
(now in [`Archive/2026-06-G-dipole/`](../../Archive/2026-06-G-dipole/ARCHIVE.md)).
Companion to [`directions.md`](./directions.md).

## The thesis in one line

Every existing >5σ cosmic-dipole result (CatWISE, Quaia, NVSS/RACS/LoTSS) shares a
mid-IR / radio / Gaia-flavoured selection function. **Euclid wide is the first deep
optical+NIR sample with an *independent* selection function** — so a Euclid dipole is the
cleanest available test of whether the anomaly is real or a shared systematic. The
desk-tractable, uncrowded, high-stakes move is to be *ready to run on day one of DR1*.

## The calendar (the whole reason this is time-sensitive)

| Release | Date | Relevance |
|---|---|---|
| Q2 — Galactic Bulge Survey | 24 Jun 2026 | **off our axes** (stars, one crowded field) |
| **DR1 — ~1900 deg² wide** | **21 Oct 2026** | **the target dataset** |
| DR2 / later wide | 2027+ | full-sky build-up → definitive dipole |

~4 months of prep window. DR1's ~1900 deg² is *not* full sky and *not* uniform — that is
itself the central systematics problem, and therefore exactly our kind of work.

## Why this is in the target quadrant (effort × tractability)

- **Foundational stakes** — isotropy / Copernican principle, not a parameter.
- **Desk-tractable** — source counting + Ellis–Baldwin dipole fit on a public catalogue;
  we already have the machinery (M1–M5).
- **The fight is systematics** — DR1's partial, non-uniform footprint, depth variations,
  star/galaxy separation, and photo-z selection *are* the contamination question. Our edge.
- **Uncrowded + first-mover** — the community explicitly names Euclid as the awaited test;
  nobody has run a published Euclid dipole because the wide data isn't out yet.

## Pre-DR1 desk tasks (what we can do in the 4-month window, no DR1 needed)

1. **Selection-function spec.** Pin down DR1's expected footprint, depth, bands
   (VIS I_E; NISP Y/J/H), star/galaxy classifier, and quasar/AGN selection. Build the
   mask + completeness model *parametrically* so it slots in when the real footprint lands.
2. **Ellis–Baldwin for Euclid bands.** Our M3 measured (x, α) for Quaia photometry; redo
   the derivation for Euclid's flux limit and spectral index in I_E / NISP. The dipole
   amplitude prediction `D = (2 + x(1+α)) β` depends entirely on these — get them right
   pre-registered, before seeing the data.
3. **Partial-sky dipole estimator.** Our current fit assumes near-full sky. DR1 is ~1900
   deg² in disjoint patches → mask-coupling between dipole and higher multipoles is severe.
   Port/validate a quadratic-estimator or pixel-likelihood fit that handles a small,
   irregular footprint (this is the genuinely new code vs. the archive).
4. **Mock + null suite.** Generate ΛCDM-kinematic-only mocks on the DR1 footprint to
   calibrate the significance of any measured excess *before* we see real numbers — the
   clustering-floor lesson from M5, redone for partial sky.
5. **Cross-match plan.** Where DR1 overlaps Quaia/CatWISE/the deep fields, design the
   consistency check: same sky, different selection → does the dipole agree? This is the
   cross-catalogue audit nobody owns.

## Honest open questions (verify before committing numbers)

- Does DR1 ship a science-ready extragalactic/QSO catalogue, or only images + photometry
  we'd have to select from ourselves? (Changes task 1 from "audit" to "build".)
- Is ~1900 deg² enough sky for a dipole measurement with competitive error bars, or is DR1
  strictly a *methods dress-rehearsal* for DR2? Quantify the forecast σ_D on the DR1
  footprint early (task 4 output) — that decides whether Oct is "measure" or "rehearse".
- Footprint geometry: contiguous vs. scattered drives how bad mask-coupling is.

## First action

Forecast σ_D on a realistic DR1 footprint (tasks 1+4, parametric) → tells us whether DR1 is
a measurement or a rehearsal, which sets everything downstream. Start there.
