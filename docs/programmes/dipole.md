# P1 · Cosmic dipole (Euclid DR1 line)

**Status**: active; **Euclid DR1 lands 2026-10-21** · **Code**: [`celestrium/dipole/`](../../celestrium/dipole/) · [`scripts/dipole/`](../../scripts/dipole/) · earlier pipeline in [`Archive/2026-06-G-dipole/`](../../Archive/2026-06-G-dipole/ARCHIVE.md)
**Ledger**: EXP-2026-B, E, I, J, K, L, O, P, W, 01, 05

## What stands after the 2026-10-14 audit

- **Machinery worth keeping**: pseudo-C_l mask deconvolution (B), injection–recovery linearity (K, sound as synthetic validation), affine-invariant MCMC with R̂ diagnostics (P), raw-sample selection-function deprojection and literature reconciliation (L), real-catalogue object mining (I).
- **Not supported as claimed**: the "4.9σ" tension (never computed), the >100σ cloud null (E), the ΔBIC = +180.7 multi-tracer result (W, nested-likelihood violation), all significances from independent-Poisson likelihoods (O, P, W), and anything using classifier weights trained with (l, b) features (B, J).

## Rerun queue

1. **Clustering-aware errors** for every dipole amplitude: lognormal/Gaussian mocks with the real mask and selection function (≥10³), or jackknife over sky patches. Requirement for any significance.
2. **Purification without spatial priors**: retrain without (l, b), assert checkpoint loads, model the post-cut selection mask (fixes B, J, L).
3. **Multi-tracer co-inference** with a working optimiser (verify max lnL(H2) ≥ max lnL(H1)) and overdispersed likelihood (fixes W).
4. **Euclid DR1 day-one** (21 Oct): σ_D forecast on the real DR1 footprint first, then the measurement. The headline is a measurement with honest errors, whatever its sign.

---

## Euclid DR1 prep notes (opened 2026-06-25)

Opened **2026-06-25**, the day after Euclid Q2. Successor line to project **G**
(now in [`Archive/2026-06-G-dipole/`](../../Archive/2026-06-G-dipole/ARCHIVE.md)).
Companion to [`directions.md`](../reference/directions.md).

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

## First action — Gate Verdict Answered (2026-09)

Forecast σ_D on a realistic DR1 footprint (tasks 1+4, parametric) → tells us whether DR1 is
a measurement or a rehearsal:

**Verdict: DRESS REHEARSAL (1.8σ distinguishability)**

On the ~1,900 deg² multi-patch footprint (f_sky ≈ 4.4% across EDF-N, EDF-S, EDF-Fornax):
- **Shot noise** is negligible: σ_shot ≈ 0.00039 (for ~1.97×10⁸ sources at 30 gal/arcmin²).
- **Harmonic leakage** from cosmic clustering (C_2 ≈ 5×10⁻⁵) dominates: σ_leak ≈ 0.00401.
- **Combined uncertainty**: σ_total ≈ 0.00403.
- With ΔD = |0.0120 - 0.0047| = 0.0073, SNR is **1.8σ (< 3σ)**.
- **Strategic conclusion**: DR1 cannot decisively reject D_kin at 3σ without joint multipole
  deprojection or the full-sky DR2 release. October 2026 is definitively a **methods dress-rehearsal
  and cross-catalogue consistency audit**, not the final word.

## Implementations Built & Operational

All three proposed modules are implemented, hermetically tested, and integrated:

### I. `celestrium/dipole/forecast.py` — σ_D partial-sky forecast (tasks 1+4, gate task) [COMPLETE]
- Models DR1 wide-survey footprint as a HEALPix mask at configurable NSIDE across 3 high-latitude patches.
- Exact partial-sky Fisher information matrix & covariance.
- Harmonic leakage projection matrix mapping quadrupole clustering C_2 into dipole estimator.
- CLI: `celestrium forecast [--area 1900] [--density 30] [--nside 32] [--c2 5e-5] [--json]`
- Capability: `analysis.dr1_forecast` and `analysis.dr1_footprint`.

### II. `celestrium/dipole/mocks.py` — mock + null pipeline (task 4) [COMPLETE]
- Generates Poisson realizations with prescribed source density and injected dipole modulation.
- Full-sky and partial-sky dipole least-squares fits.
- Computes empirical null distribution & 3σ significance threshold.
- CLI: `celestrium mock [--nsources 100000] [--nmock 50] [--seed 42] [--json]`
- Capability: `analysis.dr1_mocks`.

### III. `celestrium/dipole/ellis_baldwin.py` — D_kin for Euclid bands (task 2) [COMPLETE]
- Pre-registers number-count slope x and spectral index α for Euclid VIS (I_E), NISP Y, J, H,
  combined galaxy sample, and AGN/quasars.
- Monte Carlo propagation (50,000 draws) of parameter uncertainties into D_kin.
- Results:
  - VIS (I_E): D_kin = 0.0047 ± 0.0002 toward (l=264.021°, b=48.253°)
  - Combined galaxies: D_kin = 0.0046 ± 0.0001
  - Euclid quasars: D_kin = 0.0056 ± 0.0004
- CLI: `celestrium ellis-baldwin [--band all] [--json]`
- Capability: `analysis.dr1_ellis_baldwin`.

