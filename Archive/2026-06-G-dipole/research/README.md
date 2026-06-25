# Project G — reading notes: the cosmic number-count dipole

Orientation for the cosmic-dipole / isotropy-test literature before we touch data or write a
spec. Papers pulled to markdown in this folder via [`../fetch_papers.sh`](../fetch_papers.sh)
(ar5iv/arXiv-HTML → Pandoc). Citations below use each paper's own reference labels.

## Files / suggested reading order
1. `secrest2021_catwise_dipole.md` — **2009.14826**, Secrest+2021. The CatWISE 4.9σ result; read §2 (sample) + §3 (estimator). The template.
2. `secrest2022_challenge_lcdm.md` — **2206.05624**, Secrest+2022 (ApJL). Joint CatWISE×NVSS, "A Challenge to the Standard Cosmological Model"; pushes to ~5σ.
3. `storeyfisher2024_quaia_catalog.md` — **2306.17749** (facts stub). The clean-selection-function quasar catalog (Quaia).
4. `oayda2024_quaia_bayesian_dipole.md` — **2311.14938**, Oayda/Mittal/Lewis. Bayesian (evidence-based) dipole estimation; the methodological alternative to least-squares.
5. `guandalin2023_theoretical_systematics.md` — **2212.04925**, Guandalin, Piat, Clarkson, Maartens. **The key systematics paper** (see Gaps).
6. `2025_kinematic_contribution_dipole.md` — **2503.02470**. Joint kinematic-dipole estimation, prior/QLF-evolution treatment (2025).
7. `landstrykowski2025_dipole_tensions.md` — **2509.18689**, Land-Strykowski, Lewis & Murphy 2025. Bayesian **cross-dataset tension** (Planck vs NVSS/RACS/CatWISE). Current frontier.

## The physics in one screen — the Ellis & Baldwin (1984) test
Our peculiar motion (β = v/c) Doppler-boosts + aberrates a flux-limited source count into a
**dipole** whose amplitude, for sources with integral count slope `x` (N(>S) ∝ S⁻ˣ) and
spectral index `α` (S ∝ ν⁻ᵅ), is

> **𝒟 = [2 + x(1+α)] β**

ΛCDM + the Cosmological Principle predict this matter dipole should **match the CMB kinematic
dipole** (v = 369.82 ± 0.11 km/s toward (l,b)=(264.0°, 48.3°), Planck 2018). The anomaly:
the measured 𝒟 points *roughly* at the CMB direction but is **~2–4× too large**. So either
(a) we move faster than the CMB says, (b) there's an intrinsic/large-scale-structure dipole,
i.e. the CP is violated, or (c) `x`, `α`, evolution, or selection are biasing 𝒟. (a)/(b) =
new physics; (c) = the systematics battle, which is our entry point.

## Datasets (all public)
| Catalog | Band | N (typical cut) | Sky | Selection function | Notes |
|---|---|--:|---|---|---|
| **CatWISE2020** | IR W1/W2 | ~1.36M QSO (Secrest cuts) | all-sky, \|b\|>30° | heuristic (ecliptic-lat fit + 291 masks) | strongest signal; Vega 9>W1>16.4 |
| **Quaia** | Gaia×unWISE | 1.30M (G<20.5) / 0.76M (G<20.0) | all-sky | **modelled & published** | cleanest selection; Zenodo 8060755 |
| **NVSS** | radio 1.4 GHz | ~0.34M (15–1000 mJy) | δ ≥ −40° | flux-cut + masks; 'B' removes z<0.1 via 2MRS/NED | Condon+1998 |
| **RACS-low** | radio 887 MHz | ~2.1M raw | −80°<δ<30°, \|b\|>5° | possible catalogue systematic (see L-S 2025) | Hale+2021 |
| **LoTSS-DR2** | radio 144 MHz | deep, partial sky | northern | partial-sky → harder | used in some multi-survey fits |
| **Planck PR3 / BeyondPlanck LFI** | CMB μW | — | all-sky | component-separated (SMICA etc.) | the reference dipole |

## Citation web (the lineage)
- **Foundational:** Ellis & Baldwin 1984 (the test) → Blake & Wall 2002 (first NVSS, 1.5–2× excess).
- **Radio era:** Singal 2011; Gibelyou & Huterer 2012; Rubart & Schwarz 2013; Tiwari+2015; Colin+2017 (~2× excess, direction ≈ CMB).
- **IR/quasar era (the escalation):** **Secrest+2021** (CatWISE, 4.9σ) → **Secrest+2022** (CatWISE×NVSS, "Challenge to ΛCDM", ~5σ).
- **Bayesian turn:** Dam+2023; Wagenveld+2023; Mittal+2023; **Oayda+2024** (evidence/suspiciousness) → **Land-Strykowski+2025** (cross-dataset tension).
- **Systematics/theory critiques:** **Guandalin+2023** (QLF evolution); Nadolny+2021; Dalang & Bonvin 2022; **2025 kinematic-contribution** (2503.02470); Cheng+2024.
- **Reviews:** Peebles 2022; Abdalla+2022 (Snowmass-ish); Kumar Aluri+2023; Secrest+2025.
- **Data papers:** Wright+2010 (WISE); CatWISE2020; Storey-Fisher+2024 (Quaia); Condon+1998 (NVSS); McConnell+2020/Hale+2021 (RACS).

## Methods — two camps
- **Frequentist (Secrest):** least-squares dipole estimator on the masked, weighted count map;
  compare amplitude to kinematic expectation; quote p-value/σ. Simple, reproducible — our v1.
- **Bayesian (Oayda, Land-Strykowski):** estimate the dipole via nested sampling; compare
  hypotheses with the **Bayes ratio R** and **suspiciousness log S** (prior-independent
  tension; d − 2logS ~ χ²_d → N_σ). Also the right tool for **catalogue-vs-catalogue**
  consistency, not just catalogue-vs-CMB.

## State of the tension (numbers to anchor on)
- CatWISE: 4.9σ (Secrest 2021) → ~5σ joint (2022); Planck–CatWISE >5σ Bayesian (L-S 2025).
- NVSS: amplitude ~2–3× kinematic; moderate Bayesian tension with Planck.
- RACS: strong tension with Planck **but discordant with CatWISE & NVSS** → likely a RACS
  catalogue systematic, not cosmology (L-S 2025). **Important cautionary result.**
- Concordance CatWISE ↔ NVSS (IR vs radio, independent systematics) → suggests a *common*
  signal, the strongest pro-anomaly argument.
- Forecast: ~**10⁶** radio sources needed for a 5σ radio-only result → SKA-era.

## The gaps — where simple desk work actually contributes
1. **No uniform cross-catalogue audit under a *common* pipeline.** Each group applies its own
   masks, flux/α/x choices, and estimator. A single re-analysis applying **identical** masks,
   cuts, and both estimators across CatWISE / Quaia / NVSS / RACS would isolate how much
   "excess" is method vs sky. L-S 2025 started this pairwise (Bayesian) but not with a unified
   measurement pipeline, and **Quaia is under-used for the dipole** despite its clean selection.
2. **The `x` and `α` inputs are often assumed, not re-measured per catalogue.** 𝒟 depends on
   [2+x(1+α)]; small errors here move the "expected" amplitude. Re-deriving x (count slope)
   and α (spectral index) directly from each public catalogue is a contained, high-value task.
3. **QLF redshift-evolution bias (Guandalin 2023).** The standard 2D Ellis–Baldwin formula
   ignores luminosity-function evolution; folding in radial/QLF info changes the predicted
   velocity amplitude by **>3σ** across plausible QLF models. *This is unresolved* and directly
   weakens/strengthens the claim. Applying the evolution-corrected expectation uniformly is a
   real contribution — and it's exactly a theory-judgment task.
4. **Local-structure / low-z contamination.** The NVSS 'B' variant (drop z<0.1 via 2MRS/NED)
   matters; a systematic study of low-z contamination across *all* catalogues (esp. Quaia,
   which has redshifts!) is tractable and under-done.
5. **RACS-type catalogue systematics.** L-S 2025 flags RACS as internally discordant. A focused
   "is this survey's dipole trustworthy?" null-test protocol (declination stripes, flux-scale
   drifts) is the unglamorous reanalysis departments skip.
6. **Estimator cross-check.** Frequentist vs Bayesian give different emphases; running both on
   one common sample and reporting where they diverge is itself informative and rarely done.

## Implication for our spec (next step, not yet written)
The cleanest first contribution is **#1 + #2 on Quaia vs CatWISE**: one pipeline, identical
masks/cuts, x and α re-measured from the data, both estimators, asking *does the >5σ excess
survive a principled, uniform selection treatment?* — with #3 (evolution-corrected expectation)
as the theory layer that makes it more than a reproduction. All inputs are public
(Zenodo/IRSA/CDS). Spec to follow once we've sanity-checked data access.
