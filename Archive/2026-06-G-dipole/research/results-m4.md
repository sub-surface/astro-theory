# Milestone 4 results — CatWISE through the same pipeline; the cross-catalogue verdict (2026-06-23)

Code: [`../g_catwise_fetch.py`](../g_catwise_fetch.py) (IRSA TAP async pull, resumable RA
stripes), [`../g_catwise_dipole.py`](../g_catwise_dipole.py) (raw + principled + null),
[`../g_figures.py`](../g_figures.py). Data: 3,598,533 CatWISE2020 quasar candidates pulled
all-sky via IRSA TAP with Secrest's cuts (W1−W2≥0.8, 9<W1<16.4 Vega), → 1,360,788 after masking
(Secrest's final: 1,355,352 — within 0.4%). This is the project's central deliverable: **the
same estimator, masks and null on CatWISE and Quaia.**

## Reproduction of Secrest+2021 (validation, Milestone 4 step 1) — PASSED

| | this work (principled Poisson) | this work (raw LS) | Secrest+2021 |
|---|---|---|---|
| amplitude D | **0.0170** | 0.0180 | 0.01554 |
| direction (l,b) | **(237, +26)** | (242, +23) | (238.2, +28.8) |
| offset from CMB | 30.7° | 30.7° | 27.8° |

Direction matches Secrest to ~1° in l and ~3° in b; amplitude ~10% high (residual contamination
+ no dust correction — see below). The estimator and sample are validated against the strongest
published anomaly claim.

### What it took to get there (the contamination story)
A naive |b|>30 cut gave D=0.0315 @ (238,+11) — **2× too large, pulled toward the plane**. The fix,
in two reproducible steps that stand in for Secrest's 291 hand-drawn masks:
1. **Magellanic Clouds** survive |b|>30 (LMC b≈−33°, SMC b≈−44°) and inject a huge southern
   overdensity. Masking them as galactic cones (9°/7°) removed 25k sources → D=0.0269.
2. **Automatic hot-pixel mask** (robust 6σ clip on pixel counts) removed 9,469 sources in just 40
   pixels — dominated by a resolved galaxy/artifact at RA≈147°, Dec≈+13° (l,b≈220,+45) with 896
   sources inside 0.1°. → D=0.0180 @ (242,23), matching Secrest.
The recovered ecliptic-latitude trend (density = 57.5 − 0.04·|elat|) is consistent with Secrest's
(68.89 − 0.051·|elat|); applying it as a forward-model weight barely changes the dipole
(0.018→0.017), i.e. the CatWISE dipole is **robust to the principled selection correction.**

## D_kin for CatWISE
Our near-limit W1 count slope is x=1.36±0.004 (flatter than the literature x≈1.9, which is
measured even closer to the limit / via integral counts on a different cut). For the headline we
adopt the literature kinematic expectation (2025 kinematic paper x=1.90, α=1.07 → **D_kin=0.0073**;
Secrest quote ~0.007). With our flatter x, D_kin would be ~0.0062, making CatWISE *more* anomalous,
so adopting the literature value is the conservative choice.

## THE CROSS-CATALOGUE VERDICT (the gap nobody owned)
One pipeline, one estimator, one masking philosophy, one isotropic null, run on both catalogues:

| measurement | D | D/D_kin | σ above isotropic floor | direction | verdict |
|---|---|---|---|---|---|
| **CatWISE** (principled) | 0.0170 | **2.32** | **11.9σ** | (237,+26), 31° off CMB | **anomalous — confirms Secrest** |
| **Quaia low** (principled, \|b\|>40) | 0.0109 | **1.38** | ~2σ | (255,+50), **6° off CMB** | **CMB-consistent — anomaly collapses** |
| Quaia high (principled, \|b\|>40) | 0.0124 | 1.91 | ~4σ | (323,+64), 35° off CMB | residual, non-kinematic |
| (Quaia high raw, \|b\|>30, M1) | 0.0353 | 5.4 | — | ~CMB dir | selection-inflated |

**CatWISE is genuinely anomalous where clean Quaia is not.** The same principled correction that
collapses Quaia-low onto the CMB leaves CatWISE sitting at 2.3× the kinematic expectation, 12σ
above the noise floor, pointing 31° off the CMB toward (237,26) — exactly Secrest's signal. The
two catalogues **disagree under a uniform analysis.**

### Interpretation (where we land)
This is a genuine, literature-anchored result and it cuts both ways:
- It is *not* a blanket vindication of the anomaly: the cleanest-selection catalogue (Quaia) does
  **not** reproduce it, supporting the selection-skeptic camp (Oayda, von Hausegger) for Quaia.
- It is *not* a refutation either: CatWISE's >2× dipole is **robust** to our principled correction,
  so it is not trivially a selection artifact of the kind that fakes Quaia's raw signal.
- The honest reading: **the dipole excess is catalogue-dependent.** Either CatWISE carries a
  mid-infrared/WISE-specific systematic that our ecliptic correction doesn't capture (and Quaia is
  clean), or there is a real excess that Quaia — shallower (z̄~1.5 fewer sources) and noisier — lacks
  the statistical power to confirm (Quaia-high hints at 1.9× but mispointed). Discriminating these
  is the next scientific step.

## Caveats (carried from the approach review)
- **Poisson-only nulls** → significances (11.9σ, 4σ, 2σ) are upper bounds; clustering variance would
  lower them. This weakens Quaia-low's already-marginal excess further (good for "CMB-consistent")
  and leaves CatWISE's 12σ comfortably significant even if halved.
- **No dust/extinction correction** on CatWISE and only an automatic hot-pixel mask (not Secrest's
  291 bespoke regions) → our CatWISE D is ~10% high; direction is unaffected.
- α is the effective band index (Gaia for Quaia; adopted WISE literature value for CatWISE).

## Next
- **M5 (Guandalin 2023):** fold QLF redshift-evolution into D_kin and recompute CatWISE significance
  — the one correction that could move the tension by >3σ.
- **Clustering-aware nulls** to convert upper-bound significances into honest ones.
- A targeted **WISE-systematics probe**: is CatWISE's residual dipole correlated with ecliptic /
  scan-coverage structure beyond the linear |elat| term? That would localise the discrepancy.
