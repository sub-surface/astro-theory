# Milestone 1 results — Quaia dipole reproduction (2026-06-23)

First empirical pass. Code: [`../g_dipole.py`](../g_dipole.py). Data: `quaia_G20.5.fits`
(1,295,502 QSO) + Quaia Nside-64 selection-function map. Single-threaded, ~2 s, <0.3 GB.

## What was done
- Binned Quaia into HEALPix Nside=64; **empirically determined the selection-function map is
  in equatorial/ICRS coords** (corr(counts, selfunc) = +0.67 ICRS vs +0.06 Galactic — no
  assumption needed).
- Fit the number-count dipole 3 ways across a mask sweep: **raw** (counts on the mask),
  **divide** (counts/selfunc), **fwd** (forward-model selfunc as a weight).

## Result: Milestone 1 PASSED — estimator reproduces the literature
**Raw dipole, |b|>30° cut:** D = **0.0353**, direction **RA=166°, Dec=+7°** (l,b)=(246,58).

| quantity | this work (raw, \|b\|>30) | published Quaia \|b\|>30 | verdict |
|---|---|---|---|
| amplitude D | 0.0353 | 0.033 ± 0.005 | ✓ within 1σ |
| direction | RA 166, Dec +7 | RA 181±14, Dec +20±13 | ✓ within ~1.5σ |
| velocity ratio D/D_kin | ~4.5–5.3 | 4.2 ± 0.6 | ✓ consistent |

→ Our independent pipeline recovers the published Quaia number-count dipole. The estimator is
validated.

## Cross-reference to the wider literature
- **The anomaly is qualitatively confirmed at every masking choice:** D/D_kin ranges 2.0–11
  but is **always ≫ 1** (kinematic expectation D_kin ≈ 0.006–0.008). The Quaia matter dipole
  exceeds the CMB-kinematic expectation regardless of treatment → consistent with the
  Cosmological-Principle tension reported by Secrest+2021/2022 (CatWISE D~0.0155) and the
  Quaia analyses.
- Quaia's raw amplitude (0.035) is **~2× CatWISE's** (0.0155) — consistent with Quaia carrying
  a larger Gaia-scanning systematic on top of any cosmic signal (why the selection treatment
  matters so much here).

## The key methodological finding (and the trap)
The dipole is **highly sensitive to selection treatment**, exactly the field's contested point:

| estimator (\|b\|>30) | D | offset from CMB |
|---|---|---|
| raw | 0.035 | **14°** |
| divide (counts/selfunc) | 0.021 | 54° |
| forward-model weight | 0.019 | 51° |

Naive **division by the completeness map rotates the dipole ~40° and cuts its amplitude ~40%**
— because dividing inflates noisy low-completeness pixels near the mask edge. This is *not* how
the literature applies the selection function: standard practice forward-models it via the
**random catalogs** Quaia ships (or a Poisson likelihood), never by direct division. So:
- The **raw** number is the correct like-for-like comparison → and it matches. ✓
- My **divide/fwd** numbers are artifact-contaminated and should not be trusted as "the
  selection-corrected dipole." Proper correction is the next milestone.

## Iteration / next steps (Milestone 2 proper)
1. **Do selection correction the right way:** use Quaia's `random_G20.5_10x.fits` (encodes the
   selection function) to build expected counts; measure the dipole of (data − randoms)/randoms
   or via a Poisson likelihood. *Prediction to test:* if the dipole **survives** this principled
   correction near the CMB direction → supports the anomaly; if it **collapses** → supports a
   selection-systematic origin for Quaia specifically.
2. **Mask-induced bias:** quantify the linear estimator's bias on a cut sky with isotropic mock
   catalogs (inject zero dipole, measure recovered) — gentle batched mocks.
3. **Re-measure x and α** from Quaia counts (Milestone 3) to pin D_kin instead of the placeholder.
4. Bring in **CatWISE** via IRSA TAP and run the *same* pipeline → the cross-catalogue audit (gap #1).

## Honest status
Positive result: **independent reproduction of the published Quaia dipole + qualitative
confirmation of the >kinematic anomaly.** Open: whether the excess survives a *principled*
selection correction (the crux) — naive division is insufficient and must be replaced by the
randoms-based forward model before we can make a positive/negative claim on the anomaly itself.
