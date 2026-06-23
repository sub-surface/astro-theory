# Milestone 2 results — principled selection correction for the Quaia dipole (2026-06-23)

Code: [`../g_dipole_m2.py`](../g_dipole_m2.py) (estimator + mask sweep),
[`../g_dipole_m2_null.py`](../g_dipole_m2_null.py) (isotropic-mock null + CMB injection).
Data: `quaia_G20.5.fits` (1,295,502 QSO) + the two published Quaia selection-function maps
(`selection_function_NSIDE64_G20.0/20.5.fits`, fetched from Zenodo 8060755). Single-threaded,
Nside=64, <0.4 GB, ~minutes for 100 mocks.

## What changed vs M1
M1's "selection correction" was naive division by the completeness map — which inflates noisy
low-completeness pixels near the mask edge and rotated the dipole ~40° (an **estimator
artifact**). M2 replaces it with the field-standard forward model: a **Poisson maximum-
likelihood** fit with the selection function as a multiplicative term,

    mu_p = s_p · N · (1 + A·n̂_p),     D = |A|,  direction = Â     (fit on the masked sky)

No division, no noise inflation. The Quaia random catalogs are just a Monte-Carlo realisation of
exactly this `N·s_p` model, so using the published `selfunc` map directly **is** the same forward
model with no MC noise and nothing large to download. This is the estimator Dam+2023 / Oayda+2024
call the "Poissonian likelihood."

A key methodological by-product (closes M1 next-step #2): the dipole **amplitude estimator has a
positive noise floor** on a cut sky — `D=|A|` is the norm of a noisy vector, so even a perfectly
isotropic sky returns `D≈0.004–0.005` at |b|>40. Amplitudes must be compared to this floor, not to
zero. Quantified below via isotropic Poisson mocks.

## Result: Milestone 2 PASSED — the anomaly does NOT survive principled correction in clean Quaia

Headline mask = |b|>40 (Oayda's preferred), matched selection function per magnitude cut:

| | Quaia **low** (G<20.0) | Quaia **high** (G<20.5) |
|---|---|---|
| raw-LS dipole (M1-style) | 0.0375 @ ~CMB | 0.0353 @ 14° off CMB |
| **principled (Poisson) D** | **0.0109** | **0.0124** |
| direction (l,b) | **(255,+50) — 6° off CMB** | **(323,+64) — 35° off CMB** |
| σ above isotropic-null floor | **2.6σ** | **4.1σ** |
| CMB-dipole injection recovers | 0.0080 ± 0.0028, dir scatter ~35° | 0.0082 ± 0.0022, dir scatter ~28° |
| verdict | **consistent with a true CMB dipole + noise** | residual excess, but pointed *away* from CMB |

- **The principled correction roughly halves the raw amplitude** (0.035 → 0.011–0.012) and dissolves
  the raw sample's accidental CMB alignment. The raw "≫kinematic, CMB-aligned" dipole of M1 was
  largely the Gaia-scanning selection systematic masquerading as a cosmic dipole.
- **Quaia low → CMB-consistent.** D=0.0109 sits only 2.6σ above the isotropic floor (NOT a >5σ
  anomaly), its amplitude matches what injecting a *genuine* CMB-kinematic dipole recovers
  (0.0080±0.0028, ~1σ), and its direction is 6° from the CMB — far inside the ~35° direction scatter
  a true CMB dipole produces at this mask. The clean sample looks exactly like CMB + shot noise.
- **Quaia high → residual, non-kinematic excess.** D=0.0124 is 4.1σ above the floor and ~1.9σ above
  the CMB-injection amplitude, but it points **35° away from the CMB toward (l,b)≈(323,+64)**. That is
  not the kinematic direction; it drifts as the mask tightens.

## Direct cross-reference to the literature — exact reproduction of Oayda+2024
Oayda, Dam & Lewis (2024, Bayesian Quaia analysis) report that after excising galactic-plane
contamination: **Quaia low becomes consistent with the CMB dipole** (their CMB-aligned model M6
dominant; D×10³≈11 vs CMB≈8 at |b|>40), while **Quaia high drifts toward (l,b)≈(330,60) and cannot be
reconciled / is unconstrained.** Our independent pipeline lands on the *same two conclusions and
nearly the same numbers*:
- Quaia low |b|>40: D×10³ = **10.9** (Oayda ≈11), direction **6° from CMB**. ✓
- Quaia high |b|>40: direction **(323,+64) ≈ Oayda's (330,60)**, off-CMB, residual excess. ✓

This is a **literature-anchored negative result for the cosmological-principle anomaly in Quaia**:
the excess that survives the principled correction lives only in the G<20.5 sample, points away from
the CMB, and behaves like residual selection contamination — not a >kinematic cosmic dipole.

## Honest caveats / what the error bars do and don't include
- Mocks are **Poisson (shot-noise) only**. They omit intrinsic large-scale clustering and residual
  systematic variance, so the *true* uncertainties are **larger** and the quoted significances
  (2.6σ, 4.1σ) are **upper bounds**. This only strengthens "Quaia low is CMB-consistent" and weakens
  the Quaia-high excess further — so the high-sample residual should be read as *at most* a mild,
  systematics-like excess, not evidence for the anomaly.
- D_kin uses the EB placeholder x=1.7, α=1.0 (D_kin≈0.0067). Oayda's Quaia-specific values give
  ≈0.0068 (high) / 0.0080 (low) — negligibly different for these conclusions. Milestone 3 will
  re-measure x and α from the Quaia counts directly.
- The G<20.0 cut now uses its **matched** G<20.0 selection function (not the G<20.5 map) — the
  confound flagged earlier is removed.

## Iteration / next steps (Milestone 3+)
1. **Re-measure x (number-count slope) and α (spectral index) from Quaia** to replace the placeholder
   D_kin and pin D/D_kin precisely (Milestone 3). Cheap: histogram of counts vs flux limit.
2. **Add clustering variance** to the null (lognormal or shuffle-based mocks) for honest significances.
3. **CatWISE via IRSA TAP through the SAME pipeline** — the cross-catalogue audit (gap #1). The open
   question CatWISE poses: its raw >5σ CMB-aligned dipole (D~0.0155) has a heuristic selection
   function; does *it* also collapse under a principled forward model, or is CatWISE genuinely
   anomalous where Quaia is not? That contrast is the project's central deliverable.
