# Project G — approach review (2026-06-23, after M1–M3, during M4)

A deliberate step back: is the pipeline measuring the right thing the right way, and where
are the threats to validity? Written before the CatWISE result lands so it isn't post-hoc.

## What we are actually testing
The Ellis-Baldwin number-count dipole: an observer moving at velocity β w.r.t. the source rest
frame sees an anisotropy D = [2 + x(1+α)]β. The CMB fixes β (369.8 km/s) and a direction. The
test of the Cosmological Principle: does the *matter* dipole match that kinematic expectation in
amplitude and direction, or is it anomalously large (Secrest+2021: CatWISE D≈0.0155 ≈ 2× too big,
4.9σ)? Our angle — the gap nobody owns — is a **uniform cross-catalogue audit**: one pipeline,
one estimator, one masking scheme, run on Quaia *and* CatWISE, so the catalogues are compared
like-for-like instead of across heterogeneous published methods.

## Estimator choices and why
1. **Linear least-squares (raw):** `n_p = A0 + A·n̂_p`. Fast, closed-form, what Secrest use. We use
   it only for like-for-like reproduction of published *raw* numbers (M1, M4 step 1). It has no
   selection model, so it is not the number we draw conclusions from.
2. **Poisson forward model (principled):** `μ_p = s_p·N·(1 + A·n̂_p)`, ML fit. The selection
   function enters multiplicatively, never by division → no noise inflation at low completeness.
   This is the Dam+2023 / Oayda+2024 "Poissonian likelihood." For Quaia, `s_p` = the published
   selection-function map (the random catalogs are just a MC draw of `N·s_p`, so the map *is* the
   forward model, with no MC noise). For CatWISE, `s_p` = the ecliptic-latitude density trend
   (Secrest's documented systematic), fit from our own counts.
   → The same estimator on both catalogues is the whole point of the audit.

## The methodological catches (things that would have fooled a naive run)
- **Division-by-completeness is a trap.** It inflates noisy low-`s` pixels near the mask edge and
  rotated the Quaia dipole ~40° in M1 — an artifact, not signal. Replaced by the forward model.
- **The amplitude estimator has a positive noise floor.** `D=|A|` is the norm of a noisy vector,
  so even a perfectly isotropic sky returns D≈0.004–0.008 on a cut sky (rising as the mask shrinks
  the usable sky / source count). Amplitudes MUST be compared to this floor, not to zero. We
  measure it per-config with isotropic Poisson mocks and quote significance as
  (D_data − ⟨D_null⟩)/σ_null. Skipping this would overstate every detection.
- **Frame of the selection map is empirical, not assumed** (M1 found ICRS via corr, +0.67 vs +0.06).

## Validation chain (each milestone is a literature checkpoint, pass/fail)
- M1: reproduce Quaia *raw* dipole (D=0.035 @ |b|>30) — ✓ within 1σ of published.
- M3: reproduce the *kinematic expectation* D_kin from measured x, α — ✓ 0.0065/0.0079 vs Oayda
  0.0068/0.0080 (~3%).
- M2: principled correction reproduces Oayda's *two-arm* finding (low→CMB, high→drift) — ✓.
- M4 (running): reproduce Secrest's *CatWISE* raw dipole (0.0155 @ (238,29)) before correcting it.
A pipeline that independently reproduces four published numbers across two catalogues and two
analysis groups is well-validated.

## Threats to validity (honest list)
1. **Poisson-only nulls.** Our mocks contain shot noise only — NOT intrinsic large-scale
   clustering (the local-structure/LSS dipole) or residual systematic variance. So the true
   uncertainties are *larger* and our significances (e.g. Quaia-high 3.9σ over floor) are **upper
   bounds**. This is the single biggest caveat. Mitigation (future): lognormal or
   shuffled-redshift mocks, or fold in the published clustering covariance. Note this *strengthens*
   the negative result for Quaia-low (it's already only ~2σ over floor).
2. **Incomplete CatWISE masking.** We apply |b|>30 + the ecliptic-latitude correction but not
   Secrest's 291 bespoke point-source/nebula masks. Secrest showed these change D by <5% / <5°, so
   the effect on conclusions is small — but our CatWISE D may differ from 0.0155 at the few-percent
   level for this reason; we report it transparently.
3. **α is an effective band index.** From the Gaia G−BP monochromatic approximation (Oayda's
   method), not a true SED fit. It reproduces Oayda's D_kin, so it's fit-for-purpose, but it is not
   a physical spectral index.
4. **Direction uncertainty blows up at high masks.** At |b|>50 < half the sky remains; the |b|>50
   points wander (e.g. Quaia-low swings to (231,34)). We lean on |b|>40 as the literature sweet
   spot, not the most aggressive mask.
5. **Nside=64** (~0.9° pixels). Ample for a dipole; higher multipoles not our target.

## Where this lands us in the debate
Two camps: Secrest/Dam (pro-anomaly, mainly CatWISE) vs Oayda / von Hausegger / the 2025 kinematic
paper (selection- and method-skeptic). Our independent pipeline so far supports the **skeptic
reading for Quaia**: the raw excess is largely Gaia-scanning selection + galactic-plane
contamination, and the clean (low) sample reconciles with the CMB. The decisive open question M4
answers: **does CatWISE behave the same, or is CatWISE genuinely anomalous where Quaia is not?**
That contrast — same pipeline, opposite catalogues — is the deliverable, and either outcome is
publishable as a citizen-science cross-check.

## Next, in priority order
- **M4 finish:** CatWISE raw (validate) → principled (ecliptic) → null → put on the figures.
- **M5 (Guandalin):** fold QLF redshift-evolution into the expectation; recompute significance.
- **Clustering-aware nulls:** upgrade significances from upper bounds to honest values.
- **Bayesian cross-check:** a small dynesty run mirroring Oayda to confirm the ML results.
