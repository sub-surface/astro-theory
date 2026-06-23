# Explorations beyond M1–M4: multipoles, a third catalogue (NVSS), and redshift tomography (2026-06-23)

Three creative cross-checks, each attacking the dipole from an angle the basic estimator can't.
Code: [`../g_multipoles.py`](../g_multipoles.py), [`../g_nvss_fetch.py`](../g_nvss_fetch.py) +
[`../g_nvss_dipole.py`](../g_nvss_dipole.py), [`../g_tomography.py`](../g_tomography.py).
Figure: `figures/fig4_tomography.png`.

## 1. Multipole spectrum — is it a clean dipole, or quadrupole contamination?
Fit real spherical harmonics ℓ=0–2 (well-conditioned on a half-sky mask, cond≈7; ℓ≥3 lets
mode-coupling corrupt the dipole — verified cond 15→285 at ℓ=3→6) with the same selfunc
forward model, isotropic mocks through the identical fit. Dipole D = A₁·√3 (calibrated: at ℓ=1
the harmonic fit reproduces the Cartesian dipole exactly).

| sample (\|b\| cut) | dipole D | dipole σ | quadrupole A₂ | quad σ | A₂/A₁ |
|---|---|---|---|---|---|
| Quaia low (\|b\|>40) | 0.0110 | 2.8σ | 0.0126 | 3.1σ | 1.98 |
| Quaia high (\|b\|>40) | 0.0125 | 4.8σ | 0.0054 | 0.8σ | 0.75 |
| CatWISE uniform-s (\|b\|>30) | 0.0171 | 13.7σ | 0.0183 | **19.9σ** | 1.85 |
| **CatWISE ecliptic-corrected** | 0.0169 | **15.8σ** | 0.0075 | **7.2σ** | 0.77 |

- The dipoles **reconcile exactly** with M2/M4 (0.011, 0.0125, 0.017) — independent validation.
- **CatWISE has a spectacular quadrupole** — bigger than its dipole (20σ) under uniform
  selection. **Forward-modelling the ecliptic-latitude trend collapses it 20σ→7σ** (the |elat|
  scanning systematic is intrinsically quadrupolar), while the **dipole is untouched (15.8σ)**.
  → The CatWISE dipole is robust to the dominant WISE systematic, but a **residual 7σ quadrupole
  remains**, and a pure kinematic dipole predicts *zero* quadrupole — so non-kinematic structure
  is definitely present at some level alongside the dipole.
- Quaia low's dipole (2.8σ) and quadrupole (3.1σ) are both marginal and comparable → a weakly
  anisotropic field, consistent with "no clean kinematic dipole" (reinforces M2's collapse).

## 2. NVSS (1.4 GHz radio) — a third, independent catalogue
Pulled 420k NVSS sources (S₁.₄>15 mJy) via VizieR TAP; radio systematics share nothing with
optical (Quaia) or IR (CatWISE). Same pipeline, footprint Dec>−40 handled by the mask + matched
mocks; declination systematic detrended (quadratic in Dec). EB inputs x=0.74, α=0.75 → D_kin=0.0041.

| mask | D | D/D_kin | direction (l,b) | off CMB | σ over floor |
|---|---|---|---|---|---|
| \|b\|>30, Dec>−37 | 0.0108 | **2.66** | (257,+26) | 23° | 1.4σ |
| \|b\|>40, Dec>−37 | 0.0104 | 2.55 | (302,+21) | 40° | 0.2σ |

- NVSS alone is **low-significance** (1.4σ) — the Dec>−40 footprint + 200k sources give a high
  noise floor (as in the radio literature, where NVSS dipole significance has always been modest).
- BUT its amplitude (**2.6× kinematic**) and \|b\|>30 direction (**l≈257, b≈26**) sit **right next
  to CatWISE's (237,26), ~2.3×** — and radio vs IR systematics are unrelated. Two independent deep
  high-z AGN populations agreeing at ~2.5× and b≈26 is hard to fake with systematics, and is the
  more suggestive (pro-signal) half of the picture.

## 3. Redshift tomography of Quaia — a test only Quaia enables
Quaia ships per-source photo-z (CatWISE has none). A *kinematic* dipole is CMB-aligned at every z
and ~flat in amplitude; a *local-structure* dipole is strongest at low z and need not point at the
CMB. Split into redshift quartiles, principled dipole per bin, |b|>40.

| z-bin | z_med | D | direction (l,b) | off CMB | σ |
|---|---|---|---|---|---|
| 0.0–1.0 | 0.72 | 0.0217 | (350,+33) | **63°** | 3.3σ |
| 1.0–1.5 | 1.24 | 0.0073 | (293,+86) | 38° | noise |
| 1.5–2.0 | 1.71 | 0.0120 | (52,+31) | 95° | noise |
| **2.0+** | **2.41** | **0.0257** | **(248,+46)** | **11°** | **5.0σ** |

- **The high-z bin (z̄≈2.4) — where local structure is weakest — gives the strongest (5σ), most
  CMB-aligned (11°) dipole.** That is exactly the kinematic signature, and it's the cleanest
  CMB-aligned signal found anywhere in Quaia.
- **The low-z bin (z̄≈0.7) points 63° AWAY** toward (350,33) — a local-structure (or low-z
  selection) dipole, unrelated to the CMB.
- Mid-z bins are noise. The full-sample Quaia signal is thus a **blend** of a high-z CMB-aligned
  component and a low-z structure component — which is why naive full-sample fits give muddled,
  mask-dependent directions.
- *Caveat:* per-bin selection functions differ from the full-sample selfunc used as proxy →
  amplitudes are approximate; the **direction trend is the robust observable.** A proper per-z
  selection function (or the published Quaia per-bin maps) would sharpen the amplitudes.

## Synthesis — the picture after the explorations
- **Deep, high-z AGN (CatWISE IR, NVSS radio, and Quaia's z>2 slice) all show a ~2–4× dipole within
  ~10–30° of the CMB**, by three independent selection routes. That concordance is the strongest
  pro-anomaly (or at least pro-real-signal) evidence we've assembled.
- **Clean optical selection (full Quaia-low) and the multipole separation pull the other way:** the
  full-sample dipole is marginal once the quadrupole/low-z structure is removed.
- **A real non-kinematic component exists** (CatWISE's residual 7σ quadrupole; Quaia's low-z
  structure dipole) — the field is not purely kinematic in any catalogue.
- Best current reading: a **CMB-aligned dipole consistent with — but larger than — kinematic is
  present in deep samples, superposed on catalogue-specific structure/systematics**. Whether the
  *excess* over kinematic is real hinges on (a) the evolution correction to D_kin (M5) and (b)
  honest, clustering-aware error bars.

## Next
- **M5 (Guandalin 2023):** QLF redshift-evolution correction to D_kin — could raise the kinematic
  expectation for deep samples and shrink the excess (the high-z tomography bin makes this timely).
- **Clustering-aware nulls** (lognormal mocks with a quasar Cℓ) for honest significances.
- **Per-redshift selection functions** to firm up the tomography amplitudes.
- Optional 4th catalogue: **RACS-low** (southern radio) to complement NVSS's northern bias.
