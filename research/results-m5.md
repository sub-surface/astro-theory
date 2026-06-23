# Milestone 5 + clustering-aware significance (2026-06-23)

The two committed "natural next steps": (A) honest, clustering-aware significances to replace the
shot-noise-only nulls; (B) the Guandalin/Dalang-Bonvin evolution correction to the kinematic
expectation D_kin. Both turn out to *complicate the anomaly in the same direction* — they widen
the error budget around the "excess."

## A. Clustering-aware significance — `g_clustering_floor.py`
Our M1–M4 nulls were Poisson (shot) only. The fix uses a fact for free: the kinematic dipole is
**pure ℓ=1**, so the multipole power at **ℓ≥2 contains no kinematic signal** — it is an in-data
measure of the non-kinematic per-mode fluctuation (clustering + residual systematics + shot).
Since clustering Cℓ is ~flat at very low ℓ, the ℓ=2 amplitude is a fair proxy for the clustering
contribution to ℓ=1. Define a non-kinematic floor F = √(⟨A₁,shot⟩² + A₂,excess²) and compare:

| sample | dipole A₁ | **A₁ / F (vs non-kin floor)** | A₁ vs shot-noise only |
|---|---|---|---|
| Quaia low (\|b\|>40) | 0.0064 | **0.55** — *below* the floor | 2.8σ |
| Quaia high (\|b\|>40) | 0.0072 | 1.70 | 4.8σ |
| CatWISE uniform-s (\|b\|>30) | 0.0099 | 0.54 | 13.7σ |
| **CatWISE ecliptic-corrected** | 0.0098 | **1.33** | 15.8σ |

- **Quaia low's dipole is *smaller* than its own non-kinematic floor** → fully consistent with no
  special dipole. The "anomaly collapses" conclusion is reinforced at the strongest level.
- **CatWISE's dipole is 16σ over *shot noise* but only ~1.3× the data-driven non-kinematic floor.**
  The shot-noise significance vastly overstates how special the dipole is relative to the
  catalogue's own large-scale anisotropy.
- **Caveat (important):** for CatWISE the residual ℓ=2 is partly the ecliptic *systematic*, so F is
  systematic-inflated and A₁/F=1.33 is a conservative **lower bound**. If the true cosmological
  clustering Cℓ is much smaller (as theory suggests for z~1.5 AGN), the dipole significance is high.
  So this *brackets* the truth: CatWISE's dipole is somewhere between "≈1.3× the field's anisotropy"
  and "highly significant," depending on how much of the residual non-dipole power is systematic vs
  whether comparable systematic leaks into ℓ=1. **This is precisely the unresolved crux of the
  CatWISE debate** — now quantified rather than asserted. A proper lognormal-Cℓ null (fiducial
  cosmological clustering) is the next refinement.

## B. Evolution / relativistic correction to D_kin — `g_evolution.py` (Milestone 5)
The EB formula D=[2+x(1+α)]β assumes a non-evolving population. The full relativistic 2D dipole
factor (Maartens 2018; Dalang-Bonvin 2022; Guandalin 2023) projects a redshift-dependent factor
over the source dN/dz: D = D_cosmo + D_mag + D_evol, with
  D_cosmo = ∫f[2 + 2/(rℋ) + Ḣ/ℋ²],  D_mag = −2∫f·x/(rℋ),  D_evol = −∫f·b_e.
We computed it over **Quaia's measured f(z)** (where Guandalin had to model a QLF), using astropy
Planck18 and the derived identities Ḣ/ℋ² = 1−(1+z)H′/H and rℋ = D_C·H/[(1+z)c].

Results (Quaia high, x=0.96 from M3):
- The **cosmological terms are O(1), not negligible** vs the EB "2": ⟨2/(rℋ)⟩=2.56, ⟨Ḣ/ℋ²⟩=−0.26.
- The magnification term (−2.46) **largely cancels** the +2/(rℋ).
- **D_cosmo+D_mag (b_e=0) = 1.85**, vs **D_EB = 5.30** — they do *not* coincide; the gap is the
  evolution term. Matching EB requires **b_e ≈ −3.5** (a strongly rising population, plausible for
  quasars toward z~2).
- The evolution bias is the decisive, **QLF-model-dependent** piece. A naive empirical b_e from
  N(z) is volume-contaminated (the 1/D_C² term dominates) and unreliable; the proper value needs a
  QLF fit at fixed absolute-magnitude cut (Guandalin's method — the source of their ~3σ spread
  between QLF models).

**Takeaway:** D_kin carries a **real theoretical systematic at the tens-of-percent-to-factor level**
from the evolution bias. This propagates into *every* anomaly significance, including our CatWISE
D/D_kin≈2.3. It does not by itself erase a 2× excess (and von Hausegger 2024 argues EB is adequate
for redshift-integrated samples when x is measured at the limit), but it widens the uncertainty on
"how anomalous" enough to matter. Caveat: matching the EB↔relativistic conventions exactly is
subtle; we present the cosmo/magnification terms (exact) and treat b_e as the bracketed unknown
rather than quoting a single corrected D_kin.

## Where the project stands (synthesis across M1–M5 + explorations)
The "anomaly" fragments into a defensible, nuanced picture:
1. **Deep high-z AGN concur on a ~2–4×, ~CMB-ward dipole** by three independent routes — CatWISE
   (IR, 16σ vs shot), NVSS (radio, 2.6×, marginal), Quaia's z>2 slice (5σ, 11° off CMB). Radio/IR
   systematics are unrelated, so their concordance is the strongest pro-signal evidence.
2. **Clean optical selection + mode separation pull the other way:** full Quaia-low's dipole is
   below its non-kinematic floor; the multipole-separated dipole is marginal.
3. **A real non-kinematic component is present everywhere** (CatWISE residual 7σ quadrupole; Quaia
   low-z structure dipole), and **the kinematic *baseline* D_kin is itself uncertain by a factor**
   (evolution bias). So the "excess" is squeezed from both sides: the measurement's significance
   over a clustering-aware floor is smaller than the headline, and the baseline it's compared to is
   soft.
4. **Net:** we neither confirm a clean >5σ Cosmological-Principle violation nor cleanly refute it.
   The robust, defensible result is the **cross-catalogue contrast** (deep high-z anomalous,
   clean-selection optical consistent) plus a **quantified shrinking of the anomaly** once
   clustering-aware floors and evolution-corrected baselines are admitted. That is a genuine,
   literature-anchored contribution to a live, unresolved debate.

## Next refinements (if continued)
- Proper lognormal-Cℓ clustering nulls (fiducial cosmological Cℓ) to turn the A₁/F bracket into a
  calibrated significance.
- A small QLF fit (or adopt Guandalin's best-fit b_e(z)) to pin D_evol and quote a corrected D_kin
  with a model-spread error bar.
- Per-redshift selection functions to firm up the tomography amplitudes.
- RACS-low (southern radio) as a 4th catalogue to balance NVSS's northern footprint.
