<!-- arXiv:2306.17749  https://arxiv.org/abs/2306.17749 -->
<!-- ar5iv conversion crashed (LaTeX datatool macros); this is a hand-built facts stub. -->

# Quaia, the Gaia–unWISE Quasar Catalog (Storey-Fisher, Hogg, Rix et al. 2024, ApJ)

The cleanest-selection-function all-sky spectroscopic quasar sample — the reference
dataset for project G's "clean catalogue" arm.

## Key facts
- **Construction:** starts from **6.6M** Gaia DR3 quasar *candidates* (with BP/RP
  low-resolution spectroscopic redshifts), then cross-matches **unWISE** (W1/W2 IR) and
  applies cuts on proper motion + Gaia/unWISE colours to kill contaminants. Redshifts
  improved via k-NN trained on colours with SDSS labels.
- **Sample sizes:** **1,295,502** quasars at G < 20.5; **755,850** in the cleaner G < 20.0
  subsample.
- **Redshift quality (G<20.0):** ~6% (10%) catastrophic errors at |Δz/(1+z)| > 0.2 (0.1) —
  ~3× / 2× better than raw Gaia.
- **The headline feature:** ships with a **rigorous, published all-sky selection-function
  model** — the thing CatWISE-style samples lack, and the reason Quaia is the ideal testbed
  for whether the dipole excess survives clean selection.
- **Public data:** Zenodo **10.5281/zenodo.8060755** (also 10403370 for updates),
  https://zenodo.org/records/8060755 . Selection-function maps included.

## Why it matters for G
CatWISE2020 (Secrest+) carries the strongest dipole signal but a *heuristic* selection
function (ecliptic-latitude correction, masks). Quaia trades a bit of sky/number for a
*modelled* selection function. The live question: **does the >5σ dipole excess hold when you
swap CatWISE's heuristic corrections for Quaia's principled selection function?** That
comparison is the spine of our cross-catalogue audit.
