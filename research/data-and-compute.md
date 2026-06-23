# G — data & compute feasibility (verified 2026-06-23)

**Verdict: this runs comfortably on the home machine. No cloud needed.** The "big data" fear
is unfounded because (a) catalogue selection cuts run *server-side* via TAP — we never
download the multi-TB raw CatWISE — and (b) the estimator works on a 49,152-pixel HEALPix
map, not the raw sources. Confirmed by direct measurement, not estimation.

## Data budget (all public, all small)
| Item | Size | How |
|---|--:|---|
| Quaia `quaia_G20.5.fits` (1.30M QSO) | **171 MB** | Zenodo 8060755 (one download) |
| Quaia `quaia_G20.0.fits` (0.76M, cleaner) | 100 MB | Zenodo |
| Quaia selection-function maps (Nside64) | **0.4 MB each** | Zenodo |
| Quaia random catalogs (10×) | 259 + 151 MB | Zenodo — *optional*, only if we use random-based masking |
| CatWISE2020 quasar sample (~1.36M) | **~50–80 MB** | IRSA TAP, cuts applied server-side (verified) |
| NVSS / RACS (later) | ~tens of MB each | VizieR/TAP, server-side cuts |
| **Core first analysis (Quaia + CatWISE)** | **< 300 MB** | — |
| Everything incl. randoms + radio | ~1.5 GB | — |

Verified: a 1° CatWISE cone with the quasar cut (`w1mpro<16.4`, `w1−w2≥0.8`) returns
**63 sources/deg²**, matching Secrest's published density → ~1.3M over `|b|>30°`. Table is
`catwise_2020` on `https://irsa.ipac.caltech.edu/TAP`.

## Compute budget (measured, `g_sanity.py`)
- CatWISE-scale dipole fit (Nside=64, 1.36M sources): **0.03 s wall, 5.2 MB peak Python mem.**
- Recovered injected dipole to ~95% amplitude / 4° direction on a single noisy realization.
- Even a full Bayesian run (dynesty, ~10⁴–10⁵ likelihood evals over ~6 params with this fast
  pixel likelihood) is **seconds-to-minutes, tens of MB** single-core.
- → The machine is over-provisioned by orders of magnitude. The only heavy thing we could
  ever choose to do is large mock-injection suites (1000s of realizations for null tests);
  even that runs overnight locally, gently — see below.

## Environment (installed & verified)
`numpy pandas astropy astroquery pyvo dynesty astropy-healpix matplotlib` — all import OK.
**`healpy` deliberately NOT used** (no Windows wheel; source build fails). `astropy_healpix`
covers everything we need (ang2pix, pixel centers, areas). Don't reintroduce healpy.

## Hardware-safe operating plan (fragile machine, overheating/OOM-prone)
Per the user's constraint — *run longer at lower intensity, anticipate OOM/overheat*:
- **Single-threaded by default.** Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` before
  numpy-heavy work; never spawn process pools without an explicit small cap.
- **Chunk all network pulls.** TAP queries in sky stripes (e.g. 30° RA bands) with pauses
  between, written to disk incrementally — never one giant in-memory result. Resumable.
- **Stream, don't slurp.** Read FITS column-subsets (`fitsio`/astropy `columns=`) not whole
  tables; bin to the HEALPix map and discard raw rows immediately.
- **Checkpoint everything** to `data/` (git-ignored) so a crash/overheat never loses work.
- **Cap mock suites**: run null-test realizations in small batches with cooldown gaps rather
  than a tight max-core loop. Prefer wall-time over CPU saturation.
- If anything ever does strain the machine, the fallback is a *single* cheap cloud VM for that
  one batch job — but nothing seen so far comes close to needing it.

## Path to a result (empirical, iterative — see goal)
Each milestone is a genuine positive/negative checkpoint cross-referenced to the literature:
1. **Reproduce** the published CatWISE dipole (Secrest 2021: D≈0.0155, ~28° from CMB dir,
   4.9σ) with our own pipeline → validates the estimator. *Pass/fail vs literature.*
2. **Cross-catalogue audit (the gap):** run Quaia through the *same* pipeline + masks; compare
   to CatWISE. Does the >5σ excess survive Quaia's principled selection function? → positive
   (excess robust) or negative (excess shrinks under clean selection) result.
3. **Re-measure x and α** per catalogue from the data (don't assume) → does the *expected*
   kinematic amplitude shift enough to matter?
4. **Evolution-corrected expectation (Guandalin 2023):** fold QLF redshift evolution into the
   prediction → recompute significance. Could move the tension by >3σ either way.
5. Synthesize: where do we land vs Secrest (pro-anomaly) vs Guandalin/selection-skeptics?
