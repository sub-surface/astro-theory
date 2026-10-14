# CLAUDE.md — Onboarding & Technical Standards for Celestrium (astro-theory)

Read this first when picking up work in this repo.
(Parent context: `Psychograph/` hub — see `../AGENTS.md`.)

---

## 1. What Celestrium Is

**Celestrium** is Leon's and Claude's astrophysics research instrument: data access to the major archives,
calibrated inference machinery, and a set of research programmes run on real data with honest uncertainties.

**Research programmes** (one page each in [`docs/programmes/`](./docs/programmes/)):
- **P1 Cosmic dipole** ([`dipole.md`](./docs/programmes/dipole.md)): Quaia/CatWISE/NVSS machinery; Euclid DR1 (21 Oct 2026).
- **P2 Accreting white dwarfs & recurrent novae** ([`novae.md`](./docs/programmes/novae.md)): T CrB status, the symbiotic RN family, "who's next".
- **P3 Calibrated triage engines** ([`transients.md`](./docs/programmes/transients.md)): evidential classifiers + conformal gating, being rebuilt on real labelled data.
- **P4 Theory threads** ([`theory.md`](./docs/programmes/theory.md)): idea ledger (inflation after ACT, f_NL, denoising, ...).

**Where things live**:
- Experiment ledger (single source of truth, with the 2026-10-14 audit status of every result): [`docs/ledger.md`](./docs/ledger.md).
- Docs map: [`docs/README.md`](./docs/README.md). Results JSON: `docs/results/`. Figures: `docs/figures/`.
- Observing calendar: [`celestrium/watchlist.py`](./celestrium/watchlist.py). Run `python -m celestrium calendar` at the start of a session and score any window it lists as closed (`--ics file.ics` exports).
- Engineering roadmap: [`docs/engineering/roadmap.md`](./docs/engineering/roadmap.md). Web platform: https://astro.subsurfaces.net (`site/`).

**Audit rule**: results marked SYNTHETIC ONLY / NEEDS RERUN / WITHDRAWN in the ledger must not be cited as findings.
New results enter the ledger only with real data (or are labelled synthetic validation), clustering-aware or
mock-validated errors, and no leaks of labels, coordinates or simulation truth (section 3).

---

## 2. Repo Architecture

```
celestrium/                 the package
  core/  caps/  study/      kernel (content-addressed artifacts, ledger, capabilities), capability registry, study runner
  archives/                 data access: TAP, HEASARC, MAST, VizieR, X-Match, SDSS, Data Lab, resolvers, cutouts, ADS/arXiv, ALeRCE, feeds
  engines/                  evidential ML (AstroJev family, fused ops, calibration, data streamers)
  dipole/                   P1 code     novae/  P2 code     transients/  P3 code     explore/  parked synthetic-only modules
  cli.py hub.py dashboard.py config.py paths.py net.py watchlist.py modal_app.py   surfaces and shared infrastructure
scripts/<programme>/        runnable experiments: dipole/ novae/ engines/ transients/ explore/ infra/
docs/                       ledger.md, programmes/, results/, figures/, reference/, engineering/, manuscripts/, history/
tests/                      hermetic test suite (`python -m pytest`; 259 fast tests, slow ones marked `slow`)
data/ checkpoints/ cache/   local only (git-ignored)
Archive/                    shelved June 2026 dipole pipeline (G)
site/                       one-page public listing (astro.subsurfaces.net); wrangler.toml deploys it to Cloudflare Workers
```

New programme code goes in its subpackage and its scripts in `scripts/<programme>/`; scripts set
`ROOT = Path(__file__).resolve().parents[2]`. Data for a programme goes in `data/<programme-or-object>/`.

---

## 3. Core Technical Standards & "Gotchas" (Mandatory)

### A. Retention of Analytical Error Bars on Decisions (Zero Naked Predictions)
- **Standard**: Every model decision (classification, candidate triage, follow-up scheduling) MUST retain explicit analytical uncertainties. Never emit naked point probabilities.
- **Formulas**:
  - Dirichlet posterior standard deviation: $\sigma_k = \sqrt{\frac{p_k(1 - p_k)}{S + 1}}$
  - 95% Credible Interval: $[p_k - 1.96\sigma_k, p_k + 1.96\sigma_k]$ (clamped to $[0, 1]$)
  - Conformal prediction set: $C_\lambda(X) = \{ k : p_k \ge 1 - \hat{\lambda}_{\rm CRC} \}$
  - Epistemic vacuity: $u_{\rm epi} = K / S$ with $S = \sum_k \alpha_k$
- **Gating Policy**: High-cost follow-up (e.g. 8m GMOS spectroscopy `GEMINI_RAPID_TOO`) requires $p_{\rm target} \ge \hat{\lambda}_{\rm CRC}$ **AND** $u_{\rm epi} \le 0.35$ **AND** lower bound $p - 1.96\sigma \ge 0.40$.
- **Reward Doubt**: Ambiguous or high-vacuity targets must route to low-cost robotic screening (`LCOGT_SCREENING_TOO`) to collapse uncertainty before spending scarce 8m aperture time.

### B. Disentangled RLCD Optimization (TUM 2026 / Bani-Harouni et al.)
- **Gotcha**: Training confidence or calibration heads end-to-end with representation backbones corrupts embedding geometry and leads to mode collapse.
- **Standard**: Strictly freeze the representation trunk (`input_proj`, `recurrent_cell`) during calibration. Fine-tune ONLY the Dirichlet readout head under composite loss:
  $$\mathcal{L} = \mathcal{L}_{\rm Brier} + \beta \mathcal{L}_{\rm doubt} + \gamma \mathcal{L}_{\rm CARL}$$
  using clipped logarithmic scoring for doubt: $R = \log(\max(\hat{p}, \epsilon))$ for correct, $\log(\max(1 - \hat{p}, \epsilon))$ for incorrect.

### C. Calibration Validation: Stanford Debiased Error (Kumar et al. NeurIPS 2019)
- **Gotcha**: Standard binned plugin ECE has an intrinsic positive variance bias $\sim \frac{\hat{y}_b(1-\hat{y}_b)}{n_b - 1}$ that inflates apparent error on small or imbalanced validation sets.
- **Standard**: Always evaluate and report the Stanford Debiased Squared Calibration Error $\hat{E}^2_{\rm db}$ alongside ECE with bootstrap 95% confidence intervals and RMSCE.

### D. Memory-Efficient Real Data Streaming
- **Gotcha**: Loading multi-million-row catalogs (e.g. 1.3M Quaia sources) into memory exhausts RAM.
- **Standard**: Always stream large FITS catalogs in micro-chunks using `astropy.io.fits.open(..., memmap=True)` (<50 MB RAM footprint).
- **Heteroscedasticity**: Always perturb or evaluate features using observational error bars: $\tilde{\mathbf{x}}_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ on active observational masks.

### E. PyTorch Cross-Device Comparisons
- **Gotcha**: Multi-device tensor assertions (e.g. comparing frozen CPU reference weights against GPU-trained model weights) crash with device mismatch errors.
- **Standard**: Explicitly align devices (`ref.to(device)` or `tensor.cpu()`) before performing any state-dict or weight verification assertions.

### F. Hermetic Test Integrity
- **Standard**: All tests in `tests/` must execute 100% hermetically without internet access. Data streamers, brokers (GraceDB, ALeRCE), and TAP clients must supply local synthetic/cached fallbacks.
- Verify regularly: `python -m pytest` (259 fast tests pass as of 2026-10-14).

### G. Windows PowerShell UTF-8 Encoding for Modal CLI
- **Gotcha**: Windows PowerShell defaults to `cp1252` encoding, causing Modal CLI to crash with `'charmap' codec can't encode character '\u2713'` (checkmark) when rendering terminal status.
- **Standard**: Always prefix Modal CLI commands with `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; modal run ...`.

### H. Astronomical Time & Ephemeris Standards (GPS vs Unix Epoch)
- **Gotcha**: GraceDB trigger times `t_0` are in **GPS seconds** (elapsed since 1980-01-06 00:00:00 UTC), NOT Unix epoch timestamps. Treating GPS seconds as Unix seconds shifts dates by ~3,657 days (~10 years), placing GW170817 in 2007.
- **Standard**: Always convert GPS timestamps via `astropy.time.Time(float(gps_seconds), format="gps").utc.mjd` (properly accounting for leap seconds).

### I. Astrometric Coordinates, Sexagesimal Formatting, & Spatial Prior Leaks
- **Gotcha 1 (Coordinate rotations)**: Never approximate Galactic coordinates $(l, b)$ using naive linear offsets like `(ra + 60, dec + 15)` or swap $(l, b)$ with $(\alpha, \delta)$. Always use strict spherical transforms via `astropy.coordinates.SkyCoord(ra, dec, unit="deg", frame="icrs").galactic`.
- **Gotcha 2 (Sexagesimal formatting)**: Formatting coordinates into sexagesimal strings without rounding arcseconds first can produce invalid outputs like `01:59:60.00` when `59.999` rounds up. Round total seconds before integer division/modulo, and wrap RA modulo 24h / 360°.
- **Gotcha 3 (Spatial Prior Leakage)**: Never feed sky coordinates $(\alpha, \delta)$ or $(l, b)$ as input features to classifiers intended to purify cosmological samples (e.g. Quasar dipole). If the model learns non-uniform sky coverage, the resulting "purified" catalog will encode the classifier's spatial imprint into the very dipole being measured.

### J. Null Audits, False Discovery Rate (FDR), & Conformal Bounds
- **Gotcha 1 (FDR vs FAR under pure null)**: Under an empty-sky null hypothesis (zero true kilonovae), *every* trigger is a false discovery (${\rm FDP} = 1$). A field's FDR is 1 if it triggered at all, and 0 otherwise. Empirical FDR across fields is $\frac{\text{fields with } \ge 1 \text{ trigger}}}{N_{\rm fields}}$ (the quantity CRC bounds in expectation). Reporting per-candidate trigger rates as "FDR" conceals high field-level false alarm risk. Report both `empirical_fdr` and `per_candidate_false_alarm_rate` transparently.
- **Gotcha 2 (Conformal finite-sample edge case)**: When $\lceil (n+1)(1-\alpha) \rceil > n$, the conformal quantile is unbounded ($+\infty$), not finite. Never silently fall back to an arbitrary threshold (e.g. $\hat{\lambda}=1.0$) claiming zero empirical risk. The calibration routine must flag feasibility and issue explicit warnings when no threshold achieves the risk bound.
- **Gotcha 3 (Oracle / Ground-Truth Leaks)**: Active triage gates and alert filters must never read simulation metadata (e.g. `alert.metadata['archetype']` or synthetic visibility tags like `mag=99` on non-kilonova tiles). Triage decisions must depend exclusively on observational features and model evidential posteriors.

### K. Dipole Statistics, Harmonic Sign Conventions, & Deprojection
- **Gotcha 1 (Condon-Shortley Phase Sign)**: Spherical harmonic decompositions $Y_\ell^m$ require strict adherence to the Condon-Shortley phase $(-1)^m$. Omitting it causes the recovered dipole direction to be flipped by $180^\circ$ in Galactic longitude ($l \to l \pm 180^\circ$).
- **Gotcha 2 (Pseudo-Cl Error Scale)**: Pseudo-$C_\ell$ dipole error bars scale with whole-sky Poisson variance, not reduced by an erroneous factor of $\sqrt{N_{\rm pix}}$. Always validate analytical error bars against empirical scatter across $\ge 300$ Poisson mocks.
- **Gotcha 3 (Purification Cancellation Formula)**: Applying a deprojection correction to a purified catalog using $S_{\rm eff} = S_{\rm map} \cdot \frac{N_{\rm pur}}{N_{\rm raw}}$ algebraically cancels out the purification, recovering the raw contaminated estimate. Deprojection must model the actual post-cut selection mask.

### L. Nested Model Likelihoods & MCMC Convergence Diagnostics
- **Gotcha 1 (Nested Model Likelihoods)**: If hypothesis $H_1$ is nested inside $H_2$ ($H_1 \subset H_2$), then by definition $\max \ln \mathcal{L}(H_2) \ge \max \ln \mathcal{L}(H_1)$. If an MCMC or nested sampler reports $\ln \mathcal{L}(H_2) < \ln \mathcal{L}(H_1)$, the sampler failed to locate the global mode and $\Delta{\rm BIC} = {\rm BIC}(H_2) - {\rm BIC}(H_1)$ is a sampler artifact. Furthermore, for nested models $\Delta{\rm BIC}(H_2 - H_1)$ cannot exceed $\Delta k \cdot \ln(n)$.
- **Gotcha 2 (MCMC Convergence)**: Never report posterior kinematic parameters (e.g. bulk flow velocity $v_{\rm bulk}$) without evaluating the Gelman-Rubin diagnostic $\hat{R} < 1.05$ across multiple chains.

### M. Observational Error Alignment & Synthetic Data Realism
- **Gotcha 1 (Column-by-column error pairing)**: When perturbing features with heteroscedastic noise $\tilde{\mathbf{x}}_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, bind errors strictly by column name rather than positional index to prevent swapping photometric noise with astrometric proper motion errors.
- **Gotcha 2 (Tautological Vetoes & Error-Bar Leakage)**: Never encode ground-truth class identity into feature uncertainties (e.g. setting $\sigma_{W1}/\sigma_G$ or $\sigma_{\rm pm}$ to class-specific formulas). When hard heuristic cuts purge 100% of synthetic contaminants by construction, claiming "0.00% false alarms" reflects the veto boundary rather than evidential network discriminative power.

### N. Web Platform, Astro Scripts, & Edge Routing
- **Gotcha 1 (Astro Inline Scripts)**: Inside Astro `<script is:inline>` blocks, browsers do NOT decode HTML entities like `&gt;` or `&lt;`. They cause unhandled `SyntaxError` exceptions at runtime that freeze client-side components. Always write literal `<` and `>` in inline scripts.
- **Gotcha 2 (Monorepo Lockfile Integrity)**: Root `package.json` must not declare `"workspaces": ["site"]` if `site/` maintains its own `package-lock.json`. Always run `npm --prefix site ci` for hermetic builds.
- **Gotcha 3 (Cloudflare Workers 404 Routing)**: Multi-page static sites must serve a dedicated `404.html` with `"404-page"` routing, not `"single-page-application"` (which masks 404s with 200 OK index pages).
- **Gotcha 4 (Distance Modulus Scaling)**: Apparent magnitude scales linearly with distance modulus difference $\Delta \mu = 5 \log_{10}(d / d_{\rm ref})$, NOT $0.4 \Delta \mu$ (which erroneously incorporates the Pogson flux factor $10^{-0.4 m}$).

### O. Untrained Model Checkpoints & Test Fixtures
- **Gotcha**: When neural network checkpoints are gitignored, instantiating triage engines without an explicit checkpoint or trained model silently runs on untrained random weights.
- **Standard**: Engines must explicitly warn when running on uncalibrated/random weights. Unit and integration tests must train small hermetic models or use dedicated test fixtures (`trained_multimessenger_model`) rather than assuming untracked checkpoints exist on disk.

---

## 4. Hardware & Cloud Environments

- **Local Machine**: Windows 11 host (PowerShell), NVIDIA GeForce RTX 2060 (6 GB VRAM). Fast local iteration and unit testing.
- **Modal Cloud Compute**: Active grant balance ~$22.16 USD. Use for scaled H100 SXM5 multi-GPU training, large-scale Monte Carlo runs, and broker streaming workers (`modal run modal_app.py`).
- **Cloudflare Edge Deployment**: Web platform deployed to Cloudflare Workers with static assets serving [https://astro.subsurfaces.net](https://astro.subsurfaces.net). Automated builds via `wrangler.toml` (`[build] command = "npm --prefix site install && npm --prefix site run build"`). Custom OpenGraph suite and SVG/ICO favicons live at edge.

---

## 5. Collaboration Conventions

- Leon and Claude work here under the local git identity `Sub-Surface` (Leon's account). Older results written by earlier agents (Codex, Gemini; mid-2026) are treated as unverified until rerun with principled methods.
- Always check `git status` and test suite before modifying code.
- Avoid introducing extra markdown files as redundant sources of truth: experiments go in [`docs/ledger.md`](./docs/ledger.md), programme context and literature matrices in the programme page.
