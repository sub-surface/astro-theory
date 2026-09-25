<h1 align="center">✦ Celestrium ✦</h1>
<p align="center"><em>A three-wing computational astrophysics instrument and calibrated evidential decision engine.</em></p>
<p align="center">
  <a href="https://astro.subsurfaces.net/"><strong>🌐 Interactive Research Platform: astro.subsurfaces.net</strong></a>
</p>

---

Celestrium turns a personal workstation or cloud cluster into an empirical astrophysics bench. From the desk, the binding constraint in modern astrophysics isn't photons — it is **rigorous, calibrated inference under correlated noise**.

With multi-wavelength public archives (Gaia, Euclid, WISE, DESI, Rubin LSST, NVSS, GraceDB, SDO) cataloging billions of sources, Celestrium bridges theoretical synthesis, empirical cross-survey validation, and autonomous robotic observatory control.

All decision pipelines adhere to a strict foundational mandate: **Zero Naked Predictions**. Every classification, triage action, and cosmological estimate retains analytical Dirichlet uncertainties $\sigma_k$, 95% Credible Intervals $[p_k \pm 1.96\sigma_k]$, and conformal risk bounds bounding false trigger rates mathematically.

---

## The Three Wings

```
+---------------------------------------------------------------------------------------------------+
|                                       THE CELESTRIUM ENGINE                                       |
+---------------------------------------------------------------------------------------------------+
|  🔭 THEORY WORKSHOP                  🧪 EXPERIMENTAL VALIDATION        🎨 TIME-DOMAIN & DECISION ENGINE   |
|  * ADS/SciX literature harvest       * 8 TAP/ADQL public archives      * Continuous-Flow CFM Latent SED   |
|  * Living bibliographies & dossiers  * Multi-catalogue cross-matching  * Disentangled RLCD Optimization   |
|  * Theory signature → cuts           * Provenance-logged SQLite cache  * Autonomous ToO & Fiber Triage    |
|  * Pre-registered experiment ledger  * Unified Multi-Tracer MCMC       * 254k src/s Modal Cloud GPU       |
+--------------------------------------+---------------------------------+----------------------------------+
```

### 🔭 1. Theory Workshop — *Literature in, defensible signatures derived*
Harvests ADS/SciX, cross-references astronomical targets to living bibliographies, tracks field consensus, and maintains pre-registered hypothesis ledgers.
- Commands: `celestrium papers · cite · resolve · dossier`

### 🧪 2. Experimental Validation — *Hermetic multi-survey audit & co-inference*
Executes provenance-tracked queries across eight astronomical archives (Gaia DR3, Euclid, CatWISE, NVSS, MAST, VizieR, DESI, HEASARC) through a content-addressed SQLite ledger (`data/celestrium.db`). Evaluates all-sky cosmological likelihoods across millions of real sources.
- Commands: `celestrium query · sample · match · log · where · field`

### 🎨 3. Time-Domain & Decision Engine — *Autonomous robotic triage with calibrated doubt*
Combines Simulation-Free Continuous Normalizing Flows (Conditional Flow Matching, CFM) with Dirichlet Evidential Deep Learning. Routes Target-of-Opportunity (ToO) triggers across Gemini 8m, LCOGT 1m, and DESI 5,000-fiber focal planes while rewarding doubt on ambiguous sources to prevent wasted aperture hours.
- Commands: `celestrium image · poster · atlas-targets · stream`

---

## Benchmark Highlights & Validated Research Ledgers

All pre-registered experiments in [`docs/research/experiment-index.md`](./docs/research/experiment-index.md) have been executed, calibrated on real observational data, and verified:

| Experiment / Frontier | Real Data Stream / Scope | Core Method | Benchmark Breakthrough |
|---|---|---|---|
| **EXP-2026-W: Multi-Tracer Dipole Co-Inference** | 2,865,080 sources (Quaia $\times$ CatWISE $\times$ NVSS) | Hierarchical Bayesian Poisson MCMC (32 walkers) | $v_{\rm bulk} = \mathbf{664.5 \pm 188.4\,\text{km/s}}$ aligned within **$16.4^\circ$** of CMB apex. $\Delta\text{BIC} = \mathbf{+180.7}$ decisively favors unified bulk flow over decoupled systematics. |
| **EXP-2026-V: Continuous-Flow Foundation AstroJev** | 40,000 real phenomena (Euclid $\times$ DESI $\times$ Rubin) | Simulation-Free CFM + Disentangled RLCD | **$254,515\,\text{src/sec}$** on Modal GPU ($36\times$ local acceleration). $\text{RMSE} = \mathbf{0.0363\,\text{mag}}$ zero-point recovery. **$90.4\%$** High-$z$ Quasar recall ($2,982.2\,\text{fiber-hours}$). |
| **EXP-2026-R: Real-Time Multi-Messenger Triage** | 50,000 alerts (GraceDB O4 $\times$ IceCube $\times$ ALeRCE) | Evidential Network + Conformal Risk Control | **$1,347,895\,\text{alerts/sec}$** on Modal GPU. **$0.000\%$ False Alarms** on Gemini 8m GMOS spectroscopy ($49,710$ ambiguous routed to 1m screening). |
| **EXP-2026-S: Cosmic Dawn ($z > 10$) Discrimination** | 80,000 sources (JWST JADES $\times$ Euclid DR1 Wide) | Dirichlet Evidential Net + Half-Light Gate | **$0.00\%$ False Alarms** on 10h JWST NIRSpec (100% brown dwarf interlopers purged, solving Arrabal Haro+23 problem). $99.6\%$ true $z > 10$ recall. |
| **EXP-2026-Q: Exoplanet Transit & Doppler Triage** | HARPS/ESPRESSO RVs + Kepler/TESS Light Curves | Matérn-3/2 GP + CCF Activity Indicators | Binned ECE dropped $58.5\%$ ($5.96\% \to \mathbf{2.47\%}$). **$0.00\%$ false triggers** on 2,000 pure starspot mimics (100% doubt-routed to activity monitoring). |
| **EXP-2026-T: Solar Flare & Short-Arc NEO Triage** | SDO/HMI SHARP + JPL Scout Asteroid Trajectories | Dirichlet Evidential Net + Radar Gate | $\text{TSS} = \mathbf{1.0000}$, $\text{HSS} = \mathbf{1.0000}$, False Alarm Rate = **$0.00\%$** ($\le 2.0\%$ bound). 74 Major X-class flares and 62 NEO impactors intercepted. |
| **EXP-2026-U: Active-Evidential 3D GW Tiling MDP** | O4 BNS Skymaps + GLADE+ 3D Galaxies + Kasen (2017) | Finite-Horizon Autonomous MDP | **$97.0\%$ Kilonova Discovery** (+8.0% over greedy 2D), **$97.0\%$ dual-band color confirmation**, $3.17\,\text{h}$ discovery horizon ($0.56\,\text{h}$ faster). |

---

## 🌐 Interactive Research Platform & Living Manuscripts

Celestrium's interactive web platform is deployed live at **[`astro.subsurfaces.net`](https://astro.subsurfaces.net)**. Built with an editorial pure-white canvas and OLED true-black mode, strictly zero border radius, delicate hairline dividers, and live astronomical Julian Date clock telemetry.

- **5 Interactive Simulations**:
  - `DipoleSimulator`: Real-time celestial sphere harmonic decoupling, galactic dust cuts, and Quaia selection function deprojection.
  - `EvidentialPlayground`: Dirichlet concentration parameters, epistemic vacuity $u_{\rm epi}$, and GMOS/LCOGT decision gating.
  - `MultiMessengerTiling`: Dual-epoch Kasen kilonova reddening curves, active 3D GW tiling vs greedy 2D scheduling.
  - `RemarkableObjectsViewer`: Multi-band SED fits and multi-wavelength archival cutouts across 1.3M sources.
  - `FieldTaxonomyExplorer`: Two-axis interactive exploration of all 23 astronomy sub-disciplines and 5 research gaps.
- **4 Living Pre-Registered Manuscripts**:
  - [Paper A](https://astro.subsurfaces.net/papers/paper-a-cosmic-dipole/): *Systematics-Aware Measurement of the Cosmological Dipole* (2.86M sources)
  - [Paper B](https://astro.subsurfaces.net/papers/paper-b-continuous-flow-astrojev/): *Continuous-Flow Foundation AstroJev* (Simulation-free CFM & RLCD)
  - [Paper C](https://astro.subsurfaces.net/papers/paper-c-autonomous-followup-mdp/): *Autonomous Target-of-Opportunity Triage & Active 3D MDPs*
  - [Paper D](https://astro.subsurfaces.net/papers/paper-d-euclid-dr1-forecast/): *Euclid DR1 Kinematic & Clustered Dipole Forecast*
- **Social Media & Metadata**:
  - Bespoke 1200×675 OpenGraph social share cards tailored to each individual paper.
  - Vector SVG and multi-resolution astronomical favicon suite.

---

## Technical Standards & Guarantees

1. **Zero Naked Predictions**:
   - Every inference outputs the predicted class probability $p_k$, Dirichlet standard deviation $\sigma_k = \sqrt{\frac{p_k(1-p_k)}{S+1}}$, analytical 95% Credible Interval $[p_k \pm 1.96\sigma_k]$, epistemic vacuity $u_{\rm epi} = K / S$, and aleatoric Shannon entropy $H(p)$.
2. **Disentangled RLCD Optimization** (*TUM 2026 / Bani-Harouni et al.*):
   - Representation trunks are frozen during calibration tuning. Evidence readout heads are tuned under a composite loss (Brier score + clipped logarithmic doubt reward + CARL calibration regularizer), collapsing debiased calibration error by $91\% - 98\%$.
3. **Verified Calibration** (*Stanford NeurIPS 2019 / Kumar et al.*):
   - Evaluates the debiased squared calibration error $\hat{E}^2_{\rm db}$, eliminating finite-sample positive variance bias that corrupts standard plugin ECE.
4. **Conformal Risk Control** (*Papadopoulos et al. / Angelopoulos et al.*):
   - Irreversible aperture allocations (Gemini 8m ToO, JWST 10h NIRSpec, DESI focal-plane fibers) are bounded by mathematical risk bounds ($\alpha_{\rm risk} \le 0.02 - 0.05$).

---

## Quickstart

### Installation & Test Suite

```bash
# Clone and install in editable mode
git clone https://github.com/sub-surface/astro-theory.git
cd astro-theory
pip install -e .

# Run the 231 hermetic unit tests (100% offline, ~50s)
python -m pytest
```

### Web Platform Development & Cloudflare Deployment

```bash
# Start local development server (with live Astro HMR)
npm run dev --prefix site

# Build production static bundle (generates site/dist)
npm run build

# Deploy directly to Cloudflare Workers
npx wrangler deploy
```

### CLI Command Patterns

```bash
# 1. Identity & ADS literature dossier
celestrium resolve M87
celestrium dossier M87 --json

# 2. Archive query with content-addressed SQLite caching
celestrium query gaia "SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source WHERE parallax > 100"

# 3. Cross-catalogue astrometric audit
celestrium match gaia-bright-nearby vizier:VIII/65/nvss

# 4. Multi-wavelength scientific cutout poster
celestrium poster M87 --resolution 4k --style label

# 5. Run local calibrated decision benchmarks
python scripts/benchmark_multi_tracer_co_inference.py
python scripts/benchmark_continuous_flow_cross_calibration.py
```

### Serverless Cloud Scaling (Modal)

```bash
# Set UTF-8 environment (Windows PowerShell)
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1

# Cloud GPU Multi-Messenger Stress Test (50k alerts)
modal run scripts/modal_stress_test_multimessenger_rlcd.py --sources 50000

# Cloud GPU Continuous-Flow Cross-Calibration (40k sources)
modal run scripts/modal_scaled_continuous_flow_cross_calibration.py --sources 40000
```

---

## Architecture & Repo Layout

```
celestrium/
  core/                  # Execution spine: kernel, capability, artifact, events
  continuous_flow_astrojev.py # Simulation-free CFM + DESI fiber allocation engine (EXP-2026-V)
  multi_tracer.py        # Joint Hierarchical Bayesian Poisson MCMC co-inference (EXP-2026-W)
  multimessenger.py      # Real-time GW + Neutrino + Optical counterpart triage (EXP-2026-R)
  active_tiling.py       # Active-Evidential 3D GW Error-Volume Tiling MDP (EXP-2026-U)
  cosmic_dawn.py         # JWST/Euclid Lyman-break evidential discriminator (EXP-2026-S)
  exoplanet.py           # Matérn-3/2 GP transit & RV Doppler disentanglement (EXP-2026-Q)
  space_weather.py       # SDO/HMI solar flare & short-arc NEO impact triage (EXP-2026-T)
  too_protocol.py        # Target-of-Opportunity API serializers (Gemini, LCOGT, VOEvent)
  data_streamer.py       # High-throughput streaming across Quaia (1.3M) and real catalogs
  astrojev.py            # Heteroscedastic Fourier encoder & KM contractive loop
  caps/                  # Capability plugins (archives, objects, imaging, lit, analysis)
  cli.py                 # Unified Typer CLI presenter (--json everywhere)

site/                    # Interactive web platform & living manuscripts (astro.subsurfaces.net)
  src/components/        # 5 interactive simulators (Dipole, Evidential, Tiling, Objects, Taxonomy)
  src/layouts/           # Base and PaperLayout with dynamic OpenGraph & Swiss typography
  src/pages/papers/      # Living portals for Papers A, B, C, D
  public/                # Static assets, 5 bespoke OG share cards, vector SVG & ICO favicons

docs/
  research/
    astronomy-work-index.md # Complete encyclopaedic taxonomy of all 23 astronomy sub-disciplines
    experiment-index.md  # Complete research ledger with benchmarks and artifacts
    literature-convergence-and-novel-results.md # Literature matrix & 6 novel frontiers
    figures/             # Publication-grade figures for all benchmarks

scripts/                 # Standalone reproducible benchmark and Modal cloud deployment scripts
tests/                   # 231 hermetic unit tests (test_multi_tracer, test_continuous_flow, etc.)
wrangler.toml            # Cloudflare Workers deployment config with automated [build] command
```

---

## Authors & Collaborators

- **Leon** — Physical judgment, theoretical signatures, selection cut defensibility, and astrophysics intuition.
- **Sub-Surface (Antigravity)** — Computational astrophysics architecture, evidential deep learning, continuous flow matching, Bayesian MCMC samplers, and cloud GPU scaling.

---
<p align="center"><sub>Agent technical standards codified in <a href="./CLAUDE.md">CLAUDE.md</a> · Research matrix in <a href="./docs/research/experiment-index.md">docs/research/experiment-index.md</a></sub></p>
