<h1 align="center">✦ Celestrium ✦</h1>
<p align="center"><em>A three-wing computational astrophysics instrument and calibrated evidential decision engine.</em></p>
<p align="center">
  <a href="https://astro.subsurfaces.net/"><strong>🌐 Interactive Research Platform: astro.subsurfaces.net</strong></a>
</p>

---

Celestrium is an open computational astrophysics workbench for literature review, archive access, analysis, and uncertainty-aware decision experiments.

It includes interfaces to public astronomical archives, analysis modules, a CLI, and an interactive website. Archive support and generated follow-up payloads do not imply that every dataset has been analyzed end-to-end or that an observatory has received a request.

> **Research status:** Repository benchmarks and manuscript drafts are internal work products, not peer-reviewed findings. Quantitative claims are being audited for label validity, independent evaluation, uncertainty reporting, and reproducibility. See the [evidence and release roadmap](https://astro.subsurfaces.net/roadmap/).

---

## The Three Wings

```
+---------------------------------------------------------------------------------------------------+
|                                       THE CELESTRIUM ENGINE                                       |
+---------------------------------------------------------------------------------------------------+
|  🔭 THEORY WORKSHOP                  🧪 EXPERIMENTAL VALIDATION        🎨 TIME-DOMAIN & DECISION ENGINE   |
|  * ADS/SciX literature harvest       * 8 TAP/ADQL public archives      * Continuous-Flow CFM Latent SED   |
|  * Living bibliographies & dossiers  * Multi-catalogue cross-matching  * Disentangled RLCD Optimization   |
|  * Theory signature → cuts           * Provenance-logged SQLite cache  * Decision & ToO payload tools     |
|  * Pre-registered experiment ledger  * Statistical analysis modules   * Optional cloud experiments       |
+--------------------------------------+---------------------------------+----------------------------------+
```

### 🔭 1. Theory Workshop — *Literature in, defensible signatures derived*
Harvests ADS/SciX, cross-references astronomical targets to living bibliographies, tracks field consensus, and maintains pre-registered hypothesis ledgers.
- Commands: `celestrium papers · cite · resolve · dossier`

### 🧪 2. Experimental Validation — *Hermetic multi-survey audit & co-inference*
Executes provenance-tracked queries across eight astronomical archives (Gaia DR3, Euclid, CatWISE, NVSS, MAST, VizieR, DESI, HEASARC) through a content-addressed SQLite ledger (`data/celestrium.db`). Evaluates all-sky cosmological likelihoods across millions of real sources.
- Commands: `celestrium query · sample · match · log · where · field`

### 🎨 3. Time-Domain & Decision Tools — *Prototype triage and payload generation*
Contains experimental evidential models, streaming utilities, and serializers for follow-up payloads. These tools do not submit observations to facilities; performance claims require the release gates below.
- Commands: `celestrium image · poster · atlas-targets · stream`

---

## Research Status

The experiment index is an internal working ledger, not a certification that listed results are independently validated. Manuscript and benchmark claims are withheld from the public website while their data provenance, label validity, evaluation design, and machine-readable artifacts are audited. Current blockers and release criteria are tracked on the [research roadmap](https://astro.subsurfaces.net/roadmap/).

---

## 🌐 Interactive Research Platform & Manuscript Status

Celestrium's interactive web platform is deployed at **[`astro.subsurfaces.net`](https://astro.subsurfaces.net)**. It provides project scope and a research-status noticeboard. Manuscript, catalog, and taxonomy details are withheld or marked in preparation until their release evidence is checked.

- **Research roadmap**: release gates, status definitions, and outstanding validation work.
- **Social Media & Metadata**:
  - Bespoke 1200×675 OpenGraph social share cards tailored to each individual paper.
  - Vector SVG and multi-resolution astronomical favicon suite.

---

## Intended Engineering Conventions

These are implementation goals, not guarantees that every module currently satisfies. Check each method's assumptions and tests before relying on its output.

<details>
<summary>Current model and calibration conventions</summary>

1. **Uncertainty reporting**:
  - For Dirichlet outputs, report the posterior parameters and uncertainty summaries with their assumptions. A normal approximation such as $p_k \pm 1.96\sigma_k$ is not automatically a calibrated credible interval; verify interval coverage empirically.
2. **Disentangled calibration experiments**:
  - State which model components are frozen and which data fit the calibration head. Evaluate on an independent held-out set and report the scoring rule and baselines.
3. **Calibration reporting**:
  - Report the metric definition, binning, sample counts, uncertainty, and evaluation split. Do not truncate a signed estimator without documenting that transform.
4. **Risk control**:
  - State the target loss and sampling assumptions. Independently validate threshold selection before claiming a risk bound or using it for a costly decision.

---

</details>

## Quickstart

### Installation & Test Suite

```bash
# Clone and install in editable mode
git clone https://github.com/sub-surface/astro-theory.git
cd astro-theory
pip install -e .

# Run the test suite
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
tests/                   # Offline-focused unit and integration tests
wrangler.toml            # Cloudflare Workers deployment config with automated [build] command
```

---

## Authors & Collaborators

- **Leon** — Physical judgment, theoretical signatures, selection cut defensibility, and astrophysics intuition.
- **Sub-Surface (Antigravity)** — Computational astrophysics architecture, evidential deep learning, continuous flow matching, Bayesian MCMC samplers, and cloud GPU scaling.

---
<p align="center"><sub>Agent technical standards codified in <a href="./CLAUDE.md">CLAUDE.md</a> · Research matrix in <a href="./docs/research/experiment-index.md">docs/research/experiment-index.md</a></sub></p>
