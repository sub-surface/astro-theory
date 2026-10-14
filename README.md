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

## Website

**[`astro.subsurfaces.net`](https://astro.subsurfaces.net)** is one plain page (`site/src/pages/index.astro`): the programmes,
the T CrB figures, and every experiment with its ledger status and figure. `tests/test_site.py` checks that its statuses
match `docs/ledger.md` and that every figure it lists exists. Build with `npm --prefix site run build`.

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
python scripts/dipole/benchmark_multi_tracer_co_inference.py
python scripts/engines/benchmark_continuous_flow_cross_calibration.py
```

### Serverless Cloud Scaling (Modal)

```bash
# Set UTF-8 environment (Windows PowerShell)
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1

# Cloud GPU Multi-Messenger Stress Test (50k alerts)
modal run scripts/transients/modal_stress_test_multimessenger_rlcd.py --sources 50000

# Cloud GPU Continuous-Flow Cross-Calibration (40k sources)
modal run scripts/engines/modal_scaled_continuous_flow_cross_calibration.py --sources 40000
```

---

## Architecture & Repo Layout

```
celestrium/              the package
  core/ caps/ study/     execution kernel, capability registry, study runner
  archives/              TAP/ADQL, HEASARC, MAST, VizieR, X-Match, SDSS, Data Lab, resolvers, cutouts, ADS/arXiv, feeds
  engines/               evidential ML (AstroJev family), calibration, data streamers
  dipole/ novae/ transients/   programme code (P1, P2, P3); explore/ holds parked synthetic-only modules
  cli.py watchlist.py ...      CLI surface, observing calendar, shared infrastructure
scripts/<programme>/     runnable experiments (dipole, novae, engines, transients, explore, infra)
docs/                    ledger.md (all experiments + audit status), programmes/, results/, figures/, reference/, history/
site/                    web platform (astro.subsurfaces.net); wrangler.toml deploys it
tests/                   hermetic test suite
```

Research programmes: **P1** cosmic dipole / Euclid DR1, **P2** accreting white dwarfs & recurrent novae (T CrB),
**P3** calibrated triage engines on real data, **P4** theory threads. See [`docs/README.md`](./docs/README.md).
The [ledger](./docs/ledger.md) records an audit (2026-10-14) of every earlier result; most pre-October benchmarks
are synthetic-only or need reruns and should not be cited.

---

## Authors & Collaborators

- **Leon**: physical judgment, theoretical signatures, research direction.
- **Claude** (Anthropic): analysis, code, and documentation, working with Leon.

---
<p align="center"><sub>Agent technical standards codified in <a href="./CLAUDE.md">CLAUDE.md</a> · Research matrix in <a href="./docs/ledger.md">docs/ledger.md</a></sub></p>
