# CLAUDE.md — Onboarding & Technical Standards for Celestrium (astro-theory)

Read this first when picking up work in this repo.
(Parent context: `Psychograph/` hub — see `../AGENTS.md`.)

---

## 1. What Celestrium Is

**Celestrium** is an operational astrophysics research instrument operating across three wings:
1. **Theory Workshop & Evidential AI**: Deep evidential decision engines (`celestrium/astrojev.py`, `celestrium/multimessenger.py`), Dirichlet calibration, and Conformal Risk Control.
2. **Experimental Validation & Data Streaming**: Multi-catalog stream ingestion (`celestrium/data_streamer.py`), TAP queries (`celestrium/tap.py`), selection deprojection, and MCMC inference.
3. **Observatory Follow-Up & ToO Dispatch**: Autonomous Multi-Tier Target-of-Opportunity serialization (`celestrium/too_protocol.py`) for Gemini 8m GMOS and LCOGT 1m networks.

**Primary Living Ledgers**:
- Experiments Ledger: [`docs/research/experiment-index.md`](./docs/research/experiment-index.md) (All experiments EXP-2026-A through T).
- Literature Convergence & Novelty Matrix: [`docs/research/literature-convergence-and-novel-results.md`](./docs/research/literature-convergence-and-novel-results.md).
- Active Roadmap: [`docs/roadmap.md`](./docs/roadmap.md).

---

## 2. Repo Architecture & Key Pointers

```
celestrium/                 Core instrument package
  core/                     Kernel: artifact (content-addressed DAG), ledger, capability, events
  caps/                     Capability registry (@capability declarations)
  astrojev.py               Evidential deep learning (Dirichlet, BALD, Krasnoselskii-Mann)
  multimessenger.py         Multi-messenger GW+Neutrino+Optical triage & RLCD doubt head
  data_streamer.py          Memory-mapped real data streaming & heteroscedastic noise
  stream.py                 Streaming triage engine with conformal sets & error bars
  too_protocol.py           Turnkey ToO protocol serializers (Gemini GMOS, LCOGT)
  tap.py                    Unified Table Access Protocol client (pyvo + driver fallback)
  cli.py                    Headless Typer CLI interface (run with --json)
docs/research/              Pre-registered proposals, experiment indexes, and literature dossiers
tests/                      Hermetic test suite (184+ tests, 100% network-independent)
modal_app.py                Distributed GPU cloud training & triage service (Modal)
```

---

## 3. Core Technical Standards & "Gotchas" (Mandatory)

### A. Retention of Analytical Error Bars on Decisions (Zero Naked Predictions)
- **Standard**: Every model decision (classification, candidate triage, follow-up scheduling) MUST retain explicit analytical uncertainties. Never emit naked point probabilities.
- **Formulas**:
  - Dirichlet posterior standard deviation: $\sigma_k = \sqrt{\frac{p_k(1 - p_k)}{S + 1}}$
  - 95% Credible Interval: $[p_k - 1.96\sigma_k, p_k + 1.96\sigma_k]$ (clamped to $[0, 1]$)
  - Conformal prediction set: $C_\lambda(X) = \{ k : p_k \ge 1 - \hat{\lambda}_{\rm CRC} \}$
  - Epistemic vacuity: $u_{\rm epi} = K / (S + K)$
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
- Verify regularly: `python -m pytest` (currently 200 passing tests).

### G. Windows PowerShell UTF-8 Encoding for Modal CLI
- **Gotcha**: Windows PowerShell defaults to `cp1252` encoding, causing Modal CLI to crash with `'charmap' codec can't encode character '\u2713'` (checkmark) when rendering terminal status.
- **Standard**: Always prefix Modal CLI commands with `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; modal run ...`.

---

## 4. Hardware & Cloud Environments

- **Local Machine**: Windows 11 host (PowerShell), NVIDIA GeForce RTX 2060 (6 GB VRAM). Fast local iteration and unit testing.
- **Modal Cloud Compute**: Active grant balance ~$22.16 USD. Use for scaled H100 SXM5 multi-GPU training, large-scale Monte Carlo runs, and broker streaming workers (`modal run modal_app.py`).

---

## 5. Collaboration Conventions

- Three agents (Claude, Codex, Gemini) collaborate here under the local git identity `Sub-Surface`.
- Always check `git status` and test suite before modifying code.
- Avoid introducing extra markdown files as redundant sources of truth; update [`docs/research/experiment-index.md`](./docs/research/experiment-index.md) and [`docs/research/literature-convergence-and-novel-results.md`](./docs/research/literature-convergence-and-novel-results.md).
