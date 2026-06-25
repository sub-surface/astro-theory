# Theoretical threads — the idea ledger

The theory wing's running list of **lines of inquiry**: physics questions and methodological
bets we're thinking about, *before* (and whether) they harden into a concrete build. This is
deliberately more speculative than its neighbours:

- [`directions.md`](./directions.md) / [`candidates.md`](./candidates.md) / [`field-map.md`](./field-map.md)
  rank the **data-project backlog** (A–H) on leverage — the "what to pull and test" map.
- [`../roadmap.md`](../roadmap.md) tracks the **Celestrium tool's** technical development.
- **This file** tracks the *ideas* — including ones that may never become a data pull, or that
  cut across several projects. A thread graduates to `directions.md` when it becomes a ranked,
  desk-tractable build.

**Thread template:** one-line question · why it's live · key refs · honest desk-tractability ·
status (`watch` / `exploring` / `graduating` / `parked`) · cross-links.

---

## T1 · Non-Gaussian tails, ζ⁴, and "beyond perturbation theory" (the Creminelli line)
**Question.** Primordial black hole (PBH) abundance is set by the *far tail* of the probability
distribution P(ζ) of the curvature perturbation — you need ζ above an O(1) collapse threshold.
But that tail is exactly where the standard expansion in correlators (bispectrum ⟨ζ³⟩↔f_NL,
trispectrum ⟨ζ⁴⟩↔g_NL/τ_NL) breaks down: for ζ ≳ 1/f_NL each higher cumulant matters *more*, not
less, so people "resort to numerics" (stochastic-inflation Fokker–Planck, δN, lattice).

**Why it's live.** Creminelli & collaborators' **"Beyond perturbation theory in inflation"**
(2021, JCAP 06 051) computes the tail non-perturbatively via a semiclassical / saddle-point
(ħ→0) treatment — resumming all tree-level Witten diagrams — giving a tail like
P(ζ) ~ exp(−c·ζ^(3/2)/√λ), *non-analytic* in the coupling. The 2024 follow-up
("Non-perturbative wavefunction of the universe…", JHEP 03 010) and 2026 "Large n-point
functions in resonant inflation" (JCAP 01 064) push the same program. The community tail
literature is converging here too (e.g. "non-perturbative non-Gaussianity and PBHs"
arXiv:2211.08348; "The hand-made tail" JHEP 05 (2022) 052; "Large power spectrum and PBHs in
the EFT of inflation" JHEP 01 (2022) 074). The talk's "ζ⁴ for PBHs at the tail" is the entry
point; the "and beyond" is precisely that truncating at ζ³/ζ⁴ is *not enough* — the tower
resums into a non-perturbative exponential.

**Desk-tractability.** The *analytics* (saddle-point profiles, large-deviation / instanton
methods for P(ζ)) are genuinely paper-and-pencil + light numerics — within reach of a theory
desk with the right hands. The *forecasting* angle (which inflation+tail model predicts what PBH
mass function, vs. which observational window) is a structured-table artifact we build well.

**Status:** `exploring`. Refs added to `refs.bib`. Cross-links: PBH attack-surface (project **D**,
[`candidates.md`](./candidates.md)); reheating/oscillons (project **F**, [`directions.md`](./directions.md)).

---

## T2 · Learned PDE surrogates for the "resort to numerics" bottleneck (incl. Neural Cellular Automata)
**Question.** The recurring frustration in T1 (and across inflation/reheating) is being forced
into expensive numerics. Can data-driven PDE methods — in particular **Neural Cellular Automata**
(*Discovering Partial Differential Equations With Neural Cellular Automata*, Artificial Life,
2026, [doi:10.1162/ARTL.a.454](https://doi.org/10.1162/ARTL.a.454)) — either *accelerate* those
simulations or *discover effective* equations for the coarse dynamics?

**What NCA actually are.** A small neural net applies a *local* update rule repeatedly over a
grid; its learned convolution kernels are structurally finite-difference stencils, so a trained
NCA *is* a discretized PDE and the kernels can be read back as differential operators. It's a
learnable, differentiable cousin of sparse-regression discovery (SINDy / PDE-FIND), reportedly
strong on pattern formation and on learning long-range dynamics from sparsely sampled data.

**Honest read on the promise (see the writeup below for the long version).**
- ✅ **Strongest fit — lattice surrogates, not the ζ-tail directly.** NCA are *local grid update
  rules* = a near-perfect structural match to **lattice field theory time-stepping**. The sharpest
  "resort to numerics" pain with that shape is **preheating / oscillon** lattice simulations
  (project **F**, the Copeland thread). An NCA/neural-operator surrogate trained on lattice
  snapshots could make parameter scans cheap. That's the concrete entry point.
- ⚠️ **Weaker fit — the ζ-tail itself.** There the governing equations (stochastic-inflation
  Fokker–Planck; the saddle-point profile ODE/PDE) are largely *known* — so "discovery" isn't the
  bottleneck; *rare-event sampling* is. ML-for-rare-events (normalising flows for importance
  sampling) is a better-matched tool than NCA, and Creminelli's saddle-point is the analytic
  answer. NCA discovering an equation you already know analytically is not the win.
- ⚠️ **Interpretability gap.** NCA yields a *discretized operator*, not necessarily a parsimonious
  *symbolic* PDE; extracting clean physics from the learned rule is non-trivial (SINDy is more
  interpretable by construction).
- ⚠️ **Compute/skill mismatch with our edge.** Training surrogates needs GPU compute + ML
  expertise — outside a desk team's current strengths.

**The desk-tractable sub-question** (this is the part *we* could actually do): a **literature
meta-analysis** — "which numerical bottlenecks in the inflation/PBH/reheating program have the
right structure for ML surrogates (local, grid-based, smooth) vs. which don't (rare-event,
global, stiff)?" That is a census in the spirit of project C/C′, on-brand for our wing, and it
would tell us whether T2 ever deserves to graduate.

**Status:** `watch`. Cross-links: T1 (the motivating bottleneck); project **F** (lattice
surrogates); project **C′** (the meta-analysis framing).

---

## Backlog (unstarted threads to flesh out later)
- Consistency relations / squeezed limits as model-independent discriminators (the other half of
  the Creminelli toolkit) — does any live anomaly touch them?
- EFT-of-inflation operator basis → which operators are observationally reachable by 2027 data.
