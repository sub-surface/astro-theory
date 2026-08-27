# Modal ML Project Brainstorm

Context: a small Modal account with roughly $30/month of burst compute. Best used for scale-to-zero inference, embedding/indexing, narrow fine-tunes, eval sweeps, synthetic data, anomaly triage, and artifact generation rather than always-on GPU services.

## Shortlist from the first pass

### 1. Embedding Map of the Internet's Weird Corners

Crawl niche public archives, forums, datasets, bibliographies, or subcultural corpora; embed them; cluster latent topics; surface strange conceptual neighborhoods and overlooked connections.

Why it is interesting:
- Turns obscure text worlds into navigable maps.
- Could support research, writing, market discovery, aesthetic scouting, or cultural intelligence.
- Works well with cheap CPU embedding batches and occasional GPU reranking.

Possible shape:
- Start with one corpus.
- Produce a static atlas: clusters, representative passages, surprising bridges, and "unknown unknown" prompts.
- Later expose as a paid research artifact or custom mapping service.

### 2. Ad Creative Evolution Engine

Generate hundreds of ad variants, score them with multimodal models, mutate the strongest candidates, and output a compact set of high-performing directions.

Why it is interesting:
- A real business use case for iterative ML search.
- Uses cheap bursts: image/text generation, scoring, and ranking.
- Could target small brands, musicians, indie games, Etsy sellers, or local businesses.

Possible shape:
- Input: product URL, audience, constraints, brand voice.
- Output: copy variants, image concepts, ranked hypotheses, and test matrix.
- The real edge is not generation, but model-guided evolutionary search plus clear rationale.

### 3. Autonomous Literature Hypothesis Miner

Ingest papers from a field, extract claims, assumptions, contradictions, neglected variables, suggested experiments, and "if A and B are true, test C" hypotheses.

Why it is interesting:
- High leverage for research-heavy domains.
- Plays well with your existing taste for mechanism-to-observable maps.
- Can produce useful artifacts without requiring giant training runs.

Possible shape:
- Start with a focused field and a curated seed bibliography.
- Extract claim graphs, observable hooks, dataset hooks, and weak assumptions.
- Score hypotheses by novelty, testability, and data availability.

### 4. Prediction-Market Research Agent

An ML system that reads news, papers, filings, datasets, and market questions; generates calibrated forecasts; tracks resolution; and learns from its own misses.

Why it is interesting:
- Potentially lucrative if it becomes even slightly useful.
- Produces a continuous benchmark of reasoning quality.
- Combines retrieval, calibration, decomposition, and evaluation rather than just chat.

Possible shape:
- Start with one domain and public resolved questions.
- Build a pipeline: question decomposition, evidence retrieval, base-rate search, forecast, confidence, postmortem.
- Treat it as an evaluation system first, trading system second.

## Machine Learning for Astronomy

This feels like a particularly good fit for small burst compute: astronomy has enormous public data, expensive human attention, sparse labels, and many workflows where a useful model does not have to be frontier-scale. The promising angle is not "train a giant astronomy model"; it is to build clever, inspectable systems that rank, triage, compress, and connect observations to hypotheses.

### Field Notes

- Survey scale is exploding. Rubin/LSST is designed around a real-time alert stream, with official Rubin material describing millions of alerts per night. That makes triage, anomaly detection, prioritization, and broker-side intelligence central rather than optional.
- Euclid and other space surveys are producing enormous imaging catalogs where morphology, lenses, artifacts, redshifts, and weak-lensing-quality measurements all need automation plus human verification.
- The ML frontier is moving toward astronomy foundation models: masked/contrastive pretraining over images, spectra, light curves, and metadata, then adaptation to downstream tasks.
- A recurring gap is transfer: models often work well within one survey/instrument/simulation distribution, then become brittle across cadence, noise, wavelength, PSF, selection effects, and calibration regimes.
- Another gap is epistemic humility. Astronomers do not only need a class label; they need "why this object", uncertainty, nearest known analogues, failure modes, and what observation would reduce ambiguity.

Useful source anchors:
- Rubin alert-scale context: https://rubinobservatory.org/
- Rubin data/broker ecosystem: https://www.lsst.org/scientists/alert-brokers
- Euclid mission/data context: https://www.esa.int/Science_Exploration/Space_Science/Euclid
- AstroM3 multimodal astronomy foundation model: https://arxiv.org/abs/2411.08842
- FALCO, a foundation model for astronomical light curves: https://arxiv.org/abs/2504.20290

### Gaps That Look Promising

1. Human attention is the scarce resource, not GPU time.
2. Cross-survey transfer is under-solved.
3. Astronomical anomaly detection often lacks useful explanations.
4. Synthetic data is abundant, but simulation-to-real mismatch is dangerous.
5. Follow-up resources are limited, so ranking what to observe next matters.
6. Literature, survey data, and object-level evidence are still poorly joined.
7. Small labs and amateurs can access data, but not always the tooling needed to make it tractable.

### Potential Projects

#### 1. Astronomical Anomaly Triage Workbench

Build a pipeline that embeds light curves, spectra, images, and metadata; identifies outliers; then explains each candidate with nearest neighbors, possible known classes, confidence, and recommended follow-up.

Why it fits Modal:
- Batch embedding and clustering can run cheaply.
- Occasional GPU jobs can process new object batches.
- Results can be served as static reports or a lightweight web app.

Why it might matter:
- "Anomaly" alone is weak. "This is anomalous because it resembles class X in cadence but class Y in color, and no nearby training examples share both" is much more useful.

#### 2. Cross-Survey Translator

Train small models that map object representations between surveys: e.g. ZTF/Rubin-like light curves, Euclid/ground-based image features, Gaia + photometry + spectra feature spaces.

Goal:
- Learn which features survive instrument differences and which predictions break under domain shift.

Possible output:
- A benchmark suite and practical transfer-risk score for a model or object class.

Why it is interesting:
- This is less glamorous than discovery, but potentially foundational. Survey transfer is where many ML astronomy tools become fragile.

#### 3. Follow-Up Value Estimator

Given an object and limited follow-up budget, estimate which next observation would be most informative: another photometric band, higher cadence, spectrum, deeper imaging, or no follow-up.

ML form:
- Active learning / Bayesian decision model / uncertainty reduction predictor.

Possible first domain:
- Transients, variable stars, supernova candidates, strong lens candidates, or unusual galaxy morphologies.

Why it could be high leverage:
- Telescope time and human review are expensive. A decent prioritizer could be genuinely useful even if it is simple.

#### 4. Literature-to-Sky Hypothesis Miner

Combine the "Autonomous Literature Hypothesis Miner" with astronomical archives. Read a small literature area, extract predicted observational signatures, then search public catalogs for candidate objects or fields.

Example:
- Paper claims a mechanism should produce a certain light-curve shape, color evolution, spectral line, or spatial environment.
- System turns that into catalog queries and candidate-ranking logic.

Why it fits your style:
- It joins theory, observables, and data rather than stopping at summaries.

#### 5. Citizen-Science Copilot for Rare Objects

Build a tool that helps volunteers or amateurs inspect candidate lenses, weird galaxies, transients, or artifacts. It gives calibrated hints, similar examples, and reasons to agree/disagree, without replacing human judgment.

Why it is promising:
- Human pattern recognition remains valuable.
- The tool can amplify attention while preserving interpretability.

Potential public artifact:
- A weekly "strangest sky objects" gallery with model explanations and links to data.

#### 6. Simulation-to-Real Failure Finder

Train on simulated astronomical data, test on real survey data, and automatically identify where the model's confidence fails. Produce a report of covariate shift, missing noise sources, and misleading synthetic shortcuts.

Why it is useful:
- Astronomy depends heavily on simulations.
- A small, rigorous failure-finding tool could be more valuable than yet another classifier.

#### 7. Tiny Astronomy Foundation Model Benchmark

Do not try to beat major labs. Instead, build a compact benchmark that evaluates small open models and embeddings across practical tasks: morphology, photometric redshift proxying, transient classification, anomaly retrieval, and spectra tagging.

Why it fits $30/month:
- Small datasets, cached embeddings, periodic eval sweeps.
- The artifact is the benchmark and leaderboard, not an expensive model.

Why it could become useful:
- Many people need to know which cheap model is "good enough" for a specific astronomy workflow.

### Best First Bet

Start with the Astronomical Anomaly Triage Workbench, then connect it to Literature-to-Sky Hypothesis Mining. The first gives a concrete ML artifact; the second gives it a more original research identity. Together they point toward a system that does not merely classify the sky, but asks: "What is strange, why is it strange, and what would make it scientifically meaningful?"

## Theory-ML Cosmology Avenues

These are distinct from ordinary astronomy data-analysis projects. The goal is to use ML as a theoretical instrument: discover compact structure, test which dynamics are learnable, generate model-space maps, and turn learned representations back into physics.

### 1. Epiplexity Map of Cosmological Numerics

Measure which cosmological systems are learnable by bounded models and which are effectively irreducible at a given compute budget. Candidate domains: PBH curvature-perturbation tails, stochastic-inflation trajectories, reheating / preheating toy lattices, oscillon dynamics, CMB maps, and cosmic-web fields.

Core thesis:
- Measure first, build surrogates second.
- Use known analytic cases as calibration checks.
- Rank where ML can actually help theory, rather than assuming it can.

### 2. Rare-Event Engine for PBH Tails

Use normalising flows / importance sampling to reach rare curvature-perturbation tails that brute-force sampling cannot touch. Validate against non-perturbative saddle-point or large-deviation results.

Core thesis:
- PBH abundance lives in the distribution tail.
- The ML contribution is not "predict PBHs" but making rare-event theory numerically reachable.
- Tail work needs conservative probability updates, good validation cases, and explicit uncertainty.

### 3. Reheating-to-GW Differentiable Emulator

Build a small emulator mapping inflaton potential parameters, reheating equation-of-state history, and decay/channel assumptions to gravitational-wave spectral features and detector bands.

Core thesis:
- Reheating is observationally underconstrained but increasingly testable through GW spectra.
- The useful artifact is a falsifiable map: model family -> spectral feature -> detector.
- Epiplexity should decide whether an emulator is worth training for each toy regime.

### 4. Symbolic Inflation / Reheating Discovery

Use symbolic regression after ML triage to recover compact relations from simulations or learned emulators: potential shape -> tensor tilt, reheating history -> spectral break, tail parameters -> PBH mass-function scaling.

Core thesis:
- The model is disposable; the equation is the result.
- If the learned structure cannot be compressed into a legible formula or scaling law, it is less useful as theory.
- This complements, rather than replaces, analytic work.

### 5. Simulation-Based Falsifiability Forecasts

Use SBI to ask which future measurements could actually distinguish theory families under realistic degeneracies, selection effects, and survey geometry.

Core thesis:
- Do not merely forecast constraints; forecast whether rival models are distinguishable at all.
- Useful targets: Euclid dipole/isotropy tests, reheating/GW signatures, PTA backgrounds, and selection-systematics-heavy anomalies.

### 6. Mechanistic Interpretability of Astronomy Foundation Models

Interpret models such as AstroPT, AstroCLIP, or FALCO as learned coordinate systems over astronomical phenomena. Search for internal features corresponding to known physical variables, then look for stable unknown features that predict real observables.

Core thesis:
- A pretrained astronomy model is not just a classifier; it is a compressed representation of the sky.
- If internal directions correspond to redshift, morphology, dust, metallicity, lensing, age, or variability class, they can be audited as machine-discovered observables.
- The high-upside win is a latent feature that is stable, predictive, physically meaningful, and not already a standard variable.

### 7. The Main Ladder

Working name candidate: **Celestrial Semantics**.

Ladder:
1. Start from pretrained astronomy foundation models and small theory/simulation models.
2. Mechanistically identify learned features that correspond to known physics.
3. Search for stable unknown features.
4. Test those features against simulations, catalogues, and theory predictions.
5. Use epiplexity to decide where learned structure is compressible.
6. Use symbolic regression or analytic fitting to turn useful features into equations, selection rules, or new phenomenological variables.
7. Feed the results back through Celestrium: literature, observables, archive pulls, validation, and reports.

Possible one-line identity:
> A project to recover the hidden physical language learned by models of the sky, then compress it back into cosmological theory.

### 8. Renormalization-Aware ML for Stochastic Cosmological Fields

Source anchor: Bruned, Chandra, Chevyrev, Hairer, "Renormalising SPDEs in regularity structures" (`10.4171-jems-1025.pdf`).

Core idea:
- Singular stochastic PDEs can converge to the wrong or ill-defined object under naive discretization.
- The regularity-structures/BPHZ framework says the correct continuum object often requires explicit counterterms.
- This matters for ML because neural PDE surrogates trained on discretized stochastic fields may silently learn grid-scale artifacts instead of the renormalized continuum dynamics.

ML angles:
- Learn the counterterms required for cutoff-invariant stochastic dynamics.
- Train renormalization-aware neural operators and test whether predictions remain stable under mollifier/grid changes.
- Use symbolic regression to recover counterterm structure from multiscale simulations.
- Use regularity-structure trees as a mathematically exact feature grammar for stochastic dynamics.
- Build SPDE epiplexity benchmarks where symbolic complexity is known from renormalization theory.

Cosmology hook:
- Stochastic inflation, reheating-like scalar-field dynamics, phase transitions, noisy effective fields, and early-universe stochastic processes can all become dangerous if learned at one resolution and trusted at another.
- The high-level warning: if we train on stochastic field simulations without respecting renormalization, the model may learn the wrong continuum theory.

Possible project:
> Renormalization-Conscious Neural Fields: neural operators for stochastic cosmological fields whose learned dynamics are explicitly tested for cutoff invariance and counterterm structure.

### 9. Learned Lyapunov / Harris Diagnostics for Cosmological Inference

Source anchor: Hairer and Mattingly, "Yet another look at Harris' ergodic theorem for Markov chains" (`0810.2777v1.pdf`).

Core idea:
- A Markov chain has a unique invariant measure and exponential convergence if it satisfies a Lyapunov drift condition plus a small-set/minorization condition.
- The paper gives a clean contraction proof using tunable weighted total-variation norms.

ML angles:
- Learn Lyapunov functions for complicated stochastic simulators or inference chains.
- Use learned Lyapunov candidates to diagnose sampler stability and mixing.
- Design normalizing-flow/MCMC hybrids with explicit drift/minorization checks.
- Use Harris-style criteria to evaluate agentic theory-search loops over model space.
- Apply the same attractor/stability language to stochastic versions of dark-sector SOC dynamics.

Cosmology hook:
- SBI, MCMC, sequential Monte Carlo, neural posterior estimation, stochastic reheating models, and flow-based rare-event samplers all need stability diagnostics.
- This is a practical theoretical tool: not a flashy model, but a way to know whether an inference loop actually converges rather than merely producing nice traces.

Possible project:
> Learned Lyapunov Certificates for Cosmological Simulators: train neural Lyapunov candidates over parameter/dynamical state space, then test whether an SBI sampler, stochastic reheating model, or dark-sector process is stable/mixing.

Take:
- The SPDE renormalization paper is the deeper wild-theory ML hook: it says ML over stochastic fields must respect the continuum limit.
- The Harris paper is the practical inference hook: it gives a stability/mixing lens for flows, MCMC, SBI, stochastic simulators, and learned search processes.

### 10. Relational / Gauge-Dressed Representation Learning

Source anchor: Susskind, "Is Time Reversal in de Sitter Space a Spontaneously Broken Gauge Symmetry?" (`2603.12434v1.pdf`).

Core idea:
- In de Sitter/static-patch holography, physically meaningful bulk observables may need dressing to a physical clock/reference.
- Gauge-variant scaffold variables can be useful internally, but only dressed/gauge-invariant operators behave semiclassically.
- Susskind's signal for hidden time-reversal gauge symmetry is nonlocal: long-range order / cluster-decomposition failure plus a holonomy exchanging forward-going and backward-going clocks.

ML angles:
- Train models on relational or gauge-invariant observables rather than coordinate/scaffold variables.
- Use representation learning to distinguish physical variables from gauge artifacts.
- Search toy holographic/quantum systems for hidden symmetry breaking through nonlocal correlation diagnostics.
- Use graph/geometric learning to detect holonomies or global topological features rather than local field patterns.
- Treat learned internal variables as scaffolds: useful for computation, but suspect until compressed into gauge-invariant/dressed observables.

Cosmology hook:
- In cosmology and de Sitter-like settings, "time", "observer", and "local observable" are subtle.
- Any ML system trained naively on coordinate-dependent data may learn gauge scaffolding rather than physics.

Possible project:
> Dressed Observable Learning: train models to predict only relational/gauge-invariant quantities, then audit whether their latent space separates scaffold variables from physically meaningful dressed variables.

Take:
- This is not the first build, but it is conceptually important for Celestial Semantics.
- It says the learned language of the sky may need to be relational, nonlocal, and dressed to physical reference systems.

### 11. Complexity Barriers In Cosmological Dynamics

Source anchor: Susskind, "Computational complexity and black hole horizons" (`Computational complexity and black hole horizons -- Leonard Susskind.pdf`).

Core idea:
- Black-hole interior/horizon structure is controlled not just by entanglement, but by computational complexity.
- Entanglement saturates around scrambling time, while complexity keeps growing until exponentially long times.
- Some states or operations, such as firewall-producing precursors, may be physically possible but computationally inaccessible.

ML angles:
- Build toy quantum-circuit benchmarks where entanglement saturates but complexity continues growing.
- Test whether models learn genuine complexity-growth structure or collapse to entanglement proxies.
- Study precursor learning: can ML approximate `U(t) W U†(t)`, and where does that become computationally inaccessible?
- Use complexity barriers as a theory-search diagnostic: distinguish formally possible regions from reachable regions under bounded compute.
- Connect this to epiplexity: some structure may be real but inaccessible to observers/models below a compute threshold.

Cosmology hook:
- Reheating, stochastic inflation, black-hole interiors, and quantum-gravity toy models may contain real structure hidden behind computational barriers.
- The question is not just "does a pattern exist?" but "is it reachable by bounded inference?"

Possible project:
> Complexity Barriers in Cosmological Dynamics: use quantum-circuit and field-theory toy models to map the boundary between physically possible and computationally reachable structures, then test whether ML can learn that boundary.

Take:
- This is a strong conceptual ingredient for the ML-theory agenda.
- It gives a physical language for bounded-observer learnability: some variables may be real, but only become visible past a scrambling/complexity threshold.
