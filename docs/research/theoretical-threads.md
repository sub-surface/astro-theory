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

## T2 · Learnability under a compute bound — *epiplexity as the diagnostic* (surrogates downstream)
**Question.** The recurring frustration in T1 (and across inflation/reheating) is being forced
into expensive numerics. Before asking *how* to learn a surrogate, ask the prior question:
**for a computationally bounded observer, how much of this is learnable structure vs. irreducible
noise — at our budget?** That is exactly what **epiplexity** measures.

**Epiplexity (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, 2026, [arXiv:2601.03220](https://arxiv.org/abs/2601.03220)).**
Relative to a compute budget T, it splits a source's information into **epiplexity** S_T (learnable
structure) and **time-bounded entropy** H_T (what *looks* like noise to a T-bounded observer).
Formally S_T(X)=|P⋆| for the program minimizing a time-bounded two-part code
P⋆=argmin_{P∈𝒫_T}{|P|+𝔼[log 1/P(X)]}; H_T(X)=𝔼[log 1/P⋆(X)]. Estimated by **prequential coding**
(area under the training loss curve above the final loss — cheap, heuristic) or **requential
coding** (cumulative teacher→student KL — rigorous, 2–10× compute). Tellingly, the paper's
testbed includes **elementary cellular automata**: Rule 30 (chaotic) → high H_T, *low* S_T =
computationally irreducible; Rule 54 → *high* S_T = learnable emergent structure. "The same object
may appear random or structured depending on the computational resources of the observer."

**Why this is the stronger tool (and reframes the NCA idea).** Epiplexity is a *diagnostic*
(is there a learnable nail, and can our budget reach it?); a surrogate like NCA is a *generative*
hammer. They compose — **measure first, build second** — and epiplexity directly adjudicates the
T1/T2 questions:
- The **ζ-tail** is *epiplectically simple*: Creminelli's saddle-point P(ζ)~exp(−c·ζ^{3/2}/√λ) IS
  the short program (high S_T, low H_T). So epiplexity predicts "don't throw a surrogate at it —
  a cheap closed form exists"; the win is analytic, exactly as Creminelli found. (Caveat: epiplexity
  says a short program *exists*; it doesn't hand you the physics — it's descriptive, not generative.)
- **Preheating / oscillon lattices** (project **F**) are the open case: parametric resonance →
  turbulent fields → is the evolution Rule-54-like (surrogate-worthy) or Rule-30-like (irreducible,
  surrogate will memorise noise and fail to extrapolate)? **Measure the epiplexity of the lattice
  data before sinking compute into any NCA/neural-operator surrogate.** This is the rigorous version
  of "which bottlenecks are ML-tractable."
- A deep reframing of "resort to numerics": simulation *creates* epiplexity a bounded observer
  couldn't access from the initial data + equations alone (the paper's first paradox). "Resort to
  numerics" = the structure is real but locked behind compute; the live question is whether a
  *cheaper* program — analytic (Creminelli) or learned — can unlock it at our budget.

**The desk-tractable build (this is what *we* could actually do — and it fits Leon's ML background):**
an **"epiplexity triage of cosmological numerics."** Generate 2–3 cheap datasets (a 1D
stochastic-inflation ζ-trajectory ensemble; a 1D/2D preheating toy lattice; a δN map), estimate
epiplexity via prequential coding at a fixed FLOP budget, and rank them simple↔irreducible — using
the ζ-tail (known to be epiplectically simple) as a *validation* case the estimator should confirm.
Output: a principled map of where ML surrogates can help in the inflation/PBH/reheating program.
Small-GPT-scale, on-brand for our wing, and hard for a pure-physics *or* pure-ML group to produce.

**Honest caveats.** Epiplexity is (1) **budget-relative** — conclusions can flip with T, so the
FLOP budget must be fixed and reported; (2) **representation-dependent** — measured on a *encoding*
of the field (real-space vs. Fourier modes vs. multipoles), and the right physics variables can
expose structure a naïve tokenisation hides (this is where physics judgment enters, and it rhymes
with the EFT/squeezed-limit choice of variables); (3) **estimator-young** — prequential is
heuristic, requential is compute-hungry; (4) NCA-as-surrogate keeps its own caveats (discretised
not symbolic operator; chaotic fields won't extrapolate) — but now we'd only reach for it *after*
epiplexity says the structure is there.

**Status:** `exploring` (upgraded from `watch` — epiplexity gives the thread a concrete, testable
first move). Cross-links: T1 (the ζ-tail as the epiplectically-simple validation case); project
**F** (preheating lattices = the open epiplexity question); project **C′** (the triage as a
methods census).

---

## Backlog (unstarted threads to flesh out later)
- Consistency relations / squeezed limits as model-independent discriminators (the other half of
  the Creminelli toolkit) — does any live anomaly touch them?
- EFT-of-inflation operator basis → which operators are observationally reachable by 2027 data.
