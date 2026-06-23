# Research directions — leverage map & expanded candidates

Companion to [`README.md`](./README.md). The README holds the four "sexy" candidate
projects (A–D). This file adds the **higher-leverage / less-glamorous** areas and, more
importantly, sets out a **field-cartography** approach so our choices are guided by *where
effort is and isn't being spent*, not just by what's exciting.

## The leverage thesis

Excitement and leverage are weakly correlated. A crowded frontier (e.g. "explain JWST's
early massive galaxies") has hundreds of sharp groups racing — our marginal contribution
is ~zero. A neglected-but-tractable corner can be wide open precisely because it's
unglamorous or needs a particular cross-disciplinary angle. So we want an **effort × tractability**
matrix and to fish in the **low-effort / desk-tractable / scientifically-live** quadrant.

```
                  high tractability (desk-doable)        low tractability (needs big iron)
high effort   │  crowded — low marginal value         │  pro-dominated — spectate only
(crowded)     │  (e.g. yet another H0 estimate)        │  (e.g. JWST high-z luminosity fn)
──────────────┼───────────────────────────────────────┼──────────────────────────────────
low effort    │  ★ THE TARGET QUADRANT ★               │  parked — note for the future
(neglected)   │  (curation/re-analysis, method audits, │  (needs LISA/ET/next survey)
              │   cross-matches nobody bothered with)  │
```

A–D from the README mostly already sit in the target quadrant (A: selection-purity
re-analysis; B: re-analysis of a now-frozen archive; C: a census nobody maintains;
D: a falsification map). The two additions below are deliberately chosen for leverage,
not glamour.

## Expanded candidates

### E. Galaxy formation — *don't* chase the crowded frontier; find its quiet flank
The headline — JWST's unexpectedly **massive, bright, mature galaxies at z ≳ 10** — is one
of the most crowded problems in the field (luminosity-function excess, bursty SF + Eddington
bias, top-heavy IMF, enhanced SF efficiency, early SMBHs, even modified inflation spectra
are all being raced). **We should not compete there directly.** The leverage is on the flanks:
- **Method/systematics audits** of the photometric-redshift and stellar-mass pipelines that
  produce the "impossible galaxies" — how much of the tension is mass/redshift inference
  error? This is desk work on public catalogs (CEERS, JADES, COSMOS-Web are open).
- **Cross-matching/meta-analysis**: collate the claimed z≳10 massive galaxies across papers,
  normalise their assumptions (IMF, dust law, SFH prior), and see how much of the "excess"
  survives a common pipeline. This is a *census* in the spirit of project C.
- Anchors: massive galaxy at z≈... 400 Myr after BB (Feb 2026 JWST); "Accelerated Structure
  Formation" (arXiv:2406.17930); "Cosmic Rush Hour" (arXiv:2509.19427); halo mass functions
  at high-z (arXiv:2408.15194); "Impossible Galaxies & the Hubble Tension" (arXiv:2501.04065).

### F. Reheating & the post-inflationary equation of state (the Copeland thread)
This is the high-leverage one: reheating is the **least observationally constrained epoch**
of the whole cosmic history (between the end of inflation and BBN), yet it's increasingly
*falsifiable* via the primordial gravitational-wave spectrum. The physics Ed Copeland's
circle works on:
- **Oscillons** — after inflation the inflaton condensate fragments into long-lived
  soliton-like lumps that can drive an **early matter-dominated phase**; their rapid final
  decay produces **enhanced induced GWs**. Lozanov, Sasaki & Tränkle (2026, **arXiv:2601.11360**,
  PRD) use the ΔN_eff bound on this GW background to **constrain the inflaton mass and cubic/
  quartic self-couplings** — regions inaccessible to the CMB. (Copeland is a foundational
  figure on oscillons; this is the modern, observational descendant of that line.)
- **The reheating equation of state imprints on the inflationary GW spectrum** — the
  expansion history during reheating tilts/breaks the primordial tensor spectrum, so a
  measured GW spectrum becomes a probe of the unknown post-inflationary universe
  (Nottingham, PRD Nov 2025, https://link.aps.org/doi/10.1103/81wm-jlm6). Related:
  "Self-resonance preheating in deformed attractor models" (arXiv:2602.07972, Feb 2026);
  GW from particle decays during reheating (ScienceDirect S0370269324003654);
  Resonant Reheating (arXiv:2404.16090).
- **What's concretely falsifiable / desk-doable for us:** the mapping
  *(inflaton potential shape, reheating T, EoS history) → GW spectral feature → detector
  band (LISA / ET / PTA / DECIGO)*. A clean **candidate-model table** — for each well-posed
  inflation+reheating model, the predicted GW peak frequency/amplitude and which experiment
  could see it — is exactly the kind of structured, literature-derived artifact we can build,
  and it doubles as a "what could validate Copeland's program" scorecard. Strong **D**-style
  project with a real theorist's payoff.

## Field cartography — building the effort map (the meta-project, "C′")

To make the leverage thesis operational rather than vibes-based, build an actual map of
where the field spends its effort, then read the gaps.

- **Empirical anchor:** "Astrophysics Wrapped 2025 — Year-in-Review of Every Astrophysics
  arXiv Paper from 2025" (**arXiv:2602.12303**) statistically profiles the *entire* year of
  astro-ph: keyword/subfield frequencies, telescopes, phenomena (GW, GRB, FRB, exoplanets),
  an "Astrophysical Spectral Fingerprint" (effort by wavelength and by redshift), 13 tables /
  24 figures. **First task: pull the full PDF and extract its tables** to get hard per-topic
  paper counts — the backbone of our effort axis.
- **Our own slice:** scrape arXiv `astro-ph.CO` + `gr-qc` listing counts by month/keyword
  (the API is free) to track *trajectories* — what's accelerating (crowded, avoid) vs flat/
  declining-but-unresolved (possible neglected leverage).
- **Output:** a living table of {topic → recent paper volume, trend, desk-tractability,
  open-question status} → ranked shortlist for the target quadrant. This *is* project C
  generalised from "one tension" to "the whole field's attention budget," and it should
  drive which of A–F we invest in.

## Honest status of what we know vs. still need

- We have the *structure* of the effort map and the empirical source (2602.12303) but **not
  yet the extracted per-topic counts** — that PDF was too large to pull in one shot; needs a
  chunked fetch or arXiv-API tally. Flagged as task 1 for C′.
- F's attribution: oscillon→GW and reheating-EoS→GW are real, current, and falsifiable; the
  specific 2026 oscillon-constraint paper is Lozanov/Sasaki/Tränkle. We should read
  Copeland's own recent papers/talks to pin exactly which validation *he* is pursuing before
  committing F's framing. (Get the exact reference from Leon if he can find it.)

## Updated shortlist

| Project | Quadrant | Type | First build? |
|---|---|---|---|
| A wide-binary gravity | low-effort / desk | re-analysis, real result | **yes** |
| C tension census | low-effort / desk | living DB | **yes (background)** |
| C′ field effort map | low-effort / desk | meta / steering | start small alongside C |
| F reheating GW scorecard | low-effort / desk | theory table, high payoff | strong next |
| B Dyson re-analysis | low-effort / desk | frozen-archive re-analysis | later |
| D PBH attack-surface | low-effort / desk | falsification map | later |
| E galaxy-formation audit | flank of crowded | census/systematics | opportunistic |
