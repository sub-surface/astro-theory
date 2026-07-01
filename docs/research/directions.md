# Research directions — leverage map & expanded candidates

Companion to [`README.md`](../../README.md). The README holds the four "sexy" candidate
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

## Round 2 — the neglected-anomaly hunt (the structural gap)

After mapping effort (C′) and rejecting holography as high-effort/low-leverage, we asked the
sharper question: *what do active cosmology departments structurally under-invest in?* The
answer is consistent across the 2025 anomaly reviews (PNAS "Everyone wants something better
than ΛCDM" 2025; the Oxford "Manifesto"; Peebles' "Anomalies in Physical Cosmology"):

**Departments are organised around new data + new models. They under-reward three things we
are unusually well-suited to do from a desk:** (1) **cross-catalogue consistency audits**,
(2) **systematics / selection-function re-analysis of live anomalies**, (3) **living
meta-analyses**. Our edge (theory judgment + tireless literature/data curation, no telescope)
maps exactly onto that gap.

Crucial steering signal from the reviews: the **CMB large-angle anomalies are *fading*** (peak
smoothing, low-ℓ curiosities drifting back toward statistical noise), while the
**catalogue-based isotropy anomalies are *growing*** (now >5σ). Point effort at the growing,
catalogue-based, systematics-limited ones — not the fading CMB curiosities.

### G. Cosmic number-count dipole / isotropy tests — *new top desk pick*
The flagship instance and, on the leverage metric, the best thing we've found. Quasar/radio
source counts show a dipole **3.7× larger than the CMB kinematic expectation — a 5.4σ
violation** of the cosmological principle (Böhme et al. 2025, NVSS+RACS+LoTSS; CatWISE 4.9σ,
Secrest 2021; Quaia 1.3M-quasar Bayesian analyses). Why it's the target quadrant:
- **Foundational stakes** — it tests *isotropy / the Copernican principle* itself, not a
  parameter. A higher-stakes target than yet another H₀ number.
- **Genuinely simple work** — it is *source counting + dipole fitting* (the Ellis–Baldwin
  test) on fully public catalogues (CatWISE2020, Quaia, NVSS, RACS-low, LoTSS-DR2). Simpler
  than Gaia wide-binary vetting.
- **The fight is systematics** — is the excess real, or selection/flux-calibration/masking
  contamination? That is *exactly* our curation edge, mirroring the wide-binary logic.
- **Uncrowded + growing** — ~12 papers/yr vs. 200 for the Hubble tension (~17× less worked),
  yet at *higher* significance and rising. Nobody owns the **cross-catalogue consistency
  audit** (do CatWISE/Quaia/radio agree once a *common* selection mask + flux/colour cuts are
  imposed? how much excess survives?). That meta-analysis is a real, near-term contribution.
- Anchors: Secrest et al. 2021 (arXiv:2009.14826); Land-Strykowski, Lewis & Murphy 2025
  (arXiv:2509.18689, Bayesian cross-dataset tension — note CatWISE↔NVSS concordant but RACS
  discordant); Guandalin et al. 2023 (arXiv:2212.04925, QLF-evolution systematic, the key
  caveat); Böhme et al. 2025 (radio NVSS+RACS+LoTSS, ~5.4σ — ref to verify);
  Quaia Bayesian (MNRAS 527, 8497); "Kinematic contribution to the number-count dipole"
  (A&A 2025); "Testing the cosmological principle with quasars" (A&A 2026, aa56955-25).
- **Sibling probe:** the **bulk-flow non-convergence** (peculiar-velocity flows not settling
  to the CMB frame by the expected scale; ~18 papers/yr) is the same isotropy question from
  CosmicFlows data — a natural second front in the same family.

### H. Cosmic birefringence — strong but steeper
A ~**0.3° rotation** of CMB polarization (E→B), a parity-violation smoking gun for axion-like
fields, at ~3.6σ from Planck/ACT/SPIDER (~24 papers/yr). Genuinely live and not crowded, BUT
the entire result is **limited by the instrumental polarization-angle miscalibration**, which
needs detector-team calibration knowledge — far less desk-tractable than G. Keep as a
"watch / partial-entry" candidate, not a first build.

## Updated shortlist

| Project | Quadrant | Type | 2025 vol. | First build? |
|---|---|---|--:|---|
| **G cosmic-dipole / isotropy audit** | **low-effort / desk** | **cross-catalogue re-analysis, >5σ live** | **~12/yr** | **NEW TOP PICK** |
| A wide-binary gravity | low-effort / desk | re-analysis, real result | ~11/yr | yes (twin of G) |
| F reheating→GW scorecard (narrow) | low-effort / desk | theory table, high ceiling | ~11–13/yr | strong next |
| C / C′ census + field map | low-effort / desk | living DB + steering | — | background |
| H cosmic birefringence | low-effort but calib-limited | anomaly, needs instrument cal | ~24/yr | watch |
| B Dyson re-analysis | low-effort / desk | frozen-archive re-analysis | ~2/yr | later |
| D PBH attack-surface | crowded | falsification map | ~353/yr | later (map only) |
| E galaxy-formation audit | flank of crowded | systematics audit | ~313/yr | opportunistic |

**Revised operating decision:** **G is the new lead** — it dominates A on stakes (isotropy vs.
a binary-star test), matches it on tractability, beats it on simplicity (catalogue counting),
and is comparably uncrowded but at *higher, rising* significance. G and A are methodological
twins (both are "the anomaly is real iff the selection function is clean"), so doing G first
builds the exact curation muscle A needs. Keep **F** as the high-ceiling theory bet and **C′**
as the standing steering layer.

## Implementation proposals (Jul 2026)

Three concrete `celestrium/` modules for G-Euclid pre-DR1 work are now documented in
[`euclid-dr1-prep.md`](./euclid-dr1-prep.md#proposed-implementations-jul-2026):
**forecast.py** (σ_D partial-sky forecast — the gate task), **mocks.py** (mock + null
pipeline for significance calibration), and **ellis_baldwin.py** (D_kin pre-registration
for Euclid bands). Priority order: forecast → mocks → Ellis–Baldwin.
