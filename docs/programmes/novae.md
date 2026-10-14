# P2 · Accreting white dwarfs & recurrent novae

**Status**: active · **Lead objects**: T CrB, plus the symbiotic recurrent-nova family (RS Oph, V3890 Sgr, V745 Sco)
**Code**: [`celestrium/novae/`](../../celestrium/novae/) · [`scripts/novae/`](../../scripts/novae/) · **Data**: `data/tcrb/`, `data/srn/<slug>/`
**Ledger**: EXP-2026-Y, Z in [`../ledger.md`](../ledger.md) · **Pages**: https://astro.subsurfaces.net/tcrb/, https://astro.subsurfaces.net/v347nor/
**Dated tests**: `python -m celestrium calendar` (entries in [`celestrium/watchlist.py`](../../celestrium/watchlist.py))

## Question

How do we see a recurrent nova coming? T CrB (eruptions 1866, 1946; next expected 2026–2030) is the test case:
what state is its accretion flow in, and which pre-eruption signatures are real? The longer-range aim is a
calibrated "who's next" list across the Galaxy's symbiotic stars, tested backwards against a century of plates.

## Material results (EXP-2026-Y, real data)

- **September 2026 UV collapse.** Swift UVM2 fell 2.05 mag in 30 days (12.12 → 14.17), back to the pre-2015 quiescent level, while 2–10 keV X-rays rose. Unreported as of 2026-10-14.
- **Not a supply cut.** A viscous disk with its supply switched off fades at most 0.11 mag in 30 days; the fastest 2-mag fade takes 288 days. The inner disk changed state.
- **Inner-edge history** (robust across inversion variants): ~3 R☉ in quiescence → 0.75–1 R☉ through the 2016–22 high state → 2.7 R☉ in the 2023 dip and again now. Absolute Ṁ is **not** identified (brightness–truncation degeneracy; nebular continuum missing).
- **Inclination**: quiescent ellipsoidal half-amplitude 0.182 ± 0.011 mag needs M-giant TiO temperature sensitivity; implies i ≥ 59–76° (lower limit).
- **TESS flicker**: no significant periodicity in S24/25/51/78 (60 red-noise simulations per sector; the 300-sim run was interrupted).
- **Family**: in quiescence RS Oph and V745 Sco are ~2,800× more UV-luminous than T CrB (±0.4–0.8 dex); T CrB's high state reached ~60× its own quiescence.
- **1946 overlay**: the 2023 alignment is falsified; the Sep 2026 alignment implies ~2027.4.

### Literature matrix (from the 2026-10-13 review; curated values with bibcodes in `celestrium/novae/literature.py`)

| Question | Literature position | Key references | This programme |
| :--- | :--- | :--- | :--- |
| White-dwarf mass | 1.29–1.37 M☉ | Stanishev+04; Hinkle+25; Munari+25; Baptista+26 | adopted 1.35 |
| Inclination | 55–70°, method-dependent | Tatarnikova 13; Munari+25; Baptista+26 | i ≥ 59–76° (ellipsoidal, TiO-corrected) |
| High-state driver | enhanced mass transfer through a large disk | Planquart+25; Schlindwein+25; Baptista+26 (vs Iłkiewicz+23) | inner-edge migration; Ṁ not identified |
| UV vs X-ray | anticorrelated | Luna+18; Pei+25; Luna+26 | ρ = −0.49 over 366 same-day pairs (p not autocorrelation-corrected) |
| 2023 dip = 1945 dip? | no | Merc+25 | confirmed (alignment falsified) |
| Dust in the dips | disfavoured | Luna+26; Evans+19 | 30-day collapse too fast for a disk drain → state change |
| ~6000 s periodicity | disputed | Luna+26 | not significant in TESS |
| Eruption date | 15 published windows, 5 lapsed | Schaefer 23a,b; Schneider 24; Pei+26; Sello+26 | recurrence clock median 2028.1 (68%: 2027.1–2030.9) |
| Super-remnant | ~30 pc nebula, optically thin | Shara+24 | eruption-day test in the calendar |

## Rerun / next queue

1. **UV-recovery test** (pre-registered, 5–30 Nov 2026): orbit-locked obscuration vs accretion-state change.
2. **Inversion v4**: add a nebular (Balmer) continuum component and an X-ray-luminosity energy budget; re-test whether Ṁ becomes identifiable (fuel-budget question vs Schaefer 2025's 7.4×10⁻⁴ M☉ ejecta).
3. **Autocorrelation-aware UV/X test** (block bootstrap over visits).
4. **DASCH plates**: real photometry of the 1930s–40s pre-eruption dip.
5. **"Who's next?"**: recurrence-time predictions for known symbiotics (Gaia distances + Swift UV + nova grids), validated retrospectively on DASCH and forward with Gaia DR4 epoch photometry (~Dec 2026). Run an ADS novelty check first.
6. **TESS flicker census** of symbiotics and accreting WDs (extends `tcrb_tess_flicker.py`).
7. Eruption playbook (calendar `tcrb-eruption-playbook`).

## EXP-2026-Z · Hidden recurrent novae in a century of Harvard plates (pre-registered 2026-10-14)

Written before any light curve beyond T CrB's was pulled. Thresholds below are fixed; changes after data are logged as amendments with reasons.

**Question.** Are any known Galactic novae or symbiotic stars recurrent novae (RNe) whose earlier eruption sits unnoticed on the Harvard plates (DASCH DR7, 1885–1992, complete sky since 2024)?
How many RNe hide among the "classical" novae? Pagnotta & Schaefer 2014 give 24 ± 4% (three methods spanning 12–35%); ours would be a fourth, direct method.

**Literature position and novelty (ADS, 2026-10-14).**
- Schaefer has hand-searched plates for the 10 known RNe (Schaefer 2010) and for some of the strong candidates:
  - KT Eri, shown to be recurrent (2022);
  - V2487 Oph, superflares in 1941–42 (2024);
  - V745 Sco, an 1897 eruption found via DASCH in 2024.
- Pagnotta & Schaefer 2014 rank RN candidates by 7 criteria.
- DASCH blind-transient searches (Miller 2014; Grindlay 2019) target >5 mag outbursts of unknown objects, not nova positions.
- No systematic, calibrated sweep of all ~400 Galactic nova positions and ~600 symbiotics, with measured completeness, was found.
- The novelty is therefore the census and its completeness, not the method.
- Caveat on stakes: Schaefer 2025 (ApJ 991, 111; 994, 187) argues from period changes that RNe eject more than they accrete, and that symbiotics are not SN Ia progenitors. The SN Ia relevance of an RN census is contested.

**Targets** (`data/rn/targets.csv`).
- Galactic novae from Schaefer 2022 (402, with eruption year, V_peak, E(B−V), light-curve class), merged with VSX types N*/NR* (Galactic only).
- Symbiotics from Akras+ 2019 (257 Galactic + candidates) merged with VSX ZAND*.
- Duplicates are merged within 5″.
- Negative controls: per target, 2 field stars 3–10′ away in the same refcat, matched within 0.5 mag, not flagged variable.

**Data and cuts.**
- DASCH light curves come from the public API, using the ATLAS refcat (APASS as fallback), with the nearest source within 5″.
- Points flagged high background, bad plate, multi-exposure unmatched or blended, uncertain date, large ISO/local RMS, unknown or large Δr, B/C blend, or Pickering wedge are dropped.
- Points near the plate limit are kept, with their errors inflated.
- Non-detections are used as censored limits.

**Detectors.**
- **D0 (naive).** The baseline is the median of good detections. Event points are at least max(1.0 mag, 5σ_MAD) brighter. An event needs ≥ 2 such points on different nights within 200 d. The score is the excess of the second-brightest point.
- **D1 (physics-informed).**
  - Quiescence: a running-median envelope over ±500 d (event window excluded), with Student-t (ν = 3) errors and a fitted jitter.
  - Censored likelihood for non-detections, with a 2% floor for plate misses.
  - Eruption template: a flux power-law decline F ∝ (1 + Δt/τ)^−1.75 (Hachisu & Kato universal decline), with t₂ from 2 to 128 d, peak time t₀ and amplitude profiled, added in flux to quiescence.
  - Score: 2Δln L.

**Result conditions.**
- **G0, pipeline gate on T CrB.** (a) D1 recovers the 1946 eruption at its date. If not, the pipeline is broken; stop and fix. (b) The 1945 pre-eruption dip: mean B over 1945.3–1946.0 is ≥ 0.3 mag fainter than over 1943.0–1945.0, at ≥ 3σ. (c) The 1936–45 high state: mean B over 1938–45 is ≥ 0.3 mag brighter than over 1950–60. If (b) or (c) fails, the secondary precursor sweep is declared infeasible with DASCH.
- **R1, recall on real eruptions.**
  - Positives are catalogued eruptions from 1890 to 1991 of the targets themselves, plus the known RN eruptions.
  - An eruption is recovered if an event's t₀ falls within [Year − 0.5, Year + 1.5].
  - Recall is reported both unconditional and conditional on DASCH coverage (≥ 1 good plate within −10/+60 d), each with a binomial CI.
- **H_PI, physics-informed denoising.** At the threshold where each detector's control false-alarm rate is 1% per light curve, D1 recall − D0 recall ≥ 0.10, and the paired-bootstrap 95% CI lower bound is > 0. Also reported at 0.1%. Anything else counts as "no benefit".
- **R2, discovery.**
  - The search looks for events ≥ 2 yr from every catalogued eruption, with a score above the maximum control score (or a GPD-tail false-alarm rate < 0.5/N_targets).
  - **New RN candidate** requires ≥ 2 nights on ≥ 2 plates, a decline consistent with the template, plate-image vetting (PSF/ellipticity matching neighbours, no defect, no blend, not an asteroid), and no prior report (ADS/VSX/CBAT).
  - **Confirmed** needs an independent archive (APPLAUSE, Sonneberg) or inspection of the original plate. Nothing is announced externally without Leon.
- **R3, population.** Combine detections with per-target completeness from injecting eruptions at each nova's own catalogued peak into its real plate cadence and limits. Report the posterior on the fraction of CNe that are RNe with T_rec < 100 yr. If the 95% upper bound is < 12%, that is in tension with all three Pagnotta & Schaefer estimates.

**Amendments.**
- **A1 (2026-10-14, on the G0 target T CrB, before any search data).** LARGE_ISO_RMS, LARGE_LOCAL_SMOOTH_RMS and LARGE_DRAD flag 26 of the 27 points at T CrB's February 1946 peak: they fire on points outside the calibrated range. These flags are now soft: the point is kept and 0.5 mag is added to its error in quadrature. Targets and controls are treated the same way, so false-alarm calibration is unaffected.
- **G0 result (2026-10-14): passed.**
  - (a) D1 recovers the 1946 eruption at t₀ = 1946.108 (the actual peak was 1946 Feb 9 = 1946.107).
  - (b) The 1945 dip is 0.86 ± 0.06 mag fainter than 1943–45 (14σ).
  - (c) 1938–45 is 0.37 ± 0.04 mag brighter than the 1950s (10σ).
  - The precursor sweep is therefore feasible with DASCH.
- **A2–A5 (2026-10-14).** Each was triggered by G0, or by the recall test on known eruptions on the first 111 targets. Discovery-score summaries had been seen, but no candidate list had been looked at.
  - **A2.** An eruption must be ≥ 1.5 mag above quiescence.
  - **A3.** Saturated points are treated as "at least this bright".
  - **A4, position-based light curves.** DASCH filed the eruption detections of faint novae under neighbouring catalogue stars: V528 Aql's 1945 eruption, 61 plates at ~4″, sat under a 16.5 mag star 12.7″ away. Light curves are now assembled from every neighbour within 20″, keeping detections measured within 6″ of the position and nearer to it than to their owner. CASE_B_BLEND ("may belong to a fainter catalogue star nearby") becomes a soft flag. Non-detections count in the quiescent envelope as max(limit, catalogue mag).
  - **A5.** D1's null is the better of quiescence and a free brightness step over the same window, i.e. a slow state change, so symbiotic high states do not score as eruptions. Discovery scans drop all data within ±2 yr of catalogued eruptions instead of only masking t₀.
  - Controls get identical treatment throughout.
- **A6–A7 (2026-10-14).** Both were triggered by vetting the top events of a partial run (116 targets). They were fixed before the full run.
  - **A6, blended neighbours.** V1404 Cen's top event was a 13.5 mag star 4.3″ away. A neighbour closer than 8″ is unresolved on most plates, so its detections are reassigned to the target only if they are ≥ 1 mag brighter than the neighbour's own median. A resolved neighbour's detection is reassigned only if it lies at least twice as close to the target as to the neighbour.
  - **A8, two-night events (2026-10-14, interim run on 554 targets).** The control tail was set by single-plate events: 9 of the top 10 control scores had 0–1 supporting nights. R2 could never accept such events, yet they set the 1% threshold (Δln L = 84). The threshold must be computed on the same statistic the decision uses, so D1's best event is now the best event supported on ≥ 2 nights. This applies identically to controls, targets, recall and completeness. Disclosure: this interim run had also shown me the target list, including DE Cir (score 30, 2 nights), below that threshold.
  - **Night counting fix.** Nights were counted as distinct round(JD) values, which roll over at 0h UT. That split single Boyden nights (16–02 UT) and merged some consecutive ones. Nights are now clusters of epochs separated by > 12 h (`plates.n_nights`). D2's training labels use the same rule.
  - **A7, repeaters.** Z Cam (a dwarf nova) topped the partial list. After the best event, D1 rescans twice with ±0.5 yr dropped around each event found. A candidate with ≥ 2 further events above the 1% threshold within 5 yr is vetoed as dwarf-nova-like. No RN recurs that fast. The veto removes candidates only, so control thresholds are unchanged. The number vetoed is reported.

**H_ML: learned vs physics-informed denoising (pre-registered 2026-10-14, before any D2 code).**
- **D2.** A sequence model sees ±3 yr windows of plate rows: Δt, magnitude or limit, detection flag, error, and the position-assembled envelope. It is trained to output P(eruption in window).
- **Training data.** Synthetic universal-decline eruptions (t₂ 2–128 d, peaks spanning the plate limits) injected into real control light curves from half the targets, split by target ID. Negatives are uninjected windows from the same light curves. No nova-position light curve is seen in training.
- **Evaluation.**
  - Light-curve score = max over windows.
  - False-alarm thresholds come from the held-out half of the controls, and the same held-out half sets D1's threshold.
  - Positives are the real catalogued eruptions (R1 set).
- **Outcomes.**
  - Recall_D2 − recall_D1 ≥ 0.05 at FAR 1%, with paired-bootstrap CI > 0: learned denoising adds information beyond the physics likelihood.
  - |Δ| < 0.05, or a CI spanning 0: the physics prior already captures it.
  - D2 worse by ≥ 0.05 (CI < 0): the sim-to-real gap dominates.
- **Compute.** Prototype on the local RTX 2060; the full run on one Modal B200, capped at $1. The spend is logged in ledger §4.

**Prior expectation (calibrated).**
- ~400 novae × 24% RN ⇒ ~100 hidden RNe. Most recur on 30–100 yr scales, have quiescence too faint for the refcat, or erupt when no plate was taken.
- With roughly 5–15% per-target completeness we expect O(1–10) detectable earlier eruptions. Some are already known to Schaefer.
- P(≥ 1 genuinely new candidate) is guessed at 0.3–0.5.

**Results (2026-10-14).** Full census: 1,211 targets with a refcat match (104 had none), 2,101 visibility-matched controls, and 197 catalogued eruptions with ≥ 3 plates in their window. JSON: [`experiment_z_rn_results.json`](../results/experiment_z_rn_results.json).
- **H_PI: NOT SUPPORTED at the pre-registered 1% FAR.**
  - Recall D1 = 0.39 [0.32, 0.46] vs D0 = 0.41 [0.34, 0.48]: Δ = −0.02, CI [−0.09, +0.10].
  - At 0.1% FAR the physics likelihood wins clearly: 0.28 vs 0.14, Δ = +0.14, CI [+0.02, +0.26]. It earns its keep in the far tail, where discoveries live, not at 1%.
  - [`rn_scores_recall.png`](../figures/rn/rn_scores_recall.png).
- **R1.**
  - 76/197 covered eruptions recovered at 1% FAR (0.39). That rises to 0.56 [0.43, 0.68] where plates reach ≥ 6 mag below the catalogued peak.
  - Recall is limited mainly by cadence: most non-RN eruptions are dated only to the year, and their peaks fell in seasonal or wartime gaps.
  - Separately, 5.7% of targets but 1.0% of controls exceed the 1% threshold. Nova and symbiotic positions are strongly enriched in real activity.
- **H_ML: sim-to-real gap dominates.**
  - D2 (transformer, one B200, $0.42) recalls 0.10 vs D1 0.38 on the same held-out controls: Δ = −0.28, CI [−0.38, −0.21].
  - Diagnosis: D2's AUC for real eruptions vs controls is 0.25, below chance (D1: 0.72). Its control scores track detection fraction (Spearman 0.75) and source brightness. It learned "more detections than usual" from the injector. Real eruptions carry signatures the injector never made: saturated peaks, soft-flag errors, and blends split across catalogue stars.
  - A fair retry needs an injector that reproduces those plate-pipeline effects, which would be a new pre-registration.
- **R2.** Ten events exceed the control maximum (Δln L = 278), all at symbiotic stars. After image vetting ([`vetted.csv`](../../data/rn/vetted.csv), montages in `data/rn/vet/`):
  - **V347 Nor = Hen 2-147 (genuine, new): an unrecorded nova eruption in February 1943.** Audit: [`v347nor_audit.json`](../results/v347nor_audit.json), [`v347nor_stamps.json`](../results/v347nor_stamps.json) (`scripts/novae/v347nor_audit.py`, `v347nor_stamps.py`).
    - **Identity.** V347 Nor = Hen 2-147 = Wray 16-208 = PN G327.9-04.3: a D-type symbiotic with a Mira (P = 373 d) inside an expanding [N II] ring. Long called a "symbiotic nova" (Munari & Patat 1993; Corradi+ 1999; Santander-García+ 2007), but no outburst has been reported (ADS, VSX, GCVS).
    - **Onset (from plate cutouts).** Absent on 1943 Feb 9.07 UT (bi00717, limit 12.9) and Feb 10.07 (bi00722, 11.7). Present on Feb 11.10 on two cameras (am22958 S = 17, rb11996 S = 22) with an image as strong as on Feb 26. The first catalogued magnitude is B = 7.99 on rb12036, Feb 26.06 UT (flagged saturated). The DASCH catalogue logs all the Feb 11–25 plates as non-detections. An earlier note here ("fainter than 13.0 on Feb 14") was one of those misses; the dates were also off by a day (decimal-year conversion).
    - **Cutout gate (G7b).** Cutout significance S (10″ core against a 100–200″ annulus) on the plates DASCH logs as non-detections: 74 of 79 in 1943 exceed the 99th percentile (1.9) of 725 quiet-year plates (1938–42, 1947–52; max 4.7). The rate falls to 54/136 in 1944 and 19/160 in 1945 as the star fades. A 20–40″ annulus fails because saturated patrol images are ~70″ across.
    - **Decline.** 70 catalogued detections on 64 nights in seven series (a, am, ax, b, bi, mf, rb); t2 = 58 d (43–96), t3 = 153 d (132–194) from Feb 26. Below the plate limits by 1947.
    - **Uniqueness and position.** 758 deep plates in 1890–1942 and 315 in 1947–90 show nothing comparable. The mean offset of 104 bright detections from Gaia is (+0.4 ± 0.6, −1.2 ± 0.4)″; the 6.5″ neighbour is excluded at 12σ. Chance of a catalogued nova within 2″ of any of the 1,211 targets: 2 × 10⁻⁴.
    - **Luminosity.** E(B−V) = 0.47 ± 0.08. At the Mira P-L distance of 3.0 ± 0.4 kpc, M_B = −6.3 (−6.8 to −5.8) on Feb 26. At the 1.5 kpc expansion parallax it is −4.8. The first unsaturated plate (bi00897, day 74, B = 9.65) gives −4.7 at 3 kpc. A ~1-day rise and an amplitude ≥ 9 mag exclude a Mira maximum and disfavour a Z And-type outburst (weeks-long rises, 2–4 mag). Classed as a symbiotic nova, like RR Tel 1944, but faster (t3 ~150 d).
    - **The nebula (G8: consistent, not diagnostic).** Santander-García+ (2007) measured the ring's growth with HST in 2001 and 2004. They found an expansion parallax (1.5 kpc) at odds with the Mira distance (3 kpc), proposed a shock front, and inferred an outburst "~200 yr ago". A 1943 origin requires deceleration r ∝ t^n with n = 0.25–0.57 (snowplough/wind regimes). Leon's alternative: the moving feature may be older gas lit up or shocked by a recent outburst, a pattern speed rather than material motion. Either way the ring cannot date the eruption. Spectra separating shock from photoionisation, and a pattern-vs-gas velocity comparison, would say more than a third imaging epoch.
    - **Status.** Strong candidate; G1–G7b pass. Confirmation (G9) needs inspection of the original plates (Feb 9–11 and rb12036) and a check of the 1943 Harvard Announcement Cards. Draft request: [`v347nor_dasch_query.md`](../manuscripts/v347nor_dasch_query.md). Note draft: [`v347nor_rnaas.md`](../manuscripts/v347nor_rnaas.md). Neither has been sent. Public page: https://astro.subsurfaces.net/v347nor/ (labelled candidate).
    - Figures: [`v347nor_1943.png`](../figures/rn/v347nor_1943.png), [`v347nor_onset.png`](../figures/rn/v347nor_onset.png), [`v347nor_vetting.png`](../figures/rn/v347nor_vetting.png), [`v347nor_dasch_misses.png`](../figures/rn/v347nor_dasch_misses.png).
  - **Known activity or textbook outbursts:** Z And 1939 and AX Per 1950.
  - **Z And-type activity of 1.5–3.5 mag, not nova eruptions:** RT Cru, CL Sco, Hen 3-1103, V1413 Aql, TX CVn (an A7 repeater) and V3804 Sgr (a ~1 mag high-state step).
  - **Assembly artifact:** SS73 141, where the star is present on plates listed as deep non-detections.
  - X Ser 1935 scored 185 in the interim run. It is the known stunted dwarf-nova activity of this old nova (Šimon et al. 2018, A&A 614, A141).
- **DE Cir 1929: an exploratory finding outside the pre-registered statistic.**
  - It scores Δln L = 30, below the 1% threshold of 45. The automated R2 search does not flag it.
  - The case rests on images: a stellar source at DE Cir on two plates on Apr 4 and 5, 1929 (B ≈ 12.5), and on rb00288 on Apr 16 (13.15). It is absent on Mar 15 at limit 16, and nothing else appears in about 800 deep plates.
  - DE Cir is one of Pagnotta & Schaefer's (2014) ten strong RN candidates but not among the seven whose plates they searched (their Table 7).
  - No dwarf-nova outburst appears in 2014–2025: Gaia DR3 (392 epochs), Gaia Alerts and TESS (7 sectors). That disfavours a frequent dwarf-nova cycle ([`decir_modern.json`](../../data/rn/decir_modern.json)).
  - Reading: a possible earlier eruption of a hidden RN (T_rec ≈ 74 yr, or ≈ 37 yr if one fell in the 1953–68 plate gap), caught on its tail. It is weaker than V347 Nor, and the original plates are the test.
  - [`decir_1929.png`](../figures/rn/decir_1929.png).
- **R3 (population).**
  - Completeness from recurrence-aware injection into all 356 Schaefer (2022) classical novae with assembled light curves: own cadence, catalogued peak, T = 10–100 yr, 6 realisations each. The threshold is the discovery threshold, the control maximum.
  - Genuine earlier eruptions among them at that threshold: k = 0.
  - **Typical-speed eruptions** (t₂ 5–100 d): Σ completeness = 104, giving f_RN < 2.5% (95%). A large hidden population of slow recurrent eruptions is excluded.
  - **Fast, RN-like eruptions** (t₂ = 3 d): Σ completeness = 14.8, giving f_RN < 20% (95%). That is above the pre-registered 12% tension line, so there is **no tension** with Pagnotta & Schaefer's 24 ± 4%. The plates are too sparse for few-day eruptions: completeness is 0.11 at T = 10 yr and 0.02 at 40 yr.
  - Since real RNe are fast, Harvard plates alone cannot settle the RN fraction. A denser archive or the original plates would be needed.
  - [`rn_completeness.png`](../figures/rn/rn_completeness.png).
- **Verdict on the prior.** P(≥ 1 genuinely new candidate) was 0.3–0.5. One emerged (V347 Nor), from the symbiotic arm rather than the hidden-RN arm.

## Open questions

Abrupt state changes (thermal switch vs truncation vs in-binary obscuration); why T CrB is the UV-faint family member;
the fuel budget; scoring the published forecasts after the eruption.

**Which minimum is deeper (found 2026-10-14).** Folded on our ephemeris, quiescent AAVSO V (2006–14) has its deeper
minimum at phase 0, which is phase 0.99 of the literature light-minimum ephemeris (taken as giant in front). Our
gravity-darkened Roche giant (`celestrium/novae/roche.py`, blackbody, i = 61°) puts the deeper minimum half an orbit
later, with the L1 face towards us. Two possibilities: light the model omits (disk, hot spot, irradiated giant), or
a half-orbit offset in what the literature phase zero means (check it against the spectroscopic conjunction in
Fekel+ 2000). This matters for the inclination limit from the ellipsoidal amplitude. Shown in `tcrb_orbit.gif`.
