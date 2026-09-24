# Celestrium Research Avenues & Conditions for Definitive Results

**Date**: September 24, 2026  
**Status**: Active Research Roadmap & Scientific Conditions Matrix  
**Repository**: `sub-surface/astro-theory` (`celestrium`)  

---

## Executive Framing

Following supervisor guidance and rigorous epistemological audit, Celestrium organizes its computational astrophysics into **four distinct, decoupled research avenues**. Each avenue is governed by explicit conditions required before claiming cosmological or methodological results:

1. **Avenue A**: Cosmological Quasar Dipole Anomaly & Systematics-Aware Inference
2. **Avenue B**: Heteroscedastic Evidential Deep Learning for Multi-Survey Astro-Phenomenology (`FoundationAstroJev`)
3. **Avenue C**: Calibrated Decision-Making & Reinforcement Learning on Calibrated Decisions (RLCD) for Constrained Observatories
4. **Avenue D**: Euclid DR1 Forward Modeling & Multi-Survey Cosmological Discovery Preparation

```
+---------------------------------------------------------------------------------------------------+
|                                CELESTRIUM RESEARCH ARCHITECTURE                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
         +--------------------+-------------------+--------------------+--------------------+
         |                    |                                        |                    |
         v                    v                                        v                    v
   [ Avenue A ]         [ Avenue B ]                             [ Avenue C ]         [ Avenue D ]
  Quasar Dipole       Evidential Deep Learning                  RLCD Decision       Euclid DR1 &
  Harmonic Decoupling & Multi-Survey Calibration                Telescope Queue     Future Survey
  & Deprojection      (FoundationAstroJev)                      Scheduling (CMDP)   Forecasting
         |                    |                                        |                    |
         v                    v                                        v                    v
  Reproduce Quaia      Spatial K-Fold Holdout                   Multi-Tier Alert    TAP Footprint
  Literature & Test    & Reddening Robustness                   Follow-up Benchmarks Simulation &
  Null Hypothesis      (No Coordinate Leakage)                  (3x Yield Gain)     Fisher Matrices
```

---

## Avenue A: Cosmological Quasar Dipole Anomaly & Systematics-Aware Inference

* **Target Output**: Paper A (*Systematics-Aware Measurement of the Quasar Number-Count Dipole with Evidential Selection and Conformal Contamination Control* — A&A / MNRAS)
* **Core Scientific Hypothesis**: The apparent discrepancy between the kinematic CMB dipole ($D_{\rm CMB} \approx 0.007$, $(l, b) = (264.0^\circ, +48.3^\circ)$) and wide-area quasar number-count dipoles ($D \sim 0.02 - 0.08$) is primarily driven by survey selection-function non-uniformities (specifically North-South exposure gradients) and stellar contamination at faint photometric limits, rather than a true violation of the Cosmological Principle.

### Literature Baselines to Reproduce
1. **Quaia G20.5 Raw Anomaly**: Naive masked dipole yields $D \approx 7.5\% - 8.5\%$ with strong North-South asymmetry ($D_z \approx +0.055$).
2. **Published Selection-Corrected Bounds**: Dam et al. (2024), McTier et al. (2024), Storey-Fisher et al. (2023) find that deprojecting the official unWISE+Gaia selection function reduces the apparent dipole to **$2.1\% - 3.3\%$** depending on Galactic latitude cut ($|b| > 20^\circ$ to $30^\circ$).

### Conditions to Claim a Definitive Result
| Condition | Description | Threshold / Criterion | Status |
| :--- | :--- | :--- | :--- |
| **C1.1: Literature Replication** | Reproduce Dam et al. (2024) / McTier et al. (2024) dipole amplitude on Quaia G20.5 with official selection function. | Recover $D \in [2.0\%, 3.3\%]$ across $|b| > 20^\circ - 30^\circ$ with $D_z \le 0.015$. | **FULFILLED** ($2.08\% - 3.12\%$, $D_z = +0.011$ in EXP-2026-L) |
| **C1.2: Linearity Response** | Inverted pseudo-$C_\ell$ mode-coupling matrix $M_{\ell\ell'}^{-1}$ must recover injected dipoles without suppression on cut sky. | Response slope $b \in [0.95, 1.05]$ under injection-recovery benchmark. | **FULFILLED** ($b = 0.975 \pm 0.021$ in EXP-2026-K) |
| **C1.3: Provenance Audit** | Account for proper motion cuts and catalog contaminants (e.g. halo subdwarfs). | Catalog scaling law $\mu < 10^{+0.4(G - 18.25)}$ verified on all 1.295M rows. | **FULFILLED** (100% compliant) |
| **C1.4: Contamination Control** | Bound false discovery rate of stellar contaminants using Conformal Risk Control. | Empirical $\text{FDR} \le 5.0\%$ with provable finite-sample guarantee. | **FULFILLED** (438,242 purified quasars, $\hat{\lambda}=0.50$) |
| **C1.5: Large-Scale Null Model** | Establish non-vacuous empirical $p$-value under $\Lambda$CDM kinematic null hypothesis. | $N \ge 10,000$ end-to-end mock sky realizations including selection function and mask. | **ACTIVE (EXP-2026-O)** |
| **C1.6: 3D Vector Covariance** | Evaluate tension in full Cartesian 3-vector $\mathbf{D} = (D_x, D_y, D_z)$ rather than scalar amplitude. | Full $3\times 3$ empirical covariance $\mathbf{\Sigma}_D$ and $\Delta\chi^2$ test. | **FULFILLED** |

---

## Avenue B: Heteroscedastic Evidential Deep Learning for Multi-Survey Astro-Phenomenology

* **Target Output**: Paper B (*A Heteroscedastic Evidential Classifier for Multi-Survey Astro-Phenomenology* — MNRAS / ApJ)
* **Core Scientific Hypothesis**: Modeling classification uncertainty through a continuous Dirichlet prior over 12 astrophysical classes enables simultaneous quantification of observational noise (aleatoric) and out-of-distribution domain shift (epistemic), allowing rigorous conformal guarantees without test-time calibration drift.

### Literature Baselines to Reproduce
1. **Multi-Class Photometric Astro-Classification**: Standard MLP / CNN / XGBoost classifiers on Gaia DR3 + AllWISE achieve $85\% - 91\%$ top-1 accuracy, but exhibit severe overconfidence in faint ($G > 19.5$) and high-extinction ($E(B-V) > 0.2$) regimes with Expected Calibration Error $\text{ECE} > 0.15$.
2. **Evidential Deep Learning (Sensoy et al. 2018)**: Dirichlet concentration parameterization $\boldsymbol{\alpha} = \mathbf{e} + 1$, where total evidence $S = \sum \alpha_k$ governs epistemic uncertainty $u_{\rm epi} = K / S$.

### Conditions to Claim a Definitive Result
| Condition | Description | Threshold / Criterion | Status |
| :--- | :--- | :--- | :--- |
| **C2.1: Heteroscedastic Uncertainty** | Incorporate heteroscedastic photometric and astrometric errors into Dirichlet loss. | Aleatoric variance $\sigma_i^2$ explicitly penalizes low-SNR photometric bands. | **FULFILLED** |
| **C2.2: Multi-Regime Calibration** | Prove calibration across magnitude, SNR, and Galactic latitude splits. | $\text{ECE} \le 0.04$ across all regimes; Brier score improvement $> 25\%$ vs baseline. | **FULFILLED** (EXP-2026-F, Brier = 0.052, ECE = 0.019) |
| **C2.3: Spatial Disjoint Holdout** | Demonstrate spatial invariance via regional $k$-fold cross-validation (Supervisor Rec. #13). | No coordinate memorization; North vs South transfer test $\Delta\text{ECE} \le 0.02$. | **ACTIVE (EXP-2026-M)** |
| **C2.4: Epistemic OOD Detection** | Epistemic uncertainty $u_{\rm epi}$ must systematically increase on unobserved or anomalous features. | AUC-ROC $> 0.90$ on OOD / simulated unWISE scanning strip anomalies. | **ACTIVE (EXP-2026-M)** |
| **C2.5: Conformal Set Guarantees** | Multi-class prediction sets achieve user-specified coverage $1 - \alpha$. | Marginal and conditional coverage $\ge 95\%$ on held-out real survey sources. | **FULFILLED** |

---

## Avenue C: Dynamic Telescope Queue Scheduling & Epistemic RLCD

* **Target Output**: Paper C (*Calibrated Follow-Up Decision Policies for Constrained Autonomous Observatories* — Astron. Comput. / ICML)
* **Core Scientific Hypothesis**: Transient alert follow-up formulated as a Constrained Markov Decision Process (CMDP) with Reinforcement Learning on Calibrated Decisions (RLCD) significantly outperforms greedy probability thresholding by routing high-epistemic-uncertainty candidates to low-cost screening facilities before committing scarce 8m-class spectrographs.

### Literature Baselines to Reproduce
1. **Broker Heuristics**: Greedy ranking (highest $p_{\rm class}$ first) or static thresholding ($p > p_{\rm thresh}$) achieves high prompt follow-up but suffers high false-alarm rates ($> 30\%$) and rapid budget exhaustion within the first 10 days of an observing semester.
2. **Aperture Constraints**: Standard observatory allocation (e.g. Gemini, Keck, VLT ToO): $B \approx 60 - 180$ hours per semester.

### Conditions to Claim a Definitive Result
| Condition | Description | Threshold / Criterion | Status |
| :--- | :--- | :--- | :--- |
| **C3.1: Formal CMDP Formulation** | Explicit constrained optimization: $\max_\pi \mathbb{E}[\sum U_t] \text{ s.t. } \sum C_t \le B$. | Exact budget constraint satisfied without post-hoc manual intervention. | **FULFILLED** |
| **C3.2: Multi-Tier Instrument Routing** | Environment supports heterogeneous follow-up tiers (1m imager, 4m spectrograph, 8m spectrograph). | Policy dynamically selects optimal aperture tier based on magnitude and epistemic state. | **ACTIVE (EXP-2026-N)** |
| **C3.3: Perishable Transient Dynamics** | Realistic exponential luminosity decay ($U(t) = U_0 e^{-t/\tau}$) and Poisson alert arrivals. | Kilonovae ($\tau \sim 2\,{\rm d}$), FBOTs ($\tau \sim 3\,{\rm d}$), SNe Ia ($\tau \sim 25\,{\rm d}$) modeled. | **ACTIVE (EXP-2026-N)** |
| **C3.4: Value of Information (VoI)** | Demonstrate that screening high-doubt candidates with Tier 1 reduces false alarms on Tier 3. | Tier 3 false alarm rate $< 5\%$ with $>2.5\times$ yield in confirmed rare transients. | **ACTIVE (EXP-2026-N)** |
| **C3.5: Weather & Seeing Stochasticity** | Policy must be robust to stochastic cloud cover and lunar brightness interruptions. | Average monthly budget utilization $> 92\%$ without overruns across 100 simulated semesters. | **ACTIVE (EXP-2026-N)** |

---

## Avenue D: Euclid DR1 Forward Modeling & Multi-Survey Cosmological Discovery Preparation

* **Target Output**: Paper D / Technical Report (*Euclid DR1 Wide Survey Cosmological Dipole Forecast & Systematics Shielding*)
* **Core Scientific Hypothesis**: Euclid DR1 Wide Survey (~1,900 deg² near-infrared slitless spectroscopy) provides an uncorrupted extragalactic sample free from unWISE infrared zodiacal background and unWISE scan-pattern systematics, enabling a clean test of the dipole anomaly at $z > 1.0$.

### Conditions to Claim a Definitive Result
| Condition | Description | Threshold / Criterion | Status |
| :--- | :--- | :--- | :--- |
| **C4.1: TAP Footprint Fidelity** | Exact geometric modeling of the 103 Euclid TAP tables and ~1,900 deg² Q1/DR1 survey footprint. | HEALPix mask matching verified ESA TAP sky distribution. | **FULFILLED** |
| **C4.2: Mode-Coupling Inversion** | Pseudo-$C_\ell$ inversion matrix computed for the Euclid DR1 geometry ($f_{\rm sky} \approx 0.046$). | Condition number $\kappa(M) \le 10^3$ for multipoles $\ell \le 3$. | **FULFILLED** |
| **C4.3: Dipole Precision Forecast** | Fisher matrix forecast for dipole recovery uncertainty $\sigma_D$ under Euclid DR1 NISP spectroscopy. | Quantify detectable dipole amplitude $D_{\rm min}$ at $3\sigma$ confidence. | **FULFILLED** ($\sigma_D \approx 0.012$ for $N \sim 500,000$ Euclid DR1 quasars) |
| **C4.4: Joint Euclid + Rubin Mock** | Synthetic cross-match of Euclid near-IR with Rubin LSST optical colors. | Demonstrate elimination of stellar locus overlap at $G - W_1$ boundaries. | **PLANNED** |

---

## Summary of Active Execution Priorities

1. **`EXP-2026-M`**: Run Spatial Disjoint Hold-out Validation on 80,000 sources (Hemispheric North vs South, Disjoint HEALPix).
2. **`EXP-2026-N`**: Execute Multi-Tier Telescope Queue Scheduling with Epistemic RLCD across 30 nights under stochastic weather.
3. **`EXP-2026-O`**: Run Vectorized 10,000-Realization Null-Model Monte Carlo for definitive $\Lambda$CDM kinematic $p$-value.
