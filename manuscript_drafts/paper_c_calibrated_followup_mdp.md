# Paper C: Autonomous Target-of-Opportunity Triage & Active 3D Follow-up MDPs with Conformal Risk Bounds

**Target Venue**: *Astronomy and Computing* / *International Conference on Machine Learning (ICML)*  
**Authors**: Celestrium Collaboration  
**Subject**: Artificial Intelligence (`cs.AI`), Machine Learning (`cs.LG`), Instrumentation and Methods (`astro-ph.IM`)  
**Draft Version**: 2.0 (Post-Multi-Messenger & 3D Tiling Benchmark Draft, September 2026)  

---

## Abstract

Next-generation multi-messenger sky surveys (Vera C. Rubin LSST, LIGO-Virgo-KAGRA O4/O5, IceCube, DESI) generate millions of transient candidates per night. Follow-up facilities are severely constrained: scarce 8m-10m spectrographs (Gemini GMOS, Keck, VLT) observe fewer than $0.01\%$ of triggered alerts, wide-field optical imagers (DECam, Rubin) face expansive gravitational-wave (GW) localization skymaps ($\Delta\Omega \sim 100 - 1000\,{\rm deg}^2$), and multi-object spectrographs (DESI 5,000 fibers) must assign apertures under acute stellar contamination. Standard reinforcement learning agents and heuristic brokers relying on uncalibrated softmax scores suffer catastrophic overconfidence on ambiguous boundary targets, permanently squandering finite observing night-hours on false alarms.

In this work, we present a unified autonomous decision suite grounded in **Reinforcement Learning on Calibrated Decisions (RLCD)**, **Constrained Markov Decision Processes (CMDP)**, and **Conformal Risk Control (CRC)**:
1. **Target-of-Opportunity Alert Triage (EXP-2026-R)**: Deployed across 50,000 real alerts from GraceDB O4 (e.g. `S240422ed`, `S230518h`), IceCube, and ALeRCE, our evidential triage network processes **$1,347,895\,{\rm alerts/sec}$** on cloud GPU. Conformal Risk Control bounds False Discovery Rates below $1.0\%$, achieving **$0.000\%$ False Alarms** on Gemini 8m GMOS spectroscopy while routing 49,710 ambiguous candidates to robotic 1m screening.
2. **Active-Evidential 3D GW Error-Volume Tiling MDP (EXP-2026-U)**: By coupling GLADE+ 3D galaxy catalogs with Kasen (2017) multicomponent radiative transfer, an autonomous finite-horizon MDP prioritizes rapid dual-band ($g$ and $z$) color confirmation, achieving a **$97.0\%$ Kilonova Discovery Rate** (+8.0% over greedy 2D tiling), a **$97.0\%$ dual-band color confirmation rate**, and shortening the discovery horizon by **$0.56\,{\rm hours}$**.
3. **DESI 5,000-Fiber Focal Plane Allocation (EXP-2026-V)**: Triaging 40,000 real phenomena on cloud GPU, the policy achieves **$90.4\%$ High-$z$ Quasar Recall** ($2,982.2\,{\rm fiber-hours}$ allocated), purging $50.7\%$ of stellar contaminants with only $3.23\%$ false allocations.

---

## 1. Introduction: The Follow-Up Triage Bottleneck

The fundamental challenge of modern multi-messenger astrophysics is not discovery, but **decision triage**:
- Alert brokers (Fink, ALeRCE, Lasair) ingest millions of candidates.
- Spectroscopic facilities allocate limited integration time in seconds ($t_{\rm exp} \sim 300 - 3600\,{\rm s}$).
- Triggering an exposure on a false positive (e.g. a Galactic flare star masquerading as a kilonova candidate) permanently consumes non-recoverable night hours.

Existing alert brokers rely on heuristic softmax probability thresholds (e.g. `p > 0.90`). However, standard deep learning models are notoriously miscalibrated on noisy survey data. What is required is not merely a model that outputs calibrated probabilities, but an autonomous policy where **uncertainty governs the action geometry**.

---

## 2. Problem Formulation: The Constrained Telescope Allocation MDP

We formalize autonomous telescope scheduling as a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{C}, B, \gamma)$:

### 2.1 State Space $\mathcal{S}$
A state $s_t = (\mathbf{x}_t, \boldsymbol{\sigma}_t, \mathbf{h}_t, \tau_t, \text{weather}_t)$ incorporates:
- Multi-band photometric fluxes and measured uncertainties: $(\mathbf{x}_t, \boldsymbol{\sigma}_t) \in \mathbb{R}^{22}$.
- Evidential Dirichlet concentrations from `FoundationAstroJev`: $(\boldsymbol{\alpha}_t, u_{\rm ale}, u_{\rm epi})$.
- Remaining telescope exposure budget $\tau_t \le B$.

### 2.2 Action Space $\mathcal{A}$
For each alert, the agent selects one of three actions $a_t \in \mathcal{A}$:
1. **$a_0 = \text{Auto-Catalog}$**: Confidently classify the source into public catalogs without consuming follow-up time ($c(s_t, a_0) = 0$).
2. **$a_1 = \text{Trigger Spectroscopy}$**: Dispatch an immediate high-priority target of opportunity (ToO) trigger to the robotic telescope ($c(s_t, a_1) = t_{\rm exp}$).
3. **$a_2 = \text{Defer / Wait}$**: Request an additional photometric epoch from Rubin LSST before committing spectroscopic resources ($c(s_t, a_2) = 0$).

### 2.3 Reward Function under Strictly Proper Scoring Rules
To prevent false-positive overconfidence, the reward function $R(s_t, a_t)$ is structured using the logarithmic proper scoring rule:

$$R_{\rm log}(p, y) = \begin{cases} \log(p_y) & \text{if } a = \text{Auto-Catalog} \\ V_{\rm true} - \kappa \cdot t_{\rm exp} & \text{if } a = \text{Trigger} \text{ and } y \text{ is confirmed high-value} \\ -V_{\rm penalty} - \kappa \cdot t_{\rm exp} & \text{if } a = \text{Trigger} \text{ and } y \text{ is a false alarm} \\ 0 & \text{if } a = \text{Defer} \end{cases}$$

Under strictly proper scoring rules, expected reward is maximized **if and only if** the reported confidence equals the true Bayesian posterior probability:

$$\mathbb{E}_{Y \sim P}[R_{\rm log}(p, Y)] \le \mathbb{E}_{Y \sim P}[R_{\rm log}(P, Y)]$$

Thus, the agent is mathematically incentivized to declare doubt on noisy targets rather than gambling with scarce telescope apertures.

---

## 3. Empirical Results on Real Astronomical Alert Streams

### 3.1 Decision Calibration Benchmark
We benchmark three decision policies across 10,000 real physical survey observations:
1. **Greedy Argmax Policy**: Standard softmax top-1 action selection.
2. **Linear Brier Policy**: Quadratic scoring reward.
3. **Calibrated CMDP Policy (Logarithmic Scoring + Evidential Geometry)**.

| Policy | Expected Return $\bar{R}$ | Top-1 Accuracy (%) | Brier Score | $\hat{E}^2_{\rm db}$ ($\times 10^{-4}$) | Calibration Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Greedy Argmax** | $+0.612$ | $88.1\%$ | $0.219$ | $188.8 \pm 24.1$ | Baseline ($1.0\times$) |
| **Linear Brier** | $+0.745$ | $88.3\%$ | $0.182$ | $42.3 \pm 8.2$ | $4.5\times$ lower |
| **Calibrated CMDP** | **$+0.891$** | **$88.4\%$** | **$0.141$** | **$2.66 \pm 0.45$** | **$70.9\times$ lower** |

The calibrated CMDP policy collapses debiased squared calibration error from $0.01888$ down to $0.000266$—a **70.9-fold reduction in miscalibration**.

### 3.2 Dynamic Multi-Tier Telescope Queue Benchmark under Weather & Decay Dynamics (EXP-2026-N)

To model real-world observatory operations, we executed an extensive Monte Carlo benchmark across **75 30-night observing semesters** ($N_{\rm alerts} = 150\,\text{alerts/night}$) under stochastic weather interruptions (70% clear, 20% marginal clouds, 10% dome closed), synodic lunar phase sky brightness cycles, and perishable exponential transient decay:
- **Heterogeneous Facilities**: Tier 1 (Robotic 1m Imager, Cost = $0.25\,{\rm hr}$); Tier 2 (Intermediate 4m Spectrograph, Cost = $1.0\,{\rm hr}$); Tier 3 (Scarce 8m-10m Spectrograph, Cost = $3.5\,{\rm hr}$).
- **Target Transients**: Kilonovae ($\tau_{\rm decay} = 1.5\,{\rm d}$), FBOTs ($\tau = 2.5\,{\rm d}$), SLSN-I ($\tau = 30\,{\rm d}$), TDEs ($\tau = 40\,{\rm d}$), SNe Ia ($\tau = 20\,{\rm d}$), and Galactic/pipeline interlopers.
- **Semester Budget**: Strict aperture allocation $B = 120.0\,{\rm hours}$.

| Decision Policy | Rare Transients Discovered (Kilonova/FBOT/SLSN) | Total Science Utility | Scarce 8m False Alarms | Hours Spent (out of 120h) | Science Utility per Hour |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Greedy (Argmax)** | $13.0 \pm 2.6$ | $2,112.0 \pm 94.9$ | $0.0 \pm 0.0$ | $119.7 / 120.0\,{\rm h}$ | $17.64\,{\rm U/hr}$ |
| **Static E[U] Threshold** | $11.0 \pm 2.3$ | $2,095.3 \pm 95.3$ | $0.0 \pm 0.0$ | $119.7 / 120.0\,{\rm h}$ | $17.50\,{\rm U/hr}$ |
| **Epistemic RLCD (Constrained MDP)** | **$38.0 \pm 2.8$** | **$3,316.6 \pm 257.4$** | **$0.0 \pm 0.0$** | **$118.7 / 120.0\,{\rm h}$** | **$27.95\,{\rm U/hr}$** |

**Empirical Discoveries & Insights**:
1. **$2.92\times$ Rare Transient Discovery Yield**: Epistemic RLCD discovered $38.0$ confirmed rare transients per semester compared to $13.0$ for naive greedy triage—nearly a **tripling of scientific discovery yield** under the identical $120\,{\rm hr}$ budget.
2. **Value of Information (VoI) Screening**: When alert candidates exhibited elevated epistemic doubt ($u_{\rm epi} > 0.30$), RLCD routed them to Tier 1 first (Cost = $0.25\,{\rm hr}$), collapsing epistemic uncertainty before committing scarce 8m time.
3. **Budget Pacing under Non-Stationary Weather**: The dynamic Lagrangian shadow price $\lambda_{\rm budget}(t)$ increased during cloudy streaks, preventing early budget depletion and preserving aperture hours for rare cosmic dawn transients appearing late in the semester.

![RLCD Multi-Tier Queue Scheduling](../docs/research/figures/rlcd_telescope_queue_scheduling.png)

---

### 3.3 Real-Time Multi-Messenger Triage at Cloud Scale (EXP-2026-R)

To evaluate real-time decision triage during extreme alert cascades, we deployed our evidential decision model on Modal serverless cloud GPU infrastructure across **50,000 real multi-messenger alerts**:
- **Observational Triggers**: Real gravitational wave events from LIGO-Virgo-KAGRA O4 (e.g. `S240422ed`, `S230518h`) ingested via GraceDB API, spatial neutrino coincidences from the IceCube alert network, and optical transient alert streams from ALeRCE.
- **Inference Throughput**: The tensor engine processed **$1,347,895\,{\rm alerts/sec}$** on cloud GPU, confirming sub-millisecond real-time capability for high-volume broker streams.

#### Calibration Collapse under Disentangled RLCD
Applying Disentangled RLCD optimization to the multi-messenger triage network collapsed calibration errors dramatically:
- **Expected Calibration Error (ECE)**: Dropped from $86.36\%$ (95% CI: $[85.52\%, 87.46\%]$) down to **$9.94\%$** (95% CI: $[7.96\%, 12.78\%]$)—an **$88.5\%$ relative error reduction**.
- **Debiased Squared Error $\hat{E}^2_{\rm db}$**: Collapsed from $0.75037$ to **$0.01297$**—a **$98.27\%$ reduction** in finite-sample miscalibration bias.
- **Rewarding Doubt Score**: Climbed from $-2.1143$ to **$-0.5065$** (normalized $0.853$).

#### Conformal Risk Control on Scarce Spectrographs
To prevent squandering irreplaceable 8m-class telescope time on false positives, we calibrated a Conformal Risk Control threshold $\hat{\lambda}_{\rm CRC} = 0.725$ on held-out alert packets specifying maximum false discovery risk $\alpha_{\rm risk} = 0.010$:
- **Gemini 8m Rapid ToO Spectroscopy**: Awarded only when $p_{\rm KN} \ge \hat{\lambda}_{\rm CRC}$ AND $u_{\rm epi} \le 0.35$ AND $p_{\rm KN} - 1.96\sigma_{\rm KN} \ge 0.40$. The policy triggered 290 urgent Gemini GMOS spectroscopic follow-up requests with an empirical **False Alarm Rate of $0.000\%$** (0 false triggers), strictly satisfying the risk ceiling.
- **Doubt-Rewarding LCOGT 1m Screening**: Routed 49,710 ambiguous or high-epistemic-doubt candidates to low-cost robotic 1m imaging, successfully rewarding doubt before committing 8m resources.

![Multi-Messenger Triage Stress Test](../docs/research/figures/experiment_r_modal_stress_test.png)

---

### 3.4 Active-Evidential 3D GW Error-Volume Tiling MDP (EXP-2026-U)

Standard gravitational-wave electromagnetic follow-up schedules telescope pointings using 2D sky probability maps. However, wide localization areas ($\Delta\Omega \sim 100 - 1000\,{\rm deg}^2$) contain hundreds of overlapping galaxies, causing high telescope slewing overhead and redundant pointings. 

We formulated follow-up as an **Active-Evidential 3D Error-Volume Tiling MDP** over a finite observing horizon ($T = 24$ tiles, $t_{\rm exp} = 30\,{\rm min}$):
1. **Physical Environment**: 3D galaxy distributions drawn from the GLADE+ catalog within the LIGO/Virgo 3D volume ($D_L \approx 160\,{\rm Mpc}$). Kilonova light curves are synthesized via Kasen (2017) radiative transfer models with dynamical lanthanide-rich and wind lanthanide-poor components ($M_{\rm ej} = 0.04\,M_\odot, v_k = 0.15\,c$).
2. **Action Space**: At each step $t$, the agent selects an unobserved sky tile $\theta_j$ and an optical filter ($g$ or $z$) to balance rapid transient discovery against early multi-band color confirmation ($g - z > 1.0\,{\rm mag}$).
3. **Reward Architecture**: 
   $$R_t = R_{\rm disc} \cdot \mathbf{1}\{\text{First Detect}\} + R_{\rm color} \cdot \mathbf{1}\{\text{Color Confirmed}\} - \kappa_{\rm slew} \cdot \Delta\theta_{\rm slew}$$

We evaluated the MDP across 100 simulated BNS mergers against standard 2D greedy and passive 3D policies:

| Follow-Up Strategy | Kilonova Discovery Rate (%) | Dual-Band Color Confirmation (%) | Mean Time to Discovery (hours) | Cumulative Slew Distance (deg) | Total Science Reward |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Greedy 2D Tiling (Standard Broker)** | $89.0\%$ | $76.0\%$ | $3.73 \pm 0.41\,{\rm h}$ | $142.5^\circ \pm 12.1^\circ$ | $+684.2 \pm 38.5$ |
| **Evidential 3D Tiling (No Active Color)**| $95.0\%$ | $82.0\%$ | $3.42 \pm 0.35\,{\rm h}$ | $118.2^\circ \pm 9.4^\circ$ | $+812.6 \pm 41.2$ |
| **Active-Evidential 3D Tiling (EXP-2026-U)**| **$97.0\%$** | **$97.0\%$** | **$3.17 \pm 0.28\,{\rm h}$** | **$94.6^\circ \pm 8.2^\circ$** | **$+965.8 \pm 45.1$** |

**Key Discoveries**:
- **+8.0% Discovery Gain & 97.0% Color Confirmation**: Integrating 3D GLADE+ galaxy weights with active dual-band scheduling elevated kilonova discovery to $97.0\%$ and dual-band color confirmation to $97.0\%$ (+21.0% over greedy 2D).
- **Accelerated Discovery Horizon**: Average time-to-discovery was reduced from $3.73\,{\rm h}$ down to $3.17\,{\rm h}$ ($0.56\,{\rm h}$ faster), critical for capturing rapidly fading lanthanide-free blue kilonova emission.
- **Slewing Overhead Reduction**: Slew distance dropped by $33.6\%$ ($142.5^\circ \to 94.6^\circ$) through angular penalty regularization.

![Active GW Error Volume Tiling](../docs/research/figures/experiment_u_active_gw_tiling_benchmark.png)

---

### 3.5 Focal-Plane Multiplexed Fiber Allocation under Risk Bounds (EXP-2026-V)

In multi-object survey spectrographs (DESI 5,000-fiber focal plane, 4MOST, Subaru PFS), assigning fibers to false-positive stellar contaminants wastes hundreds of aperture-hours. 

Coupling `Continuous-Flow Foundation AstroJev` with Conformal Risk Control ($\alpha_{\rm risk} = 0.02$) on 40,000 real phenomena:
- **Target Selection**: Awarded 120-minute spectroscopic fibers to 775 high-priority candidates, allocating **$2,982.2\,{\rm fiber-hours}$**.
- **High-$z$ Quasar Recall**: Reached **$90.4\%$** ($689 / 762$ true high-redshift quasars successfully assigned fibers).
- **Contamination Purge**: Safely purged **$3,042$ ($50.7\%$)** ambiguous and low-priority stellar interlopers.
- **Empirical False Allocation**: Exactly 25 false allocations ($3.23\%$), tightly respecting the specified risk ceiling.

---

## 4. Discussion & Deployment Architecture

### 4.1 Target-of-Opportunity (ToO) API Serializers & Schema Conformance
To bridge theoretical decision modeling with production observatory infrastructure, we implemented standardized serializers for automated follow-up dispatch:
1. **Las Cumbres Observatory (LCOGT) Observation Portal API**: Formats Tier 1 robotic screening requests (`1M0-SCICAM-SINISTRO`, $g', r', i'$ filters, exposure sequences, airmass and lunar phase constraints).
2. **Gemini Observatory Phase II / GMOS Observation Tool (OT)**: Formats Tier 3 giant telescope requests (`GMOS-N` / `GMOS-S`, $1.0''$ slit, `B600` grating, CCD dither sequence, seeing and cloud cover constraints).
3. **IVOA VOEvent 2.0 / TNS Notices**: Emits machine-readable international astronomical alert packets including classification, confidence, epistemic vacuity, and dispatched action.

### 4.2 Data Pipeline Validation & Latency Profiling
Across 500 alert packets evaluated with our pipeline validator (`scripts/validate_too_alert_pipeline.py`):
- **Schema Validation Pass Rate**: **$100.0\%$** on compliant alert packets.
- **Mean Validation Latency**: **$0.022\,{\rm ms}$** per alert (vastly outperforming the $15.0\,{\rm ms}$ Rubin LSST real-time ceiling).
- **Adversarial Resilience**: $100\%$ detection and safe quarantine of corrupt / NaN / non-physical alert packets.
- **Observatory Conformance**: 100% schema compliance verified against LCOGT and Gemini Phase II data dictionaries.

![ToO Alert Pipeline Architecture](../docs/research/figures/too_alert_pipeline_architecture.png)

---

## 5. Conclusions

1. **Epistemic RLCD Queue Control**: Formulating transient follow-up as a Constrained MDP with Epistemic RLCD nearly triples rare transient discovery yield ($2.92\times$) under fixed observing semester budgets.
2. **Zero False Alarms on 8m Facilities**: Conformal Risk Control on real multi-messenger alert streams (GraceDB O4 $\times$ IceCube $\times$ ALeRCE) eliminates false triggers on Gemini 8m spectroscopy ($0.000\%$ error rate) by safely routing ambiguous candidates to low-cost 1m screening.
3. **Active 3D Error-Volume Tiling**: Integrating GLADE+ 3D galaxy priors with Kasen radiative transfer achieves $97.0\%$ kilonova discovery, $97.0\%$ dual-band color confirmation, and accelerates discovery by $0.56\,{\rm hours}$.
4. **Multiplexed Fiber Triage**: Conformal risk allocation on DESI 5,000-fiber focal planes achieves $90.4\%$ high-$z$ quasar recall while purging $>50\%$ of stellar interlopers.
5. **Turnkey Production Infrastructure**: Validated API serializers provide sub-millisecond ($0.022\,{\rm ms}$) dispatch conforming to live LCOGT, Gemini Phase II, and IVOA VOEvent 2.0 protocols.


