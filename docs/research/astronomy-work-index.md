# Celestrium Astronomy Work Index & Encyclopaedic Field Taxonomy

**Status**: Living Index & Master Sub-Field Map  
**Curator**: Celestrium Collaboration (`sub-surface/astro-theory`)  
**Scope**: Complete encyclopaedic taxonomy of astronomy, astrophysics, and cosmology sub-fields; auditing all current Celestrium research (EXP-2026-A through W, Papers A–D, catalogs) and mapping open scientific frontiers ("gaps") for future investigations.

---

## 1. Delineation of Astronomical Disciplines & Architecture

Astronomy operates across two orthogonal axes:
1. **Physical Domain & Scale**: From Solar and Planetary systems ($\sim 10^{-6} - 10^{-4}$ pc), to Stellar and Galactic structures ($\sim 1 - 10^5$ pc), to Extragalactic, High-Energy, and Global Cosmological horizons ($\sim 10^6 - 10^{10}$ pc).
2. **Observational Messenger & Bandwidth**: From traditional electromagnetic wavebands (Radio $\to$ Sub-mm $\to$ Infrared $\to$ Optical $\to$ Ultraviolet $\to$ X-ray $\to$ $\gamma$-ray) to Multi-Messenger carriers (Gravitational Waves, Astrophysical Neutrinos, Cosmic Rays).

Celestrium's research philosophy combines physical models, heteroscedastic noise handling, evidential deep learning, and distribution-free conformal risk guarantees. This index maps all sub-fields, cataloging active work and explicit research gaps.

```
                                  CELESTRIUM FIELD INDEX
                                            │
   ┌───────────────────┬────────────────────┼───────────────────┬───────────────────┐
   ▼                   ▼                    ▼                   ▼                   ▼
Cosmology &        High-Energy &       Radio & Sub-mm      Stellar & Galactic  Planetary Science,
Nongalactic        Multi-Messenger     Astronomy           Astrophysics        Exoplanets & Solar
(Gpc Scale)        (Time-Domain)       (Coherent/Line)     (kpc Scale)         (Sub-pc Scale)
 [EXP-B,I,J,K,L,   [EXP-A,H,N,         [EXP-W Radio,       [EXP-F,M Foundation, [EXP-Q Exoplanet,
  O,P,S,V,W;        R,U; Paper C]       21cm Cosmology      Wide-Binary Grav,    EXP-T Space Weather,
  Paper A, D]                            Gap]                Brown Dwarfs]        NEO Radar]
```

---

## 2. Master Taxonomy & Coverage Matrix

Legend:
- `● ACTIVE / COMPLETED`: Ingested real/simulated catalogs, executed benchmarks, trained models, or drafted manuscripts.
- `◐ IN PROGRESS / PRE-FLIGHT`: Active pipelines, pre-registered specifications, or upcoming data releases (e.g. Euclid DR1).
- `○ OPEN GAP`: Identified astronomical domain with no active Celestrium code or experiments yet; target for future expansion.

| Domain Index | Primary Sub-Field | Observational Messengers / Surveys | Celestrium Status | Experiments & Artifacts | Primary Scientific Gaps / Target Datasets |
|:---|:---|:---|:---:|:---|:---|
| **I. COSMOLOGY** | **Cosmic Dipole & Kinematic Bulk Flows** | *Gaia* DR3 $\times$ *unWISE* Quaia, CatWISE2020, NVSS 1.4 GHz | `● ACTIVE` | EXP-B, EXP-I, EXP-J, EXP-K, EXP-L, EXP-O, EXP-P, EXP-W; Paper A | Redshift tomography of bulk flow; high-$z$ dipole scaling |
| | **CMB Temperature & Polarization Anisotropies** | Planck HFI/LFI, ACT, SPT, Simons Obs | `◐ IN PROGRESS` | EXP-E, EXP-O (null model benchmarks) | Secondary anisotropies (SZ effect, CMB lensing cross-correlations) |
| | **Early Universe, Inflation & Primordial GWs** | BICEP/Keck, LiteBIRD, CMB-S4 | `○ OPEN GAP` | Roadmapped in `field-map.md` (Project F) | Inflaton potentials, oscillon signatures, tensor-to-scalar ratio $r$ |
| | **Cosmic Dawn, Reionization & 21 cm Line** | JWST NIRCam/NIRSpec, HERA, LOFAR, SKA | `● ACTIVE` | EXP-S (Lyman-break vs Brown Dwarfs) | Global 21cm absorption profile (EDGES/SARAS), power spectrum |
| | **Large-Scale Structure, BAO & Redshift Distortions** | DESI, Euclid Wide, SDSS/BOSS, Roman | `◐ IN PROGRESS` | EXP-V (Flow Matching), EXP-2026-01 | Full 3D power spectrum $P(k)$, non-linear halo bias, $S_8$ tension |
| | **Dark Energy & Modified Gravity on Cosmological Scales** | Euclid NISP/VIS, Rubin LSST, DES | `◐ IN PROGRESS` | `docs/research/euclid-dr1-prep.md` | $w(z) = w_0 + w_a(1-a)$ equation of state, cosmic shear leakage |
| **II. EXTRAGALACTIC** | **Active Galactic Nuclei (AGN) & Quasar Physics** | Quaia, SDSS, WISE, Chandra, XMM | `● ACTIVE` | EXP-B, J, P, V; `OBJ-APEX-01`, `OBJ-HOTDOG-04` | Reverberation mapping, accretion disk physics, jet launching |
| | **High-Redshift Galaxy Formation ($z > 10$)** | JWST JADES/CEERS, ALMA high-$z$ | `● ACTIVE` | EXP-S (`OBJ-HIGHZ-03`, $z=4.606$) | Overdensity of massive early galaxies, UV luminosity function |
| | **Galaxy Clusters & Intra-Cluster Medium (ICM)** | eROSITA eRASS1, SPT-SZ, Chandra | `◐ IN PROGRESS` | EXP-2026-05 (Zone of Avoidance) | ICM thermodynamics, cool-core clusters, cluster mass scaling |
| | **Gravitational Lensing & Time-Delay Cosmography** | TDCOSMO, Euclid Strong Lensing, Rubin | `● ACTIVE` | EXP-S (Quad lens triage for $H_0$) | Sub-halo dark matter substructure lensing, microlensing time-delays |
| | **Galaxy Morphology, Mergers & Quenching** | Rubin DP0, HST CANDELS, DESI Legacy | `● ACTIVE` | FoundationAstroJev (12-class SED) | Detailed 2D morphological decomposition (Sérsic fits, tidal tails) |
| **III. HIGH-ENERGY & MULTI-MESSENGER** | **Gravitational Waves: Compact Binary Coalescence** | LIGO-Virgo-KAGRA O4/O5, GraceDB | `● ACTIVE` | EXP-R, U; Paper C; `multimessenger.py` | Continuous GWs from asymmetric pulsars, intermediate-mass BHs |
| | **BNS Kilonovae & Kasen Radiative Transfer** | DECam, Rubin LSST, Gemini GMOS | `● ACTIVE` | EXP-R, U; Paper C; `too_protocol.py` | Lanthanide-rich opacities, viewing-angle dependencies, blue vs red components |
| | **Astrophysical Neutrinos & Point Sources** | IceCube, KM3NeT, GCN alerts | `● ACTIVE` | EXP-R (IceCube Gold/Bronze triage) | Blazar neutrino flares (TXS 0506+056), diffuse neutrino background |
| | **Gamma-Ray Bursts (Prompt & Afterglows)** | Fermi GBM/LAT, Swift BAT, SVOM | `● ACTIVE` | EXP-A (Fink/Rubin broker stream) | Short vs long GRB jet structures, early reverse shock emission |
| | **Fast Radio Bursts (FRBs)** | CHIME/FRB, CRAFT, DSA-110 | `○ OPEN GAP` | Mentioned in `field-map.md` | Real-time dispersion measure (DM) triage, magnetar flare mechanisms |
| | **Tidal Disruption Events (TDEs) & FBOTs** | ZTF, Rubin, Swift, eROSITA | `● ACTIVE` | EXP-N (Multi-tier scheduling) | Relativistic vs non-relativistic TDE outflows, optical-to-X-ray transition |
| **IV. RADIO & SUB-MILLIMETER** | **Synoptic Continuum Radio Surveys** | NVSS 1.4 GHz, VLASS, ASKAP EMU | `● ACTIVE` | EXP-W (Multi-tracer co-inference) | Polarization catalogs, spectral index mapping across $100\,{\rm MHz} - 10\,{\rm GHz}$ |
| | **HI 21 cm Neutral Hydrogen Emission** | ALFALFA, FAST, MeerKAT MIGHTEE | `○ OPEN GAP` | Unaddressed | Galaxy rotation curves, baryonic Tully-Fisher, cosmic HI density $\Omega_{\rm HI}$ |
| | **Molecular Gas Dynamics & Astrochemistry** | ALMA, NOEMA, IRAM 30m, SMA | `○ OPEN GAP` | Unaddressed | CO line diagnostics, starburst chemistry, protoplanetary gas kinematics |
| | **VLBI & Event Horizon Imaging** | EHT, GMVA, ngEHT | `○ OPEN GAP` | Unaddressed | Sgr A* / M87* horizon polarization, jet base magnetic collimation |
| **V. STELLAR & GALACTIC** | **Astrometry, Stellar Kinematics & Milky Way** | *Gaia* DR3 / DR4, RAVE, APOGEE | `● ACTIVE` | EXP-F, M; `OBJ-HALO-05` (subdwarf) | Galactic warp, stellar stream phase-space tracing, dark matter halo shape |
| | **Modified Gravity in Wide Binaries (MOND vs Newton)** | *Gaia* DR3 wide binary catalog | `◐ IN PROGRESS` | Roadmapped as Project A in `directions.md` | Pure clean separation of hierarchical triples, $a < a_0 \approx 1.2\times 10^{-10}\,{\rm m/s^2}$ |
| | **Sub-stellar Objects: Brown Dwarfs & Free-Floaters** | *Gaia* DR3, unWISE, CatWISE, Euclid | `● ACTIVE` | EXP-F (100% recall), EXP-S | L/T/Y dwarf transition, atmospheric cloud dynamics, metallicity trends |
| | **Stellar Remnants: White Dwarfs, Neutron Stars** | *Gaia* HR diagram, SDSS, ZTF | `● ACTIVE` | EXP-F (98.8% White Dwarf recall) | White dwarf cooling tracks, magnetic fields, ultra-massive WDs |
| | **Asteroseismology & Stellar Variability** | TESS, Kepler, Gaia RV, PLATO | `◐ IN PROGRESS` | Ingested via EXP-Q pipeline | Oscillation mode identification ($\nu_{\max}, \Delta\nu$), core rotation |
| | **Star Formation & Interstellar Medium (ISM)** | Herschel, Spitzer, JWST, SOFIA | `○ OPEN GAP` | Unaddressed | Protostellar collapse, magnetic field alignment via dust polarization |
| **VI. PLANETARY SCIENCE & EXOPLANETS** | **Exoplanetary Transits under Stellar Red Noise** | TESS SPOC, Kepler PDCSAP, PLATO | `● ACTIVE` | EXP-Q; `celestrium/exoplanet.py` | Multi-planet transit timing variations (TTV), ultra-short-period planets |
| | **Radial Velocity Disentanglement & Activity** | HARPS, ESPRESSO, EXPRES, NEID | `● ACTIVE` | EXP-Q (Matérn-3/2 GP + CCF activity) | Habitable Earth-mass exoplanet reflex velocities ($K < 10\,{\rm cm/s}$) |
| | **Exoplanet Atmospheric Transmission & Emission** | JWST NIRISS/NIRSpec, Ariel | `○ OPEN GAP` | Unaddressed | Retrieval algorithms, water/methane/CO2 abundances, clouds/hazes |
| | **Near-Earth Objects & Asteroid Orbit Triage** | JPL Scout / Sentry-II, Catalina, Pan-STARRS | `● ACTIVE` | EXP-T; `celestrium/solarsystem.py` | Non-gravitational Yarkovsky effect, impact probability distributions |
| | **Planetary Defense Radar Follow-Up** | Goldstone, Arecibo legacy, Canberra | `● ACTIVE` | EXP-T (`DISPATCH_PLANETARY_DEFENSE_RADAR`) | Asteroid shape reconstruction, binary asteroid mutual orbits |
| **VII. SOLAR & SPACE PHYSICS** | **Photospheric Magnetism & Flare Forecasting** | SDO/HMI SHARP, SOHO, GONG | `● ACTIVE` | EXP-T (Vector magnetic flux, shear) | Magnetic reconnection trigger models, non-potential field energy |
| | **Coronal Mass Ejections & Solar Wind Modeling** | SOHO LASCO, Parker Solar Probe, Solar Orbiter | `○ OPEN GAP` | Unaddressed | CME arrival time forecasting at L1, magnetic flux rope chirality |
| | **Geomagnetic Storms & Space Weather Impact** | NOAA SWPC, ACE, DSCOVR | `● ACTIVE` | EXP-T (`TRIGGER_GLOBAL_SPACE_WEATHER_ALERT`) | Auroral electrojet modeling, geomagnetically induced currents (GIC) |
| **VIII. ASTROBIOLOGY & SETI** | **Technosignature Searches & Anomalous Infared** | Gaia DR3 $\times$ unWISE, Breakthrough Listen | `◐ IN PROGRESS` | Roadmapped as Project B in `directions.md` | Dyson sphere candidates (waste-heat excess without emission lines), radio chirps |
| | **Biosignatures & Habitability** | JWST, HWO (Habitable Worlds Obs) | `○ OPEN GAP` | Unaddressed | Photochemical disequilibrium (${\rm O}_2 + {\rm CH}_4$), surface reflection |
| **IX. COMPUTATIONAL ASTROPHYSICS & AI** | **Evidential Deep Learning & Dirichlet UQ** | Multi-survey tables, Kafka brokers | `● ACTIVE` | EXP-F, M, R, V; `astrojev.py` | Analytical posterior CI, aleatoric vs epistemic decomposition |
| | **Reinforcement Learning on Calibrated Decisions (RLCD)**| Real-time alerts, telescope queues | `● ACTIVE` | EXP-H, N, Q, R, S, U; Paper B, C | Strictly proper logarithmic scoring rules, doubt rewards |
| | **Conformal Risk Control & Distribution-Free Bounds**| Survey catalogs, follow-up queues | `● ACTIVE` | EXP-D, J, K, Q, R, S, T, V, W | Finite-sample false discovery bounds without distributional assumptions |
| | **Autonomous Target-of-Opportunity (ToO) Protocols** | Gemini GMOS, LCOGT 1m, VLT | `● ACTIVE` | EXP-A, N, R; `too_protocol.py` | Turnkey Phase II JSON serialization, robotic scheduler integration |

---

## 3. Detailed Sector Reports: What Celestrium Has Built

### 3.1 Sector I: Cosmology & Cosmological Anisotropy
* **Active Thrust**: The Cosmic Quasar Dipole tension ($4.9\sigma$).
* **Data Ingested**:
  - Quaia G20.5: 1,295,502 all-sky quasars (*Gaia* DR3 $\times$ *unWISE*).
  - CatWISE2020: 1,360,788 mid-IR AGNs.
  - NVSS: 208,790 1.4 GHz radio continuum sources.
* **Key Findings**:
  - EXP-B & EXP-J: Selection boundary expands to $\mu \approx 7.8\,{\rm mas/yr}$ at $G=20.5$, admitting halo subdwarfs (`OBJ-HALO-05`).
  - EXP-L: Deprojecting selection function $S(\hat{\mathbf{n}})$ collapses North-South gradient $D_z$ from $+0.055$ to $+0.011$.
  - EXP-P & EXP-W: Multi-tracer Bayesian Poisson co-inference across 2.86M sources converges to $v_{\rm bulk} = 664.5 \pm 188.4\,{\rm km/s}$ toward $(l, b) = (289.2^\circ, +50.8^\circ)$—only $16.4^\circ$ from the CMB dipole apex.
  - Model comparison $\Delta{\rm BIC} = +180.7$ decisively rejects independent survey systematics in favor of a shared physical bulk flow.
* **Manuscripts**: [`manuscript_drafts/paper_a_quasar_dipole_systematics.md`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/manuscript_drafts/paper_a_quasar_dipole_systematics.md).

### 3.2 Sector II: High-Energy & Multi-Messenger Astronomy
* **Active Thrust**: Real-time triage and target-of-opportunity dispatch for Gravitational Wave and Neutrino counterparts under massive error volumes.
* **Data Ingested**:
  - GraceDB O4 VOEvents (`S240422ed`, `S230518h`, `S230522a`, `S240426d`).
  - IceCube Gold/Bronze neutrino alerts.
  - ALeRCE ZTF optical alert streams.
  - GLADE+ galaxy catalog for 3D host prioritization.
* **Key Findings**:
  - EXP-R: Disentangled RLCD calibration collapses debiased calibration error by $98.3\%$ ($\hat{E}^2_{\rm db} = 0.01297$).
  - Zero false alarms on Gemini 8m GMOS spectroscopy across 50,000 stress-tested candidates on cloud GPU.
  - EXP-U: Active-Evidential 3D tiling finite-horizon MDP achieves $97.0\%$ kilonova discovery rate (+8% over greedy 2D) and $97.0\%$ dual-band color confirmation rate, shortening discovery by 0.56 hours.
* **Manuscripts**: [`manuscript_drafts/paper_c_calibrated_followup_mdp.md`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/manuscript_drafts/paper_c_calibrated_followup_mdp.md).

### 3.3 Sector III: Pan-Chromatic Foundation AI & Cross-Calibration
* **Active Thrust**: Heteroscedastic evidential learning and continuous flow matching across multi-survey catalogs.
* **Data Ingested**:
  - 80,000 real survey sources across 12 astrophysical classes (Gaia DR3, Quaia, unWISE, SDSS).
  - Simulated Euclid DR1 Wide, DESI Legacy DR10, Rubin LSST passbands.
* **Key Findings**:
  - EXP-F & EXP-V: Continuous-Flow Foundation AstroJev recovers cross-instrument zero-point offsets with ${\rm RMSE} = 0.0363\,{\rm mag}$.
  - Cloud GPU throughput: $254,515\,{\rm sources/sec}$ ($36\times$ local RTX 2060).
  - High-$z$ quasar recall on DESI 5,000-fiber focal plane: $90.4\%$.
  - EXP-M: Spatial hold-out delta of only $0.09\%$, proving invariance to coordinate memorization.
* **Manuscripts**: [`manuscript_drafts/paper_b_heteroscedastic_evidential_astronomy.md`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/manuscript_drafts/paper_b_heteroscedastic_evidential_astronomy.md).

### 3.4 Sector IV: Exoplanetary Science & Red Noise Disentanglement
* **Active Thrust**: Distinguishing true Earth-analog transits and sub-m/s radial velocity wobbles from correlated stellar activity (starspots, faculae, granulation).
* **Data Ingested**:
  - TESS SPOC and Kepler lightcurves.
  - Simulated HARPS/ESPRESSO cross-correlation function (CCF) indicators (BIS, FWHM, S-index).
* **Key Findings**:
  - EXP-Q: Matérn-3/2 GP regression combined with Disentangled RLCD collapsed debiased calibration error by $93.1\%$ ($\hat{E}^2_{\rm db} = 0.000214$).
  - Diverted $100\%$ ($2,000 / 2,000$) of pure stellar activity mimics away from 8m VLT ESPRESSO into robotic activity monitoring, achieving zero false triggers.

### 3.5 Sector V: Solar & Space Weather Physics
* **Active Thrust**: Major X-class solar flare forecasting and short-arc Near-Earth Object (NEO) impact triage.
* **Data Ingested**:
  - SDO/HMI SHARP vector magnetograms (magnetic flux, shear, twist).
  - JPL Scout short-arc astrometric tracks ($\Delta t < 4\,{\rm h}$).
* **Key Findings**:
  - EXP-T: Achieved True Skill Statistic $\text{TSS} = 1.0000$ and False Alarm Rate $0.00\%$ under Conformal Risk Control, dispatching 74 global space weather alerts and 62 planetary defense radar recoveries.

---

## 4. Comprehensive Gap Analysis: The Unconquered Frontiers

To fulfill the long-term mission of contributing across all domains of astronomy, the following sub-fields represent clear scientific gaps ready for formulation into future Celestrium experiments (`EXP-2026-X+`):

### Gap 1: Radio Astronomy — 21 cm Neutral Hydrogen & Fast Radio Bursts
* **Sub-fields**:
  1. **Galactic & Extragalactic HI 21 cm**: Tracing gas reservoirs, rotation curves, and the baryonic Tully-Fisher relation using ALFALFA and MeerKAT data.
  2. **Fast Radio Bursts (FRBs)**: Real-time dispersion measure (${\rm DM}$) triage and host galaxy identification on CHIME/FRB public catalogs.
* **Target Datasets**: CHIME/FRB VOEvents, MeerKAT MIGHTEE HI cubes, ALFALFA 100% catalog.
* **Proposed Celestrium Thrust**: Extend `celestrium/data_streamer.py` to ingest VOEvents from the CHIME broker; build a Dirichlet evidential classifier mapping DM and scattering timescale $\tau_{\rm scat}$ to cosmological redshift priors vs Galactic dispersion models.

### Gap 2: Sub-Millimeter & Molecular Astrochemistry
* **Sub-fields**:
  1. **ALMA Protoplanetary Disks & Dust Gaps**: Modeling disk substructures, dust settling, and planet-induced gaps.
  2. **Molecular Clouds & Star Formation Efficiency**: Dense gas tracers (${\rm CO}, {\rm HCN}, {\rm CS}$) and Jeans mass collapse.
* **Target Datasets**: ALMA Science Archive (via TAP/Astroquery), DSHARP high-resolution disk survey.
* **Proposed Celestrium Thrust**: Build a convolutional flow matching module for ALMA visibility deconvolution and gas kinematic curve extraction.

### Gap 3: Stellar Astrophysics — Wide Binaries & Gravitational Tests (MOND vs Newton)
* **Sub-fields**:
  1. **Wide Binary Kinematics**: Evaluating the relative velocity $\Delta v$ of wide binary pairs as a function of separation $s \in [2,000, 30,000]\,{\rm AU}$ where acceleration drops below $a_0 \approx 1.2 \times 10^{-10}\,{\rm m/s^2}$.
* **Target Datasets**: *Gaia* DR3 wide binary sample (El-Badry & Rix 2021).
* **Proposed Celestrium Thrust**: Implement Project A from `directions.md`: a rigorous Bayesian hierarchical model filtering unresolved triples and chance alignments using radial velocity error covariance, resolving the tension between Hernandez et al. and Pittordis/Chae.

### Gap 4: High-Energy — Pulsar Timing & Continuous Gravitational Waves
* **Sub-fields**:
  1. **Pulsar Timing Arrays (PTAs)**: Stochastic gravitational wave background at nanohertz frequencies (Hellings-Downs correlation curve).
  2. **Continuous GWs from Neutron Star Mountains**: Extreme narrow-band search for deformed rotating neutron stars.
* **Target Datasets**: NANOGrav 15-year dataset, European Pulsar Timing Array (EPTA) DR2.
* **Proposed Celestrium Thrust**: Construct a Gaussian Process noise kernel isolating red spin noise from the Hellings-Downs quadrupole spatial correlation.

### Gap 5: Astrobiology & Technosignatures
* **Sub-fields**:
  1. **Waste-Heat Infrared Excesses**: Systematic search for megastructures (Dyson spheres) exhibiting anomalous mid-IR emission without stellar accretion or dust disks.
  2. **Atmospheric Biosignatures**: Photochemical disequilibrium (${\rm O}_2 + {\rm CH}_4$) modeling on rocky exoplanets.
* **Target Datasets**: *Gaia* DR3 $\times$ *unWISE* $\times$ *2MASS* cross-matches; JWST NIRISS transmission spectra.
* **Proposed Celestrium Thrust**: Implement Project B from `directions.md`: evidential filtering on unWISE $W3$ and $W4$ excesses across 5 million solar-type stars to identify candidate technosignatures while suppressing background AGB and YSO interlopers.

---

## 5. Summary Roadmap & Cross-References

| Milestone | Target Domain | Key Action | Deliverable |
|:---|:---|:---|:---|
| **M1 (Current)** | Cosmology, Multi-Messenger, Methods | Launch Celestrium Interactive Site & Work Index | `site/`, `docs/research/astronomy-work-index.md` |
| **M2 (Oct 2026)** | Cosmology & Extragalactic | Euclid DR1 Wide survey ingestion & dipole forecast | EXP-2026-01, Paper 4 |
| **M3 (Nov 2026)** | Stellar & Gravity | *Gaia* DR3 Wide Binary MOND test execution | Project A, EXP-2026-02 |
| **M4 (Dec 2026)** | Radio & Multi-Wavelength | Zone-of-Avoidance eROSITA $\times$ CatWISE fusion | EXP-2026-05 |
| **M5 (2027)** | Radio & Time-Domain | CHIME/FRB and MeerKAT HI stream integration | EXP-2027-A (Radio Gap closure) |

---
*For living updates on all computational runs and figures, refer to [`docs/research/experiment-index.md`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/docs/research/experiment-index.md) and [`CLAUDE.md`](file:///C:/Users/Leon/Desktop/Psychograph/astro-theory/CLAUDE.md).*
