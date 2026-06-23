# astro-theory

Theory-driven **candidate-generation pipelines** for citizen cosmology & astronomy.
The premise: from a desk, the binding constraint isn't photons — it's *ideas and analysis*.
The public archives (Gaia, WISE, 2MASS, Planck, DESI) are richer than the community can
exploit. So we work the literature side: turn a theoretical signature into concrete
selection cuts, pull a *small* local dataset (a few thousand rows via TAP/ADQL, not a
petabyte), produce a ranked candidate list, triage it against follow-up papers, iterate.

## The working loop

```
theory / arXiv paper
  → distinctive observable signature
    → selection cuts (the intellectual core — where a theorist's judgment competes with pros)
      → cross-match a small public catalog locally  (SQLite / parquet, kept and grown)
        → ranked candidate list
          → triage against follow-up literature
            → iterate
```

The "small local dataset" is usually the *output* of the cuts, not a big download.

## Roles

- **Leon** — physical judgment: what's worth predicting, which cuts are defensible.
- **Claude** — literature throughput, turning signatures into ADQL/Python, bookkeeping,
  maintaining the candidate DB and the literature census.

## The four candidate projects

### A. Wide-binary gravity test (MOND vs. Newton) — *first build*
Wide binaries at ~1–30 kAU separation probe accelerations near Milgrom's
a₀ ≈ 1.2×10⁻¹⁰ m/s². The candidate list **is** the experiment, and the whole result
hinges on sample-selection choices — exactly where careful, theory-aware curation moves
the needle. Highest "real physics per laptop-hour."

### B. Technosignature / Dyson-sphere IR-excess candidates
Cross-match Gaia × 2MASS × WISE for stars with anomalous W3/W4 infrared excess not
explained by debris disks / YSOs / blends. Reproduce a published template, then extend
to new excess models and host types.

### C. Tension census (living literature database) — *persistent background project*
Pick a tension (Hubble H₀, or S₈) and build a structured DB of *every* measurement —
method, value, error, sample, systematics, date — from continuous arXiv scraping.
The dataset is the literature, distilled. Feeds future candidate hunts and plays to
Claude's reading throughput.

### D. Theory-falsification map for one non-standard model
Pick one model (primordial black holes as DM, cosmic strings, a modified-gravity theory)
and systematically extract *every* distinctive observable from the literature, then rank
each by which public dataset could test it and how desk-tractable that is. Output: a
prioritized "attack surface."

**Plan:** start with **A** as the first concrete build; run **C** as the persistent
background project. (A + C.)

## State of the field — June 2026 snapshot

> Captured at project start so we know our baseline. Cite-and-update as we go.

### A — Wide binaries: the tide has turned toward Newton (but on *method*)
- **Cookson, Banik, El-Badry, Sutherland, Penoyre, Pittordis & Clarke (2026)**, *MNRAS*
  547(2), "A quality framework for testing gravity with wide binaries: no evidence for
  MOND." A rigorous quality checklist — degrouping to remove triples, `RUWE < 1.25`,
  HR-diagram main-sequence selection, scaled-velocity cut ṽ < 2.5, ΔRV < 10 km/s —
  applied to Gaia DR3 within **130 pc**, separations **1–30 kAU**, yields 1,421 clean
  systems. Conclusion: **no MOND velocity boost; Newton up to ~1500× more likely** for
  the cleanest sample. https://academic.oup.com/mnras/article/547/2/stag342/8497444
- Banik et al. (2024, *MNRAS* 527) earlier claimed strong constraints *against* MOND;
  Chae's independent analyses of the *same* Gaia data claimed the opposite. The fight has
  always been about **contamination treatment** (hidden tertiaries, chance alignments,
  projection/eccentricity priors) — i.e. sample selection. The 2025 OJA "realistic triple
  modelling" paper (arXiv:2504.07569) found Newton fits better but flagged that the triple
  population must be better understood to be decisive.
- **Implication for us:** the frontier is now *purity of selection*, which is precisely a
  desk task. A defensible independent re-derivation of the cuts (and an honest look at
  which side's priors hold up) is a genuine contribution, not a toy.

### B — Dyson spheres: archives are now *frozen*, raising the value of re-analysis
- **Suazo et al. (2024)**, *Project Hephaistos II* (arXiv:2405.02927): pipeline over ~5M
  objects → **7 M-dwarf candidates** with strong W3/W4 excess within ~900 ly (+~53 larger
  hosts out to ~6500 ly).
- **Contamination pushback:** arXiv:2405.14921 argues several candidates suffer background
  contamination (e.g. blended galaxies); 2025 high-res radio imaging (arXiv:2501.05152,
  *MNRAS Letters*) found no radio signal for one candidate.
- **Amiri (2026)**, "Dyson spheres on the H-R diagram" (arXiv:2602.23270, accepted to
  *Universe*): radiative-balance placement of spheres on the HRD, T ∝ R_D^(−1/2), arguing
  **white dwarfs and M-dwarfs are the cleanest host regimes**. A fresh theoretical handle
  for *new* cuts.
- **Implication for us:** Gaia is decommissioned and WISE expired (2024) — no successor
  wide IR survey imminent. So the *frozen* archive elevates the value of smarter
  re-analysis (better confounder rejection, WD/M-dwarf-targeted cuts).

### C — Hubble tension: persists, but the SH0ES-vs-CCHP gap is narrowing on *method*
- **SH0ES (Cepheids):** ~73 km/s/Mpc, late 2025.
- **CCHP (JWST):** TRGB H₀ = 69.85 ± 1.75(stat) ± 1.54(sys); Cepheids 72.05 ± 1.86 ± 3.10;
  best TRGB-only ≈ 70.39 ± 1.22 ± 1.33. TRGB & JAGB agree at ~1%; differ from Cepheids at
  2.5–4%. https://iopscience.iop.org/article/10.3847/1538-4357/adce78
- Riess et al.: JWST **rejects Cepheid crowding** as the explanation at 8σ
  (https://iopscience.iop.org/article/10.3847/2041-8213/ad1ddd).
- **Implication for us:** the live question is the **distance-ladder method spread**
  (Cepheid vs TRGB vs JAGB), not just "early vs late." A census structured by *method and
  calibration choice* is the useful cut.

### D — PBH dark matter: asteroid-mass window still (barely) open
- The window sits between evaporation (~10¹⁷ g) and microlensing (~10²³ g). Recent work
  (arXiv:2403.03839) says it stays open for all-DM **unless the mass function is wide**.
- 2026 papers keep probing it: synchrotron constraints (arXiv:2601.19386), SUSY/MSSM
  production shifting peaks into the window (arXiv:2604.26005). Near-future MeV telescopes
  are the decisive probe (arXiv:2102.06714).
- **Implication for us:** good **D** candidate — the "attack surface" is unusually
  well-mapped and several rungs are desk-analysable.

## Stack (planned)

Python: `astropy`, `astroquery` (Gaia/Vizier TAP), `lightkurve`, `pandas`, `CAMB`/`CLASS`
for C-side cosmology. Candidate DBs as SQLite/parquet under `data/` (git-ignored).

## Status

- [x] Repo + scaffold
- [x] State-of-field baseline (June 2026)
- [ ] **A:** pull the Cookson et al. 2026 cut-list → reproduce as ADQL against Gaia DR3
- [ ] **C:** schema for the H₀ measurement DB + first arXiv ingestion pass
