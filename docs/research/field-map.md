# Field map (C′) — where the field spends its effort, June 2026

The operational backbone of the [leverage thesis](./directions.md). Two axes:

- **Effort level** — how crowded a topic is *now* — from "Astrophysics Wrapped 2025"
  (arXiv:2602.12303), a statistical profile of all **18,660** astro-ph papers of 2025.
- **Trajectory** — whether a topic is accelerating or flat — from our own arXiv-API tally
  (`scripts/arxiv_tally.py` → `data/topic_trajectories.csv`), counts per phrase 2022→2025.

Read the gaps: high-effort + accelerating = crowded, avoid. Live-but-flat/neglected +
desk-tractable = our quadrant.

## Effort level — the 2025 census (arXiv:2602.12303)

**18,660 astro-ph papers in 2025** (up from 16,333 in 2024 — ~14% YoY growth, ~51 papers/day
incl. weekends). Most active months: Oct, Sep, Jul.

### Primary subject — paper volume vs. citation impact (Table 1)
| Primary subject | Papers 2025 | /month | Avg citations (AAC) |
|---|--:|--:|--:|
| Astrophysics of Galaxies (GA) | **4,761** | 397 | 2.78 |
| High Energy Astrophysical Phenomena (HE) | 3,869 | 322 | 2.29 |
| Solar & Stellar Astrophysics (SR) | 2,958 | 246 | 1.41 |
| **Cosmology & Nongalactic (CO)** | 2,789 | 232 | **4.83** ← highest impact |
| Earth & Planetary (EP) | 2,293 | 191 | 2.25 |
| Instrumentation & Methods (IM) | 1,722 | 144 | 1.06 |
| Astrophysics (general) | 268 | — | 2.06 |

> Key reading: **Cosmology has the fewest papers of the big four but by far the highest
> citations/paper (4.83 vs ~2.5)** — a small, high-attention field. Crowding there is by
> *quality of competition*, not volume. Galaxy astrophysics is the volume monster (4,761).

### Secondary subjects (Table 2, top of 8,684 with a 2nd subject)
GA 2049 · SR 1632 · IM 1224 · **GR & Quantum Cosmology 1205** · HE 983 · CO 892 ·
HEP-Phenomenology 831 · EP 605 · HEP-Theory 340 · Space Physics 263 · Plasma 223 ·
**Machine Learning 191** · Nuclear Theory 149 · HEP-Experiment 111 · Computational 80.

### Spectral fingerprint — effort by waveband (Fig 4b)
IR **24.4%** (JWST effect) · Visible ~21.5% · UV 16.0% · Microwave 15.9% · Radio 9.4% ·
X-ray 9.1% · Gamma **3.4%**. Redshift effort: huge spike at z≈0, falling with z, small
*reversal at z≈6* (reionization interest).

### Telescopes (Fig 5) — volume vs. citation leaders
- **Volume:** JWST ~1,600 (runaway #1) · GAIA ~900 (decommissioned, still #2 — DR4 awaited) ·
  DESI ~700 · HST ~600 · ALMA ~550 · SDSS ~450 · TESS ~440 · Fermi ~420 · Rubin ~360 · Euclid ~290.
- **Citation/paper leaders:** ACT 14.1 · DESI 10.5 · JWST 5.2 · Einstein Probe 5.1. (ACT/DESI
  punch far above volume — cosmology's outsized impact again.)

### Most-studied specific events (Tables 3–4)
GW events drew 193 papers (top: GW170817 still 76, 8 yrs on); GRB221009A 39; molecular-line
work led by CO 622 / H₂ 365 / HI 335; SN Type Ia 323. Only **6 GW, 41 GRB, 23 FRB** distinct
events account for 80% of each subfield's event papers — attention piles onto a handful of objects.

## Trajectory — our arXiv tally (2022→2025)

`scripts/arxiv_tally.py` → `data/topic_trajectories.csv`. Counts of papers per phrase per year in
`astro-ph.{CO,GA,HE}`+`gr-qc`. `2025/mean` = 2025 vs. the 2022–24 mean (>1 accelerating,
<1 cooling). **Absolute volume matters as much as the ratio** — a small base is the tell of
an uncrowded niche.

| Topic | 2022 | 2023 | 2024 | 2025 | 2025/mean | Read |
|---|--:|--:|--:|--:|--:|---|
| **JWST high-z galaxies** | 88 | 173 | 212 | **313** | **1.99** | exploding + huge → **avoid (E confirmed)** |
| primordial black hole | 257 | 287 | 332 | **353** | 1.21 | huge + growing → crowded (D is a *map*, not a race) |
| fast radio burst | 173 | 200 | 201 | 248 | 1.30 | huge + growing → crowded |
| **Hubble tension** | 124 | 166 | 165 | **204** | 1.35 | big + still hot → don't add an estimate; census only (C) |
| 21 cm cosmology | 125 | 135 | 124 | 170 | 1.33 | big + growing |
| reheating (inflation) | 69 | 95 | 99 | 103 | 1.17 | **moderate** (~100/yr) — *broadly* populated |
| early dark energy | 37 | 52 | 40 | 45 | 1.05 | flat |
| MOND / Milgromian | 41 | 60 | 58 | **40** | **0.75** | **cooling** — hype past peak |
| preheating | 28 | 29 | 26 | 25 | 0.90 | flat/cooling, specialised |
| **inflationary grav. waves** | 4 | 11 | 8 | **13** | **1.70** | **tiny + accelerating → opening (F)** |
| **oscillon** | 5 | 13 | 9 | **11** | 1.22 | **tiny + live → opening (F)** |
| **wide binary gravity** | 4 | 8 | 7 | **11** | **1.74** | **tiny + accelerating inside a cooling parent → ideal (A)** |
| technosignature | 1 | 3 | 8 | 7 | 1.75 | tiny, growing from ~0 |
| **Dyson sphere** | 1 | 2 | 3 | **2** | 1.00 | **near-abandoned** — frozen-archive re-analysis (B) |
| S8 / σ8 tension | 2 | 1 | 1 | n/a | — | phrase too narrow (query noise; widen later) |

## Synthesis → where we fish

Two findings reshape the earlier vibes-based ranking:

**1. A's niche is the *live sub-front of a cooling field* — the best possible signature.**
Broad **MOND is cooling** (0.75, hype past its peak → fewer aggressive competitors), yet the
**wide-binary test specifically is accelerating** (11 papers in 2025, ratio 1.74) off a tiny
base. So the specific, desk-tractable question is alive while the surrounding field has
thinned out. That's exactly the quadrant we want, and it's where curation *is* the science.

**2. F's leverage is real but ONLY in the narrow GW-signature niche.** "Reheating" broadly is
~100 papers/yr — not neglected. But the *observational-falsification-via-GW* micro-niche is
tiny and accelerating (**oscillon 11/yr ×1.22; inflationary GWs 13/yr ×1.70**). So F must be
framed narrowly as the **(inflaton potential + reheating EoS) → GW spectral feature → detector
band** scorecard, not "reheating" in general. Framed that way it's a genuine opening.

**3. The big confirmations:** JWST-high-z is *exploding* (×1.99, 313/yr) — E stays a flank
audit, never a frontal contribution. PBH (353/yr, growing) and Hubble tension (204/yr,
growing) are crowded — so D stays a *falsification map* and C stays a *census/meta*, never
"one more estimate." Dyson spheres are near-abandoned (2/yr) — B's value is entirely in
smarter re-analysis of the now-frozen Gaia/WISE archive.

### Leverage ranking (updated, data-backed)
| Rank | Project | Why the data supports it |
|---|---|---|
| **1** | **A — wide-binary gravity** | tiny live niche (11/yr ↑) inside a cooling parent; curation = science; near-term falsifiable result on public Gaia DR3 |
| **2** | **F — reheating→GW scorecard** (narrow) | highest ceiling; micro-niche (oscillon/infl-GW) tiny + accelerating; theorist-judgment-heavy; payoff is a forward-looking map |
| **3** | **C / C′ — census + cartography** | crowded-but-loud cosmology (4.83 cites/paper) rewards the meta layer nobody maintains; steers 1–2 |
| 4 | B — Dyson re-analysis | near-abandoned topic + frozen archive = re-analysis leverage, but low scientific liveness |
| 5 | D — PBH attack-surface | important but crowded (353/yr); value is a structured map, not a race |
| 6 | E — galaxy-formation audit | only as a flank/systematics audit; the frontier itself is the most crowded thing in the field |

**Operating decision:** build **A** first (real near-term result), develop **F** (narrow GW
scorecard) in parallel as the high-ceiling theory bet, and keep **C′** as the standing steering
layer — re-run `scripts/arxiv_tally.py` quarterly to watch the trajectories move.

### Caveats / next refinements
- The S8/σ8 phrase was too narrow (one query also failed to parse → `n/a`); widen the query
  set and add error-tolerant retries before trusting the small-N rows.
- Counts are title+abstract+full-text `all:` matches → some cross-topic bleed; fine for
  *trajectory*, not for absolute field size. The Wrapped census is the authority on absolute size.
- Quarterly re-run will show *acceleration changes*, which matter more than levels.
