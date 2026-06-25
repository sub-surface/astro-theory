# Data Atlas — public astronomical data, ranked for desk work

The standing map for this repo. Premise (from [`README.md`](../README.md)): our binding
constraint isn't photons, it's ideas + analysis. So we catalogue **what's publicly
queryable from a laptop**, rate it on **data quality** and — separately — on **leverage for
*us*** (a desk team whose edge is selection-function systematics, cross-catalogue audits,
and living meta-analyses, *not* new observations). Quality ≠ leverage: Rubin is superb data
we mostly can't touch yet; NVSS is old radio data that sits at the heart of a live >5σ anomaly.

> **Living document.** Update release states as data drops. Companion to
> [`directions.md`](./directions.md) (leverage thesis) and
> [`field-map.md`](./field-map.md) (where the field spends effort).

---

## 1. How to read access (the primitives that make something "desk-tractable")

| Primitive | What it is | Desk verdict |
|---|---|---|
| **TAP + ADQL** | SQL-like queries to a catalogue server; returns rows, not images | ★ ideal — pull a few k–M rows, never the petabyte |
| **astroquery** | Python wrappers (`astroquery.gaia`, `.mast`, `.ipac.irsa`, `.esa.euclid`, `.heasarc`, …) | ★ our default interface |
| **Direct catalogue download** | FITS/parquet/CSV tables (e.g. Legacy Surveys sweeps, CatWISE) | ★ fine if ≲ a few GB; stream/filter |
| **ESA Datalabs / Rubin RSP / cloud notebooks** | Run analysis *next to* the data, no download | ◐ great for big images; some are access-gated |
| **Bulk image archives** | Pixel data (HST/JWST/Euclid VIS) | ✗ usually too big / not our mode unless cutout API |
| **Data-rights-gated** | Account required, restricted membership (Rubin DP) | ✗ spectate until open |

**The desk filter:** prefer *catalogue-level* data reachable by TAP/astroquery, where the
science lives in the *selection* of rows, not in pixel reduction.

---

## 2. Key release calendar (2026–2027)

| Date | Release | Why it matters to us |
|---|---|---|
| 19 Mar 2025 | Euclid **Q1** (deep fields, ~63 deg²) | photo-z / high-z flank (E); live now |
| Jul 2025 | **SDSS-V DR19** | spectro census, BHM/MWM |
| **24 Jun 2026** | Euclid **Q2** — Galactic Bulge Survey | stellar/microlensing; **off our cosmology axes** |
| **Jul–Sep 2026** | Rubin **DP2** (deep coadds) | data-rights-gated; watch for open subsets |
| **21 Oct 2026** | Euclid **DR1** (~1900 deg² wide) | ★ first deep optical/NIR, WISE/Gaia-independent → dipole test ([`euclid-dr1-prep.md`](./euclid-dr1-prep.md)) |
| **~late 2026** | **Gaia DR4** (full astrometry + epoch data) | huge: wide binaries (A), QSO astrometry, variability |
| 2026 (rolling) | **SPHEREx** quick releases (all-sky NIR spectra) | ★ all-sky selection-independent — dipole sibling probe |
| ~2027 | **DESI DR2**, Rubin **DR1** | next-gen spectro + optical; mostly forecast/prep now |

*Verify each date before quoting in an analysis — release schedules slip.*

---

## 3. The archives, by ecosystem

### ESA — `astroquery.esa.*`, ESASky, ESA Datalabs (GitHub tutorials)
- **Gaia** (`astroquery.gaia`) — DR3 public (1.8B sources, astrometry+photometry+QSO/galaxy
  tables); **DR4 ~late 2026**. ★ Our workhorse: wide binaries (A), QSO catalogue, proper
  motions. Anonymous ADQL, results auto-purged 72h.
- **Euclid** (`astroquery.esa.euclid`) — Q1/Q2 public; DR1 Oct. ESA-Datalabs/Euclid-Q1
  GitHub tutorials (cutouts, ADQL, jdaviz). Run on Datalabs to avoid bulk download.
- **Planck** — Planck Legacy Archive: CMB maps, component-separated, catalogues (PCCS, SZ).
- **XMM-Newton, INTEGRAL, Herschel** — via ESASky / dedicated archives.

### NASA MAST — `astroquery.mast` (STScI; optical/UV/NIR space)
- **JWST** — imaging+spectra; deep fields (JADES, CEERS, COSMOS-Web). ◐ Pixel-heavy, but
  catalogues from survey teams are tractable; the **high-z census (E)** lives here.
- **HST** — decades of imaging; legacy fields.
- **TESS / Kepler / K2** — full-frame + light curves. ★ Time-domain desk gold via
  `lightkurve`: transits, rotation, flares, asteroseismology.
- **Pan-STARRS, GALEX** — optical/UV all-(north-)sky catalogues, TAP-queryable.

### NASA IRSA — `astroquery.ipac.irsa` (IPAC; infrared + survey catalogues)
- **WISE / AllWISE / CatWISE2020** — mid-IR all-sky. ★ Heart of the dipole anomaly
  (Secrest); already in our archive.
- **NEOWISE** — time-domain mid-IR.
- **2MASS** — NIR all-sky (J/H/Ks).
- **Spitzer, IRAS** — legacy IR.
- **SPHEREx** (2025-launch, releasing 2026) — ★ **all-sky NIR spectro-photometry**, a
  selection-*independent* probe → natural dipole + ISW + high-z sibling.
- **COSMOS / deep-field value-added** — multiwavelength photo-z catalogues for (E).

### NASA HEASARC — `astroquery.heasarc` (high-energy; cloud-friendly)
- **Chandra, XMM, NuSTAR, Swift, Fermi (γ-ray), eROSITA-DR1 (western gal. hemisphere, X-ray)**,
  ROSAT. ◐ More specialised; AGN/cluster cross-matches, X-ray dipole as another isotropy front.

### NOIRLab Astro Data Lab — `datalab` client (ground-based optical, server-side notebooks)
- **DESI** (DR1 public: 18.7M spectra; DR2 ~2027) — ★ spectroscopic redshifts, BAO, QSO/ELG
  catalogues; peculiar-velocity survey (bulk-flow sibling probe).
- **DES, DECaLS / Legacy Surveys** (sweeps), **NSC**, **SDSS**. ★ Server-side SQL; huge
  imaging catalogues without downloading pixels.

### Radio — NRAO / CASDA / LOFAR / SARAO
- **NVSS, FIRST, VLASS** (NRAO), **RACS** (ASKAP/CASDA), **LoTSS-DR2** (LOFAR). ★ The radio
  leg of the dipole anomaly (Böhme et al.); NVSS already in our archive.

### CMB & multimessenger
- **Planck PLA, ACT DR6, SPT** — CMB maps/spectra, lensing, SZ clusters.
- **GWOSC** (LIGO/Virgo/KAGRA, fully open strain + event catalogues) — desk-doable parameter
  re-analysis, population studies.
- **NASA Exoplanet Archive** — confirmed planets + TESS candidates, TAP-queryable.

---

## 4. Quality × leverage ranking (for *us*)

Quality = intrinsic data depth/precision. Leverage = marginal value *a desk team with our
edge* can add now (open access + selection-systematics-limited + uncrowded + foundational).

| Dataset | Quality | Leverage (us) | Why |
|---|:--:|:--:|---|
| **Euclid DR1** (Oct) | ★★★★★ | ★★★★★ | WISE/Gaia-independent dipole test; we're building the pipeline now |
| **CatWISE/Quaia/NVSS** | ★★★☆ | ★★★★★ | live >5σ anomaly, selection-limited = our edge (the G work) |
| **SPHEREx** (2026) | ★★★★ | ★★★★ | all-sky, selection-independent; new, uncrowded dipole/ISW sibling |
| **Gaia DR3 → DR4** | ★★★★★ | ★★★★ | wide-binary gravity (A), QSO astrometry; DR4 is a big new surface |
| **DESI DR1** | ★★★★★ | ★★★★ | spectro-z truth, BAO, peculiar-velocity/bulk-flow front; open |
| **TESS/Kepler** | ★★★★ | ★★★☆ | time-domain desk gold; crowded but vast, niche cuts open |
| **JWST deep fields** | ★★★★★ | ★★★☆ | high-z census flank (E); systematics audit, not pixel races |
| **eROSITA DR1** | ★★★★ | ★★★ | X-ray isotropy/AGN; only half-sky public, more specialised |
| **GWOSC** | ★★★★ | ★★★ | open, re-analysable, but pro-saturated population work |
| **Rubin DP1/DP2** | ★★★★★ | ★☆ | superb but **data-rights-gated** — spectate until open |

---

## 5. Opportunity map — keyed to our three repeatable moves

Our edge (from `directions.md`) is three under-rewarded things: **(a) cross-catalogue
consistency audits, (b) selection-function re-analysis of live anomalies, (c) living
meta-analyses.** Each archive above feeds at least one:

- **(a) Cross-catalogue audits** — the dipole across CatWISE × Quaia × NVSS × **Euclid DR1**
  × **SPHEREx** under a *common* mask/cut. *This is the live G-Euclid line.* Also: do
  DESI spectro-z's confirm the photometric QSO selections the dipole relies on?
- **(b) Selection-function re-analysis** — wide-binary purity in **Gaia DR4** (A); high-z
  "impossible galaxy" photo-z systematics in **JWST/Euclid deep fields** (E); Dyson-sphere
  IR-excess confounders in **WISE×Gaia×2MASS** (B, frozen archive).
- **(c) Living meta-analyses** — H₀/S₈ tension census (C); the field-effort map (C′,
  `field-map.md`); a *dipole-measurement* census (every D, direction, significance, mask).

**Sibling isotropy probes worth a standing watch:** bulk-flow non-convergence (DESI PV
survey, CosmicFlows), X-ray source dipole (eROSITA), and the all-sky SPHEREx number-count
dipole — same Copernican-principle question, different systematics. Agreement across
*independent* selection functions is the whole game.

---

## 6. Immediate next steps

1. **Stand up the access layer** — a tiny `celestrium/` cookbook: one minimal working
   astroquery/TAP snippet per archive (Gaia, Euclid, IRSA/WISE, MAST, DESI Data Lab,
   HEASARC) so any future probe starts from a known-good query. Mirror the ESA-Datalabs
   tutorial patterns.
2. **G-Euclid** continues as the lead build ([`euclid-dr1-prep.md`](./euclid-dr1-prep.md)):
   σ_D forecast on a realistic DR1 footprint → measure-vs-rehearse decision.
3. **Scope SPHEREx** as the all-sky dipole sibling — its selection-independence may matter
   more than DR1's depth for the isotropy question. Check first-release date + catalogue form.
4. Keep this atlas current; re-rank when DR1 / Gaia DR4 / SPHEREx land.
