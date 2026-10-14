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

### Rubin access reality -> Celestrium feature shape

Rubin is the main case where "open-source" and "open data" can mislead us. The
pipelines, schemas, and documentation are inspectable, but the full Rubin Science
Platform archive is not an anonymous public survey backend. DP1/DP2, prompt
processed images, and annual release products are data-rights/RSP surfaces with
proprietary periods. Do not model Rubin as another `gaia`/`irsa`/`euclid` TAP
archive until we have data rights or a genuinely public query surface.

The open feature worth adding is narrower and still useful: a **Rubin watch**
layer for live sky activity.

- **Public inputs:** broker alert APIs/substreams, world-public alert packet
  contents, MPC solar-system submissions from Rubin observatory code `X05`,
  and the public scheduler/live-status pages.
- **Useful Celestrium actions:** "show Rubin alerts near this object/field",
  "watch this candidate list for new transients", "summarize last-night Rubin
  activity in these atlas fields", and "cross-match a broker alert against our
  existing object/candidate context."
- **Artifact shape:** row-capped alert tables, small alert cutout references or
  extracted postage stamps when provided by the broker, per-field watch reports,
  and provenance entries in `data/manifest.jsonl`.
- **Implementation boundary:** keep this out of the generic archive registry
  unless/until the PPDB or a public TAP-like service is actually available.
  Start with a dedicated service module plus packet builders so the CLI/TUI can
  share it; tests should use canned broker/MPC responses and remain hermetic.

In short: first Rubin feature = **live alert intelligence around our targets**,
not "download Rubin images".

---

## 2. Key release calendar (2026–2027)

| Date | Release | Why it matters to us |
|---|---|---|
| 19 Mar 2025 | Euclid **Q1** (deep fields, ~63 deg²) | photo-z / high-z flank (E); live now |
| Jul 2025 | **SDSS-V DR19** | spectro census, BHM/MWM |
| **24 Jun 2026** | Euclid **Q2** — Galactic Bulge Survey | stellar/microlensing; **off our cosmology axes** |
| **Jul–Sep 2026** | Rubin **DP2** (deep coadds) | data-rights-gated; watch for open subsets |
| **21 Oct 2026** | Euclid **DR1** (~1900 deg² wide) | ★ first deep optical/NIR, WISE/Gaia-independent → dipole test ([`euclid-dr1-prep.md`](../programmes/dipole.md)) |
| **~late 2026** | **Gaia DR4** (full astrometry + epoch data) | huge: wide binaries (A), QSO astrometry, variability |
| 2026 (rolling) | **SPHEREx** quick releases (all-sky NIR spectra) | ★ all-sky selection-independent — dipole sibling probe |
| ~2027 | **DESI DR2**, Rubin **DR1** | next-gen spectro + optical; mostly forecast/prep now |

*Verify each date before quoting in an analysis — release schedules slip.*

---

## 3. Data sources by wavelength and modality

Access key: **open** (anonymous) · **have** (credential configured here) · **want** (free account worth getting) · **gated** (data rights / paid).
"Module" is our client, if any. ✓ = already pulled real data in this repo.

### Time domain: optical and archival plates (P2 core)
| Source | Span / depth | What it gives | Access | Module |
|---|---|---|---|---|
| **DASCH DR7** (Harvard plates) | 1885–1992, all sky, B≈12–18 | century light curves, plate cutouts, exposure lists | open (public API); full tier needs `DASCHLAB_API_KEY` | `archives/dasch.py` ✓ |
| **APPLAUSE DR4** (Hamburg, Bamberg, Potsdam, Tartu, ... plates) | 1890s–1990s, mostly north | plate catalogues + scans; independent check on DASCH | open TAP; account for bulk | — (next) |
| **AAVSO** International Database + VSX | 1900s→now, visual + CCD | long light curves; variable-star catalogue (VSX) | **have** (`AAVSO_TOKEN` in `.env`); VSX open | `scripts/novae/fetch_tcrb_aavso.py` ✓ |
| **ASAS-SN Sky Patrol v2** | 2012→now, V/g≲17, all sky | light curves anywhere | open (`pyasassn`) | — |
| **ZTF** (IRSA light curves; forced photometry ZFPS) | 2018→now, r≲20.5, north | DR light curves; forced phot at any position | open (DR); **want** ZFPS account (email registration) | via `tap.py` (IRSA) |
| **ATLAS** forced photometry | 2015→now, c/o≲19.5, all sky | forced phot at any position | **want** (free account + token) | — |
| **Gaia** DR3 epoch phot / alerts → **DR4** (Dec 2026) | 2014→2020(DR4), G≲20 | epoch photometry, XP spectra, astrometry; Gaia Alerts | open; optional archive account for big async jobs | `tap.py` ✓ |
| **OGLE** (bulge/disk), **KMTNet**, **VVV/VVVX** (NIR) | 1992→now, dense fields | bulge light curves; NIR variability | open (download / VizieR) | `vizier.py` |
| **TESS / Kepler / K2** | 2009→now, minutes cadence | flicker, rotation, eclipses | open (MAST) | `mast.py` ✓ |
| **Brokers**: ALeRCE, Fink, Lasair, ANTARES | live alerts (ZTF → Rubin) | alert streams, cross-matches | open (Lasair token: **want**) | `alerce.py` |
| **Rubin** alerts / DP1–DR1 | 2025→ | alerts public via brokers; images & catalogues data-rights | alerts open; archive **gated** | — |
| **TNS** (Transient Name Server) | live | official transient reports, novae (AstroNotes) | read open; **want** bot API key for programmatic queries | — |

### Ultraviolet & X-ray & gamma (high-energy accretion, shocks)
| Source | What | Access | Module |
|---|---|---|---|
| **Swift** UVOT / XRT (HEASARC) | UV + soft X-ray monitoring of novae/symbiotics | open | `heasarc.py`, `fetch_tcrb_swift_*.py` ✓ |
| **GALEX** | FUV/NUV all-sky snapshot (2003–13) | open (MAST) | `mast.py` |
| **HST/IUE** spectra | UV spectroscopy archives (IUE: 1978–96, many symbiotics) | open (MAST) | `mast.py` |
| **XMM-Newton**, **Chandra**, **NuSTAR**, **NICER**, **eROSITA DR1** | X-ray spectra/timing; eROSITA half-sky survey | open (HEASARC / ESA / MPE) | `heasarc.py` |
| **MAXI**, **Swift/BAT** | daily all-sky X-ray monitoring | open | — |
| **Fermi-LAT** | GeV γ-rays (nova shocks) | open (FSSC) | — |

### Infrared & spectra
| Source | What | Access | Module |
|---|---|---|---|
| **2MASS, AllWISE/CatWISE, unWISE, NEOWISE** | NIR/MIR colours; NEOWISE 2014–24 MIR light curves (dusty symbiotics) | open (IRSA) | `tap.py`, `xmatch.py` ✓ |
| **SPHEREx** (2026 rolling) | all-sky 0.75–5 µm spectrophotometry | open (IRSA) | — |
| **Gaia XP / RVS** | low-res optical spectra for ~220M stars | open | `tap.py` |
| **SDSS-V / LAMOST / DESI / GALAH / APOGEE** | optical/IR spectra; symbiotic and accretor discovery | open (SDSS, DESI); LAMOST registration for some DRs | `sdss.py`, `datalab_desi.py` ✓ |
| **ARAS** amateur spectroscopy database | dense spectral monitoring of symbiotics/novae | open download | — |
| **ESO archive** (Phase 3 products) | reduced spectra/images (X-shooter, UVES, ...) | open TAP; account for raw data | `tap.py` |

### Radio, CMB, multimessenger, catalogues
| Source | What | Access | Module |
|---|---|---|---|
| **NVSS, FIRST, VLASS, RACS, LoTSS** | radio continuum surveys (dipole; nova shocks) | open (CASDA download needs OPAL account: **want** if used) | `vizier.py`, `cutouts.py` ✓ |
| **Planck PLA, ACT DR6, SPT** | CMB maps, lensing, SZ | open | — |
| **GWOSC**, **GraceDB**, **IceCube/GCN** | GW strain/events; alerts | open | `transients/` |
| **Euclid** Q1 / DR1 (21 Oct 2026) | deep optical/NIR wide survey | open (anonymous TAP; account for Datalabs) | `tap.py` |
| **SIMBAD, VizieR, NED, CDS X-Match** | identifiers, published tables, cross-matching | open | `resolvers.py`, `vizier.py`, `xmatch.py` ✓ |
| **ADS / SciX**, arXiv | literature, citations, BibTeX | **have** (`~/.ads/dev_key`); arXiv open | `ads.py` ✓ |
| **CBAT** CBETs | nova/comet discovery telegrams | **gated** (subscription) | — |
| **ATel** | astronomer's telegrams | open read; posting requires account (Leon's decision) | — |

## 3b. Credentials

**Configured here** (never print or commit values):
- `AAVSO_TOKEN` in the gitignored `.env`: AAVSO data downloads.
- ADS dev key in `~/.ads/dev_key` (or `$ADS_DEV_KEY`): literature search.
- Modal token in `~/.modal.toml`: cloud GPU (balance in ledger §4).
- Cloudflare OAuth via `npx wrangler login`: site deploys.
- Git identity Sub-Surface: GitHub pushes.

**Worth getting (free), in rough priority for current work:**
1. **ATLAS forced photometry** account. Modern all-sky light curves at any position, for symbiotic and nova candidates.
2. **ZTF forced photometry (ZFPS)** registration. Deeper northern forced photometry.
3. **TNS bot API key**. Programmatic queries of reported transients and novae.
4. **DASCH full-tier API key** (`DASCHLAB_API_KEY`, request from the DASCH team). The public tier has worked for everything so far.
5. **Lasair token**. Rubin-era alert filters.
6. **Gaia / ESA Datalabs account**. Large async ADQL jobs and Euclid notebooks.
7. **CASDA OPAL** (only if we use RACS images) and **LAMOST** registration (only if we use its spectra).

**Not available to us**: Rubin data rights (spectate via brokers), CBAT subscription (paid).

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
| **DASCH / APPLAUSE plates** | ★★★ | ★★★★★ | century baseline, complete since 2024, barely mined systematically (EXP-2026-Z) |
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
2. **G-Euclid** continues as the lead build ([`euclid-dr1-prep.md`](../programmes/dipole.md)):
   σ_D forecast on a realistic DR1 footprint → measure-vs-rehearse decision.
3. **Scope SPHEREx** as the all-sky dipole sibling — its selection-independence may matter
   more than DR1's depth for the isotropy question. Check first-release date + catalogue form.
4. Keep this atlas current; re-rank when DR1 / Gaia DR4 / SPHEREx land.
