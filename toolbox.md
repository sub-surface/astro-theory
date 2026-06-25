# Toolbox — research force-multipliers & quality-of-life

The third pillar of the hub. [`data-atlas.md`](./data-atlas.md) = *what data exists*;
[`access/`](./access/) = *how to pull it*; **this file = everything around the query** that
makes the work faster and more correct. Ordered by leverage *for our edge* (cross-catalogue
audits, selection-function re-analysis, living meta-analyses), not by fame.

★ = set this up soon, it compounds. ◐ = grab when the relevant project needs it.

---

## 1. Literature layer — our census edge, made programmatic

- **★ ADS / SciX (Science Explorer)** — the big one beyond EsaSky's lookup. Free API token
  (ui.adsabs.harvard.edu/user/settings/token); use via `astroquery.nasa_ads` (`na.ADS.TOKEN`)
  or `pip install ads`. Gives **citation/reference graphs, full-text search, bibcodes, and
  saved "libraries"** — i.e. *programmatic* versions of what we do by hand for projects C/C′.
  Supersedes hand-scraping; `arxiv_tally.py` becomes one input among many.
- **INSPIRE-HEP** (inspirehep.net, open API) — the hep-th/gr-qc index. The right tool for the
  **F reheating→GW** thread (Copeland-circle papers live here, not always in ADS-astro).
- **arXiv API** — already wired (`fetch_papers.sh`, `arxiv_tally.py`); keep for trajectories.
- **Semantic Scholar / Connected Papers / Litmaps** — citation-graph *discovery* ("what cites
  this anomaly paper, what does it cite") to seed a census without missing a branch.
- **★ A repo-local `refs.bib`** — one BibTeX file as the spine of every dossier/census, fed by
  ADS exports. Cheap now, invaluable once the literature DB (C) grows.
- **Per-object bibliography** — SIMBAD/NED (below) return *the papers on a given object*,
  automating the EsaSky→ADS lookup you've been doing by hand.

## 2. Object resolution & cross-ID

- **SIMBAD** (`astroquery.simbad`) — name → type, coords, basic data, **bibliography per
  object**. The resolver for "what is this source."
- **NED** (`astroquery.ipac.ned`) — extragalactic objects: redshifts, cross-IDs, photometry.
  Essential for QSO/galaxy work (the dipole populations).
- **Sesame** — lightweight name→coordinate resolver (under the hood of the above).
- → see new `access/resolvers.py`.

## 3. Cross-matching at scale — directly serves move (a)

- **★ CDS X-Match** (`astroquery.xmatch`) — server-side cross-match of *our* table against any
  VizieR/SIMBAD catalogue, millions of rows, by sky position. This **is** the cross-catalogue
  consistency audit primitive (CatWISE ↔ Quaia ↔ Euclid under a common match radius).
  → see new `access/xmatch.py`.
- **STILTS / TOPCAT** — the table workhorse. **STILTS** is scriptable/CLI: crossmatch, HEALPix
  density maps, sky plots, format conversion — many `g_*.py` operations done faster and
  battle-tested. Worth having even though we script in Python.

## 4. Footprint / mask / HEALPix — the partial-sky need (Euclid DR1)

- **healpy** — already in use (`selfunc_*_nside64.fits`). Pixelised maps, `anafast`, masks.
- **★ MOCpy** — Multi-Order Coverage maps: **footprint algebra** (union/intersection/area of
  survey footprints). Exactly the tool for DR1's irregular ~1900 deg² and for "where do
  Euclid and Quaia overlap" — the mask-coupling problem in `euclid-dr1-prep.md`.
- **astropy-healpix, regions** — coordinate/region helpers; ecliptic/galactic frames for
  isotropy systematics.
- **Aladin Lite / pyESASky** — visual footprint overlays (sanity-check masks by eye).

## 5. Selection-function reference data — the systematics fight

- **★ dustmaps** (`pip install dustmaps`; SFD, Planck, Bayestar) — Galactic extinction
  correction. Extinction + stellar density are *top* contaminants of a number-count dipole;
  having this on hand is non-optional for defensible dipole/isotropy cuts.
- **Gaia star-density / scanning-law maps**, survey **depth/completeness maps** — the inputs
  that turn "raw counts" into "selection-corrected counts." Pull per-survey as needed.

## 6. Reproducibility & QoL — compounds every session

- **✅ Query cache + provenance manifest** — *built*: [`access/cache.py`](./access/cache.py).
  `cached_query(archive, adql, fetch)` hashes each query, caches the result as ECSV under
  `data/cache/`, and appends `{query, archive, UTC date, nrows}` to `data/manifest.jsonl`.
  Stops us re-pulling and makes every figure traceable to its query.
- **✅ Multi-wavelength image cutouts** — *built*: [`access/cutouts.py`](./access/cutouts.py).
  `panel(ra, dec)` renders a position across UV→optical→near-IR→mid-IR→radio (SkyView) and
  `color(ra, dec)` a CDS-HiPS colour image — a quick "what's actually here?" from coordinates.
- **Environment pinning** — one `requirements.txt` / lockfile for the whole repo (we have
  `access/requirements.txt`; promote to a repo-level env when the analysis deps settle).
- **Server-side notebooks** — ESA Datalabs, NOIRLab Astro Data Lab, SciServer, Rubin RSP: run
  *next to* the data for anything image-heavy instead of downloading.
- **Units & frames discipline** — `astropy.units`/`coordinates` everywhere; never pass bare
  floats between galactic/ecliptic/ICRS in isotropy work (a classic silent-bug source).

## 7. Quick wins — status

1. ✅ **`access/resolvers.py` + `access/xmatch.py`** — object-lookup + cross-match audit core.
2. ✅ **`access/cache.py` + `access/cutouts.py`** — query cache/provenance + multi-λ imaging.
3. ◻ **ADS/SciX token** → drop in `~/.ads/dev_key`; start a repo `refs.bib`. (Unlocks §1.)
4. ◻ **`pip install dustmaps mocpy`** — the reference tools the dipole work needs next
   (extinction corrections + Euclid DR1 partial-sky footprint algebra).
