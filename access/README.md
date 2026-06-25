# access/ — query cookbook

Known-good, minimal query patterns for every public archive in
[`../data-atlas.md`](../data-atlas.md). The point: when we start a new probe, we don't
re-derive how to talk to an archive — we copy the locked-in pattern and change the ADQL.

Each file is a **standalone, runnable** script: `python access/<archive>.py` prints a few
rows so you can confirm connectivity before building anything on top. They are deliberately
tiny — the value is the *exact* import + method + return-type, which differs per archive.

## The one pattern, six dialects

Almost everything here is **TAP + ADQL** (SQL for catalogues). The differences are only the
client class and how you get an astropy `Table` back:

`✓` = smoke-tested live on this machine (astroquery 0.4.11 / astropy 8.0 / pyvo 1.9.1,
2026-06-25); `·` = API verified against current docs, not yet run here.

| Archive | Script | Import / call | Status | Notes |
|---|---|---|:--:|---|
| ESA Gaia | `gaia.py` | `Gaia.launch_job_async(adql).get_results()` | ✓ | anonymous OK; results purged 72h |
| ESA Euclid | `euclid.py` | `Euclid.launch_job(adql).get_results()` | · | Q1/Q2 public; DR1 Oct 2026 |
| NASA IRSA | `irsa.py` | `Irsa.query_tap(adql).to_table()` | ✓ | WISE/CatWISE/2MASS/SPHEREx |
| NASA MAST | `mast.py` | `Observations.query_criteria(...)` | · | JWST/HST/TESS; obs-metadata model |
| NASA HEASARC | `heasarc.py` | `Heasarc.query_tap(adql)` | · | X-ray/γ; returns Table directly |
| VizieR | `vizier.py` | `Vizier(...).query_constraints(...)` | ✓ | any *published* catalogue (NVSS, …) |
| NOIRLab Data Lab | `datalab_desi.py` | `queryClient.query(sql=...)` | · | needs `astro-datalab`; server-side |
| Generic IVOA TAP | `vo_generic.py` | `pyvo.dal.TAPService(url).search(adql)` | ✓ | RACS/CASDA, LoTSS, anything VO |
| SIMBAD + NED | `resolvers.py` | `Simbad.query_object` / `Ned.query_object` | ✓¹ | object ID, type, **per-object bibliography** |
| CDS X-Match | `xmatch.py` | `XMatch.query(cat1, cat2, max_distance)` | ✓ | scale cross-match — the audit primitive |
| *(cache)* | `cache.py` | `cached_query(archive, adql, fetch)` | ✓ | wrap any pull → local cache + provenance log |
| *(imaging)* | `cutouts.py` | `panel(ra, dec)` / `color(ra, dec)` | ✓ | multi-wavelength image panel from coords |

¹ SIMBAD paths live-tested; NED depends on a frequently-slow server (snippet degrades
gracefully on timeout). See [`../toolbox.md`](../toolbox.md) for the wider tooling map.

## Install

```bash
pip install -r access/requirements.txt
```

## Composing: cache any pull, then look

```python
from access import gaia, cache, cutouts
adql = "SELECT TOP 100 source_id, ra, dec FROM gaiadr3.gaia_source WHERE ..."
tab = cache.cached_query("gaia", adql, lambda: gaia.query(adql))  # cached + logged
cutouts.panel(213.6905918, -12.5801013)  # multi-wavelength PNG of a position
```

`cache.cached_query` wraps *any* of the query helpers: identical query → instant disk
hit; every fetch is appended to `data/manifest.jsonl` so each figure traces to its query.
`cutouts.panel(ra, dec)` / `cutouts.color(ra, dec)` render a position across the spectrum
(SkyView UV→radio + CDS HiPS colour). Both write to `data/` (git-ignored).

## Conventions baked into every snippet

- **`TOP n` / row caps** — never pull the whole table; the desk filter is "select rows, not
  pixels." Start with `TOP 5`, widen deliberately.
- **Discovery helpers** — most clients can list tables/columns (`Gaia.load_tables()`,
  `Irsa.list_catalogs()`, `Heasarc.list_catalogs()`, `Vizier.find_catalogs()`,
  `queryClient.schema()`). Use them before guessing column names.
- **Cone vs. full-table** — for a position use a `CONTAINS(POINT, CIRCLE)` ADQL clause or the
  client's `query_region`; for a science cut use a `WHERE` on magnitudes/flags.
- **Not included: Rubin** — DP1/DP2 are data-rights-gated (Rubin Science Platform account),
  not anonymously queryable. Add a `rubin.py` only once we have rights / an open release.

## Gotchas worth knowing once

- **Table/column names drift between releases** — always confirm against the live schema,
  don't trust a hard-coded name from an old script.
- **Sync vs async** — sync (`launch_job`) is fine under ~2k rows; use `_async` above that.
- **IRSA `query_tap`** returns a PyVO result → call `.to_table()`.
- **Euclid bulk** — for images/cutouts run on ESA Datalabs (notebooks next to the data),
  not local download; see the ESA-Datalabs/Euclid-Q1 GitHub tutorials.
