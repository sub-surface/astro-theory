#!/usr/bin/env python
"""registry.py — the repo's data tables for the hub, in one place.

This is the *data* layer the CLI (`hub.py`) and the future TUI (`tui/`) both read,
so neither surface hardcodes archives/recipes/targets/runbooks. Design rule: the
CLI and TUI present; this module (with `packets.py`) holds what they present.

Four tables:
  - ARCHIVES        queryable archives, each a uniform `.query(adql) -> Table`
  - SAMPLE_RECIPES  named low-compute science pulls
  - ATLAS_TARGETS   curated visual targets for contact sheets / posters
  - RUNBOOKS        data-driven multi-step desk-work packets

Archive note: the underlying modules are NOT uniform — gaia/irsa/euclid/heasarc
take ADQL directly, but datalab_desi returns pandas unless fmt='table', and
vo_generic needs an endpoint URL first. We wrap the non-uniform ones in lazy
adaptors so (a) the call signature is always `.query(adql)` and (b) heavy/optional
deps (`dl`, `pyvo`) are imported only when that archive is actually used.
"""
import importlib
from dataclasses import dataclass
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Archives
# --------------------------------------------------------------------------- #
class _LazyAdaptor:
    """Expose a uniform `.query(adql)` over a module whose real signature differs.

    The module is imported on first use (not at registry import) so optional deps
    like `dl` (astro-datalab) and `pyvo` never break `import celestrium.hub`.
    """

    def __init__(self, module: str, call: Callable):
        self._module = module
        self._call = call

    def query(self, adql: str):
        mod = importlib.import_module(self._module)
        return self._call(mod, adql)


@dataclass(frozen=True)
class Archive:
    key: str
    label: str
    adql: bool          # True if `query` takes ADQL/SQL; False = different mode
    source: object      # module path str (lazy-imported) OR object with .query
    health_url: str = ""  # VO /availability (or equivalent) endpoint for `doctor`


# Native-ADQL archives are kept as module-path strings: lazily imported via
# importlib, exactly as the original hub did (keeps anonymous/optional deps lazy).
# Non-uniform archives use a _LazyAdaptor so the registry stays import-safe.
ARCHIVES = {
    "gaia": Archive("gaia", "ESA Gaia DR3 (TAP/ADQL)", True, "celestrium.gaia",
                    "https://gea.esac.esa.int/tap-server/tap/availability"),
    "irsa": Archive("irsa", "NASA/IPAC IRSA — WISE/2MASS/SPHEREx", True, "celestrium.irsa",
                    "https://irsa.ipac.caltech.edu/TAP/availability"),
    "euclid": Archive("euclid", "ESA Euclid Q1/Q2 (DR1 prep)", True, "celestrium.euclid",
                      "https://eas.esac.esa.int/tap-server/tap/availability"),
    "heasarc": Archive("heasarc", "NASA HEASARC — X-ray/gamma", True, "celestrium.heasarc",
                       "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/availability"),
    "desi": Archive(
        "desi", "NOIRLab Astro Data Lab — DESI/Legacy", True,
        _LazyAdaptor("celestrium.datalab_desi", lambda m, q: m.query(q, fmt="table")),
        "https://datalab.noirlab.edu/tap/availability",
    ),
    "casda": Archive(
        "casda", "CSIRO CASDA — ASKAP/RACS (TAP)", True,
        _LazyAdaptor("celestrium.vo_generic",
                     lambda m, q: m.query(m.ENDPOINTS["casda"], q)),
        "https://casda.csiro.au/casda_vo_tools/tap/availability",
    ),
    "vizier-tap": Archive(
        "vizier-tap", "VizieR over generic TAP", True,
        _LazyAdaptor("celestrium.vo_generic",
                     lambda m, q: m.query(m.ENDPOINTS["vizier"], q)),
        "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/availability",
    ),
}

# Legacy mapping kept for backward-compat: hub.py re-exports this, and the test
# suite patches it via `monkeypatch.setitem(hub.QUERY_ARCHIVES, ...)`. Values are
# module-path strings or objects exposing `.query(adql)` — resolved by
# resolve_query_source below. Extends the original four with the adaptor archives.
QUERY_ARCHIVES = {k: a.source for k, a in ARCHIVES.items()}


def resolve_query_source(key: str, table: Optional[dict] = None):
    """Resolve an archive key to an object exposing `.query(adql)`, or None.

    `table` lets callers pass a patched/overridden mapping (the CLI passes its
    re-exported QUERY_ARCHIVES so tests that monkeypatch it still win)."""
    src = (table or QUERY_ARCHIVES).get(key.lower())
    if src is None:
        return None
    if isinstance(src, str):
        return importlib.import_module(src)
    return src


# --------------------------------------------------------------------------- #
# Global feeds (live sky/event streams — not target/field products)
# --------------------------------------------------------------------------- #
# key -> (module path, zero-arg fetch function returning an astropy Table|None).
# Data, not code: both surfaces resolve these lazily via importlib so the CLI
# `feed` command and the TUI `/global` browser share one executor table (and
# the same cache tags in cache.cached_query).
GLOBAL_FEED_EXECUTORS = {
    "neo": ("celestrium.neos", "fetch_close_approaches"),
    "satellite": ("celestrium.satellites", "fetch_visible_satellites"),
    "transient": ("celestrium.transients", "fetch_latest_transients"),
}

# Friendly plurals/synonyms -> canonical feed keys (shared by CLI + TUI prompts).
FEED_ALIASES = {
    "neos": "neo",
    "cneos": "neo",
    "satellites": "satellite",
    "tle": "satellite",
    "tles": "satellite",
    "transients": "transient",
    "alerts": "transient",
}


def feed_key(text: str) -> str:
    """Normalise user feed spellings ('neos', 'tles', …) to canonical keys."""
    key = text.strip().lower()
    return FEED_ALIASES.get(key, key)


# --------------------------------------------------------------------------- #
# Sample recipes (low-compute, row-capped science pulls)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleRecipe:
    description: str
    archive: str
    adql: str


SAMPLE_RECIPES = {
    "gaia-bright-nearby": SampleRecipe(
        "Gaia DR3 bright nearby seed stars",
        "gaia",
        "SELECT TOP 25 source_id, ra, dec, parallax, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source "
        "WHERE parallax > 20 AND phot_g_mean_mag < 10 "
        "ORDER BY phot_g_mean_mag",
    ),
    "wise-agn-colors": SampleRecipe(
        "AllWISE colour-selected AGN-like sources in a tiny row-capped pull",
        "irsa",
        "SELECT TOP 25 designation, ra, dec, w1mpro, w2mpro, w3mpro "
        "FROM allwise_p3as_psd "
        "WHERE w1mpro - w2mpro > 0.8 AND w2mpro < 15",
    ),
    "heasarc-chandra-gc": SampleRecipe(
        "HEASARC Chandra observations near the Galactic Centre",
        "heasarc",
        "SELECT TOP 25 name, ra, dec, time, status "
        "FROM chanmaster "
        "WHERE CONTAINS(POINT('ICRS', ra, dec), "
        "CIRCLE('ICRS', 266.4, -29.0, 0.5))=1",
    ),
    "euclid-q1-smoke": SampleRecipe(
        "Euclid archive smoke query; table names may drift between public releases",
        "euclid",
        "SELECT TOP 10 object_id, right_ascension, declination "
        "FROM catalogue.mer_catalogue",
    ),
}


# --------------------------------------------------------------------------- #
# Atlas targets (curated visual contact sheet / poster anchors)
# --------------------------------------------------------------------------- #
ATLAS_TARGETS = [
    {"name": "M87", "ra": 187.7059, "dec": 12.3911, "fov": 8.0},
    {"name": "3C 273", "ra": 187.2779, "dec": 2.0524, "fov": 4.0},
    {"name": "Centaurus A", "ra": 201.3651, "dec": -43.0191, "fov": 14.0},
    {"name": "Bullet Cluster", "ra": 104.6583, "dec": -55.9475, "fov": 10.0},
    {"name": "CDFS", "ra": 53.1250, "dec": -28.1000, "fov": 12.0},
    {"name": "Sombrero Galaxy", "ra": 189.9976, "dec": -11.6231, "fov": 10.0},
]


# --------------------------------------------------------------------------- #
# Runbooks (data-driven, multi-step desk-work packets)
# --------------------------------------------------------------------------- #
# A runbook is a list of step dicts interpreted by packets.run_runbook. Each step
# has a "kind" and kind-specific keys. Adding a runbook = adding data here, no code.
@dataclass(frozen=True)
class Runbook:
    description: str
    steps: list


RUNBOOKS = {
    "euclid-q1": Runbook(
        "Euclid Q1 desk-work packet: papers, deep-field anchor, samples, visuals",
        [
            {"kind": "papers", "phrase": "Euclid Quick Data Release",
             "query": "year:2025-2026"},
            {"kind": "papers", "phrase": "Euclid Q1 AGN", "query": "year:2025-2026"},
            {"kind": "field", "ra": 53.1250, "dec": -28.1000, "fov": 10.0,
             "note": "CDFS / Euclid deep-field visual anchor"},
            {"kind": "sample", "recipe": "gaia-bright-nearby"},
            {"kind": "atlas"},
            {"kind": "poster", "target": "M87", "fov": 8.0},
            {"kind": "poster", "target": "3C 273", "fov": 4.0},
        ],
    ),
}
