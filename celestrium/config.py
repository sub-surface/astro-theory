"""Tiny local instrument config — theme persistence today, one future home
for prefs/credentials (a `~/.celestrium/config.toml`-style need flagged in
the 2026-07 feature review) once more than one setting needs to survive a
restart. Deliberately minimal: a flat JSON dict under data/config.json
(already git-ignored alongside cache/manifest/candidates).
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import paths

CONFIG_PATH = paths.DATA / "config.json"


def load() -> dict[str, Any]:
    """Read the config dict, or {} if it doesn't exist yet or is corrupt."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(updates: dict[str, Any]) -> None:
    """Merge `updates` into the on-disk config and write it back."""
    cfg = load()
    cfg.update(updates)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set(key: str, value: Any) -> None:
    save({key: value})


# --------------------------------------------------------------------------- #
# Static Instrument Data Tables
# --------------------------------------------------------------------------- #
class _LazyAdaptor:
    """Expose a uniform `.query(adql)` over a module whose real signature differs."""

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
    adql: bool
    source: object
    health_url: str = ""


ARCHIVES = {
    "gaia": Archive("gaia", "ESA Gaia DR3 (TAP/ADQL)", True, "celestrium.gaia",
                    "https://gea.esac.esa.int/tap-server/tap/availability?archive=gaia"),
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

QUERY_ARCHIVES = {k: a.source for k, a in ARCHIVES.items()}


def resolve_query_source(key: str, table: Optional[dict] = None):
    src = (table or QUERY_ARCHIVES).get(key.lower())
    if src is None:
        return None
    if isinstance(src, str):
        return importlib.import_module(src)
    return src


GLOBAL_FEED_EXECUTORS = {
    "neo": ("celestrium.neos", "fetch_close_approaches"),
    "satellite": ("celestrium.satellites", "fetch_visible_satellites"),
    "transient": ("celestrium.transients", "fetch_latest_transients"),
}

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
    key = text.strip().lower()
    return FEED_ALIASES.get(key, key)


CENSUS_TOPICS = {
    "wide binary gravity": 'all:"wide binary" AND (all:MOND OR all:gravity)',
    "MOND / Milgromian": "all:MOND OR all:Milgromian",
    "Dyson sphere": 'all:"Dyson sphere"',
    "technosignature": "all:technosignature OR all:technosignatures",
    "Hubble tension": 'all:"Hubble tension"',
    "S8 / sigma8 tension": 'all:"S8 tension" OR all:"sigma8 tension"',
    "primordial black hole": 'all:"primordial black hole"',
    "reheating (inflation)": "all:reheating AND all:inflation",
    "preheating": "all:preheating",
    "oscillon": "all:oscillon OR all:oscillons",
    "inflationary grav. waves": 'all:"inflationary gravitational waves"',
    "JWST high-z galaxies": 'all:JWST AND all:"high redshift" AND all:galaxies',
    "early dark energy": 'all:"early dark energy"',
    "21 cm cosmology": 'all:"21 cm" AND all:cosmology',
    "fast radio burst": 'all:"fast radio burst"',
}


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


ATLAS_TARGETS = [
    {"name": "M87", "ra": 187.7059, "dec": 12.3911, "fov": 8.0},
    {"name": "3C 273", "ra": 187.2779, "dec": 2.0524, "fov": 4.0},
    {"name": "Centaurus A", "ra": 201.3651, "dec": -43.0191, "fov": 14.0},
    {"name": "Bullet Cluster", "ra": 104.6583, "dec": -55.9475, "fov": 10.0},
    {"name": "CDFS", "ra": 53.1250, "dec": -28.1000, "fov": 12.0},
    {"name": "Sombrero Galaxy", "ra": 189.9976, "dec": -11.6231, "fov": 10.0},
]


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

