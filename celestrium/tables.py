"""Shared table-reference resolution — one name for "a table I already have".

A *table-ref* is, in resolution order: a saved candidate-list name, a sample
recipe name (runs through the provenance cache), or a manifest hash prefix
(see `hub log`). Commands on both surfaces (`export`, `plot`, `match`) accept
the same refs, so the lookup lives here rather than in a presenter.
"""
from typing import Optional, Tuple

from astropy.table import Table

from . import cache, candidates, registry

# RA/Dec column-name aliases seen across archives (Gaia, VizieR/CDS, ALeRCE,
# hand-built tables). Shared so crossmatch (tui/app.py), the watch packet's
# per-row cone search, and any future table consumer agree on one list.
RA_ALIASES = ("ra", "RA", "ra_deg", "RA_ICRS", "RAJ2000", "_RAJ2000", "raj2000", "RAdeg")
DEC_ALIASES = ("dec", "DEC", "dec_deg", "DE_ICRS", "DEJ2000", "_DEJ2000", "dej2000", "DEdeg")


def find_coord_columns(tab: Table) -> Tuple[str, str]:
    """Identify RA/Dec columns by name, or raise a helpful KeyError."""
    ra = next((c for c in RA_ALIASES if c in tab.colnames), None)
    dec = next((c for c in DEC_ALIASES if c in tab.colnames), None)
    if not ra or not dec:
        raise KeyError(f"could not identify RA/Dec columns; available: {tab.colnames}")
    return ra, dec


def load_table_ref(ref: str, archives: Optional[dict] = None) -> Tuple[Table, str]:
    """Resolve a table-ref to ``(table, provenance_note)``.

    `archives` overrides the query-source mapping for recipe refs (the CLI
    passes its patched `QUERY_ARCHIVES` so tests that monkeypatch it still win).
    Raises ``KeyError`` when the ref matches nothing.
    """
    rec = candidates.find_record(ref)
    if rec is not None:
        return candidates.load(ref), f"candidate list {ref!r} ({rec.get('origin', '')})"

    recipe = registry.SAMPLE_RECIPES.get(ref)
    if recipe is not None:
        source = registry.resolve_query_source(recipe.archive, archives)
        tab = cache.cached_query(f"sample:{ref}", recipe.adql,
                                 lambda: source.query(recipe.adql))
        return tab, f"sample recipe {ref!r} ({recipe.archive})"

    try:
        return cache.load_cached(ref), f"cached pull {ref}"
    except KeyError:
        raise KeyError(
            f"unknown table ref {ref!r} — not a candidate list, sample recipe, "
            "or manifest hash (see `candidates`, `sample list`, `log`)")
