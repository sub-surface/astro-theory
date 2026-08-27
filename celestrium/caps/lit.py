"""The theory wing — literature in, effort maps out."""
from __future__ import annotations

from ..core.capability import Param, capability


@capability(
    name="lit.papers", kind="data", wing="theory", cost="network",
    params={"query": Param("str", help="ADS/SciX query"), "rows": Param("int", 10)},
    summary="Literature search via ADS/SciX.",
)
def lit_papers(ctx, query, rows):
    from .. import ads
    ctx.progress(f"ADS: {query}")
    docs = ads.search(query, rows=rows)
    payload = {"query": query, "docs": docs}
    docs = payload.get("docs", payload.get("papers", []))
    return ctx.data(payload, label=f"papers: {query}", nrows=len(docs), query=query)


@capability(
    name="lit.cite", kind="data", wing="theory", cost="network",
    params={"bibcodes": Param("str", help="comma-separated bibcodes")},
    summary="Append BibTeX for these bibcodes to refs.bib (deduped).",
)
def lit_cite(ctx, bibcodes):
    from .. import ads
    codes = [c.strip() for c in bibcodes.replace(" ", ",").split(",") if c.strip()]
    if not codes:
        raise ValueError("lit.cite needs at least one bibcode")
    added = ads.add_to_refs(codes)
    ctx.note(f"{added} new entries in refs.bib")
    return ctx.data({"bibcodes": codes, "added": added},
                    label=f"cite {len(codes)} → +{added}")


@capability(
    name="lit.census", kind="table", wing="theory", cost="network",
    params={"topics": Param("json", None, help="topic → arXiv phrase; null = registry")},
    summary="Field-effort census: where a field actually spends its effort.",
)
def lit_census(ctx, topics):
    from .. import census
    ctx.progress("tallying arXiv topic volumes")
    table = census.tally(topics or None)
    return ctx.table(table, label="field-effort census")


@capability(
    name="lit.bibliography", kind="data", wing="theory", cost="network",
    params={"target": Param("str"), "rows": Param("int", 10)},
    summary="SIMBAD per-object bibliography.",
    tags=("target",),
)
def lit_bibliography(ctx, target, rows):
    from .. import resolvers
    docs = resolvers.bibliography(target, limit=rows)
    listed = list(docs) if docs is not None else []
    return ctx.data({"target": target, "docs": [dict(d) if hasattr(d, "keys") else str(d)
                                                for d in listed]},
                    label=f"bibliography {target}", nrows=len(listed))
