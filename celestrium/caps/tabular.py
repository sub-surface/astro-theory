"""Table-to-table capabilities — the middle of the validation loop.

Every one takes a `table` parameter of kind ``artifact``, so the kernel folds
the input into that run's lineage automatically. Chaining query → filter →
crossmatch → plot therefore produces a real provenance chain with no
bookkeeping at any call site.
"""
from __future__ import annotations

import ast

import numpy as np

from ..core.capability import Param, capability

TABLE_IN = Param("artifact", help="artifact id (or unambiguous prefix) of a table")

# Column algebra only. Attribute access, subscripting, comprehensions, lambdas,
# imports and walrus are all absent from this list, so `__class__`-style escapes
# and arbitrary calls cannot be expressed — the expression is validated against
# the allowlist *before* it is compiled, not sandboxed after.
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)
_FUNCS = {
    "abs": np.abs, "log10": np.log10, "log": np.log, "exp": np.exp,
    "sqrt": np.sqrt, "isfinite": np.isfinite, "isnan": np.isnan,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "deg2rad": np.deg2rad, "rad2deg": np.rad2deg,
    "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
}


def _safe_eval(expr: str, table) -> np.ndarray:
    """Evaluate a boolean row expression over table columns.

    Validates the parse tree against an allowlist first (see `_ALLOWED_NODES`),
    so only arithmetic/comparison over column names and the whitelisted numpy
    functions can run. Anything else is rejected before compilation.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse filter expression: {exc}") from exc

    names = {name: np.asarray(table[name]) for name in table.colnames}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"filter expression may not use {type(node).__name__}; "
                "only column arithmetic, comparisons and "
                f"{', '.join(sorted(_FUNCS))} are allowed")
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError("only whitelisted functions may be called: "
                             + ", ".join(sorted(_FUNCS)))
        if isinstance(node, ast.Name) and node.id not in names and node.id not in _FUNCS:
            raise ValueError(f"unknown name {node.id!r}; columns are "
                             f"{', '.join(table.colnames[:20])}")

    mask = eval(compile(tree, "<filter>", "eval"),  # noqa: S307 — allowlisted above
                {"__builtins__": {}}, {**names, **_FUNCS})
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape[0] != len(table):
        raise ValueError(f"filter expression must yield one boolean per row; "
                         f"got {mask.dtype} shape {mask.shape}")
    return mask


@capability(
    name="table.filter", kind="table", wing="validation", cost="free",
    params={"table": TABLE_IN, "expr": Param("str", help="e.g. 'w1mpro - w2mpro > 0.8'")},
    summary="Row-filter a table with a numpy expression over its columns.",
)
def table_filter(ctx, table, expr):
    source = ctx.load(table)
    mask = _safe_eval(expr, source)
    kept = source[mask]
    ctx.progress(f"{len(kept)}/{len(source)} rows kept")
    return ctx.table(kept, label=f"filter: {expr}", expr=expr,
                     kept=int(len(kept)), dropped=int(len(source) - len(kept)))


@capability(
    name="table.columns", kind="table", wing="validation", cost="free",
    params={"table": TABLE_IN, "keep": Param("str", help="comma-separated columns")},
    summary="Project a table down to named columns.",
)
def table_columns(ctx, table, keep):
    source = ctx.load(table)
    wanted = [c.strip() for c in keep.split(",") if c.strip()]
    missing = [c for c in wanted if c not in source.colnames]
    if missing:
        raise KeyError(f"no column(s) {', '.join(missing)}; "
                       f"have {', '.join(source.colnames[:20])}")
    return ctx.table(source[wanted], label=f"columns: {', '.join(wanted)}")


@capability(
    name="table.xmatch", kind="table", wing="validation", cost="network",
    params={"table": TABLE_IN, "catalog": Param("str", "vizier:VIII/65/nvss"),
            "radius_arcsec": Param("float", 5.0)},
    summary="CDS X-Match a table against a VizieR catalogue.",
)
def table_xmatch(ctx, table, catalog, radius_arcsec):
    from .. import tables, xmatch
    source = ctx.load(table)
    ra_col, dec_col = tables.find_coord_columns(source)
    ctx.progress(f"X-matching {len(source)} rows against {catalog}")
    matched = xmatch.match(source, cat2=catalog, ra=ra_col, dec=dec_col,
                           radius_arcsec=radius_arcsec)
    return ctx.table(matched, label=f"×{catalog}", catalog=catalog,
                     radius_arcsec=radius_arcsec, input_rows=int(len(source)))


@capability(
    name="table.plot", kind="figure", wing="validation", cost="free",
    params={"table": TABLE_IN, "x": Param("str", ""), "y": Param("str", ""),
            "kind": Param("enum", "scatter", ("scatter", "hist", "sky", "cmd")),
            "title": Param("str", "")},
    summary="Quick-look plot of table columns → PNG.",
)
def table_plot(ctx, table, x, y, kind, title):
    from .. import plots
    source = ctx.load(table)
    path = plots.plot_table(source, x=x or None, y=y or None, kind=kind,
                            title=title, out=ctx.out_path(".png"))
    return ctx.figure(path, label=f"{kind}: {x or 'ra'}/{y or ''}".rstrip("/"),
                      kind=kind, x=x, y=y)


@capability(
    name="table.stats", kind="data", wing="validation", cost="free",
    params={"table": TABLE_IN, "columns": Param("str", "", help="comma-separated; blank = all numeric")},
    summary="Per-column summary statistics (count, mean, sd, min/median/max).",
)
def table_stats(ctx, table, columns):
    source = ctx.load(table)
    wanted = [c.strip() for c in columns.split(",") if c.strip()] or list(source.colnames)
    out = {}
    for name in wanted:
        if name not in source.colnames:
            continue
        try:
            values = np.asarray(source[name], dtype="float64")
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        out[name] = {
            "n": int(finite.size), "missing": int(values.size - finite.size),
            "mean": float(np.mean(finite)), "sd": float(np.std(finite, ddof=1))
            if finite.size > 1 else 0.0,
            "min": float(np.min(finite)), "median": float(np.median(finite)),
            "max": float(np.max(finite)),
        }
    return ctx.data({"rows": int(len(source)), "columns": out},
                    label=f"stats ({len(out)} numeric columns)")


@capability(
    name="table.export", kind="table", wing="validation", cost="free",
    params={"table": TABLE_IN,
            "fmt": Param("enum", "csv", ("csv", "ecsv", "fits", "votable"))},
    summary="Export a table to data/exports/ in a portable format.",
)
def table_export(ctx, table, fmt):
    from .. import paths
    source = ctx.load(table)
    art = ctx.artifact(table)
    paths.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = {"csv": ".csv", "ecsv": ".ecsv", "fits": ".fits", "votable": ".vot"}[fmt]
    writer = {"csv": "csv", "ecsv": "ascii.ecsv", "fits": "fits",
              "votable": "votable"}[fmt]
    out = paths.EXPORTS_DIR / f"{art.id if art else 'table'}{suffix}"
    source.write(out, format=writer, overwrite=True)
    ctx.note(f"exported → {out}")
    return ctx.table(source, label=f"export {fmt}", export_path=out.as_posix(),
                     fmt=fmt)


@capability(
    name="table.save_list", kind="table", wing="validation", cost="free",
    params={"table": TABLE_IN, "name": Param("str"), "note": Param("str", "")},
    summary="Persist a table as a named candidate list.",
)
def table_save_list(ctx, table, name, note):
    from .. import candidates
    source = ctx.load(table)
    art = ctx.artifact(table)
    path = candidates.save(name, source, origin=(art.cap if art else "kernel"),
                           note=note or (art.label if art else ""))
    ctx.note(f"saved candidate list → {path}")
    return ctx.table(source, label=f"list {name}", list_name=name,
                     list_path=str(path))


@capability(
    name="table.load_list", kind="table", wing="validation", cost="free",
    params={"name": Param("str")},
    summary="Load a saved candidate list back into the ledger.",
)
def table_load_list(ctx, name):
    from .. import candidates
    return ctx.table(candidates.load(name), label=f"list {name}", list_name=name)
