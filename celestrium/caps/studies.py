"""Studies as capabilities — a grid of runs that produces one answer.

`study.run` is itself a capability, which is what makes the whole thing hold
together: the result surface is a normal artifact, its per-combination runs are
its lineage children, and re-running the summary reuses the cached science.
"""
from __future__ import annotations

import numpy as np

from ..core.capability import Param, capability
from ..study import P, Ref, dig


def _resolve_value(value, combo: dict, named: dict, prev):
    """Fill a step parameter: grid ref → value, step ref → artifact id,
    templated string → formatted with the combination."""
    if isinstance(value, P):
        if value.name not in combo:
            raise KeyError(f"grid has no value for free parameter {value.name!r}")
        return combo[value.name]
    if isinstance(value, Ref):
        if value.step in ("prev", ""):
            if prev is None:
                raise ValueError("Ref('prev') used before any step has run")
            return prev
        if value.step not in named:
            raise KeyError(f"no step named {value.step!r} has run yet")
        return named[value.step]
    if isinstance(value, str) and "{" in value and combo:
        try:
            return value.format(**combo)
        except (KeyError, IndexError):
            return value
    return value


def _column(values: list):
    """Build a column that survives mixed/missing values (failed combinations)."""
    cleaned = [v for v in values if v is not None]
    if cleaned and all(isinstance(v, (int, float, np.integer, np.floating))
                       and not isinstance(v, bool) for v in cleaned):
        return np.array([float(v) if v is not None else np.nan for v in values],
                        dtype="float64")
    return np.array(["" if v is None else str(v) for v in values], dtype=object)


@capability(
    name="study.run", kind="surface", wing="validation", cost="heavy",
    params={"study": Param("str", help="study id"),
            "grid": Param("json", None, help="override the study's grid"),
            "limit": Param("int", 0, help="cap combinations (0 = all)")},
    summary="Run a study's pipeline across its parameter grid → a result surface.",
)
def study_run(ctx, study, grid, limit):
    from astropy.table import Table
    from ..study import library

    spec = library.get(study)
    if not spec.pipeline:
        raise ValueError(f"study {study!r} has no pipeline yet "
                         f"(status: {spec.status}) — it is a claim, not a run")
    missing = spec.unbound(grid)
    if missing:
        raise ValueError(f"study {study!r} has unbound parameters: "
                         f"{', '.join(missing)}")

    combos = spec.combos(grid)
    if limit > 0:
        dropped = len(combos) - limit
        combos = combos[:limit]
        if dropped > 0:
            ctx.note(f"limit={limit}: {dropped} combinations not run "
                     "(this surface is a sample, not the full grid)")

    # Preregistration: record the analysis decisions the first time, and say so
    # loudly if they changed once results already existed.
    prereg = spec.prereg_hash()
    stored = ctx.ledger.get_study(study)
    if stored is None:
        ctx.ledger.put_study(study, spec.to_dict(), prereg=prereg)
        ctx.note(f"preregistered analysis {prereg}")
    else:
        ctx.ledger.put_study(study, spec.to_dict(), prereg=prereg)
        if stored.get("prereg") and stored["prereg"] != prereg:
            ctx.note(f"⚠ analysis changed since preregistration "
                     f"({stored['prereg']} → {prereg}) — results after this "
                     "point are exploratory, not confirmatory")

    estimate = ctx.kernel.estimate(spec.calls(grid))
    ctx.progress(f"{len(combos)} combinations · {estimate['cached']} already cached")

    metric_names = (spec.metric, *spec.metrics)
    records = []
    for index, combo in enumerate(combos, start=1):
        ctx.check_cancel()
        label = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        ctx.progress(f"[{index}/{len(combos)}] {label}")
        row = dict(combo)
        prev, named = None, {}
        try:
            for entry in spec.pipeline:
                params = {key: _resolve_value(value, combo, named, prev)
                          for key, value in entry.params.items()}
                artifact = ctx.child(entry.cap, **params)
                prev = artifact.id
                if entry.into:
                    named[entry.into] = artifact.id
            final = ctx.artifact(prev)
            payload = ctx.load(prev) if final and final.kind == "data" else {}
            for name in metric_names:
                row[name] = dig(payload, name, dig(final.meta if final else {}, name))
            row["artifact"] = prev
            row["status"] = "ok"
        except Exception as exc:                       # one combination ≠ the study
            ctx.note(f"{label}: {type(exc).__name__}: {exc}")
            for name in metric_names:
                row.setdefault(name, None)
            row["artifact"] = prev or ""
            row["status"] = f"failed: {type(exc).__name__}"
        records.append(row)

    columns = list(combos[0].keys()) if combos else []
    columns += [*metric_names, "artifact", "status"]
    surface = Table({name: _column([record.get(name) for record in records])
                     for name in columns})

    ok = sum(1 for record in records if record["status"] == "ok")
    values = [record.get(spec.metric) for record in records
              if record["status"] == "ok" and record.get(spec.metric) is not None]
    spread = {}
    if values:
        array = np.asarray(values, dtype="float64")
        spread = {"metric_min": float(array.min()), "metric_max": float(array.max()),
                  "metric_median": float(np.median(array)),
                  "metric_spread": float(array.max() - array.min())}
        ctx.progress(f"{spec.metric}: {array.min():.5g} … {array.max():.5g} "
                     f"across {ok} runs")
    return ctx.surface(surface, label=f"study {study} ({ok}/{len(records)} ok)",
                       study_id=study, metric=spec.metric, combinations=len(records),
                       succeeded=ok, prereg=prereg, title=spec.title, **spread)


@capability(
    name="study.rank", kind="table", wing="theory", cost="free",
    params={},
    summary="Rank studies by leverage per row — how a desk competes with a telescope.",
)
def study_rank(ctx):
    from astropy.table import Table
    from ..study import library
    from ..study.model import leverage_score

    rows = sorted(library.STUDIES.values(), key=leverage_score, reverse=True)
    table = Table({
        "study": np.array([s.id for s in rows], dtype=object),
        "status": np.array([s.status for s in rows], dtype=object),
        "leverage": np.array([s.leverage or "—" for s in rows], dtype=object),
        "rows": np.array([s.rows or "—" for s in rows], dtype=object),
        "score": np.array([leverage_score(s) for s in rows], dtype="float64"),
        "runnable": np.array(["yes" if s.pipeline else "no" for s in rows],
                             dtype=object),
    })
    return ctx.table(table, label=f"{len(rows)} studies by leverage/row")


@capability(
    name="study.estimate", kind="data", wing="validation", cost="free",
    params={"study": Param("str"), "grid": Param("json", None)},
    summary="What a study would cost before you commit to running it.",
)
def study_estimate(ctx, study, grid):
    from ..study import library
    spec = library.get(study)
    if not spec.pipeline:
        return ctx.data({"study": study, "runnable": False, "status": spec.status},
                        label=f"{study}: no pipeline")
    calls = spec.calls(grid)
    estimate = ctx.kernel.estimate(calls)
    combos = spec.combos(grid)
    payload = {"study": study, "runnable": True, "combinations": len(combos),
               "steps": len(spec.pipeline), "declared_rows": spec.rows, **estimate}
    ctx.progress(f"{len(combos)} combinations · {estimate['to_run']} to run · "
                 f"{estimate['cached']} cached")
    return ctx.data(payload, label=f"{study}: {estimate['to_run']} runs to do")
