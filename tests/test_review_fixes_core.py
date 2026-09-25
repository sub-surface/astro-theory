"""Regression tests for review fixes in the kernel, ledger, resolvers,
tabular filter, cutouts and CLI. Hermetic: no network, temp ledger/artifacts."""
import math

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from celestrium import paths, resolvers
from celestrium.caps.tabular import _safe_eval
from celestrium.core import events as ev
from celestrium.core.artifact import Artifact, jsonable
from celestrium.core.capability import Param, capability
from celestrium.core.kernel import CapabilityError, Kernel
from celestrium.core.ledger import Ledger


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    return Kernel(Ledger(tmp_path / "ledger.db"))


@capability(name="rfix.rows", kind="table", params={"n": Param("int", 3)},
            summary="a table of n rows")
def _rows(ctx, n):
    return ctx.table(Table({"i": np.arange(n), "x": np.arange(n) * 1.5}))


@capability(name="rfix.leaf", kind="data", params={"n": Param("int", 1)},
            summary="leaf")
def _leaf(ctx, n):
    return ctx.data({"n": n})


@capability(name="rfix.parent", kind="data", params={}, summary="calls a child")
def _parent(ctx):
    ctx.child("rfix.leaf", n=7)
    return ctx.data({"ok": 1})


@capability(name="rfix.boom", kind="data", params={}, summary="always fails")
def _boom(ctx):
    raise ValueError("kaboom")


# ----- kernel ------------------------------------------------------------- #
def test_repro_keeps_the_address_of_artifact_param_runs(kernel):
    table = kernel.run("rfix.rows", {"n": 5})
    filtered = kernel.run("table.filter", {"table": table.id, "expr": "i > 1"})
    assert filtered.meta["_inputs"] == [table.id]
    # repro hands _inputs back alongside the artifact param: must not double up
    aid, _, inputs = kernel.resolve_id("table.filter", filtered.params,
                                       tuple(filtered.meta["_inputs"]))
    assert aid == filtered.id and inputs == (table.id,)
    assert kernel.repro(filtered.id, refresh=False).id == filtered.id


def test_child_events_reach_on_event_once(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    seen, caller = [], []
    k = Kernel(Ledger(tmp_path / "l.db"), on_event=seen.append)
    k.run("rfix.parent", {}, emit=caller.append)
    for events in (seen, caller):
        child_done = [e for e in events if e.cap == "rfix.leaf" and e.type == ev.DONE]
        assert len(child_done) == 1


def test_failed_run_records_the_real_traceback(kernel):
    with pytest.raises(CapabilityError):
        kernel.run("rfix.boom", {})
    error = [r for r in kernel.ledger.runs() if r["cap"] == "rfix.boom"][0]["error"]
    assert "kaboom" in error and "NoneType: None" not in error


# ----- ledger ------------------------------------------------------------- #
def test_ambiguous_prefix_resolves_to_nothing(tmp_path):
    store = Ledger(tmp_path / "l.db")
    for aid in ("abc111", "abc222"):
        store.put(Artifact(id=aid, kind="data", cap="t.c", params={}))
    assert store.find("abc") is None
    assert store.find("abc1").id == "abc111"
    assert store.forget("abc") is False
    assert store.count() == 2


def test_prefix_lookup_treats_like_wildcards_literally(tmp_path):
    store = Ledger(tmp_path / "l.db")
    store.put(Artifact(id="abc111", kind="data", cap="t.c", params={}))
    assert store.find("a_c") is None
    assert store.find("%1") is None
    assert store.find("abc").id == "abc111"


def test_nan_is_stored_as_null():
    assert jsonable(float("nan")) is None
    assert jsonable({"ra": np.float64("nan")}) == {"ra": None}
    assert jsonable(float("inf")) == "inf"


# ----- resolvers ---------------------------------------------------------- #
def test_bibliography_escapes_quotes(monkeypatch):
    seen = []
    monkeypatch.setattr(resolvers.Simbad, "query_tap", lambda q: seen.append(q))
    resolvers.bibliography("Barnard's Star")
    assert "WHERE id = 'Barnard''s Star'" in seen[0]
    assert resolvers.adql_string("x' OR '1'='1") == "'x'' OR ''1''=''1'"


def _none(_text):
    return None


def test_target_without_finite_position_is_unresolved():
    ambiguous = [{"main_id": "Io", "otype": "Solar System",
                  "ra": float("nan"), "dec": float("nan")}]
    assert resolvers.resolve_target("Io", identify_fn=_none,
                                    dynamic_fn=lambda t: ambiguous,
                                    search_fn=_none) is None


def test_sun_resolves_through_horizons_with_a_real_position():
    sun = [{"main_id": "Sun", "otype": "Solar System", "ra": 183.2, "dec": -1.4}]
    found = resolvers.resolve_target("Sun", identify_fn=_none,
                                     dynamic_fn=lambda t: sun, search_fn=_none)
    assert found is not None
    assert math.isfinite(found.ra) and math.isfinite(found.dec)


# ----- tabular ------------------------------------------------------------ #
def test_masked_values_never_pass_a_filter(tmp_path):
    t = Table({"p": MaskedColumn([25.0, 5.0, 99.0], mask=[False, False, True])})
    t.write(tmp_path / "m.ecsv")
    t = Table.read(tmp_path / "m.ecsv")
    assert list(np.flatnonzero(_safe_eval("p > 20", t))) == [0]
    assert list(np.flatnonzero(_safe_eval("p < 30", t))) == [0, 1]
    assert list(np.flatnonzero(_safe_eval("~(p > 20)", t))) == [1]


def test_scalar_filter_expression_is_a_value_error():
    t = Table({"p": [1.0, 2.0]})
    with pytest.raises(ValueError):
        _safe_eval("1 > 0", t)


# ----- cutouts / CLI ------------------------------------------------------ #
def test_zero_fov_does_not_divide_by_zero():
    from celestrium import cutouts
    assert cutouts.fetch_direct_cutout("CDS/P/DESI-Legacy-Surveys/DR10/color",
                                       10.0, 20.0, 0.0) is None
    with pytest.raises(ValueError):
        cutouts.poster(10.0, 20.0, fov_arcmin=0.0)


def test_poster_resolution_maps_to_width_height():
    from celestrium import cli
    assert cli._poster_size("1080p") == (1920, 1080)
    assert cli._poster_size("800x600") == (800, 600)
    with pytest.raises(ValueError):
        cli._poster_size("huge")
