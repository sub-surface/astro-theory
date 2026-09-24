"""Hermetic tests for the kernel spine: artifacts, ledger, capabilities, runs.

No network. Every test gets its own ledger and artifact tree, which is the
point of the design — capabilities receive a `ctx`, so nothing reaches into
process-global state the way the old surfaces did.
"""
import json

import numpy as np
import pytest
from astropy.table import Table

from celestrium import paths
from celestrium.core import capability as capmod
from celestrium.core import events as ev
from celestrium.core.artifact import Artifact, artifact_id, canonical_text, jsonable
from celestrium.core.capability import Param, ParamError, capability
from celestrium.core.kernel import CapabilityError, Kernel
from celestrium.core.ledger import Ledger


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "EXPORTS_DIR", tmp_path / "exports")
    return Kernel(Ledger(tmp_path / "ledger.db"))


# --------------------------------------------------------------------------- #
# content addressing
# --------------------------------------------------------------------------- #
def test_artifact_id_is_stable_and_order_independent():
    a = artifact_id("x.y", {"b": 2, "a": 1})
    b = artifact_id("x.y", {"a": 1, "b": 2})
    assert a == b
    assert a != artifact_id("x.y", {"a": 1, "b": 3})
    assert a != artifact_id("x.z", {"a": 1, "b": 2})


def test_floats_hash_exactly_not_by_display():
    """0.1 must never drift to 0.09999999999999999 between sessions."""
    assert canonical_text(0.1) == repr(0.1)
    assert artifact_id("c", {"v": 0.1}) == artifact_id("c", {"v": 0.1})
    assert artifact_id("c", {"v": 0.1}) != artifact_id("c", {"v": 0.100000001})


def test_canonical_handles_numpy_and_nan():
    assert canonical_text(np.float64(2.5)) == repr(2.5)
    assert canonical_text(float("nan")) == "nan"
    assert jsonable({"a": np.int64(3), "b": np.array([1, 2])}) == {"a": 3, "b": [1, 2]}


def test_inputs_change_the_address():
    assert artifact_id("c", {}, ("aaa",)) != artifact_id("c", {}, ("bbb",))


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #
def test_ledger_round_trip_and_prefix_lookup(tmp_path):
    store = Ledger(tmp_path / "l.db")
    art = Artifact(id="abcdef1234", kind="data", cap="t.c", params={"a": 1})
    store.put(art)
    assert store.get("abcdef1234").cap == "t.c"
    assert store.find("abcd").id == "abcdef1234"
    assert store.find("zzzz") is None
    assert store.count() == 1


def test_ledger_records_lineage_as_a_graph(tmp_path):
    store = Ledger(tmp_path / "l.db")
    store.put(Artifact(id="a1", kind="table", cap="pull", params={}))
    store.put(Artifact(id="b2", kind="table", cap="filter", params={}, inputs=("a1",)))
    store.put(Artifact(id="c3", kind="data", cap="fit", params={}, inputs=("b2",)))

    assert [a.id for a in store.ancestors("c3")] == ["b2", "a1"]
    assert [a.id for a in store.chain("c3")] == ["a1", "b2", "c3"]
    assert {a.id for a in store.descendants("a1")} == {"b2", "c3"}


def test_chain_is_topologically_ordered_on_a_diamond(tmp_path):
    """Every artifact must appear after everything it was derived from —
    otherwise the generated methods paragraph reads as a bag of ids."""
    store = Ledger(tmp_path / "l.db")
    store.put(Artifact(id="root", kind="table", cap="pull", params={}))
    store.put(Artifact(id="left", kind="table", cap="a", params={}, inputs=("root",)))
    store.put(Artifact(id="right", kind="table", cap="b", params={}, inputs=("root",)))
    store.put(Artifact(id="join", kind="data", cap="c", params={},
                       inputs=("left", "right")))

    order = [a.id for a in store.chain("join")]
    assert len(order) == 4 and order[0] == "root" and order[-1] == "join"
    for artifact in store.chain("join"):
        for parent in artifact.inputs:
            assert order.index(parent) < order.index(artifact.id)


def test_preregistration_is_written_once_and_never_replaced(tmp_path):
    store = Ledger(tmp_path / "l.db")
    store.put_study("s", {"v": 1}, prereg="FIRST")
    store.put_study("s", {"v": 2}, prereg="SECOND")
    assert store.get_study("s")["prereg"] == "FIRST"
    assert store.get_study("s")["spec"] == {"v": 2}


# --------------------------------------------------------------------------- #
# capability declaration
# --------------------------------------------------------------------------- #
def test_param_binding_fills_defaults_and_coerces():
    cap = capmod.Capability(name="t", kind="data", fn=lambda ctx: None,
                            params={"n": Param("int", 5), "s": Param("str")})
    assert cap.bind({"s": "x"}) == {"n": 5, "s": "x"}
    assert cap.bind({"s": "x", "n": "7"})["n"] == 7


def test_unknown_parameter_is_an_error_not_a_silent_noop():
    """A typo'd parameter that changes nothing is the worst outcome for a
    content-addressed cache — it would silently return the wrong artifact."""
    cap = capmod.Capability(name="t", kind="data", fn=lambda ctx: None,
                            params={"n": Param("int", 5)})
    with pytest.raises(ParamError, match="unknown parameter"):
        cap.bind({"nn": 3})


def test_missing_required_parameter_is_an_error():
    cap = capmod.Capability(name="t", kind="data", fn=lambda ctx: None,
                            params={"s": Param("str")})
    with pytest.raises(ParamError, match="required"):
        cap.bind({})


def test_enum_rejects_unknown_choice():
    cap = capmod.Capability(name="t", kind="data", fn=lambda ctx: None,
                            params={"k": Param("enum", "a", ("a", "b"))})
    with pytest.raises(ParamError):
        cap.bind({"k": "c"})


def test_every_registered_capability_is_well_formed():
    capmod.load_all()
    assert len(capmod.CAPS) > 20
    for cap in capmod.all_caps():
        assert cap.kind in ("table", "image", "figure", "spectrum", "document",
                            "data", "surface"), cap.name
        assert cap.wing in capmod.WINGS, cap.name
        assert cap.cost in capmod.COSTS, cap.name
        assert cap.summary, f"{cap.name} has no summary"
        assert "." in cap.name, cap.name


# --------------------------------------------------------------------------- #
# the kernel
# --------------------------------------------------------------------------- #
@capability(name="test.double", kind="data", params={"n": Param("int", 1)},
            summary="double a number")
def _double(ctx, n):
    ctx.progress(f"doubling {n}")
    return ctx.data({"result": n * 2}, label=f"{n}→{n * 2}")


@capability(name="test.rows", kind="table", params={"n": Param("int", 3)},
            summary="a table of n rows")
def _rows(ctx, n):
    return ctx.table(Table({"i": np.arange(n), "x": np.arange(n) * 1.5}))


@capability(name="test.boom", kind="data", params={}, summary="always fails")
def _boom(ctx):
    raise RuntimeError("nope")


@capability(name="test.chain", kind="data", params={"n": Param("int", 2)},
            summary="calls another capability")
def _chain(ctx, n):
    child = ctx.child("test.double", n=n)
    return ctx.data({"from_child": ctx.load(child.id)["result"]})


def test_run_produces_a_persisted_artifact(kernel):
    art = kernel.run("test.double", {"n": 21})
    assert art.kind == "data" and art.cap == "test.double"
    assert kernel.load(art.id) == {"result": 42}
    assert art.exists and art.meta["exec_ms"] >= 0
    assert kernel.ledger.get(art.id) is not None


def test_identical_run_is_a_cache_hit(kernel):
    seen = []
    first = kernel.run("test.double", {"n": 3})
    second = kernel.run("test.double", {"n": 3}, emit=seen.append)
    assert first.id == second.id
    assert any(e.type == ev.DONE and e.cached for e in seen)


def test_refresh_bypasses_the_cache_but_keeps_the_address(kernel):
    first = kernel.run("test.double", {"n": 4})
    again = kernel.run("test.double", {"n": 4}, refresh=True)
    assert first.id == again.id            # same recipe, same address


def test_table_payload_records_shape(kernel):
    art = kernel.run("test.rows", {"n": 5})
    assert art.meta["nrows"] == 5 and art.meta["ncols"] == 2
    assert len(kernel.load(art.id)) == 5


def test_failure_raises_and_is_recorded(kernel):
    with pytest.raises(CapabilityError):
        kernel.run("test.boom", {})
    runs = kernel.ledger.runs()
    assert runs and runs[0]["status"] == "failed"


def test_child_calls_become_lineage_but_not_identity(kernel):
    art = kernel.run("test.chain", {"n": 5})
    assert kernel.load(art.id) == {"from_child": 10}
    stored = kernel.ledger.get(art.id)
    assert len(stored.inputs) == 1                      # the child edge
    # ...and the hashed inputs stay empty, so the address is reproducible.
    assert stored.meta["_inputs"] == []
    assert artifact_id("test.chain", {"n": 5}, ()) == art.id


def test_estimate_reports_what_is_already_cached(kernel):
    kernel.run("test.double", {"n": 1})
    est = kernel.estimate([("test.double", {"n": 1}), ("test.double", {"n": 2})])
    assert est == {"total": 2, "cached": 1, "to_run": 1, "cost_mix": {"free": 2}}


def test_methods_paragraph_is_generated_from_lineage(kernel):
    art = kernel.run("test.chain", {"n": 3})
    text = kernel.methods(art.id)
    assert "test.chain" in text and "test.double" in text and art.id in text


def test_artifact_params_become_inputs_automatically(kernel):
    table = kernel.run("test.rows", {"n": 4})
    filtered = kernel.run("table.filter", {"table": table.id[:6], "expr": "i > 1"})
    assert filtered.inputs == (table.id,)               # prefix resolved to full id
    assert len(kernel.load(filtered.id)) == 2


# --------------------------------------------------------------------------- #
# table capabilities
# --------------------------------------------------------------------------- #
def test_filter_expression_rejects_attribute_escapes(kernel):
    table = kernel.run("test.rows", {"n": 4})
    for hostile in ("i.__class__", "__import__('os')", "[c for c in i]"):
        with pytest.raises(CapabilityError):
            kernel.run("table.filter", {"table": table.id, "expr": hostile})


def test_filter_expression_allows_column_maths(kernel):
    table = kernel.run("test.rows", {"n": 6})
    out = kernel.run("table.filter", {"table": table.id, "expr": "sqrt(x) > 1.5"})
    assert 0 < len(kernel.load(out.id)) < 6


def test_table_stats_summarises_numeric_columns(kernel):
    table = kernel.run("test.rows", {"n": 4})
    stats = kernel.load(kernel.run("table.stats", {"table": table.id}).id)
    assert stats["rows"] == 4
    assert stats["columns"]["i"]["max"] == 3.0


# --------------------------------------------------------------------------- #
# the analysis wing — the real end-to-end check
# --------------------------------------------------------------------------- #
def test_injected_dipole_is_recovered(kernel):
    """Inject a known dipole, run the pipeline, get it back."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    sources = kernel.run("analysis.synthetic_sky",
                         {"nsources": 10000, "amplitude": 0.08, "seed": 7})
    density = kernel.run("analysis.sky_density",
                         {"table": sources.id, "nside": 8, "frame": "galactic"})
    fit = kernel.load(kernel.run("analysis.dipole_fit", {"density": density.id}).id)

    assert fit["amplitude"] == pytest.approx(0.08, abs=0.04)
    assert fit["n_sources"] == 10000

    recovered = SkyCoord(fit["direction"]["l"] * u.deg, fit["direction"]["b"] * u.deg,
                         frame="galactic")
    truth = SkyCoord(264.021 * u.deg, 48.253 * u.deg, frame="galactic")
    sigma_angle = np.rad2deg(fit["sigma_amplitude"] / fit["amplitude"])
    assert recovered.separation(truth).deg < 5 * sigma_angle


def test_reported_uncertainty_scales_as_one_over_root_n(kernel):
    """σ_D ∝ 1/√N — verify scaling."""
    sigmas = []
    for count in (5000, 20000):
        sources = kernel.run("analysis.synthetic_sky",
                             {"nsources": count, "amplitude": 0.05, "seed": 7})
        density = kernel.run("analysis.sky_density",
                             {"table": sources.id, "nside": 8, "frame": "galactic"})
        fit = kernel.load(kernel.run("analysis.dipole_fit",
                                     {"density": density.id}).id)
        sigmas.append(fit["sigma_amplitude"])
    assert sigmas[1] / sigmas[0] == pytest.approx(0.5, rel=0.3)


def test_no_injected_dipole_stays_near_shot_noise(kernel):
    sources = kernel.run("analysis.synthetic_sky",
                         {"nsources": 5000, "amplitude": 0.0, "seed": 3})
    density = kernel.run("analysis.sky_density", {"table": sources.id, "nside": 8})
    fit = kernel.load(kernel.run("analysis.dipole_fit", {"density": density.id}).id)
    assert fit["amplitude"] < 20 * fit["shot_noise_amplitude"]


def test_empty_pixels_are_not_fitted_as_real_zeros(kernel):
    """Regression: a footprint-limited catalogue must not manufacture a dipole."""
    common = {"nsources": 10000, "sky_fraction": 0.046, "gal_lat_min": 25.0,
              "seed": 1}
    recovered = []
    for injected in (0.0, 0.08):
        sources = kernel.run("analysis.synthetic_sky",
                             {**common, "amplitude": injected})
        density = kernel.run("analysis.sky_density",
                             {"table": sources.id, "nside": 16, "frame": "galactic"})
        assert density.meta["sky_fraction"] < 0.15
        fit = kernel.load(kernel.run("analysis.dipole_fit",
                                     {"density": density.id}).id)
        recovered.append(fit["amplitude"])

    assert recovered[0] < 0.06
    assert recovered[1] == pytest.approx(0.08, abs=0.04)

    # ...and the old behaviour is still reachable, loudly, for all-sky data.
    naive = kernel.run("analysis.sky_density",
                       {"table": kernel.run("analysis.synthetic_sky",
                                            {**common, "amplitude": 0.0}).id,
                        "nside": 16, "frame": "galactic", "empty": "observed"})
    bogus = kernel.load(kernel.run("analysis.dipole_fit", {"density": naive.id}).id)
    assert bogus["amplitude"] > 0.05


def test_mask_reduces_sky_fraction_and_survives_the_fit(kernel):
    sources = kernel.run("analysis.synthetic_sky",
                         {"nsources": 5000, "amplitude": 0.05, "seed": 5})
    density = kernel.run("analysis.sky_density", {"table": sources.id, "nside": 8})
    masked = kernel.run("analysis.mask", {"density": density.id, "gal_lat_min": 30.0})
    assert masked.meta["sky_fraction"] < 0.55
    fit = kernel.load(kernel.run("analysis.dipole_fit", {"density": masked.id}).id)
    assert fit["sky_fraction"] < 0.55 and fit["amplitude"] > 0


def test_null_shuffle_gives_a_high_p_value_for_noise(kernel):
    sources = kernel.run("analysis.synthetic_sky",
                         {"nsources": 3000, "amplitude": 0.0, "seed": 11})
    density = kernel.run("analysis.sky_density", {"table": sources.id, "nside": 8})
    null = kernel.load(kernel.run("analysis.null_shuffle",
                                  {"density": density.id, "n": 10}).id)
    assert null["p_value"] > 0.01


def test_kinematic_dipole_matches_ellis_baldwin(kernel):
    got = kernel.load(kernel.run("analysis.kinematic_dipole", {}).id)
    assert got["amplitude"] == pytest.approx(0.006137, abs=1e-5)
    assert got["beta"] == pytest.approx(369.82 / 299792.458, rel=1e-9)


def test_compare_reports_ratio_and_tension(kernel):
    sources = kernel.run("analysis.synthetic_sky",
                         {"nsources": 10000, "amplitude": 0.0167, "seed": 2})
    density = kernel.run("analysis.sky_density", {"table": sources.id, "nside": 8})
    measured = kernel.run("analysis.dipole_fit", {"density": density.id})
    expected = kernel.run("analysis.kinematic_dipole", {})
    out = kernel.load(kernel.run("analysis.compare",
                                 {"measured": measured.id,
                                  "expected": expected.id}).id)
    assert "ratio" in out


# --------------------------------------------------------------------------- #
# studies
# --------------------------------------------------------------------------- #
def test_grid_expands_to_every_combination():
    from celestrium.study import Study
    spec = Study(id="s", grid={"a": [1, 2], "b": [10, 20, 30]})
    combos = spec.combos()
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos


def test_prereg_hash_changes_when_the_analysis_changes():
    from celestrium.study import P, Ref, Study, step
    base = Study(id="s", pipeline=(step("test.double", n=P("n")),), grid={"n": [1, 2]})
    same = Study(id="s", pipeline=(step("test.double", n=P("n")),), grid={"n": [1, 2]})
    moved = Study(id="s", pipeline=(step("test.double", n=P("n")),), grid={"n": [1, 3]})
    assert base.prereg_hash() == same.prereg_hash()
    assert base.prereg_hash() != moved.prereg_hash()


def test_unbound_free_parameters_are_detected():
    from celestrium.study import P, Study, step
    spec = Study(id="s", pipeline=(step("test.double", n=P("missing")),))
    assert spec.unbound() == ("missing",)


def test_leverage_score_prefers_high_leverage_per_row():
    from celestrium.study import Study
    from celestrium.study.model import leverage_score
    cheap = Study(id="a", leverage="high", rows="~50000")
    dear = Study(id="b", leverage="high", rows="~500000000")
    assert leverage_score(cheap) > leverage_score(dear)


def test_study_run_produces_a_surface_over_the_grid(kernel, monkeypatch):
    from celestrium.study import P, Ref, Study, library, step
    spec = Study(id="demo", title="demo",
                 pipeline=(step("test.double", n=P("n"), into="fit"),),
                 grid={"n": [1, 2, 3]}, metric="result")
    monkeypatch.setitem(library.STUDIES, "demo", spec)

    art = kernel.run("study.run", {"study": "demo"})
    surface = kernel.load(art.id)
    assert art.kind == "surface"
    assert len(surface) == 3
    assert list(surface["result"]) == [2.0, 4.0, 6.0]
    assert art.meta["succeeded"] == 3
    assert kernel.ledger.get_study("demo")["prereg"] == spec.prereg_hash()


def test_one_failing_combination_does_not_kill_the_study(kernel, monkeypatch):
    from celestrium.study import P, Study, library, step
    spec = Study(id="halfbad", pipeline=(step("test.double", n=P("n")),),
                 grid={"n": [1, "not-an-int"]}, metric="result")
    monkeypatch.setitem(library.STUDIES, "halfbad", spec)
    art = kernel.run("study.run", {"study": "halfbad"})
    surface = kernel.load(art.id)
    assert art.meta["succeeded"] == 1
    assert any("failed" in str(s) for s in surface["status"])


def test_study_without_a_pipeline_refuses_to_run(kernel):
    with pytest.raises(CapabilityError, match="claim, not a run"):
        kernel.run("study.run", {"study": "wide-binaries-gravity-test"})


def test_builtin_studies_are_internally_consistent():
    from celestrium.study import library
    for spec in library.STUDIES.values():
        assert not spec.unbound(), f"{spec.id} has unbound params"
        for entry in spec.pipeline:
            capmod.load_all()
            assert entry.cap in capmod.CAPS, f"{spec.id} → unknown cap {entry.cap}"


def test_euclid_forecast_runs_offline_end_to_end(kernel, monkeypatch):
    """The active line, in miniature: the DR1 forecast needs no network."""
    from celestrium.study import library
    from dataclasses import replace
    spec = library.get("euclid-dr1-dipole-forecast")
    small = replace(spec, grid={**spec.grid, "nsources": [50000],
                                "injected": [0.01], "seed": [1]})
    monkeypatch.setitem(library.STUDIES, spec.id, small)

    art = kernel.run("study.run", {"study": spec.id})
    surface = kernel.load(art.id)
    assert len(surface) == 1
    assert float(surface["amplitude"][0]) > 0
    assert float(surface["sigma_amplitude"][0]) > 0


