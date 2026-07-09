"""Wave-1 parity commands: `fetch`, `feed`, `export` (hermetic, no network).

These pin the CLI side of the "one brain, two surfaces" repair: the CLI drives
the same planner/products/feed/table-ref machinery the TUI uses.
"""
import json

from astropy.table import Table
from typer.testing import CliRunner

from celestrium import hub, planner, products, registry

runner = CliRunner()


def _fake_target(**kw):
    base = dict(display_name="Demo", aliases=(), ra=10.0, dec=-5.0, otype="G",
                object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    base.update(kw)
    return planner.ResolvedTarget(**base)


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def test_fetch_runs_named_product_and_prints_table(monkeypatch):
    target = _fake_target()
    monkeypatch.setattr(planner, "resolve_target", lambda text: target)

    seen = {}

    def fake_execute(tgt, plan, settings=None, emit=None):
        seen["key"] = plan.product.key
        seen["settings"] = settings
        return products.ProductResult(
            "table", plan.product.label, tgt.display_name,
            table=Table({"a": [1, 2]}), provenance={"archive": "x"})

    monkeypatch.setattr(products, "execute_product", fake_execute)
    result = runner.invoke(hub.app, ["fetch", "Demo", "--product", "vizier"])
    assert result.exit_code == 0
    assert seen["key"] == "vizier"
    assert seen["settings"]["fov"] == "auto"
    assert "2 rows" in result.output


def test_fetch_defaults_to_top_executable_plan(monkeypatch):
    target = _fake_target()
    monkeypatch.setattr(planner, "resolve_target", lambda text: target)
    seen = {}

    def fake_execute(tgt, plan, settings=None, emit=None):
        seen["key"] = plan.product.key
        return products.ProductResult(
            "image", "Demo survey", tgt.display_name, path="demo.jpg", fov=8.0)

    monkeypatch.setattr(products, "execute_product", fake_execute)
    result = runner.invoke(hub.app, ["fetch", "Demo"])
    assert result.exit_code == 0
    # top-ranked executable product for a galaxy is the colour image
    assert seen["key"] == "colour-image"
    assert "demo.jpg" in result.output


def test_fetch_unknown_product_lists_fetchable_keys(monkeypatch):
    monkeypatch.setattr(planner, "resolve_target", lambda text: _fake_target())
    result = runner.invoke(hub.app, ["fetch", "Demo", "--product", "nope"])
    assert result.exit_code == 1
    assert "colour-image" in result.output


def test_fetch_unresolved_target_fails(monkeypatch):
    monkeypatch.setattr(planner, "resolve_target", lambda text: None)
    result = runner.invoke(hub.app, ["fetch", "Nothing"])
    assert result.exit_code == 1


def test_fetch_json_payload_serialises_table(monkeypatch):
    target = _fake_target()
    monkeypatch.setattr(planner, "resolve_target", lambda text: target)
    monkeypatch.setattr(
        products, "execute_product",
        lambda tgt, plan, settings=None, emit=None: products.ProductResult(
            "table", plan.product.label, tgt.display_name,
            table=Table({"a": [1]}), provenance={"archive": "x"}))
    result = runner.invoke(hub.app, ["--json", "fetch", "Demo", "--product", "vizier"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "table"
    assert payload["nrows"] == 1
    assert payload["target"]["display_name"] == "Demo"


# --------------------------------------------------------------------------- #
# feed
# --------------------------------------------------------------------------- #
def test_feed_runs_executor_through_cache(monkeypatch):
    from celestrium import cache, neos
    monkeypatch.setattr(neos, "fetch_close_approaches",
                        lambda: Table({"des": ["2026 AB"], "dist": ["0.01"]}))
    seen = {}

    def fake_cached_query(archive, query, fetch, refresh=False, **kw):
        seen.update(archive=archive, query=query, refresh=refresh)
        return fetch()

    monkeypatch.setattr(cache, "cached_query", fake_cached_query)
    result = runner.invoke(hub.app, ["feed", "neos", "--refresh"])
    assert result.exit_code == 0
    assert seen["archive"] == "global-neo"      # alias resolved, TUI-shared tag
    assert "global-feed:neo" in seen["query"]
    assert seen["refresh"] is True
    assert "1 rows" in result.output


def test_feed_transient_is_executable_since_wave_4(monkeypatch):
    """transient flipped from planned to executable once Wave 4 wired a real
    ALeRCE-backed transients.py — the CLI must actually run and cache it now,
    exactly like neo/satellite (mirrors test_feed_runs_executor_through_cache)."""
    from celestrium import cache, transients

    monkeypatch.setattr(transients, "fetch_latest_transients",
                        lambda: Table({"main_id": ["ZTF25abc"], "ra": [10.0]}))
    seen = {}

    def fake_cached_query(archive, query, fetch, refresh=False, **kw):
        seen.update(archive=archive, query=query)
        return fetch()

    monkeypatch.setattr(cache, "cached_query", fake_cached_query)
    result = runner.invoke(hub.app, ["--json", "feed", "transient"])
    assert result.exit_code == 0
    assert seen["archive"] == "global-transient"
    payload = json.loads(result.output)
    assert payload["nrows"] == 1


def test_feed_gates_on_planner_status_generically(monkeypatch):
    """The parity fix (feed must not run a 'planned' capability) is generic,
    not transient-specific — verified by temporarily marking a feed planned."""
    from celestrium import planner
    caps = list(planner.SOURCE_CAPABILITIES)
    idx = next(i for i, c in enumerate(caps) if c.key == "neo")
    from dataclasses import replace
    caps[idx] = replace(caps[idx], status="planned")
    monkeypatch.setattr(planner, "SOURCE_CAPABILITIES", tuple(caps))

    called = {"n": 0}
    from celestrium import neos
    monkeypatch.setattr(neos, "fetch_close_approaches",
                        lambda: called.__setitem__("n", called["n"] + 1))
    result = runner.invoke(hub.app, ["--json", "feed", "neo"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert called["n"] == 0


def test_feed_unknown_key_lists_supported():
    result = runner.invoke(hub.app, ["feed", "quasars"])
    assert result.exit_code == 1
    for key in registry.GLOBAL_FEED_EXECUTORS:
        assert key in result.output


def test_feed_aliases_shared_between_surfaces():
    # registry owns the aliases; the TUI re-exports them
    assert registry.feed_key("tles") == "satellite"
    assert registry.feed_key("alerts") == "transient"


# --------------------------------------------------------------------------- #
# export + table refs
# --------------------------------------------------------------------------- #
def test_export_candidate_list_to_csv(tmp_path, monkeypatch):
    from celestrium import candidates, paths, tables
    monkeypatch.setattr(candidates, "CAND_DIR", tmp_path)
    monkeypatch.setattr(candidates, "INDEX", tmp_path / "index.jsonl")
    monkeypatch.setattr(candidates, "_REPO", tmp_path)
    monkeypatch.setattr(paths, "EXPORTS_DIR", tmp_path / "exports")
    candidates.save("agn", Table({"ra": [1.0], "dec": [2.0]}), origin="test")

    tab, note = tables.load_table_ref("agn")
    assert len(tab) == 1 and "candidate list" in note

    result = runner.invoke(hub.app, ["export", "agn"])
    assert result.exit_code == 0
    out = tmp_path / "exports" / "agn.csv"
    assert out.exists()
    assert "ra,dec" in out.read_text(encoding="utf-8").splitlines()[0]


def test_export_unknown_ref_fails_helpfully():
    result = runner.invoke(hub.app, ["export", "no-such-ref-xyz"])
    assert result.exit_code == 1
    assert "no-such-ref-xyz" in result.output


def test_load_table_ref_resolves_recipe_through_cache(monkeypatch):
    from celestrium import cache, tables

    class FakeArchive:
        @staticmethod
        def query(adql):
            return Table({"x": [1, 2, 3]})

    monkeypatch.setitem(registry.SAMPLE_RECIPES, "demo-ref",
                        registry.SampleRecipe("Demo", "gaia", "SELECT 1"))
    monkeypatch.setattr(cache, "cached_query",
                        lambda archive, query, fetch, refresh=False, **kw: fetch())
    tab, note = tables.load_table_ref("demo-ref", {"gaia": FakeArchive})
    assert len(tab) == 3
    assert "recipe" in note
