"""Tests for the Phase 0/1 service layer: registry, packets, match, actionable log, --json."""
import json

from astropy.table import Table
from typer.testing import CliRunner

from access import hub, packets, registry


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_completes_archive_set():
    keys = set(registry.QUERY_ARCHIVES)
    assert {"gaia", "irsa", "euclid", "heasarc", "desi", "casda", "vizier-tap"} <= keys


def test_resolve_query_source_handles_known_and_unknown():
    assert registry.resolve_query_source("does-not-exist") is None
    src = registry.resolve_query_source("gaia")
    assert hasattr(src, "query")


def test_hub_reexports_share_registry_objects():
    # The test suite patches these on `hub`; they must be the same objects.
    assert hub.QUERY_ARCHIVES is registry.QUERY_ARCHIVES
    assert hub.SAMPLE_RECIPES is registry.SAMPLE_RECIPES


# --------------------------------------------------------------------------- #
# packets dataclasses
# --------------------------------------------------------------------------- #
def test_paper_set_round_trips():
    ps = packets.PaperSet("q", [{"bibcode": "b1", "year": 2026, "title": ["T"]}])
    assert ps.to_dict()["n"] == 1
    assert "b1" in ps.to_markdown()


def test_object_packet_markdown_and_dict():
    op = packets.ObjectPacket("M  87", "G", 1.0, 2.0, "Survey", 3.0, redshift=0.004)
    assert "# Dossier: M  87" in op.to_markdown()
    assert "Redshift (NED): 0.004" in op.to_markdown()
    assert op.to_dict()["name"] == "M  87"


def test_field_packet_nearest_text():
    fp = packets.FieldPacket(1.0, 2.0, 5.0, "Survey", ("Obj", "G", 4.2))
    assert "Obj (G), 4.2 arcsec away" in fp.to_markdown()
    blank = packets.FieldPacket(1.0, 2.0, 5.0, "Survey", None)
    assert "No SIMBAD object" in blank.to_markdown()


# --------------------------------------------------------------------------- #
# match
# --------------------------------------------------------------------------- #
def test_match_recipe_against_catalog(monkeypatch):
    runner = CliRunner()

    class FakeArchive:
        @staticmethod
        def query(adql):
            return Table({"ra": [1.0], "dec": [2.0]})

    monkeypatch.setitem(hub.QUERY_ARCHIVES, "gaia", FakeArchive)
    monkeypatch.setitem(hub.SAMPLE_RECIPES, "demo",
                        hub.SampleRecipe("Demo", "gaia", "SELECT 1"))
    monkeypatch.setattr(hub.cache, "cached_query",
                        lambda archive, query, fetch, refresh=False: fetch())

    import access.xmatch as xm
    seen = {}

    def fake_match(local, cat2, ra, dec, radius_arcsec):
        seen.update(cat2=cat2, radius=radius_arcsec, nrows=len(local))
        return Table({"angDist": [0.5]})

    monkeypatch.setattr(xm, "match", fake_match)

    result = runner.invoke(hub.app, ["match", "demo", "vizier:VIII/65/nvss", "--radius", "7"])

    assert result.exit_code == 0
    assert seen == {"cat2": "vizier:VIII/65/nvss", "radius": 7.0, "nrows": 1}
    assert "angDist" in result.output


# --------------------------------------------------------------------------- #
# actionable log
# --------------------------------------------------------------------------- #
def test_log_open_loads_cached_table(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(hub.cache, "load_cached", lambda h: Table({"colx": [1, 2]}))
    result = runner.invoke(hub.app, ["log", "--open", "abc123"])
    assert result.exit_code == 0
    assert "colx" in result.output


def test_log_rerun_reexecutes_query(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(hub.cache, "find_record",
                        lambda h: {"archive": "gaia", "query": "SELECT 1"})

    class FakeArchive:
        @staticmethod
        def query(adql):
            return Table({"coly": [9]})

    monkeypatch.setitem(hub.QUERY_ARCHIVES, "gaia", FakeArchive)
    seen = {}

    def fake_cached_query(archive, query, fetch, refresh=False):
        seen.update(archive=archive, refresh=refresh)
        return fetch()

    monkeypatch.setattr(hub.cache, "cached_query", fake_cached_query)

    result = runner.invoke(hub.app, ["log", "--rerun", "abc"])
    assert result.exit_code == 0
    assert seen == {"archive": "gaia", "refresh": True}
    assert "coly" in result.output


# --------------------------------------------------------------------------- #
# --json
# --------------------------------------------------------------------------- #
def test_json_resolve_emits_machine_readable(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        hub.resolvers, "identify",
        lambda name: Table({"main_id": ["M  87"], "otype": ["G"], "ra": [187.7], "dec": [12.4]}),
    )
    result = runner.invoke(hub.app, ["--json", "resolve", "M87"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "M  87"
    assert payload["otype"] == "G"
