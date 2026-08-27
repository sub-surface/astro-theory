"""Hermetic tests for Celestrium CLI (hub.py) and ADS/literature integration.

Tests CLI presenter commands using CliRunner with monkeypatched kernels and
mocked network responses to keep the entire test suite deterministic and offline.
"""
from pathlib import Path
import json
import pytest
from typer.testing import CliRunner
from astropy.table import Table

from celestrium import hub, ads, config, cutouts, resolvers
from celestrium.core.kernel import Kernel
from celestrium.core.artifact import Artifact

runner = CliRunner()


def test_add_to_refs_appends_only_new_bibtex_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(ads, "bibtex", lambda codes: "@article{2024Test,\n  author = {Test, A.},\n  title = {Paper One}\n}\n\n@article{2025Test,\n  author = {Test, B.},\n  title = {Paper Two}\n}")
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text("@article{2024Test,\n  author = {Test, A.},\n  title = {Paper One}\n}\n", encoding="utf-8")
    
    added = ads.add_to_refs(["2024Test", "2025Test"], path=bib_file)
    assert added == 1
    content = bib_file.read_text(encoding="utf-8")
    assert "2025Test" in content


def test_cite_reports_add_to_refs_failures(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        if cap_name == "lit.papers":
            return Artifact(id="p1", kind="data", cap="lit.papers")
        if cap_name == "lit.cite":
            raise RuntimeError("ADS token expired")
        return Artifact(id="a", kind="data", cap=cap_name)

    def fake_load(self, art_id):
        return {"query": "test", "docs": [{"bibcode": "2026Test", "year": 2026, "title": ["Test"]}]}

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    result = runner.invoke(hub.app, ["cite", "dipole", "--add"])
    assert result.exit_code != 0


def test_cite_joins_multi_token_query(monkeypatch):
    seen_query = {}

    def fake_run(self, cap_name, params=None, **kwargs):
        seen_query["query"] = params.get("query")
        return Artifact(id="p1", kind="data", cap="lit.papers")

    def fake_load(self, art_id):
        return {"query": seen_query.get("query"), "docs": [{"bibcode": "2026Test", "year": 2026, "title": ["Cosmic Dipole"]}]}

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    result = runner.invoke(hub.app, ["cite", "cosmic", "dipole", "2026"])
    assert result.exit_code == 0
    assert seen_query["query"] == "cosmic dipole 2026"


def test_where_reports_coverage_note(monkeypatch):
    monkeypatch.setattr(cutouts, "identify_field", lambda ra, dec: ("M87", "Galaxy", 0.5))
    monkeypatch.setattr(cutouts, "best_color_hips", lambda dec: ("CDS/P/DESI", "DESI Legacy"))
    monkeypatch.setattr(resolvers, "image_coverage_note", lambda dec: "Deep Legacy coverage")

    result = runner.invoke(hub.app, ["where", "187.7", "12.3"])
    assert result.exit_code == 0
    assert "DESI Legacy" in result.output
    assert "Deep Legacy coverage" in result.output


def test_plan_resolves_and_ranks_products(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="p1", kind="data", cap="object.products")

    def fake_load(self, art_id):
        return {
            "target": {"display_name": "M87", "otype": "Galaxy", "object_class": "galaxy_agn", "ra": 187.7, "dec": 12.3},
            "products": [{"capability": "imaging.cutout", "kind": "image", "cost": "network", "summary": "Colour image"}]
        }

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    result = runner.invoke(hub.app, ["plan", "M87"])
    assert result.exit_code == 0
    assert "M87" in result.output
    assert "imaging.cutout" in result.output


def test_plan_unresolvable_target_exits_nonzero(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        raise LookupError("nothing resolves for NonExistentTarget")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["plan", "NonExistentTarget"])
    assert result.exit_code != 0


def test_spectrum_renders_via_ned(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="spec1", kind="spectrum", cap="object.spectrum", path="data/artifacts/spec1.png", label="spectrum M87")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["spectrum", "M87"])
    assert result.exit_code == 0
    assert "spec1.png" in result.output


def test_spectrum_missing_exits_nonzero(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        raise LookupError("no NED spectrum for 'StarXYZ'")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["spectrum", "StarXYZ"])
    assert result.exit_code != 0


def test_image_accepts_fov_option(monkeypatch):
    seen = {}

    def fake_run(self, cap_name, params=None, **kwargs):
        seen.update(params)
        return Artifact(id="img1", kind="image", cap="imaging.cutout", path="data/artifacts/img1.png", label="cutout")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["image", "187.7", "12.3", "--fov", "10.5"])
    assert result.exit_code == 0
    assert seen["fov"] == 10.5
    assert seen["target"] == "187.7 12.3"


def test_log_prints_manifest_records(monkeypatch):
    class FakeArtifact:
        id = "art12345"
        cap = "archive.query"
        kind = "table"
        label = "gaia query"
        created = "2026-08-27T12:00:00"

    class FakeLedger:
        def search(self, limit=20):
            return [FakeArtifact()]

    class FakeKernel:
        ledger = FakeLedger()
        def load(self, id):
            return Table({"a": [1]})

    monkeypatch.setattr(hub, "_kernel", lambda: FakeKernel())

    result = runner.invoke(hub.app, ["log"])
    assert result.exit_code == 0
    assert "art12345" in result.output


def test_query_uses_supported_archive_and_cache(monkeypatch):
    seen = {}

    def fake_run(self, cap_name, params=None, **kwargs):
        seen.update(params)
        return Artifact(id="q1", kind="table", cap="archive.query")

    def fake_load(self, art_id):
        return Table({"source_id": [100, 200], "ra": [10.0, 20.0], "dec": [0.0, 5.0]})

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    result = runner.invoke(hub.app, ["query", "gaia", "SELECT TOP 2 source_id FROM gaiadr3.gaia_source"])
    assert result.exit_code == 0
    assert seen["archive"] == "gaia"
    assert "2 rows" in result.output


def test_papers_writes_report_and_uses_phrase(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)
    seen = {}

    def fake_run(self, cap_name, params=None, **kwargs):
        seen.update(params)
        return Artifact(id="p1", kind="data", cap="lit.papers")

    def fake_load(self, art_id):
        return {"query": seen.get("query"), "docs": [{"bibcode": "2026Test", "year": 2026, "title": ["Euclid Q1 Discovery"]}]}

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    result = runner.invoke(hub.app, ["papers", "Euclid", "--phrase", "Cosmic Dipole", "--report"])
    assert result.exit_code == 0
    assert 'abs:"Cosmic Dipole"' in seen["query"]
    reports = list(tmp_path.glob("papers-*.md"))
    assert len(reports) == 1


def test_dossier_writes_report_and_calls_imaging(tmp_path, monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="doss1", kind="document", cap="object.dossier", path=str(tmp_path / "dossier-m87.md"), label="dossier M87")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["dossier", "M87"])
    assert result.exit_code == 0
    assert "dossier" in result.output


def test_field_writes_report(tmp_path, monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="field1", kind="document", cap="object.field", path=str(tmp_path / "field.md"), label="field 187.7 12.3")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["field", "187.7", "12.3"])
    assert result.exit_code == 0
    assert "field packet" in result.output


def test_field_keeps_panel_when_colour_fails(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="field1", kind="document", cap="object.field", path="field.md", label="field")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["field", "10.0", "-5.0", "--fov", "8.0"])
    assert result.exit_code == 0


def test_sample_runs_named_recipe_through_cache(monkeypatch):
    seen = {}

    def fake_run(self, cap_name, params=None, **kwargs):
        seen.update(params)
        return Artifact(id="s1", kind="table", cap="archive.sample")

    def fake_load(self, art_id):
        return Table({"source_id": [1, 2], "ra": [10.0, 11.0], "dec": [0.0, 1.0]})

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    result = runner.invoke(hub.app, ["sample", "gaia-bright-nearby"])
    assert result.exit_code == 0
    assert seen["recipe"] == "gaia-bright-nearby"
    assert "2 rows" in result.output


def test_atlas_targets_makes_contact_sheet(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="atlas1", kind="figure", cap="imaging.atlas", path="atlas.png", label="atlas sheet")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["atlas-targets", "--limit", "3"])
    assert result.exit_code == 0
    assert "atlas ->" in result.output


def test_poster_calls_cutout_poster(monkeypatch):
    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="post1", kind="image", cap="imaging.poster", path="poster.jpg", label="poster M87")

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["poster", "M87", "--resolution", "1080p", "--style", "clean"])
    assert result.exit_code == 0
    assert "poster ->" in result.output


def test_runbook_euclid_q1_writes_index(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)

    def fake_run(self, cap_name, params=None, **kwargs):
        return Artifact(id="step1", kind="data", cap=cap_name)

    monkeypatch.setattr(Kernel, "run", fake_run)

    result = runner.invoke(hub.app, ["runbook", "euclid-q1"])
    assert result.exit_code == 0
    assert "runbook ->" in result.output
    index_file = tmp_path / "runbook-euclid-q1.md"
    assert index_file.exists()
