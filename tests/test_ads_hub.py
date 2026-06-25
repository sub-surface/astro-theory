from typer.testing import CliRunner
from astropy.table import Table

from access import ads, hub


def test_add_to_refs_appends_only_new_bibtex_entries(monkeypatch, tmp_path):
    refs = tmp_path / "refs.bib"
    refs.write_text("@article{oldkey,\n  title = {Old}\n}\n", encoding="utf-8")

    monkeypatch.setattr(
        ads,
        "bibtex",
        lambda bibcodes: (
            "@article{oldkey,\n  title = {Old}\n}\n\n"
            "@article{newkey,\n  title = {New}\n}\n"
        ),
    )

    added = ads.add_to_refs(["old", "new"], path=refs)

    text = refs.read_text(encoding="utf-8")
    assert added == 1
    assert text.count("@article{oldkey,") == 1
    assert "@article{newkey," in text


def test_cite_reports_add_to_refs_failures(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        ads,
        "search",
        lambda query, rows: [{"bibcode": "abc", "year": 2026, "title": ["Title"]}],
    )

    def fail_export(bibcodes):
        raise RuntimeError("export failed")

    monkeypatch.setattr(ads, "add_to_refs", fail_export)

    result = runner.invoke(hub.app, ["cite", "arxiv:1234.5678", "--add"])

    assert result.exit_code == 1
    assert "export failed" in result.output


def test_cite_joins_multi_token_query(monkeypatch):
    runner = CliRunner()
    calls = []

    def fake_search(query, rows):
        calls.append((query, rows))
        return [{"bibcode": "abc", "year": 2026, "title": ["Title"]}]

    monkeypatch.setattr(ads, "search", fake_search)

    result = runner.invoke(
        hub.app,
        ["cite", "abs:Euclid", "Quick", "Data", "Release", "year:2025-2026", "--rows", "8"],
    )

    assert result.exit_code == 0
    assert calls == [("abs:Euclid Quick Data Release year:2025-2026", 8)]
    assert "abc" in result.output


def test_image_accepts_fov_option(monkeypatch):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(hub.cutouts, "smart", lambda ra, dec, fov_arcmin=None: calls.append((ra, dec, fov_arcmin)))

    result = runner.invoke(hub.app, ["image", "1.5", "-2.5", "--fov", "7.0"])

    assert result.exit_code == 0
    assert calls == [(1.5, -2.5, 7.0)]


def test_log_prints_manifest_records(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        hub.cache,
        "manifest",
        lambda: [{
            "utc": "2026-06-25T05:00:00+00:00",
            "archive": "gaia",
            "hash": "abc123",
            "nrows": 5,
            "cache_file": "data/cache/gaia_abc123.ecsv",
            "query": "SELECT TOP 5 source_id FROM gaiadr3.gaia_source",
        }],
    )

    result = runner.invoke(hub.app, ["log"])

    assert result.exit_code == 0
    assert "gaia" in result.output
    assert "abc123" in result.output
    assert "SELECT TOP 5" in result.output


def test_query_uses_supported_archive_and_cache(monkeypatch):
    runner = CliRunner()
    fetched = []
    cached = []

    class FakeArchive:
        @staticmethod
        def query(adql):
            fetched.append(adql)
            return Table({"source_id": [1], "ra": [2.0]})

    monkeypatch.setitem(hub.QUERY_ARCHIVES, "demo", FakeArchive)

    def fake_cached_query(archive, query, fetch, refresh=False):
        cached.append((archive, query, refresh))
        return fetch()

    monkeypatch.setattr(hub.cache, "cached_query", fake_cached_query)

    result = runner.invoke(hub.app, ["query", "demo", "SELECT 1"])

    assert result.exit_code == 0
    assert fetched == ["SELECT 1"]
    assert cached == [("demo", "SELECT 1", False)]
    assert "source_id" in result.output
    assert "1" in result.output


def test_papers_writes_report_and_uses_phrase(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)
    calls = []

    def fake_search(query, rows):
        calls.append((query, rows))
        return [{"bibcode": "abc", "year": 2026, "title": ["Euclid Q1"]}]

    monkeypatch.setattr(ads, "search", fake_search)

    result = runner.invoke(
        hub.app,
        ["papers", "year:2025-2026", "--phrase", "Euclid Quick Data Release", "--rows", "3", "--report"],
    )

    assert result.exit_code == 0
    assert calls == [('abs:"Euclid Quick Data Release" year:2025-2026', 3)]
    reports = list(tmp_path.glob("papers-*.md"))
    assert len(reports) == 1
    assert "abc" in reports[0].read_text(encoding="utf-8")


def test_dossier_writes_report_and_calls_imaging(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        hub.resolvers,
        "identify",
        lambda name: Table({"main_id": ["M  87"], "otype": ["G"], "ra": [187.7], "dec": [12.4]}),
    )
    monkeypatch.setattr(hub.cutouts, "color", lambda *args, **kwargs: tmp_path / "color.jpg")
    monkeypatch.setattr(hub.cutouts, "panel", lambda *args, **kwargs: tmp_path / "panel.png")
    monkeypatch.setattr(hub.cutouts, "best_color_hips", lambda dec: ("hips-id", "Deep Survey"))
    monkeypatch.setattr(ads, "search", lambda query, rows: [{"bibcode": "abc", "year": 2026, "title": ["Paper"]}])

    result = runner.invoke(hub.app, ["dossier", "M87", "--rows", "1"])

    assert result.exit_code == 0
    reports = list(tmp_path.glob("dossier-*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "M  87" in text
    assert "color.jpg" in text
    assert "abc" in text


def test_field_writes_report(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(hub.cutouts, "identify_field", lambda ra, dec: ("Obj", "G", 4.2))
    monkeypatch.setattr(hub.cutouts, "best_color_hips", lambda dec: ("hips-id", "Deep Survey"))
    monkeypatch.setattr(hub.cutouts, "color", lambda *args, **kwargs: tmp_path / "field.jpg")
    monkeypatch.setattr(hub.cutouts, "panel", lambda *args, **kwargs: tmp_path / "field.png")

    result = runner.invoke(hub.app, ["field", "1.5", "-2.5", "--fov", "6"])

    assert result.exit_code == 0
    reports = list(tmp_path.glob("field-*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "Obj" in text
    assert "Deep Survey" in text


def test_sample_runs_named_recipe_through_cache(monkeypatch):
    runner = CliRunner()
    cached = []

    class FakeArchive:
        @staticmethod
        def query(adql):
            return Table({"id": [1]})

    monkeypatch.setitem(hub.QUERY_ARCHIVES, "gaia", FakeArchive)
    monkeypatch.setitem(
        hub.SAMPLE_RECIPES,
        "demo",
        hub.SampleRecipe("Demo recipe", "gaia", "SELECT TOP 1 id FROM demo"),
    )

    def fake_cached_query(archive, query, fetch, refresh=False):
        cached.append((archive, query, refresh))
        return fetch()

    monkeypatch.setattr(hub.cache, "cached_query", fake_cached_query)

    result = runner.invoke(hub.app, ["sample", "demo"])

    assert result.exit_code == 0
    assert cached == [("sample:demo", "SELECT TOP 1 id FROM demo", False)]
    assert "Demo recipe" in result.output


def test_atlas_targets_makes_contact_sheet(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(hub, "ATLAS_DIR", tmp_path)
    monkeypatch.setattr(hub.cutouts, "color", lambda *args, **kwargs: tmp_path / f"{kwargs['out'].stem}.jpg")
    monkeypatch.setattr(hub, "_contact_sheet", lambda paths, out, title: out.write_text(title, encoding="utf-8") or out)

    result = runner.invoke(hub.app, ["atlas-targets", "--limit", "2"])

    assert result.exit_code == 0
    assert (tmp_path / "atlas-targets.png").read_text(encoding="utf-8") == "atlas targets"


def test_poster_calls_cutout_poster(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(hub, "POSTERS_DIR", tmp_path)
    monkeypatch.setattr(
        hub.resolvers,
        "identify",
        lambda name: Table({"main_id": ["M  87"], "otype": ["G"], "ra": [187.7], "dec": [12.4]}),
    )
    calls = []

    def fake_poster(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "poster.jpg"

    monkeypatch.setattr(hub.cutouts, "poster", fake_poster)

    result = runner.invoke(hub.app, ["poster", "M87", "--resolution", "1080p", "--style", "label"])

    assert result.exit_code == 0
    assert calls[0][1]["width"] == 1920
    assert calls[0][1]["height"] == 1080
    assert calls[0][1]["label"] == "M  87"
