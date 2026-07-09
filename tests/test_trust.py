"""Wave-2 trust layer: cache retry/backoff, structured --json errors, doctor.

All hermetic — the pinger is injected, the retry sleep is stubbed, and the
CLI error paths are driven through monkeypatched service calls.
"""
import json

import pytest
from astropy.table import Table
from typer.testing import CliRunner

from celestrium import cache, doctor, hub, planner

runner = CliRunner()


# --------------------------------------------------------------------------- #
# cache retry/backoff
# --------------------------------------------------------------------------- #
def test_cached_query_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(cache, "_REPO", tmp_path)
    naps = []
    monkeypatch.setattr(cache, "_sleep", naps.append)

    calls = {"n": 0}

    def flaky_fetch():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("archive hiccup")
        return Table({"x": [1]})

    tab = cache.cached_query("demo", "SELECT 1", flaky_fetch)
    assert len(tab) == 1
    assert calls["n"] == 3               # two failures + one success
    assert naps == list(cache.RETRY_BACKOFFS)


def test_cached_query_raises_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(cache, "_sleep", lambda s: None)

    def always_fails():
        raise ConnectionError("archive down")

    with pytest.raises(ConnectionError):
        cache.cached_query("demo", "SELECT 2", always_fails)


def test_cached_query_zero_retries_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "MANIFEST", tmp_path / "manifest.jsonl")
    naps = []
    monkeypatch.setattr(cache, "_sleep", naps.append)
    calls = {"n": 0}

    def fails(_=None):
        calls["n"] += 1
        raise ValueError("bad adql")

    with pytest.raises(ValueError):
        cache.cached_query("demo", "SELECT 3", fails, retries=0)
    assert calls["n"] == 1 and naps == []


# --------------------------------------------------------------------------- #
# structured --json errors
# --------------------------------------------------------------------------- #
def test_json_mode_emits_structured_error(monkeypatch):
    monkeypatch.setattr(planner, "resolve_target", lambda text: None)
    result = runner.invoke(hub.app, ["--json", "fetch", "Nothing"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "Error"
    assert "Nothing" in payload["message"]
    assert "hint" in payload


def test_json_mode_error_carries_exception_type(monkeypatch):
    def boom(archive, query, fetch, refresh=False, **kw):
        raise TimeoutError("archive timed out")

    monkeypatch.setattr(hub.cache, "cached_query", boom)
    result = runner.invoke(
        hub.app, ["--json", "query", "gaia", "SELECT 1"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {"error": "TimeoutError", "message": "archive timed out"}


def test_rich_mode_error_is_unchanged(monkeypatch):
    monkeypatch.setattr(planner, "resolve_target", lambda text: None)
    result = runner.invoke(hub.app, ["fetch", "Nothing"])
    assert result.exit_code == 1
    assert "could not resolve" in result.output


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def test_doctor_checks_with_injected_pinger():
    seen = []

    def fake_pinger(url, timeout):
        seen.append(url)
        return ("gaia" in url or "irsa" in url), "HTTP 200"

    report = doctor.run_checks(pinger=fake_pinger)
    assert report["summary"]["archives_checked"] == len(seen) > 0
    by_key = {a["archive"]: a for a in report["archives"]}
    assert by_key["gaia"]["status"] == "up"
    assert by_key["heasarc"]["status"] == "down"
    assert {"cache_files", "manifest_rows", "candidate_lists",
            "ads_token", "cache_mb"} <= set(report["local"])


def test_doctor_pinger_exception_is_a_down_not_a_crash():
    def exploding_pinger(url, timeout):
        raise ConnectionError("no route to host")

    report = doctor.run_checks(pinger=exploding_pinger)
    assert all(a["status"] == "down" for a in report["archives"])
    assert report["summary"]["archives_up"] == 0


def test_doctor_cli_json(monkeypatch):
    monkeypatch.setattr(doctor, "_default_pinger",
                        lambda url, timeout: (True, "HTTP 200"))
    result = runner.invoke(hub.app, ["--json", "doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["archives_up"] == payload["summary"]["archives_checked"]


def test_every_registered_archive_has_a_health_endpoint():
    from celestrium import registry
    for key, archive in registry.ARCHIVES.items():
        assert archive.health_url.startswith("https://"), key
