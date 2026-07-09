"""Fixes for defects Codex's Wave-3 stress-test review surfaced.

See docs/devlogs/2026-07-09-codex-review.md for the original findings:
Windows cp1252 console crashes, the transient-feed parity gap (CLI executed
a planned stub the TUI already knew to gate), `python -m celestrium.tui
--help` launching the app, an unguarded optional `dl` import, and plot/sweep
worker sequences not tied into clear/resolve invalidation.
"""
import asyncio
import io
import subprocess
import sys

import pytest

from celestrium import hub


# --------------------------------------------------------------------------- #
# UTF-8 console reconfiguration (Windows cp1252 crash fix)
# --------------------------------------------------------------------------- #
def test_ensure_utf8_console_reconfigures_real_streams(monkeypatch):
    calls = []

    class FakeStream:
        def reconfigure(self, encoding=None, errors=None):
            calls.append((encoding, errors))

    monkeypatch.setattr(hub.sys, "stdout", FakeStream())
    monkeypatch.setattr(hub.sys, "stderr", FakeStream())
    hub._ensure_utf8_console()
    assert calls == [("utf-8", "replace"), ("utf-8", "replace")]


def test_ensure_utf8_console_is_a_safe_noop_without_reconfigure(monkeypatch):
    # Typer's CliRunner-captured streams (and plain io.StringIO) have no
    # .reconfigure — the helper must swallow that, not raise.
    monkeypatch.setattr(hub.sys, "stdout", io.StringIO())
    monkeypatch.setattr(hub.sys, "stderr", io.StringIO())
    hub._ensure_utf8_console()  # must not raise


# --------------------------------------------------------------------------- #
# datalab_desi optional dependency (deferred ImportError)
# --------------------------------------------------------------------------- #
def test_datalab_desi_imports_without_optional_dl_package(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "dl", None)  # simulate 'dl' not installed
    sys.modules.pop("celestrium.datalab_desi", None)
    import celestrium.datalab_desi as dd
    assert dd.qc is None


def test_datalab_desi_raises_actionable_error_only_at_call_time(monkeypatch):
    from celestrium import datalab_desi as dd
    monkeypatch.setattr(dd, "qc", None)
    with pytest.raises(ImportError, match="astro-datalab"):
        dd.query("SELECT 1")
    with pytest.raises(ImportError, match="astro-datalab"):
        dd.schema()


# --------------------------------------------------------------------------- #
# TUI: plot/sweep sequences tied into clear + resolve invalidation
# --------------------------------------------------------------------------- #
def test_clear_invalidates_in_flight_plot_and_sweep():
    from celestrium.tui.app import CelestriumApp

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            plot_seq = app._bump("plot")
            sweep_seq = app._bump("sweep")
            app.action_clear()
            assert not app._current("plot", plot_seq)
            assert not app._current("sweep", sweep_seq)

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# `python -m celestrium.tui --help` must behave like a normal entry point
# --------------------------------------------------------------------------- #
def test_tui_module_help_exits_without_launching_the_app():
    result = subprocess.run(
        [sys.executable, "-m", "celestrium.tui", "--help"],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0
    assert "python -m celestrium tui" in result.stdout


def test_resolve_invalidates_in_flight_sweep_for_previous_target():
    from celestrium.tui.app import CelestriumApp

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            stale_sweep_seq = app._bump("sweep")
            app.do_resolve("NGC 1275")
            await pilot.pause()
            assert not app._current("sweep", stale_sweep_seq)

    asyncio.run(scenario())
