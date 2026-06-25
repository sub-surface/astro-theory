"""Headless smoke test for the Celestrium TUI (Textual pilot — no TTY needed).

Validates that the app composes, the CSS parses, the wired widgets exist, and a
network-free action (History) runs. Mode workers (resolve/query/literature) hit
the network, so they're out of scope here — this guards the skeleton.
"""
import asyncio

import pytest

pytest.importorskip("textual")
from celestrium.tui.app import CelestriumApp  # noqa: E402


def test_tui_mounts_and_history_runs():
    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            # core widgets are wired
            assert app.query_one("#entry") is not None
            assert app.query_one("#results") is not None
            assert app.query_one("#status") is not None
            # header subtitle carries the active line + ADS status
            assert "test-line" in app.sub_title
            # a network-free action populates without error
            app.mode = "history"
            app.action_history()
            await pilot.pause()
        # clean exit
        assert True

    asyncio.run(scenario())


def test_cockpit_polish_network_free():
    """Theme default, wireframe panel, debug + theme + image-settings controls."""
    from celestrium.tui.app import AboutScreen, ImageSettingsScreen
    from celestrium.tui.wireframe import Wireframe

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            await pilot.pause()
            # ansi-dark is the requested default theme
            assert app.theme == "ansi-dark"
            # the wireframe panel is mounted and spinning (frame advances on its timer)
            wf = app.query_one("#orrery", Wireframe)
            f0 = wf.frame
            await asyncio.sleep(0.25)
            assert wf.frame > f0
            # F2 cycles the solid; its render is Braille
            s0 = wf.solid_name
            app.action_next_solid()
            assert wf.solid_name != s0
            assert any(0x2800 <= ord(c) <= 0x28ff for c in wf.render().plain)
            # debug + theme-cycle controls
            app.action_toggle_debug()
            assert app.debug_mode is True
            app.action_cycle_theme()
            assert app.theme != "ansi-dark"
            # image-settings modal applies FOV/pixels
            app.action_image_settings()
            await pilot.pause()
            assert isinstance(app.screen, ImageSettingsScreen)
            app.screen.query_one("#img-fov").value = "30"
            app.screen.query_one("#img-ok").press()
            await pilot.pause()
            assert app.image_cfg["fov"] == 30.0
            # About card opens and closes
            app.action_about()
            await pilot.pause()
            assert isinstance(app.screen, AboutScreen)
            app.screen.query_one("#about-close").press()
            await pilot.pause()
            # refresh with no prior action is a safe no-op
            app.action_refresh()

    asyncio.run(scenario())


def test_easter_egg_is_network_free():
    """A magic word in Resolve mode resolves to an egg, never hitting SIMBAD."""
    from textual.widgets import Input

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.mode = "resolve"
            entry = app.query_one("#entry", Input)
            entry.value = "42"
            app.on_input_submitted(Input.Submitted(entry, "42"))
            await pilot.pause()
            assert "42" in str(app.query_one("#detail").render())

    asyncio.run(scenario())


def test_phase3_candidate_save_and_load(tmp_path, monkeypatch):
    """The Phase 3 desk loop, network-free: a retained table saves + reloads as a
    candidate list, and the Candidates browser lists it."""
    from astropy.table import Table

    from celestrium import candidates
    monkeypatch.setattr(candidates, "CAND_DIR", tmp_path)
    monkeypatch.setattr(candidates, "INDEX", tmp_path / "index.jsonl")
    monkeypatch.setattr(candidates, "_REPO", tmp_path)

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            # simulate a query having populated the result table
            app.last_table = Table({"id": [1, 2], "ra": [1.0, 2.0], "dec": [0.0, 1.0]})
            app.mode = "query"
            # save directly through the service layer the action delegates to
            candidates.save("tui-test", app.last_table, origin="tui:query")
            assert candidates.find_record("tui-test")["nrows"] == 2

            # the Candidates browser is network-free and shows the saved list
            app.mode = "candidates"
            app.action_candidates()
            await pilot.pause()
            assert app.query_one("#results").row_count == 1

            # guard rail: saving with no table is a no-op, not a crash
            app.last_table = None
            app.action_save_candidate()
            await pilot.pause()

    asyncio.run(scenario())
