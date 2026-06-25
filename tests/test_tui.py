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
