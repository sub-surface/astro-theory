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


def test_planner_settings_can_select_spectrum_network_free():
    from celestrium.tui.app import ImageSettingsScreen

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.action_image_settings()
            await pilot.pause()
            assert isinstance(app.screen, ImageSettingsScreen)
            app.screen.query_one("#img-product").value = "spectrum"
            app.screen.query_one("#img-ok").press()
            await pilot.pause()
            assert app.image_cfg["product"] == "spectrum"

    asyncio.run(scenario())


def test_render_preview_uses_spectrum_backend_for_spectrum_product(tmp_path, monkeypatch):
    from celestrium import spectra

    class Result:
        path = tmp_path / "spec.png"
        summary = "fake spectrum"
        source = "NED"

    seen = {}

    def fake_fetch(name):
        seen["name"] = name
        Result.path.write_text("fake", encoding="utf-8")
        return Result()

    monkeypatch.setattr(spectra, "fetch_ned_spectrum", fake_fetch)
    app = CelestriumApp(active_line="test-line")
    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    app._set_detail = lambda text: None
    app.image_cfg = {"fov": "auto", "pix": 512, "survey": "auto",
                     "product": "spectrum", "wavelength": "auto"}
    row = {"otype": "QSO"}
    app._render_preview_image("3C 273", row, 187.2, 2.0, "detail")
    assert seen["name"] == "3C 273"
    assert app.last_image == Result.path


def test_resolve_detail_includes_planner_recommendations():
    app = CelestriumApp(active_line="test-line")
    detail = app._planner_detail("3C 273", "QSO", 187.2779, 2.0524, "base")
    assert "recommended" in detail.lower()
    assert "NED" in detail or "SDSS" in detail
    assert "HEASARC" in detail


def test_resolve_plans_first_without_fetching(monkeypatch):
    """Deliberate fetch: Resolve populates ranked product rows and fetches nothing."""
    from celestrium import planner

    target = planner.ResolvedTarget(
        display_name="3C 273", aliases=(), ra=187.2779, dec=2.0524, otype="QSO",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    monkeypatch.setattr(planner, "resolve_target", lambda name: target)

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.mode = "resolve"
            app.do_resolve("3C 273")
            for _ in range(100):
                await asyncio.sleep(0.02)
                if app.last_plans:
                    break
            # the ranked plan list landed in the grid, but nothing was fetched
            assert app.last_target is target
            assert app.last_plans
            assert app.query_one("#results").row_count == len(app.last_plans)
            assert app.last_image is None  # no auto image/spectrum

    asyncio.run(scenario())


def test_slash_query_dispatches_without_network():
    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            seen = {}

            def fake_do_query(archive, adql, refresh):
                seen["archive"] = archive
                seen["adql"] = adql
                seen["refresh"] = refresh

            app.do_query = fake_do_query
            app._handle_command("/query irsa: SELECT TOP 1 ra, dec FROM demo")
            await pilot.pause()

            assert seen == {
                "archive": "irsa",
                "adql": "SELECT TOP 1 ra, dec FROM demo",
                "refresh": False,
            }
            assert app.mode == "query"

    asyncio.run(scenario())


def test_enter_in_prompt_runs_current_text_without_panel_focus():
    from textual.widgets import Input

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            seen = {}

            def fake_resolve(name):
                seen["name"] = name

            app.do_resolve = fake_resolve
            entry = app.query_one("#entry", Input)
            entry.value = "M87"
            entry.focus()

            await pilot.press("enter")
            await pilot.pause()

            assert seen == {"name": "M87"}

    asyncio.run(scenario())


def test_resolve_row_selection_runs_current_product():
    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            seen = {}
            app.mode = "resolve"
            app.last_plans = [object()]
            app._fill_table(["action", "product"], [("[b green]Run[/]", "demo")])
            app._current_row_index = lambda: 0
            app.execute_plan = lambda index: seen.setdefault("index", index)

            table = app.query_one("#results")

            class Event:
                data_table = table

            app.on_data_table_row_selected(Event())
            await pilot.pause()

            assert seen == {"index": 0}

    asyncio.run(scenario())


def test_stale_product_fetch_cannot_overwrite_new_target(tmp_path):
    from celestrium import planner

    old_target = planner.ResolvedTarget(
        display_name="Old", aliases=(), ra=1.0, dec=2.0, otype="G",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    new_target = planner.ResolvedTarget(
        display_name="New", aliases=(), ra=3.0, dec=4.0, otype="G",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    path = tmp_path / "old.jpg"
    path.write_text("old", encoding="utf-8")

    app = CelestriumApp(active_line="test-line")
    app._fetch_seq = 1
    app.last_target = old_target
    old_key = app._target_key(old_target)
    app.last_target = new_target

    app._accept_plan_image_result(path, "Old survey", "detail", 5.0, 1, old_key)

    assert app.last_image is None


def test_tabular_product_fetch_uses_cache(monkeypatch):
    from astropy.table import Table
    from celestrium import cache, planner, vizier

    target = planner.ResolvedTarget(
        display_name="3C 273", aliases=(), ra=187.2779, dec=2.0524, otype="QSO",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    plan = next(p for p in planner.recommend_plans(target)
                if p.product.key == "vizier")
    seen = {}

    monkeypatch.setattr(
        vizier, "cone_pull",
        lambda ra, dec: Table({"ra": [ra], "dec": [dec]}))

    def fake_cached_query(archive, query, fetch, refresh=False):
        seen["archive"] = archive
        seen["query"] = query
        seen["refresh"] = refresh
        return fetch()

    monkeypatch.setattr(cache, "cached_query", fake_cached_query)

    app = CelestriumApp(active_line="test-line")
    app._fetch_seq = 1
    app.last_target = target
    target_key = app._target_key(target)
    app.call_from_thread = lambda fn, *args, **kwargs: None

    app._fetch_tabular_product(
        target, plan, 1, target_key, "card",
        module_name="celestrium.vizier", fetch_func="cone_pull",
        args=(target.ra, target.dec))

    assert seen["archive"] == "product-vizier"
    assert "3C 273" in seen["query"]
    assert seen["refresh"] is False


def test_query_table_updates_scatter_scene_from_coordinates():
    from astropy.table import Table
    from celestrium.tui.wireframe import SkyScatterScene

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.switch_to_mode("query")
            app._accept_table_result(Table({"ra": [0.0, 180.0],
                                            "dec": [-90.0, 90.0]}))
            await pilot.pause()
            scene = app.query_one("#orrery").scene
            assert isinstance(scene, SkyScatterScene)
            assert scene.pts == [(0.0, 0.0), (0.5, 1.0)]

    asyncio.run(scenario())


def test_global_feeds_browser_is_distinct_from_target_products():
    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.switch_to_mode("global")
            await pilot.pause()

            assert app.mode == "global"
            assert app.query_one("#results").row_count > 0
            assert all(payload["scope"] == "global"
                       for payload in app._row_payloads)

    asyncio.run(scenario())


def test_global_feed_runs_through_cache(monkeypatch):
    from astropy.table import Table
    from celestrium import cache, neos

    monkeypatch.setattr(neos, "fetch_close_approaches",
                        lambda: Table({"des": ["demo"], "dist": ["0.01"]}))
    seen = {}

    def fake_cached_query(archive, query, fetch, refresh=False):
        seen["archive"] = archive
        seen["query"] = query
        seen["refresh"] = refresh
        return fetch()

    monkeypatch.setattr(cache, "cached_query", fake_cached_query)

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.switch_to_mode("global")
            app.do_global_feed("neos", refresh=True)
            for _ in range(100):
                await asyncio.sleep(0.02)
                if app.last_table is not None:
                    break

            assert seen["archive"] == "global-neo"
            assert "global-feed:neo" in seen["query"]
            assert seen["refresh"] is True
            assert app.mode == "query"
            assert len(app.last_table) == 1

    asyncio.run(scenario())


def test_execute_plan_row_fetches_colour_image(monkeypatch, tmp_path):
    """Enter on a colour-image plan row fetches via cutouts.color_auto (no network)."""
    from celestrium import cutouts

    path = tmp_path / "color.jpg"
    path.write_text("img", encoding="utf-8")
    monkeypatch.setattr(cutouts, "color_auto",
                        lambda *a, **k: (path, "Legacy Surveys DR10 (deep)"))

    app = CelestriumApp(active_line="test-line")
    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    app._set_detail = lambda text: None
    app._fetch_colour_image("3C 273", "QSO", 187.2779, 2.0524, "card")
    assert app.last_image == path


def test_runbooks_browser_is_network_free():
    from celestrium import registry

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.mode = "runbooks"
            app.action_runbooks()
            await pilot.pause()
            assert app.query_one("#results").row_count == len(registry.RUNBOOKS)

    asyncio.run(scenario())


def test_do_runbook_writes_index_with_live_progress(monkeypatch, tmp_path):
    from celestrium import packets, paths

    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path)
    steps = ["- step 1 done", "- step 2 done"]

    def fake_run_runbook(name, *, reports_dir, atlas_dir, posters_dir,
                         contact_sheet, on_step=None, **kw):
        for s in steps:
            if on_step:
                on_step(s)
        return packets.RunbookResult(name, "desc", list(steps), [])

    monkeypatch.setattr(packets, "run_runbook", fake_run_runbook)

    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            app.mode = "runbooks"
            app.do_runbook("euclid-q1")
            index = tmp_path / "runbook-euclid-q1.md"
            for _ in range(100):
                await asyncio.sleep(0.02)
                if app.last_report == index:
                    break
            assert index.exists()
            assert app.last_report == index
            assert "step 1 done" in index.read_text(encoding="utf-8")

    asyncio.run(scenario())


def test_trail_selection_is_safe_against_race(monkeypatch):
    """Clicking/selecting a trail item restores it without raising ValueError."""
    async def scenario():
        app = CelestriumApp(active_line="test-line")
        async with app.run_test() as pilot:
            # Add trail items
            app.add_trail("gaia:10r", "query", {"archive": "gaia", "query": "SELECT 1"})
            app.add_trail("SIMBAD:M31", "resolve", {"name": "M31"})

            # Wait for deferred _rebuild_trail_ui callback to run
            await pilot.pause()

            list_view = app.query_one("#trail-list")
            assert len(list_view) > 0

            # Select the first item in the trail-list (triggers selection event/rebuild)
            list_view.index = 0
            await pilot.pause()

            # Defer selection again to ensure no errors raised on event loop ticks
            await pilot.pause()

    asyncio.run(scenario())
