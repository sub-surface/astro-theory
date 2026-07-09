"""The Celestrium cockpit — a Textual app over registry + packets + cache.

Reimagined as a minimal, prompt-centric instrument:
- Unified prompt input with auto-parsing and Tab-cycle prefixing.
- Session context (remembers target and dataset across calls).
- Adaptive canvas vertical layout (collapses/expands views dynamically).
- Inline details panel (drawer style, toggles on Enter/Esc).
- Ambient strip containing the mini-orrery and breadcrumb session trail.

Launch with: python -m celestrium.tui
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Input, Label,
                             ListItem, ListView, RichLog, Select, Static)
from astropy.table import Table
from rich.markup import escape
from rich.text import Text

from .. import (cache, candidates, cutouts, packets, paths, planner, products,
                 registry, xmatch)
from .wireframe import Wireframe, PlatonicScene, TransitScene, SkyScatterScene

_PROMPTS = {
    "resolve": "object / 'RA Dec' — plan first, Enter a product row to fetch",
    "literature": 'ADS query, e.g. abs:"cosmic dipole" year:2024-2026',
    "query": "ADQL, e.g. SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source",
    "crossmatch": "catalogue id, e.g. vizier:VIII/65/nvss  (matches the current table)",
    "global": "(Global feeds load automatically — Enter runs the highlighted feed)",
    "candidates": "(Candidates load automatically — select a row to open it)",
    "runbooks": "(Runbooks load automatically — Enter runs the highlighted one)",
    "history": "(History loads automatically — no input)",
}
THEME_RING = ["ansi-dark", "tokyo-night", "nord", "gruvbox", "dracula",
              "catppuccin-mocha", "monokai", "textual-dark"]

_EGGS = {
    "42": "[b gold1]42[/] — the Answer to the Ultimate Question of Life, "
          "the Universe, and Everything. (Now find the Question.)",
    "elite": "[b green]RIGHT ON, COMMANDER![/] Wireframe drive engaged.",
    "thargoid": "[b green]⚠ THARGOID DETECTED[/] — raise shields, deploy E.C.M.",
    "xyzzy": "[dim]Nothing happens.[/]",
    "tea": "[b]Share and Enjoy.[/] ☕  (the Sirius Cybernetics Corp. thanks you)",
    "cake": "[yellow]The cake is a lie.[/]",
}

ARCHIVES_LIST = list(registry.ARCHIVES)
GLOBAL_FEED_EXECUTORS = {
    "neo": ("celestrium.neos", "fetch_close_approaches"),
    "satellite": ("celestrium.satellites", "fetch_visible_satellites"),
    "transient": ("celestrium.transients", "fetch_latest_transients"),
}
GLOBAL_FEED_ALIASES = {
    "neos": "neo",
    "cneos": "neo",
    "satellites": "satellite",
    "tle": "satellite",
    "tles": "satellite",
    "transients": "transient",
    "alerts": "transient",
}


def _global_feed_key(text: str) -> str:
    key = text.strip().lower()
    return GLOBAL_FEED_ALIASES.get(key, key)


def _ads_token_ok() -> bool:
    try:
        from .. import ads
        ads._token()
        return True
    except Exception:
        return False


def _find_coord_cols(tab: Table) -> tuple[str, str]:
    """Identify RA/Dec columns or raise a helpful error."""
    aliases_ra = ("ra", "RA", "ra_deg", "RA_ICRS", "RAJ2000", "_RAJ2000", "raj2000", "RAdeg")
    aliases_dec = ("dec", "DEC", "dec_deg", "DE_ICRS", "DEJ2000", "_DEJ2000", "dej2000", "DEdeg")
    ra = next((c for c in aliases_ra if c in tab.colnames), None)
    dec = next((c for c in aliases_dec if c in tab.colnames), None)
    if not ra or not dec:
        raise ValueError(
            f"Could not identify RA/Dec columns. Available columns: {tab.colnames}"
        )
    return ra, dec


def _sky_points(tab: Table, limit: int = 500) -> list[tuple[float, float]]:
    """Normalize RA/Dec columns to 0..1 canvas coordinates for ambient plots."""
    try:
        ra_col, dec_col = _find_coord_cols(tab)
    except ValueError:
        return []
    points = []
    for row in tab[:limit]:
        try:
            ra = float(row[ra_col]) % 360.0
            dec = float(row[dec_col])
        except (TypeError, ValueError):
            continue
        if ra != ra or dec != dec or dec < -90.0 or dec > 90.0:
            continue
        points.append((ra / 360.0, (dec + 90.0) / 180.0))
    return points


def open_path(path: str | os.PathLike) -> None:
    """Platform-agnostic path opener."""
    system = platform.system()
    path_str = str(path)
    if system == "Windows":
        os.startfile(path_str)
    elif system == "Darwin":
        subprocess.Popen(["open", path_str])
    else:
        subprocess.Popen(["xdg-open", path_str])


@dataclass
class SessionContext:
    target: planner.ResolvedTarget | None = None
    dataset: Table | None = None
    trail: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Modal screens
# --------------------------------------------------------------------------- #
class SaveCandidateScreen(ModalScreen):
    """Prompt for a name, then save the app's current table as a candidate list."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, default: str = "candidates"):
        super().__init__()
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("Save current rows as candidate list", classes="heading")
            yield Input(value=self._default, placeholder="list name", id="save-name")
            with Horizontal(classes="modal-actions"):
                yield Button("Save", variant="success", id="save-ok")
                yield Button("Cancel", id="save-cancel")

    def on_mount(self) -> None:
        self.query_one("#save-name", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self.query_one("#save-name", Input).value.strip()
                     if event.button.id == "save-ok" else None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class ImageSettingsScreen(ModalScreen):
    """Set product-planner parameters used by Resolve product executors."""

    BINDINGS = [("escape", "dismiss", "Cancel")]
    PRESETS = [
        ("Quick look", "quick-look"),
        ("Morphology", "morphology"),
        ("Wide context", "wide-context"),
        ("Paper figure", "paper-figure"),
        ("Time-domain host", "time-domain-host"),
    ]
    SURVEYS = [
        ("Auto best coverage", "auto"),
        ("Legacy Surveys", "legacy"),
        ("Pan-STARRS", "panstarrs"),
        ("DES", "des"),
        ("DSS2 fallback", "dss2"),
        ("SDSS optical", "sdss"),
        ("MAST UV/optical", "mast"),
        ("TESS", "tess"),
        ("Kepler/K2", "kepler"),
    ]
    PRODUCTS = [
        ("Auto quick-look", "auto"),
        ("HiPS colour image", "colour_image"),
        ("Multi-wavelength panel", "multi_panel"),
        ("Spectrum", "spectrum"),
        ("SDSS spectrum/image", "sdss"),
        ("MAST UV/optical observations", "mast"),
        ("TESS/Kepler lightcurves", "lightcurve"),
    ]
    WAVELENGTHS = [
        ("Auto", "auto"),
        ("UV", "UV"),
        ("Optical", "optical"),
        ("Near-IR", "near-IR"),
        ("Mid-IR", "mid-IR"),
        ("Radio", "radio"),
        ("X-ray", "X-ray"),
        ("Time-domain", "time-domain"),
    ]

    def __init__(self, settings: dict, target: planner.ResolvedTarget | None = None):
        super().__init__()
        self._s = dict(settings)
        self._target = target

    def _coverage_note(self) -> str:
        if self._target is None:
            return "[dim]Resolve a target to see coverage-aware product advice.[/]"
        survey = str(self._s.get("survey", "auto"))
        if survey not in cutouts.COLOR_SURVEYS:
            survey = "auto"
        note = planner.image_coverage_note(self._target.dec, survey)
        return (
            f"[b]{escape(self._target.display_name)}[/]  "
            f"{escape(self._target.object_class)}  "
            f"RA {self._target.ra:.5f} Dec {self._target.dec:+.5f}\n"
            f"{escape(note)}"
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("◇ PRODUCT PLANNER", classes="heading")
            yield Static(self._coverage_note(), id="img-recommendation")
            yield Label("Preset", classes="dim")
            yield Select(self.PRESETS,
                         value=self._s.get("preset", "quick-look"),
                         id="img-preset", allow_blank=False)
            yield Label("Product intent", classes="dim")
            yield Select(self.PRODUCTS,
                         value=self._s.get("product", "colour_image"),
                         id="img-product", allow_blank=False)
            yield Label("Wavelength", classes="dim")
            yield Select(self.WAVELENGTHS,
                         value=self._s.get("wavelength", "auto"),
                         id="img-wavelength", allow_blank=False)
            yield Label("Field of view (arcmin, or 'auto')", classes="dim")
            yield Input(value=str(self._s.get("fov", "auto")), id="img-fov")
            yield Label("Pixels (per side)", classes="dim")
            yield Input(value=str(self._s.get("pix", 512)), id="img-pix")
            yield Label("Preferred source / survey", classes="dim")
            yield Select(self.SURVEYS, value=self._s.get("survey", "auto"),
                         id="img-survey", allow_blank=False)
            with Horizontal(classes="modal-actions"):
                yield Button("Apply", variant="success", id="img-ok")
                yield Button("Cancel", id="img-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "img-ok":
            self.dismiss(None)
            return
        fov_raw = self.query_one("#img-fov", Input).value.strip().lower()
        try:
            pix = max(64, min(2048, int(self.query_one("#img-pix", Input).value)))
        except ValueError:
            pix = 512
        fov = "auto"
        if fov_raw not in ("", "auto"):
            try:
                fov = max(0.5, float(fov_raw))
            except ValueError:
                fov = "auto"
        self.dismiss({"preset": self.query_one("#img-preset", Select).value,
                      "fov": fov, "pix": pix,
                      "survey": self.query_one("#img-survey", Select).value,
                      "product": self.query_one("#img-product", Select).value,
                      "wavelength": self.query_one("#img-wavelength", Select).value})

    def action_dismiss(self) -> None:
        self.dismiss(None)


class AboutScreen(ModalScreen):
    """A tiny about/help card."""

    BINDINGS = [("escape", "dismiss", "Close"), ("enter", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(
                "[b cyan]CELESTRIUM[/] — A Zero-Clutter Astrophysics Terminal\n"
                "[dim]Target Resolution · Time-Domain Data · ADQL Queries · Runbook Workflows[/]\n\n"

                "Celestrium operates on a command-palette philosophy. Type slash commands to transition modes, "
                "or directly enter targets and ADQL queries.\n\n"

                "━━━━━━━━━ [b]CORE COMMANDS[/] ━━━━━━━━━━━━━━━━━\n"
                " [green]/resolve[/] [dim]<target>[/]  Resolve a star, exoplanet, NEO, or galaxy via SIMBAD/NASA.\n"
                " [green]/query[/] [dim]<adql>[/]      Execute raw ADQL against Gaia, VizieR, or MAST.\n"
                " [green]match[/] [dim]<catalog>[/]    Crossmatch the current active table against a catalog.\n"
                " [green]papers[/] [dim]<target>[/]   Search the ADS astrophysical literature database.\n"
                " [green]run[/] [dim]<workflow>[/]    Execute an automated multi-step runbook.\n"
                " [green]/global[/]                  Browse global feeds, separate from target/field products.\n\n"

                "━━━━━━━━━ [b]HUD PANELS[/] ━━━━━━━━━━━━━━━━━━━━\n"
                " [cyan]/history[/]      Browse past queries and results.\n"
                " [cyan]/candidates[/]   View saved target shortlists.\n"
                " [cyan]/runbooks[/]     View available automated workflows.\n"
                " [cyan]/global[/]       View live sky/event feeds.\n\n"

                "━━━━━━━━━ [b]PRODUCT SCOPES[/] ━━━━━━━━━━━━━━━━\n"
                " [b]Target[/] products belong to the resolved object.\n"
                " [b]Field[/] products belong to the sky position or blank field.\n"
                " [b]Global[/] feeds are live streams and are never mixed into Resolve plans.\n\n"

                "━━━━━━━━━ [b]SHORTCUTS[/] ━━━━━━━━━━━━━━━━━━━━━\n"
                "  [b]Ctrl+S[/] Save List       [b]Ctrl+O[/] Open Output     [b]Ctrl+G[/] Image Settings\n"
                "  [b]Ctrl+R[/] Refresh Cache   [b]F5[/] Re-run Action       [b]Ctrl+T[/] Toggle Theme\n"
                "  [b]F2[/] Next Solid          [b]F3[/] Toggle Spin         [b]i[/] or [b]/[/] Focus Prompt\n\n"

                "[dim]'Pull rows, not pixels.' — the house rule[/]",
                id="about-body")
            yield Button("Close", id="about-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_dismiss(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
class CelestriumApp(App):
    """Celestrium — a three-wing astrophysics instrument (TUI cockpit)."""

    CSS_PATH = "theme.tcss"
    TITLE = "Celestrium"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+s", "save_candidate", "Save list"),
        ("ctrl+o", "open_image", "Open img"),
        ("ctrl+g", "image_settings", "Image cfg"),
        ("ctrl+r", "refresh", "Refresh cache"),
        ("ctrl+t", "cycle_theme", "Theme"),
        ("ctrl+d", "toggle_debug", "Debug"),
        ("ctrl+h", "toggle_trail", "Trail"),
        ("f1", "about", "About"),
        ("f2", "next_solid", "Solid"),
        ("f3", "toggle_spin", "Spin"),
        ("f5", "rerun", "Re-run"),
        ("enter", "row_action", "Open row"),
        # Mode switch fallbacks
        ("alt+r", "mode_resolve", "Mode Resolve"),
        ("alt+l", "mode_literature", "Mode Lit"),
        ("alt+q", "mode_query", "Mode Query"),
        ("alt+x", "mode_crossmatch", "Mode X-match"),
        ("alt+g", "mode_global", "Mode Global"),
        ("alt+c", "mode_candidates", "Mode Cand"),
        ("alt+b", "mode_runbooks", "Mode Runbooks"),
        ("alt+h", "mode_history", "Mode Hist"),
    ]

    def __init__(self, active_line: str = "G-Euclid DR1 prep"):
        super().__init__()
        self.active_line = active_line
        self.mode = "resolve"
        self.last_table = None          # astropy Table from the last query/crossmatch
        self.last_image = None          # Path of the last rendered colour preview
        self.last_report = None         # Path of the last runbook index report
        self.last_target = None         # planner.ResolvedTarget from the last Resolve
        self.last_plans = []            # ranked ObservationPlans for last_target
        self._resolve_seq = 0           # request id: guards stale Resolve workers
        self._fetch_seq = 0             # fetch request id: guards stale product fetches
        self.debug_mode = False
        self.image_cfg = {
            "preset": "quick-look", "fov": "auto", "pix": 512, "survey": "auto",
            "product": "colour_image", "wavelength": "auto",
        }
        self._last_action = None        # (fn_name, *args) for refresh / re-run
        self._row_payloads = []         # parallel to displayed rows (contextual detail)
        self._row_render = None         # callable(payload) -> detail markup
        self.context = SessionContext()

    def action_mode_resolve(self) -> None: self.switch_to_mode("resolve")
    def action_mode_literature(self) -> None: self.switch_to_mode("literature")
    def action_mode_query(self) -> None: self.switch_to_mode("query")
    def action_mode_crossmatch(self) -> None: self.switch_to_mode("crossmatch")
    def action_mode_global(self) -> None: self.switch_to_mode("global")
    def action_mode_candidates(self) -> None: self.switch_to_mode("candidates")
    def action_mode_runbooks(self) -> None: self.switch_to_mode("runbooks")
    def action_mode_history(self) -> None: self.switch_to_mode("history")

    # ----- layout ----------------------------------------------------------- #
    def compose(self) -> ComposeResult:
        # Keep the Select widget hidden so we don't break code references
        archive_select = Select([(k, k) for k in registry.ARCHIVES],
                                 prompt="archive", id="archive", value="gaia")
        archive_select.display = False
        yield archive_select

        with Horizontal(id="body"):
            with Vertical(id="canvas-container"):
                with Horizontal(id="controls"):
                    yield Label("❯ ", id="prompt-prefix")
                    yield Input(placeholder="enter object, /query, /help, etc.", id="entry")
                    yield Label("", id="sys-status", classes="dim")

                # Adaptive Canvas Area
                with Vertical(id="canvas"):
                    yield Label("", id="target-hud", classes="dim")
                    yield Wireframe(id="orrery-idle")
                    with VerticalScroll(id="detail-scroll"):
                        yield Static("Pick a mode and submit.", id="detail")
                    yield DataTable(id="results", zebra_stripes=True)
                    with VerticalScroll(id="inline-detail"):
                        yield Static(id="inline-detail-text")

            with Vertical(id="ambient-strip"):
                yield Label("◇ ORRERY", classes="heading")
                yield Wireframe(id="orrery")  # ID is "#orrery" so test pilot functions
                yield Label("◇ TRAIL", classes="heading")
                yield ListView(id="trail-list")

        yield RichLog(id="status", max_lines=12, wrap=True, markup=True)

    def on_mount(self) -> None:
        self.theme = "ansi-dark"  # the requested default
        self.query_one("#results", DataTable).cursor_type = "row"
        self._refresh_subtitle()
        self._log(f"[b]Celestrium[/] ready — {len(registry.ARCHIVES)} archives, "
                  f"{len(registry.SAMPLE_RECIPES)} recipes.  [dim]F1 for keys[/]")
        self.query_one("#entry", Input).focus()

        # Hide non-active canvas items
        self.query_one("#orrery-idle", Wireframe).display = True
        self.query_one("#detail-scroll", VerticalScroll).display = False
        self.query_one("#results", DataTable).display = False
        self.query_one("#inline-detail", VerticalScroll).display = False
        self.query_one("#ambient-strip", Vertical).display = False  # Hide sidebar in idle

    # ----- helpers ---------------------------------------------------------- #
    def _refresh_subtitle(self) -> None:
        token = "[green]ADS ✓[/]" if _ads_token_ok() else "[yellow]ADS –[/]"
        dbg = "[magenta]DBG[/]" if self.debug_mode else ""
        plain_token = "ADS ok" if _ads_token_ok() else "ADS missing"
        plain_dbg = " DEBUG" if self.debug_mode else ""
        self.sub_title = f"active: {self.active_line}   {plain_token}{plain_dbg}"
        self.query_one("#sys-status", Label).update(f"{token} {dbg}")

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#status", RichLog).write(msg)
        except Exception:
            pass

    def _feedback(self, stage: str, msg: str, style: str = "cyan") -> None:
        self._log(f"[{style}]{escape(stage)}[/] {escape(msg)}")

    def _dbg(self, msg: str) -> None:
        if self.debug_mode:
            self._log(f"[magenta]· {msg}[/]")

    def _set_detail(self, text: str) -> None:
        try:
            self.query_one("#detail", Static).update(text)
        except Exception:
            pass

        # Instead of the heavy detail scroll, we parse out the critical info for the HUD.
        # An identity card puts "RA <deg>  Dec <deg>" on its third line — that's the cue.
        clean = Text.from_markup(text).plain
        lines = clean.split('\n')
        if len(lines) >= 3 and lines[2].startswith("RA "):
            self.query_one("#target-hud", Label).update(
                f"[b]{escape(lines[0])}[/b]  {escape(lines[1])}")
        else:
            self.query_one("#target-hud", Label).update(f"[dim]{escape(clean.replace(chr(10), ' · '))}[/]")

        # Also log it
        if self.mode != "resolve":
            self._log(f"[dim]{escape(clean.replace(chr(10), ' · '))}[/]")

    def _fill_table(self, columns, rows, payloads=None, render=None) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        if not columns:
            columns = [" "]
        table.add_columns(*columns)
        for row in rows:
            # We must use Text.from_markup if we want Rich markup in cells
            table.add_row(*[Text.from_markup(str(c)[:200]) for c in row])
        self._row_payloads = payloads or []
        self._row_render = render

    def _show_astropy(self, tab, ncols: int = 8, nrows: int = 50) -> None:
        """Render an astropy Table into the results grid."""
        cols = list(tab.colnames)[:ncols]
        rows = [tuple(r[c] for c in cols) for r in tab[:nrows]]
        # payload per row: the full (untruncated) record for the detail panel
        payloads = [{c: tab[c][i] for c in tab.colnames} for i in range(min(len(tab), nrows))]
        self._fill_table(cols, rows, payloads, self._render_row_detail)

    def _render_row_detail(self, payload: dict) -> str:
        lines = ["[b]row[/]"]
        for k, v in payload.items():
            lines.append(f"[cyan]{escape(str(k))}[/] {escape(str(v))}")
        return "\n".join(lines)

    # ----- mode switching --------------------------------------------------- #
    def switch_to_mode(self, mode: str) -> None:
        self.mode = mode

        # In a true CLI approach, the prompt stays minimalist.
        self.query_one("#prompt-prefix", Label).update("❯ ")
        self.query_one("#entry", Input).placeholder = _PROMPTS.get(mode, "enter command or object")

        # Adaptive views in canvas
        self.query_one("#orrery-idle", Wireframe).display = False
        self.query_one("#inline-detail", VerticalScroll).display = False
        self.query_one("#ambient-strip", Vertical).display = True  # Show sidebar in active mode

        results = self.query_one("#results", DataTable)
        detail_scroll = self.query_one("#detail-scroll", VerticalScroll)

        detail_scroll.display = False
        results.display = True
        results.styles.height = "1fr"
        results.show_header = (mode != "resolve")

        # Set ambient scene based on mode
        orrery = self.query_one("#orrery", Wireframe)
        if mode == "query":
            orrery.scene = SkyScatterScene()
        elif mode == "resolve":
            if self.context.target and self.context.target.match_kind == "exoplanet":
                orrery.scene = TransitScene()
            else:
                orrery.scene = PlatonicScene("dodecahedron")
        else:
            orrery.scene = PlatonicScene("icosahedron")

        # Auto-populate tables for browser modes
        if mode == "history":
            self.action_history()
        elif mode == "global":
            self.action_global()
        elif mode == "candidates":
            self.action_candidates()
        elif mode == "runbooks":
            self.action_runbooks()

    # ----- trail list ------------------------------------------------------- #
    def add_trail(self, label: str, mode: str, action_data: dict) -> None:
        # Avoid duplicate trail entries
        for idx, item in enumerate(self.context.trail):
            if item.get("label") == label and item.get("mode") == mode:
                self.context.trail.pop(idx)
                break

        self.context.trail.append({
            "label": label,
            "mode": mode,
            "action_data": action_data
        })

        if len(self.context.trail) > 15:
            self.context.trail.pop(0)

        # Postpone UI rebuilding to avoid ValueError race when clicked
        self.call_later(self._rebuild_trail_ui)

    def _rebuild_trail_ui(self) -> None:
        try:
            trail_list = self.query_one("#trail-list", ListView)
        except Exception:
            return  # in case the app is shutting down during deferred call
        trail_list.clear()
        for idx, item in enumerate(self.context.trail):
            active = "active" if idx == len(self.context.trail) - 1 else ""
            trail_list.append(ListItem(Label(f"· {item['label']}", classes=f"trail-item {active}")))

    def restore_trail_item(self, item: dict) -> None:
        mode = item["mode"]
        data = item["action_data"]
        self.switch_to_mode(mode)

        if mode == "resolve":
            self.do_resolve(data["name"])
        elif mode == "literature":
            self.do_literature(data["query"])
        elif mode == "query":
            self.do_query(data["archive"], data["query"], False)
        elif mode == "crossmatch":
            # Semantic crossmatch restore via exact cached tag loader
            self.do_open_cached(data["key"])
        elif mode == "global":
            # Trail restore is a revisit — serve the cached pull, don't re-hit the feed.
            self.do_global_feed(data["key"], refresh=False)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "trail-list":
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(self.context.trail):
                item = self.context.trail[idx]
                self._log(f"[dim]restoring trail:[/] {item['label']}")
                self.restore_trail_item(item)
                self.query_one("#entry", Input).focus()

    # ----- unified input submission ---------------------------------------- #
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._handle_command(event.value.strip())

    def _handle_command(self, text: str) -> None:
        """Central command dispatcher."""
        if not text:
            return

        lower = text.lower()
        if lower == "clear" or lower == "/clear":
            self.action_clear()
            return

        # Slash Commands
        if lower in ("/help", "?"):
            self.action_about()
            return
        if lower.startswith("/history"):
            self.switch_to_mode("history")
            return
        if lower.startswith("/global") or lower.startswith("/feeds"):
            parts = text.split(maxsplit=1)
            self.switch_to_mode("global")
            if len(parts) > 1:
                self.do_global_feed(_global_feed_key(parts[1]), refresh=True)
            return
        if lower.startswith("/runbooks"):
            self.switch_to_mode("runbooks")
            return
        if lower.startswith("/candidates"):
            self.switch_to_mode("candidates")
            return
        if lower.startswith("/resolve"):
            target = text.removeprefix("/resolve").strip()
            if target:
                self.switch_to_mode("resolve")
                self._last_action = ("resolve", target)
                self.do_resolve(target)
            else:
                self.switch_to_mode("resolve")
                self._log("[dim]usage:[/] /resolve <object name or coordinates>\n[dim]example:[/] /resolve TRAPPIST-1")
            return

        if lower.startswith("/query"):
            q = text.removeprefix("/query").strip()
            if not q:
                self.switch_to_mode("query")
                self._log("[dim]usage:[/] /query <ADQL statement>\n[dim]example:[/] /query SELECT TOP 5 source_id FROM gaiadr3.gaia_source")
                return
            archive = "gaia"
            query_text = q
            for arch in registry.ARCHIVES:
                prefix = f"{arch}:"
                if q.startswith(prefix):
                    archive = arch
                    query_text = q[len(prefix):].strip()
                    break
            self.query_one("#archive", Select).value = archive
            self.switch_to_mode("query")
            self._last_action = ("query", archive, query_text, False)
            self.do_query(archive, query_text, False)
            return

        # Easter eggs
        if self.mode == "resolve" and lower in _EGGS:
            # Shift canvas to display the detail Scroll panel, hiding others
            self.query_one("#orrery-idle", Wireframe).display = False
            self.query_one("#results", DataTable).display = False
            self.query_one("#detail-scroll", VerticalScroll).display = True
            self.query_one("#ambient-strip", Vertical).display = True  # Show sidebar

            self._set_detail(_EGGS[lower])
            if lower in ("elite", "thargoid"):
                # Engage correct solid in both mini and idle orreries
                self.query_one("#orrery", Wireframe).set_solid("icosahedron")
                try:
                    self.query_one("#orrery-idle", Wireframe).set_solid("icosahedron")
                except Exception:
                    pass
            return

        self._log(f"[dim]{self.mode}:[/] {text}")

        # Crossmatch prefix
        if text.startswith("×") or text.startswith("match "):
            cat = text.removeprefix("×").removeprefix("match ").strip()
            self.switch_to_mode("crossmatch")
            self._last_action = ("crossmatch", cat, False)
            self.do_crossmatch(cat, False)
            return

        # Runbook prefix
        if text.startswith("run "):
            rb = text.removeprefix("run ").strip()
            self.switch_to_mode("runbooks")
            self.do_runbook(rb)
            return

        if text.startswith("feed "):
            key = _global_feed_key(text.removeprefix("feed "))
            self.switch_to_mode("global")
            self.do_global_feed(key, refresh=True)
            return

        # Query prefixes (e.g. gaia: SELECT ...) or starting with SELECT
        is_query = False
        archive = "gaia"
        query_text = text
        for arch in registry.ARCHIVES:
            if text.startswith(f"{arch}:"):
                is_query = True
                archive = arch
                query_text = text[len(arch)+1:].strip()
                break
        if not is_query and (lower.startswith("select ") or lower.startswith("select\n")):
            is_query = True
            archive = self.query_one("#archive", Select).value or "gaia"

        if is_query:
            # Set the Select value before switching mode to prevent label race
            self.query_one("#archive", Select).value = archive
            self.switch_to_mode("query")
            self._last_action = ("query", archive, query_text, False)
            self.do_query(archive, query_text, False)
            return

        # Literature prefixes
        is_lit = False
        if any(p in lower for p in ("abs:", "author:", "year:", "title:", "bibcode:")):
            is_lit = True
        elif text.startswith("papers "):
            is_lit = True
            text = text.removeprefix("papers ").strip()
        elif lower == "papers":
            if self.context.target:
                is_lit = True
                text = f'"{self.context.target.display_name}"'
            else:
                self._log("[yellow]No active target for papers search — specify a query[/]")
                return

        if is_lit:
            self.switch_to_mode("literature")
            self._last_action = ("literature", text)
            self.do_literature(text)
            return

        # Product fetch command for target
        if lower in ("image", "cutout", "panel", "spectrum"):
            if not self.context.target:
                self._log("[yellow]No active target to fetch products for[/]")
                return
            plans = self.last_plans or planner.recommend_plans(self.context.target)
            self.last_plans = plans

            matched_idx = -1
            for idx, p in enumerate(plans):
                mods = p.product.modalities
                if lower == "spectrum" and "spectrum" in mods:
                    matched_idx = idx
                    break
                elif lower == "panel" and "multi_panel" in mods:
                    matched_idx = idx
                    break
                elif lower in ("image", "cutout") and "colour_image" in mods:  # Alias fix
                    matched_idx = idx
                    break

            if matched_idx != -1:
                self.switch_to_mode("resolve")
                self.execute_plan(matched_idx)
            else:
                self._log(f"[yellow]Product {lower} not available/executable for {escape(self.context.target.display_name)}[/]")
            return

        # Default: coordinates / object name -> Resolve
        if text.startswith("/"):
            self._log(f"[yellow]Unknown command:[/] {text.split()[0]}\n[dim]Type /help for available commands.[/]")
            return

        self.switch_to_mode("resolve")
        self._last_action = ("resolve", text)
        self.do_resolve(text)

    # ----- key / focus overrides ------------------------------------------- #
    def on_key(self, event) -> None:
        if event.key == "escape":
            entry = self.query_one("#entry", Input)
            detail_pane = self.query_one("#inline-detail", VerticalScroll)

            if self.focused == entry:
                entry.blur()
                if self.mode == "resolve":
                    self.query_one("#detail-scroll").focus()
                else:
                    self.query_one("#results", DataTable).focus()
                event.prevent_default()
                event.stop()
            elif detail_pane.display and self.focused == detail_pane:
                detail_pane.display = False
                self.query_one("#results", DataTable).focus()
                event.prevent_default()
                event.stop()

        elif event.key in ("i", "/"):
            if self.focused != self.query_one("#entry", Input):
                self.query_one("#entry", Input).focus()
                event.prevent_default()
                event.stop()

        elif event.key == "tab":
            entry = self.query_one("#entry", Input)
            if self.focused == entry:
                val = entry.value
                m = re.match(r"^(\w+[-_]?\w*):\s*(.*)$", val)
                if m:
                    prefix, rest = m.groups()
                    if prefix in ARCHIVES_LIST:
                        idx = ARCHIVES_LIST.index(prefix)
                        nxt = ARCHIVES_LIST[(idx + 1) % len(ARCHIVES_LIST)]
                        entry.value = f"{nxt}: {rest}"
                        entry.cursor_position = len(entry.value)

                        # Update prompt prefix label too
                        self.query_one("#prompt-prefix", Label).update(f"{nxt} ❯ ")
                        event.prevent_default()
                        event.stop()
                elif val.lower().startswith("select"):
                    entry.value = f"gaia: {val}"
                    entry.cursor_position = len(entry.value)
                    self.query_one("#prompt-prefix", Label).update("gaia ❯ ")
                    event.prevent_default()
                    event.stop()

    # ----- simple actions --------------------------------------------------- #
    def action_clear(self) -> None:
        self.query_one("#entry", Input).value = ""
        self._fill_table([], [])
        self.query_one("#orrery-idle", Wireframe).display = True
        self.query_one("#detail-scroll", VerticalScroll).display = False
        self.query_one("#results", DataTable).display = False
        self.query_one("#inline-detail", VerticalScroll).display = False
        self.query_one("#ambient-strip", Vertical).display = False  # Hide sidebar in idle

        # Clear session context as well to prevent stale state bugs
        self.last_table = None
        self.last_target = None
        self.last_plans = []
        self.context.target = None
        self.context.dataset = None
        self._resolve_seq += 1
        self._fetch_seq += 1

        self._log("Canvas and session state cleared.")

    def action_about(self) -> None:
        self.push_screen(AboutScreen())

    def action_next_solid(self) -> None:
        name = self.query_one("#orrery", Wireframe).next_solid()
        try:
            self.query_one("#orrery-idle", Wireframe).set_solid(name)
        except Exception:
            pass
        self._dbg(f"orrery → {name}")

    def action_toggle_spin(self) -> None:
        spinning = self.query_one("#orrery", Wireframe).toggle()
        try:
            self.query_one("#orrery-idle", Wireframe).toggle()
        except Exception:
            pass
        self._dbg(f"orrery spin {'on' if spinning else 'off'}")

    def action_cycle_theme(self) -> None:
        cur = self.theme if self.theme in THEME_RING else THEME_RING[0]
        nxt = THEME_RING[(THEME_RING.index(cur) + 1) % len(THEME_RING)]
        self.theme = nxt
        self._log(f"theme → [b]{nxt}[/]")

    def action_toggle_debug(self) -> None:
        self.debug_mode = not self.debug_mode
        self._refresh_subtitle()
        self._log(f"debug [b]{'ON' if self.debug_mode else 'off'}[/]")

    def action_image_settings(self) -> None:
        def _apply(cfg):
            if cfg:
                self.image_cfg = cfg
                self._log(
                    f"product planner: {cfg['product']} · {cfg['wavelength']} · "
                    f"FOV {cfg['fov']} · {cfg['pix']}px · {cfg['survey']}")
        self.push_screen(ImageSettingsScreen(self.image_cfg, self.last_target), _apply)

    def action_open_image(self) -> None:
        target = (self.last_report if self.mode == "runbooks" and self.last_report
                  else self.last_image)
        if not target:
            self._log("[yellow]nothing to open yet (fetch a product or run a runbook)[/]")
            return
        try:
            open_path(target)  # Platform-agnostic
        except Exception as e:
            self._log(f"[red]open failed: {escape(type(e).__name__)}[/]")

    def action_refresh(self) -> None:
        """Re-run the last data action bypassing the cache (refresh=True)."""
        act = self._last_action
        if not act:
            self._log("[yellow]nothing to refresh yet[/]")
            return
        self._dbg(f"refresh {act}")
        if act[0] == "query":
            self.do_query(act[1], act[2], True)
        elif act[0] == "crossmatch":
            self.do_crossmatch(act[1], True)
        elif act[0] == "resolve":
            self.do_resolve(act[1])
        elif act[0] == "literature":
            self.do_literature(act[1])
        elif act[0] == "global":
            self.do_global_feed(act[1], True)
        else:
            self._log("[dim]nothing cacheable to refresh[/]")

    def action_rerun(self) -> None:
        if self.mode == "history":
            self._history_row_action(rerun=True)
            return
        entry = self.query_one("#entry", Input)
        self._handle_command(entry.value.strip())

    def action_toggle_trail(self) -> None:
        trail = self.query_one("#trail-list", ListView)
        if self.focused == trail:
            self.query_one("#entry", Input).focus()
        else:
            trail.focus()

    def action_save_candidate(self) -> None:
        if self.last_table is None or len(self.last_table) == 0:
            self._log("[yellow]nothing to save — run a Query or Crossmatch first[/]")
            return
        origin = f"tui:{self.mode}"

        def _save(name):
            if not name:
                self._log("[dim]save cancelled[/]")
                return
            try:
                path = candidates.save(name, self.last_table, origin=origin)
                self._log(f"[green]saved {len(self.last_table)} rows -> {name}[/] "
                          f"[dim]{path}[/]")
            except Exception as e:
                self._log(f"[red]save failed: {escape(type(e).__name__)}: {escape(str(e))}[/]")

        self.push_screen(SaveCandidateScreen(default=packets.slug(origin)), _save)

    # ----- list/grid actions ------------------------------------------------ #
    def action_history(self) -> None:
        records = cache.manifest()
        if not records:
            self._fill_table(["(empty)"], [])
            self._set_detail("No cached queries yet — run a Query.")
            return
        shown = records[-50:][::-1]
        rows = [(r.get("utc", ""), r.get("archive", ""), r.get("nrows", ""),
                 r.get("hash", ""), str(r.get("query", ""))[:60]) for r in shown]
        self._fill_table(["utc", "archive", "rows", "hash", "query"], rows,
                         payloads=shown, render=self._render_history_detail)
        self._set_detail(f"{len(records)} cached pulls.\n"
                         "[dim]Enter[/] open cached rows · [dim]F5[/] re-run query")

    def _render_history_detail(self, rec: dict) -> str:
        return (f"[b]hash[/] {rec.get('hash', '')}\n[b]archive[/] {rec.get('archive', '')}\n"
                f"[b]rows[/] {rec.get('nrows', '')}\n[b]utc[/] {rec.get('utc', '')}\n\n"
                f"[b]query[/]\n{rec.get('query', '')}\n\n"
                "[dim]Enter[/] open · [dim]F5[/] re-run")

    def action_global(self) -> None:
        feeds = [cap for cap in planner.SOURCE_CAPABILITIES if cap.scope == "global"]
        if not feeds:
            self._fill_table(["(none)"], [])
            self._set_detail("No global feeds registered.")
            return
        rows = []
        payloads = []
        for cap in feeds:
            action = "[b green]Run[/]" if cap.status == "executable" else "[dim]Planned[/]"
            rows.append((action, cap.label, cap.status, cap.coverage_hint))
            payloads.append({"key": cap.key, "label": cap.label,
                             "status": cap.status, "scope": cap.scope,
                             "description": cap.coverage_hint})
        self._fill_table(["action", "global feed", "status", "description"],
                         rows, payloads, self._render_global_detail)
        self._set_detail("[b]Global Feeds[/]\n"
                         "These are live sky/event streams, not target or field products.\n"
                         "[dim]Enter[/] runs an executable feed into Query view.")

    def _render_global_detail(self, rec: dict) -> str:
        action = "Enter runs this feed" if rec.get("status") == "executable" else "planned only"
        return (f"[b cyan]{escape(rec.get('label', ''))}[/]\n"
                f"[b]scope[/] global\n[b]status[/] {escape(rec.get('status', ''))}\n"
                f"{escape(rec.get('description', ''))}\n\n[dim]{action}[/]")

    def action_candidates(self) -> None:
        records = candidates.latest()
        if not records:
            self._fill_table(["(empty)"], [])
            self._set_detail("No candidate lists yet.\n"
                             "Run a Crossmatch, then [dim]Ctrl+S[/] to save one.")
            return
        rows = [(r.get("name", ""), r.get("nrows", ""), str(r.get("origin", ""))[:40],
                 r.get("utc", "")) for r in records]
        self._fill_table(["name", "rows", "origin", "utc"], rows,
                         payloads=records, render=self._render_candidate_detail)
        self._set_detail(f"{len(records)} candidate lists.\n[dim]Enter[/] loads the rows.")

    def _render_candidate_detail(self, rec: dict) -> str:
        cols = ", ".join(rec.get("columns", []))
        return (f"[b cyan]{rec.get('name', '')}[/]\n[b]rows[/] {rec.get('nrows', '')}\n"
                f"[b]origin[/] {rec.get('origin', '')}\n[b]saved[/] {rec.get('utc', '')}\n"
                f"[b]note[/] {rec.get('note', '') or '—'}\n[b]columns[/] {cols}\n\n"
                "[dim]Enter[/] load these rows into the grid")

    def action_runbooks(self) -> None:
        specs = registry.RUNBOOKS
        if not specs:
            self._fill_table(["(none)"], [])
            self._set_detail("No runbooks registered.")
            return
        rows = [(name, spec.description, str(len(spec.steps)))
                for name, spec in specs.items()]
        payloads = [{"name": name, "description": spec.description,
                     "steps": len(spec.steps)} for name, spec in specs.items()]
        self._fill_table(["runbook", "description", "steps"], rows,
                         payloads, self._render_runbook_detail)
        self._set_detail(f"{len(specs)} runbooks.\n"
                         "[dim]Enter[/] runs the highlighted one → live progress + index report.")

    def _render_runbook_detail(self, rec: dict) -> str:
        return (f"[b cyan]{rec.get('name', '')}[/]\n{rec.get('description', '')}\n"
                f"[b]steps[/] {rec.get('steps', '')}\n\n"
                f"[dim]Enter[/] run → writes data/reports/runbook-{rec.get('name','')}.md\n"
                "[dim]Ctrl+O[/] opens the index when done")

    def _run_selected_runbook(self) -> None:
        i = self._current_row_index()
        if i is None or i >= len(self._row_payloads):
            return
        name = str(self._row_payloads[i].get("name", ""))
        if not name:
            return
        self._log(f"[dim]running runbook[/] {name} …")
        self.do_runbook(name)

    def _run_selected_global_feed(self) -> None:
        i = self._current_row_index()
        if i is None or i >= len(self._row_payloads):
            return
        rec = self._row_payloads[i]
        key = _global_feed_key(str(rec.get("key", "")))
        if not key:
            return
        if rec.get("status") != "executable":
            self._set_detail(
                f"[b cyan]{escape(rec.get('label', key))}[/]\n"
                "[yellow]This global feed is planned, not executable yet.[/]"
            )
            return
        self._log(f"[dim]running global feed[/] {key} …")
        self.do_global_feed(key, refresh=True)

    def _current_row_index(self):
        table = self.query_one("#results", DataTable)
        if table.row_count == 0:
            return None
        return table.cursor_row

    def _history_row_action(self, *, rerun: bool) -> None:
        i = self._current_row_index()
        if i is None or i >= len(self._row_payloads):
            return
        h = str(self._row_payloads[i].get("hash", ""))
        if not h:
            return
        if rerun:
            self._log(f"[dim]re-running[/] {h}")
            self.do_rerun(h)
        else:
            self._log(f"[dim]opening cached[/] {h}")
            self.do_open_cached(h)

    def _open_candidate(self) -> None:
        i = self._current_row_index()
        if i is None or i >= len(self._row_payloads):
            return
        name = str(self._row_payloads[i].get("name", ""))
        try:
            tab = candidates.load(name)
            self._accept_table_result(tab)  # Updates last_table/context.dataset
            self._set_detail(f"[b]{escape(name)}[/] — {len(tab)} candidate rows loaded.\n"
                             "[dim]Crossmatch[/] mode can now match these.")
            self._log(f"[green]loaded candidate list {name} ({len(tab)} rows)[/]")
        except Exception as e:
            self._log(f"[red]{escape(type(e).__name__)}: {escape(str(e))}[/]")

    def action_row_action(self) -> None:
        entry = self.query_one("#entry", Input)
        if self.focused == entry:
            self._handle_command(entry.value.strip())
            return
        self._activate_current_row()

    def _activate_current_row(self) -> None:
        if self.mode == "history":
            self._history_row_action(rerun=False)
        elif self.mode == "candidates":
            self._open_candidate()
        elif self.mode == "global":
            self._run_selected_global_feed()
        elif self.mode == "runbooks":
            self._run_selected_runbook()
        elif self.mode == "resolve":
            i = self._current_row_index()
            if i is not None and 0 <= i < len(self.last_plans):
                self.execute_plan(i)
        else:
            # Inline detail drawer toggle
            detail_pane = self.query_one("#inline-detail", VerticalScroll)
            if detail_pane.display:
                detail_pane.display = False
                self.query_one("#results", DataTable).focus()
            else:
                detail_pane.display = True
                detail_pane.focus()

    def on_data_table_row_selected(self, event) -> None:
        """Enter/click on a result row should run the row's contextual action."""
        table = getattr(event, "data_table", None) or getattr(event, "control", None)
        if getattr(table, "id", None) != "results":
            return
        self._activate_current_row()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Contextual detail panel: expand the highlighted row."""
        try:
            i = self.query_one("#results", DataTable).get_row_index(event.row_key)
        except Exception:
            return
        if self._row_render and 0 <= i < len(self._row_payloads):
            markup = self._row_render(self._row_payloads[i])
            if self.mode == "resolve":
                self._set_detail(markup)
            else:
                self.query_one("#inline-detail-text", Static).update(markup)

    # ----- UI thread safe result acceptors ----------------------- #
    def _accept_table_result(self, tab: Table) -> None:
        """Central table state committer."""
        self.last_table = tab
        self.context.dataset = tab
        self._show_astropy(tab)
        points = _sky_points(tab)
        if points:
            try:
                self.query_one("#orrery", Wireframe).scene = SkyScatterScene(points)
            except Exception:
                pass

    def _target_key(self, target: planner.ResolvedTarget | None) -> str | None:
        if target is None:
            return None
        return (
            f"{target.display_name}|{target.ra:.8f}|{target.dec:.8f}|"
            f"{target.object_class}|{target.match_kind}"
        )

    def _fetch_is_current(self, seq: int, target_key: str | None) -> bool:
        if seq != self._fetch_seq:
            return False
        return target_key is None or target_key == self._target_key(self.last_target)

    def _accept_resolve_result(self, seq: int, target: planner.ResolvedTarget | None, plans: list) -> None:
        """Apply state changes and update UI on the main thread."""
        # Resolve seq check
        if seq != self._resolve_seq:
            return
        if target is None:
            self._feedback("resolve", "no match for target", "yellow")
            self._set_detail("No match for target.")
            self._fill_table(["(no match)"], [])
            self.last_target, self.last_plans = None, []
            return

        self.last_target = target
        self.context.target = target

        cols = ["action", "product"]
        rows = []
        for p in plans:
            formatted_text = f"[b]{p.product.label}[/] · {p.product.status} · [dim]{p.product.coverage_hint}[/]"
            action = "[b green]Run[/]" if p.next_action == "fetch" else "[dim]Inspect[/]"
            rows.append((action, formatted_text))
        self._fill_table(cols, rows, list(plans), self._render_plan_detail)
        self._set_detail(self._identity_card(target))
        self.add_trail(target.display_name, "resolve", {"name": target.display_name})

        # Update ambient scene if we just resolved an exoplanet
        orrery = self.query_one("#orrery", Wireframe)
        if target.match_kind == "exoplanet":
            orrery.scene = TransitScene()
        else:
            orrery.scene = PlatonicScene("dodecahedron")

        self.last_plans = plans
        self._feedback(
            "resolve",
            f"{target.display_name}: {target.match_kind}, confidence {target.confidence:.2f}, "
            f"{len(plans)} ranked products",
            "green")

    def _accept_literature_result(self, docs: list, query_text: str) -> None:
        """Apply literature search result on the UI thread."""
        rows = [(str(d.get("bibcode", "")), str(d.get("year", "")),
                 packets.title_of(d)[:50]) for d in docs]
        self._fill_table(["bibcode", "year", "title"], rows, docs, self._render_paper_detail)
        self._set_detail(f"{len(docs)} papers for:\n{escape(query_text)}\n"
                          "[dim]highlight a row for authors + abstract[/]")
        self.add_trail(query_text[:12], "literature", {"query": query_text})

    def _accept_query_result(self, tab: Table, archive: str, adql: str) -> None:
        """Apply ADQL query result on the UI thread."""
        self._accept_table_result(tab)
        self._set_detail(f"[b]{escape(archive)}[/] → {len(tab)} rows (cached + logged).\n"
                          "[dim]Crossmatch[/] matches these · [dim]Ctrl+S[/] saves · "
                          "[dim]highlight a row[/] for full values")
        self.add_trail(f"{archive}:{len(tab)}r", "query", {"archive": archive, "query": adql})

    def _accept_crossmatch_result(self, out: Table, catalog: str, key: str) -> None:
        """Apply crossmatch result on the UI thread."""
        self._accept_table_result(out)
        self._set_detail(f"[b]xmatch[/] × {escape(catalog)}\n{len(out)} matches (r≤5\")\n"
                          "[b green]Ctrl+S[/] to save as candidates")
        self.add_trail(f"×:{catalog[:8]}", "crossmatch", {"catalog": catalog, "key": key})

    def _accept_plan_image_result(self, path: str, label: str, detail: str, fov: float,
                                  seq: int, target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self.last_image = path
        self._feedback("product", f"image ready: {label}; fov={fov}' path={path}", "green")
        self._set_detail(f"{detail}\n\n[b]image[/] {escape(label)}\nFOV {fov}'  ·  {self.image_cfg['pix']}px\n"
                         f"[dim]{escape(str(path))}[/]\n[b green]Ctrl+O[/] open · [b]Ctrl+G[/] settings")

    def _accept_plan_panel_result(self, path: str, detail: str, fov: float,
                                  seq: int, target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self.last_image = path
        self._feedback("product", f"multi-panel ready; fov={fov}' path={path}", "green")
        self._set_detail(f"{detail}\n\n[b]multi-wavelength panel[/]\nFOV {fov}'\n"
                         f"[dim]{escape(str(path))}[/]\n[b green]Ctrl+O[/] open")

    def _accept_plan_spectrum_result(self, path: str, source: str, summary: str,
                                     detail: str, seq: int,
                                     target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self.last_image = path
        self._feedback("product", f"spectrum ready: {source}; path={path}", "green")
        self._set_detail(f"{detail}\n\n[b]spectrum[/] {escape(source)}\n{escape(summary)}\n"
                         f"[dim]{escape(str(path))}[/]\n[b green]Ctrl+O[/] open")

    def _accept_plan_error(self, plan_label: str, detail: str, err_name: str,
                           err_msg: str, seq: int,
                           target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self._feedback("error", f"{plan_label} failed: {err_name}: {err_msg}", "red")
        self._set_detail(f"{detail}\n\n[red]Fetch failed:[/] {escape(err_name)} - {escape(err_msg)}\n"
                         "[dim]Please check connection or try an alternate survey.[/]")

    def _accept_plan_no_records(self, target_name: str, plan_label: str, detail: str,
                                seq: int, target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self._feedback("product", f"{plan_label}: no records for {target_name}", "yellow")
        self._set_detail(
            f"{detail}\n\n[yellow]No records found for {escape(target_name)} "
            f"in {escape(plan_label)}.[/]"
        )

    def _accept_tabular_product_result(self, tab: Table, target_name: str,
                                       plan_label: str, detail: str, seq: int,
                                       target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self.switch_to_mode("query")
        self._accept_table_result(tab)
        self._feedback("product", f"{plan_label}: loaded {len(tab)} rows for {target_name}", "green")
        self._set_detail(
            f"{detail}\n\n[green]Loaded {len(tab)} rows into Query view from "
            f"{escape(plan_label)} for {escape(target_name)}.[/]"
        )

    def _accept_product_result(self, result: products.ProductResult | None,
                               plan_label: str, detail: str, seq: int,
                               target_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        target_name = self.last_target.display_name if self.last_target else ""
        if result is None:
            self._accept_plan_no_records(target_name, plan_label, detail, seq, target_key)
            return
        if result.kind == "image":
            self._accept_plan_image_result(
                str(result.path), result.label, detail, result.fov or 0.0,
                seq, target_key)
        elif result.kind == "panel":
            self._accept_plan_panel_result(
                str(result.path), detail, result.fov or 0.0, seq, target_key)
        elif result.kind == "spectrum":
            self._accept_plan_spectrum_result(
                str(result.path), result.source or result.label, result.summary,
                detail, seq, target_key)
        elif result.kind == "table" and result.table is not None:
            if len(result.table) == 0:
                self._accept_plan_no_records(
                    result.target_name, plan_label, detail, seq, target_key)
            else:
                self._accept_tabular_product_result(
                    result.table, result.target_name, plan_label, detail,
                    seq, target_key)
        else:
            self._accept_plan_no_records(target_name, plan_label, detail, seq, target_key)

    def _accept_runbook_result(self, index: Path, name: str, steps_count: int, created_count: int) -> None:
        """Central runbook state committer on the main thread."""
        self.last_report = index
        self._set_detail(f"[b green]runbook {name} done[/]\n{steps_count} steps · "
                          f"{created_count} artifacts\n[dim]{escape(str(index))}[/]\n[b]Ctrl+O[/] open index")
        self._log(f"[green]runbook {name} → {index}[/]")

    def _accept_global_feed_result(self, tab: Table, key: str, label: str) -> None:
        self._last_action = ("global", key)
        self.switch_to_mode("query")
        self._accept_table_result(tab)
        self._feedback("global", f"{label}: loaded {len(tab)} rows", "green")
        self._set_detail(
            f"[b green]Global feed loaded[/]\n{escape(label)} · {len(tab)} rows\n"
            "[dim]This is a live/global stream, not a target or field product.[/]"
        )
        self.add_trail(f"feed:{key}", "global", {"key": key})

    # ----- workers (threaded; blocking calls off the UI loop) --------------- #
    def do_resolve(self, name: str) -> None:
        """Resolve trigger on the UI thread to prevent sequence race."""
        self._resolve_seq += 1
        self._fetch_seq += 1
        seq = self._resolve_seq
        self._feedback("resolve", f"queued identity lookup for {name}")
        self._set_detail(
            f"[b]Resolving[/] {escape(name)}\n"
            "[dim]Trying coordinate parsing, exact name services, Horizons-style aliases, "
            "and relaxed nearby matches.[/]"
        )
        self._run_resolve_worker(name, seq)

    @work(thread=True, exclusive=True)
    def _run_resolve_worker(self, name: str, seq: int) -> None:
        try:
            self.call_from_thread(
                self._feedback, "resolve",
                f"querying identity services for {name}")
            target = planner.resolve_target(name)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"resolve failed: {type(e).__name__}: {str(e)}", "red")
            return

        # Thread scheduling race check
        if seq != self._resolve_seq:
            return

        plans = []
        if target is not None:
            self.call_from_thread(
                self._feedback, "resolve",
                f"ranking product capabilities for {target.display_name}")
            plans = planner.recommend_plans(target)

        self.call_from_thread(self._accept_resolve_result, seq, target, plans)

    def _identity_card(self, target) -> str:
        """Identity + ambiguity + ranked products + coverage — the planner card."""
        head = [
            f"[b cyan]{escape(target.display_name)}[/]",
            f"[yellow]{escape(target.otype or '?')}[/] · {escape(target.object_class)} · "
            f"[magenta]{escape(target.match_kind)}[/] (conf {target.confidence:.2f})",
            f"RA {target.ra:.5f}  Dec {target.dec:+.5f}",
        ]
        if target.alternatives:
            alts = ", ".join(escape(str(a.get("name", "?"))) for a in target.alternatives[:4])
            head.append(f"[dim]also matched: {alts}[/]")
        lines = head + ["", "[b]recommended products[/]  [dim](Enter a row to fetch)[/]"]
        for plan in planner.recommend_plans(target):
            mark = "▶" if plan.next_action == "fetch" else "·"
            lines.append(f"{mark} {escape(plan.product.label)} [{escape(plan.product.status)}]")
        lines += ["", planner.image_coverage_note(
            target.dec, self.image_cfg.get("survey", "auto"))]
        return "\n".join(lines)

    def _render_plan_detail(self, plan) -> str:
        head = self._identity_card(self.last_target) if self.last_target else ""
        fetch = ("Enter to fetch" if plan.next_action == "fetch"
                 else "inspect only — not yet fetchable here")
        return (f"{head}\n\n[b]selected:[/] {escape(plan.product.label)}\n"
                f"[dim]{escape(plan.product.coverage_hint)}[/]\n"
                f"access {escape(plan.product.access)} · cost {plan.product.fetch_cost} · "
                f"status {escape(plan.product.status)}\n[b green]{fetch}[/]")

    def execute_plan(self, index: int) -> None:
        """Trigger background product fetch execution with sequence guarding."""
        target = self.last_target
        if target is None:
            self._feedback("product", "no active target to fetch from", "yellow")
            return
        if index < 0 or index >= len(self.last_plans):
            self._feedback("product", f"row {index} is not a valid product", "yellow")
            return
        plan = self.last_plans[index]
        if plan.next_action != "fetch":
            self._feedback(
                "product",
                f"{plan.product.label} is inspect-only for this target", "yellow")
            self._set_detail(self._render_plan_detail(plan))
            return
        self._fetch_seq += 1
        seq = self._fetch_seq
        self._feedback(
            "product",
            f"queued {plan.product.label} for {target.display_name}; "
            f"status={plan.product.status}, access={plan.product.access}")
        self._run_execute_plan_worker(index, seq, self._target_key(target))

    @work(thread=True, exclusive=True)
    def _run_execute_plan_worker(self, index: int, seq: int, target_key: str | None) -> None:
        target = self.last_target
        if target is None or index >= len(self.last_plans):
            return
        if not self._fetch_is_current(seq, target_key):
            return
        plan = self.last_plans[index]
        card = self._identity_card(target)
        self.call_from_thread(
            self._feedback, "product",
            f"fetching {plan.product.label}; datatype={','.join(plan.product.modalities) or 'table'}")

        # Show working feedback immediately in the details panel
        loading_msg = (
            f"{card}\n\n"
            f"[b yellow]⠋ Fetching {escape(plan.product.label)}...[/]\n"
            f"[dim]Running background query to pull data slice...[/]"
        )
        self.call_from_thread(self._set_detail, loading_msg)

        try:
            result = products.execute_product(
                target,
                plan,
                self.image_cfg,
                emit=lambda msg: self.call_from_thread(self._feedback, "product", msg),
            )
        except NotImplementedError:
            self.call_from_thread(
                self._feedback, "product",
                f"{plan.product.label} is inspect-only in the cockpit", "yellow")
            self.call_from_thread(
                self._set_detail,
                f"{card}\n\n[yellow]{escape(plan.product.label)} is not yet fetchable from "
                f"the cockpit.[/]\n[dim]{escape(plan.product.coverage_hint)}[/]\n"
                f"[dim]status: {escape(plan.product.status)} · access: {escape(plan.product.access)}[/]")
            return
        except Exception as e:
            self.call_from_thread(
                self._accept_plan_error, plan.product.label, card,
                type(e).__name__, str(e), seq, target_key
            )
            return

        self.call_from_thread(
            self._accept_product_result, result, plan.product.label, card, seq, target_key)

    @work(thread=True, exclusive=True)
    def do_literature(self, query_text: str) -> None:
        try:
            self.call_from_thread(
                self._feedback, "literature",
                f"ADS search queued; rows=15 query={query_text}")
            from .. import ads
            docs = ads.search(query_text, rows=15,
                              fl=("bibcode", "title", "author", "year",
                                  "citation_count", "abstract"))
            self.call_from_thread(
                self._feedback, "literature",
                f"ADS returned {len(docs)} documents", "green")
            self.call_from_thread(self._accept_literature_result, docs, query_text)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"ADS failed: {type(e).__name__}: {str(e)} (ADS token?)", "red")

    def _render_paper_detail(self, doc: dict) -> str:
        authors = doc.get("author") or []
        alist = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        abstract = doc.get("abstract") or "[dim](no abstract)[/]"
        cites = doc.get("citation_count", "—")
        return (f"[b]{escape(packets.title_of(doc))}[/]\n\n[cyan]{escape(alist)}[/]\n"
                f"[dim]{doc.get('year', '?')} · {escape(doc.get('bibcode', ''))} · "
                f"{cites} cites[/]\n\n{escape(str(abstract))}")

    @work(thread=True, exclusive=True)
    def do_global_feed(self, key: str, refresh: bool = True) -> None:
        key = _global_feed_key(key)
        self.call_from_thread(
            self._feedback, "global",
            f"feed request {key}; refresh={refresh}")
        caps = {cap.key: cap for cap in planner.SOURCE_CAPABILITIES
                if cap.scope == "global"}
        cap = caps.get(key)
        if cap is None:
            self.call_from_thread(
                self._feedback, "error",
                f"unknown global feed {key!r}", "red")
            return
        if cap.status != "executable":
            self.call_from_thread(
                self._feedback, "global",
                f"{cap.label} is planned, not executable", "yellow")
            self.call_from_thread(
                self._set_detail,
                f"[b cyan]{escape(cap.label)}[/]\n"
                "[yellow]This global feed is planned, not executable yet.[/]"
            )
            return

        module_name, func_name = GLOBAL_FEED_EXECUTORS.get(key, (None, None))
        if module_name is None:
            self.call_from_thread(
                self._feedback, "error",
                f"no executor for global feed {key}", "red")
            return

        try:
            import importlib
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            self.call_from_thread(
                self._feedback, "global",
                f"{cap.label}; executor={module_name}.{func_name}")

            def _fetch():
                result = func()
                return result if result is not None else Table()

            query = f"global-feed:{key}|executor={module_name}.{func_name}"
            tab = cache.cached_query(f"global-{key}", query, _fetch, refresh=refresh)
            if len(tab) == 0:
                self.call_from_thread(
                    self._feedback, "global",
                    f"{cap.label}: no rows returned", "yellow")
                self.call_from_thread(
                    self._set_detail,
                    f"[b cyan]{escape(cap.label)}[/]\n[yellow]No rows returned.[/]\n"
                    "[dim]The feed may be unavailable or not wired yet.[/]"
                )
                return
            self.call_from_thread(self._accept_global_feed_result, tab, key, cap.label)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"global feed failed: {type(e).__name__}: {str(e)}", "red"
            )

    @work(thread=True, exclusive=True)
    def do_query(self, archive: str, adql: str, refresh: bool) -> None:
        try:
            self.call_from_thread(
                self._feedback, "query",
                f"{archive} request; refresh={refresh}; {len(adql)} characters")
            source = registry.resolve_query_source(archive)
            if source is None:
                self.call_from_thread(
                    self._feedback, "error",
                    f"unknown archive {archive!r}", "red")
                return
            self.call_from_thread(
                self._feedback, "query",
                f"source={source.__class__.__name__}; cache={'refresh' if refresh else 'normal'}")
            tab = cache.cached_query(archive, adql, lambda: source.query(adql), refresh=refresh)
            self.call_from_thread(
                self._feedback, "query",
                f"{archive} returned {len(tab)} rows", "green")
            self.call_from_thread(self._accept_query_result, tab, archive, adql)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"query failed: {type(e).__name__}: {str(e)}", "red")

    @work(thread=True, exclusive=True)
    def do_crossmatch(self, catalog: str, refresh: bool) -> None:
        if self.last_table is None or len(self.last_table) == 0:
            self.call_from_thread(
                self._feedback, "crossmatch",
                "no table to match; run a Query or open a Candidate first", "yellow")
            return
        local = self.last_table
        self.call_from_thread(
            self._feedback, "crossmatch",
            f"{len(local)} local rows -> {catalog}; refresh={refresh}")

        try:
            ra, dec = _find_coord_cols(local)  # Explicit columns
        except ValueError as e:
            self.call_from_thread(
                self._feedback, "error",
                f"crossmatch coordinate error: {str(e)}", "red")
            return

        try:
            # Strong unique cache key based on hashed coordinates
            coords = []
            for r in local:
                try:
                    coords.append(f"{r[ra]},{r[dec]}")
                except Exception:
                    pass
            coord_str = "|".join(coords)
            fingerprint = hashlib.md5(coord_str.encode('utf-8')).hexdigest()[:10]
            tag = f"tui|{catalog}|r5|{ra}|{dec}|{fingerprint}"

            self.call_from_thread(
                self._feedback, "crossmatch",
                f"columns={ra}/{dec}; radius=5 arcsec; key={fingerprint}")
            out = cache.cached_query(
                "xmatch", tag,
                lambda: xmatch.match(local, cat2=catalog, ra=ra, dec=dec, radius_arcsec=5.0),
                refresh=refresh)
            self.call_from_thread(
                self._feedback, "crossmatch",
                f"{catalog} returned {len(out)} matches", "green")

            # Extract key for precise trail recovery
            key = hashlib.sha1(f"xmatch\n{tag}".encode()).hexdigest()[:16]
            self.call_from_thread(self._accept_crossmatch_result, out, catalog, key)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"crossmatch failed: {type(e).__name__}: {str(e)}", "red")

    @work(thread=True, exclusive=True)
    def do_open_cached(self, hash_prefix: str) -> None:
        try:
            self.call_from_thread(
                self._feedback, "history",
                f"opening cached table {hash_prefix}")
            tab = cache.load_cached(hash_prefix)
            self.call_from_thread(self._accept_table_result, tab)  # Updates last_table/context.dataset
            self.call_from_thread(
                self._feedback, "history",
                f"{hash_prefix}: loaded {len(tab)} rows", "green")
            self.call_from_thread(self._set_detail,
                                  f"cached [b]{escape(hash_prefix)}[/] → {len(tab)} rows loaded.\n"
                                  "[dim]Crossmatch[/]/[dim]Ctrl+S[/] now work on these.")
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"open cached failed: {type(e).__name__}: {str(e)}", "red")

    @work(thread=True, exclusive=True)
    def do_rerun(self, hash_prefix: str) -> None:
        try:
            self.call_from_thread(
                self._feedback, "history",
                f"re-running cached query {hash_prefix}")
            rec = cache.find_record(hash_prefix)
            if rec is None:
                self.call_from_thread(
                    self._feedback, "error",
                    f"no record for {hash_prefix}", "red")
                return
            source = registry.resolve_query_source(rec["archive"])
            if source is None:
                self.call_from_thread(
                    self._feedback, "error",
                    f"archive {rec['archive']!r} not re-runnable", "red")
                return
            tab = cache.cached_query(rec["archive"], rec["query"],
                                     lambda: source.query(rec["query"]), refresh=True)
            self.call_from_thread(self._accept_table_result, tab)  # Updates last_table/context.dataset
            self.call_from_thread(
                self._feedback, "history",
                f"{hash_prefix}: rerun returned {len(tab)} rows", "green")
            self.call_from_thread(self._set_detail,
                                  f"re-ran [b]{escape(hash_prefix)}[/] → {len(tab)} fresh rows.")
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"rerun failed: {type(e).__name__}: {str(e)}", "red")

    @work(thread=True, exclusive=True)
    def do_runbook(self, name: str) -> None:
        """Run a registry runbook with live per-step progress, then save its index."""
        def on_step(note: str) -> None:
            self.call_from_thread(self._log, note)
        try:
            self.call_from_thread(
                self._feedback, "runbook",
                f"starting {name}; report dir={paths.REPORTS_DIR}")
            result = packets.run_runbook(
                name, reports_dir=paths.REPORTS_DIR, atlas_dir=paths.ATLAS_DIR,
                posters_dir=paths.POSTERS_DIR, contact_sheet=cutouts.contact_sheet,
                on_step=on_step)
            paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            index = paths.REPORTS_DIR / f"runbook-{name}.md"
            index.write_text(result.to_markdown() + "\n", encoding="utf-8")
            self.call_from_thread(self._accept_runbook_result, index, name, len(result.notes), len(result.created))
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"runbook failed: {type(e).__name__}: {str(e)}", "red")


def main() -> None:
    CelestriumApp().run()


if __name__ == "__main__":
    main()
