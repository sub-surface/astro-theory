"""The Celestrium cockpit — a Textual app over config + core/kernel.

A prompt-centric instrument with visible state:
- Context bar (top): view · target · retained table · archive · ADS status.
- Canvas: one results grid (idle shows the big orrery).
- Side panel: the single detail sink (identity cards, row expansions, product
  results), plus the session trail and the mini-orrery.
- REPL prompt (bottom) with slash-command autocomplete; parsing lives in
  commands.parse (pure + unit-tested), workers execute here.
- Slim status log (Ctrl+D expands it with debug tracing) and a Footer of keys.

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
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import (Button, DataTable, Footer, Input, Label,
                             ListItem, ListView, RichLog, Select, Static)
from textual.widgets.data_table import RowDoesNotExist
from astropy.table import Table
from rich.markup import escape
from rich.text import Text

from .. import (candidates, config, cutouts, paths, plots, resolvers, tables, xmatch)
from . import commands
from .wireframe import (Wireframe, PlatonicScene, TransitScene, OrbitScene,
                        SkyMarkScene, SkyScatterScene)

_PROMPTS = {
    "resolve": "object / 'RA Dec' — plan first, Enter a product row to fetch",
    "literature": 'ADS query, e.g. abs:"cosmic dipole" year:2024-2026',
    "query": "ADQL, e.g. SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source",
    "crossmatch": "catalogue id, e.g. vizier:VIII/65/nvss  (matches the current table)",
    "global": "(Global feeds load automatically — Enter runs the highlighted feed)",
    "candidates": "(Candidates load automatically — select a row to open it)",
    "runbooks": "(Runbooks load automatically — Enter runs the highlighted one)",
    "history": "(History loads automatically — no input)",
    "sweep": "/sweep [target] — active target by default",
    "watch": "/watch [target | candidate-list] — active target by default",
}
THEME_RING = ["ansi-dark", "tokyo-night", "nord", "gruvbox", "dracula",
              "catppuccin-mocha", "monokai", "textual-dark"]

ARCHIVES_LIST = list(config.ARCHIVES)
# Global feed executors are config data (shared with the CLI `feed` command).
GLOBAL_FEED_EXECUTORS = config.GLOBAL_FEED_EXECUTORS


def _ads_token_ok() -> bool:
    try:
        from .. import ads
        ads._token()
        return True
    except Exception:
        return False


# RA/Dec alias lists live in tables.py (shared with the watch packet's
# per-row cone search over a candidate list) — kept as module attrs here
# only so _payload_coords can scan a plain dict (not a Table).
_RA_ALIASES = tables.RA_ALIASES
_DEC_ALIASES = tables.DEC_ALIASES
_NAME_ALIASES = ("main_id", "name", "designation", "objectId", "Satellite", "source_id")


def _find_coord_cols(tab: Table) -> tuple[str, str]:
    """Identify RA/Dec columns or raise a helpful error."""
    try:
        return tables.find_coord_columns(tab)
    except KeyError as e:
        raise ValueError(str(e))


def _payload_coords(payload: dict) -> tuple[float, float] | None:
    """RA/Dec from a row-detail payload dict, or None if it isn't sky-shaped."""
    ra_key = next((k for k in _RA_ALIASES if k in payload), None)
    dec_key = next((k for k in _DEC_ALIASES if k in payload), None)
    if not ra_key or not dec_key:
        return None
    try:
        ra, dec = float(payload[ra_key]), float(payload[dec_key])
    except (TypeError, ValueError):
        return None
    if ra != ra or dec != dec or not (-90.0 <= dec <= 90.0):
        return None
    return ra % 360.0, dec


def _payload_name(payload: dict, ra: float, dec: float) -> str:
    """A displayable name for a row payload (falls back to its coordinates)."""
    for key in _NAME_ALIASES:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return f"field {ra:.4f} {dec:+.4f}"


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
    target: resolvers.ResolvedTarget | None = None
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

    def __init__(self, settings: dict, target: resolvers.ResolvedTarget | None = None):
        super().__init__()
        self._s = dict(settings)
        self._target = target

    def _coverage_note(self) -> str:
        if self._target is None:
            return "[dim]Resolve a target to see coverage-aware product advice.[/]"
        survey = str(self._s.get("survey", "auto"))
        if survey not in cutouts.COLOR_SURVEYS:
            survey = "auto"
        note = resolvers.image_coverage_note(self._target.dec, survey)
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
    """The command reference — generated from commands.COMMANDS so it can't drift."""

    BINDINGS = [("escape", "dismiss", "Close"), ("enter", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        width = max(len(usage) for usage, _ in commands.help_rows())
        cmd_rows = "\n".join(
            f" [green]{escape(usage.ljust(width))}[/]  [dim]{escape(desc)}[/]"
            for usage, desc in commands.help_rows())
        with Vertical(id="modal-box"):
            yield Static(
                "[b cyan]CELESTRIUM[/] — a three-wing astrophysics instrument\n"
                "[dim]One prompt, many idioms: a bare name resolves; "
                "everything else is a command.[/]\n\n"

                f"━━━━━━━━━ [b]COMMANDS[/] ━━━━━━━━━━━━━━━━━━━━━\n{cmd_rows}\n\n"

                "━━━━━━━━━ [b]PRODUCT SCOPES[/] ━━━━━━━━━━━━━━━━\n"
                " [b]Target[/] products belong to the resolved object.\n"
                " [b]Field[/] products belong to the sky position or blank field.\n"
                " [b]Global[/] feeds are live streams, never mixed into Resolve plans.\n\n"

                "━━━━━━━━━ [b]SHORTCUTS[/] ━━━━━━━━━━━━━━━━━━━━━\n"
                "  [b]Enter[/] Act on row      [b]Ctrl+S[/] Save List     [b]Ctrl+O[/] Open Artifact\n"
                "  [b]Ctrl+G[/] Planner Cfg    [b]Ctrl+R[/] Refresh       [b]F5[/] Re-run Action\n"
                "  [b]Ctrl+T[/] Theme          [b]Ctrl+D[/] Log + Debug   [b]Ctrl+H[/] Focus Trail\n"
                "  [b]F2[/] Next Solid         [b]F3[/] Toggle Spin       [b]Tab[/] Cycle Archive\n"
                "  [b]Esc[/] Focus Table       [b]i[/] or [b]/[/] Focus Prompt\n\n"

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
    # show=True bindings surface in the Footer; the rest stay chord-only (F1 lists all).
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear", show=False),
        Binding("ctrl+s", "save_candidate", "Save list"),
        Binding("ctrl+o", "open_image", "Open artifact"),
        Binding("ctrl+g", "image_settings", "Planner cfg"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("ctrl+d", "toggle_debug", "Log+debug"),
        Binding("ctrl+h", "toggle_trail", "Trail", show=False),
        Binding("f1", "about", "Help"),
        Binding("f2", "next_solid", "Solid", show=False),
        Binding("f3", "toggle_spin", "Spin", show=False),
        Binding("f5", "rerun", "Re-run", show=False),
        Binding("ctrl+b", "cite_paper", "Cite→refs.bib"),
        Binding("ctrl+e", "export_table", "Export CSV"),
        Binding("enter", "row_action", "Open row", show=False),
        # Mode switch fallbacks (chord-only; slash commands are the front door)
        Binding("alt+r", "mode_resolve", "Mode Resolve", show=False),
        Binding("alt+l", "mode_literature", "Mode Lit", show=False),
        Binding("alt+q", "mode_query", "Mode Query", show=False),
        Binding("alt+x", "mode_crossmatch", "Mode X-match", show=False),
        Binding("alt+g", "mode_global", "Mode Global", show=False),
        Binding("alt+c", "mode_candidates", "Mode Cand", show=False),
        Binding("alt+b", "mode_runbooks", "Mode Runbooks", show=False),
        Binding("alt+h", "mode_history", "Mode Hist", show=False),
    ]

    def __init__(self, active_line: str = "G-Euclid DR1 prep"):
        super().__init__()
        self.active_line = active_line
        self.mode = "resolve"
        self.archive = "gaia"           # active archive for bare-SELECT queries
        self.last_table = None          # astropy Table from the last query/crossmatch
        self.last_image = None          # Path of the last rendered colour preview
        self.last_report = None         # Path of the last runbook index report
        self.last_target = None         # resolvers.ResolvedTarget from the last Resolve
        self.last_plans = []            # ranked ObservationPlans for last_target
        # Named request sequences guard EVERY worker against stale commits:
        # @work(exclusive=True) cancels *future* calls, but an in-flight thread
        # can still finish late and call back with old data. Sinks: "resolve"
        # (identity+plans), "fetch" (product results), "table" (query/crossmatch/
        # feed/cached commits to last_table), "literature" (ADS results).
        self._seq: dict[str, int] = {}
        self.debug_mode = False
        self._ads_ok = False            # checked once on mount, not per redraw
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
        yield Static("", id="context-bar")

        with Horizontal(id="body"):
            with Vertical(id="canvas"):
                yield Wireframe(id="orrery-idle")
                yield DataTable(id="results", zebra_stripes=True)

            with Vertical(id="side"):
                with VerticalScroll(id="detail-scroll"):
                    yield Static("Resolve a target or type /help.", id="detail")
                yield Label("◇ TRAIL", classes="heading")
                yield ListView(id="trail-list")
                yield Wireframe(id="orrery")  # ID is "#orrery" so test pilot functions

        yield RichLog(id="status", max_lines=40, wrap=True, markup=True)
        with Horizontal(id="controls"):
            yield Label("❯ ", id="prompt-prefix")
            yield Input(placeholder="object · 'RA Dec' · /query · match <cat> · /help",
                        id="entry",
                        suggester=SuggestFromList(commands.suggestions(),
                                                  case_sensitive=False))
        yield Footer()

    def on_mount(self) -> None:
        saved_theme = config.get("theme")
        self.theme = saved_theme if saved_theme in THEME_RING else "ansi-dark"
        self._ads_ok = _ads_token_ok()
        self.query_one("#results", DataTable).cursor_type = "row"
        self._refresh_subtitle()
        self._log(f"[b]Celestrium[/] ready — {len(config.ARCHIVES)} archives, "
                  f"{len(config.SAMPLE_RECIPES)} recipes.  [dim]F1 for keys[/]")
        self.query_one("#entry", Input).focus()

        # Idle state: the big orrery holds the canvas until the first action
        self.query_one("#orrery-idle", Wireframe).display = True
        self.query_one("#results", DataTable).display = False
        self.query_one("#side", Vertical).display = False

    # ----- helpers ---------------------------------------------------------- #
    def _bump(self, name: str) -> int:
        """Start a new request in a named sequence; returns its token.

        Call on the UI thread when *queuing* a worker, pass the token through,
        and gate the acceptor with `_current` — a stale in-flight thread then
        commits nothing instead of overwriting newer state.
        """
        self._seq[name] = self._seq.get(name, 0) + 1
        return self._seq[name]

    def _current(self, name: str, seq: int) -> bool:
        return self._seq.get(name, 0) == seq

    def _refresh_subtitle(self) -> None:
        """Sync the window subtitle and the context bar with session state.

        The context bar is the one always-visible answer to "what am I
        operating on?" — view, target, retained table, archive, ADS status.
        """
        plain_token = "ADS ok" if self._ads_ok else "ADS missing"
        plain_dbg = " DEBUG" if self.debug_mode else ""
        self.sub_title = f"active: {self.active_line}   {plain_token}{plain_dbg}"

        parts = [f"[b]{escape(self.mode.upper())}[/]"]
        target = self.context.target
        if target is not None:
            chip = (f"[b cyan]{escape(target.display_name)}[/] "
                    f"[dim]{escape(target.object_class)}[/]")
            if target.ra == target.ra:  # NaN-safe (the Sun has no fixed RA)
                chip += f" [dim]RA {target.ra:.4f} Dec {target.dec:+.4f}[/]"
            parts.append(chip)
        if self.last_table is not None and len(self.last_table):
            parts.append(f"table [b]{len(self.last_table)}r × "
                         f"{len(self.last_table.colnames)}c[/]")
        parts.append(f"archive [b]{escape(self.archive)}[/]")
        spark = ""
        if spark:
            parts.append(f"cache [dim]{escape(spark)}[/]")
        parts.append("[green]ADS ✓[/]" if self._ads_ok else "[yellow]ADS –[/]")
        if self.debug_mode:
            parts.append("[magenta]DBG[/]")
        try:
            self.query_one("#context-bar", Static).update(
                "  [dim]·[/]  ".join(parts))
        except NoMatches:
            pass

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#status", RichLog).write(msg)
        except NoMatches:
            pass

    def _feedback(self, stage: str, msg: str, style: str = "cyan") -> None:
        self._log(f"[{style}]{escape(stage)}[/] {escape(msg)}")

    def _dbg(self, msg: str) -> None:
        if self.debug_mode:
            self._log(f"[magenta]· {msg}[/]")

    def _set_detail(self, text: str) -> None:
        """The one detail sink: everything renders in the side panel."""
        try:
            self.query_one("#detail", Static).update(text)
        except NoMatches:
            pass

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

    def _resolve_ambient_scene(self, target: resolvers.ResolvedTarget | None):
        """The default Resolve-mode scene: a sky-position mini-map centred on
        the target's real coordinates (the roadmap's 'ASCII sky-position
        mini-map', zero network) — or a plain solid when there's no valid
        position yet (no target, or a NaN-RA/Dec special case like the Sun).
        A subsequent transit/ephemeris product fetch overrides this with the
        real TransitScene/OrbitScene once there's actual data to show."""
        if target is not None and target.ra == target.ra and target.dec == target.dec:
            return SkyMarkScene(target.ra, target.dec)
        return PlatonicScene("dodecahedron")

    # ----- mode switching --------------------------------------------------- #
    def switch_to_mode(self, mode: str) -> None:
        self.mode = mode

        prefix = f"{self.archive} ❯ " if mode == "query" else "❯ "
        self.query_one("#prompt-prefix", Label).update(prefix)
        self.query_one("#entry", Input).placeholder = _PROMPTS.get(mode, "enter command or object")

        # Leave idle: canvas shows results, side panel carries detail + trail
        self.query_one("#orrery-idle", Wireframe).display = False
        self.query_one("#side", Vertical).display = True

        results = self.query_one("#results", DataTable)
        results.display = True
        results.show_header = (mode != "resolve")
        self._refresh_subtitle()

        # Set ambient scene based on mode
        orrery = self.query_one("#orrery", Wireframe)
        if mode == "query":
            orrery.scene = SkyScatterScene()
        elif mode == "resolve":
            orrery.scene = self._resolve_ambient_scene(self.context.target)
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
        except NoMatches:
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
        elif mode == "sweep":
            self.trigger_sweep(data["target"])
        elif mode == "watch":
            self.trigger_watch(data["arg"])

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
        """Prompt dispatcher: commands.parse decides, the app's workers execute."""
        intent = commands.parse(text, mode=self.mode)
        args = intent.args

        if intent.kind == "empty":
            return
        if intent.kind == "clear":
            self.action_clear()
            return
        if intent.kind == "help":
            self.action_about()
            return
        if intent.kind == "mode":
            self.switch_to_mode(args["mode"])
            if "usage" in args:
                self._log(f"[dim]usage:[/] {escape(args['usage'])}")
            if "feed" in args:
                self.do_global_feed(args["feed"], refresh=True)
            return
        if intent.kind == "egg":
            self._show_egg(args["word"], args["markup"])
            return
        if intent.kind == "unknown":
            self._log(f"[yellow]Unknown command:[/] {escape(args['command'])}\n"
                      "[dim]Type /help for available commands.[/]")
            return

        self._log(f"[dim]{self.mode}:[/] {escape(text)}")

        if intent.kind == "resolve":
            self.switch_to_mode("resolve")
            self._last_action = ("resolve", args["target"])
            self.do_resolve(args["target"])
        elif intent.kind == "query":
            # archive=None means "whatever archive is active" (bare SELECT)
            self.archive = args["archive"] or self.archive or "gaia"
            self.switch_to_mode("query")
            self._last_action = ("query", self.archive, args["adql"], False)
            self.do_query(self.archive, args["adql"], False)
        elif intent.kind == "crossmatch":
            self.switch_to_mode("crossmatch")
            self._last_action = ("crossmatch", args["catalog"], False)
            self.do_crossmatch(args["catalog"], False)
        elif intent.kind == "runbook":
            self.switch_to_mode("runbooks")
            self.do_runbook(args["name"])
        elif intent.kind == "feed":
            self.switch_to_mode("global")
            self.do_global_feed(args["key"], refresh=True)
        elif intent.kind == "literature":
            self.switch_to_mode("literature")
            self._last_action = ("literature", args["query"])
            self.do_literature(args["query"])
        elif intent.kind == "papers_context":
            if self.context.target:
                query = f'"{self.context.target.display_name}"'
                self.switch_to_mode("literature")
                self._last_action = ("literature", query)
                self.do_literature(query)
            else:
                self._log("[yellow]No active target for papers search — specify a query[/]")
        elif intent.kind == "product":
            self._fetch_product_word(args["word"])
        elif intent.kind == "dossier":
            self.trigger_dossier(args.get("target"))
        elif intent.kind == "field":
            self.trigger_field(args.get("position"))
        elif intent.kind == "poster":
            self.trigger_poster(args.get("target"))
        elif intent.kind == "plot":
            self.trigger_plot(args.get("x"), args.get("y"), args.get("kind", "scatter"))
        elif intent.kind == "sweep":
            self.trigger_sweep(args.get("target"))
        elif intent.kind == "watch":
            self.trigger_watch(args.get("arg"))

    def _for_each_orrery(self, fn) -> None:
        """Apply fn(widget) to both Wireframe widgets in one guarded place —
        replaces the historical pattern of a direct call on #orrery plus a
        hand-wrapped try/except mirroring it onto #orrery-idle at every
        call site (F2/F3/easter-egg all did this independently)."""
        for widget_id in ("#orrery", "#orrery-idle"):
            try:
                fn(self.query_one(widget_id, Wireframe))
            except NoMatches:
                pass

    def _show_egg(self, word: str, markup: str) -> None:
        self.query_one("#side", Vertical).display = True
        self._set_detail(markup)
        if word in ("elite", "thargoid"):
            self._for_each_orrery(lambda w: w.set_solid("icosahedron"))

    def _fetch_product_word(self, word: str) -> None:
        """'image' / 'cutout' / 'panel' / 'spectrum' fetch that product for the
        active target, if a matching ranked plan exists."""
        if not self.context.target:
            self._log("[yellow]No active target to fetch products for[/]")
            return
        plans = self.last_plans or resolvers.recommend_plans(self.context.target)
        self.last_plans = plans

        wanted = {"spectrum": "spectrum", "panel": "multi_panel",
                  "image": "colour_image", "cutout": "colour_image"}[word]
        matched_idx = next(
            (idx for idx, p in enumerate(plans) if wanted in p.product.modalities),
            -1)
        if matched_idx != -1:
            self.switch_to_mode("resolve")
            self.execute_plan(matched_idx)
        else:
            self._log(f"[yellow]Product {word} not available/executable for "
                      f"{escape(self.context.target.display_name)}[/]")

    # ----- key / focus overrides ------------------------------------------- #
    def on_key(self, event) -> None:
        if event.key == "escape":
            entry = self.query_one("#entry", Input)
            if self.focused == entry:
                entry.blur()
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
                        self.archive = nxt
                        self.query_one("#prompt-prefix", Label).update(f"{nxt} ❯ ")
                        self._refresh_subtitle()
                        event.prevent_default()
                        event.stop()
                elif val.lower().startswith("select"):
                    entry.value = f"{self.archive}: {val}"
                    entry.cursor_position = len(entry.value)
                    self.query_one("#prompt-prefix", Label).update(f"{self.archive} ❯ ")
                    event.prevent_default()
                    event.stop()

    # ----- simple actions --------------------------------------------------- #
    def action_clear(self) -> None:
        self.query_one("#entry", Input).value = ""
        self._fill_table([], [])
        self._set_detail("Resolve a target or type /help.")
        self.query_one("#orrery-idle", Wireframe).display = True
        self.query_one("#results", DataTable).display = False
        self.query_one("#side", Vertical).display = False

        # Clear session context as well to prevent stale state bugs
        self.last_table = None
        self.last_target = None
        self.last_plans = []
        self.context.target = None
        self.context.dataset = None
        for name in ("resolve", "fetch", "table", "literature", "plot", "sweep", "watch"):
            self._bump(name)
        self._refresh_subtitle()

        self._log("Canvas and session state cleared.")

    def action_about(self) -> None:
        self.push_screen(AboutScreen())

    def action_next_solid(self) -> None:
        name = self.query_one("#orrery", Wireframe).next_solid()
        self._for_each_orrery(lambda w: w.set_solid(name))
        self._dbg(f"orrery → {name}")

    def action_toggle_spin(self) -> None:
        spinning = self.query_one("#orrery", Wireframe).toggle()
        # Set both to the SAME final value rather than toggling the idle
        # widget independently — correct even if the two had drifted apart.
        self._for_each_orrery(lambda w: setattr(w, "spinning", spinning))
        self._dbg(f"orrery spin {'on' if spinning else 'off'}")

    def action_cycle_theme(self) -> None:
        cur = self.theme if self.theme in THEME_RING else THEME_RING[0]
        nxt = THEME_RING[(THEME_RING.index(cur) + 1) % len(THEME_RING)]
        self.theme = nxt
        config.set("theme", nxt)
        self._log(f"theme → [b]{nxt}[/]")

    def action_toggle_debug(self) -> None:
        """One toggle for both: verbose tracing and an expanded log panel."""
        self.debug_mode = not self.debug_mode
        try:
            self.query_one("#status", RichLog).styles.height = (
                10 if self.debug_mode else 3)
        except NoMatches:
            pass
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

    def action_cite_paper(self) -> None:
        """Append the highlighted paper's canonical BibTeX to refs.bib (deduped)."""
        if self.mode != "literature":
            self._log("[yellow]cite works on a highlighted paper row — "
                      "run a papers search first[/]")
            return
        i = self._current_row_index()
        if i is None or i >= len(self._row_payloads):
            return
        bibcode = str(self._row_payloads[i].get("bibcode", "") or "")
        if bibcode:
            self.do_cite(bibcode)

    @work(thread=True, exclusive=True)
    def do_cite(self, bibcode: str) -> None:
        try:
            from .. import ads
            self.call_from_thread(
                self._feedback, "cite", f"exporting BibTeX for {bibcode}")
            n = ads.add_to_refs([bibcode])
            if n:
                self.call_from_thread(
                    self._feedback, "cite", f"+{n} new entry -> refs.bib", "green")
            else:
                self.call_from_thread(
                    self._feedback, "cite", f"{bibcode} already in refs.bib", "yellow")
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"cite failed: {type(e).__name__}: {str(e)} (ADS token?)", "red")

    def action_export_table(self) -> None:
        """Write the retained table to data/exports/ as CSV; Ctrl+O opens it."""
        if self.last_table is None or len(self.last_table) == 0:
            self._log("[yellow]nothing to export — run a Query or Crossmatch first[/]")
            return
        try:
            paths.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stem = candidates.slug(f"tui-{self.mode}-{len(self.last_table)}r")
            out = paths.EXPORTS_DIR / f"{stem}.csv"
            self.last_table.write(out, format="ascii.csv", overwrite=True)
            self.last_image = out  # the "last artifact" — Ctrl+O opens it
            self._feedback("export",
                           f"{len(self.last_table)} rows -> {out}", "green")
        except Exception as e:
            self._feedback("error",
                           f"export failed: {type(e).__name__}: {str(e)}", "red")

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

        self.push_screen(SaveCandidateScreen(default=candidates.slug(origin)), _save)

    # ----- list/grid actions ------------------------------------------------ #
    def action_history(self) -> None:
        records = [{"utc": r.timestamp, "archive": "run", "nrows": "-", "hash": r.id, "query": r.cap} for r in __import__("celestrium.core.ledger", fromlist=["ledger"]).ledger.list_runs()]
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
        feeds = [
            {"key": "neo", "label": "JPL CNEOS close approaches", "status": "executable",
             "description": "Near-Earth asteroids and close approaches (JPL Horizons/CNEOS)"},
            {"key": "satellite", "label": "Celestrak visible satellites (TLEs)", "status": "executable",
             "description": "Earth orbiters with active two-line elements"},
            {"key": "transient", "label": "ALeRCE/ZTF transient alerts", "status": "executable",
             "description": "Recent optical transient events from Zwicky Transient Facility"},
        ]
        rows = []
        payloads = []
        for cap in feeds:
            action = "[b green]Run[/]" if cap["status"] == "executable" else "[dim]Planned[/]"
            rows.append((action, cap["label"], cap["status"], cap["description"]))
            payloads.append(cap)
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
        specs = config.RUNBOOKS
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
        key = commands.feed_key(str(rec.get("key", "")))
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

    def _open_paper(self) -> None:
        """Enter on a paper row opens its ADS abstract page in the browser."""
        i = self._current_row_index()
        if i is None or i >= len(self._row_payloads):
            return
        bibcode = str(self._row_payloads[i].get("bibcode", "") or "")
        if not bibcode:
            return
        url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
        try:
            open_path(url)
            self._log(f"[dim]opened ADS[/] {escape(bibcode)}")
        except Exception as e:
            self._log(f"[red]open failed: {escape(type(e).__name__)}[/]")

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
        elif self.mode == "literature":
            self._open_paper()
        # other table modes: the highlighted row already renders in the side panel

    def on_data_table_row_selected(self, event) -> None:
        """Enter/click on a result row should run the row's contextual action."""
        table = getattr(event, "data_table", None) or getattr(event, "control", None)
        if getattr(table, "id", None) != "results":
            return
        self._activate_current_row()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Contextual detail: the highlighted row expands in the side panel."""
        try:
            i = self.query_one("#results", DataTable).get_row_index(event.row_key)
        except (NoMatches, RowDoesNotExist):
            return  # widget unmounted, or the grid moved on before we got here
        if self._row_render and 0 <= i < len(self._row_payloads):
            self._set_detail(self._row_render(self._row_payloads[i]))

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
            except NoMatches:
                pass

    def _target_key(self, target: resolvers.ResolvedTarget | None) -> str | None:
        if target is None:
            return None
        return (
            f"{target.display_name}|{target.ra:.8f}|{target.dec:.8f}|"
            f"{target.object_class}|{target.match_kind}"
        )

    def _fetch_is_current(self, seq: int, target_key: str | None) -> bool:
        if not self._current("fetch", seq):
            return False
        return target_key is None or target_key == self._target_key(self.last_target)

    def _accept_resolve_result(self, seq: int, target: resolvers.ResolvedTarget | None, plans: list) -> None:
        """Apply state changes and update UI on the main thread."""
        if not self._current("resolve", seq):
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

        orrery = self.query_one("#orrery", Wireframe)
        orrery.scene = self._resolve_ambient_scene(target)

        self.last_plans = plans
        self._feedback(
            "resolve",
            f"{target.display_name}: {target.match_kind}, confidence {target.confidence:.2f}, "
            f"{len(plans)} ranked products",
            "green")

    def _accept_literature_result(self, docs: list, query_text: str, seq: int) -> None:
        """Apply literature search result on the UI thread."""
        if not self._current("literature", seq):
            return
        rows = [(str(d.get("bibcode", "")), str(d.get("year", "")),
                 (lambda d: d.get("title", [""])[0])(d)[:50]) for d in docs]
        self._fill_table(["bibcode", "year", "title"], rows, docs, self._render_paper_detail)
        self._set_detail(f"{len(docs)} papers for:\n{escape(query_text)}\n"
                          "[dim]highlight a row for authors + abstract[/]")
        self.add_trail(query_text[:12], "literature", {"query": query_text})

    def _accept_query_result(self, tab: Table, archive: str, adql: str, seq: int) -> None:
        """Apply ADQL query result on the UI thread."""
        if not self._current("table", seq):
            return
        self._accept_table_result(tab)
        self._set_detail(f"[b]{escape(archive)}[/] → {len(tab)} rows (cached + logged).\n"
                          "[dim]Crossmatch[/] matches these · [dim]Ctrl+S[/] saves · "
                          "[dim]highlight a row[/] for full values")
        self.add_trail(f"{archive}:{len(tab)}r", "query", {"archive": archive, "query": adql})

    def _accept_crossmatch_result(self, out: Table, catalog: str, key: str, seq: int) -> None:
        """Apply crossmatch result on the UI thread."""
        if not self._current("table", seq):
            return
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

    def _apply_transit_scene(self, tab: Table) -> None:
        """Parametrize TransitScene from the first row's real predicted depth
        and duration (exoplanet.predict_transits); silently no-ops (keeps
        whatever scene was already showing) if the values are unavailable —
        e.g. 'Unknown'/'No ephemeris' rows for a system with no light-curve fit."""
        try:
            depth_pct = float(str(tab["Depth"][0]).split()[0])
            duration_h = float(str(tab["Duration"][0]).split()[0])
        except (KeyError, IndexError, ValueError):
            return
        try:
            self.query_one("#orrery", Wireframe).scene = TransitScene(
                depth_pct=depth_pct, duration_hours=duration_h)
        except NoMatches:
            pass

    def _apply_orbit_scene(self, tab: Table) -> None:
        """Draw the real 30-day Horizons RA/Dec track instead of a placeholder."""
        try:
            ra_col, dec_col = tables.find_coord_columns(tab)
            points = [(float(tab[ra_col][i]), float(tab[dec_col][i]))
                     for i in range(len(tab))]
        except (KeyError, TypeError, ValueError):
            return
        if len(points) < 2:
            return
        try:
            self.query_one("#orrery", Wireframe).scene = OrbitScene(points)
        except NoMatches:
            pass

    def _accept_tabular_product_result(self, tab: Table, target_name: str,
                                       plan_label: str, detail: str, seq: int,
                                       target_key: str | None = None,
                                       product_key: str | None = None) -> None:
        if not self._fetch_is_current(seq, target_key):
            return
        self.switch_to_mode("query")
        self._accept_table_result(tab)
        if product_key == "transit":
            self._apply_transit_scene(tab)
        elif product_key == "ephemeris":
            self._apply_orbit_scene(tab)
        self._feedback("product", f"{plan_label}: loaded {len(tab)} rows for {target_name}", "green")
        self._set_detail(
            f"{detail}\n\n[green]Loaded {len(tab)} rows into Query view from "
            f"{escape(plan_label)} for {escape(target_name)}.[/]"
        )

    def _accept_product_result(self, result: __import__("celestrium.core.artifact", fromlist=["Artifact"]).Artifact | None,
                               plan_label: str, detail: str, seq: int,
                               target_key: str | None = None,
                               product_key: str | None = None) -> None:
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
                    seq, target_key, product_key)
        else:
            self._accept_plan_no_records(target_name, plan_label, detail, seq, target_key)

    def _accept_runbook_result(self, index: Path, name: str, steps_count: int, created_count: int) -> None:
        """Central runbook state committer on the main thread."""
        self.last_report = index
        self._set_detail(f"[b green]runbook {name} done[/]\n{steps_count} steps · "
                          f"{created_count} artifacts\n[dim]{escape(str(index))}[/]\n[b]Ctrl+O[/] open index")
        self._log(f"[green]runbook {name} → {index}[/]")

    def _accept_global_feed_result(self, tab: Table, key: str, label: str, seq: int) -> None:
        if not self._current("table", seq):
            return
        self._last_action = ("global", key)
        self.switch_to_mode("query")
        self._accept_table_result(tab)
        self._feedback("global", f"{label}: loaded {len(tab)} rows", "green")
        self._set_detail(
            f"[b green]Global feed loaded[/]\n{escape(label)} · {len(tab)} rows\n"
            "[dim]This is a live/global stream, not a target or field product.[/]"
        )
        self.add_trail(f"feed:{key}", "global", {"key": key})

    def _accept_plot_result(self, path: Path, x: str | None, y: str | None,
                            kind: str, seq: int) -> None:
        if not self._current("plot", seq):
            return
        self.last_image = path
        label = f"{kind}: {x or 'ra'}" + ("" if kind == "hist" else f" vs {y or 'dec'}")
        self._feedback("plot", f"{label} -> {path}", "green")
        self._set_detail(f"[b green]plot ready[/] {escape(label)}\n"
                         f"[dim]{escape(str(path))}[/]\n[b]Ctrl+O[/] open")

    def _render_sweep_detail(self, entry: dict) -> str:
        rows = "n/a" if entry.get("rows") is None else str(entry["rows"])
        return (f"[b]{escape(str(entry.get('label', entry.get('key', '?'))))}[/]\n"
                f"status {escape(str(entry.get('status', '?')))}  rows {escape(rows)}\n"
                f"{escape(str(entry.get('note', '')))}")

    def _accept_sweep_result(self, pkt: Any, seq: int) -> None:
        if not self._current("sweep", seq):
            return
        self.switch_to_mode("sweep")
        rows = []
        for entry in pkt.entries:
            count = "" if entry.get("rows") is None else str(entry["rows"])
            rows.append((entry.get("label", entry.get("key", "?")),
                         entry.get("status", ""), count,
                         str(entry.get("note", ""))[:80]))
        self._fill_table(["archive", "status", "rows", "note"], rows,
                         pkt.entries, self._render_sweep_detail)
        name = str(pkt.target.get("display_name", "?"))
        self._set_detail(escape(pkt.to_markdown()))
        self._feedback("sweep", f"{name}: {pkt.hits}/{len(pkt.entries)} hits", "green")
        self.add_trail(f"sweep:{name[:12]}", "sweep", {"target": name})

    def _render_watch_detail(self, entry) -> str:
        return (f"[b]{escape(entry.label)}[/]\n"
                f"RA {entry.ra:.5f}  Dec {entry.dec:+.5f}\n"
                f"status {escape(entry.status)}  alerts {entry.nrows}\n"
                f"{escape(entry.note)}")

    def _accept_watch_result(self, pkt: Any, seq: int) -> None:
        if not self._current("watch", seq):
            return
        self.switch_to_mode("watch")
        rows = [(e.label, e.status, str(e.nrows), e.note[:80]) for e in pkt.entries]
        self._fill_table(["position", "status", "alerts", "note"], rows,
                         pkt.entries, self._render_watch_detail)
        self._set_detail(escape(pkt.to_markdown()))
        self._feedback("watch", f"{pkt.label}: {pkt.total_alerts} alert(s) "
                                f"across {len(pkt.entries)} position(s)", "green")
        self.add_trail(f"watch:{pkt.label[:12]}", "watch", {"arg": pkt.label})

    # ----- workers (threaded; blocking calls off the UI loop) --------------- #
    def trigger_plot(self, x: str | None, y: str | None, kind: str = "scatter") -> None:
        if self.last_table is None or len(self.last_table) == 0:
            self._log("[yellow]no retained table — run a query, feed, product, or candidate first[/]")
            return
        if kind not in plots.KINDS:
            self._log(f"[yellow]unknown plot kind {escape(kind)}[/]")
            return
        if kind != "sky" and not x:
            self._log("[yellow]usage: /plot <x> [y] [scatter|hist|sky|cmd][/]")
            return
        if kind not in ("hist", "sky") and not y:
            self._log(f"[yellow]{escape(kind)} plot needs X and Y columns[/]")
            return
        self._feedback("plot", f"queued {kind} plot")
        self.do_plot(x, y, kind)

    def do_plot(self, x: str | None, y: str | None, kind: str) -> None:
        self._run_plot_worker(self.last_table, x, y, kind, self._bump("plot"))

    @work(thread=True, exclusive=True)
    def _run_plot_worker(self, tab: Table, x: str | None, y: str | None,
                         kind: str, seq: int) -> None:
        try:
            path = plots.plot_table(tab, x=x, y=y, kind=kind)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"plot failed: {type(e).__name__}: {str(e)}", "red")
            return
        self.call_from_thread(self._accept_plot_result, path, x, y, kind, seq)

    def trigger_sweep(self, target: str | None) -> None:
        text = (target or "").strip()
        target_arg = text or self.context.target
        if target_arg is None:
            self._log("[yellow]no target — /sweep <target>, or resolve one first[/]")
            return
        label = text or self.context.target.display_name
        self.switch_to_mode("sweep")
        self._feedback("sweep", f"queued archive sweep for {label}")
        self.do_sweep(target_arg)

    def do_sweep(self, target) -> None:
        self._run_sweep_worker(target, self._bump("sweep"))

    @work(thread=True, exclusive=True)
    def _run_sweep_worker(self, target, seq: int) -> None:
        try:
            pkt = (lambda *args, **kwargs: type("Obj", (), {"entries": [], "to_dict": lambda self: {}, "to_markdown": lambda self: ""})())(
                target,
                on_note=lambda m: self.call_from_thread(self._feedback, "sweep", m),
            )
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"sweep failed: {type(e).__name__}: {str(e)}", "red")
            return
        self.call_from_thread(self._accept_sweep_result, pkt, seq)

    def trigger_watch(self, arg: str | None) -> None:
        text = (arg or "").strip()
        if not text and self.context.target is None:
            self._log("[yellow]no target — /watch <target|list>, or resolve one first[/]")
            return
        # A bare word that names a saved candidate list wins over treating it
        # as a target name — watch's whole point is chasing your shortlist.
        list_name = text if text and candidates.find_record(text) else None
        target_arg = None if list_name else (text or self.context.target)
        label = list_name or (text or self.context.target.display_name)
        self.switch_to_mode("watch")
        self._feedback("watch", f"queued alert watch for {label}")
        self.do_watch(target_arg, list_name)

    def do_watch(self, target, list_name: str | None) -> None:
        self._run_watch_worker(target, list_name, self._bump("watch"))

    @work(thread=True, exclusive=True)
    def _run_watch_worker(self, target, list_name: str | None, seq: int) -> None:
        try:
            pkt = (lambda *args, **kwargs: type("Obj", (), {"entries": [], "to_dict": lambda self: {}, "to_markdown": lambda self: ""})())(
                target=target, candidate_list=list_name,
                on_note=lambda m: self.call_from_thread(self._feedback, "watch", m),
            )
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"watch failed: {type(e).__name__}: {str(e)}", "red")
            return
        self.call_from_thread(self._accept_watch_result, pkt, seq)

    def do_resolve(self, name: str) -> None:
        """Resolve trigger on the UI thread to prevent sequence race."""
        seq = self._bump("resolve")
        self._bump("fetch")
        # A late-landing sweep/watch for the *previous* target would silently
        # flip the mode back over whatever Resolve is about to show.
        self._bump("sweep")
        self._bump("watch")
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
            target = resolvers.resolve_target(name)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"resolve failed: {type(e).__name__}: {str(e)}", "red")
            return

        # Thread scheduling race check
        if not self._current("resolve", seq):
            return

        plans = []
        if target is not None:
            self.call_from_thread(
                self._feedback, "resolve",
                f"ranking product capabilities for {target.display_name}")
            plans = resolvers.recommend_plans(target)

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
        for plan in resolvers.recommend_plans(target):
            mark = "▶" if plan.next_action == "fetch" else "·"
            lines.append(f"{mark} {escape(plan.product.label)} [{escape(plan.product.status)}]")
        lines += ["", resolvers.image_coverage_note(
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
        seq = self._bump("fetch")
        # A product fetch is the newest table intent too: invalidate any older
        # in-flight query/crossmatch so it can't overwrite the product's table.
        self._bump("table")
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
            result = resolvers.execute_product(
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
            self._accept_product_result, result, plan.product.label, card, seq,
            target_key, plan.product.key)

    def do_literature(self, query_text: str) -> None:
        self._run_literature_worker(query_text, self._bump("literature"))

    @work(thread=True, exclusive=True)
    def _run_literature_worker(self, query_text: str, seq: int) -> None:
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
            self.call_from_thread(self._accept_literature_result, docs, query_text, seq)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"ADS failed: {type(e).__name__}: {str(e)} (ADS token?)", "red")

    def _render_paper_detail(self, doc: dict) -> str:
        authors = doc.get("author") or []
        alist = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        abstract = doc.get("abstract") or "[dim](no abstract)[/]"
        cites = doc.get("citation_count", "—")
        return (f"[b]{escape((lambda d: d.get("title", [""])[0])(doc))}[/]\n\n[cyan]{escape(alist)}[/]\n"
                f"[dim]{doc.get('year', '?')} · {escape(doc.get('bibcode', ''))} · "
                f"{cites} cites[/]\n\n{escape(str(abstract))}")

    def do_global_feed(self, key: str, refresh: bool = True) -> None:
        self._run_global_feed_worker(key, refresh, self._bump("table"))

    @work(thread=True, exclusive=True)
    def _run_global_feed_worker(self, key: str, refresh: bool, seq: int) -> None:
        feed_key = commands.feed_key(key)
        self.call_from_thread(
            self._feedback, "global",
            f"feed request {feed_key}; refresh={refresh}")
        if feed_key not in config.GLOBAL_FEED_EXECUTORS:
            self.call_from_thread(
                self._feedback, "error",
                f"unknown global feed {key!r}", "red")
            return

        labels = {
            "neo": "JPL CNEOS close approaches",
            "satellite": "Celestrak visible satellites (TLEs)",
            "transient": "ALeRCE/ZTF transient alerts",
        }
        label = labels.get(feed_key, feed_key)

        try:
            from ..core.kernel import Kernel
            k = Kernel()
            art = k.run(f"feed.{feed_key}", refresh=refresh)
            tab = k.load(art.id)
            if len(tab) == 0:
                self.call_from_thread(
                    self._feedback, "global",
                    f"{label}: no rows returned", "yellow")
                self.call_from_thread(
                    self._set_detail,
                    f"[b cyan]{escape(label)}[/]\n[yellow]No rows returned.[/]\n"
                    "[dim]The feed may be unavailable or returned 0 entries.[/]"
                )
                return
            self.call_from_thread(self._accept_global_feed_result, tab, feed_key, label, seq)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"global feed failed: {type(e).__name__}: {str(e)}", "red"
            )

    def do_query(self, archive: str, adql: str, refresh: bool) -> None:
        self._run_query_worker(archive, adql, refresh, self._bump("table"))

    @work(thread=True, exclusive=True)
    def _run_query_worker(self, archive: str, adql: str, refresh: bool, seq: int) -> None:
        try:
            self.call_from_thread(
                self._feedback, "query",
                f"{archive} request; refresh={refresh}; {len(adql)} characters")
            source = config.resolve_query_source(archive)
            if source is None:
                self.call_from_thread(
                    self._feedback, "error",
                    f"unknown archive {archive!r}", "red")
                return
            self.call_from_thread(
                self._feedback, "query",
                f"source={source.__class__.__name__}; cache={'refresh' if refresh else 'normal'}")
            tab = __import__("celestrium.core.kernel", fromlist=["Kernel"]).Kernel().load(Kernel().run("archive.query", archive=archive, adql=adql).id)
            self.call_from_thread(
                self._feedback, "query",
                f"{archive} returned {len(tab)} rows", "green")
            self.call_from_thread(self._accept_query_result, tab, archive, adql, seq)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"query failed: {type(e).__name__}: {str(e)}", "red")

    def do_crossmatch(self, catalog: str, refresh: bool) -> None:
        self._run_crossmatch_worker(catalog, refresh, self._bump("table"))

    @work(thread=True, exclusive=True)
    def _run_crossmatch_worker(self, catalog: str, refresh: bool, seq: int) -> None:
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
                f"columns={ra}/{dec}; radius=5 arcsec; matching {catalog}")
            out = xmatch.match(local, cat2=catalog, ra=ra, dec=dec, radius_arcsec=5.0)
            self.call_from_thread(
                self._feedback, "crossmatch",
                f"{catalog} returned {len(out)} matches", "green")

            # Extract key for trail recovery
            key = hashlib.sha1(f"xmatch\n{tag}".encode()).hexdigest()[:16]
            self.call_from_thread(self._accept_crossmatch_result, out, catalog, key, seq)
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"crossmatch failed: {type(e).__name__}: {str(e)}", "red")

    def do_open_cached(self, hash_prefix: str) -> None:
        self._run_open_cached_worker(hash_prefix, self._bump("table"))

    @work(thread=True, exclusive=True)
    def _run_open_cached_worker(self, hash_prefix: str, seq: int) -> None:
        try:
            self.call_from_thread(
                self._feedback, "history",
                f"opening cached table {hash_prefix}")
            tab = __import__("celestrium.core.kernel", fromlist=["Kernel"]).Kernel().load(hash_prefix)
            if not self._current("table", seq):
                return
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

    def do_rerun(self, hash_prefix: str) -> None:
        self._run_rerun_worker(hash_prefix, self._bump("table"))

    @work(thread=True, exclusive=True)
    def _run_rerun_worker(self, hash_prefix: str, seq: int) -> None:
        try:
            self.call_from_thread(
                self._feedback, "history",
                f"re-running cached query {hash_prefix}")
            rec = None
            if rec is None:
                self.call_from_thread(
                    self._feedback, "error",
                    f"no record for {hash_prefix}", "red")
                return
            source = config.resolve_query_source(rec["archive"])
            if source is None:
                self.call_from_thread(
                    self._feedback, "error",
                    f"archive {rec['archive']!r} not re-runnable", "red")
                return
            tab = None # rerun not supported via old tui
            if not self._current("table", seq):
                return
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

    # ----- imaging & recreation (the third wing, in the cockpit) ------------ #
    def _accept_report_result(self, kind: str, name: str, path) -> None:
        """Shared committer for imaging-wing artifacts (dossier/field/poster)."""
        self.last_image = path
        self._feedback(kind, f"{name} -> {path}", "green")
        self._set_detail(f"[b green]{escape(kind)} ready[/] — [b]{escape(str(name))}[/]\n"
                         f"[dim]{escape(str(path))}[/]\n[b]Ctrl+O[/] open")

    def trigger_dossier(self, target: str | None) -> None:
        name = (target or "").strip() or (
            self.context.target.display_name if self.context.target else "")
        if not name:
            self._log("[yellow]no target — /dossier <name>, or resolve one first[/]")
            return
        self._feedback("dossier", f"queued object packet for {name}")
        self._set_detail(f"[b]Building dossier[/] {escape(name)}\n"
                         "[dim]identity · images · literature → Markdown report[/]")
        self.do_dossier(name)

    @work(thread=True, exclusive=True)
    def do_dossier(self, name: str) -> None:
        try:
            from ..core.kernel import Kernel
            k = Kernel()
            art = k.run("object.dossier", {"target": name})
            path = art.full_path
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"dossier failed: {type(e).__name__}: {str(e)}", "red")
            return
        self.call_from_thread(self._accept_report_result, "dossier", name, path)

    def trigger_field(self, position: str | None) -> None:
        coords = None
        if position:
            coords = resolvers.parse_request(position).coordinates
            if coords is None:
                self._log(f"[yellow]could not parse a position from "
                          f"{escape(position)} — try '/field 187.70 12.39'[/]")
                return
        elif self.context.target is not None:
            t = self.context.target
            if t.ra == t.ra and t.dec == t.dec:  # NaN-safe
                coords = (t.ra, t.dec)
        if coords is None:
            self._log("[yellow]no position — /field <RA Dec>, or resolve a target first[/]")
            return
        ra, dec = coords
        self._feedback("field", f"queued field packet for {ra:.5f} {dec:+.5f}")
        self.do_field(ra, dec)

    @work(thread=True, exclusive=True)
    def do_field(self, ra: float, dec: float) -> None:
        try:
            from ..core.kernel import Kernel
            k = Kernel()
            art = k.run("object.field", {"target": f"{ra:.5f} {dec:+.5f}"})
            path = art.full_path
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"field packet failed: {type(e).__name__}: {str(e)}", "red")
            return
        self.call_from_thread(self._accept_report_result, "field",
                              f"{ra:.5f} {dec:+.5f}", path)

    def trigger_poster(self, target: str | None) -> None:
        """Poster of an explicit target, else the highlighted row, else the active target."""
        target = (target or "").strip()
        if target:
            self._feedback("poster", f"queued poster for {target}")
            self.do_poster(target, None, None, None)
            return
        i = self._current_row_index()
        if (i is not None and 0 <= i < len(self._row_payloads)
                and isinstance(self._row_payloads[i], dict)):
            payload = self._row_payloads[i]
            coords = _payload_coords(payload)
            if coords:
                ra, dec = coords
                name = _payload_name(payload, ra, dec)
                self._feedback("poster", f"queued poster for highlighted row: {name}")
                self.do_poster(None, name, ra, dec)
                return
        t = self.context.target
        if t is not None and t.ra == t.ra:
            self._feedback("poster", f"queued poster for {t.display_name}")
            self.do_poster(None, t.display_name, t.ra, t.dec, t.otype)
            return
        self._log("[yellow]nothing to poster — /poster <target>, highlight a row "
                  "with RA/Dec, or resolve a target[/]")

    @work(thread=True, exclusive=True)
    def do_poster(self, target: str | None, name: str | None,
                  ra: float | None, dec: float | None, otype: str = "") -> None:
        try:
            from ..core.kernel import Kernel
            k = Kernel()
            tgt_name = target or name or f"{ra} {dec}"
            fov = self.image_cfg.get("fov", 8.0)
            if not isinstance(fov, (int, float)):
                fov = 8.0
            art = k.run("imaging.poster", {"target": tgt_name, "fov": fov, "width": 1920, "height": 1080})
            path = art.full_path
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"poster failed: {type(e).__name__}: {str(e)}", "red")
            return
        self.call_from_thread(self._accept_report_result, "poster", str(tgt_name), path)

    @work(thread=True, exclusive=True)
    def do_runbook(self, name: str) -> None:
        """Run a configured study/runbook with live per-step progress, then save its index."""
        try:
            self.call_from_thread(
                self._feedback, "runbook",
                f"starting study {name}")
            from ..core.kernel import Kernel
            k = Kernel()
            art = k.run("study.run", {"study": name})
            paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            index = paths.REPORTS_DIR / f"runbook-{name}.md"
            meta = art.meta
            summary = (
                f"# Study Run: {name}\n\n"
                f"- **Status:** {meta.get('succeeded', 0)}/{meta.get('combinations', 0)} succeeded\n"
                f"- **Metric:** {meta.get('metric', 'N/A')}\n"
                f"- **Preregistration Hash:** `{meta.get('prereg', 'N/A')}`\n"
                f"- **Artifact ID:** `{art.id}`\n"
            )
            index.write_text(summary, encoding="utf-8")
            self.call_from_thread(self._accept_runbook_result, index, name, 1, meta.get("succeeded", 0))
        except Exception as e:
            self.call_from_thread(
                self._feedback, "error",
                f"runbook failed: {type(e).__name__}: {str(e)}", "red")


def main() -> None:
    CelestriumApp().run()


if __name__ == "__main__":
    main()
