"""The Celestrium cockpit — a Textual app over registry + packets + cache.

One brain, two surfaces: this imports `celestrium/*` exactly like `hub.py` does and
never the other way round. Blocking astroquery/ADS calls run in threads so the UI
stays live. Modes:

  Resolve     object name  -> SIMBAD identity + papers + colour-image preview
  Literature  ADS query    -> paper set (highlight a row for full author/abstract)
  Query       ADQL         -> rows from an archive, through the provenance cache
  Crossmatch  catalog id   -> match the current result table to a VizieR catalogue
  Candidates  (none)       -> saved candidate lists; pick one to load its rows
  History     (none)       -> the cached-query manifest; open / re-run a past pull

Phase 3 made the desk loop interactive (retain the table, crossmatch it, Ctrl+S a
candidate list). This pass adds the cockpit's comforts: a theme system that actually
recolours (ansi-dark by default), debug + refresh-cache controls, an ELITE-style
rotating-wireframe panel under the detail pane, a contextual detail panel that
expands the highlighted row, an image-settings modal (FOV/pixels/survey) with a
no-coverage fallback so big galaxies stop rendering blank — and a few easter eggs.
"""
from __future__ import annotations

import os

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             ListItem, ListView, RichLog, Select, Static)

from .. import cache, candidates, cutouts, packets, registry, xmatch
from .wireframe import Wireframe

MODES = [
    ("Resolve  ·  object → identity + papers + image", "resolve"),
    ("Literature  ·  ADS query → paper set", "literature"),
    ("Query  ·  ADQL → rows (cached)", "query"),
    ("Crossmatch  ·  current rows × catalogue", "crossmatch"),
    ("Candidates  ·  saved short-lists", "candidates"),
    ("History  ·  the provenance manifest", "history"),
]
_PROMPTS = {
    "resolve": "object name, e.g. M87  ·  Ctrl+G for image settings",
    "literature": 'ADS query, e.g. abs:"cosmic dipole" year:2024-2026',
    "query": "ADQL, e.g. SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source",
    "crossmatch": "catalogue id, e.g. vizier:VIII/65/nvss  (matches the current table)",
    "candidates": "(Candidates load automatically — select a row to open it)",
    "history": "(History loads automatically — no input)",
}
_NO_INPUT = {"candidates", "history"}
# A small curated theme ring for Ctrl+T (ansi-dark first = the default).
THEME_RING = ["ansi-dark", "tokyo-night", "nord", "gruvbox", "dracula",
              "catppuccin-mocha", "monokai", "textual-dark"]

# Easter eggs: magic words typed in Resolve mode (before hitting SIMBAD).
_EGGS = {
    "42": "[b gold1]42[/] — the Answer to the Ultimate Question of Life, "
          "the Universe, and Everything. (Now find the Question.)",
    "elite": "[b green]RIGHT ON, COMMANDER![/] Wireframe drive engaged.",
    "thargoid": "[b green]⚠ THARGOID DETECTED[/] — raise shields, deploy E.C.M.",
    "xyzzy": "[dim]Nothing happens.[/]",
    "tea": "[b]Share and Enjoy.[/] ☕  (the Sirius Cybernetics Corp. thanks you)",
    "cake": "[yellow]The cake is a lie.[/]",
}


def _ads_token_ok() -> bool:
    try:
        from .. import ads
        ads._token()
        return True
    except Exception:
        return False


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
    """Set colour-cutout parameters (FOV / pixels / survey) used by Resolve."""

    BINDINGS = [("escape", "dismiss", "Cancel")]
    SURVEYS = ["auto", "legacy", "panstarrs", "des", "dss2"]
    PRODUCTS = ["colour_image", "multi_panel", "spectrum", "metadata", "exoplanet"]
    WAVELENGTHS = ["auto", "UV", "optical", "near-IR", "mid-IR", "radio", "X-ray"]

    def __init__(self, settings: dict):
        super().__init__()
        self._s = dict(settings)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("◇ IMAGE SETTINGS", classes="heading")
            yield Label("Product", classes="dim")
            yield Select([(p, p) for p in self.PRODUCTS],
                         value=self._s.get("product", "colour_image"),
                         id="img-product", allow_blank=False)
            yield Label("Wavelength", classes="dim")
            yield Select([(w, w) for w in self.WAVELENGTHS],
                         value=self._s.get("wavelength", "auto"),
                         id="img-wavelength", allow_blank=False)
            yield Label("Field of view (arcmin, or 'auto')", classes="dim")
            yield Input(value=str(self._s.get("fov", "auto")), id="img-fov")
            yield Label("Pixels (per side)", classes="dim")
            yield Input(value=str(self._s.get("pix", 512)), id="img-pix")
            yield Label("Survey", classes="dim")
            yield Select([(s, s) for s in self.SURVEYS], value=self._s.get("survey", "auto"),
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
        self.dismiss({"fov": fov, "pix": pix,
                      "survey": self.query_one("#img-survey", Select).value,
                      "product": self.query_one("#img-product", Select).value,
                      "wavelength": self.query_one("#img-wavelength", Select).value})

    def action_dismiss(self) -> None:
        self.dismiss(None)


class AboutScreen(ModalScreen):
    """A tiny about/help card — also where `elite` lands."""

    BINDINGS = [("escape", "dismiss", "Close"), ("enter", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(
                "[b cyan]CELESTRIUM[/] — a three-wing astrophysics instrument\n"
                "[dim]theory · validation · imaging[/]\n\n"
                "[b]Keys[/]\n"
                "  Ctrl+S save list   Ctrl+O open image   Ctrl+G image settings\n"
                "  Ctrl+R refresh cache   F5 re-run   Ctrl+T theme   Ctrl+D debug\n"
                "  F2 next solid   F3 spin on/off   Enter open highlighted row\n\n"
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
        ("f1", "about", "About"),
        ("f2", "next_solid", "Solid"),
        ("f3", "toggle_spin", "Spin"),
        ("f5", "rerun", "Re-run"),
        ("enter", "row_action", "Open row"),
    ]

    def __init__(self, active_line: str = "G-Euclid DR1 prep"):
        super().__init__()
        self.active_line = active_line
        self.mode = "resolve"
        self.last_table = None          # astropy Table from the last query/crossmatch
        self.last_image = None          # Path of the last rendered colour preview
        self.debug_mode = False
        self.image_cfg = {
            "fov": "auto", "pix": 512, "survey": "auto",
            "product": "colour_image", "wavelength": "auto",
        }
        self._last_action = None        # (fn_name, *args) for refresh / re-run
        self._row_payloads = []         # parallel to displayed rows (contextual detail)
        self._row_render = None         # callable(payload) -> detail markup

    # ----- layout ----------------------------------------------------------- #
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("◇ WINGS", classes="heading")
                yield Static("Theory · Validation · Imaging", classes="dim")
                yield Label("◇ MODE", classes="heading")
                yield ListView(
                    *[ListItem(Label(label), id=f"m-{key}") for label, key in MODES],
                    id="modes",
                )
            with Vertical(id="main"):
                with Horizontal(id="controls"):
                    yield Select([(k, k) for k in registry.ARCHIVES],
                                 prompt="archive", id="archive", value="gaia")
                    yield Input(placeholder=_PROMPTS["resolve"], id="entry")
                yield DataTable(id="results", zebra_stripes=True)
            with Vertical(id="preview"):
                yield Label("◇ DETAIL", classes="heading")
                with VerticalScroll(id="detail-scroll"):
                    yield Static("Pick a mode and submit.", id="detail")
                yield Label("◇ ORRERY  ·  F2 solid · F3 spin", classes="heading")
                yield Wireframe(id="orrery")
        yield RichLog(id="status", max_lines=8, wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "ansi-dark"  # the requested default
        self.query_one("#archive", Select).display = False
        self.query_one("#results", DataTable).cursor_type = "row"
        self._refresh_subtitle()
        self._log(f"[b]Celestrium[/] ready — {len(registry.ARCHIVES)} archives, "
                  f"{len(registry.SAMPLE_RECIPES)} recipes.  [dim]F1 for keys[/]")
        self.query_one("#entry", Input).focus()

    # ----- helpers ---------------------------------------------------------- #
    def _refresh_subtitle(self) -> None:
        token = "[green]ADS ✓[/]" if _ads_token_ok() else "[yellow]ADS –[/]"
        dbg = "  [magenta]DEBUG[/]" if self.debug_mode else ""
        self.sub_title = f"active: {self.active_line}   {token}{dbg}"

    def _log(self, msg: str) -> None:
        self.query_one("#status", RichLog).write(msg)

    def _dbg(self, msg: str) -> None:
        if self.debug_mode:
            self._log(f"[magenta]· {msg}[/]")

    def _set_detail(self, text: str) -> None:
        self.query_one("#detail", Static).update(text)

    def _fill_table(self, columns, rows, payloads=None, render=None) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        if columns:
            table.add_columns(*columns)
        for row in rows:
            table.add_row(*[str(c)[:60] for c in row])
        self._row_payloads = payloads or []
        self._row_render = render

    def _show_astropy(self, tab, ncols: int = 8, nrows: int = 50) -> None:
        """Render an astropy Table into the results grid and retain it (Phase 3)."""
        self.last_table = tab
        cols = list(tab.colnames)[:ncols]
        rows = [tuple(r[c] for c in cols) for r in tab[:nrows]]
        # payload per row: the full (untruncated) record for the detail panel
        payloads = [{c: tab[c][i] for c in tab.colnames} for i in range(min(len(tab), nrows))]
        self._fill_table(cols, rows, payloads, self._render_row_detail)

    def _render_row_detail(self, payload: dict) -> str:
        lines = ["[b]row[/]"]
        for k, v in payload.items():
            lines.append(f"[cyan]{k}[/] {v}")
        return "\n".join(lines)

    # ----- mode switching --------------------------------------------------- #
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        key = event.item.id.removeprefix("m-")
        self.mode = key
        self.query_one("#entry", Input).placeholder = _PROMPTS[key]
        self.query_one("#archive", Select).display = (key == "query")
        self._set_detail(f"[b]{key}[/] mode.")
        if key == "history":
            self.action_history()
        elif key == "candidates":
            self.action_candidates()
        else:
            self.query_one("#entry", Input).focus()

    # ----- input ------------------------------------------------------------ #
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self.mode in _NO_INPUT or not text:
            return
        # easter eggs first (Resolve mode only — they never hit the network)
        if self.mode == "resolve" and text.lower() in _EGGS:
            self._set_detail(_EGGS[text.lower()])
            if text.lower() in ("elite", "thargoid"):
                self.query_one("#orrery", Wireframe).set_solid("icosahedron")
                self.push_screen(AboutScreen())
            return
        self._log(f"[dim]{self.mode}:[/] {text}")
        if self.mode == "resolve":
            self._last_action = ("resolve", text)
            self.do_resolve(text)
        elif self.mode == "literature":
            self._last_action = ("literature", text)
            self.do_literature(text)
        elif self.mode == "query":
            arc = self.query_one("#archive", Select).value
            self._last_action = ("query", arc, text, False)
            self.do_query(arc, text, False)
        elif self.mode == "crossmatch":
            self._last_action = ("crossmatch", text, False)
            self.do_crossmatch(text, False)

    # ----- simple actions --------------------------------------------------- #
    def action_clear(self) -> None:
        self.query_one("#entry", Input).value = ""
        self._fill_table([], [])
        self._set_detail("Cleared.")

    def action_about(self) -> None:
        self.push_screen(AboutScreen())

    def action_next_solid(self) -> None:
        name = self.query_one("#orrery", Wireframe).next_solid()
        self._dbg(f"orrery → {name}")

    def action_toggle_spin(self) -> None:
        spinning = self.query_one("#orrery", Wireframe).toggle()
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
                self._log(f"image: FOV {cfg['fov']} · {cfg['pix']}px · {cfg['survey']}")
        self.push_screen(ImageSettingsScreen(self.image_cfg), _apply)

    def action_open_image(self) -> None:
        if not self.last_image:
            self._log("[yellow]no image rendered yet (resolve an object first)[/]")
            return
        try:
            os.startfile(self.last_image)  # win32 (Phase 3 decision: external open)
        except AttributeError:  # non-Windows fallback
            self._log(f"[dim]open manually:[/] {self.last_image}")
        except Exception as e:
            self._log(f"[red]open failed: {type(e).__name__}[/]")

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
        else:
            self._log("[dim]nothing cacheable to refresh[/]")

    def action_rerun(self) -> None:
        if self.mode == "history":
            self._history_row_action(rerun=True)
            return
        entry = self.query_one("#entry", Input)
        self.on_input_submitted(Input.Submitted(entry, entry.value))

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
                self._log(f"[red]save failed: {type(e).__name__}: {e}[/]")

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
            self._show_astropy(tab)
            self._set_detail(f"[b]{name}[/] — {len(tab)} candidate rows loaded.\n"
                             "[dim]Crossmatch[/] mode can now match these.")
            self._log(f"[green]loaded candidate list {name} ({len(tab)} rows)[/]")
        except Exception as e:
            self._log(f"[red]{type(e).__name__}: {e}[/]")

    def action_row_action(self) -> None:
        if self.mode == "history":
            self._history_row_action(rerun=False)
        elif self.mode == "candidates":
            self._open_candidate()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Contextual detail panel: expand the highlighted row."""
        try:
            i = self.query_one("#results", DataTable).get_row_index(event.row_key)
        except Exception:
            return
        if self._row_render and 0 <= i < len(self._row_payloads):
            self._set_detail(self._row_render(self._row_payloads[i]))

    # ----- workers (threaded; blocking calls off the UI loop) --------------- #
    @work(thread=True, exclusive=True)
    def do_resolve(self, name: str) -> None:
        try:
            info = packets.resolvers.identify(name)
            if info is None or len(info) == 0:
                self.call_from_thread(self._set_detail, f"No SIMBAD match for {name!r}.")
                return
            row = info[0]
            ra, dec = float(row["ra"]), float(row["dec"])
            detail = (f"[b cyan]{row['main_id']}[/]\n[yellow]{row.get('otype', '?')}[/]\n"
                      f"RA {ra:.5f}  Dec {dec:+.5f}")
            self.call_from_thread(self._set_detail, detail)
            try:
                bib = packets.resolvers.bibliography(str(row["main_id"]), limit=12)
                rows = [(str(r["bibcode"]), str(r["title"])[:55]) for r in bib]
                payloads = [{"bibcode": str(r["bibcode"]), "title": str(r["title"])}
                            for r in bib]
                self.call_from_thread(self._fill_table, ["bibcode", "title"], rows,
                                      payloads, self._render_bib_detail)
            except Exception as e:
                self.call_from_thread(self._log, f"[yellow]bibliography unavailable ({type(e).__name__})[/]")
            self._render_preview_image(name, row, ra, dec, detail)
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    def _render_bib_detail(self, payload: dict) -> str:
        return (f"[b]{payload.get('title', '')}[/]\n\n[cyan]bibcode[/] {payload.get('bibcode', '')}")

    def _render_preview_image(self, name, row, ra, dec, detail) -> None:
        """Phase-3 image preview with the new settings + no-coverage fallback."""
        cfg = self.image_cfg
        fov = cfg["fov"] if isinstance(cfg["fov"], (int, float)) \
            else packets.default_fov(str(row.get("otype", "")), 8.0, 3.0)
        survey = None if cfg["survey"] == "auto" else cfg["survey"]
        self.call_from_thread(self._dbg, f"image {name}: fov={fov} pix={cfg['pix']} survey={cfg['survey']}")
        try:
            path, label = cutouts.color_auto(ra, dec, fov_arcmin=fov,
                                              pix=cfg["pix"], survey=survey)
            self.last_image = path
            self.call_from_thread(
                self._set_detail,
                f"{detail}\n\n[b]image[/] {label}\nFOV {fov}'  ·  {cfg['pix']}px\n"
                f"[dim]{path}[/]\n[b green]Ctrl+O[/] open · [b]Ctrl+G[/] settings")
        except Exception as e:
            self.call_from_thread(self._log, f"[yellow]image unavailable ({type(e).__name__})[/]")

    @work(thread=True, exclusive=True)
    def do_literature(self, query_text: str) -> None:
        try:
            from .. import ads
            docs = ads.search(query_text, rows=15,
                              fl=("bibcode", "title", "author", "year",
                                  "citation_count", "abstract"))
            rows = [(str(d.get("bibcode", "")), str(d.get("year", "")),
                     packets.title_of(d)[:50]) for d in docs]
            self.call_from_thread(self._fill_table, ["bibcode", "year", "title"], rows,
                                  docs, self._render_paper_detail)
            self.call_from_thread(self._set_detail,
                                  f"{len(docs)} papers for:\n{query_text}\n"
                                  "[dim]highlight a row for authors + abstract[/]")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/] (ADS token?)")

    def _render_paper_detail(self, doc: dict) -> str:
        authors = doc.get("author") or []
        alist = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        abstract = doc.get("abstract") or "[dim](no abstract)[/]"
        cites = doc.get("citation_count", "—")
        return (f"[b]{packets.title_of(doc)}[/]\n\n[cyan]{alist}[/]\n"
                f"[dim]{doc.get('year', '?')} · {doc.get('bibcode', '')} · "
                f"{cites} cites[/]\n\n{abstract}")

    @work(thread=True, exclusive=True)
    def do_query(self, archive: str, adql: str, refresh: bool) -> None:
        try:
            source = registry.resolve_query_source(archive)
            if source is None:
                self.call_from_thread(self._log, f"[red]unknown archive {archive!r}[/]")
                return
            self.call_from_thread(self._dbg,
                                  f"query {archive} {'(refresh)' if refresh else ''}: {adql[:50]}…")
            tab = cache.cached_query(archive, adql, lambda: source.query(adql), refresh=refresh)
            self.call_from_thread(self._show_astropy, tab)
            self.call_from_thread(self._set_detail,
                                  f"[b]{archive}[/] → {len(tab)} rows (cached + logged).\n"
                                  "[dim]Crossmatch[/] matches these · [dim]Ctrl+S[/] saves · "
                                  "[dim]highlight a row[/] for full values")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    @work(thread=True, exclusive=True)
    def do_crossmatch(self, catalog: str, refresh: bool) -> None:
        if self.last_table is None or len(self.last_table) == 0:
            self.call_from_thread(self._log,
                                  "[yellow]no table to match — run a Query or open a Candidate first[/]")
            return
        local = self.last_table
        ra = "ra" if "ra" in local.colnames else local.colnames[1]
        dec = "dec" if "dec" in local.colnames else local.colnames[2]
        try:
            tag = f"tui|{catalog}|{len(local)}rows"
            self.call_from_thread(self._dbg, f"xmatch × {catalog} on {len(local)} rows")
            out = cache.cached_query(
                "xmatch", tag,
                lambda: xmatch.match(local, cat2=catalog, ra=ra, dec=dec, radius_arcsec=5.0),
                refresh=refresh)
            self.call_from_thread(self._show_astropy, out)
            self.call_from_thread(
                self._set_detail,
                f"[b]xmatch[/] × {catalog}\n{len(out)} matches (r≤5\")\n"
                "[b green]Ctrl+S[/] to save as candidates")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    @work(thread=True, exclusive=True)
    def do_open_cached(self, hash_prefix: str) -> None:
        try:
            tab = cache.load_cached(hash_prefix)
            self.call_from_thread(self._show_astropy, tab)
            self.call_from_thread(self._set_detail,
                                  f"cached [b]{hash_prefix}[/] → {len(tab)} rows loaded.\n"
                                  "[dim]Crossmatch[/]/[dim]Ctrl+S[/] now work on these.")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    @work(thread=True, exclusive=True)
    def do_rerun(self, hash_prefix: str) -> None:
        try:
            rec = cache.find_record(hash_prefix)
            if rec is None:
                self.call_from_thread(self._log, f"[red]no record for {hash_prefix}[/]")
                return
            source = registry.resolve_query_source(rec["archive"])
            if source is None:
                self.call_from_thread(self._log,
                                      f"[red]archive {rec['archive']!r} not re-runnable[/]")
                return
            tab = cache.cached_query(rec["archive"], rec["query"],
                                     lambda: source.query(rec["query"]), refresh=True)
            self.call_from_thread(self._show_astropy, tab)
            self.call_from_thread(self._set_detail,
                                  f"re-ran [b]{hash_prefix}[/] → {len(tab)} fresh rows.")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")


def main() -> None:
    CelestriumApp().run()


if __name__ == "__main__":
    main()
