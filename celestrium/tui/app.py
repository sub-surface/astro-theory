"""The Celestrium cockpit — a Textual app over registry + packets + cache.

One brain, two surfaces: this imports `celestrium/*` exactly like `hub.py` does and
never the other way round. Blocking astroquery/ADS calls run in threads so the UI
stays live. Modes:

  Resolve     object name  -> SIMBAD identity + recent papers + colour-image preview
  Literature  ADS query    -> paper set
  Query       ADQL         -> rows from an archive, through the provenance cache
  Crossmatch  catalog id   -> match the current result table to a VizieR catalogue
  Candidates  (none)       -> saved candidate lists; pick one to load its rows
  History     (none)       -> the cached-query manifest; open / re-run a past pull

Phase 3 (docs/roadmap.md) is the interactive leap: the current result table is kept
in `self.last_table`, so Crossmatch matches it against a catalogue and Ctrl+S saves
the rows as a named candidate list — the desk loop, made interactive. Image preview
follows the decided design: survey + FOV + saved thumbnail *path* with an [Open]
action (`os.startfile`), no sixel/kitty dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             ListItem, ListView, RichLog, Select, Static)

from .. import cache, candidates, cutouts, packets, registry, xmatch

MODES = [
    ("Resolve  ·  object → identity + papers + image", "resolve"),
    ("Literature  ·  ADS query → paper set", "literature"),
    ("Query  ·  ADQL → rows (cached)", "query"),
    ("Crossmatch  ·  current rows × catalogue", "crossmatch"),
    ("Candidates  ·  saved short-lists", "candidates"),
    ("History  ·  the provenance manifest", "history"),
]
_PROMPTS = {
    "resolve": "object name, e.g. M87",
    "literature": 'ADS query, e.g. abs:"cosmic dipole" year:2024-2026',
    "query": "ADQL, e.g. SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source",
    "crossmatch": "catalogue id, e.g. vizier:VIII/65/nvss  (matches the current table)",
    "candidates": "(Candidates load automatically — select a row to open it)",
    "history": "(History loads automatically — no input)",
}
_NO_INPUT = {"candidates", "history"}


def _ads_token_ok() -> bool:
    try:
        from .. import ads
        ads._token()
        return True
    except Exception:
        return False


class SaveCandidateScreen(ModalScreen):
    """Prompt for a name, then save the app's current table as a candidate list."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, default: str = "candidates"):
        super().__init__()
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical(id="save-box"):
            yield Label("Save current rows as candidate list", classes="heading")
            yield Input(value=self._default, placeholder="list name", id="save-name")
            with Horizontal(id="save-actions"):
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


class CelestriumApp(App):
    """Celestrium — a three-wing astrophysics instrument (TUI cockpit)."""

    CSS_PATH = "theme.tcss"
    TITLE = "Celestrium"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+s", "save_candidate", "Save list"),
        ("ctrl+o", "open_image", "Open img"),
        ("enter", "row_action", "Open row"),
        ("f5", "rerun", "Re-run"),
    ]

    def __init__(self, active_line: str = "G-Euclid DR1 prep"):
        super().__init__()
        self.active_line = active_line
        self.mode = "resolve"
        self.last_table = None      # astropy Table from the last query/crossmatch
        self.last_image = None      # Path of the last rendered colour preview

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
                yield Static("Pick a mode and submit.", id="detail")
        yield RichLog(id="status", max_lines=8, wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        token = "[green]ADS ✓[/]" if _ads_token_ok() else "[yellow]ADS –[/]"
        self.sub_title = f"active: {self.active_line}   {token}"
        self.query_one("#archive", Select).display = False
        self.query_one("#results", DataTable).cursor_type = "row"
        self._log(f"[b]Celestrium[/] ready — {len(registry.ARCHIVES)} archives, "
                  f"{len(registry.SAMPLE_RECIPES)} recipes.")
        self.query_one("#entry", Input).focus()

    # ----- helpers ---------------------------------------------------------- #
    def _log(self, msg: str) -> None:
        self.query_one("#status", RichLog).write(msg)

    def _set_detail(self, text: str) -> None:
        self.query_one("#detail", Static).update(text)

    def _fill_table(self, columns, rows) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        if columns:
            table.add_columns(*columns)
        for row in rows:
            table.add_row(*[str(c)[:60] for c in row])

    def _show_astropy(self, tab, ncols: int = 8, nrows: int = 50) -> None:
        """Render an astropy Table into the results grid and retain it (Phase 3)."""
        self.last_table = tab
        cols = list(tab.colnames)[:ncols]
        rows = [tuple(r[c] for c in cols) for r in tab[:nrows]]
        self._fill_table(cols, rows)

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

    # ----- actions ---------------------------------------------------------- #
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self.mode in _NO_INPUT:
            return
        if not text:
            return
        self._log(f"[dim]{self.mode}:[/] {text}")
        if self.mode == "resolve":
            self.do_resolve(text)
        elif self.mode == "literature":
            self.do_literature(text)
        elif self.mode == "query":
            self.do_query(self.query_one("#archive", Select).value, text)
        elif self.mode == "crossmatch":
            self.do_crossmatch(text)

    def action_clear(self) -> None:
        self.query_one("#entry", Input).value = ""
        self._fill_table([], [])
        self._set_detail("Cleared.")

    def action_rerun(self) -> None:
        if self.mode == "history":
            self._history_row_action(rerun=True)
            return
        self.on_input_submitted(
            Input.Submitted(self.query_one("#entry", Input),
                            self.query_one("#entry", Input).value))

    def action_row_action(self) -> None:
        """Enter on a result row: context-dependent open."""
        if self.mode == "history":
            self._history_row_action(rerun=False)
        elif self.mode == "candidates":
            self._open_candidate()

    def action_open_image(self) -> None:
        """Open the last rendered colour preview with the OS image viewer."""
        if not self.last_image:
            self._log("[yellow]no image rendered yet (resolve an object first)[/]")
            return
        try:
            os.startfile(self.last_image)  # win32 (Phase 3 decision: external open)
        except AttributeError:  # non-Windows fallback
            self._log(f"[dim]open manually:[/] {self.last_image}")
        except Exception as e:
            self._log(f"[red]open failed: {type(e).__name__}[/]")

    def action_save_candidate(self) -> None:
        """Persist the current result table as a named candidate list (Phase 3)."""
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

    def action_history(self) -> None:
        records = cache.manifest()
        if not records:
            self._fill_table(["(empty)"], [])
            self._set_detail("No cached queries yet — run a Query.")
            return
        rows = [(r.get("utc", ""), r.get("archive", ""), r.get("nrows", ""),
                 r.get("hash", ""), str(r.get("query", ""))[:60])
                for r in records[-50:][::-1]]
        self._fill_table(["utc", "archive", "rows", "hash", "query"], rows)
        self._set_detail(f"{len(records)} cached pulls.\n"
                         "[dim]Enter[/] open cached rows · [dim]F5[/] re-run query")

    def action_candidates(self) -> None:
        records = candidates.latest()
        if not records:
            self._fill_table(["(empty)"], [])
            self._set_detail("No candidate lists yet.\n"
                             "Run a Crossmatch, then [dim]Ctrl+S[/] to save one.")
            return
        rows = [(r.get("name", ""), r.get("nrows", ""), str(r.get("origin", ""))[:40],
                 r.get("utc", "")) for r in records]
        self._fill_table(["name", "rows", "origin", "utc"], rows)
        self._set_detail(f"{len(records)} candidate lists.\n[dim]Enter[/] loads the rows.")

    # ----- row-driven actions ---------------------------------------------- #
    def _current_row(self):
        table = self.query_one("#results", DataTable)
        if table.row_count == 0:
            return None
        try:
            return table.get_row_at(table.cursor_row)
        except Exception:
            return None

    def _history_row_action(self, *, rerun: bool) -> None:
        row = self._current_row()
        if not row or len(row) < 4:
            return
        h = str(row[3])
        if rerun:
            self._log(f"[dim]re-running[/] {h}")
            self.do_rerun(h)
        else:
            self._log(f"[dim]opening cached[/] {h}")
            self.do_open_cached(h)

    def _open_candidate(self) -> None:
        row = self._current_row()
        if not row:
            return
        name = str(row[0])
        try:
            tab = candidates.load(name)
            self._show_astropy(tab)
            self._set_detail(f"[b]{name}[/] — {len(tab)} candidate rows loaded.\n"
                             "[dim]Crossmatch[/] mode can now match these.")
            self._log(f"[green]loaded candidate list {name} ({len(tab)} rows)[/]")
        except Exception as e:
            self._log(f"[red]{type(e).__name__}: {e}[/]")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.mode == "history":
            try:
                row = self.query_one("#results", DataTable).get_row(event.row_key)
                self._set_detail(f"[b]hash[/] {row[3]}\n[b]archive[/] {row[1]}\n"
                                 f"[b]query[/]\n{row[4]}\n\n"
                                 "[dim]Enter[/] open · [dim]F5[/] re-run")
            except Exception:
                pass

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
                self.call_from_thread(self._fill_table, ["bibcode", "title"], rows)
            except Exception as e:
                self.call_from_thread(self._log, f"[yellow]bibliography unavailable ({type(e).__name__})[/]")
            # Phase 3 image preview: render a colour thumbnail, show survey + path.
            try:
                fov = packets.default_fov(str(row.get("otype", "")), 8.0, 3.0)
                hips, survey = cutouts.best_color_hips(dec)
                path = cutouts.color(ra, dec, fov_arcmin=fov, hips=hips)
                self.last_image = path
                self.call_from_thread(
                    self._set_detail,
                    f"{detail}\n\n[b]image[/] {survey}\nFOV {fov}'\n"
                    f"[dim]{path}[/]\n[b green]Ctrl+O[/] to open")
            except Exception as e:
                self.call_from_thread(self._log, f"[yellow]image unavailable ({type(e).__name__})[/]")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    @work(thread=True, exclusive=True)
    def do_literature(self, query_text: str) -> None:
        try:
            ps = packets.build_paper_set(query_text, rows=15)
            rows = [(str(d.get("bibcode", "")), str(d.get("year", "")),
                     packets.title_of(d)[:50]) for d in ps.docs]
            self.call_from_thread(self._fill_table, ["bibcode", "year", "title"], rows)
            self.call_from_thread(self._set_detail, f"{len(ps.docs)} papers for:\n{query_text}")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/] (ADS token?)")

    @work(thread=True, exclusive=True)
    def do_query(self, archive: str, adql: str) -> None:
        try:
            source = registry.resolve_query_source(archive)
            if source is None:
                self.call_from_thread(self._log, f"[red]unknown archive {archive!r}[/]")
                return
            tab = cache.cached_query(archive, adql, lambda: source.query(adql))
            self.call_from_thread(self._show_astropy, tab)
            self.call_from_thread(self._set_detail,
                                  f"[b]{archive}[/] → {len(tab)} rows (cached + logged).\n"
                                  "[dim]Crossmatch[/] mode matches these · [dim]Ctrl+S[/] saves")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    @work(thread=True, exclusive=True)
    def do_crossmatch(self, catalog: str) -> None:
        """Match the retained result table against a VizieR catalogue (Phase 3)."""
        if self.last_table is None or len(self.last_table) == 0:
            self.call_from_thread(self._log,
                                  "[yellow]no table to match — run a Query or open a Candidate first[/]")
            return
        local = self.last_table
        ra = "ra" if "ra" in local.colnames else local.colnames[1]
        dec = "dec" if "dec" in local.colnames else local.colnames[2]
        try:
            tag = f"tui|{catalog}|{len(local)}rows"
            out = cache.cached_query(
                "xmatch", tag,
                lambda: xmatch.match(local, cat2=catalog, ra=ra, dec=dec, radius_arcsec=5.0))
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
