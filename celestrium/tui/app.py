"""The Celestrium cockpit — a Textual app over registry + packets + cache.

One brain, two surfaces: this imports `celestrium/*` exactly like `hub.py` does and
never the other way round. Phase 2 is read-only and async (blocking astroquery/ADS
calls run in threads so the UI stays live). Four modes:

  Resolve     object name  -> SIMBAD identity + recent papers
  Literature  ADS query    -> paper set
  Query       ADQL         -> rows from an archive, through the provenance cache
  History     (none)       -> the cached-query manifest

Phase 3 (docs/roadmap.md): inline-ish image preview (external open), interactive
crossmatch from selected rows -> candidate lists, runbook runner.
"""
from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (DataTable, Footer, Header, Input, Label, ListItem,
                             ListView, RichLog, Select, Static)

from .. import cache, cutouts, packets, registry

MODES = [
    ("Resolve  ·  object → identity + papers", "resolve"),
    ("Literature  ·  ADS query → paper set", "literature"),
    ("Query  ·  ADQL → rows (cached)", "query"),
    ("History  ·  the provenance manifest", "history"),
]
_PROMPTS = {
    "resolve": "object name, e.g. M87",
    "literature": 'ADS query, e.g. abs:"cosmic dipole" year:2024-2026',
    "query": "ADQL, e.g. SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source",
    "history": "(History loads automatically — no input)",
}


def _ads_token_ok() -> bool:
    try:
        from .. import ads
        ads._token()
        return True
    except Exception:
        return False


class CelestriumApp(App):
    """Celestrium — a three-wing astrophysics instrument (TUI cockpit)."""

    CSS_PATH = "theme.tcss"
    TITLE = "Celestrium"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("f5", "rerun", "Re-run"),
    ]

    def __init__(self, active_line: str = "G-Euclid DR1 prep"):
        super().__init__()
        self.active_line = active_line
        self.mode = "resolve"

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
                    yield Select(list(registry.ARCHIVES.items()) if False else
                                 [(k, k) for k in registry.ARCHIVES],
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

    # ----- mode switching --------------------------------------------------- #
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        key = event.item.id.removeprefix("m-")
        self.mode = key
        self.query_one("#entry", Input).placeholder = _PROMPTS[key]
        self.query_one("#archive", Select).display = (key == "query")
        self._set_detail(f"[b]{key}[/] mode.")
        if key == "history":
            self.action_history()
        else:
            self.query_one("#entry", Input).focus()

    # ----- actions ---------------------------------------------------------- #
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self.mode == "history":
            self.action_history()
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

    def action_clear(self) -> None:
        self.query_one("#entry", Input).value = ""
        self._fill_table([], [])
        self._set_detail("Cleared.")

    def action_rerun(self) -> None:
        self.on_input_submitted(
            Input.Submitted(self.query_one("#entry", Input),
                            self.query_one("#entry", Input).value))

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
        self._set_detail(f"{len(records)} cached pulls. Select a row for the hash.")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.mode == "history":
            try:
                row = self.query_one("#results", DataTable).get_row(event.row_key)
                self._set_detail(f"[b]hash[/] {row[3]}\n[b]archive[/] {row[1]}\n"
                                 f"[b]query[/]\n{row[4]}")
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
            detail = (f"[b cyan]{row['main_id']}[/]\n[yellow]{row.get('otype', '?')}[/]\n"
                      f"RA {float(row['ra']):.5f}  Dec {float(row['dec']):+.5f}")
            self.call_from_thread(self._set_detail, detail)
            try:
                bib = packets.resolvers.bibliography(str(row["main_id"]), limit=12)
                rows = [(str(r["bibcode"]), str(r["title"])[:55]) for r in bib]
                self.call_from_thread(self._fill_table, ["bibcode", "title"], rows)
            except Exception as e:
                self.call_from_thread(self._log, f"[yellow]bibliography unavailable ({type(e).__name__})[/]")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")

    @work(thread=True, exclusive=True)
    def do_literature(self, query_text: str) -> None:
        try:
            ps = packets.build_paper_set(query_text, rows=15)
            rows = [(str(d.get("bibcode", "")), str(d.get("year", "")),
                     str(d.get("title", ["?"])[0])[:50]) for d in ps.docs]
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
            cols = list(tab.colnames)[:8]
            rows = [tuple(r[c] for c in cols) for r in tab[:50]]
            self.call_from_thread(self._fill_table, cols, rows)
            self.call_from_thread(self._set_detail,
                                  f"[b]{archive}[/] → {len(tab)} rows (cached + logged).")
        except Exception as e:
            self.call_from_thread(self._log, f"[red]{type(e).__name__}: {e}[/]")


def main() -> None:
    CelestriumApp().run()


if __name__ == "__main__":
    main()
