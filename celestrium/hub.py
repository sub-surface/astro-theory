#!/usr/bin/env python
"""Celestrium CLI — the operable front-end to the astrophysics instrument.

Three wings, one tool: a **theory workshop** (literature in, ideas assessed), an
**experimental validation** bench (pull data slices, test the empirical claims),
and an **imaging & recreation** studio (scientific diagrams + beautiful renders).
A thin presenter: all real logic lives in celestrium/ (config.py = data,
ads/cutouts/resolvers… = primitives), so this CLI and the Textual TUI (tui/) share one brain.

    python -m celestrium --help
    python -m celestrium resolve M87
    python -m celestrium fetch M87 --product colour-image   # plan -> deliberate fetch
    python -m celestrium feed neos                          # global live feeds
    python -m celestrium export agn                         # any table-ref -> CSV
    python -m celestrium image 187.7059 12.3911
    python -m celestrium where 213.6906 -12.5801
    python -m celestrium cite 'abs:"cosmic dipole" year:2024-2026' --add
    python -m celestrium query gaia "SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source"
    python -m celestrium match gaia-bright-nearby vizier:VIII/65/nvss --radius 5 --save agn
    python -m celestrium candidates          # browse saved lists; NAME to show one
    python -m celestrium poster M87 --resolution 1080p --style label
    python -m celestrium runbook euclid-q1
    python -m celestrium atlas | toolbox     # pretty-print the repo maps

Global: add --json before any subcommand for machine-readable output.
"""
import json as _json
import re
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table as RichTable


def _ensure_utf8_console() -> None:
    """Windows consoles often default to cp1252, which can't encode the
    arrows/em-dashes in help text and Markdown docs (atlas/toolbox) — every
    such command then crashes with UnicodeEncodeError instead of printing.
    Reconfigure stdout/stderr to UTF-8 with a safe fallback; no-ops under
    Typer's CliRunner (its captured streams don't support `.reconfigure`)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

from . import (candidates, census as census_mod, config, cutouts, plots, resolvers, spectra, tables, xmatch)
# Re-exported so the test suite (and any importer) can patch these on `hub`.
from .config import QUERY_ARCHIVES, SAMPLE_RECIPES, SampleRecipe, ATLAS_TARGETS  # noqa: F401

# The three wings, surfaced as command groups in --help.
THEORY = "Theory workshop"
VALIDATION = "Experimental validation"
IMAGING = "Imaging & recreation"
WORKFLOWS = "Workflows & maps"

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help=("[bold cyan]Celestrium[/] — a three-wing astrophysics instrument: "
                        "a theory workshop, an experimental-validation bench, and an "
                        "imaging & recreation studio. Add [yellow]--json[/] for machine output."))
console = Console()
REPO = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO / "data" / "reports"
ATLAS_DIR = REPO / "data" / "atlas"
POSTERS_DIR = REPO / "data" / "posters"
_COORD_CONTEXT = {"ignore_unknown_options": True}
RESOLUTIONS = {"1080p": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160)}
_STATE = {"json": False}

# Thin aliases over the service layer (kept as module attrs for patchability).
_slug = candidates.slug


@app.callback()
def _main(json_out: bool = typer.Option(False, "--json", help="machine-readable output")):
    _STATE["json"] = json_out


# --------------------------------------------------------------------------- #
# Infrastructure shared across all commands
# --------------------------------------------------------------------------- #
def _emit(payload, render):
    """JSON-print payload in --json mode, else call render() for rich output."""
    if _STATE["json"]:
        console.print_json(_json.dumps(payload, default=str))
    else:
        render()


def _fail(err, hint: Optional[str] = None) -> typer.Exit:
    """Uniform error exit: structured JSON under --json, Rich red otherwise.

    Usage: ``raise _fail(e)`` / ``raise _fail("message", hint="try …")``.
    Under --json an agent gets ``{"error": …, "message": …}`` instead of Rich
    markup, so errors are parseable too."""
    name = type(err).__name__ if isinstance(err, BaseException) else "Error"
    if _STATE["json"]:
        payload = {"error": name, "message": str(err)}
        if hint:
            payload["hint"] = hint
        console.print_json(_json.dumps(payload, default=str))
    else:
        console.print(f"[red]{err}[/]")
        if hint:
            console.print(f"[dim]{hint}[/]")
    return typer.Exit(1)


def _write_report(kind: str, stem: str, body: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "-", stem.strip()).strip("-").lower() or kind
    path = REPORTS_DIR / f"{kind}-{safe}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def _rich_table_from_rows(title, columns, rows) -> RichTable:
    table = RichTable(*[str(c) for c in columns], title=title)
    for row in rows:
        table.add_row(*[str(row[c])[:80] for c in columns])
    return table


def _print_astropy_table(tab, title: str, limit: int = 12):
    columns = list(tab.colnames)[:8]
    console.print(f"[bold]{title}[/]")
    console.print(_rich_table_from_rows(title, columns, tab[:limit]))
    if len(tab) > limit:
        console.print(f"[dim]showing {limit} of {len(tab)} rows[/]")


def _table_payload(tab) -> dict:
    """Serialise an astropy Table for --json output (values via default=str)."""
    return {"nrows": len(tab), "columns": list(tab.colnames),
            "rows": [{c: row[c] for c in tab.colnames} for row in tab]}


def _contact_sheet(paths, out: Path, title: str):
    # Thin re-export so the test suite can still patch hub._contact_sheet; the
    # implementation lives in cutouts so the TUI runbook runner shares it.
    return cutouts.contact_sheet(paths, out, title)


def _open_path(path: Path) -> None:
    """Best-effort OS opener used only for explicit --open requests."""
    import os
    import platform
    import subprocess

    path_str = str(path)
    system = platform.system()
    if system == "Windows":
        os.startfile(path_str)
    elif system == "Darwin":
        subprocess.Popen(["open", path_str])
    else:
        subprocess.Popen(["xdg-open", path_str])


def _markdown_table(tab, title: str) -> str:
    lines = [f"# {title}", ""]
    cols = list(tab.colnames)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for row in tab:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _kernel():
    from .core import capability as capmod, kernel as kernelmod
    capmod.load_all()
    return kernelmod.default()


def _printer():
    """Live progress for interactive runs; silent under --json."""
    if _STATE["json"]:
        return None
    from .core import events as ev

    def listen(event) -> None:
        if event.type == ev.PROGRESS:
            console.print(f"  [dim]· {event.message}[/]")
        elif event.type == ev.STARTED:
            console.print(f"[cyan]▶ {event.cap}[/]")
        elif event.type == ev.DONE and event.cached:
            console.print(f"[green]✓ cached[/] [dim]{event.cap}[/]")
        elif event.type == ev.DONE:
            console.print(f"[green]✓[/] {event.cap} [dim]{event.ms} ms[/]")
        elif event.type == ev.FAILED:
            console.print(f"[red]✗ {event.cap}[/] {event.message}")

    return listen


def _run(cap: str, emit=None, **params):
    """Run a capability and return the artifact. Central dispatch for all commands."""
    return _kernel().run(cap, params, emit=emit or _printer())


def _load(artifact_id: str):
    """Load the payload of an artifact by id."""
    return _kernel().load(artifact_id)


# --------------------------------------------------------------------------- #
# Object / position
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=THEORY)
def resolve(name: str):
    """Identify an object (SIMBAD) and list recent papers about it."""
    try:
        art = _run("object.resolve", target=name)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)

    def render():
        console.print(f"[bold cyan]{payload.get('display_name', name)}[/]  "
                      f"[yellow]{payload.get('otype', '?')}[/]  "
                      f"RA={payload.get('ra', 0):.5f}  Dec={payload.get('dec', 0):+.5f}")
        try:
            bib_art = _run("lit.bibliography", target=str(payload.get("display_name", name)), rows=8)
            bib_data = _load(bib_art.id)
            docs = bib_data.get("docs", [])
            if docs:
                t = RichTable("bibcode", "title", title="recent references")
                for r in docs:
                    t.add_row(str(r.get("bibcode", "")), str(r.get("title", ""))[:70])
                console.print(t)
        except Exception as e:
            console.print(f"[dim]bibliography unavailable ({type(e).__name__})[/]")
    _emit(payload, render)


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=IMAGING)
def image(ra: float, dec: float,
          fov: Optional[float] = typer.Option(None, help="field of view in arcmin")):
    """Smart colour cutout of a position (auto survey/FOV)."""
    try:
        art = _run("imaging.cutout", target=f"{ra} {dec}", fov=fov or 0.0)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(),
          lambda: console.print(f"image -> [green]{art.path}[/]  [dim]{art.label}[/]"))


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=VALIDATION)
def where(ra: float, dec: float):
    """What's here + which deep survey covers this declination (no download)."""
    obj = cutouts.identify_field(ra, dec)
    hips, label = cutouts.best_color_hips(dec)
    coverage = resolvers.image_coverage_note(dec)
    payload = {"nearest": obj, "survey": label, "hips": hips, "coverage": coverage}

    def render():
        if obj:
            console.print(f"nearest: [bold]{obj[0]}[/] ([yellow]{obj[1]}[/]) {obj[2]:.1f}\" away")
        else:
            console.print("[dim]no catalogued SIMBAD object within 2'[/]")
        console.print(f"best colour survey: [green]{label}[/]  [dim]{hips}[/]")
        console.print(f"[dim]{coverage}[/]")
    _emit(payload, render)


@app.command(rich_help_panel=IMAGING)
def plan(target: str,
         modality: Optional[str] = typer.Option(
             None, help="filter products: colour_image, multi_panel, spectrum, metadata, …")):
    """Plan observations for a target: identity, confidence, ranked capabilities.

    Resolve TARGET through the layered resolver, classify it, and rank what
    capabilities apply. This is the planning step before `image`/`spectrum`/
    `dossier` — no data is pulled.
    """
    try:
        art = _run("object.products", target=target)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)

    def render():
        tgt = payload.get("target", {})
        console.print(
            f"[bold cyan]{tgt.get('display_name', target)}[/] "
            f"[yellow]{tgt.get('otype', '?')}[/] ({tgt.get('object_class', '?')}) · "
            f"RA={tgt.get('ra', 0):.5f} Dec={tgt.get('dec', 0):+.5f}")
        products = payload.get("products", [])
        t = RichTable("capability", "kind", "cost", "what it does",
                      title=f"products for {tgt.get('display_name', target)}")
        for p in products:
            t.add_row(f"[cyan]{p['capability']}[/]", p.get("kind", ""),
                      p.get("cost", ""), p.get("summary", "")[:60])
        console.print(t)
    _emit(payload, render)


@app.command(rich_help_panel=IMAGING)
def fetch(target: str,
          product: Optional[str] = typer.Option(
              None, help="capability name (e.g. imaging.cutout, object.spectrum); "
                         "default = top-ranked product"),
          fov: Optional[float] = typer.Option(None, help="field of view in arcmin"),
          pix: int = typer.Option(512, help="pixels per side for image products"),
          survey: str = typer.Option("auto", help="preferred colour survey")):
    """Fetch a product for a target — the deliberate step after `plan`."""
    cap = product or "imaging.cutout"
    try:
        art = _run(cap, target=target, fov=fov or 0.0, pix=pix, survey=survey)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(
        f"{art.kind} -> [green]{art.path or art.id}[/]  [dim]{art.label}[/]"))


@app.command(rich_help_panel=IMAGING)
def spectrum(target: str):
    """Fetch and render the first available spectrum for a target (NED)."""
    try:
        art = _run("object.spectrum", target=target)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(),
          lambda: console.print(f"spectrum -> [green]{art.path}[/]  [dim]{art.label}[/]"))


# --------------------------------------------------------------------------- #
# Literature
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=THEORY)
def cite(query: List[str] = typer.Argument(..., help="ADS query"),
         add: bool = typer.Option(False, help="append to refs.bib"),
         rows: int = 8):
    """Search ADS/SciX; optionally append the hits to refs.bib (needs a token)."""
    try:
        art = _run("lit.papers", query=" ".join(query), rows=rows)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)
    docs = payload.get("docs", [])

    def render():
        t = RichTable("bibcode", "year", "title", title=f"ADS: {payload.get('query', '')}")
        for d in docs:
            t.add_row(str(d.get("bibcode", "")), str(d.get("year", "")),
                      str((d.get("title") or ["?"])[0])[:60])
        console.print(t)
    _emit(payload, render)
    if add and docs:
        try:
            bibcodes = ",".join(d["bibcode"] for d in docs)
            cite_art = _run("lit.cite", bibcodes=bibcodes)
            cite_data = _load(cite_art.id)
            console.print(f"[green]+{cite_data.get('added', 0)} new entries -> refs.bib[/]")
        except Exception as e:
            raise _fail(e)


@app.command(rich_help_panel=THEORY)
def papers(query: List[str] = typer.Argument(..., help="ADS query"),
           phrase: Optional[str] = typer.Option(None, help="exact phrase in abstracts"),
           add: bool = typer.Option(False, help="append returned records to refs.bib"),
           rows: int = typer.Option(8, help="ADS rows to request"),
           report: bool = typer.Option(False, help="write a Markdown paper-set report")):
    """Search ADS/SciX with conveniences for exact phrases and reports."""
    query_text = " ".join(query)
    if phrase:
        query_text = f'abs:"{phrase}" {query_text}'.strip()
    try:
        art = _run("lit.papers", query=query_text, rows=rows)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)
    docs = payload.get("docs", [])

    def render():
        t = RichTable("bibcode", "year", "title", title=f"ADS papers: {query_text}")
        for d in docs:
            t.add_row(str(d.get("bibcode", "")), str(d.get("year", "")),
                      str((d.get("title") or ["?"])[0])[:70])
        console.print(t)
    _emit(payload, render)
    if add and docs:
        try:
            bibcodes = ",".join(d["bibcode"] for d in docs)
            cite_art = _run("lit.cite", bibcodes=bibcodes)
            cite_data = _load(cite_art.id)
            console.print(f"[green]+{cite_data.get('added', 0)} new entries -> refs.bib[/]")
        except Exception as e:
            raise _fail(e)
    if report:
        lines = [f"# ADS: {query_text}\n"]
        for d in docs:
            title = (d.get("title") or ["?"])[0]
            lines.append(f"- **{d.get('bibcode', '')}** ({d.get('year', '')}) {title}")
        path = _write_report("papers", query_text, "\n".join(lines))
        console.print(f"report -> [green]{path}[/]")


# --------------------------------------------------------------------------- #
# Data: query / sample / match / candidates / log
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=VALIDATION)
def query(archive: str, adql: str,
          refresh: bool = typer.Option(False, help="bypass cached result"),
          show: int = typer.Option(12, help="rows to display")):
    """Run ADQL against a supported archive through the kernel."""
    key = archive.lower()
    try:
        art = _run("archive.query", archive=key, adql=adql)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    _print_astropy_table(tab, f"{key}: {len(tab)} rows", limit=show)


@app.command(rich_help_panel=VALIDATION)
def sample(name: str,
           show: int = typer.Option(12, help="rows to display")):
    """Run a named sample recipe (row-capped, repeatable) through the kernel."""
    if name == "list":
        t = RichTable("recipe", "archive", "description", title="sample recipes")
        for key, rec in SAMPLE_RECIPES.items():
            t.add_row(key, rec.archive, rec.description)
        console.print(t)
        return
    try:
        art = _run("archive.sample", recipe=name)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    _print_astropy_table(tab, f"sample {name}: {len(tab)} rows", limit=show)


@app.command(rich_help_panel=VALIDATION)
def match(recipe: str, catalog: str,
          radius: float = typer.Option(5.0, help="cross-match radius in arcsec"),
          save: Optional[str] = typer.Option(None, help="save result as a candidate list"),
          show: int = typer.Option(12, help="rows to display")):
    """Cross-match a sample against a VizieR catalogue through the kernel."""
    try:
        sample_art = _run("archive.sample", recipe=recipe)
        art = _run("table.xmatch", table=sample_art.id, catalog=catalog,
                    radius_arcsec=radius)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    _print_astropy_table(tab, f"×{catalog}: {len(tab)} rows", limit=show)
    if save:
        try:
            _run("table.save_list", table=art.id, name=save,
                 note=f"xmatch {recipe}×{catalog}")
            console.print(f"[green]saved candidate list -> {save}[/]")
        except Exception as e:
            raise _fail(e)


@app.command(name="candidates", rich_help_panel=VALIDATION)
def candidates_cmd(
        name: Optional[str] = typer.Argument(None, help="list name to show; omit to list all"),
        drop: Optional[str] = typer.Option(None, "--drop", help="delete a candidate list"),
        show: int = typer.Option(12, help="rows to display")):
    """Browse curated candidate lists (the output side of the crossmatch loop).

    No NAME lists every saved list with provenance; a NAME prints that list's rows.
    """
    if drop:
        rec = candidates.find_record(drop)
        removed = candidates.drop(drop)
        msg = (f"[green]dropped {drop!r}[/]" if removed
               else f"[yellow]no table file for {drop!r}[/]")
        if rec:
            msg += f" [dim](was {rec['nrows']} rows, {rec.get('origin', '')})[/]"
        console.print(msg)
        return
    if name:
        try:
            art = _run("table.load_list", name=name)
            tab = _load(art.id)
        except Exception as e:
            raise _fail(e)
        _emit({"name": name, "nrows": len(tab)},
              lambda: _print_astropy_table(tab, f"candidates {name}: {len(tab)} rows", limit=show))
        return
    records = candidates.latest()
    if not records:
        console.print("[dim]no candidate lists yet — try `match <recipe> <catalog> --save NAME`[/]")
        return
    _emit({"lists": records},
          lambda: console.print(_rich_table_from_rows(
              "candidate lists",
              ["name", "nrows", "origin", "utc"],
              [{"name": r["name"], "nrows": r["nrows"], "origin": r.get("origin", ""),
                "utc": r.get("utc", "")} for r in records])))


@app.command(rich_help_panel=VALIDATION)
def log(limit: int = typer.Option(20, help="ledger rows to show"),
        open: Optional[str] = typer.Option(None, "--open", help="reload a cached table by id")):
    """Browse artifact provenance; reopen a past pull by its id."""
    if open:
        try:
            tab = _load(open)
        except Exception as e:
            raise _fail(e)
        _print_astropy_table(tab, f"cached {open}: {len(tab)} rows")
        return
    store = _kernel().ledger
    found = store.search(limit=limit)
    rows = [{"id": a.id, "cap": a.cap, "kind": a.kind, "label": a.label,
             "created": str(a.created)[:16]} for a in found]
    if not rows:
        console.print("[dim]no artifacts in the ledger yet[/]")
        return
    t = RichTable("id", "capability", "kind", "label", "created", title="ledger")
    for r in rows:
        t.add_row(f"[cyan]{r['id']}[/]", r["cap"], r["kind"],
                  str(r["label"])[:40], r["created"])
    console.print(t)


# --------------------------------------------------------------------------- #
# Feeds
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=VALIDATION)
def feed(name: str,
         refresh: bool = typer.Option(False, help="bypass cached result"),
         show: int = typer.Option(12, help="rows to display")):
    """Run a live global feed (neo / satellite / transient) through the kernel."""
    key = config.feed_key(name)
    if key not in config.GLOBAL_FEED_EXECUTORS:
        supported = ", ".join(sorted(config.GLOBAL_FEED_EXECUTORS))
        raise _fail(f"unknown feed {name!r}", hint=f"supported: {supported}")
    try:
        art = _run(f"feed.{key}")
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    if len(tab) == 0:
        _emit({"feed": key, "status": "empty"},
              lambda: console.print(f"[yellow]{key}: no rows returned[/]"))
        return
    _emit({"feed": key, **_table_payload(tab)},
          lambda: _print_astropy_table(tab, f"feed {key}: {len(tab)} rows", limit=show))


# --------------------------------------------------------------------------- #
# Tables: export / plot
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=VALIDATION)
def export(ref: str,
           out: Optional[Path] = typer.Option(None, help="output CSV path "
                                                         "(default data/exports/)")):
    """Export a table (candidate list, sample recipe, or artifact id) to CSV."""
    from . import paths
    try:
        tab, note = tables.load_table_ref(ref, QUERY_ARCHIVES)
    except Exception as e:
        raise _fail(e)
    if out is None:
        paths.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = paths.EXPORTS_DIR / f"{_slug(ref)}.csv"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    tab.write(out, format="ascii.csv", overwrite=True)
    _emit({"ref": ref, "source": note, "nrows": len(tab), "path": str(out)},
          lambda: console.print(f"exported {len(tab)} rows ({note}) -> [green]{out}[/]"))


@app.command(rich_help_panel=VALIDATION)
def plot(ref: str,
         x: Optional[str] = typer.Argument(None, help="X column"),
         y: Optional[str] = typer.Argument(None, help="Y column"),
         kind: str = typer.Option("scatter", help="scatter, hist, sky, or cmd"),
         out: Optional[Path] = typer.Option(None, help="output PNG path"),
         open_file: bool = typer.Option(False, "--open", help="open the PNG after rendering")):
    """Render a quick-look PNG from any table reference."""
    if kind not in plots.KINDS:
        raise _fail(f"unknown plot kind {kind!r}", hint=f"choose: {', '.join(plots.KINDS)}")
    try:
        tab, note = tables.load_table_ref(ref, QUERY_ARCHIVES)
        path = plots.plot_table(tab, x=x, y=y, kind=kind, title=f"{ref} ({note})", out=out)
    except Exception as e:
        raise _fail(e)

    def render():
        console.print(f"plot -> [green]{path}[/]  [dim]{kind}; {note}; {len(tab)} rows[/]")
        if open_file:
            try:
                _open_path(path)
            except Exception as e:
                console.print(f"[yellow]open failed ({type(e).__name__})[/]")

    _emit({"ref": ref, "source": note, "kind": kind, "x": x, "y": y,
           "nrows": len(tab), "path": str(path)}, render)


# --------------------------------------------------------------------------- #
# Compound: sweep / watch / dossier / field / census
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=VALIDATION)
def sweep(target: str,
          report: bool = typer.Option(False, help="write a Markdown sweep report")):
    """Ask every table-capable archive what it knows about a target."""
    try:
        art = _run("object.sweep", target=target)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)

    def render():
        _print_astropy_table(tab, f"sweep: {target}", limit=len(tab))
        if report:
            path = _write_report("sweep", target, _markdown_table(tab, f"Sweep: {target}"))
            console.print(f"report -> [green]{path}[/]")
    _emit({**_table_payload(tab), "target": target}, render)


@app.command(rich_help_panel=VALIDATION)
def watch(target: Optional[str] = typer.Argument(
              None, help="a target name/coordinates to watch"),
          list_name: Optional[str] = typer.Option(
              None, "--list", help="watch every row of a saved candidate list instead"),
          radius: float = typer.Option(2.0, help="cone-search radius in arcmin"),
          days: float = typer.Option(7.0, help="lookback window in days"),
          report: bool = typer.Option(False, help="write a Markdown watch report")):
    """Live alert intelligence: ALeRCE transient alerts near a target or a
    whole candidate list (the 'Rubin watch' scoping in docs/data-atlas.md).
    """
    if bool(target) == bool(list_name):
        raise _fail("give exactly one of TARGET or --list NAME")
    try:
        art = _run("object.watch", target=target or "",
                    candidate_list=list_name or "",
                    radius_arcmin=radius, days=days)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    label = target or list_name

    def render():
        _print_astropy_table(tab, f"watch: {label}", limit=len(tab))
        if report:
            path = _write_report("watch", label, _markdown_table(tab, f"Watch: {label}"))
            console.print(f"report -> [green]{path}[/]")
    _emit({**_table_payload(tab), "label": label}, render)


@app.command(rich_help_panel=THEORY)
def census(report: bool = typer.Option(False, help="write a Markdown census report"),
           out: Optional[Path] = typer.Option(None, help="output CSV path")):
    """Count recent arXiv effort by Celestrium topic and write the CSV."""
    try:
        art = _run("lit.census", topics=None)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    csv_path = census_mod.write_csv(tab, out)
    report_path = None
    if report:
        report_path = _write_report("census", "topic-trajectories",
                                    _markdown_table(tab, "Field-Effort Census"))

    def render():
        _print_astropy_table(tab, f"field-effort census: {len(tab)} topics", limit=len(tab))
        console.print(f"csv -> [green]{csv_path}[/]")
        if report_path:
            console.print(f"report -> [green]{report_path}[/]")
    _emit({**_table_payload(tab), "csv": str(csv_path)}, render)


@app.command(rich_help_panel=THEORY)
def dossier(target: str,
            rows: int = typer.Option(6, help="literature rows to include"),
            fov: Optional[float] = typer.Option(None, help="field of view in arcmin"),
            ned: bool = typer.Option(False, help="add NED redshift (slower; extragalactic)"),
            images: bool = typer.Option(True, help="render colour and panel images")):
    """Build a compact object packet: resolve, image, literature, Markdown report."""
    try:
        art = _run("object.dossier", target=target, rows=rows,
                    fov=fov or 0.0, images=images, ned=ned)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(),
          lambda: console.print(f"dossier -> [green]{art.path or art.id}[/]  [dim]{art.label}[/]"))


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=VALIDATION)
def field(ra: float, dec: float,
          fov: float = typer.Option(5.0, help="field of view in arcmin"),
          images: bool = typer.Option(True, help="render colour and panel images")):
    """Build a compact packet for a sky position, including blank fields."""
    try:
        art = _run("object.field", target=f"{ra} {dec}", fov=fov, images=images)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(),
          lambda: console.print(f"field packet -> [green]{art.path or art.id}[/]  "
                                f"[dim]{art.label}[/]"))


# --------------------------------------------------------------------------- #
# Visuals: atlas-targets / poster
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=IMAGING)
def atlas_targets(limit: int = typer.Option(6, help="number of curated targets"),
                  fov_scale: float = typer.Option(1.0, help="multiply each target FOV")):
    """Render a curated contact sheet of visually useful astronomy targets."""
    try:
        art = _run("imaging.atlas", limit=limit, fov=0.0)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(),
          lambda: console.print(f"atlas -> [green]{art.path or art.id}[/]  "
                                f"[dim]{art.label}[/]"))


@app.command(rich_help_panel=IMAGING)
def poster(target: str,
           resolution: str = typer.Option("1080p", help="1080p, 2k, or 4k"),
           style: str = typer.Option("clean", help="clean, label, or science"),
           fov: Optional[float] = typer.Option(None, help="field of view in arcmin")):
    """Render a desktop-wallpaper style astronomy poster."""
    if resolution not in RESOLUTIONS:
        raise _fail(f"unknown resolution {resolution!r}", hint="choose 1080p, 2k, or 4k")
    if style not in {"clean", "label", "science"}:
        raise _fail(f"unknown style {style!r}", hint="choose clean, label, or science")
    width, height = RESOLUTIONS[resolution]
    try:
        art = _run("imaging.poster", target=target, fov=fov or 8.0,
                    width=width, height=height, style=style)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(),
          lambda: console.print(f"poster -> [green]{art.path or art.id}[/]"))


# --------------------------------------------------------------------------- #
# Runbooks + map quick-reference
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=WORKFLOWS)
def runbook(name: str,
            limit: int = typer.Option(4, help="rows/images per runbook section"),
            images: bool = typer.Option(True, help="render field and atlas images"),
            posters: bool = typer.Option(True, help="render poster anchors"),
            samples: bool = typer.Option(True, help="run row-capped sample pulls")):
    """Run a curated repeatable workflow and write an index report."""
    if name == "list":
        t = RichTable("runbook", "description", title="runbooks")
        for key, spec in config.RUNBOOKS.items():
            t.add_row(key, spec.description)
        console.print(t)
        return
    if name not in config.RUNBOOKS:
        raise _fail("unknown runbook", hint="run: python -m celestrium runbook list")
    spec = config.RUNBOOKS[name]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    for i, step in enumerate(spec.steps, 1):
        kind = step.get("kind", "unknown")
        console.print(f"  [dim]{i}/{len(spec.steps)}: {kind}[/]")
        try:
            if kind == "papers":
                q = step.get("query", "")
                if step.get("phrase"):
                    q = f'abs:"{step["phrase"]}" {q}'.strip()
                _run("lit.papers", query=q, rows=step.get("rows", 8))
            elif kind == "field":
                _run("object.field", target=f"{step['ra']} {step['dec']}",
                     fov=step.get("fov", 5.0), images=images)
            elif kind == "sample":
                _run("archive.sample", recipe=step["recipe"])
            elif kind == "atlas":
                _run("imaging.atlas", limit=limit, fov=0.0)
            elif kind == "poster" and posters:
                _run("imaging.poster", target=step["target"],
                     fov=step.get("fov", 8.0), width=1920, height=1080, style="clean")
            notes.append(f"step {i} ({kind}): ok")
        except Exception as e:
            notes.append(f"step {i} ({kind}): {type(e).__name__}: {e}")
            console.print(f"  [yellow]{type(e).__name__}: {e}[/]")
    index = REPORTS_DIR / f"runbook-{name}.md"
    body = f"# Runbook: {name}\n\n{spec.description}\n\n" + "\n".join(f"- {n}" for n in notes)
    index.write_text(body + "\n", encoding="utf-8")
    _emit({"runbook": name, "steps": len(spec.steps), "index": str(index)},
          lambda: console.print(f"runbook -> [green]{index}[/]"))


def _print_map(filename: str):
    path = REPO / filename
    if not path.exists():
        raise _fail(f"{filename} not found")
    console.print(Markdown(path.read_text(encoding="utf-8")))


@app.command(rich_help_panel=WORKFLOWS)
def doctor(ping: bool = typer.Option(True, help="live-ping each archive's availability endpoint"),
           timeout: float = typer.Option(10.0, help="per-ping timeout in seconds")):
    """Instrument health: archive reachability, ADS token, cache + manifest state.

    Answers "does half my instrument currently work?" before a session — each
    registered archive's TAP availability endpoint is pinged (no data pulled).
    """
    from . import doctor as doctor_mod
    report = doctor_mod.run_checks(ping=ping, timeout=timeout)

    def render():
        local = report["local"]
        token = "[green]ok[/]" if local["ads_token"] else "[yellow]missing[/]"
        console.print(f"ADS token: {token}   cache: {local['cache_files']} files "
                      f"({local['cache_mb']} MB)   manifest: {local['manifest_rows']} rows   "
                      f"candidates: {local['candidate_lists']} lists")
        if report["archives"]:
            t = RichTable("archive", "status", "ms", "note", title="archive availability")
            for a in report["archives"]:
                colour = {"up": "green", "down": "red"}.get(a["status"], "yellow")
                t.add_row(a["archive"], f"[{colour}]{a['status']}[/]",
                          str(a.get("ms", "—")), str(a.get("note", ""))[:60])
            console.print(t)
            s = report["summary"]
            console.print(f"[dim]{s['archives_up']}/{s['archives_checked']} archives up[/]")
    _emit(report, render)


@app.command(rich_help_panel=WORKFLOWS)
def atlas():
    """Pretty-print the data atlas (what data exists)."""
    _print_map("docs/data-atlas.md")


@app.command(rich_help_panel=WORKFLOWS)
def toolbox():
    """Pretty-print the toolbox map (everything around the query)."""
    _print_map("docs/toolbox.md")


@app.command(rich_help_panel=WORKFLOWS)
def tui():
    """Launch the Celestrium cockpit (Textual TUI) — needs `pip install textual`."""
    try:
        from .tui.app import main
    except ImportError:
        raise _fail("Textual not installed", hint="run: pip install textual")
    main()


# The kernel-era commands (`caps`, `run`, `ledger`, `repro`, `methods`, plus the
# `study` and `vault` groups) are *generated from the capability registry* rather
# than written here — so a new capability appears in `--help` with no CLI edit.
# See celestrium/cli_kernel.py and celestrium/core/capability.py.
from . import cli_kernel as _cli_kernel  # noqa: E402
_cli_kernel.register(app, _STATE)


if __name__ == "__main__":
    app()
