#!/usr/bin/env python
"""Celestrium CLI — the operable front-end to the astrophysics instrument.

Three wings, one tool: a **theory workshop** (literature in, ideas assessed), an
**experimental validation** bench (pull data slices, test the empirical claims),
and an **imaging & recreation** studio (scientific diagrams + beautiful renders).
A thin presenter: all real logic lives in celestrium/ (registry.py = data,
packets.py = builders), so this CLI and the Textual TUI (tui/) share one brain.

    python -m celestrium --help
    python -m celestrium resolve M87
    python -m celestrium image 187.7059 12.3911
    python -m celestrium where 213.6906 -12.5801
    python -m celestrium cite 'abs:"cosmic dipole" year:2024-2026' --add
    python -m celestrium query gaia "SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source"
    python -m celestrium match gaia-bright-nearby vizier:VIII/65/nvss --radius 5 --save agn
    python -m celestrium candidates          # browse saved lists; NAME to show one
    python -m celestrium log                 # provenance; --open/--rerun a hash
    python -m celestrium dossier M87 --ned
    python -m celestrium field 213.6906 -12.5801
    python -m celestrium papers year:2025-2026 --phrase "Euclid Quick Data Release"
    python -m celestrium sample list
    python -m celestrium poster M87 --resolution 1080p --style label
    python -m celestrium runbook euclid-q1
    python -m celestrium atlas | toolbox     # pretty-print the repo maps

Global: add --json before any subcommand for machine-readable output.
"""
import json as _json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table as RichTable

from . import cache, candidates, cutouts, packets, planner, registry, resolvers, spectra
# Re-exported so the test suite (and any importer) can patch these on `hub`.
from .registry import QUERY_ARCHIVES, SAMPLE_RECIPES, SampleRecipe, ATLAS_TARGETS  # noqa: F401

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
_slug = packets.slug
_paper_lines = packets.paper_lines


@app.callback()
def _main(json_out: bool = typer.Option(False, "--json", help="machine-readable output")):
    _STATE["json"] = json_out


def _emit(payload, render):
    """JSON-print payload in --json mode, else call render() for rich output."""
    if _STATE["json"]:
        console.print_json(_json.dumps(payload, default=str))
    else:
        render()


def _write_report(kind: str, stem: str, body: str) -> Path:
    return packets.write_report(REPORTS_DIR, kind, stem, body)


def _rich_table_from_rows(title, columns, rows) -> RichTable:
    table = RichTable(*[str(c) for c in columns], title=title)
    for row in rows:
        table.add_row(*[str(row[c])[:80] for c in columns])
    return table


def _print_astropy_table(tab, title: str, limit: int = 12):
    columns = list(tab.colnames)[:8]
    console.print(_rich_table_from_rows(title, columns, tab[:limit]))
    if len(tab) > limit:
        console.print(f"[dim]showing {limit} of {len(tab)} rows[/]")


def _contact_sheet(paths, out: Path, title: str):
    # Thin re-export so the test suite can still patch hub._contact_sheet; the
    # implementation lives in cutouts so the TUI runbook runner shares it.
    return cutouts.contact_sheet(paths, out, title)


# --------------------------------------------------------------------------- #
# Object / position
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=THEORY)
def resolve(name: str):
    """Identify an object (SIMBAD) and list recent papers about it."""
    info = resolvers.identify(name)
    if info is None or len(info) == 0:
        console.print(f"[red]No SIMBAD match for {name!r}[/]")
        raise typer.Exit(1)
    row = info[0]
    payload = {"name": str(row["main_id"]), "otype": str(row.get("otype", "?")),
               "ra": float(row["ra"]), "dec": float(row["dec"])}

    def render():
        console.print(f"[bold cyan]{row['main_id']}[/]  "
                      f"[yellow]{row.get('otype', '?')}[/]  "
                      f"RA={row['ra']:.5f}  Dec={row['dec']:+.5f}")
        try:
            bib = resolvers.bibliography(str(row["main_id"]), limit=8)
            t = RichTable("bibcode", "title", title="recent references")
            for r in bib:
                t.add_row(str(r["bibcode"]), str(r["title"])[:70])
            console.print(t)
        except Exception as e:
            console.print(f"[dim]bibliography unavailable ({type(e).__name__})[/]")
    _emit(payload, render)


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=IMAGING)
def image(ra: float, dec: float,
          fov: Optional[float] = typer.Option(None, help="field of view in arcmin")):
    """Smart multi-wavelength + colour cutout of a position (auto survey/FOV)."""
    cutouts.smart(ra, dec, fov_arcmin=fov)


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=VALIDATION)
def where(ra: float, dec: float):
    """What's here + which deep survey covers this declination (no download)."""
    obj = cutouts.identify_field(ra, dec)
    hips, label = cutouts.best_color_hips(dec)
    coverage = planner.image_coverage_note(dec)
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
    """Plan observations for a target: identity, confidence, ambiguity, ranked products.

    Resolve TARGET through the layered resolver (exact → coordinate → relaxed →
    nearby → blank field), classify it, and rank what could be fetched. This is
    the planning step before `image`/`spectrum`/`dossier` — no data is pulled.
    """
    tgt = planner.resolve_target(target)
    if tgt is None:
        console.print(f"[red]could not resolve {target!r} (try 'RA Dec' or an alias)[/]")
        raise typer.Exit(1)
    plans = planner.recommend_plans(tgt, modality=modality)
    coverage = planner.image_coverage_note(tgt.dec)
    payload = {"target": tgt.to_dict(),
               "plans": [p.to_dict() for p in plans],
               "coverage": coverage}

    def render():
        console.print(
            f"[bold cyan]{tgt.display_name}[/] [yellow]{tgt.otype or '?'}[/] "
            f"({tgt.object_class}) · [magenta]{tgt.match_kind}[/] "
            f"conf={tgt.confidence:.2f}  RA={tgt.ra:.5f} Dec={tgt.dec:+.5f}")
        if tgt.alternatives:
            alts = ", ".join(str(a.get("name", "?")) for a in tgt.alternatives[:5])
            console.print(f"[dim]also matched: {alts}[/]")
        t = RichTable("product", "status", "action", "coverage",
                      title=f"observation plan: {tgt.display_name}")
        for p in plans:
            t.add_row(p.product.label, p.product.status, p.next_action,
                      p.product.coverage_hint)
        console.print(t)
        console.print(f"[dim]{coverage}[/]")
    _emit(payload, render)


@app.command(rich_help_panel=IMAGING)
def spectrum(target: str):
    """Fetch and render the first available spectrum for a target (NED)."""
    result = spectra.fetch_ned_spectrum(target)
    if result is None:
        console.print(f"[yellow]no spectrum found for {target!r} (NED)[/]")
        raise typer.Exit(1)
    _emit(result.to_dict(),
          lambda: console.print(f"spectrum -> [green]{result.path}[/]  "
                                f"[dim]{result.summary}[/]"))


# --------------------------------------------------------------------------- #
# Literature
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=THEORY)
def cite(query: List[str] = typer.Argument(..., help="ADS query"),
         add: bool = typer.Option(False, help="append to refs.bib"),
         rows: int = 8):
    """Search ADS/SciX; optionally append the hits to refs.bib (needs a token)."""
    from . import ads
    query_text = " ".join(query)
    try:
        docs = ads.search(query_text, rows=rows)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    t = RichTable("bibcode", "year", "title", title=f"ADS: {query_text}")
    for d in docs:
        t.add_row(d["bibcode"], str(d.get("year", "")), packets.title_of(d)[:60])
    console.print(t)
    if add and docs:
        try:
            n = ads.add_to_refs([d["bibcode"] for d in docs])
        except Exception as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        console.print(f"[green]+{n} new entries -> refs.bib[/]")


@app.command(rich_help_panel=THEORY)
def papers(query: List[str] = typer.Argument(..., help="ADS query"),
           phrase: Optional[str] = typer.Option(None, help="exact phrase in abstracts"),
           add: bool = typer.Option(False, help="append returned records to refs.bib"),
           rows: int = typer.Option(8, help="ADS rows to request"),
           report: bool = typer.Option(False, help="write a Markdown paper-set report")):
    """Search ADS/SciX with conveniences for exact phrases and reports."""
    from . import ads
    query_text = " ".join(query)
    if phrase:
        query_text = f'abs:"{phrase}" {query_text}'.strip()
    try:
        ps = packets.build_paper_set(query_text, rows=rows)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    docs = ps.docs

    def render():
        t = RichTable("bibcode", "year", "title", title=f"ADS papers: {query_text}")
        for d in docs:
            t.add_row(str(d.get("bibcode", "")), str(d.get("year", "")),
                      str((d.get("title") or ["?"])[0])[:70])
        console.print(t)
    _emit(ps.to_dict(), render)
    if add and docs:
        try:
            n = ads.add_to_refs([d["bibcode"] for d in docs])
        except Exception as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        console.print(f"[green]+{n} new entries -> refs.bib[/]")
    if report:
        path = _write_report("papers", query_text, ps.to_markdown())
        console.print(f"report -> [green]{path}[/]")


# --------------------------------------------------------------------------- #
# Data: query / sample / match / log
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=VALIDATION)
def query(archive: str, adql: str,
          refresh: bool = typer.Option(False, help="bypass cached result"),
          show: int = typer.Option(12, help="rows to display")):
    """Run ADQL against a supported archive through the local cache."""
    key = archive.lower()
    source = registry.resolve_query_source(key, QUERY_ARCHIVES)
    if source is None:
        supported = ", ".join(sorted(QUERY_ARCHIVES))
        console.print(f"[red]unknown archive {archive!r}; supported: {supported}[/]")
        raise typer.Exit(1)
    try:
        tab = cache.cached_query(key, adql, lambda: source.query(adql), refresh=refresh)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    _print_astropy_table(tab, f"{key}: {len(tab)} rows", limit=show)


@app.command(rich_help_panel=VALIDATION)
def sample(recipe: str,
           refresh: bool = typer.Option(False, help="bypass cached result"),
           show: int = typer.Option(12, help="rows to display")):
    """Run a named low-compute science recipe through the cache."""
    if recipe == "list":
        t = RichTable("recipe", "archive", "description", title="sample recipes")
        for name, rec in sorted(SAMPLE_RECIPES.items()):
            t.add_row(name, rec.archive, rec.description)
        console.print(t)
        return
    rec = SAMPLE_RECIPES.get(recipe)
    if rec is None:
        console.print(f"[red]unknown recipe {recipe!r}; run: python -m celestrium sample list[/]")
        raise typer.Exit(1)
    source = registry.resolve_query_source(rec.archive, QUERY_ARCHIVES)
    try:
        tab = cache.cached_query(f"sample:{recipe}", rec.adql,
                                 lambda: source.query(rec.adql), refresh=refresh)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{rec.description}[/]")
    _print_astropy_table(tab, f"{recipe}: {len(tab)} rows", limit=show)


@app.command(rich_help_panel=VALIDATION)
def match(source: str, catalog: str,
          radius: float = typer.Option(5.0, help="match radius in arcsec"),
          ra: str = typer.Option("ra", help="RA column in the local table"),
          dec: str = typer.Option("dec", help="Dec column in the local table"),
          refresh: bool = typer.Option(False, help="bypass cached result"),
          save: Optional[str] = typer.Option(None, "--save", help="store matches as a named candidate list"),
          show: int = typer.Option(12, help="rows to display")):
    """Cross-match a cached pull (by hash) or a sample recipe against a VizieR catalogue.

    The audit primitive: 'do two catalogues agree on the same sources?' SOURCE is
    either a manifest hash (see `log`) or a recipe name (see `sample list`).
    CATALOG is a CDS id, e.g. vizier:VIII/65/nvss. With --save NAME the matched
    rows are persisted as a candidate list (see `candidates`).
    """
    from . import xmatch
    rec = SAMPLE_RECIPES.get(source)
    try:
        if rec is not None:
            src = registry.resolve_query_source(rec.archive, QUERY_ARCHIVES)
            local = cache.cached_query(f"sample:{source}", rec.adql,
                                       lambda: src.query(rec.adql))
        else:
            local = cache.load_cached(source)
    except Exception as e:
        console.print(f"[red]could not load local table {source!r}: {e}[/]")
        raise typer.Exit(1)
    qtag = f"{source}|{catalog}|r{radius}"
    try:
        out = cache.cached_query(
            "xmatch", qtag,
            lambda: xmatch.match(local, cat2=catalog, ra=ra, dec=dec, radius_arcsec=radius),
            refresh=refresh)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    _print_astropy_table(out, f"xmatch {source} x {catalog}: {len(out)} rows", limit=show)
    if save:
        path = candidates.save(save, out, origin=f"xmatch {source} x {catalog}",
                               note=f"radius {radius}\"")
        console.print(f"candidate list [cyan]{save}[/] ({len(out)} rows) -> [green]{path}[/]")


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
            tab = candidates.load(name)
        except Exception as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        rec = candidates.find_record(name) or {}
        _emit({"name": name, "nrows": len(tab), **rec},
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
def log(limit: int = typer.Option(20, help="manifest rows to show"),
        open: Optional[str] = typer.Option(None, "--open", help="reload a cached table by hash"),
        rerun: Optional[str] = typer.Option(None, help="re-run a past query by hash (refresh)")):
    """Browse query provenance; reopen or re-run a past pull by its hash."""
    if open:
        try:
            tab = cache.load_cached(open)
        except Exception as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        _print_astropy_table(tab, f"cached {open}: {len(tab)} rows")
        return
    if rerun:
        rec = cache.find_record(rerun)
        if rec is None:
            console.print(f"[red]no manifest record for hash {rerun!r}[/]")
            raise typer.Exit(1)
        src = registry.resolve_query_source(rec["archive"], QUERY_ARCHIVES)
        if src is None:
            console.print(f"[red]archive {rec['archive']!r} not re-runnable[/]")
            raise typer.Exit(1)
        tab = cache.cached_query(rec["archive"], rec["query"],
                                 lambda: src.query(rec["query"]), refresh=True)
        _print_astropy_table(tab, f"re-ran {rerun}: {len(tab)} rows")
        return
    records = cache.manifest()
    if not records:
        console.print("[dim]no cached query manifest yet[/]")
        return
    rows = records[-limit:][::-1]
    t = RichTable("utc", "archive", "rows", "hash", "query", title="query manifest")
    for r in rows:
        t.add_row(str(r.get("utc", "")), str(r.get("archive", "")), str(r.get("nrows", "")),
                  str(r.get("hash", "")), str(r.get("query", ""))[:80])
    console.print(t)


# --------------------------------------------------------------------------- #
# Packets: dossier / field
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=THEORY)
def dossier(target: str,
            rows: int = typer.Option(6, help="ADS rows to include"),
            fov: Optional[float] = typer.Option(None, help="field of view in arcmin"),
            ned: bool = typer.Option(False, help="add NED redshift (slower; extragalactic)"),
            images: bool = typer.Option(True, help="render colour and panel images")):
    """Build a compact object packet: resolve, image, literature, Markdown report."""
    try:
        pkt = packets.build_object_packet(
            target, rows=rows, fov=fov, images=images, ned=ned,
            on_error=lambda m: console.print(f"[yellow]{m}[/]"))
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    path = _write_report("dossier", pkt.name, pkt.to_markdown())
    _emit({**pkt.to_dict(), "report": str(path)},
          lambda: console.print(f"dossier -> [green]{path}[/]"))


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=VALIDATION)
def field(ra: float, dec: float,
          fov: float = typer.Option(5.0, help="field of view in arcmin"),
          images: bool = typer.Option(True, help="render colour and panel images")):
    """Build a compact packet for a sky position, including blank fields."""
    pkt = packets.build_field_packet(
        ra, dec, fov=fov, images=images,
        on_error=lambda m: console.print(f"[yellow]{m}[/]"))
    path = _write_report("field", f"{ra:.5f}_{dec:+.5f}", pkt.to_markdown())
    _emit({**pkt.to_dict(), "report": str(path)},
          lambda: console.print(f"field packet -> [green]{path}[/]"))


# --------------------------------------------------------------------------- #
# Visuals: atlas-targets / poster
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=IMAGING)
def atlas_targets(limit: int = typer.Option(6, help="number of curated targets"),
                  fov_scale: float = typer.Option(1.0, help="multiply each target FOV")):
    """Render a curated contact sheet of visually useful astronomy targets."""
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for target in ATLAS_TARGETS[:limit]:
        out = ATLAS_DIR / f"{_slug(target['name'])}.jpg"
        try:
            paths.append(cutouts.color(target["ra"], target["dec"],
                                       fov_arcmin=target["fov"] * fov_scale, pix=768, out=out))
        except Exception as e:
            console.print(f"[yellow]skip {target['name']} ({type(e).__name__})[/]")
    sheet = _contact_sheet(paths, ATLAS_DIR / "atlas-targets.png", "atlas targets")
    console.print(f"atlas -> [green]{sheet}[/]")


@app.command(rich_help_panel=IMAGING)
def poster(target: str,
           resolution: str = typer.Option("1080p", help="1080p, 2k, or 4k"),
           style: str = typer.Option("clean", help="clean, label, or science"),
           fov: Optional[float] = typer.Option(None, help="field of view in arcmin")):
    """Render a desktop-wallpaper style astronomy poster."""
    if resolution not in RESOLUTIONS:
        console.print(f"[red]unknown resolution {resolution!r}; choose 1080p, 2k, or 4k[/]")
        raise typer.Exit(1)
    if style not in {"clean", "label", "science"}:
        console.print(f"[red]unknown style {style!r}; choose clean, label, or science[/]")
        raise typer.Exit(1)
    try:
        obj = packets.resolve_target(target)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    width, height = RESOLUTIONS[resolution]
    fov_arcmin = fov or packets.default_fov(obj["otype"], 10.0, 4.0)
    POSTERS_DIR.mkdir(parents=True, exist_ok=True)
    out = POSTERS_DIR / f"{_slug(obj['name'])}-{resolution}-{style}.jpg"
    try:
        path = cutouts.poster(obj["ra"], obj["dec"], fov_arcmin=fov_arcmin,
                              width=width, height=height, out=out,
                              label=obj["name"], style=style)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"poster -> [green]{path}[/]")


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
        for key, spec in registry.RUNBOOKS.items():
            t.add_row(key, spec.description)
        console.print(t)
        return
    if name not in registry.RUNBOOKS:
        console.print("[red]unknown runbook; run: python -m celestrium runbook list[/]")
        raise typer.Exit(1)
    result = packets.run_runbook(
        name, reports_dir=REPORTS_DIR, atlas_dir=ATLAS_DIR, posters_dir=POSTERS_DIR,
        contact_sheet=_contact_sheet, limit=limit, images=images,
        posters=posters, samples=samples)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    index = REPORTS_DIR / f"runbook-{name}.md"
    index.write_text(result.to_markdown() + "\n", encoding="utf-8")
    _emit({**result.to_dict(), "index": str(index)},
          lambda: console.print(f"runbook -> [green]{index}[/]"))


def _print_map(filename: str):
    path = REPO / filename
    if not path.exists():
        console.print(f"[red]{filename} not found[/]")
        raise typer.Exit(1)
    console.print(Markdown(path.read_text(encoding="utf-8")))


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
        console.print("[red]Textual not installed. Run: pip install textual[/]")
        raise typer.Exit(1)
    main()


if __name__ == "__main__":
    app()
