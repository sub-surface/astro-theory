#!/usr/bin/env python
"""astro-hub — a small CLI over the repo's access/ tools.

One operable front-end for the everyday moves: resolve an object, image a
position, search the literature, check which survey covers a patch of sky. Built
on typer + rich (cute output). Designed to be driven by either Leon or Claude and
to grow toward a full TUI — see ../tui-scope.md.

    python -m access.hub --help
    python -m access.hub resolve M87
    python -m access.hub image 187.7059 12.3911
    python -m access.hub where 213.6906 -12.5801
    python -m access.hub cite 'abs:"cosmic dipole" year:2024-2026' --add
    python -m access.hub log
    python -m access.hub query gaia "SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source"
    python -m access.hub dossier M87
    python -m access.hub field 213.6906 -12.5801
    python -m access.hub papers year:2025-2026 --phrase "Euclid Quick Data Release"
    python -m access.hub sample list
    python -m access.hub atlas-targets
    python -m access.hub poster M87 --resolution 1080p --style label
"""
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import typer
from rich.console import Console
from rich.table import Table as RichTable

from . import cache, cutouts, resolvers

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help=("Desk-astronomy CLI: resolve | image | where | cite | "
                        "log | query | dossier | field | papers | sample | "
                        "atlas-targets | poster."))
console = Console()
REPO = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO / "data" / "reports"
ATLAS_DIR = REPO / "data" / "atlas"
POSTERS_DIR = REPO / "data" / "posters"
QUERY_ARCHIVES = {
    "euclid": "access.euclid",
    "gaia": "access.gaia",
    "heasarc": "access.heasarc",
    "irsa": "access.irsa",
}
_COORD_CONTEXT = {"ignore_unknown_options": True}
RESOLUTIONS = {
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


@dataclass(frozen=True)
class SampleRecipe:
    description: str
    archive: str
    adql: str


SAMPLE_RECIPES = {
    "gaia-bright-nearby": SampleRecipe(
        "Gaia DR3 bright nearby seed stars",
        "gaia",
        "SELECT TOP 25 source_id, ra, dec, parallax, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source "
        "WHERE parallax > 20 AND phot_g_mean_mag < 10 "
        "ORDER BY phot_g_mean_mag",
    ),
    "wise-agn-colors": SampleRecipe(
        "AllWISE colour-selected AGN-like sources in a tiny row-capped pull",
        "irsa",
        "SELECT TOP 25 designation, ra, dec, w1mpro, w2mpro, w3mpro "
        "FROM allwise_p3as_psd "
        "WHERE w1mpro - w2mpro > 0.8 AND w2mpro < 15",
    ),
    "heasarc-chandra-gc": SampleRecipe(
        "HEASARC Chandra observations near the Galactic Centre",
        "heasarc",
        "SELECT TOP 25 name, ra, dec, time, status "
        "FROM chanmaster "
        "WHERE CONTAINS(POINT('ICRS', ra, dec), "
        "CIRCLE('ICRS', 266.4, -29.0, 0.5))=1",
    ),
    "euclid-q1-smoke": SampleRecipe(
        "Euclid archive smoke query; table names may drift between public releases",
        "euclid",
        "SELECT TOP 10 object_id, right_ascension, declination "
        "FROM catalogue.mer_catalogue",
    ),
}


ATLAS_TARGETS = [
    {"name": "M87", "ra": 187.7059, "dec": 12.3911, "fov": 8.0},
    {"name": "3C 273", "ra": 187.2779, "dec": 2.0524, "fov": 4.0},
    {"name": "Centaurus A", "ra": 201.3651, "dec": -43.0191, "fov": 14.0},
    {"name": "Bullet Cluster", "ra": 104.6583, "dec": -55.9475, "fov": 10.0},
    {"name": "CDFS", "ra": 53.1250, "dec": -28.1000, "fov": 12.0},
    {"name": "Sombrero Galaxy", "ra": 189.9976, "dec": -11.6231, "fov": 10.0},
]


def _rich_table_from_rows(title: str, columns, rows) -> RichTable:
    table = RichTable(*[str(c) for c in columns], title=title)
    for row in rows:
        table.add_row(*[str(row[c])[:80] for c in columns])
    return table


def _print_astropy_table(tab, title: str, limit: int = 12):
    columns = list(tab.colnames)[:8]
    rows = tab[:limit]
    console.print(_rich_table_from_rows(title, columns, rows))
    if len(tab) > limit:
        console.print(f"[dim]showing {limit} of {len(tab)} rows[/]")


def _query_archive(name: str):
    source = QUERY_ARCHIVES.get(name)
    if isinstance(source, str):
        return importlib.import_module(source)
    return source


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "untitled"


def _write_report(kind: str, stem: str, body: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{kind}-{_slug(stem)}.md"
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path


def _resolve_target(name: str):
    info = resolvers.identify(name)
    if info is None or len(info) == 0:
        raise RuntimeError(f"No SIMBAD match for {name!r}")
    row = info[0]
    return {
        "name": str(row["main_id"]),
        "otype": str(row.get("otype", "?")),
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
    }


def _paper_lines(docs) -> list[str]:
    if not docs:
        return ["- No ADS records returned."]
    return [
        f"- {d.get('bibcode', '?')} ({d.get('year', '?')}): "
        f"{d.get('title', ['?'])[0]}"
        for d in docs
    ]


def _contact_sheet(paths, out: Path, title: str):
    if not paths:
        raise RuntimeError("no images available for contact sheet")
    cols = min(3, len(paths))
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = [axes] if len(paths) == 1 else list(getattr(axes, "flat", axes))
    for ax, path in zip(axes, paths):
        ax.imshow(plt.imread(path))
        ax.set_title(Path(path).stem, fontsize=8)
        ax.set_axis_off()
    for ax in axes[len(paths):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


@app.command()
def resolve(name: str):
    """Identify an object (SIMBAD) and list recent papers about it."""
    info = resolvers.identify(name)
    if info is None or len(info) == 0:
        console.print(f"[red]No SIMBAD match for {name!r}[/]")
        raise typer.Exit(1)
    row = info[0]
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


@app.command(context_settings=_COORD_CONTEXT)
def image(ra: float, dec: float,
          fov: float | None = typer.Option(None, help="field of view in arcmin")):
    """Smart multi-wavelength + colour cutout of a position (auto survey/FOV)."""
    cutouts.smart(ra, dec, fov_arcmin=fov)


@app.command(context_settings=_COORD_CONTEXT)
def where(ra: float, dec: float):
    """What's here + which deep survey covers this declination (no download)."""
    obj = cutouts.identify_field(ra, dec)
    if obj:
        console.print(f"nearest: [bold]{obj[0]}[/] ([yellow]{obj[1]}[/]) "
                      f"{obj[2]:.1f}\" away")
    else:
        console.print("[dim]no catalogued SIMBAD object within 2'[/]")
    hips, label = cutouts.best_color_hips(dec)
    console.print(f"best colour survey: [green]{label}[/]  [dim]{hips}[/]")


@app.command()
def cite(query: List[str] = typer.Argument(..., help="ADS query"),
         add: bool = typer.Option(False, help="append to refs.bib"),
         rows: int = 8):
    """Search ADS/SciX; optionally append the hits to refs.bib (needs a token)."""
    from . import ads  # imported lazily so the rest of the CLI runs token-free
    query_text = " ".join(query)
    try:
        docs = ads.search(query_text, rows=rows)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    t = RichTable("bibcode", "year", "title", title=f"ADS: {query_text}")
    for d in docs:
        t.add_row(d["bibcode"], str(d.get("year", "")),
                  d.get("title", ["?"])[0][:60])
    console.print(t)
    if add and docs:
        try:
            n = ads.add_to_refs([d["bibcode"] for d in docs])
        except Exception as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        console.print(f"[green]+{n} new entries -> refs.bib[/]")


@app.command()
def log(limit: int = typer.Option(20, help="manifest rows to show")):
    """Show cached query provenance from data/manifest.jsonl."""
    records = cache.manifest()
    if not records:
        console.print("[dim]no cached query manifest yet[/]")
        return
    rows = records[-limit:][::-1]
    t = RichTable("utc", "archive", "rows", "hash", "query", title="query manifest")
    for rec in rows:
        t.add_row(
            str(rec.get("utc", "")),
            str(rec.get("archive", "")),
            str(rec.get("nrows", "")),
            str(rec.get("hash", "")),
            str(rec.get("query", ""))[:80],
        )
    console.print(t)


@app.command()
def query(archive: str, adql: str,
          refresh: bool = typer.Option(False, help="bypass cached result"),
          show: int = typer.Option(12, help="rows to display")):
    """Run ADQL against a supported archive through the local cache."""
    key = archive.lower()
    source = _query_archive(key)
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


@app.command()
def papers(query: List[str] = typer.Argument(..., help="ADS query"),
           phrase: str | None = typer.Option(None, help="exact phrase to search in abstracts"),
           add: bool = typer.Option(False, help="append returned records to refs.bib"),
           rows: int = typer.Option(8, help="ADS rows to request"),
           report: bool = typer.Option(False, help="write a Markdown paper-set report")):
    """Search ADS/SciX with conveniences for exact phrases and reports."""
    from . import ads
    query_text = " ".join(query)
    if phrase:
        query_text = f'abs:"{phrase}" {query_text}'.strip()
    try:
        docs = ads.search(query_text, rows=rows)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    t = RichTable("bibcode", "year", "title", title=f"ADS papers: {query_text}")
    for d in docs:
        t.add_row(str(d.get("bibcode", "")), str(d.get("year", "")),
                  str(d.get("title", ["?"])[0])[:70])
    console.print(t)
    if add and docs:
        try:
            n = ads.add_to_refs([d["bibcode"] for d in docs])
        except Exception as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        console.print(f"[green]+{n} new entries -> refs.bib[/]")
    if report:
        body = "\n".join([
            f"# ADS paper set: {query_text}",
            "",
            *(_paper_lines(docs)),
        ])
        path = _write_report("papers", query_text, body)
        console.print(f"report -> [green]{path}[/]")


@app.command()
def dossier(target: str,
            rows: int = typer.Option(6, help="ADS rows to include"),
            fov: float | None = typer.Option(None, help="field of view in arcmin"),
            images: bool = typer.Option(True, help="render colour and panel images")):
    """Build a compact object packet: resolve, image, literature, Markdown report."""
    from . import ads
    try:
        obj = _resolve_target(target)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    fov_arcmin = fov or (8.0 if any(k in obj["otype"] for k in ("G", "Cl", "Neb", "SNR")) else 3.0)
    hips, survey = cutouts.best_color_hips(obj["dec"])
    color_path = panel_path = None
    if images:
        try:
            color_path = cutouts.color(obj["ra"], obj["dec"], fov_arcmin=fov_arcmin, hips=hips)
            panel_path = cutouts.panel(obj["ra"], obj["dec"], fov_arcmin=fov_arcmin)
        except Exception as e:
            console.print(f"[yellow]imaging unavailable ({type(e).__name__})[/]")
    try:
        docs = ads.search(f'object:"{obj["name"]}"', rows=rows)
    except Exception:
        docs = []
    body = "\n".join([
        f"# Dossier: {obj['name']}",
        "",
        f"- Type: {obj['otype']}",
        f"- Position: RA={obj['ra']:.5f}, Dec={obj['dec']:+.5f}",
        f"- Colour survey: {survey}",
        f"- FOV: {fov_arcmin} arcmin",
        f"- Colour image: {color_path or 'not rendered'}",
        f"- Multi-wavelength panel: {panel_path or 'not rendered'}",
        "",
        "## ADS",
        *_paper_lines(docs),
    ])
    path = _write_report("dossier", obj["name"], body)
    console.print(f"dossier -> [green]{path}[/]")


@app.command(context_settings=_COORD_CONTEXT)
def field(ra: float, dec: float,
          fov: float = typer.Option(5.0, help="field of view in arcmin"),
          images: bool = typer.Option(True, help="render colour and panel images")):
    """Build a compact packet for a sky position, including blank fields."""
    obj = cutouts.identify_field(ra, dec)
    hips, survey = cutouts.best_color_hips(dec)
    color_path = panel_path = None
    if images:
        try:
            color_path = cutouts.color(ra, dec, fov_arcmin=fov, hips=hips)
            panel_path = cutouts.panel(ra, dec, fov_arcmin=fov)
        except Exception as e:
            console.print(f"[yellow]imaging unavailable ({type(e).__name__})[/]")
    nearest = (f"{obj[0]} ({obj[1]}), {obj[2]:.1f} arcsec away"
               if obj else "No SIMBAD object within 2 arcmin")
    body = "\n".join([
        f"# Field: RA={ra:.5f}, Dec={dec:+.5f}",
        "",
        f"- Nearest object: {nearest}",
        f"- Colour survey: {survey}",
        f"- FOV: {fov} arcmin",
        f"- Colour image: {color_path or 'not rendered'}",
        f"- Multi-wavelength panel: {panel_path or 'not rendered'}",
    ])
    path = _write_report("field", f"{ra:.5f}_{dec:+.5f}", body)
    console.print(f"field packet -> [green]{path}[/]")


@app.command()
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
        console.print(f"[red]unknown recipe {recipe!r}; run: python -m access.hub sample list[/]")
        raise typer.Exit(1)
    source = _query_archive(rec.archive)
    try:
        tab = cache.cached_query(f"sample:{recipe}", rec.adql,
                                 lambda: source.query(rec.adql), refresh=refresh)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{rec.description}[/]")
    _print_astropy_table(tab, f"{recipe}: {len(tab)} rows", limit=show)


@app.command()
def atlas_targets(limit: int = typer.Option(6, help="number of curated targets"),
                  fov_scale: float = typer.Option(1.0, help="multiply each target FOV")):
    """Render a curated contact sheet of visually useful astronomy targets."""
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for target in ATLAS_TARGETS[:limit]:
        out = ATLAS_DIR / f"{_slug(target['name'])}.jpg"
        try:
            paths.append(cutouts.color(target["ra"], target["dec"],
                                       fov_arcmin=target["fov"] * fov_scale,
                                       pix=768, out=out))
        except Exception as e:
            console.print(f"[yellow]skip {target['name']} ({type(e).__name__})[/]")
    sheet = _contact_sheet(paths, ATLAS_DIR / "atlas-targets.png", "atlas targets")
    console.print(f"atlas -> [green]{sheet}[/]")


@app.command()
def poster(target: str,
           resolution: str = typer.Option("1080p", help="1080p, 2k, or 4k"),
           style: str = typer.Option("clean", help="clean, label, or science"),
           fov: float | None = typer.Option(None, help="field of view in arcmin")):
    """Render a desktop-wallpaper style astronomy poster."""
    if resolution not in RESOLUTIONS:
        console.print(f"[red]unknown resolution {resolution!r}; choose 1080p, 2k, or 4k[/]")
        raise typer.Exit(1)
    if style not in {"clean", "label", "science"}:
        console.print(f"[red]unknown style {style!r}; choose clean, label, or science[/]")
        raise typer.Exit(1)
    try:
        obj = _resolve_target(target)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    width, height = RESOLUTIONS[resolution]
    fov_arcmin = fov or (10.0 if any(k in obj["otype"] for k in ("G", "Cl", "Neb", "SNR")) else 4.0)
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


if __name__ == "__main__":
    app()
