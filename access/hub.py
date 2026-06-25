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
"""
import typer
from rich.console import Console
from rich.table import Table

from . import resolvers, cutouts

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Desk-astronomy CLI: resolve | image | where | cite.")
console = Console()


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
        t = Table("bibcode", "title", title="recent references")
        for r in bib:
            t.add_row(str(r["bibcode"]), str(r["title"])[:70])
        console.print(t)
    except Exception as e:
        console.print(f"[dim]bibliography unavailable ({type(e).__name__})[/]")


@app.command()
def image(ra: float, dec: float):
    """Smart multi-wavelength + colour cutout of a position (auto survey/FOV)."""
    cutouts.smart(ra, dec)


@app.command()
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
def cite(query: str, add: bool = typer.Option(False, help="append to refs.bib"),
         rows: int = 8):
    """Search ADS/SciX; optionally append the hits to refs.bib (needs a token)."""
    from . import ads  # imported lazily so the rest of the CLI runs token-free
    try:
        docs = ads.search(query, rows=rows)
    except Exception as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    t = Table("bibcode", "year", "title", title=f"ADS: {query}")
    for d in docs:
        t.add_row(d["bibcode"], str(d.get("year", "")),
                  d.get("title", ["?"])[0][:60])
    console.print(t)
    if add and docs:
        n = ads.add_to_refs([d["bibcode"] for d in docs])
        console.print(f"[green]+{n} new entries -> refs.bib[/]")


if __name__ == "__main__":
    app()
