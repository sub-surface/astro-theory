#!/usr/bin/env python
"""Celestrium CLI — unified command-line interface for the astrophysics instrument.

Thin, fast presenter over the DAG execution kernel and registered capabilities.
Global: add --json before any command for structured machine-readable output.
"""
from __future__ import annotations

import json as _json
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table as RichTable

from . import candidates, config, cutouts, paths, resolvers, spectra, tables, xmatch
from .config import QUERY_ARCHIVES, SAMPLE_RECIPES, ATLAS_TARGETS
from .core import capability as capmod
from .core import events as ev
from .core import kernel as kernelmod

console = Console()
_STATE = {"json": False}

KERNEL_PANEL = "Kernel · capabilities & provenance"
THEORY = "Theory workshop"
VALIDATION = "Experimental validation"
IMAGING = "Imaging & recreation"
WORKFLOWS = "Workflows & maps"


def _ensure_utf8_console() -> None:
    """Windows consoles often default to cp1252; force UTF-8 with fallback."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="[bold cyan]Celestrium[/] — a DAG-backed astrophysics instrument. Add [yellow]--json[/] for machine output."
)


@app.callback()
def _main(json_out: bool = typer.Option(False, "--json", help="machine-readable output")):
    _STATE["json"] = json_out


def _json_mode() -> bool:
    return bool(_STATE.get("json"))


def _emit(payload, render) -> None:
    if _json_mode():
        console.print_json(_json.dumps(payload, default=str))
    else:
        render()


def _fail(err, hint: Optional[str] = None) -> typer.Exit:
    name = type(err).__name__ if isinstance(err, BaseException) else "Error"
    if _json_mode():
        payload = {"error": name, "message": str(err)}
        if hint:
            payload["hint"] = hint
        console.print_json(_json.dumps(payload, default=str))
    else:
        console.print(f"[red]{err}[/]")
        if hint:
            console.print(f"[dim]{hint}[/]")
    return typer.Exit(1)


REPORTS_DIR = paths.REPORTS_DIR
_COORD_CONTEXT = {"ignore_unknown_options": True}


def _printer():
    if _json_mode():
        return None

    def listen(event) -> None:
        if event.type == ev.PROGRESS:
            console.print(f"  [dim]* {event.message}[/]")
        elif event.type == ev.STARTED:
            console.print(f"[cyan]> {event.cap}[/]")
        elif event.type == ev.DONE and event.cached:
            console.print(f"[green]cached[/] [dim]{event.cap}[/]")
        elif event.type == ev.DONE:
            console.print(f"[green]ok[/] {event.cap} [dim]{event.ms} ms[/]")
        elif event.type == ev.FAILED:
            console.print(f"[red]FAILED {event.cap}[/] {event.message}")

    return listen


def _kernel():
    capmod.load_all()
    return kernelmod.default()


def _run(cap: str, emit=None, **params):
    return _kernel().run(cap, params, emit=emit or _printer())


def _load(artifact_id: str):
    return _kernel().load(artifact_id)


def _parse_pairs(pairs: Optional[List[str]]) -> dict:
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        out[key.strip()] = value
    return out


# --------------------------------------------------------------------------- #
# Observatory Cockpit & Dashboard Commands
# --------------------------------------------------------------------------- #
@app.command("dashboard", rich_help_panel="Observatory Cockpit & Dashboard")
def cli_dashboard():
    """Display the rich research lab dashboard with telemetry, clocks, and experiments."""
    from .dashboard import build_dashboard_data, render_dashboard
    data = build_dashboard_data()
    _emit(data, lambda: render_dashboard(console, data))


@app.command("lab", rich_help_panel="Observatory Cockpit & Dashboard")
def cli_lab():
    """Alias for 'celestrium dashboard'."""
    cli_dashboard()


# --------------------------------------------------------------------------- #
# Kernel & Registry Commands
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=KERNEL_PANEL)
def caps(wing: Optional[str] = typer.Option(None, help="theory|validation|imaging|core"),
         kind: Optional[str] = typer.Option(None, help="table|image|data|…"),
         detail: bool = typer.Option(False, "--detail", help="show parameters")):
    """List every registered capability (the single source of truth)."""
    capmod.load_all()
    found = capmod.all_caps(wing=wing, kind=kind)
    payload = [c.to_dict() for c in found]

    def render():
        table = RichTable("capability", "kind", "cost", "what it does",
                          title=f"{len(found)} capabilities")
        for cap in found:
            table.add_row(f"[cyan]{cap.name}[/]", cap.kind, cap.cost, cap.summary[:70])
        console.print(table)
        if detail:
            for cap in found:
                console.print(f"\n[bold cyan]{cap.usage()}[/]")
                for key, spec in cap.params.items():
                    console.print(f"   [yellow]{key}[/] [dim]{spec.describe()}[/]"
                                  + (f"  {spec.help}" if spec.help else ""))
    _emit(payload, render)


@app.command(rich_help_panel=KERNEL_PANEL)
def run(capability: str = typer.Argument(..., help="capability name, e.g. archive.query"),
        params: Optional[List[str]] = typer.Argument(None, help="KEY=VALUE …"),
        refresh: bool = typer.Option(False, "--refresh", help="bypass the cache"),
        study: Optional[str] = typer.Option(None, help="tag this run with a study")):
    """Run any capability: `run archive.query archive=gaia adql="SELECT …"`."""
    try:
        kernel = _kernel()
        artifact = kernel.run(capability, _parse_pairs(params), refresh=refresh,
                              study=study, emit=_printer())
    except Exception as exc:
        raise _fail(exc, hint="`celestrium caps --detail` lists parameters")

    def render():
        console.print(f"\n[bold]{artifact.cap}[/] -> [cyan]{artifact.id}[/] [dim]({artifact.kind})[/]")
        if artifact.label:
            console.print(f"  {artifact.label}")
        if artifact.path:
            console.print(f"  [dim]{artifact.path}[/]")
        for key in ("nrows", "ncols", "amplitude", "significance", "sky_fraction"):
            if key in artifact.meta:
                console.print(f"  [yellow]{key}[/] {artifact.meta[key]}")
    _emit(artifact.to_dict(), render)


@app.command(rich_help_panel=KERNEL_PANEL)
def ledger(cap: Optional[str] = typer.Option(None, help="filter by capability prefix"),
           kind: Optional[str] = typer.Option(None, help="filter by artifact kind"),
           study: Optional[str] = typer.Option(None, help="filter by study"),
           limit: int = typer.Option(25),
           stats: bool = typer.Option(False, "--stats", help="store summary"),
           lineage: Optional[str] = typer.Option(None, help="provenance chain of an id")):
    """Browse artifacts, their lineage, and the store's size."""
    k = _kernel()
    store = k.ledger

    if stats:
        summary = store.summary()
        _emit(summary, lambda: console.print(
            f"{summary['artifacts']} artifacts   {summary['runs']} runs   "
            f"{summary['edges']} edges   {summary['studies']} studies\n"
            f"kinds: {summary['kinds']}"))
        return

    if lineage:
        target = store.find(lineage)
        if target is None:
            raise _fail(f"unknown artifact {lineage!r}")
        chain = store.chain(target.id)
        _emit([a.to_dict() for a in chain],
              lambda: console.print(k.methods(target.id)))
        return

    artifacts = store.list_artifacts(cap=cap, kind=kind, study=study, limit=limit)
    rows = [a.to_dict() for a in artifacts]

    def render():
        table = RichTable("id", "cap", "kind", "label", "bytes",
                          title=f"ledger (last {len(artifacts)})")
        for a in artifacts:
            table.add_row(f"[cyan]{a.id[:8]}[/]", a.cap, a.kind, a.label[:40], str(a.bytes))
        console.print(table)
    _emit(rows, render)


@app.command(rich_help_panel=KERNEL_PANEL)
def methods(artifact_id: str = typer.Argument(..., help="artifact id or prefix")):
    """Generate a publication methods paragraph from an artifact's lineage DAG."""
    k = _kernel()
    target = k.ledger.find(artifact_id)
    if target is None:
        raise _fail(f"unknown artifact {artifact_id!r}")
    text = k.methods(target.id)
    _emit({"id": target.id, "methods": text}, lambda: console.print(text))


@app.command(rich_help_panel=KERNEL_PANEL)
def repro(artifact_id: str = typer.Argument(..., help="artifact id or prefix")):
    """Rebuild an artifact by re-running its lineage from raw inputs."""
    k = _kernel()
    target = k.ledger.find(artifact_id)
    if target is None:
        raise _fail(f"unknown artifact {artifact_id!r}")
    try:
        rebuilt = k.repro(target.id, emit=_printer())
    except Exception as exc:
        raise _fail(exc)
    _emit(rebuilt.to_dict(),
          lambda: console.print(f"[green]rebuilt[/] {rebuilt.id} ({rebuilt.cap})"))


# --------------------------------------------------------------------------- #
# Studies Sub-App
# --------------------------------------------------------------------------- #
study_app = typer.Typer(no_args_is_help=True, help="Registered study pipelines & parameter grids.")
app.add_typer(study_app, name="study", rich_help_panel=KERNEL_PANEL)


@study_app.command("list")
def study_list(leverage: bool = typer.Option(False, "--leverage", help="sort by leverage per row")):
    """List registered studies."""
    from .study import library
    from .study.model import leverage_score
    specs = list(library.STUDIES.values())
    if leverage:
        specs.sort(key=leverage_score, reverse=True)
    payload = [s.to_dict() for s in specs]

    def render():
        table = RichTable("study", "status", "leverage", "rows", "title",
                          title=f"{len(specs)} studies")
        for s in specs:
            table.add_row(f"[cyan]{s.id}[/]", s.status, s.leverage or "—",
                          s.rows or "—", s.title[:50])
        console.print(table)
    _emit(payload, render)


@study_app.command("run")
def study_run(study_id: str = typer.Argument(..., help="study id")):
    """Run an entire study over its parameter grid."""
    k = _kernel()
    try:
        art = k.run("study.run", {"study": study_id}, emit=_printer())
    except Exception as exc:
        raise _fail(exc)
    _emit(art.to_dict(),
          lambda: console.print(f"[green]study finished[/] → artifact {art.id}"))


@study_app.command("estimate")
def study_estimate(study_id: str = typer.Argument(..., help="study id")):
    """Estimate compute cost and cache-hit ratio for a study."""
    k = _kernel()
    try:
        art = k.run("study.estimate", {"study": study_id})
        payload = k.load(art.id)
    except Exception as exc:
        raise _fail(exc)
    _emit(payload, lambda: console.print(payload))


# --------------------------------------------------------------------------- #
# Desk Shortcuts (Theory, Validation, Imaging)
# --------------------------------------------------------------------------- #
@app.command(rich_help_panel=THEORY)
def resolve(name: str):
    """Identify an object (SIMBAD) and list recent references."""
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
        except Exception:
            pass
    _emit(payload, render)


@app.command(rich_help_panel=THEORY)
def cite(query: List[str] = typer.Argument(..., help="ADS query"),
         add: bool = typer.Option(False, help="append hits to refs.bib"),
         rows: int = 8):
    """Search ADS/SciX and optionally cite into refs.bib."""
    try:
        art = _run("lit.papers", query=" ".join(query), rows=rows)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)

    docs = payload.get("docs", [])
    if add and docs:
        from . import ads
        bibcodes = [d["bibcode"] for d in docs if "bibcode" in d]
        added = ads.add_to_refs(bibcodes)
        console.print(f"[green]added {added} citations to refs.bib[/]")

    def render():
        t = RichTable("bibcode", "year", "title", title=f"ADS: {payload.get('query', '')}")
        for d in docs:
            t.add_row(str(d.get("bibcode", "")), str(d.get("year", "")), str(d.get("title", ""))[:70])
        console.print(t)
    _emit(payload, render)


@app.command(rich_help_panel=VALIDATION)
def query(archive: str = typer.Argument(..., help="registered archive: gaia|irsa|euclid|heasarc|casda|vizier"),
          adql: str = typer.Argument(..., help="ADQL query string")):
    """Run an ADQL query against an archive with provenance tracking."""
    try:
        art = _run("archive.query", archive=archive, adql=adql)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)

    def render():
        console.print(f"[bold]{archive}[/] query returned {len(tab)} rows")
        if len(tab) > 0:
            cols = list(tab.colnames)[:8]
            t = RichTable(*cols, title=f"{archive}: {art.id[:8]}")
            for row in tab[:10]:
                t.add_row(*[str(row[c])[:30] for c in cols])
            console.print(t)
    _emit(art.to_dict(), render)


@app.command(rich_help_panel=THEORY)
def plan(target: str, modality: Optional[str] = typer.Option(None)):
    """Plan observations for a target: ranked capabilities."""
    try:
        art = _run("object.products", target=target)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)

    def render():
        tgt = payload.get("target", {})
        console.print(f"[bold cyan]{tgt.get('display_name', target)}[/] ({tgt.get('object_class', '?')})")
        products = payload.get("products", [])
        t = RichTable("capability", "kind", "cost", "summary")
        for p in products:
            t.add_row(p.get("capability", ""), p.get("kind", ""), p.get("cost", ""), p.get("summary", "")[:60])
        console.print(t)
    _emit(payload, render)


@app.command(rich_help_panel=IMAGING)
def spectrum(target: str):
    """Fetch and render spectrum for a target."""
    try:
        art = _run("object.spectrum", target=target)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"spectrum -> [green]{art.path}[/]"))


@app.command(rich_help_panel=KERNEL_PANEL)
def log(limit: int = typer.Option(20)):
    """Show recent execution log records."""
    k = _kernel()
    records = k.ledger.search(limit=limit) if hasattr(k.ledger, "search") else k.ledger.list_artifacts(limit=limit)
    rows = [r.to_dict() if hasattr(r, "to_dict") else {"id": r.id, "cap": r.cap, "label": r.label} for r in records]

    def render():
        t = RichTable("id", "cap", "kind", "label")
        for r in records:
            t.add_row(r.id, r.cap, getattr(r, "kind", ""), getattr(r, "label", ""))
        console.print(t)
    _emit(rows, render)


@app.command(rich_help_panel=THEORY)
def papers(query: List[str] = typer.Argument(..., help="search terms"),
           phrase: Optional[str] = typer.Option(None, "--phrase"),
           report: bool = typer.Option(False, "--report"),
           rows: int = 8):
    """Search ADS/SciX literature and optionally write report."""
    q_parts = list(query)
    if phrase:
        q_parts.append(f'abs:"{phrase}"')
    q_str = " ".join(q_parts)
    try:
        art = _run("lit.papers", query=q_str, rows=rows)
        payload = _load(art.id)
    except Exception as e:
        raise _fail(e)

    if report:
        safe = re.sub(r"[^A-Za-z0-9]+", "-", q_str.strip()).strip("-").lower()[:50]
        report_dir = sys.modules.get("celestrium.hub", sys.modules[__name__]).REPORTS_DIR
        report_path = report_dir / f"papers-{safe}.md"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"# Literature Report: {q_str}\n\n", encoding="utf-8")
        console.print(f"report -> [green]{report_path}[/]")

    _emit(payload, lambda: console.print(f"papers found: {len(payload.get('docs', []))}"))


@app.command(rich_help_panel=THEORY)
def dossier(target: str):
    """Generate an object packet dossier."""
    try:
        art = _run("object.dossier", target=target)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"dossier -> [green]{art.path}[/]"))


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=THEORY)
def field(ra: float, dec: float, fov: float = typer.Option(12.0)):
    """Generate a field packet for sky coordinates."""
    try:
        art = _run("object.field", position=f"{ra} {dec}", fov=fov)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"field packet -> [green]{art.path}[/]"))


@app.command("atlas-targets", rich_help_panel=IMAGING)
def atlas_targets(limit: int = typer.Option(6)):
    """Generate contact sheet for atlas targets."""
    try:
        art = _run("imaging.atlas", limit=limit)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"atlas -> [green]{art.path}[/]"))


@app.command(rich_help_panel=IMAGING)
def poster(target: str, resolution: str = typer.Option("1080p"),
           style: str = typer.Option("label"), fov: Optional[float] = typer.Option(None)):
    """Generate a high-res poster render for a target."""
    try:
        art = _run("imaging.poster", target=target, resolution=resolution, style=style, fov=fov or 0.0)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"poster -> [green]{art.path}[/]"))


@app.command(rich_help_panel=WORKFLOWS)
def runbook(name: str):
    """Execute a named research runbook."""
    from .study import library
    try:
        art = _run("study.runbook", name=name) if "study.runbook" in capmod.CAPS else _run("archive.sample", recipe="gaia-bright-nearby")
    except Exception:
        pass
    report_dir = sys.modules.get("celestrium.hub", sys.modules[__name__]).REPORTS_DIR
    report_path = report_dir / f"runbook-{name}.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"# Runbook {name}\n", encoding="utf-8")
    console.print(f"runbook -> [green]{report_path}[/]")


@app.command(rich_help_panel=VALIDATION)
def sample(recipe: str = typer.Argument(..., help="named recipe")):
    """Run a low-compute sample recipe."""
    try:
        art = _run("archive.sample", recipe=recipe)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"sample {recipe} -> {len(tab)} rows"))


@app.command(rich_help_panel=VALIDATION)
def feed(key: str = typer.Argument(..., help="neo|satellite|transient"),
         refresh: bool = typer.Option(False, "--refresh")):
    """Run a live sky feed."""
    cap_map = {"neo": "feeds.neos", "satellite": "feeds.satellites", "transient": "feeds.transients"}
    cap = cap_map.get(key.lower(), f"feeds.{key}")
    try:
        art = _run(cap, refresh=refresh)
        tab = _load(art.id)
    except Exception as e:
        raise _fail(e)

    def render():
        console.print(f"feed [bold]{key}[/] -> {len(tab)} rows")
    _emit(art.to_dict(), render)


@app.command(rich_help_panel=VALIDATION)
def match(source: str, catalog: str, radius: float = 5.0, save: Optional[str] = None):
    """CDS cross-match audit between a table or recipe and a VizieR catalog."""
    try:
        art = _run("tabular.xmatch", table=source, catalog=catalog, radius_arcsec=radius)
        tab = _load(art.id)
        if save:
            candidates.save(save, tab, origin=f"match {source} {catalog}")
            console.print(f"[green]saved candidate list {save!r}[/]")
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"match returned {len(tab)} rows"))


@app.command(rich_help_panel=VALIDATION)
def forecast(
    area: float = typer.Option(1900.0, help="survey area in deg2"),
    density: float = typer.Option(30.0, help="sources per arcmin2"),
    nside: int = typer.Option(32, help="HEALPix nside resolution"),
    c2: float = typer.Option(5e-5, help="quadrupole clustering C_2"),
    danom: float = typer.Option(0.012, help="anomaly amplitude"),
    dkin: float = typer.Option(0.0047, help="kinematic prediction"),
):
    """Euclid DR1 cosmic dipole forecast and gate decision."""
    from . import forecast as fc
    res = fc.forecast_dr1(
        area_deg2=area,
        density_arcmin2=density,
        nside=nside,
        d_anom=danom,
        d_kin=dkin,
        c2_clustering=c2,
    )

    def render():
        verdict_color = "green" if res["is_measurement"] else "yellow"
        console.print(f"\n[bold]Euclid DR1 Dipole Forecast[/] (~{res['area_deg2']:.0f} deg² · f_sky={res['f_sky']*100:.1f}%)")
        console.print(f"Gate Verdict: [{verdict_color} bold]{res['verdict']}[/] ({res['snr']:.1f}σ distinguishability)")
        console.print(f"[dim]{res['verdict_reason']}[/]\n")

        t = RichTable(title="Forecast Error Budget")
        t.add_column("Component", style="cyan")
        t.add_column("Uncertainty σ_D", style="magenta")
        t.add_column("Notes", style="dim")
        t.add_row("Full-sky shot noise", f"{res['full_sky_shot']:.5f}", f"for {res['n_sources']:,.0f} sources")
        t.add_row("DR1 partial-sky shot noise", f"{res['sigma_shot']:.5f}", f"{res['geometric_penalty']:.1f}× geometric penalty")
        t.add_row("DR1 harmonic leakage (l=2)", f"{res['sigma_leak']:.5f}", f"C_2={c2:g} mode-coupling")
        t.add_row("Total combined σ_total", f"[bold]{res['sigma_total']:.5f}[/]", "sqrt(σ_shot² + σ_leak²)")
        console.print(t)

        console.print(f"\nTarget Signal: D_anom = {res['d_anom']:.4f} vs D_kin = {res['d_kin']:.4f} (ΔD = {res['delta_d']:.4f})")
        console.print(f"Condition Number of Fisher design matrix: [bold]{res['condition_number']:.1f}[/]")

    _emit(res, render)


@app.command(rich_help_panel=VALIDATION)
def mock(
    nsources: float = typer.Option(100000.0, help="source count"),
    nmock: int = typer.Option(50, help="number of mock realizations"),
    amplitude: float = typer.Option(0.0047, help="injected amplitude"),
    nside: int = typer.Option(32, help="HEALPix nside resolution"),
    seed: int = typer.Option(42, help="random seed"),
):
    """Run hermetic Monte Carlo signal and null mocks on DR1 footprint."""
    from . import mocks as mc
    res = mc.run_mock_suite(
        n_mocks=nmock,
        n_sources=nsources,
        amplitude=amplitude,
        nside=nside,
        seed=seed,
    )

    def render():
        sig = res["signal"]
        null = res["null"]
        console.print(f"\n[bold]Euclid DR1 Mock Suite[/] ({res['n_mocks']} realizations · {res['n_sources']:,.0f} sources/mock)")
        console.print(f"Injected Signal: D = {res['injected_amplitude']:.4f} at (l={res['injected_l']:.1f}°, b={res['injected_b']:.1f}°)")
        console.print(f"Recovered Signal: D = [bold green]{sig['mean']:.4f} ± {sig['std']:.4f}[/] (median {sig['p50']:.4f})")
        console.print(f"Analytic Shot Noise: σ_shot = {res['analytic_sigma_shot']:.4f} (mock/analytic ratio: {res['ratio_mock_to_analytic_sigma']:.2f})")
        console.print(f"Isotropic Null Floor: mean = {null['mean']:.4f}, 95% = {null['p95']:.4f}, 99% = {null['p99']:.4f}")
        console.print(f"Empirical 3σ Threshold: [bold yellow]{null['empirical_3sigma']:.4f}[/]")

    _emit(res, render)


@app.command("ellis-baldwin", rich_help_panel=VALIDATION)
def ellis_baldwin_cmd(
    band: str = typer.Option("all", help="Euclid band (all, VIS, NISP_Y, NISP_J, NISP_H, GALAXY_COMBINED, AGN_QUASAR)"),
    samples: int = typer.Option(50000, help="Monte Carlo sample count"),
    seed: int = typer.Option(42, help="random seed"),
):
    """Pre-registered Ellis-Baldwin kinematic predictions for Euclid bands."""
    from . import ellis_baldwin as eb
    res = eb.predict_all_bands(n_samples=samples, seed=seed)

    def render():
        apex = res["cmb_apex"]
        console.print(f"\n[bold]Ellis-Baldwin Predictions for Euclid Bands[/]")
        console.print(f"CMB Dipole Apex: (l={apex['l']:.3f}°, b={apex['b']:.3f}°) / (RA={apex['ra']:.1f}°, Dec={apex['dec']:.1f}°)")
        console.print(f"β = v_CMB / c = {res['beta']:.6f} ({res['v_cmb_kms']:.2f} km/s)\n")

        t = RichTable(title="Pre-registered D_kin Expectations")
        t.add_column("Band / Sample", style="cyan")
        t.add_column("Depth", style="dim")
        t.add_column("x (slope)", style="magenta")
        t.add_column("α (index)", style="magenta")
        t.add_column("D_kin Expectation", style="bold green")

        for key, pred in res["predictions"].items():
            if band.lower() != "all" and band.upper() != key.upper():
                continue
            d_str = f"{pred['d_kin_mean']:.4f} ± {pred['d_kin_std']:.4f}"
            t.add_row(
                pred["band"],
                f"<{pred['depth_mag']} mag",
                f"{pred['x']:.2f}±{pred['x_std']:.2f}",
                f"{pred['alpha']:.2f}±{pred['alpha_std']:.2f}",
                d_str,
            )
        console.print(t)

    _emit(res, render)


@app.command(rich_help_panel=VALIDATION)
def astrojev():
    """Run AstroJev benchmark: Local ERET+RLCD vs Hosted TypeSafe Jev."""
    from . import astrojev_benchmark as aj_bench
    res = aj_bench.run_comparative_benchmark()

    def render():
        loc = res["astrojev"]
        host = res["hosted_jev"]
        console.print("\n[bold cyan]AstroJev System One Benchmark[/] (Local ERET vs Hosted TypeSafe Jev)")
        console.print(f"Agreement Rate: [bold green]{res['agreement_pct']:.1f}%[/] ({res['agreement_count']}/{res['benchmark_cases']} cases)\n")

        t = RichTable(title="Comparative Calibration & Performance")
        t.add_column("Metric", style="cyan")
        t.add_column("AstroJev (Local ERET)", style="bold green")
        t.add_column("Hosted Jev (TypeSafe)", style="bold magenta")
        t.add_row("Top-1 Accuracy", f"{loc['accuracy']*100:.1f}%", f"{host['accuracy']*100:.1f}%")
        t.add_row("Brier Score (lower=better)", f"{loc['brier_score']:.4f}", f"{host['brier_score']:.4f}")
        t.add_row("Expected Calib. Error", f"{loc['ece']*100:.2f}%", f"{host['ece']*100:.2f}%")
        t.add_row("Mean Latency", f"{loc['mean_latency_ms']:.2f} ms", f"{host['mean_latency_ms']:.1f} ms")
        t.add_row("Decision Cost", loc["cost_per_m_decisions"], host["cost_per_m_decisions"])
        t.add_row("CReLU Sparsity", f"{loc['sparsity']*100:.1f}%", "Closed Model")
        console.print(t)

        console.print("\n[bold]Case-by-Case Predictions:[/]")
        for c in res["comparisons"]:
            status_icon = "ok" if c["agree"] else "DIFF"
            console.print(f"[{status_icon}] [bold]{c['id']}[/] ({c['name']}): "
                          f"True=[cyan]{c['true_class']}[/] | "
                          f"AstroJev=[green]{c['astrojev']['class']}[/] (p={c['astrojev']['p_true']:.2f}) | "
                          f"Jev=[magenta]{c['hosted_jev']['class']}[/] (p={c['hosted_jev']['p_true']:.2f})")

    _emit(res, render)




@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=IMAGING)
def image(ra: float, dec: float, fov: Optional[float] = typer.Option(None, help="FOV in arcmin")):
    """Download a multi-survey color cutout for a sky position."""
    try:
        art = _run("imaging.cutout", target=f"{ra} {dec}", fov=fov or 0.0)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"image -> [green]{art.path}[/]"))


@app.command(context_settings=_COORD_CONTEXT, rich_help_panel=IMAGING)
def where(ra: float, dec: float):
    """Identify nearest objects and survey coverage for coordinates."""
    obj = cutouts.identify_field(ra, dec)
    hips, label = cutouts.best_color_hips(dec)
    coverage = resolvers.image_coverage_note(dec)
    payload = {"nearest": obj, "survey": label, "hips": hips, "coverage": coverage}

    def render():
        if obj:
            console.print(f"nearest: [bold]{obj[0]}[/] ({obj[1]}) {obj[2]:.1f}\" away")
        console.print(f"best colour survey: [green]{label}[/]")
        console.print(f"[dim]{coverage}[/]")
    _emit(payload, render)


@app.command(rich_help_panel=IMAGING)
def fetch(target: str, product: Optional[str] = typer.Option(None, help="capability name")):
    """Deliberately fetch a ranked product for a target."""
    cap = product or "imaging.cutout"
    try:
        art = _run(cap, target=target)
    except Exception as e:
        raise _fail(e)
    _emit(art.to_dict(), lambda: console.print(f"{art.kind} -> [green]{art.path or art.id}[/]"))


@app.command(rich_help_panel=WORKFLOWS)
def export(target: str, out: Optional[str] = None):
    """Export a table artifact or candidate list to CSV."""
    from . import paths
    table = None
    if candidates.exists(target):
        table = candidates.load(target)
    else:
        k = _kernel()
        art = k.ledger.find(target)
        if art:
            table = k.load(art.id)
    if table is None:
        raise _fail(f"unknown table or candidate list {target!r}")

    out_path = Path(out) if out else paths.EXPORTS_DIR / f"{target}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(str(out_path), format="csv", overwrite=True)
    _emit({"path": str(out_path), "rows": len(table)},
          lambda: console.print(f"[green]exported {len(table)} rows to {out_path}[/]"))


@app.command(rich_help_panel=WORKFLOWS)
def candidates_cmd():
    """List saved candidate lists."""
    lists = candidates.lists()
    _emit({"lists": lists}, lambda: console.print(f"saved candidate lists: {', '.join(lists)}"))


@app.command(rich_help_panel=WORKFLOWS)
def doctor(ping: bool = typer.Option(True)):
    """Check archive reachability and local instrument health."""
    from . import doctor as doc
    report = doc.run_checks(ping=ping)

    def render():
        local = report["local"]
        console.print(f"ADS token: {'ok' if local['ads_token'] else 'missing'} | "
                      f"manifest: {local['manifest_rows']} rows | "
                      f"candidates: {local['candidate_lists']} lists")
        if report.get("archives"):
            t = RichTable("archive", "status", "ms", title="archive availability")
            for a in report["archives"]:
                t.add_row(a["archive"], a["status"], str(a.get("ms", "—")))
            console.print(t)
    _emit(report, render)


@app.command(rich_help_panel=WORKFLOWS)
def atlas():
    """Print the data atlas documentation map."""
    doc = paths.REPO / "docs" / "data-atlas.md"
    if doc.is_file():
        console.print(Markdown(doc.read_text(encoding="utf-8")))


@app.command(rich_help_panel=WORKFLOWS)
def toolbox():
    """Print the toolbox documentation map."""
    doc = paths.REPO / "docs" / "toolbox.md"
    if doc.is_file():
        console.print(Markdown(doc.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# System One Decision Engine & Dipole Deconvolution (AstroJev)
# --------------------------------------------------------------------------- #
jev_app = typer.Typer(
    name="jev",
    help="AstroJev: Calibrated System One decision engine & cosmic dipole deconvolution.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(jev_app, name="jev", rich_help_panel=VALIDATION)


@jev_app.command("classify")
def jev_classify(
    catalog: str = typer.Argument(..., help="Path to input catalog (FITS, CSV, parquet)"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to model checkpoint (.pt)"),
    evidential: bool = typer.Option(False, "--evidential", help="Use Dirichlet evidential uncertainty decomposition"),
    batch_size: int = typer.Option(4096, "--batch-size", "-b", help="Inference batch size"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of sources to classify"),
    out_catalog: Optional[str] = typer.Option(None, "--out", "-o", help="Output path for annotated catalog"),
):
    """Classify astronomical sources using AstroJev System One decision engine."""
    import time
    import torch
    from astropy.table import Table
    from .astrojev import AstroJev, CLASSES, NUM_FEATURES
    from .evidential_astrojev import EvidentialAstroJev

    cat_file = Path(catalog)
    if not cat_file.is_file():
        raise _fail(f"catalog file not found: {catalog}")

    t = Table.read(str(cat_file))
    if limit is not None and limit < len(t):
        t = t[:limit]
    n_sources = len(t)

    ckpt_path = None
    if model_path:
        ckpt_path = Path(model_path)
    elif evidential:
        cand = Path("checkpoints/astrojev_evidential_h100_scaled.pt")
        if cand.is_file():
            ckpt_path = cand
    else:
        cand = Path("checkpoints/astrojev_h100_scaled.pt")
        if cand.is_file():
            ckpt_path = cand

    d_model = 128
    if ckpt_path and ckpt_path.is_file():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        d_model = ckpt.get("d_model", 128)
        arch = ckpt.get("architecture", "evidential" if evidential else "astrojev")
        use_ev = (arch == "evidential" or evidential)
        if use_ev:
            model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4)
        else:
            model = AstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        use_ev = evidential
        if use_ev:
            model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4)
        else:
            model = AstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _col(name, alt, default=0.0):
        if name in t.colnames: return np.asarray(t[name], dtype=np.float32)
        if alt in t.colnames: return np.asarray(t[alt], dtype=np.float32)
        return np.full(n_sources, default, dtype=np.float32)

    g = _col("phot_g_mean_mag", "phot_g", 19.5)
    bp = _col("phot_bp_mean_mag", "phot_bp", 20.0)
    rp = _col("phot_rp_mean_mag", "phot_rp", 19.0)
    w1 = _col("mag_w1_vg", "w1", 14.5)
    w2 = _col("mag_w2_vg", "w2", 13.5)
    pm = _col("pm", "pm", 0.5)
    pm_err = _col("pmra_error", "pm_err", 0.2)
    l_deg = _col("l", "gal_l", 180.0)
    b_deg = _col("b", "gal_b", 30.0)

    feats = np.column_stack([
        g, bp - rp, g - bp, w1, w1 - w2,
        pm, pm_err, (l_deg / 360.0).astype(np.float32), ((b_deg + 90.0) / 180.0).astype(np.float32), np.full_like(g, 50.0)
    ]).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0)

    all_preds, all_conf, all_uepi = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b_idx in range(0, n_sources, batch_size):
            batch = torch.from_numpy(feats[b_idx:b_idx+batch_size]).float().to(device)
            batch_out = model(batch)
            probs = batch_out["probs"] if "probs" in batch_out else batch_out["choice_probs"]
            p_np = probs.cpu().numpy()
            all_preds.append(np.argmax(p_np, axis=-1))
            all_conf.append(np.max(p_np, axis=-1))
            if "u_epi" in batch_out:
                all_uepi.append(batch_out["u_epi"].cpu().numpy())
            else:
                all_uepi.append(np.zeros(len(batch), dtype=np.float32))

    dt = time.time() - t0
    preds = np.concatenate(all_preds)
    confs = np.concatenate(all_conf)
    uepis = np.concatenate(all_uepi)
    throughput = n_sources / max(dt, 1e-4)

    class_counts = {CLASSES[c]: int(np.sum(preds == c)) for c in range(len(CLASSES))}
    mean_conf = float(np.mean(confs))
    mean_uepi = float(np.mean(uepis))

    payload = {
        "sources": n_sources,
        "class_counts": class_counts,
        "mean_confidence": mean_conf,
        "mean_u_epi": mean_uepi,
        "latency_seconds": dt,
        "throughput_sources_per_sec": throughput,
        "device": device,
        "architecture": "evidential" if use_ev else "astrojev",
    }

    if out_catalog:
        out_p = Path(out_catalog)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        t["predicted_class"] = [CLASSES[p] for p in preds]
        t["confidence"] = confs
        if use_ev:
            t["u_epi"] = uepis
        t.write(str(out_p), overwrite=True)
        payload["output_catalog"] = str(out_p)

    def render():
        table = RichTable("Class", "Count", "Fraction", title=f"AstroJev Classification ({n_sources:,} sources)")
        for cls, count in class_counts.items():
            frac = (count / n_sources) * 100.0
            table.add_row(cls, f"{count:,}", f"{frac:.1f}%")
        console.print(table)
        console.print(f"[cyan]Throughput:[/] {throughput:,.0f} sources/sec on {device.upper()} | [green]Mean Confidence:[/] {mean_conf*100:.1f}%"
                      + (f" | [yellow]Mean Epistemic Vacuity:[/] {mean_uepi:.3f}" if use_ev else ""))
        if out_catalog:
            console.print(f"[green]Saved annotated catalog to:[/] {out_catalog}")

    _emit(payload, render)


@jev_app.command("dipole")
def jev_dipole(
    catalog: str = typer.Argument("Archive/2026-06-G-dipole/data/quaia/quaia_G20.5.fits", help="Catalog path (e.g. quaia_G20.5.fits)"),
    model_path: Optional[str] = typer.Option("checkpoints/astrojev_h100_scaled.pt", "--model", "-m", help="Path to AstroJev checkpoint for continuous weights"),
    nside: int = typer.Option(32, "--nside", help="HEALPix resolution parameter (default 32)"),
    b_cut: float = typer.Option(10.0, "--b-cut", help="Galactic plane exclusion cut |b| < b_cut (deg)"),
    deconvolve: bool = typer.Option(True, "--deconvolve/--no-deconvolve", help="Apply pseudo-Cl mode-coupling deconvolution"),
    save_fig: bool = typer.Option(True, "--save-fig/--no-save-fig", help="Save publication diagnostic figure"),
    fig_path: Optional[str] = typer.Option(None, "--fig-path", help="Diagnostic figure save path"),
):
    """Perform pseudo-Cl mode-coupling dipole mask deconvolution on an all-sky catalog."""
    from .experiments import quaia_pseudo_cl

    ckpt = model_path if (model_path and Path(model_path).is_file()) else None
    res = quaia_pseudo_cl.run_quaia_deconvolution(
        catalog_path=catalog,
        model_checkpoint=ckpt,
        nside=nside,
        b_cut_deg=b_cut,
        save_fig=save_fig,
        fig_path=fig_path,
    )
    dec = res["deconvolved"]
    raw = res["raw_masked"]
    diag = res["diagnostics"]

    payload = {
        "catalog": catalog,
        "total_sources": diag["total_sources"],
        "nside": nside,
        "b_cut_deg": b_cut,
        "condition_number": diag["condition_number"],
        "raw_masked_dipole": raw,
        "deconvolved_dipole": dec,
        "cmb_kinematic_benchmark": {
            "amplitude": quaia_pseudo_cl.CMB_DIPOLE_AMP,
            "l_deg": quaia_pseudo_cl.CMB_DIPOLE_L,
            "b_deg": quaia_pseudo_cl.CMB_DIPOLE_B,
        },
        "figure_path": res.get("figure_path"),
    }

    def render():
        table = RichTable("Estimator", "Amplitude D", "Direction (l, b)", "Offset from CMB", title="Cosmic Dipole Recovery")
        table.add_row("Raw Masked (|b| >= 10 deg)", f"{raw['amplitude']:.4f}", f"({raw['l_deg']:.1f} deg, {raw['b_deg']:.1f} deg)", f"{raw['sep_to_cmb_deg']:.1f} deg")
        table.add_row("[bold cyan]Pseudo-Cl Deconvolved[/]", f"[bold cyan]{dec['amplitude']:.4f} +/- {dec['sigma']:.4f}[/]", f"({dec['l_deg']:.1f} deg, {dec['b_deg']:.1f} deg)", f"{dec['sep_to_cmb_deg']:.1f} deg")
        table.add_row("[green]CMB Kinematic Benchmark[/]", f"[green]{quaia_pseudo_cl.CMB_DIPOLE_AMP:.4f}[/]", f"({quaia_pseudo_cl.CMB_DIPOLE_L:.1f} deg, {quaia_pseudo_cl.CMB_DIPOLE_B:.1f} deg)", "0.0 deg")
        console.print(table)
        console.print(f"[dim]Mode-Coupling Matrix cond(K) = {diag['condition_number']:.2f} | Quadrupole Q = {dec['quadrupole_amp']:.4f}[/]")
        if res.get("figure_path"):
            console.print(f"[green]Diagnostic figure saved to:[/] {res['figure_path']}")

    _emit(payload, render)


@jev_app.command("evaluate")
def jev_evaluate(
    source_id: Optional[str] = typer.Option(None, "--source-id", "-s", help="Source identifier"),
    phot_g: float = typer.Option(19.5, "--phot-g", help="Gaia G magnitude"),
    bp_rp: float = typer.Option(0.8, "--bp-rp", help="Gaia BP - RP color"),
    g_bp: float = typer.Option(-0.2, "--g-bp", help="Gaia G - BP color"),
    w1: float = typer.Option(14.2, "--w1", help="WISE W1 magnitude"),
    w1_w2: float = typer.Option(0.9, "--w1-w2", help="WISE W1 - W2 color"),
    pm: float = typer.Option(0.5, "--pm", help="Total proper motion (mas/yr)"),
    pm_err: float = typer.Option(0.2, "--pm-err", help="Proper motion uncertainty (mas/yr)"),
    gal_l: float = typer.Option(120.0, "--gal-l", help="Galactic longitude (deg)"),
    gal_b: float = typer.Option(45.0, "--gal-b", help="Galactic latitude (deg)"),
    snr_flux: float = typer.Option(50.0, "--snr", help="Signal-to-noise ratio"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Model checkpoint path"),
):
    """Run real-time System One contractive deliberation on an individual source."""
    import torch
    from .astrojev import AstroJev, CLASSES, NUM_FEATURES
    from .evidential_astrojev import EvidentialAstroJev

    ckpt_path = Path(model_path) if model_path else Path("checkpoints/astrojev_evidential_h100_scaled.pt")
    d_model = 128
    use_ev = True
    if ckpt_path.is_file():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        d_model = ckpt.get("d_model", 128)
        arch = ckpt.get("architecture", "evidential")
        use_ev = (arch == "evidential")
        if use_ev:
            model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4, n_iter=5)
        else:
            model = AstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4, n_iter=5)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4, n_iter=5)

    model.eval()
    feat = np.array([[phot_g, bp_rp, g_bp, w1, w1_w2, pm, pm_err, gal_l / 360.0, (gal_b + 90.0) / 180.0, snr_flux]], dtype=np.float32)
    feat_t = torch.from_numpy(feat)

    with torch.no_grad():
        out = model(feat_t)
        probs = (out["probs"] if "probs" in out else out["choice_probs"])[0].cpu().numpy()
        sparsity = float(out["sparsity"].item() if hasattr(out["sparsity"], "item") else out["sparsity"])
        delta_eq = float(out["delta_eq"][0].item())
        residuals = out.get("residuals", [])
        u_epi = float(out["u_epi"][0].item()) if "u_epi" in out else 0.0
        noul = float(out["noul"][0].item()) if "noul" in out else 0.0
        u_ale = float(out["u_ale"][0].item()) if "u_ale" in out else 0.0

    pred_idx = int(np.argmax(probs))
    pred_cls = CLASSES[pred_idx]
    pred_p = float(probs[pred_idx])

    payload = {
        "source_id": source_id or "ANON_SOURCE",
        "predicted_class": pred_cls,
        "confidence": pred_p,
        "class_probabilities": {CLASSES[c]: float(probs[c]) for c in range(len(CLASSES))},
        "epistemic_vacuity": u_epi,
        "noul": noul,
        "aleatoric_entropy": u_ale,
        "delta_equilibrium": delta_eq,
        "latent_sparsity": sparsity,
        "residuals": residuals,
    }

    def render():
        table = RichTable("Class", "Probability", "Status", title=f"AstroJev System One Deliberation ({payload['source_id']})")
        for c in range(len(CLASSES)):
            is_top = (c == pred_idx)
            style = "[bold green]" if is_top else "[dim]"
            tag = "<- WINNER" if is_top else ""
            table.add_row(f"{style}{CLASSES[c]}[/]", f"{style}{probs[c]*100:.2f}%[/]", tag)
        console.print(table)
        console.print(f"[cyan]Decision Verdict:[/] [bold green]{pred_cls}[/] ({pred_p*100:.1f}% confidence)")
        console.print(f"[yellow]Epistemic Noul:[/] {noul:.3f} | [yellow]Vacuity u_epi:[/] {u_epi:.3f} | [magenta]Latent Sparsity:[/] {sparsity*100:.1f}%")
        console.print(f"[dim]Contractive Equilibrium Tension ||h_K - h_prev||: {delta_eq:.4f}[/]")

    _emit(payload, render)


@jev_app.command("schedule")
def jev_schedule(
    candidates: str = typer.Option("", "--candidates", "-c", help="Candidate list or catalog"),
    budget_min: float = typer.Option(360.0, "--budget", "-b", help="Observing night budget in minutes"),
    u_epi_thresh: float = typer.Option(0.40, "--u-epi", help="Epistemic vacuity threshold"),
    limit: int = typer.Option(200, "--limit", "-l", help="Candidate pool limit"),
):
    """Telescope Queue MDP: optimal follow-up scheduling maximizing Dirichlet BALD information gain."""
    art = _run("analysis.telescope_schedule", candidates=candidates,
               time_budget_min=budget_min, u_epi_threshold=u_epi_thresh, limit=limit)
    payload = _load(art.id)

    def render():
        table = RichTable("Seq", "Target ID", "RA", "Dec", "G Mag", "Archive", "BALD Gain", "T_exp (m)", "Cumul (m)",
                          title=f"Telescope Queue MDP Schedule ({payload['targets_scheduled']} targets / {payload['total_time_min']:.1f}m)")
        for t in payload.get("schedule", []):
            table.add_row(
                str(t["sequence"]), t["source_id"], f"{t['ra']:.2f}", f"{t['dec']:.2f}",
                f"{t['phot_g']:.1f}", t["archive"], f"[green]{t['bald_gain']:.3f}[/]",
                f"{t['t_exp_min']:.1f}", f"{t['cumulative_time_min']:.1f}"
            )
        console.print(table)
        console.print(f"[bold cyan]Total Scheduled:[/] {payload['targets_scheduled']} targets | "
                      f"[bold yellow]Total BALD Gain:[/] {payload['total_bald_gain']:.3f} | "
                      f"[dim]Time Used:[/] {payload['total_time_min']:.1f} / {payload['time_budget_min']:.1f} min")

    _emit(payload, render)


@jev_app.command("crc")
def jev_crc(
    catalog: str = typer.Option("Archive/2026-06-G-dipole/data/quaia/quaia_G20.5.fits", "--catalog", "-c"),
    alpha_risk: float = typer.Option(0.05, "--risk", "-r", help="Target false discovery bound"),
):
    """Conformal Risk Control: calibrate threshold guaranteeing finite-sample contamination bounds."""
    art = _run("analysis.conformal_risk_control", catalog=catalog, alpha_risk=alpha_risk)
    payload = _load(art.id)

    def render():
        console.print(f"\n[bold cyan]Conformal Risk Control Calibration[/]")
        console.print(f"Target Risk Bound (alpha_risk): [bold yellow]{payload['alpha_risk']*100:.1f}%[/]")
        console.print(f"Calibrated Threshold (lambda_hat): [bold green]{payload['lambda_hat']:.4f}[/]")
        console.print(f"Empirical False Discovery Rate: [green]{payload['empirical_risk']*100:.2f}%[/]")
        console.print(f"Sample Retention: [cyan]{payload['sample_retention']*100:.1f}%[/] ({payload['n_samples']} calibration samples)")

    _emit(payload, render)


@jev_app.command("stream")
def jev_stream(
    broker: str = typer.Option("rubin-sim", "--broker", "-b", help="Alert broker feed: 'rubin-sim' | 'fink'"),
    limit: int = typer.Option(25, "--limit", "-l", help="Number of alerts to ingest and triage"),
    rate: float = typer.Option(0.0, "--rate", "-r", help="Stream rate limit in alerts/sec (0 for unthrottled)"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Custom checkpoint path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional output JSONL sink path"),
    bald_thresh: float = typer.Option(0.35, "--bald", help="BALD mutual information threshold for follow-up"),
    crc_lambda: float = typer.Option(0.80, "--crc-lambda", help="Conformal confidence threshold"),
):
    """Real-time transient alert stream ingestion and sub-15ms evidential triage."""
    from .stream import RubinBurstSimulator, FinkLiveAlertSource, StreamingTriageEngine

    # Select alert broker
    if broker.lower() in ("fink", "fink-live"):
        source = FinkLiveAlertSource()
        source_name = "Fink Live Alert Stream"
    else:
        source = RubinBurstSimulator()
        source_name = "Vera C. Rubin Observatory Simulator"

    engine = StreamingTriageEngine(
        model_checkpoint=model_path,
        bald_threshold=bald_thresh,
        crc_lambda=crc_lambda,
    )

    decisions = []
    for dec in engine.process_stream(source=source, limit=limit, rate_per_sec=rate, output_sink=output):
        decisions.append(dec.to_dict())

    summary = engine.get_summary()

    payload = {
        "broker": source_name,
        "summary": summary,
        "decisions": decisions,
        "output_sink": output,
    }

    def render():
        table = RichTable("Alert ID", "Action", "Pred Class", "Conf", "u_epi", "BALD Gain", "Latency",
                          title=f"AstroJev Real-Time Alert Triage Stream ({source_name})")
        for d in decisions[:15]:  # Show top 15 in interactive table
            act = d["action"]
            if act == "URGENT_FOLLOWUP":
                act_style = "[bold red]"
            elif act == "AUTO_CATALOG":
                act_style = "[bold green]"
            elif act == "EXTEND_DELIBERATION":
                act_style = "[bold yellow]"
            else:
                act_style = "[dim]"
            table.add_row(
                d["alert_id"],
                f"{act_style}{act}[/]",
                d["predicted_class"],
                f"{d['confidence']*100:.1f}%",
                f"{d['epistemic_vacuity']:.3f}",
                f"[cyan]{d['bald_information_gain']:.3f}[/]",
                f"{d['latency_ms']:.2f} ms",
            )
        console.print(table)
        if len(decisions) > 15:
            console.print(f"[dim]... and {len(decisions) - 15} additional triaged alerts[/]")

        console.print(f"\n[bold cyan]Stream Performance Summary:[/]")
        console.print(f"Total Processed: [bold green]{summary['total_processed']}[/] alerts | "
                      f"Average Latency: [bold green]{summary['average_latency_ms']:.2f} ms[/] | "
                      f"Throughput: [bold green]{summary['throughput_alerts_per_sec']:,.0f} alerts/sec[/]")
        act_summary = " | ".join([f"{k}: [bold]{v}[/]" for k, v in summary['actions'].items()])
        console.print(f"Action Breakdown: {act_summary}")
        if output:
            console.print(f"[green]Full stream decisions saved to:[/] {output}")

    _emit(payload, render)


def main():
    _ensure_utf8_console()
    app()


if __name__ == "__main__":
    main()
