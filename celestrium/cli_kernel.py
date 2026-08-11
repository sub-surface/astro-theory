"""Kernel-era CLI commands, generated from the capability registry.

`hub.py` stays the presenter it always was; this module hands it a set of
commands that are *derived* rather than hand-written, so a new capability shows
up in `--help` with no CLI edit at all. That is the parity guarantee made
concrete: there is no list of commands to forget to update.
"""
from __future__ import annotations

import json as _json
import numpy as np
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table as RichTable

from . import paths
from .core import capability as capmod
from .core import events as ev
from .core import kernel as kernelmod

console = Console()
_STATE = {"json": False}

KERNEL_PANEL = "Kernel · capabilities & provenance"
STUDY_PANEL = "Studies & notebook"


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


def _printer():
    """Live progress for interactive runs; silent under --json."""
    if _json_mode():
        return None

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


def _kernel():
    capmod.load_all()
    return kernelmod.default()


def _parse_pairs(pairs: Optional[List[str]]) -> dict:
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        out[key.strip()] = value
    return out


def _artifact_row(artifact) -> dict:
    return {"id": artifact.id, "cap": artifact.cap, "kind": artifact.kind,
            "label": artifact.label, "rows": artifact.meta.get("nrows", ""),
            "created": artifact.created, "study": artifact.study or ""}


def register(app: typer.Typer, state: Optional[dict] = None) -> None:
    """Mount the kernel commands onto the hub's Typer app."""
    if state is not None:
        globals()["_STATE"] = state

    # ----- capabilities ---------------------------------------------------- #
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
                table.add_row(f"[cyan]{cap.name}[/]", cap.kind, cap.cost,
                              cap.summary[:70])
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
            study: Optional[str] = typer.Option(None, help="tag this run with a study"),
            publish: bool = typer.Option(False, "--publish",
                                         help="also write it into the vault notebook")):
        """Run any capability: `run archive.query archive=gaia adql="SELECT …"`."""
        try:
            kernel = _kernel()
            artifact = kernel.run(capability, _parse_pairs(params), refresh=refresh,
                                  study=study, emit=_printer())
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc, hint="`celestrium caps --detail` lists parameters")
        if publish:
            try:
                kernel.run("notebook.log", {"artifact": artifact.id}, emit=_printer())
            except Exception as exc:                              # noqa: BLE001
                console.print(f"[yellow]published nothing:[/] {exc}")

        def render():
            console.print(f"\n[bold]{artifact.cap}[/] → [cyan]{artifact.id}[/] "
                          f"[dim]({artifact.kind})[/]")
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
        store = _kernel().ledger
        if stats:
            summary = store.stats()

            def render_stats():
                console.print("[bold]ledger[/]")
                console.print(f"  artifacts   {summary['artifacts']}")
                console.print(f"  runs        {summary['runs']} "
                              f"([green]{summary['cache_hits']} cached[/])")
                console.print(f"  on disk     {summary['bytes'] / 1e6:.1f} MB")
                console.print(f"  db          [dim]{summary['db']}[/]")
                for key, count in sorted(summary["by_kind"].items()):
                    console.print(f"    {key:<10} {count}")
            return _emit(summary, render_stats)

        if lineage:
            chain = store.chain(lineage)
            if not chain:
                raise _fail(f"no artifact {lineage!r}")

            def render_chain():
                console.print(f"[bold]provenance chain[/] ({len(chain)} steps)")
                for depth, artifact in enumerate(chain):
                    console.print(f"  {'  ' * depth}[cyan]{artifact.id}[/] "
                                  f"{artifact.cap} [dim]{artifact.label}[/]")
            return _emit([a.to_dict() for a in chain], render_chain)

        found = store.search(cap=cap, kind=kind, study=study, limit=limit)
        rows = [_artifact_row(a) for a in found]

        def render():
            table = RichTable("id", "capability", "kind", "rows", "label", "created",
                              title=f"{len(rows)} artifacts")
            for row in rows:
                table.add_row(f"[cyan]{row['id']}[/]", row["cap"], row["kind"],
                              str(row["rows"]), str(row["label"])[:40],
                              str(row["created"])[:16])
            console.print(table)
        _emit(rows, render)

    @app.command(name="ledger-import", rich_help_panel=KERNEL_PANEL)
    def ledger_import(dry_run: bool = typer.Option(False, "--dry-run")):
        """Fold the legacy `data/manifest.jsonl` into the ledger.

        Rows whose archive is a registered one are imported as real
        `archive.query` artifacts, so their content addresses match and the old
        pulls become cache hits for the new kernel rather than dead history.
        """
        from . import cache, registry
        from .core.artifact import Artifact, artifact_id, now_utc

        store = _kernel().ledger
        imported = skipped = 0
        for record in cache.manifest():
            archive = str(record.get("archive", ""))
            adql = str(record.get("query", ""))
            cache_file = record.get("cache_file")
            if not cache_file or not (paths.REPO / cache_file).exists():
                skipped += 1
                continue
            if archive in registry.ARCHIVES:
                cap, params = "archive.query", {"archive": archive, "adql": adql}
            else:
                cap, params = "legacy.pull", {"archive": archive, "query": adql}
            aid = artifact_id(cap, params, ())
            if store.get(aid) is not None:
                skipped += 1
                continue
            if not dry_run:
                store.put(Artifact(
                    id=aid, kind="table", cap=cap, params=params, path=cache_file,
                    meta={"nrows": record.get("nrows", 0), "_inputs": [],
                          "imported_from": "manifest.jsonl",
                          "legacy_hash": record.get("hash", "")},
                    label=f"{archive}: {adql[:50]}",
                    created=record.get("utc", now_utc())))
            imported += 1
        _emit({"imported": imported, "skipped": skipped, "dry_run": dry_run},
              lambda: console.print(
                  f"[green]{'would import' if dry_run else 'imported'}[/] "
                  f"{imported} manifest rows ([dim]{skipped} skipped[/])"))

    @app.command(rich_help_panel=KERNEL_PANEL)
    def repro(artifact_id: str = typer.Argument(..., help="artifact id or prefix")):
        """Rebuild an artifact's entire chain from its content address."""
        try:
            rebuilt = _kernel().repro(artifact_id, emit=_printer())
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc)
        _emit(rebuilt.to_dict(),
              lambda: console.print(f"[green]rebuilt[/] [cyan]{rebuilt.id}[/] "
                                    f"{rebuilt.cap}"))

    @app.command(rich_help_panel=KERNEL_PANEL)
    def methods(artifact_id: str = typer.Argument(...)):
        """Print a methods paragraph for an artifact, generated from its lineage."""
        text = _kernel().methods(artifact_id)
        if not text:
            raise _fail(f"no artifact {artifact_id!r}")
        _emit({"artifact": artifact_id, "methods": text},
              lambda: console.print(text))

    # ----- studies --------------------------------------------------------- #
    study_app = typer.Typer(no_args_is_help=True, help="Claims, pipelines, grids.")
    app.add_typer(study_app, name="study", rich_help_panel=STUDY_PANEL)

    @study_app.command("list")
    def study_list():
        """Every study — code pipelines merged with the vault's claims."""
        from .study import library
        from .vault import sync as vault_sync
        merged = library.merge_vault(vault_sync.read_studies_safe())
        rows = [s.to_dict() for s in merged.values()]

        def render():
            table = RichTable("study", "status", "leverage", "rows", "runs",
                              "pipeline", title=f"{len(rows)} studies")
            for entry in sorted(rows, key=lambda r: r["id"]):
                table.add_row(f"[cyan]{entry['id']}[/]", entry["status"],
                              entry["leverage"] or "—", entry["rows"] or "—",
                              str(entry["runs"]) if entry["pipeline"] else "—",
                              f"{len(entry['pipeline'])} steps"
                              if entry["pipeline"] else "[dim]claim only[/]")
            console.print(table)
        _emit(rows, render)

    @study_app.command("show")
    def study_show(study_id: str = typer.Argument(...)):
        """The full study: claim, refutation, pipeline, grid, preregistration."""
        from .study import library
        from .vault import sync as vault_sync
        merged = library.merge_vault(vault_sync.read_studies_safe())
        if study_id not in merged:
            raise _fail(f"unknown study {study_id!r}",
                        hint=f"have {', '.join(sorted(merged))}")
        spec = merged[study_id]
        payload = spec.to_dict()

        def render():
            console.print(f"[bold cyan]{spec.title or spec.id}[/]  "
                          f"[dim]{spec.status}[/]")
            if spec.claim:
                console.print(f"\n{spec.claim}")
            if spec.signature:
                console.print(f"\n[green]signature[/]  {spec.signature}")
            if spec.refutation:
                console.print(f"[red]refutation[/] {spec.refutation}")
            if spec.pipeline:
                console.print(f"\n[bold]pipeline[/] ({len(spec.combos())} combinations)")
                for index, entry in enumerate(spec.pipeline, 1):
                    console.print(f"  {index}. {entry.describe()}")
                console.print("\n[bold]grid[/]")
                for key, values in sorted(spec.grid.items()):
                    console.print(f"  [yellow]{key}[/] {values}")
                console.print(f"\n[dim]preregistered analysis: "
                              f"{spec.prereg_hash()}[/]")
            if spec.note:
                console.print(f"\n[dim]{spec.note}[/]")
        _emit(payload, render)

    @study_app.command("estimate")
    def study_estimate_cmd(study_id: str = typer.Argument(...)):
        """What it would cost before you commit."""
        try:
            artifact = _kernel().run("study.estimate", {"study": study_id},
                                     emit=_printer())
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc)
        payload = _kernel().load(artifact.id)
        _emit(payload, lambda: console.print(payload))

    @study_app.command("run")
    def study_run_cmd(study_id: str = typer.Argument(...),
                      grid: Optional[str] = typer.Option(None, help="JSON grid override"),
                      limit: int = typer.Option(0, help="cap combinations"),
                      refresh: bool = typer.Option(False, "--refresh"),
                      sync: bool = typer.Option(False, "--sync",
                                                help="write results to the vault note")):
        """Run the grid and produce the result surface."""
        kernel = _kernel()
        try:
            artifact = kernel.run(
                "study.run",
                {"study": study_id, "grid": _json.loads(grid) if grid else None,
                 "limit": limit},
                study=study_id, refresh=refresh, emit=_printer())
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc)
        table = kernel.load(artifact.id)
        if sync:
            try:
                kernel.run("notebook.sync_study",
                           {"study": study_id, "surface": artifact.id},
                           emit=_printer())
            except Exception as exc:                              # noqa: BLE001
                console.print(f"[yellow]vault not updated:[/] {exc}")

        def render():
            console.print(f"\n[bold]{artifact.label}[/]  [cyan]{artifact.id}[/]")
            # A wide grid squeezes every column to nothing in a terminal; show
            # the varied parameters and the headline metrics, and point at the
            # artifact for the rest.
            columns = [c for c in table.colnames if c not in ("artifact",)][:9]
            rich = RichTable(*columns)
            for index in range(min(len(table), 30)):
                rich.add_row(*[f"{table[c][index]:.5g}"
                               if isinstance(table[c][index], (float, np.floating))
                               else str(table[c][index])[:24] for c in columns])
            console.print(rich)
            hidden = len(table.colnames) - len(columns) - 1
            if hidden > 0:
                console.print(f"[dim]{hidden} more column(s) — "
                              f"celestrium --json study run … for all of them[/]")
        _emit({"artifact": artifact.to_dict(),
               "rows": [dict(zip(table.colnames, row)) for row in table]}, render)

    @study_app.command("rank")
    def study_rank_cmd():
        """Rank studies by leverage per row."""
        kernel = _kernel()
        artifact = kernel.run("study.rank", {}, refresh=True)
        table = kernel.load(artifact.id)

        def render():
            rich = RichTable(*table.colnames, title="leverage per row")
            for index in range(len(table)):
                rich.add_row(*[f"{table[c][index]:.3f}"
                               if isinstance(table[c][index], float)
                               else str(table[c][index]) for c in table.colnames])
            console.print(rich)
        _emit([dict(zip(table.colnames, row)) for row in table], render)

    # ----- vault ----------------------------------------------------------- #
    vault_app = typer.Typer(no_args_is_help=True, help="The Obsidian notebook bridge.")
    app.add_typer(vault_app, name="vault", rich_help_panel=STUDY_PANEL)

    @vault_app.command("status")
    def vault_status():
        """Is a vault reachable, and what claims does it hold?"""
        from .vault import sync as vault_sync
        payload = {"available": vault_sync.available(),
                   "vault": str(vault_sync.vault_root() or ""),
                   "observables": sorted(vault_sync.read_studies_safe())}

        def render():
            if not payload["available"]:
                console.print("[yellow]no vault Astronomy folder found[/]")
                console.print("[dim]set one with: celestrium vault path <dir>[/]")
                return
            console.print(f"[green]vault[/] {payload['vault']}")
            for name in payload["observables"]:
                console.print(f"  · {name}")
        _emit(payload, render)

    @vault_app.command("path")
    def vault_path(directory: str = typer.Argument(..., help="path to the vault root")):
        """Point Celestrium at a vault."""
        from . import config
        config.set("vault_path", str(directory))
        _emit({"vault_path": directory},
              lambda: console.print(f"[green]vault path set[/] {directory}"))

    @vault_app.command("init")
    def vault_init(dry_run: bool = typer.Option(False, "--dry-run")):
        """Add Celestrium's generated folders to the vault's .gitignore."""
        from .vault import sync as vault_sync
        try:
            path, missing = vault_sync.ensure_gitignore(dry_run=dry_run)
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc)
        _emit({"gitignore": str(path), "added": missing, "dry_run": dry_run},
              lambda: console.print(
                  f"[green]{'would add' if dry_run else 'added'}[/] "
                  f"{len(missing)} ignore rule(s) → {path}"))

    @vault_app.command("publish")
    def vault_publish(artifact_id: str = typer.Argument(...),
                      kind: str = typer.Option("auto", help="auto|object|field|run"),
                      dry_run: bool = typer.Option(False, "--dry-run")):
        """Write an artifact into the notebook with its provenance."""
        try:
            result = _kernel().run("notebook.publish",
                                   {"artifact": artifact_id, "kind": kind,
                                    "dry_run": dry_run}, emit=_printer())
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc)
        payload = _kernel().load(result.id)
        _emit(payload, lambda: console.print(f"[green]→[/] {payload['path']}"))

    @vault_app.command("sync")
    def vault_sync_cmd(study_id: str = typer.Argument(...),
                       verdict: str = typer.Option("", help="write a verdict"),
                       dry_run: bool = typer.Option(False, "--dry-run")):
        """Write a study's latest results back to its Observables note."""
        try:
            result = _kernel().run("notebook.sync_study",
                                   {"study": study_id, "verdict": verdict,
                                    "dry_run": dry_run}, emit=_printer())
        except Exception as exc:                                  # noqa: BLE001
            raise _fail(exc)
        payload = _kernel().load(result.id)
        _emit(payload, lambda: console.print(f"[green]synced[/] → {payload['note']}"))
