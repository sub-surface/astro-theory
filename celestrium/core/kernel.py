"""The kernel — dedupe, run, persist, record, emit.

Everything the instrument does goes through `Kernel.run`. That single funnel is
what makes provenance automatic rather than remembered: there is no path to
producing an artifact that skips the ledger.

Three deliberate design calls:

* **Synchronous core, thread pool around it.** astroquery is sync and
  occasionally thread-hostile; an async rewrite would buy little and cost a
  dependency. `run()` blocks, `submit()` backgrounds it, and Textual workers
  call `run()` directly.
* **Cancellation discards the *view*, not the work.** You cannot kill a thread
  blocked in a socket read. So a cancelled run still writes its artifact —
  content-addressed, so it is never waste — and simply stops being rendered.
  Honest, and the next identical request is instant.
* **Declared inputs are hashed; discovered children are not.** A capability
  that calls `ctx.child()` records a lineage edge, but the child cannot enter
  the parent's id (it isn't known until the body runs). `meta["_inputs"]`
  keeps the exact hashed tuple so `repro` stays exact.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from .. import paths
from . import events as ev
from .artifact import (Artifact, SUFFIX, artifact_id, jsonable, now_utc,
                       relative)
from .capability import Capability, ParamError, get as get_cap
from .ledger import Ledger, default as default_ledger

RETRY_BACKOFFS = (2.0, 5.0)
_sleep = time.sleep          # module attr so tests stub the waiting


class CapabilityError(RuntimeError):
    """A capability body failed. Carries the originating exception."""

    def __init__(self, cap: str, cause: BaseException):
        super().__init__(f"{cap}: {type(cause).__name__}: {cause}")
        self.cap = cap
        self.cause = cause


@dataclass(frozen=True)
class Payload:
    """What a capability body returns — the kernel persists it."""
    kind: str
    value: Any
    meta: dict = field(default_factory=dict)
    label: str = ""


class Context:
    """What a capability body is handed. Everything it needs, nothing global."""

    def __init__(self, kernel: "Kernel", cap: Capability, params: dict,
                 artifact_id_: str, run_id: str, study: Optional[str],
                 emit: Callable, cancel: Optional[threading.Event],
                 listener: Optional[Callable] = None):
        self.kernel = kernel
        self.ledger = kernel.ledger
        self.cap = cap
        self.params = params
        self.artifact_id = artifact_id_
        self.run_id = run_id
        self.study = study
        self._emit = emit
        # The caller's own listener, *without* kernel.on_event folded in —
        # child runs fan on_event back in themselves, so handing them `emit`
        # would deliver every child event to on_event twice.
        self._listener = listener
        self._cancel = cancel
        self.children: list = []
        self.notes: list = []

    # ----- talking back ----------------------------------------------------- #
    def progress(self, message: str) -> None:
        self._emit(ev.Event(ev.PROGRESS, self.run_id, self.cap.name, str(message),
                            study=self.study))

    def note(self, message: str) -> None:
        """A durable remark — lands in the artifact's meta, not just the log."""
        self.notes.append(str(message))
        self.progress(message)

    @property
    def cancelled(self) -> bool:
        return bool(self._cancel and self._cancel.is_set())

    def check_cancel(self) -> None:
        if self.cancelled:
            raise KeyboardInterrupt(f"{self.cap.name} cancelled")

    # ----- composing -------------------------------------------------------- #
    def child(self, cap_name: str, /, **params) -> Artifact:
        """Run another capability as part of this one; records a lineage edge."""
        art = self.kernel.run(cap_name, params, study=self.study,
                              emit=self._listener, cancel=self._cancel)
        self.children.append(art.id)
        return art

    def load(self, ref: str) -> Any:
        """Read an input artifact's payload back (Table, dict, text or Path)."""
        return self.kernel.load(ref)

    def artifact(self, ref: str) -> Optional[Artifact]:
        return self.ledger.find(ref)

    # ----- output paths ----------------------------------------------------- #
    def out_path(self, suffix: str = "") -> Path:
        """The deterministic destination for this artifact's payload. Capabilities
        that render files (cutouts, posters, plots) write straight here."""
        suffix = suffix or SUFFIX.get(self.cap.kind, ".bin")
        directory = paths.ARTIFACTS / self.cap.kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self.artifact_id}{suffix}"

    # ----- payload builders ------------------------------------------------- #
    def table(self, tab, *, label: str = "", **meta) -> Payload:
        return Payload("table", tab, meta, label)

    def surface(self, tab, *, label: str = "", **meta) -> Payload:
        return Payload("surface", tab, meta, label)

    def image(self, path, *, label: str = "", **meta) -> Payload:
        return Payload("image", path, meta, label)

    def figure(self, path, *, label: str = "", **meta) -> Payload:
        return Payload("figure", path, meta, label)

    def spectrum(self, path, *, label: str = "", **meta) -> Payload:
        return Payload("spectrum", path, meta, label)

    def document(self, markdown: str, *, label: str = "", **meta) -> Payload:
        return Payload("document", markdown, meta, label)

    def data(self, obj, *, label: str = "", **meta) -> Payload:
        return Payload("data", obj, meta, label)


class Kernel:
    def __init__(self, ledger: Optional[Ledger] = None,
                 on_event: Optional[Callable] = None, workers: int = 4):
        self.ledger = ledger if ledger is not None else default_ledger()
        self.on_event = on_event
        self._pool: Optional[ThreadPoolExecutor] = None
        self._workers = workers
        self._lock = threading.Lock()

    # ----- reading ---------------------------------------------------------- #
    def load(self, ref: str) -> Any:
        art = self.ledger.find(ref) if isinstance(ref, str) else None
        if art is None:
            raise KeyError(f"no artifact {ref!r} in the ledger")
        target = art.full_path
        if target is None or not target.exists():
            raise FileNotFoundError(f"artifact {art.id} payload missing: {art.path}")
        if art.kind in ("table", "surface"):
            from astropy.table import Table
            return Table.read(target)
        if art.kind == "data":
            return json.loads(target.read_text(encoding="utf-8"))
        if art.kind == "document":
            return target.read_text(encoding="utf-8")
        return target

    # ----- planning --------------------------------------------------------- #
    def resolve_id(self, cap_name: str, params: Optional[dict] = None,
                   inputs: tuple = ()) -> tuple:
        """(artifact_id, bound_params, resolved_inputs) without running anything."""
        cap = get_cap(cap_name)
        bound = cap.bind(params)
        resolved = list(inputs)
        for key in cap.artifact_params:
            ref = bound.get(key)
            if not ref:
                continue
            art = self.ledger.find(str(ref))
            if art is None:
                raise ParamError(f"{cap_name}.{key}: no artifact {ref!r} in the ledger")
            bound[key] = art.id            # canonicalise prefix → full id
            # `repro` hands back meta["_inputs"], which already holds these
            # ids — appending again would change the hash.
            if art.id not in resolved:
                resolved.append(art.id)
        return artifact_id(cap_name, bound, tuple(resolved)), bound, tuple(resolved)

    def is_cached(self, cap_name: str, params: Optional[dict] = None,
                  inputs: tuple = ()) -> bool:
        try:
            aid, _, _ = self.resolve_id(cap_name, params, inputs)
        except (KeyError, ParamError):
            return False
        return self.ledger.get(aid) is not None

    def estimate(self, calls: list) -> dict:
        """Cost preview for a batch: how many runs are already in the ledger.

        `calls` is a list of (cap_name, params) — study grids print this before
        committing, which is the machine version of the vault's `rows:` honesty."""
        cached = to_run = 0
        cost_mix: dict = {}
        for cap_name, params in calls:
            cap = get_cap(cap_name)
            cost_mix[cap.cost] = cost_mix.get(cap.cost, 0) + 1
            if self.is_cached(cap_name, params):
                cached += 1
            else:
                to_run += 1
        return {"total": len(calls), "cached": cached, "to_run": to_run,
                "cost_mix": cost_mix}

    # ----- running ---------------------------------------------------------- #
    def run(self, cap_name: str, params: Optional[dict] = None, *,
            inputs: tuple = (), refresh: bool = False,
            study: Optional[str] = None, label: str = "",
            emit: Optional[Callable] = None,
            cancel: Optional[threading.Event] = None) -> Artifact:
        cap = get_cap(cap_name)
        listener = emit
        emit = ev.fan_out(emit, self.on_event)
        run_id = uuid.uuid4().hex[:12]

        aid, bound, resolved_inputs = self.resolve_id(cap_name, params, inputs)

        if not refresh:
            existing = self.ledger.get(aid)
            if existing is not None and (existing.path is None or existing.exists):
                emit(ev.Event(ev.DONE, run_id, cap_name, "cached",
                              artifact=existing, cached=True, study=study))
                self.ledger.record_run(run_id=run_id, cap=cap_name, params=bound,
                                       status="cached", artifact=aid,
                                       started=now_utc(), finished=now_utc(),
                                       study=study, cached=True)
                return existing

        started = now_utc()
        clock = time.time()
        emit(ev.Event(ev.STARTED, run_id, cap_name, cap.summary, study=study))

        attempts = len(RETRY_BACKOFFS) + 1 if cap.cost == "network" else 1
        ctx = last_error = None
        error_trace = ""
        payload = None
        for attempt in range(1, attempts + 1):
            ctx = Context(self, cap, bound, aid, run_id, study, emit, cancel,
                          listener=listener)
            try:
                payload = cap.fn(ctx, **bound)
                last_error = None
                break
            except KeyboardInterrupt as exc:                 # cancellation
                elapsed = int((time.time() - clock) * 1000)
                emit(ev.Event(ev.CANCELLED, run_id, cap_name, str(exc),
                              ms=elapsed, study=study))
                self.ledger.record_run(run_id=run_id, cap=cap_name, params=bound,
                                       status="cancelled", started=started,
                                       finished=now_utc(), ms=elapsed, study=study)
                raise
            except Exception as exc:                          # noqa: BLE001
                last_error = exc
                error_trace = traceback.format_exc(limit=4)[-2000:]
                if attempt < attempts:
                    emit(ev.Event(ev.PROGRESS, run_id, cap_name,
                                  f"retry {attempt}/{attempts - 1}: {exc}",
                                  attempt=attempt, study=study))
                    _sleep(RETRY_BACKOFFS[attempt - 1])

        elapsed = int((time.time() - clock) * 1000)
        if last_error is not None:
            detail = f"{type(last_error).__name__}: {last_error}"
            emit(ev.Event(ev.FAILED, run_id, cap_name, detail, error=detail,
                          ms=elapsed, study=study))
            self.ledger.record_run(run_id=run_id, cap=cap_name, params=bound,
                                   status="failed", started=started,
                                   finished=now_utc(), ms=elapsed,
                                   error=error_trace,
                                   study=study)
            raise CapabilityError(cap_name, last_error) from last_error

        artifact = self._persist(cap, aid, bound, resolved_inputs, payload, ctx,
                                 study=study, label=label, ms=elapsed)
        emit(ev.Event(ev.ARTIFACT, run_id, cap_name, artifact.one_line(),
                      artifact=artifact, ms=elapsed, study=study))
        emit(ev.Event(ev.DONE, run_id, cap_name, "", artifact=artifact,
                      ms=elapsed, study=study))
        self.ledger.record_run(run_id=run_id, cap=cap_name, params=bound,
                               status="ok", artifact=artifact.id, started=started,
                               finished=now_utc(), ms=elapsed, study=study)
        return artifact

    def submit(self, cap_name: str, params: Optional[dict] = None, **kwargs) -> Future:
        """Background a run (the TUI's fire-and-watch path)."""
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self._workers,
                                                thread_name_prefix="celestrium")
        return self._pool.submit(self.run, cap_name, params, **kwargs)

    def shutdown(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None

    # ----- persistence ------------------------------------------------------ #
    def _persist(self, cap: Capability, aid: str, bound: dict, inputs: tuple,
                 payload: Any, ctx: Context, *, study: Optional[str],
                 label: str, ms: int) -> Artifact:
        if payload is None:
            payload = Payload(cap.kind, None, {}, "")
        if not isinstance(payload, Payload):
            payload = self._wrap_bare(cap, payload)

        meta = dict(payload.meta)
        meta["_inputs"] = list(inputs)          # the exact hashed tuple, for repro
        meta["exec_ms"] = ms
        if ctx.notes:
            meta["notes"] = list(ctx.notes)

        path = None
        value = payload.value
        if value is not None:
            path, extra = self._write_payload(payload.kind, value, aid, ctx)
            meta.update(extra)

        artifact = Artifact(
            id=aid, kind=payload.kind or cap.kind, cap=cap.name,
            params=jsonable(bound), inputs=tuple(inputs), path=path, meta=meta,
            label=label or payload.label, study=study, created=now_utc(),
        )
        self.ledger.put(artifact)
        # Children are lineage but not identity: edge only (see module docstring).
        if ctx.children:
            self.ledger.add_edges(aid, ctx.children)
            return replace(artifact,
                           inputs=tuple(inputs) + tuple(c for c in ctx.children
                                                        if c not in inputs))
        return artifact

    @staticmethod
    def _wrap_bare(cap: Capability, value: Any) -> Payload:
        """Capabilities may return a bare Table/Path/str/dict; infer the kind."""
        from astropy.table import Table
        if isinstance(value, Table):
            return Payload("surface" if cap.kind == "surface" else "table", value)
        if isinstance(value, Path):
            return Payload(cap.kind if cap.kind in
                           ("image", "figure", "spectrum") else "image", value)
        if isinstance(value, str):
            return Payload("document", value)
        return Payload("data", value)

    def _write_payload(self, kind: str, value: Any, aid: str,
                       ctx: Context) -> tuple:
        directory = paths.ARTIFACTS / kind
        directory.mkdir(parents=True, exist_ok=True)
        extra: dict = {}

        if kind in ("table", "surface"):
            target = directory / f"{aid}.ecsv"
            value.write(target, format="ascii.ecsv", overwrite=True)
            extra = {"nrows": len(value), "ncols": len(value.colnames),
                     "columns": list(value.colnames)[:40]}
        elif kind == "document":
            target = directory / f"{aid}.md"
            target.write_text(str(value), encoding="utf-8")
            extra = {"chars": len(str(value))}
        elif kind == "data":
            target = directory / f"{aid}.json"
            target.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True),
                              encoding="utf-8")
        else:                                   # image / figure / spectrum
            target = Path(value)
            if not target.exists():
                raise FileNotFoundError(f"{ctx.cap.name} reported a missing file: {target}")
        try:
            extra["bytes"] = target.stat().st_size
        except OSError:
            pass
        return relative(target), extra

    # ----- reproduction ----------------------------------------------------- #
    def repro(self, ref: str, *, refresh: bool = True,
              emit: Optional[Callable] = None) -> Artifact:
        """Rebuild an artifact's whole chain from its id — oldest input first."""
        art = self.ledger.find(ref)
        if art is None:
            raise KeyError(f"no artifact {ref!r}")
        for ancestor in self.ledger.chain(art.id)[:-1]:
            self.run(ancestor.cap, ancestor.params,
                     inputs=tuple(ancestor.meta.get("_inputs", ())),
                     refresh=refresh, emit=emit, study=ancestor.study)
        return self.run(art.cap, art.params,
                        inputs=tuple(art.meta.get("_inputs", ())),
                        refresh=refresh, emit=emit, study=art.study,
                        label=art.label)

    # ----- narrative -------------------------------------------------------- #
    def methods(self, ref: str) -> str:
        """A methods paragraph generated from lineage — the caption a figure
        should carry when it leaves the instrument. Free, given the DAG."""
        chain = self.ledger.chain(ref)
        if not chain:
            return ""
        lines = []
        for art in chain:
            params = ", ".join(f"{k}={v!r}" for k, v in sorted(art.params.items())
                               if v is not None and v != "")
            rows = art.meta.get("nrows")
            tail = f" → {rows} rows" if rows is not None else ""
            lines.append(f"- **{art.cap}**({params}){tail}  `{art.id}`")
        final = chain[-1]
        head = (f"Artifact `{final.id}` ({final.kind}, {final.cap}), "
                f"produced {final.created}.")
        return head + "\n\n" + "\n".join(lines)


_DEFAULT: Optional[Kernel] = None


def default() -> Kernel:
    global _DEFAULT
    if _DEFAULT is None:
        from .capability import load_all
        load_all()
        _DEFAULT = Kernel()
    return _DEFAULT


def use(kernel: Optional[Kernel]) -> None:
    global _DEFAULT
    _DEFAULT = kernel
