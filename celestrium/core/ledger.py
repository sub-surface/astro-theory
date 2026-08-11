"""The ledger — one SQLite file that is cache, manifest and lineage graph.

Replaces `cache.py`'s keyspace, `data/manifest.jsonl`, and
`data/candidates/index.jsonl`. Payloads stay as files on disk; this stores
metadata and the edges between them, because the edges are the part that was
missing: the old manifest could say *what* was pulled but never *what was
derived from it*.

A connection per operation (WAL mode) — SQLite handles that fine and it keeps
the store trivially safe to touch from TUI worker threads.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from .. import paths
from .artifact import Artifact, jsonable, now_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifact (
    id       TEXT PRIMARY KEY,
    kind     TEXT NOT NULL,
    cap      TEXT NOT NULL,
    params   TEXT NOT NULL,
    path     TEXT,
    meta     TEXT NOT NULL,
    label    TEXT DEFAULT '',
    study    TEXT,
    created  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edge (
    child  TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (child, parent)
);
CREATE TABLE IF NOT EXISTS run (
    run_id   TEXT PRIMARY KEY,
    cap      TEXT NOT NULL,
    params   TEXT NOT NULL,
    artifact TEXT,
    status   TEXT NOT NULL,
    started  TEXT,
    finished TEXT,
    ms       INTEGER DEFAULT 0,
    error    TEXT DEFAULT '',
    study    TEXT,
    cached   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS study (
    id      TEXT PRIMARY KEY,
    spec    TEXT NOT NULL,
    prereg  TEXT,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tag (
    artifact TEXT NOT NULL,
    tag      TEXT NOT NULL,
    PRIMARY KEY (artifact, tag)
);
CREATE INDEX IF NOT EXISTS ix_artifact_cap     ON artifact(cap);
CREATE INDEX IF NOT EXISTS ix_artifact_study   ON artifact(study);
CREATE INDEX IF NOT EXISTS ix_artifact_created ON artifact(created);
CREATE INDEX IF NOT EXISTS ix_edge_parent      ON edge(parent);
CREATE INDEX IF NOT EXISTS ix_run_study        ON run(study);
"""


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else paths.LEDGER_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- artifacts -------------------------------------------------------- #
    def put(self, artifact: Artifact) -> Artifact:
        if not artifact.created:
            artifact = replace(artifact, created=now_utc())
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifact"
                " (id, kind, cap, params, path, meta, label, study, created)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (artifact.id, artifact.kind, artifact.cap,
                 json.dumps(jsonable(artifact.params)), artifact.path,
                 json.dumps(jsonable(artifact.meta)), artifact.label,
                 artifact.study, artifact.created),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO edge (child, parent) VALUES (?,?)",
                [(artifact.id, parent) for parent in artifact.inputs],
            )
        return artifact

    def add_edges(self, child: str, parents: Iterable[str]) -> None:
        """Record lineage that isn't identity — e.g. capabilities a body called
        via `ctx.child()`, which are discovered mid-run and so cannot be part of
        the parent's content hash."""
        pairs = [(child, p) for p in parents if p and p != child]
        if not pairs:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO edge (child, parent) VALUES (?,?)", pairs)

    def _row_to_artifact(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Artifact:
        parents = [r["parent"] for r in
                   conn.execute("SELECT parent FROM edge WHERE child=?", (row["id"],))]
        return Artifact(
            id=row["id"], kind=row["kind"], cap=row["cap"],
            params=json.loads(row["params"]), inputs=tuple(parents),
            path=row["path"], meta=json.loads(row["meta"]),
            label=row["label"] or "", study=row["study"], created=row["created"],
        )

    def get(self, artifact_id: str) -> Optional[Artifact]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM artifact WHERE id=?",
                               (artifact_id,)).fetchone()
            return self._row_to_artifact(row, conn) if row else None

    def find(self, ref: str) -> Optional[Artifact]:
        """Resolve an id or unambiguous id prefix (the CLI/TUI-friendly form)."""
        if not ref:
            return None
        exact = self.get(ref)
        if exact is not None:
            return exact
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM artifact WHERE id LIKE ? ORDER BY created DESC LIMIT 2",
                (f"{ref}%",)).fetchall()
            if not rows:
                return None
            return self._row_to_artifact(rows[0], conn)

    def search(self, *, cap: Optional[str] = None, kind: Optional[str] = None,
               study: Optional[str] = None, label: Optional[str] = None,
               limit: int = 50) -> list:
        clauses, args = [], []
        if cap:
            clauses.append("cap LIKE ?"), args.append(f"{cap}%")
        if kind:
            clauses.append("kind=?"), args.append(kind)
        if study:
            clauses.append("study=?"), args.append(study)
        if label:
            clauses.append("label LIKE ?"), args.append(f"%{label}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM artifact {where} ORDER BY created DESC, rowid DESC"
                " LIMIT ?", (*args, limit)).fetchall()
            return [self._row_to_artifact(r, conn) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) c FROM artifact").fetchone()["c"]

    # ----- lineage ---------------------------------------------------------- #
    def ancestors(self, artifact_id: str, depth: int = 12) -> list:
        """Every artifact this one was derived from, nearest first."""
        seen, out, frontier = {artifact_id}, [], [artifact_id]
        for _ in range(depth):
            if not frontier:
                break
            nxt = []
            for node in frontier:
                art = self.get(node)
                if art is None:
                    continue
                for parent in art.inputs:
                    if parent in seen:
                        continue
                    seen.add(parent)
                    parent_art = self.get(parent)
                    if parent_art is not None:
                        out.append(parent_art)
                        nxt.append(parent)
            frontier = nxt
        return out

    def descendants(self, artifact_id: str, depth: int = 12) -> list:
        seen, out, frontier = {artifact_id}, [], [artifact_id]
        for _ in range(depth):
            if not frontier:
                break
            nxt = []
            with self._conn() as conn:
                for node in frontier:
                    for row in conn.execute("SELECT child FROM edge WHERE parent=?",
                                            (node,)):
                        child = row["child"]
                        if child in seen:
                            continue
                        seen.add(child)
                        nxt.append(child)
            out.extend(a for a in (self.get(c) for c in nxt) if a is not None)
            frontier = nxt
        return out

    def chain(self, artifact_id: str) -> list:
        """Full provenance chain in dependency order — every artifact appears
        after everything it was derived from. This is what makes the generated
        methods paragraph read as a procedure rather than a bag of ids."""
        art = self.find(artifact_id)
        if art is None:
            return []
        order, done, stack = [], set(), [(art.id, False)]
        while stack:
            node_id, expanded = stack.pop()
            if node_id in done:
                continue
            node = self.get(node_id)
            if node is None:
                done.add(node_id)
                continue
            if expanded:
                done.add(node_id)
                order.append(node)
                continue
            stack.append((node_id, True))
            for parent in node.inputs:
                if parent not in done:
                    stack.append((parent, False))
        return order

    # ----- runs ------------------------------------------------------------- #
    def record_run(self, *, run_id: str, cap: str, params: dict, status: str,
                   artifact: Optional[str] = None, started: str = "",
                   finished: str = "", ms: int = 0, error: str = "",
                   study: Optional[str] = None, cached: bool = False) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run"
                " (run_id, cap, params, artifact, status, started, finished, ms,"
                "  error, study, cached) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, cap, json.dumps(jsonable(params)), artifact, status,
                 started, finished, int(ms), error, study, int(bool(cached))),
            )

    def runs(self, limit: int = 40, study: Optional[str] = None) -> list:
        clause, args = ("WHERE study=?", [study]) if study else ("", [])
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                f"SELECT * FROM run {clause} ORDER BY started DESC, rowid DESC LIMIT ?",
                (*args, limit))]

    # ----- studies ---------------------------------------------------------- #
    def put_study(self, study_id: str, spec: dict,
                  prereg: Optional[str] = None) -> None:
        with self._conn() as conn:
            existing = conn.execute("SELECT prereg FROM study WHERE id=?",
                                    (study_id,)).fetchone()
            # A preregistration is written once and never silently replaced —
            # that is the entire point of it (see study/model.prereg_hash).
            keep = existing["prereg"] if existing and existing["prereg"] else prereg
            conn.execute(
                "INSERT OR REPLACE INTO study (id, spec, prereg, updated)"
                " VALUES (?,?,?,?)",
                (study_id, json.dumps(jsonable(spec)), keep, now_utc()))

    def get_study(self, study_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM study WHERE id=?",
                               (study_id,)).fetchone()
            if not row:
                return None
            return {"id": row["id"], "spec": json.loads(row["spec"]),
                    "prereg": row["prereg"], "updated": row["updated"]}

    def studies(self) -> list:
        with self._conn() as conn:
            return [{"id": r["id"], "spec": json.loads(r["spec"]),
                     "prereg": r["prereg"], "updated": r["updated"]}
                    for r in conn.execute("SELECT * FROM study ORDER BY updated DESC")]

    # ----- tags + housekeeping ---------------------------------------------- #
    def tag(self, artifact_id: str, *tags: str) -> None:
        with self._conn() as conn:
            conn.executemany("INSERT OR IGNORE INTO tag (artifact, tag) VALUES (?,?)",
                             [(artifact_id, t) for t in tags])

    def tags_of(self, artifact_id: str) -> list:
        with self._conn() as conn:
            return [r["tag"] for r in conn.execute(
                "SELECT tag FROM tag WHERE artifact=? ORDER BY tag", (artifact_id,))]

    def stats(self) -> dict:
        with self._conn() as conn:
            by_kind = {r["kind"]: r["n"] for r in conn.execute(
                "SELECT kind, COUNT(*) n FROM artifact GROUP BY kind")}
            by_cap = {r["cap"]: r["n"] for r in conn.execute(
                "SELECT cap, COUNT(*) n FROM artifact GROUP BY cap ORDER BY n DESC")}
            runs = conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(cached),0) c FROM run").fetchone()
        bytes_on_disk = 0
        for art in self.search(limit=100000):
            target = art.full_path
            if target and target.exists():
                try:
                    bytes_on_disk += target.stat().st_size
                except OSError:
                    pass
        return {
            "artifacts": self.count(), "by_kind": by_kind, "by_cap": by_cap,
            "runs": runs["n"], "cache_hits": runs["c"], "bytes": bytes_on_disk,
            "db": self.path.as_posix(),
        }

    def forget(self, artifact_id: str, *, delete_payload: bool = False) -> bool:
        art = self.find(artifact_id)
        if art is None:
            return False
        if delete_payload and art.full_path and art.full_path.exists():
            try:
                art.full_path.unlink()
            except OSError:
                pass
        with self._conn() as conn:
            conn.execute("DELETE FROM artifact WHERE id=?", (art.id,))
            conn.execute("DELETE FROM edge WHERE child=? OR parent=?", (art.id, art.id))
            conn.execute("DELETE FROM tag WHERE artifact=?", (art.id,))
        return True

    def orphans(self) -> list:
        """Artifacts whose payload file has gone missing (safe to forget)."""
        return [a for a in self.search(limit=100000)
                if a.path and not (a.full_path and a.full_path.exists())]


_DEFAULT: Optional[Ledger] = None


def default() -> Ledger:
    """The process-wide ledger (tests pass their own instance instead)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Ledger()
    return _DEFAULT


def use(ledger: Optional[Ledger]) -> None:
    """Point the process-wide ledger somewhere else (tests, alternate data dirs)."""
    global _DEFAULT
    _DEFAULT = ledger
