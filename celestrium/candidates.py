#!/usr/bin/env python
"""candidates.py — persist named candidate lists with provenance.

The output side of the desk loop: a cross-match or a row selection produces a
short-list of objects worth chasing. This module saves that astropy Table under a
*name*, logs where it came from, and reads it back — so the interactive crossmatch
(TUI Phase 3) and the `candidates` CLI command both write to one place, exactly as
`cache.py` is the one place for query provenance.

Storage mirrors cache.py: ECSV tables under data/candidates/ (lossless, pure
astropy, git-ignored) plus an index.jsonl provenance log. The cache stores raw
pulls keyed by query hash; this stores *curated* lists keyed by a human name.

    from celestrium import candidates
    candidates.save("euclid-agn", table, origin="xmatch gaia x nvss", note="r<5\"")
    tab = candidates.load("euclid-agn")
    candidates.index()          # provenance records, newest last
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from astropy.table import Table

_REPO = Path(__file__).resolve().parent.parent
CAND_DIR = _REPO / "data" / "candidates"
INDEX = CAND_DIR / "index.jsonl"


def slug(name: str) -> str:
    """Filesystem-safe name for candidate lists."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.strip()).strip("-").lower()
    return s or "candidates"


def _path(name: str) -> Path:
    return CAND_DIR / f"{slug(name)}.ecsv"


def save(name: str, table: Table, *, origin: str = "", note: str = "") -> Path:
    """Write `table` as the candidate list `name`; append a provenance record.

    name    human label; slugged for the filename (an existing list is replaced).
    table   astropy Table of the short-listed rows.
    origin  free-text provenance, e.g. 'xmatch sample:gaia-bright-nearby x nvss'.
    note    optional human note (cut radius, why these rows, ...).
    Returns the saved ECSV path.
    """
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    table.write(path, format="ascii.ecsv", overwrite=True)
    rec = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "slug": slug(name),
        "nrows": int(len(table)),
        "columns": list(table.colnames),
        "origin": origin,
        "note": note,
        "file": path.relative_to(_REPO).as_posix(),
    }
    with open(INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return path


def index() -> list:
    """Read the candidate-list provenance log as records (oldest first)."""
    if not INDEX.exists():
        return []
    with open(INDEX, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest() -> list:
    """One record per list (the most-recent save of each slug), newest first."""
    by_slug = {}
    for rec in index():  # later writes overwrite earlier -> last wins per slug
        by_slug[rec.get("slug")] = rec
    return sorted(by_slug.values(), key=lambda r: r.get("utc", ""), reverse=True)


def find_record(name: str):
    """Most-recent index record for a candidate list (by name or slug), or None."""
    target = slug(name)
    hits = [r for r in index() if r.get("slug") == target]
    return hits[-1] if hits else None


def load(name: str) -> Table:
    """Load a saved candidate list by name (or slug)."""
    path = _path(name)
    if not path.exists():
        raise KeyError(f"no candidate list named {name!r} (looked for {path.name})")
    return Table.read(path)


def drop(name: str) -> bool:
    """Delete a candidate list's table file. Returns True if a file was removed.

    The index.jsonl history is append-only and kept — provenance shouldn't vanish
    just because the working table was discarded (latest() will simply skip it)."""
    path = _path(name)
    if path.exists():
        path.unlink()
        return True
    return False


if __name__ == "__main__":
    # Self-contained demo (no network): save a tiny list and read it back.
    t = Table({"id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [-5.0, 0.0, 5.0]})
    p = save("demo-candidates", t, origin="hand-built demo", note="three points")
    back = load("demo-candidates")
    print(f"saved {len(back)} rows -> {p}")
    print("index record:", index()[-1])
