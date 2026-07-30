#!/usr/bin/env python
"""Query cache + provenance manifest — never pull the same rows twice.

Wrap any archive fetch (gaia.query, irsa.query, ...) so its result is cached
locally and every pull is logged. The cache key is sha1(archive + query), so an
identical query returns instantly from disk; a changed query re-fetches. Every
fetch appends a line to data/manifest.jsonl, making each downstream figure
traceable to the exact query + date that produced it. See ../docs/toolbox.md S6.

Storage: ECSV (pure-astropy, lossless, keeps units/metadata; no pyarrow needed).
Both cache/ and manifest live under data/ (git-ignored) — local, not committed.
"""
import collections
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from astropy.table import Table

_REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = _REPO / "data" / "cache"
MANIFEST = _REPO / "data" / "manifest.jsonl"

# Public archives hiccup (transient 500s, rate limits); a couple of short
# retries removes most "just run it again" friction. The cache seam is the one
# place every fetch already passes through, so the policy lives here.
RETRY_BACKOFFS = (2.0, 5.0)   # seconds between attempts
_sleep = time.sleep           # module attr so tests can stub the waiting

# Process-local (not persisted) hit/miss history for the TUI context bar's
# cache sparkline — "is caching actually working this session", at a glance.
_HIT_HISTORY_LEN = 40
_hit_history: "collections.deque[bool]" = collections.deque(maxlen=_HIT_HISTORY_LEN)
_SPARK_HIT, _SPARK_MISS = "█", "·"


def hit_history() -> list:
    """Recent cache outcomes, oldest first (True=hit, False=miss)."""
    return list(_hit_history)


def hit_sparkline(width: int = 12) -> str:
    """Render the last `width` outcomes as a compact glyph string."""
    recent = list(_hit_history)[-width:]
    return "".join(_SPARK_HIT if h else _SPARK_MISS for h in recent)


def _key(archive: str, query: str) -> str:
    return hashlib.sha1(f"{archive}\n{query}".encode()).hexdigest()[:16]


def _fetch_with_retries(fetch, retries: int):
    for attempt, backoff in enumerate(RETRY_BACKOFFS[:max(0, retries)], start=1):
        try:
            return fetch()
        except Exception:
            _sleep(backoff)
    return fetch()  # final attempt: let the real exception propagate


def cached_query(archive: str, query: str, fetch, refresh: bool = False,
                 retries: int = len(RETRY_BACKOFFS)) -> Table:
    """Return an astropy Table for (archive, query), from cache if present.

    archive  short tag, e.g. 'gaia', 'irsa', 'euclid' (namespaces the cache).
    query    the ADQL/SQL string (also the human-readable provenance record).
    fetch    zero-arg callable that actually runs the query -> astropy Table.
             e.g.  cached_query('gaia', adql, lambda: gaia.query(adql))
    refresh  True to bypass the cache and re-fetch.
    retries  extra fetch attempts (short backoff) before the error propagates.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(archive, query)
    path = CACHE_DIR / f"{archive}_{key}.ecsv"
    if path.exists() and not refresh:
        _hit_history.append(True)
        return Table.read(path)
    tab = _fetch_with_retries(fetch, retries)
    tab.write(path, format="ascii.ecsv", overwrite=True)
    _log(archive, query, key, path, len(tab))
    _hit_history.append(False)
    return tab


def _log(archive, query, key, path, nrows):
    rec = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": archive,
        "hash": key,
        "nrows": int(nrows),
        "cache_file": path.relative_to(_REPO).as_posix(),
        "query": " ".join(query.split()),  # collapse whitespace for one-line log
    }
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def manifest() -> list:
    """Read the provenance log back as a list of records."""
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_record(hash_prefix: str):
    """Return the most-recent manifest record whose hash starts with hash_prefix.

    Lets the hub re-open / re-run a past pull by its short hash (see `hub log`)."""
    hits = [r for r in manifest() if str(r.get("hash", "")).startswith(hash_prefix)]
    return hits[-1] if hits else None


def load_cached(hash_prefix: str) -> Table:
    """Load a previously cached table by (a prefix of) its manifest hash."""
    rec = find_record(hash_prefix)
    if rec is None:
        raise KeyError(f"no cached query with hash {hash_prefix!r}")
    path = _REPO / rec["cache_file"]
    if not path.exists():
        raise FileNotFoundError(f"cache file missing: {rec['cache_file']}")
    return Table.read(path)


if __name__ == "__main__":
    # Self-contained demo (no network): a fake fetch run twice.
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return Table({"id": [1, 2, 3], "flux": [10.0, 20.0, 30.0]})

    q = "SELECT id, flux FROM demo WHERE flux > 5"
    t1 = cached_query("demo", q, fake_fetch)          # miss -> fetch + log
    t2 = cached_query("demo", q, fake_fetch)          # hit  -> no fetch
    print(f"rows={len(t2)}  fetches={calls['n']} (expect 1: 2nd call was cached)")
    print("last manifest record:", manifest()[-1])
