"""Field-effort census — live arXiv tallies of where the field spends effort.

The theory wing's "living census" (CLAUDE.md), promoted from a standalone
script to instrument machinery: counts papers per topic phrase per year across
astro-ph + gr-qc via the arXiv API's opensearch:totalResults (max_results=1,
no payload). Topics live in config.CENSUS_TOPICS; the fetcher is injectable
so tests stay hermetic. Output: astropy Table + optional CSV under data/.
"""
from __future__ import annotations

import csv
import re
import time
import urllib.parse
from pathlib import Path
from typing import Callable, Iterable, Optional

from astropy.table import Table

from . import paths, config
from .net import DEFAULT_TIMEOUT

API = "https://export.arxiv.org/api/query"
CATEGORIES = "(cat:astro-ph.CO OR cat:astro-ph.GA OR cat:astro-ph.HE OR cat:gr-qc)"
TOTAL_RE = re.compile(r"<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>")
YEARS = (2022, 2023, 2024, 2025)
_sleep = time.sleep   # module attr so tests can stub the API pacing


def _default_fetcher(url: str) -> str:
    import requests
    r = requests.get(url, timeout=DEFAULT_TIMEOUT,
                     headers={"User-Agent": "Celestrium census"})
    r.raise_for_status()
    return r.text


def total_results(query: str, fetcher: Optional[Callable[[str], str]] = None,
                  retries: int = 3) -> int:
    """One count from the arXiv API; -1 on persistent failure (not a raise —
    a census with one dead cell is still a census)."""
    fetcher = fetcher or _default_fetcher
    # max_results=1, NOT 0 — 0 yields a synthetic totalResults=1 error feed.
    url = f"{API}?{urllib.parse.urlencode({'search_query': query, 'max_results': 1})}"
    for attempt in range(retries):
        try:
            m = TOTAL_RE.search(fetcher(url))
            if m:
                return int(m.group(1))
        except Exception:
            pass
        if attempt < retries - 1:
            _sleep(3)
    return -1


def tally(topics: Optional[dict] = None, years: Iterable[int] = YEARS,
          fetcher: Optional[Callable[[str], str]] = None,
          on_note: Optional[Callable[[str], None]] = None) -> Table:
    """Count papers per topic per year -> Table(topic, <year>..., trend).

    trend = percent change of the last year vs the first (rounded), or ''
    when either endpoint is missing.
    """
    topics = topics if topics is not None else config.CENSUS_TOPICS
    years = list(years)
    rows = []
    for topic, phrase in topics.items():
        counts = []
        for yr in years:
            q = (f"({phrase}) AND {CATEGORIES} AND "
                 f"submittedDate:[{yr}01010000 TO {yr}12312359]")
            counts.append(total_results(q, fetcher))
            _sleep(1)   # arXiv API etiquette between calls
        trend = ""
        if counts and counts[0] > 0 and counts[-1] >= 0:
            trend = f"{(counts[-1] - counts[0]) / counts[0] * 100:+.0f}%"
        if on_note:
            on_note(f"{topic}: {counts}")
        rows.append([topic, *counts, trend])
    tab = Table(rows=rows, names=["topic", *[str(y) for y in years], "trend"])
    return tab


def write_csv(tab: Table, out: Optional[Path] = None) -> Path:
    out = Path(out) if out else paths.DATA / "topic_trajectories.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(tab.colnames)
        for row in tab:
            writer.writerow([row[c] for c in tab.colnames])
    return out


def main() -> None:
    """Entry point kept for the legacy scripts/arxiv_tally.py shim."""
    tab = tally(on_note=print)
    out = write_csv(tab)
    print(f"census -> {out}")
