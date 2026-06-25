#!/usr/bin/env python3
"""
C' field-cartography: live arXiv-API tally of topic effort + trajectory.

Counts papers per topic phrase per year (2022-2025) across astro-ph + gr-qc,
using the arXiv API's opensearch:totalResults (max_results=0, no payload).
Output: a tidy CSV (data/topic_trajectories.csv) = the 'trajectory' axis of the
effort map. Pair with the Astrophysics-Wrapped 2025 tables (effort levels).

Run: python scripts/arxiv_tally.py
"""
import time
import re
import subprocess
import urllib.parse
import csv
import os

# HTTPS + curl: Python urllib can't reach export.arxiv.org from this sandbox,
# but curl is proxied fine. max_results=1 (NOT 0 — 0 yields a synthetic
# totalResults=1 error feed). We still only read the count via regex, so no
# XML parser is involved (no XXE/billion-laughs surface).
API = "https://export.arxiv.org/api/query"
TOTAL_RE = re.compile(r"<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>")

# topic -> arXiv search phrase. Tied to our projects A-F where possible.
TOPICS = {
    # --- our project micro-topics (where we'd actually fish) ---
    "wide binary gravity":      'all:"wide binary" AND (all:MOND OR all:gravity)',
    "MOND / Milgromian":        'all:MOND OR all:Milgromian',
    "Dyson sphere":             'all:"Dyson sphere"',
    "technosignature":          'all:technosignature OR all:technosignatures',
    "Hubble tension":           'all:"Hubble tension"',
    "S8 / sigma8 tension":      'all:"S8 tension" OR all:"sigma8 tension"',
    "primordial black hole":    'all:"primordial black hole"',
    "reheating (inflation)":    'all:reheating AND all:inflation',
    "preheating":               'all:preheating',
    "oscillon":                 'all:oscillon OR all:oscillons',
    "inflationary grav. waves": 'all:"inflationary gravitational waves"',
    # --- crowded frontiers (calibration / 'avoid' markers) ---
    "JWST high-z galaxies":     'all:JWST AND all:"high redshift" AND all:galaxies',
    "early dark energy":        'all:"early dark energy"',
    "21 cm cosmology":          'all:"21 cm" AND all:cosmology',
    "fast radio burst":         'all:"fast radio burst"',
}
YEARS = [2022, 2023, 2024, 2025]
CAT = "(cat:astro-ph.CO OR cat:astro-ph.GA OR cat:astro-ph.HE OR cat:gr-qc)"


def total_results(query: str, retries: int = 4) -> int:
    url = f"{API}?{urllib.parse.urlencode({'search_query': query, 'max_results': 1})}"
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sL", "--max-time", "30", url],
                capture_output=True, text=True, timeout=40,
            )
            m = TOTAL_RE.search(out.stdout)
            if m:
                return int(m.group(1))
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! failed: {e}")
                return -1
        time.sleep(5)
    return -1


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "data", "topic_trajectories.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for topic, phrase in TOPICS.items():
        counts = {}
        for yr in YEARS:
            q = (f"({phrase}) AND {CAT} AND "
                 f"submittedDate:[{yr}01010000 TO {yr}12312359]")
            n = total_results(q)
            counts[yr] = n
            print(f"{topic:28s} {yr}: {n}")
            time.sleep(3.2)  # arXiv asks ~3s between calls
        # crude trajectory: 2025 vs 2022-2024 mean
        hist = [counts[y] for y in YEARS[:-1] if counts[y] >= 0]
        base = (sum(hist) / len(hist)) if hist else 0
        trend = (counts[2025] / base) if base else float("nan")
        rows.append({"topic": topic, **{str(y): counts[y] for y in YEARS},
                     "2022-24_mean": round(base, 1), "2025/mean": round(trend, 2)})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
