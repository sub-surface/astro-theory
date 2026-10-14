"""Sampled AAVSO light-curve fetch for T CrB (EXP-2026-Y).

The AAVSO v2 photometry API returns 10 rows/page and asks for <=1 request / 10 s,
so a full pull of T CrB (~10^6 rows) is infeasible. We instead take page 1 (the
10 most recent observations) of each short date window, which yields a uniformly
sampled nightly light curve, and record the per-window `count` (observing effort).

Token: read from $AAVSO_TOKEN or the gitignored .env. Output: data/tcrb/aavso_sampled.jsonl
(append-only, resumable: already-fetched (band, window) keys are skipped).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "tcrb" / "aavso_sampled.jsonl"
API = "https://apps.aavso.org/v2/api/observations/photometry/"
PAUSE = 40.0


def token() -> str:
    tok = os.environ.get("AAVSO_TOKEN")
    if tok:
        return tok
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("AAVSO_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("AAVSO_TOKEN not set")


def windows(start: date, end: date, step_days: int):
    d = start
    while d < end:
        e = min(d + timedelta(days=step_days), end)
        yield d, e
        d = e


# Priority order: the current (2022+) state densely, then the long baseline, then 1936-46.
# Budget: the API throttles at ~100 requests/hour per token (HTTP 429 + Retry-After), so
# one request per 40 s and ~800 requests total (~9 h), most valuable windows first.
PLAN = [
    ("2", date(2022, 1, 1), date(2023, 10, 17), 6),   # V (dense, done first)
    ("all", date(1935, 1, 1), date(1946, 2, 9), None),  # previous cycle: ~500 obs total -> paginate fully
    ("2", date(2026, 8, 1), date(2026, 10, 14), 2),   # V: densely sample the Sep-2026 UV collapse epoch
    ("3", date(2026, 8, 1), date(2026, 10, 14), 3),   # B, same epoch
    ("2", date(2023, 10, 17), date(2026, 10, 14), 10),
    ("3", date(2022, 1, 1), date(2026, 10, 14), 10),  # B
    ("2", date(2005, 1, 1), date(2022, 1, 1), 30),
    ("3", date(2005, 1, 1), date(2022, 1, 1), 45),
    ("7", date(2022, 1, 1), date(2026, 10, 14), 30),  # U
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["band"], r["start"]))
    s = requests.Session()
    s.headers.update({"Authorization": f"Token {token()}", "Accept": "application/json"})
    for band, a, b, step in PLAN:
        if step is None:  # full pagination of one sparse range
            jobs = [(a, b, pg) for pg in range(1, 200)]
        else:
            jobs = [(w0, w1, 1) for w0, w1 in windows(a, b, step)]
        for w0, w1, page in jobs:
            key = (band, w0.isoformat() if page == 1 and step is not None else f"{w0.isoformat()}#p{page}")
            if key in done:
                continue
            params = {"target": "T CrB", "start_date": w0.isoformat(), "end_date": w1.isoformat(), "page": page}
            if band != "all":
                params["band"] = band
            t_req = time.monotonic()
            for attempt in range(6):
                try:
                    r = s.get(API, params=params, timeout=90)
                    if r.status_code == 429:
                        wait = int(r.headers.get("Retry-After", 600)) + 5
                        print("throttled; sleeping", wait, "s", flush=True)
                        time.sleep(wait)
                        continue
                    if r.status_code == 404:
                        d = {"count": 0, "results": [], "next": None}
                        break
                    r.raise_for_status()
                    d = r.json()
                    break
                except Exception as exc:  # network hiccup: back off and retry
                    print("retry", key, exc, flush=True)
                    time.sleep(30)
            else:
                continue
            rec = {"band": band, "start": key[1], "end": w1.isoformat(),
                   "count": d.get("count", 0), "results": d.get("results", [])}
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(band, key[1], d.get("count"), f"{time.monotonic() - t_req:.1f}s", flush=True)
            time.sleep(max(0.0, PAUSE - (time.monotonic() - t_req)))  # <= 1 request start / PAUSE s
            if step is None and not d.get("next"):
                break


if __name__ == "__main__":
    main()
