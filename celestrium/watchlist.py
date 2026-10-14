"""Observing calendar: dated data drops, pre-registered observables and forecasts to score.

One curated list, `celestrium/watchlist.json` (loaded here as `EVENTS`, and read directly by the website build). Each entry says *when* to look, *what* to look at,
and — for observables and forecasts — the pre-registered test that decides the outcome, written
before the data arrive so it cannot be tuned afterwards. When an entry's window closes, score it:
set `status` to "passed" / "failed" / "lapsed" and record what happened in `outcome`.

    celestrium calendar                  # next 120 days + anything overdue for scoring
    celestrium calendar --days 400 --ics watch.ics   # export for a phone/desktop calendar

Kinds: "release" (data drop), "observable" (a measurable prediction of ours), "forecast"
(someone else's dated prediction we are scoring), "deadline" (our own target dates),
"triggered" (fires on an event rather than a date; `start` is None).
Dates are ISO strings; the T CrB orbital dates use P = 227.60 d with the deeper (giant-in-front)
minimum at JD 2461061.17 (EXP-2026-Y ephemeris stage).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

EVENTS_FILE = Path(__file__).with_name("watchlist.json")
KINDS = ("release", "observable", "forecast", "deadline", "triggered")
STATUSES = ("open", "passed", "failed", "lapsed", "done")


@dataclass(frozen=True)
class WatchEvent:
    id: str
    start: Optional[str]          # ISO date; None for "triggered"
    end: Optional[str]            # ISO date closing the window (None = single day)
    kind: str
    target: str
    title: str
    check: str                    # what to pull / look at
    test: str = ""                # pre-registered pass/fail criterion (observables & forecasts)
    source: str = ""              # bibcode, ATel, experiment id, or URL
    project: str = ""             # e.g. EXP-2026-Y
    status: str = "open"
    outcome: str = ""
    group: str = ""               # section heading in the calendar

    @property
    def start_date(self) -> Optional[date]:
        return date.fromisoformat(self.start) if self.start else None

    @property
    def end_date(self) -> Optional[date]:
        return date.fromisoformat(self.end) if self.end else self.start_date

    def to_dict(self) -> dict:
        return asdict(self)


def load_events(path: Path = EVENTS_FILE) -> List[WatchEvent]:
    """Read the calendar from its JSON source (also read by the website build)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [WatchEvent(**e) for e in doc["events"]]


EVENTS: List[WatchEvent] = load_events()


def _today(today: Optional[date] = None) -> date:
    return today or datetime.now(timezone.utc).date()


def upcoming(days: int = 120, today: Optional[date] = None, events: Optional[List[WatchEvent]] = None):
    """Open dated events whose window overlaps [today, today + days], sorted by start."""
    t0 = _today(today)
    t1 = t0 + timedelta(days=days)
    evs = EVENTS if events is None else events
    out = [e for e in evs if e.status == "open" and e.start_date
           and e.start_date <= t1 and e.end_date >= t0]
    return sorted(out, key=lambda e: e.start_date)


def overdue(today: Optional[date] = None, events: Optional[List[WatchEvent]] = None):
    """Open dated events whose window has closed: these need scoring (passed / failed / lapsed)."""
    t0 = _today(today)
    evs = EVENTS if events is None else events
    return [e for e in evs if e.status == "open" and e.end_date and e.end_date < t0]


def triggered(events: Optional[List[WatchEvent]] = None):
    evs = EVENTS if events is None else events
    return [e for e in evs if e.kind == "triggered" and e.status == "open"]


def _ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def to_ics(events: List[WatchEvent]) -> str:
    """All-day iCalendar events (RFC 5545) for import into any calendar app."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Celestrium//watchlist//EN"]
    for e in events:
        if not e.start_date:
            continue
        desc = " | ".join(x for x in (e.check, ("TEST: " + e.test) if e.test else "", e.source) if x)
        lines += ["BEGIN:VEVENT", f"UID:{e.id}@celestrium", f"DTSTAMP:{stamp}",
                  f"DTSTART;VALUE=DATE:{e.start_date:%Y%m%d}",
                  f"DTEND;VALUE=DATE:{e.end_date + timedelta(days=1):%Y%m%d}",
                  f"SUMMARY:{_ics_escape(f'[{e.kind}] {e.target}: {e.title}')}",
                  f"DESCRIPTION:{_ics_escape(desc)}", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"
