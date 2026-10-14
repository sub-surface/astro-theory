"""Observing calendar: dated data drops, pre-registered observables and forecasts to score.

One curated list (`EVENTS`) shared by all agents. Each entry says *when* to look, *what* to look at,
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

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

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

    @property
    def start_date(self) -> Optional[date]:
        return date.fromisoformat(self.start) if self.start else None

    @property
    def end_date(self) -> Optional[date]:
        return date.fromisoformat(self.end) if self.end else self.start_date

    def to_dict(self) -> dict:
        return asdict(self)


EVENTS: List[WatchEvent] = [
    # ------------------------------------------------------------------ data releases
    WatchEvent("euclid-dr1", "2026-10-21", None, "release", "Euclid",
               "Euclid DR1 (~1900 deg2 wide survey)",
               "Run the partial-sky dipole estimator same day; DR1 selection is independent of WISE/Gaia.",
               source="docs/programmes/dipole.md", project="EXP-2026-01"),
    WatchEvent("gaia-dr4", "2026-12-01", "2026-12-31", "release", "Gaia",
               "Gaia DR4 (66 months; epoch astrometry, photometry and RVS spectra)",
               "Confirm the date on the ESA Gaia site; pull T CrB / RS Oph epoch photometry and refresh distances.",
               source="ESA Gaia DR4 schedule (verify)", project="EXP-2026-Y"),
    # ------------------------------------------------------------------ T CrB observables (ours)
    WatchEvent("tcrb-uv-recovery", "2026-11-05", "2026-11-30", "observable", "T CrB",
               "Does the Sep 2026 UV collapse recover by orbital phase ~0.3?",
               "Swift/UVOT UVM2 and XRT 2-10 keV in the newest HEASARC pointings "
               "(scripts/novae/fetch_tcrb_swift_uvot.py, fetch_tcrb_swift_xrt.py).",
               test="Geometric (orbit-locked obscuration) if UVM2 brightens by >= 1 mag from the 29 Sep 2026 level "
                    "by 2026-11-30 (phase 0.3 = 2026-11-12) with the hard X-rays falling again; a change of accretion "
                    "state if UVM2 stays within 0.5 mag of its quiescent pre-2015 level through 2026-11-30.",
               source="EXP-2026-Y uv_dips (two collapses 5.04 orbits apart, chance p = 0.08)", project="EXP-2026-Y"),
    WatchEvent("tcrb-conjunction-2027", "2027-04-20", None, "observable", "T CrB",
               "Next giant-in-front conjunction (deeper ellipsoidal minimum)",
               "If a third UV collapse begins within ~0.1 in phase after this date, the orbit-locked hypothesis "
               "gains a third coincidence.",
               test="Orbit-locked if a UV collapse (>= 1.5 mag UVM2 in < 45 d) starts between 2027-04-20 and 2027-06-10.",
               source="EXP-2026-Y ephemeris", project="EXP-2026-Y"),
    WatchEvent("tcrb-1946-analogue", "2027-03-01", "2027-08-01", "observable", "T CrB",
               "1946 analogue window: eruption 0.63 yr after the Sep 2026 UV collapse (2027.38 = 2027-05-19)",
               "AAVSO alerts / ATels; the overlay stage in scripts/novae/tcrb_analysis.py.",
               test="Analogue survives if T CrB erupts between 2027-03-01 and 2027-08-01; dead (like the 2023 alignment) "
                    "if no eruption by 2027-08-01.",
               source="EXP-2026-Y dip_overlay alignment B", project="EXP-2026-Y"),
    WatchEvent("tcrb-recurrence-median", "2027-02-14", "2030-11-25", "observable", "T CrB",
               "Recurrence-clock forecast: 68% window 2027.12-2030.90, median 2028.13",
               "Score at eruption: where does the date fall in our survival-conditioned log-normal?",
               test="Report the predictive CDF at the eruption date (calibrated forecasts give uniform CDF values).",
               source="EXP-2026-Y forecast.recurrence", project="EXP-2026-Y"),
    # ------------------------------------------------------------------ other people's forecasts to score
    WatchEvent("marchev-atel17784", "2026-10-15", "2026-11-30", "forecast", "T CrB",
               "Marchev+ ATel 17784 (May 2026): eruption within 5-6 months",
               "Lapses if no eruption by end of Nov 2026.", test="Passes if T CrB erupts by 2026-11-30.",
               source="ATel 17784", project="EXP-2026-Y"),
    WatchEvent("iijima-rsoph", "2026-10-01", "2026-11-30", "forecast", "RS Oph",
               "Iijima 2025: next RS Oph outburst ~Oct-Nov 2026",
               "Our Swift data show RS Oph quiescent on 2026-09 (UVM2 14.85). Watch AAVSO / ATels.",
               test="Passes if RS Oph erupts by 2026-11-30.", source="Iijima 2025", project="EXP-2026-Y"),
    WatchEvent("pei-dec2026", "2026-12-01", "2026-12-31", "forecast", "T CrB",
               "Pei+ 2026 early branch: eruption Dec 2026 (late branch >= 2029.4)",
               "", test="Early branch passes if T CrB erupts in Dec 2026.",
               source="2026MNRAS.550g1361P", project="EXP-2026-Y"),
    WatchEvent("schneider-2027a", "2027-01-25", "2027-02-22", "forecast", "T CrB",
               "Schneider 2024 RNAAS discrete date 2027.106 (2027-02-08)",
               "", test="Passes if the eruption lands within +/- 2 weeks of 2027-02-08.",
               source="Schneider 2024 RNAAS", project="EXP-2026-Y"),
    WatchEvent("sello-feb2027", "2027-02-12", "2027-03-12", "forecast", "T CrB",
               "Sello+ 2026: eruption 26 Feb 2027",
               "", test="Passes if the eruption lands within +/- 2 weeks of 2027-02-26.",
               source="Sello+ 2026", project="EXP-2026-Y"),
    # ------------------------------------------------------------------ our own targets
    WatchEvent("exp-04-bald", "2026-11-10", None, "deadline", "DESI / 4MOST",
               "EXP-2026-04 BALD active learning target date", "docs/ledger.md",
               project="EXP-2026-04"),
    WatchEvent("exp-05-zoa", "2026-12-01", None, "deadline", "eROSITA + CatWISE",
               "EXP-2026-05 Zone-of-Avoidance fusion target date", "docs/ledger.md",
               project="EXP-2026-05"),
    # ------------------------------------------------------------------ triggered (on eruption)
    WatchEvent("tcrb-eruption-playbook", None, None, "triggered", "T CrB",
               "On eruption: score every open T CrB forecast; pull Swift/NuSTAR/Fermi-LAT; check the "
               "Shara+ 2024 30 pc super-remnant for a light echo (they predict none); compare the shock "
               "X-ray rise with RS Oph 2006/2021 (data/srn/rs_oph).",
               "Re-run scripts/novae/tcrb_literature.py scoreboard with the eruption date.",
               test="Super-remnant: Shara+ prediction passes if no fluorescent echo is detected.",
               source="2024ApJ...977L..48S", project="EXP-2026-Y"),
]


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
