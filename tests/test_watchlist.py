"""Hermetic checks on the observing calendar (celestrium/watchlist.py)."""
from datetime import date

from celestrium import watchlist as W


def test_entries_are_well_formed():
    ids = [e.id for e in W.EVENTS]
    assert len(ids) == len(set(ids))
    for e in W.EVENTS:
        assert e.kind in W.KINDS and e.status in W.STATUSES
        if e.kind == "triggered":
            assert e.start is None
            continue
        assert e.start_date is not None and e.end_date >= e.start_date
        if e.kind in ("observable", "forecast"):
            assert e.test, f"{e.id} needs a pre-registered test"


def test_upcoming_overdue_and_ics():
    today = date(2026, 10, 13)
    soon = {e.id for e in W.upcoming(30, today=today)}
    assert "euclid-dr1" in soon and "tcrb-uv-recovery" in soon
    assert "sello-feb2027" not in soon
    late = {e.id for e in W.overdue(today=date(2026, 12, 15))}
    assert {"tcrb-uv-recovery", "marchev-atel17784"} <= late
    ics = W.to_ics(W.upcoming(400, today=today))
    assert ics.startswith("BEGIN:VCALENDAR") and ics.count("BEGIN:VEVENT") >= 8
    assert "DTSTART;VALUE=DATE:20261021" in ics
