"""Canonical on-disk locations, shared by both surfaces.

`hub.py` keeps its own module-level dir attrs (the test suite monkeypatches
them), but the TUI has no such re-export, so the runbook runner reads these.
All of `data/` is git-ignored (see CLAUDE.md)."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REPORTS_DIR = DATA / "reports"
ATLAS_DIR = DATA / "atlas"
POSTERS_DIR = DATA / "posters"
