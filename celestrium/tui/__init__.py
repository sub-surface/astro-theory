"""celestrium.tui — the Celestrium Textual cockpit.

A thin presenter over the same brain the CLI uses (config + core/kernel).
Async so the UI never blocks: Resolve · Literature · Query · Crossmatch · Candidates
· History, plus the interactive desk loop (crossmatch → candidate lists) and a
care-and-attention pass — recolouring themes (ansi-dark default), a rotating-
wireframe orrery (`wireframe.py`), a contextual detail panel, image settings with a
no-coverage fallback, and debug/refresh controls. Launch: `python -m celestrium.tui`
or `python -m celestrium tui`. See ../../docs/roadmap.md (Phase 2/3/3.5).
"""
from .app import CelestriumApp

__all__ = ["CelestriumApp"]
