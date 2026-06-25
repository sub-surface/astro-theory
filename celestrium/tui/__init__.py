"""celestrium.tui — the Celestrium Textual cockpit.

A thin presenter over the same brain the CLI uses (registry + packets + cache).
Read-only Phase 2 skeleton: Resolve · Literature · Query · History, async so the
UI never blocks. Launch: `python -m celestrium.tui` or `python -m celestrium tui`.
See ../../docs/roadmap.md (Phase 2/3).
"""
from .app import CelestriumApp

__all__ = ["CelestriumApp"]
