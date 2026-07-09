"""`python -m celestrium` → the Celestrium CLI.

The Textual TUI (Phase 2) will attach here as a `tui` subcommand / `--tui` flag.
"""
from .hub import _ensure_utf8_console, app

if __name__ == "__main__":
    # Before Typer even starts parsing args: a Windows cp1252 console must
    # never crash on --help or docs-map Unicode (see hub._ensure_utf8_console).
    _ensure_utf8_console()
    app()
