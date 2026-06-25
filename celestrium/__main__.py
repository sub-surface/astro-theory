"""`python -m celestrium` → the Celestrium CLI.

The Textual TUI (Phase 2) will attach here as a `tui` subcommand / `--tui` flag.
"""
from .hub import app

if __name__ == "__main__":
    app()
