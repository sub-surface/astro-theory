"""`python -m celestrium` → the Celestrium CLI."""
from .cli import _ensure_utf8_console, app

if __name__ == "__main__":
    _ensure_utf8_console()
    app()
