"""Compatibility layer mapping legacy hub imports to unified celestrium.cli."""
import sys
from . import cli
from .cli import (
    app,
    _ensure_utf8_console,
    _emit,
    _fail,
    _printer,
    _run,
    _load,
    _kernel,
    REPORTS_DIR,
    QUERY_ARCHIVES,
    SAMPLE_RECIPES,
    ATLAS_TARGETS,
)


class _HubModule(sys.modules[__name__].__class__):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(cli, name):
            setattr(cli, name, value)


sys.modules[__name__].__class__ = _HubModule

if __name__ == "__main__":
    _ensure_utf8_console()
    app()
