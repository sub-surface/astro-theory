"""Every capability the instrument has, one module per family.

Importing this package registers them all. Adding a capability is adding a
decorated function here — no surface edits, no parallel tables, no parity
matrix to maintain.
"""
from . import (analysis, archives, feeds, imaging, lit, notebook, objects,  # noqa: F401
               studies, tabular)

__all__ = ["analysis", "archives", "feeds", "imaging", "lit", "notebook",
           "objects", "studies", "tabular"]
