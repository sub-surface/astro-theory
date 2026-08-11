"""Studies — a claim, a pipeline, and the grid of cuts you vary against it.

The research object the instrument was missing. `Astronomy/Observables/*.md` in
the vault already defines the *claim* half (signature, refutation, leverage,
rows); this adds the machine half (pipeline, grid, metric) and binds the two by
`id`, so neither side duplicates the other.
"""
from .model import P, Ref, Step, Study, dig, step
from .library import STUDIES, get, names

__all__ = ["P", "Ref", "Step", "Study", "step", "dig",
           "STUDIES", "get", "names"]
