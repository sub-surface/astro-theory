"""The Study model: pipeline + grid + metric, and the preregistration hash.

A pipeline is a tuple of `Step`s. A step's parameters may be:

  * a literal            ``nside=64``
  * a grid reference     ``gal_lat_min=P("mask_width")``  ← filled per combination
  * a step reference     ``density=Ref("prev")``          ← the previous artifact

That third form is what makes lineage automatic: a `Ref` resolves to an
artifact id, the kernel sees an ``artifact``-kind parameter, and the edge is
recorded without the study having to know the ledger exists.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.artifact import canonical_text


@dataclass(frozen=True)
class P:
    """A free parameter, filled from the grid."""
    name: str


@dataclass(frozen=True)
class Ref:
    """A reference to an earlier step's artifact ('prev' or a step's `into` name)."""
    step: str = "prev"


@dataclass(frozen=True)
class Step:
    cap: str
    params: dict = field(default_factory=dict)
    into: str = ""

    def describe(self) -> str:
        bits = []
        for key, value in sorted(self.params.items()):
            if isinstance(value, P):
                bits.append(f"{key}=<{value.name}>")
            elif isinstance(value, Ref):
                bits.append(f"{key}=<{value.step}>")
            else:
                bits.append(f"{key}={value!r}")
        return f"{self.cap}({', '.join(bits)})"


def step(cap: str, into: str = "", **params) -> Step:
    return Step(cap=cap, params=params, into=into)


def dig(payload: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path out of a nested dict — 'direction.l' → payload[...]"""
    if not path:
        return default
    node = payload
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


@dataclass(frozen=True)
class Study:
    id: str
    title: str = ""
    claim: str = ""
    signature: str = ""
    refutation: str = ""
    pipeline: tuple = ()
    grid: dict = field(default_factory=dict)
    metric: str = "amplitude"
    metrics: tuple = ()          # extra dotted paths to carry into the surface
    archives: tuple = ()
    rows: str = ""
    leverage: str = ""
    status: str = "scoped"
    verdict: str = ""
    note: str = ""

    # ----- the grid --------------------------------------------------------- #
    def combos(self, overrides: dict | None = None) -> list:
        """Expand the parameter grid into concrete combinations (sorted, stable)."""
        grid = {**self.grid, **(overrides or {})}
        if not grid:
            return [{}]
        keys = sorted(grid)
        values = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]]
                  for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def free_params(self) -> tuple:
        found = set()
        for entry in self.pipeline:
            for value in entry.params.values():
                if isinstance(value, P):
                    found.add(value.name)
        return tuple(sorted(found))

    def unbound(self, overrides: dict | None = None) -> tuple:
        """Free parameters with nothing in the grid to fill them (a config error)."""
        grid = {**self.grid, **(overrides or {})}
        return tuple(p for p in self.free_params() if p not in grid)

    # ----- integrity -------------------------------------------------------- #
    def prereg_hash(self) -> str:
        """Fingerprint of the *analysis decisions*: pipeline, grid, metric.

        Stored on first run and never silently replaced. If it changes after
        results exist, the study is flagged — the analysis moved after seeing
        the data, which is exactly the thing the vault's refutation rule is
        reaching for and which nothing else in the toolchain records.
        """
        spine = [
            [(s.cap, sorted((k, canonical_text(v)) for k, v in s.params.items()))
             for s in self.pipeline],
            sorted((k, canonical_text(v)) for k, v in self.grid.items()),
            self.metric, list(self.metrics),
        ]
        import hashlib
        return hashlib.blake2b(canonical_text(spine).encode(),
                               digest_size=8).hexdigest()

    # ----- serialisation ---------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "claim": self.claim,
            "signature": self.signature, "refutation": self.refutation,
            "pipeline": [{"cap": s.cap, "into": s.into,
                          "params": {k: (f"<{v.name}>" if isinstance(v, P)
                                         else f"<{v.step}>" if isinstance(v, Ref)
                                         else v)
                                     for k, v in s.params.items()}}
                         for s in self.pipeline],
            "grid": dict(self.grid), "metric": self.metric,
            "metrics": list(self.metrics), "archives": list(self.archives),
            "rows": self.rows, "leverage": self.leverage, "status": self.status,
            "verdict": self.verdict, "note": self.note,
            "prereg": self.prereg_hash(), "runs": len(self.combos()),
        }

    def calls(self, overrides: dict | None = None) -> list:
        """(capability, literal-params) pairs for a cost estimate — Ref-valued
        params are omitted because their ids aren't known until the run."""
        out = []
        for combo in self.combos(overrides):
            for entry in self.pipeline:
                params = {k: (combo.get(v.name) if isinstance(v, P) else v)
                          for k, v in entry.params.items()
                          if not isinstance(v, Ref)}
                out.append((entry.cap, params))
        return out


def leverage_score(study: Study) -> float:
    """Leverage per row — the vault's own ranking rule, made computable.

    `rows: ~50000` means a lunchtime; `rows: ~500M` means a project. Ranking
    high-leverage-per-row is how a desk competes with a telescope.
    """
    weight = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(
        str(study.leverage).strip().lower(), 1.0)
    text = str(study.rows).strip().lstrip("~").replace(",", "").lower()
    multiplier = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9}
    try:
        rows = (float(text[:-1]) * multiplier[text[-1]] if text and text[-1] in multiplier
                else float(text))
    except (ValueError, IndexError):
        rows = 0.0
    if rows <= 0:
        return weight
    import math
    # Log-scaled: an order of magnitude more rows costs one unit of leverage.
    return weight / max(math.log10(rows), 1.0)
