"""Quick-look plots of tabular data — *see the numbers*, not just a grid.

The validation wing's missing primitive: after a query/crossmatch, render a
scatter, histogram, sky map, or colour–magnitude diagram of any columns as a
PNG under data/plots/. Pure matplotlib over an in-memory astropy Table, so
both surfaces (CLI `plot`, TUI `plot` idiom) share it and tests stay hermetic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # render-to-file; never require a display
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from . import paths

KINDS = ("scatter", "hist", "sky", "cmd")


def _column(tab: Table, name: str) -> np.ndarray:
    if name not in tab.colnames:
        raise KeyError(f"no column {name!r}; available: {', '.join(tab.colnames)}")
    col = np.asarray(tab[name], dtype="float64")
    return col


def plot_table(tab: Table, x: Optional[str] = None, y: Optional[str] = None,
               kind: str = "scatter", title: str = "", out: Optional[Path] = None) -> Path:
    """Render a quick-look plot of table columns; returns the saved PNG path.

    kind='scatter'  x vs y                     (needs x and y)
    kind='hist'     histogram of x             (needs x)
    kind='sky'      RA/Dec positions           (x/y default to 'ra'/'dec')
    kind='cmd'      colour–magnitude: x is the colour, y the magnitude
                    (y-axis inverted, astronomer-style)
    """
    if kind not in KINDS:
        raise ValueError(f"unknown plot kind {kind!r}; choose one of {', '.join(KINDS)}")
    if kind == "sky":
        x, y = x or "ra", y or "dec"
    if x is None:
        raise ValueError("plot needs at least an X column")
    if kind != "hist" and y is None:
        raise ValueError(f"plot kind {kind!r} needs X and Y columns")

    xs = _column(tab, x)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    try:
        if kind == "hist":
            good = xs[np.isfinite(xs)]
            ax.hist(good, bins=min(60, max(10, int(len(good) ** 0.5) * 2)),
                    color="#4c72b0", edgecolor="white", linewidth=0.3)
            ax.set_xlabel(x)
            ax.set_ylabel("N")
        else:
            ys = _column(tab, y)
            good = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[good], ys[good], s=8, alpha=0.7, color="#4c72b0",
                       edgecolors="none")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            if kind == "sky":
                ax.set_xlim(360, 0)          # RA increases leftward on the sky
                ax.set_xlabel(f"{x} [deg]")
                ax.set_ylabel(f"{y} [deg]")
            if kind == "cmd":
                ax.invert_yaxis()            # brighter is up
        n = int(np.count_nonzero(np.isfinite(xs)))
        ax.set_title(title or f"{kind}: {x}{'' if kind == 'hist' else ' vs ' + str(y)}  (n={n})",
                     fontsize=10)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        if out is None:
            paths.DATA.mkdir(parents=True, exist_ok=True)
            plots_dir = paths.DATA / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{kind}_{x}" + (f"_{y}" if y and kind != "hist" else "")
            out = plots_dir / f"{stem}.png"
        else:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    finally:
        plt.close(fig)
    return Path(out)
