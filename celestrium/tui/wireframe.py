"""wireframe.py — an ultra-light software 3D engine + a Textual wireframe widget.

The cockpit's ELITE-era flourish: a rotating Platonic-solid wireframe drawn in
the corner, all in pure Python — no numpy, no GPU, just vertices, edges, a couple
of rotation matrices and a Braille canvas. Each terminal cell becomes a 2×4 grid
of Braille dots, so a small panel still draws crisp vector lines.

It's deliberately frame-counter driven (not wall-clock), so the look is
deterministic and the renderer is unit-testable without a running event loop:

    canvas = BrailleCanvas(40, 20)
    render_solid(canvas, SOLIDS["icosahedron"], angle=0.7)
    print(canvas.to_text())          # a frame, as text

The Textual `Wireframe` widget just ticks a frame counter on a timer and re-renders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Geometry — the five Platonic solids as (vertices, edges).
# Vertices are unit-ish; edges index into the vertex list.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Solid:
    name: str
    verts: tuple        # tuple of (x, y, z)
    edges: tuple        # tuple of (i, j)


def _tetrahedron() -> Solid:
    v = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    e = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return Solid("tetrahedron", v, e)


def _cube() -> Solid:
    v = tuple((x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1))
    # indices: bit pattern xyz -> 0..7
    e = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    return Solid("cube", v, e)


def _octahedron() -> Solid:
    v = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    e = ((0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3),
         (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5))
    return Solid("octahedron", v, e)


def _icosahedron() -> Solid:
    p = (1 + math.sqrt(5)) / 2  # golden ratio
    raw = [
        (-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0),
        (0, -1, p), (0, 1, p), (0, -1, -p), (0, 1, -p),
        (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1),
    ]
    e = ((0, 1), (0, 5), (0, 7), (0, 10), (0, 11), (1, 5), (1, 7), (1, 8),
         (1, 9), (2, 3), (2, 4), (2, 6), (2, 10), (2, 11), (3, 4), (3, 6),
         (3, 8), (3, 9), (4, 5), (4, 9), (4, 11), (5, 9), (5, 11), (6, 7),
         (6, 8), (6, 10), (7, 8), (7, 10), (8, 9), (10, 11))
    return Solid("icosahedron", tuple(raw), e)


def _dodecahedron() -> Solid:
    p = (1 + math.sqrt(5)) / 2
    r = 1 / p
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                v.append((x, y, z))
    for s in (-1, 1):
        for t in (-1, 1):
            v.append((0, s * r, t * p))
            v.append((s * r, t * p, 0))
            v.append((s * p, 0, t * r))
    verts = tuple(v)
    # Build edges by nearest-neighbour distance (each vertex joins its 3 closest).
    edges = set()
    n = len(verts)
    for i in range(n):
        d = sorted(range(n), key=lambda j: _dist2(verts[i], verts[j]))
        for j in d[1:4]:
            edges.add((min(i, j), max(i, j)))
    return Solid("dodecahedron", verts, tuple(sorted(edges)))


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


SOLIDS = {s.name: s for s in (
    _tetrahedron(), _cube(), _octahedron(), _icosahedron(), _dodecahedron())}
SOLID_ORDER = list(SOLIDS)


# --------------------------------------------------------------------------- #
# Braille canvas — each cell is 2 (x) × 4 (y) dots.
# --------------------------------------------------------------------------- #
# Braille dot bit positions (Unicode U+2800 base):
#   (0,0)=0x01 (1,0)=0x08
#   (0,1)=0x02 (1,1)=0x10
#   (0,2)=0x04 (1,2)=0x20
#   (0,3)=0x40 (1,3)=0x80
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


class BrailleCanvas:
    """A monochrome dot canvas that renders to Braille glyphs."""

    def __init__(self, cols: int, rows: int):
        self.cols = max(1, cols)
        self.rows = max(1, rows)
        self.px = self.cols * 2
        self.py = self.rows * 4
        self._cells = bytearray(self.cols * self.rows)

    def clear(self) -> None:
        for i in range(len(self._cells)):
            self._cells[i] = 0

    def set(self, x: int, y: int) -> None:
        if 0 <= x < self.px and 0 <= y < self.py:
            cx, cy = x // 2, y // 4
            self._cells[cy * self.cols + cx] |= _DOTS[y % 4][x % 2]

    def line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham line into the dot grid."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, r: int, filled: bool = False) -> None:
        """Draw a circle (or disk) using the midpoint algorithm."""
        cx, cy, r = int(cx), int(cy), int(r)
        if r <= 0:
            self.set(cx, cy)
            return

        x = r
        y = 0
        err = 0

        while x >= y:
            if filled:
                self.line(cx - x, cy + y, cx + x, cy + y)
                self.line(cx - x, cy - y, cx + x, cy - y)
                self.line(cx - y, cy + x, cx + y, cy + x)
                self.line(cx - y, cy - x, cx + y, cy - x)
            else:
                self.set(cx + x, cy + y)
                self.set(cx + y, cy + x)
                self.set(cx - y, cy + x)
                self.set(cx - x, cy + y)
                self.set(cx - x, cy - y)
                self.set(cx - y, cy - x)
                self.set(cx + y, cy - x)
                self.set(cx + x, cy - y)

            if err <= 0:
                y += 1
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

    def to_text(self) -> str:
        out = []
        for cy in range(self.rows):
            row = self._cells[cy * self.cols:(cy + 1) * self.cols]
            out.append("".join(chr(0x2800 + b) for b in row))
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Projection + render.
# --------------------------------------------------------------------------- #
def _rotate(v, ax, ay, az):
    x, y, z = v
    # X
    cy_, sy_ = math.cos(ax), math.sin(ax)
    y, z = y * cy_ - z * sy_, y * sy_ + z * cy_
    # Y
    cx_, sx_ = math.cos(ay), math.sin(ay)
    x, z = x * cx_ + z * sx_, -x * sx_ + z * cx_
    # Z
    cz_, sz_ = math.cos(az), math.sin(az)
    x, y = x * cz_ - y * sz_, x * sz_ + y * cz_
    return x, y, z


def render_solid(canvas: BrailleCanvas, solid: Solid, angle: float,
                 tilt: float = 0.0) -> None:
    """Draw `solid` rotated by `angle` (radians) onto the canvas (cleared first)."""
    canvas.clear()
    ax = tilt + angle * 0.55
    ay = angle
    az = angle * 0.23
    # scale to fit, leave a margin
    scale = min(canvas.px, canvas.py) * 0.36
    cx, cy = canvas.px / 2, canvas.py / 2
    dist = 4.5
    proj = []
    for v in solid.verts:
        x, y, z = _rotate(v, ax, ay, az)
        f = dist / (dist + z)  # simple perspective
        proj.append((cx + x * scale * f, cy + y * scale * f))
    for i, j in solid.edges:
        x0, y0 = proj[i]
        x1, y1 = proj[j]
        canvas.line(x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# Scene Abstractions
# --------------------------------------------------------------------------- #
class Scene:
    """Base class for all ambient visualizations in the Braille canvas."""
    def render(self, canvas: BrailleCanvas, frame: int) -> None:
        pass


class PlatonicScene(Scene):
    """The default spinning Platonic solid scene."""
    def __init__(self, solid_name: str = "icosahedron"):
        self.solid_name = solid_name

    def render(self, canvas: BrailleCanvas, frame: int) -> None:
        if self.solid_name in SOLIDS:
            render_solid(canvas, SOLIDS[self.solid_name], angle=frame * 0.12)


class TransitScene(Scene):
    """Ambient visualization of a transiting exoplanet."""
    def render(self, canvas: BrailleCanvas, frame: int) -> None:
        cx, cy = canvas.px // 2, canvas.py // 2
        star_r = min(canvas.px, canvas.py) // 3
        # Draw host star
        canvas.circle(cx, cy, star_r, filled=False)

        # Calculate planet position (simple looping phase for now)
        phase = (frame % 200) / 200.0
        px = int(cx - star_r * 2 + phase * star_r * 4)
        py = int(cy + star_r * 0.15)

        # Draw transiting planet
        canvas.circle(px, py, max(1, star_r // 4), filled=True)


class SkyScatterScene(Scene):
    """Ambient visualization of a query scatter plot."""
    def __init__(self, points=None):
        self.pts = points or []

    def render(self, canvas: BrailleCanvas, frame: int) -> None:
        for x, y in self.pts:
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Add slight drift
                dx = math.sin(frame * 0.02 + x * 10) * 0.02
                px = int((x + dx) * (canvas.px - 1))
                py = int(y * (canvas.py - 1))
                canvas.set(px, py)


# --------------------------------------------------------------------------- #
# Textual widget.
# --------------------------------------------------------------------------- #
try:
    from rich.text import Text
    from textual.reactive import reactive
    from textual.widgets import Static

    class Wireframe(Static):
        """A rotating Platonic-solid wireframe; ticks a frame counter on a timer."""

        DEFAULT_CSS = """
        Wireframe {
            height: 1fr;
            min-height: 6;
            content-align: center middle;
            color: $primary;
            background: $panel;
        }
        """

        frame = reactive(0)
        spinning = reactive(True)

        def __init__(self, scene: Scene = None, fps: float = 12.0, **kw):
            super().__init__("", **kw)
            # Default to Platonic Scene to preserve legacy behaviour
            self.scene = scene or PlatonicScene("icosahedron")
            self._fps = fps
            self._timer = None

        def on_mount(self) -> None:
            self._timer = self.set_interval(1 / self._fps, self._tick)

        def _tick(self) -> None:
            if self.spinning:
                self.frame += 1

        def next_solid(self) -> str:
            if isinstance(self.scene, PlatonicScene):
                i = (SOLID_ORDER.index(self.scene.solid_name) + 1) % len(SOLID_ORDER)
                self.scene.solid_name = SOLID_ORDER[i]
                self.refresh()
                return self.scene.solid_name
            return ""

        def set_solid(self, name: str) -> None:
            if name in SOLIDS:
                self.scene = PlatonicScene(name)
                self.refresh()

        @property
        def solid_name(self) -> str:
            return self.scene.solid_name if isinstance(self.scene, PlatonicScene) else ""

        def toggle(self) -> bool:
            self.spinning = not self.spinning
            return self.spinning

        def watch_frame(self) -> None:
            self.refresh()

        def render(self):
            w = max(8, self.content_size.width or 24)
            h = max(4, self.content_size.height or 8)
            canvas = BrailleCanvas(w, h)
            if self.scene:
                self.scene.render(canvas, self.frame)
            return Text(canvas.to_text(), no_wrap=True, overflow="crop")

except ImportError:  # textual/rich absent — the geometry half still imports
    Wireframe = None  # type: ignore


if __name__ == "__main__":
    # Pure-text spin preview (no Textual needed): a few frames per solid.
    # Screen clear uses a constant ANSI escape (no shell — avoids os.system).
    import sys
    import time

    canvas = BrailleCanvas(48, 24)
    name = SOLID_ORDER[0]
    for f in range(200):
        if f % 40 == 0:
            name = SOLID_ORDER[(f // 40) % len(SOLID_ORDER)]
        render_solid(canvas, SOLIDS[name], angle=f * 0.12)
        sys.stdout.write("\033[2J\033[H")  # clear + home
        sys.stdout.write(f"  {name}\n\n{canvas.to_text()}\n")
        sys.stdout.flush()
        time.sleep(0.05)
