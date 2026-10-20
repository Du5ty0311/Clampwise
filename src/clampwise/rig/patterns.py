"""The Polygon's patterns: shapes on its field, and the image of mirrors that lights them.

The Polygon1000's DMD is 912 columns by 1140 rows of diamond mirrors: each row is offset by half a mirror from the
one before, and rows are half a mirror apart, so the field is 912 by 570 mirror widths (16:10), not 912 by 1140. A
shape is given in the field's own coordinates, ``u`` across and ``v`` down, each 0 to 1 (as PolyScan keeps its
regions), and lit where the centres of the mirrors fall inside it; a round spot is round on the slice.

A pattern is a list of steps, each a name and the shapes lit together: a full field, a spot, a numbered grid of
spots (1 at the top left, along the rows) over the field or over the camera's picture, a line of spots, an image
file, a PolyScan sequence (``*.seq``: its regions of interest, ROI1, ROI2...), or shapes drawn on the camera's
picture. Several steps can be lit together as one.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COLUMNS, ROWS = 912, 1140                  # the DMD's mirrors, as the image sent to it is laid out
FIELD_W, FIELD_H = 912.0, 570.0            # the field in mirror widths: rows are half a mirror apart


def mirror_centres() -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of every mirror's centre on the field, each (ROWS, COLUMNS), 0 to 1."""
    r, c = np.mgrid[0:ROWS, 0:COLUMNS].astype(np.float64)
    x = c + 0.5 + 0.5 * (r % 2)                  # odd rows sit half a mirror over
    y = (r + 1.0) / 2.0
    return x / (FIELD_W + 0.5), y / (FIELD_H + 0.5)


_CENTRES: tuple[np.ndarray, np.ndarray] | None = None


def _centres() -> tuple[np.ndarray, np.ndarray]:
    global _CENTRES
    if _CENTRES is None:
        _CENTRES = mirror_centres()
    return _CENTRES


@dataclass
class Shape:
    """A region of the field: ``kind`` rect, ellipse or polygon; ``points`` its box (left, top, right, bottom) or,
    for a polygon, its corners, all in field coordinates (0 to 1)."""
    kind: str
    points: tuple[float, ...]

    def contains(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.kind == "polygon":
            return _inside_polygon(u, v, np.asarray(self.points, float).reshape(-1, 2))
        l, t, r, b = self.points
        l, r = min(l, r), max(l, r)
        t, b = min(t, b), max(t, b)
        if self.kind == "rect":                       # half open, so a grid's cells share no mirror
            return (u >= l) & ((u < r) | (r >= 1.0)) & (v >= t) & ((v < b) | (b >= 1.0))
        cu, cv, au, av = (l + r) / 2, (t + b) / 2, max((r - l) / 2, 1e-9), max((b - t) / 2, 1e-9)
        return ((u - cu) / au) ** 2 + ((v - cv) / av) ** 2 <= 1.0

    def centre(self) -> tuple[float, float]:
        p = np.asarray(self.points, float)
        if self.kind == "polygon":
            p = p.reshape(-1, 2)
            return float(p[:, 0].mean()), float(p[:, 1].mean())
        return float((p[0] + p[2]) / 2), float((p[1] + p[3]) / 2)

    def outline(self, n: int = 48) -> np.ndarray:
        """Points around the shape (k, 2), closed, in field coordinates."""
        if self.kind == "polygon":
            p = np.asarray(self.points, float).reshape(-1, 2)
            return np.vstack([p, p[:1]])
        l, t, r, b = self.points
        if self.kind == "rect":
            return np.array([[l, t], [r, t], [r, b], [l, b], [l, t]], float)
        a = np.linspace(0, 2 * np.pi, n + 1)
        return np.column_stack([(l + r) / 2 + (r - l) / 2 * np.cos(a), (t + b) / 2 + (b - t) / 2 * np.sin(a)])


def _inside_polygon(u: np.ndarray, v: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(u.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        (ui, vi), (uj, vj) = poly[i], poly[j]
        crosses = (vi > v) != (vj > v)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = (uj - ui) * (v - vi) / (vj - vi) + ui
        inside ^= crosses & (u < at)
        j = i
    return inside


@dataclass
class Step:
    """One pattern shown at once: its name (``1``, ``ROI7``, ``Full field``) and its shapes, or a mask of mirrors
    (an image file's)."""
    name: str
    shapes: list[Shape] = field(default_factory=list)
    mask: np.ndarray | None = None

    def mirrors(self) -> np.ndarray:
        """Which mirrors light this step: bool (ROWS, COLUMNS)."""
        if self.mask is not None:
            return self.mask
        u, v = _centres()
        lit = np.zeros((ROWS, COLUMNS), bool)
        for s in self.shapes:
            lit |= s.contains(u, v)
        return lit

    def covers(self, u: float, v: float) -> bool:
        """Whether the point (u, v) of the field is in this step: in one of its shapes, or on a lit mirror of its
        image (the mirror nearest the point)."""
        if self.mask is not None:
            r = int(np.clip(round(v * (FIELD_H + 0.5) * 2 - 1), 0, ROWS - 1))
            c = int(np.clip(round(u * (FIELD_W + 0.5) - 0.5 - 0.5 * (r % 2)), 0, COLUMNS - 1))
            return bool(self.mask[r, c])
        a, b = np.array([float(u)]), np.array([float(v)])
        return any(bool(s.contains(a, b)[0]) for s in self.shapes)

    def describe(self) -> str:
        if self.mask is not None:
            return f"{self.name}: an image ({self.mask.mean():.1%} of the mirrors)"
        parts = []
        for s in self.shapes:
            pts = ", ".join(f"{x:.4f}" for x in s.points)
            parts.append(f"{s.kind}({pts})")
        return f"{self.name}: " + "; ".join(parts)


def image(step_mask: np.ndarray, on_value: int = 255) -> np.ndarray:
    """The image the Polygon is sent: one byte a mirror, ``on_value`` where lit, the other value elsewhere."""
    off = 0 if on_value else 255
    out = np.full((ROWS, COLUMNS), off, np.uint8)
    out[step_mask] = on_value
    return np.ascontiguousarray(out)


# ------------------------------------------------------------------ patterns

def full_field() -> list[Step]:
    return [Step("Full field", [Shape("rect", (0.0, 0.0, 1.0, 1.0))])]


def spot(u: float, v: float, diameter: float) -> list[Step]:
    """A round spot at (u, v), ``diameter`` across the field's width (round on the slice: its height in field
    units is scaled by the field's 16:10 shape)."""
    du, dv = diameter / 2, diameter / 2 * FIELD_W / FIELD_H
    return [Step("Spot", [Shape("ellipse", (u - du, v - dv, u + du, v + dv))])]


def grid(rows: int, columns: int, region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
         fill: float = 1.0, round_spots: bool = False) -> list[Step]:
    """``rows`` x ``columns`` spots tiling ``region`` (left, top, right, bottom), numbered from 1 at the top left
    along each row (PolyScan's grid: ROI1 to ROI100 for 10 x 10); each spot ``fill`` of its cell across."""
    l, t, r, b = region
    cw, ch = (r - l) / columns, (b - t) / rows
    steps = []
    for i in range(rows):
        for j in range(columns):
            cu, cv = l + (j + 0.5) * cw, t + (i + 0.5) * ch
            hw, hh = cw * fill / 2, ch * fill / 2
            steps.append(Step(str(i * columns + j + 1),
                              [Shape("ellipse" if round_spots else "rect", (cu - hw, cv - hh, cu + hw, cv + hh))]))
    return steps


def grid_on_picture(rows: int, columns: int, cal: "Calibration", size: tuple[int, int], fill: float = 1.0,
                    round_spots: bool = False) -> list[Step]:
    """``rows`` x ``columns`` cells tiling the camera's picture (``size``: its width and height, ``cal`` the map at
    that size) rather than the whole field, numbered as ``grid`` numbers them (1 at the picture's top left, along
    each row): when the field is larger than the picture, every cell is one the camera sees. Each cell is a shape on
    the field through the map; a round spot is round on the picture, and so on the slice."""
    w, h = size
    cw, ch = w / columns, h / rows
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    steps = []
    for i in range(rows):
        for j in range(columns):
            cx, cy = (j + 0.5) * cw, (i + 0.5) * ch
            hw, hh = cw * fill / 2, ch * fill / 2
            if round_spots:
                r = min(hw, hh)
                pts = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
            else:
                pts = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
            u, v = cal.field_of(pts[:, 0], pts[:, 1])
            steps.append(Step(str(i * columns + j + 1),
                              [Shape("polygon", tuple(np.column_stack([u, v]).ravel().tolist()))]))
    return steps


def picture_outside_field(cal: "Calibration", size: tuple[int, int]) -> bool:
    """Whether part of the camera's picture (``size``, ``cal`` the map at that size) lies outside the field: what is
    drawn there cannot be lit."""
    w, h = size
    u, v = cal.field_of(np.array([0.0, w, w, 0.0]), np.array([0.0, 0.0, h, h]))
    return bool((u < -0.005).any() or (u > 1.005).any() or (v < -0.005).any() or (v > 1.005).any())


def together(steps: list[Step]) -> Step:
    """Several steps lit at once, as one step named by theirs joined with " + " (``3 + 17``): their shapes kept
    (each region in the log), an image's mirrors joined to the others'."""
    if len(steps) == 1:
        return steps[0]
    name = " + ".join(s.name for s in steps)
    shapes = [sh for s in steps for sh in s.shapes]
    if any(s.mask is not None for s in steps):
        lit = np.zeros((ROWS, COLUMNS), bool)
        for s in steps:
            lit |= s.mirrors()
        return Step(name, shapes, mask=lit)
    return Step(name, shapes)


def line(u0: float, v0: float, u1: float, v1: float, spots: int, diameter: float) -> list[Step]:
    """``spots`` round spots evenly along a line, numbered from its start."""
    steps = []
    for k, f in enumerate(np.linspace(0.0, 1.0, max(1, spots))):
        s = spot(u0 + f * (u1 - u0), v0 + f * (v1 - v0), diameter)[0]
        steps.append(Step(str(k + 1), s.shapes))
    return steps


def from_image(path: str | Path, threshold: int = 128, invert: bool = False) -> list[Step]:
    """An image file as one step: stretched over the whole field (its own 16:10 shape fits best), lit where its
    brightness is at least ``threshold`` (or below it, ``invert``)."""
    from PIL import Image

    with Image.open(path) as im:
        grey = np.asarray(im.convert("L"), np.float64)
    h, w = grey.shape
    u, v = _centres()
    rows = np.clip((v * h).astype(int), 0, h - 1)
    cols = np.clip((u * w).astype(int), 0, w - 1)
    lit = grey[rows, cols] >= threshold
    return [Step(Path(path).stem, mask=~lit if invert else lit)]


# ------------------------------------------------------------------ PolyScan's sequences

_SHAPE_TYPE = re.compile(rb"PolyScan(?:3|\.Utilities)\.C(Rectangle|Ellipse|Polygon|Circle|PolyLine|Polyline|Line)")
_KINDS = {b"Rectangle": "rect", b"Ellipse": "ellipse", b"Circle": "ellipse"}


_OLD_MAGIC = b"\x88\x88\x88\x88"


def _read_region(data: bytes, at: int) -> tuple[Shape | None, str, int]:
    """One region written by PolyScan from ``at`` (its type's name, its number, its name, a byte, its colour and a
    value; then a rectangle's or an ellipse's flag, box and a word; a polygon's flag, a word and its corners; a
    line's word and its points): the shape (None for a line, which lights nothing), its name and where the next
    byte is."""
    n = data[at]
    m = _SHAPE_TYPE.match(data, at + 1)
    if m is None:
        raise ValueError("not a region")
    at = at + 1 + n + 4
    k = data[at]
    name = data[at + 1:at + 1 + k].decode("latin-1")
    at += 1 + k + 1 + 4 + 4
    kind = m.group(1)
    if kind == b"Line":                                       # its two ends, as a box is kept, and a word
        return None, name, at + 16 + 4
    if kind in (b"Polygon", b"PolyLine", b"Polyline"):
        at += 1 if kind == b"Polygon" else 0
        count = struct.unpack_from("<i", data, at + 4)[0]
        if not 0 <= count < 100000:
            raise ValueError("not a region")
        pts = struct.unpack_from(f"<{2 * count}f", data, at + 8)
        at += 8 + 8 * count
        shape = Shape("polygon", tuple(float(x) for x in pts)) if kind == b"Polygon" and count >= 3 else None
        return shape, name, at
    box = struct.unpack_from("<4f", data, at + 1)
    return Shape(_KINDS[kind], tuple(float(x) for x in box)), name, at + 1 + 16 + 4


def _read_old_seq(data: bytes) -> tuple[list[Step], list[str]]:
    """PolyScan's earlier sequences: each step a grid over the field (columns, rows, one bit a cell, row by row,
    the first column the highest bit), then the regions lit with it."""
    steps: list[Step] = []
    notes: list[str] = []
    n = struct.unpack_from("<i", data, 4)[0]
    at = 8
    for k in range(n):
        cols, rows, _ = struct.unpack_from("<3i", data, at)
        at += 12
        per_row = (cols + 7) // 8
        bits = data[at:at + rows * per_row]
        at += rows * per_row
        _, _, _, nreg = struct.unpack_from("<iffi", data, at)
        at += 16
        shapes = []
        for i in range(rows):
            for j in range(cols):
                if bits[i * per_row + j // 8] & (0x80 >> (j % 8)):
                    shapes.append(Shape("rect", (j / cols, i / rows, (j + 1) / cols, (i + 1) / rows)))
        for _ in range(max(0, nreg)):
            shape, name, at = _read_region(data, at)
            if shape is None:
                notes.append(f"step {k + 1}: {name} is a line (it lights nothing: left out)")
            else:
                shapes.append(shape)
        steps.append(Step(str(k + 1), shapes))
    return steps, notes


def read_seq(path: str | Path) -> tuple[list[Step], list[str]]:
    """A PolyScan sequence as steps, and notes on what could not be read. Its earlier files hold each step as a
    grid of cells and the regions lit with it (read step by step); its later ones hold the regions of interest
    (each a step in their order, ROI1, ROI2..., as its grids number them). The later files also hold who saved
    them and the computer's identifiers: only the shapes are kept. The region PolyScan calls EWA (the whole area,
    the field itself) is left out."""
    data = Path(path).read_bytes()
    if data[:4] == _OLD_MAGIC:
        try:
            return _read_old_seq(data)
        except (IndexError, struct.error, ValueError, UnicodeDecodeError):
            return [], ["the sequence could not be read (PolyScan's earlier format)"]
    steps: list[Step] = []
    notes: list[str] = []
    for m in _SHAPE_TYPE.finditer(data):
        try:
            shape, name, _ = _read_region(data, m.start() - 1)
        except (IndexError, struct.error, UnicodeDecodeError, ValueError):
            notes.append(f"a {m.group(1).decode()} could not be read")
            continue
        if shape is None:
            notes.append(f"{name}: a line (it lights nothing: left out)")
            continue
        p = shape.points
        if name.upper() == "EWA" or not all(-0.01 <= x <= 1.01 for x in p):
            continue
        if shape.kind != "polygon" and (p[2] <= p[0] or p[3] <= p[1]):
            continue
        steps.append(Step(name, [shape]))
    if not steps and not notes:
        notes.append("no regions of interest found")
    return steps, notes


# ------------------------------------------------------------------ the camera and the field

@dataclass
class Calibration:
    """How the Polygon sits under the camera: ``on_value`` the byte that lights a mirror (0 or 255), and the
    affine map from the camera's pixels (x across, y down) to the field (u, v): [u, v] = A @ [x, y, 1], measured
    with one ``objective`` (and camera adapter): the field's place shifts a little between objectives (their
    centring), so a map is used only with the objective it was measured with. ``check`` holds the check grid
    projected after the fit: each spot where it was found and where the map put it (x, y, x, y, camera pixels)."""
    on_value: int = 255
    light_tested: bool = False                                        # on_value seen to light, not assumed
    to_field: list[list[float]] = field(default_factory=list)       # 2 x 3; empty: not calibrated
    camera_size: tuple[int, int] = (0, 0)                             # (width, height) it was measured at
    residual_px: float = float("nan")
    objective: str = ""                                               # "40x, adapter 1x"; "" not known
    check: list[list[float]] = field(default_factory=list)
    um_per_px: float = 0.0                                            # the camera's scale when it was measured

    @property
    def ready(self) -> bool:
        return len(self.to_field) == 2

    def check_errors_px(self) -> np.ndarray:
        """How far each check spot landed from where the map put it, camera pixels."""
        c = np.asarray(self.check, float).reshape(-1, 4)
        return np.hypot(c[:, 0] - c[:, 2], c[:, 1] - c[:, 3])

    def field_area_px(self) -> float:
        """The whole field's area on the camera's picture, square pixels (the field is 1 x 1 in its own units)."""
        a = np.asarray(self.to_field, float)
        det = abs(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) if self.ready else 0.0
        return 1.0 / det if det > 0 else float("nan")

    def field_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(self.to_field, float)
        return a[0, 0] * x + a[0, 1] * y + a[0, 2], a[1, 0] * x + a[1, 1] * y + a[1, 2]

    def camera_of(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.vstack([np.asarray(self.to_field, float), [0.0, 0.0, 1.0]])
        inv = np.linalg.inv(a)
        return inv[0, 0] * u + inv[0, 1] * v + inv[0, 2], inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]

    def at_size(self, size: tuple[int, int]) -> "Calibration":
        """The calibration for the camera's pictures at ``size`` (width, height): measured at one binning, used at
        another (a binned pixel is n pixels across, its centre half way along them)."""
        w0, h0 = self.camera_size
        w, h = size
        if not self.ready or not w0 or not h0 or not w or not h or (w0, h0) == (w, h):
            return self
        a = np.asarray(self.to_field, float).copy()
        sx, sy = w0 / w, h0 / h
        a[:, 2] += a[:, 0] * (sx - 1) / 2 + a[:, 1] * (sy - 1) / 2
        a[:, 0] *= sx
        a[:, 1] *= sy
        check = [[(x0 - (sx - 1) / 2) / sx, (y0 - (sy - 1) / 2) / sy, (x1 - (sx - 1) / 2) / sx,
                  (y1 - (sy - 1) / 2) / sy] for x0, y0, x1, y1 in self.check]
        return Calibration(self.on_value, self.light_tested, a.tolist(), (int(w), int(h)), self.residual_px / sx,
                           self.objective, check, self.um_per_px * sx)

    def to_dict(self) -> dict:
        return {"on_value": self.on_value, "light_tested": self.light_tested, "to_field": self.to_field,
                "camera_size": list(self.camera_size), "residual_px": self.residual_px,
                "objective": self.objective, "check": self.check, "um_per_px": self.um_per_px}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(int(d.get("on_value", 255)), bool(d.get("light_tested", False)),
                   [list(map(float, r)) for r in d.get("to_field", [])],
                   tuple(int(x) for x in d.get("camera_size", (0, 0))), float(d.get("residual_px", float("nan"))),
                   str(d.get("objective", "")), [list(map(float, r)) for r in d.get("check", [])],
                   float(d.get("um_per_px", 0.0) or 0.0))


def objective_key(magnification: float | None, adapter: float | None = 1.0) -> str:
    """The name a calibration is kept under: the objective and the camera adapter it was measured with."""
    if not magnification:
        return ""
    return f"{magnification:g}x, adapter {adapter or 1.0:g}x"


class Calibrations:
    """The calibrations kept: which byte lights a mirror (the Polygon's own: the same with every objective) and one
    map of the field for each objective (and camera adapter) it was measured with."""

    def __init__(self, on_value: int = 255, light_tested: bool = False, maps: dict | None = None) -> None:
        self.on_value, self.light_tested = on_value, light_tested
        self.maps: dict[str, Calibration] = dict(maps or {})

    def at(self, key: str) -> Calibration:
        """The calibration to use with the objective ``key``: its map when one was measured with it (else none),
        and the light test's byte."""
        c = self.maps.get(key)
        if c is None:
            return Calibration(self.on_value, self.light_tested, objective=key)
        c.on_value, c.light_tested = self.on_value, self.light_tested
        return c

    def keep(self, c: Calibration) -> None:
        self.on_value, self.light_tested = c.on_value, c.light_tested
        if c.ready:
            self.maps[c.objective] = c

    def refusal(self, key: str) -> str:
        """Why no map can be used with ``key`` ("" when one can)."""
        if key in self.maps:
            return ""
        others = [k for k in self.maps if k != key]
        if not others:
            return "calibrate first (Test and calibrate)"
        return (f"calibrated at {', '.join(sorted(others))}, not at {key or 'this objective'}: calibrate with this "
                "objective (each objective centres the field a little differently)")

    def to_dict(self) -> dict:
        return {"on_value": self.on_value, "light_tested": self.light_tested,
                "maps": {k: c.to_dict() for k, c in self.maps.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrations":
        """Also reads the single calibration kept before objectives were (its byte only: its map's objective is not
        known)."""
        out = cls(int(d.get("on_value", 255)), bool(d.get("light_tested", False)))
        for k, m in (d.get("maps") or {}).items():
            c = Calibration.from_dict(m)
            if c.ready:
                out.maps[str(k)] = c
        return out


CALIBRATION_SPOTS = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5))
CALIBRATION_DIAMETER = 0.06
# a field larger than the camera's picture puts the outer spots off it: rings of spots nearer the centre then, each
# its corners this far from the centre (field units), smaller spots with them
CLOSER_RINGS = (0.1, 0.04)
# after the fit, a check grid of spots not used by it: how far each lands from where the map puts it
CHECK_SPOTS = tuple((u, v) for v in (0.35, 0.5, 0.65) for u in (0.35, 0.5, 0.65) if (u, v) != (0.5, 0.5))
CHECK_TARGET_UM = 5.0


def ring(d: float) -> tuple[tuple[tuple[float, float], ...], float]:
    """Four calibration spots at the corners of a square ``d`` from the field's centre, and their diameter."""
    return ((0.5 - d, 0.5 - d), (0.5 + d, 0.5 - d), (0.5 - d, 0.5 + d), (0.5 + d, 0.5 + d)), \
        min(CALIBRATION_DIAMETER, 0.4 * d)


def spread_px(points: np.ndarray) -> float:
    """How far points spread across their narrowest direction (camera pixels, the root mean square): near 0 when
    they lie on one line, which cannot fix a map."""
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    return float(np.linalg.svd(p - p.mean(0), compute_uv=False)[-1] / np.sqrt(len(p)))


def check_spots_on_picture(cal: Calibration, size: tuple[int, int]) -> tuple[list[tuple[float, float]], float]:
    """The check grid laid over the camera's picture rather than the field (3 x 3 at a quarter, half and three
    quarters of it, less its centre), so every check spot is one the camera can see when the field is larger than
    the picture; those outside the field left out. And their diameter: a twentieth of the picture's width."""
    w, h = size
    xs = np.array([fx * w for fy in (0.25, 0.5, 0.75) for fx in (0.25, 0.5, 0.75) if (fx, fy) != (0.5, 0.5)])
    ys = np.array([fy * h for fy in (0.25, 0.5, 0.75) for fx in (0.25, 0.5, 0.75) if (fx, fy) != (0.5, 0.5)])
    u, v = cal.field_of(xs, ys)
    (u0, u1), (v0, v1) = cal.field_of(np.array([0.0, w]), np.array([h / 2, h / 2]))
    width = float(np.hypot(u1 - u0, (v1 - v0) * FIELD_H / FIELD_W))           # the picture's width, field widths
    diameter = float(np.clip(0.05 * width, 0.01, CALIBRATION_DIAMETER))
    r = diameter / 2
    spots = [(float(a), float(b)) for a, b in zip(u, v) if r <= a <= 1 - r and r <= b <= 1 - r]
    return spots, diameter


def shape_area(shape: Shape) -> float:
    """A shape's area in field units (the field is 1 x 1)."""
    if shape.kind == "polygon":
        p = np.asarray(shape.points, float).reshape(-1, 2)
        x, y = p[:, 0], p[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
    l, t, r, b = shape.points
    w, h = abs(r - l), abs(b - t)
    return float(w * h if shape.kind == "rect" else np.pi * w * h / 4)


def dose(step: "Step", mask: np.ndarray, cal: Calibration, um_per_px: float) -> dict:
    """What a step lights on the slice: the fraction of the field's mirrors lit; its area in um^2 (from the mirrors
    lit, through the calibration and the camera's scale, when both are known); each shape's centre and area on the
    camera's picture (pixels and um from the picture's top left). The irradiance where it lights is the same for any
    pattern at one LED command (each mirror passes the same light); the power on the slice goes with the area."""
    fraction = float(np.asarray(mask, bool).mean())
    field_px = cal.field_area_px() if cal.ready else float("nan")
    um2 = um_per_px ** 2 if um_per_px else float("nan")
    shapes = []
    for s in step.shapes:
        d = {"kind": s.kind, "field": [round(float(x), 6) for x in s.points]}
        if cal.ready:
            cu, cv = s.centre()
            x, y = cal.camera_of(np.array([cu]), np.array([cv]))
            d["centre_px"] = [round(float(x[0]), 2), round(float(y[0]), 2)]
            d["area_px2"] = round(shape_area(s) * field_px, 2)
            if um_per_px:
                d["centre_um"] = [round(float(x[0]) * um_per_px, 2), round(float(y[0]) * um_per_px, 2)]
                d["area_um2"] = round(shape_area(s) * field_px * um2, 1)
        shapes.append(d)
    area = fraction * field_px * um2
    return {"field_fraction": round(fraction, 6), "area_um2": round(area, 1) if np.isfinite(area) else None,
            "shapes": shapes}


def fit_affine(camera_xy: np.ndarray, field_uv: np.ndarray) -> tuple[list[list[float]], float]:
    """The affine map camera -> field by least squares (at least 3 points), and its residual in camera pixels."""
    cam = np.asarray(camera_xy, float)
    uv = np.asarray(field_uv, float)
    m = np.column_stack([cam, np.ones(len(cam))])
    a, *_ = np.linalg.lstsq(m, uv, rcond=None)
    to_field = a.T.tolist()
    c = Calibration(to_field=to_field)
    back = np.column_stack(c.camera_of(uv[:, 0], uv[:, 1]))
    return to_field, float(np.sqrt(np.mean(np.sum((back - cam) ** 2, axis=1))))


def lit_difference(on: np.ndarray, off: np.ndarray) -> float:
    """How much brighter the picture ``on`` is than ``off`` (their mean grey levels)."""
    return float(np.asarray(on, float).mean() - np.asarray(off, float).mean())


def find_spot(on: np.ndarray, off: np.ndarray) -> tuple[float, float] | None:
    """The centre (x, y) of the spot lit in ``on`` and not in ``off`` (two grey pictures of the same view): the
    weighted centre of the pixels more than half way from the background to the spot's brightest, or None when no
    spot stands clear of the noise."""
    d = np.asarray(on, float) - np.asarray(off, float)
    k = max(1, min(d.shape) // 100)                                    # a little smoothing: single pixels are noise
    if k > 1:
        c = np.cumsum(np.cumsum(np.pad(d, ((1, 0), (1, 0))), 0), 1)
        d = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
        shift = k / 2
    else:
        shift = 0.0
    background, peak = float(np.median(d)), float(d.max())
    noise = float(np.median(np.abs(d - background))) * 1.4826 + 1e-9
    if peak - background < 6 * noise or peak - background < 3:
        return None
    mask = d > background + 0.5 * (peak - background)
    ys, xs = np.nonzero(mask)
    w = d[ys, xs] - background
    return float((xs * w).sum() / w.sum() + shift), float((ys * w).sum() / w.sum() + shift)
