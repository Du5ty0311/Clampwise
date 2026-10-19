"""The Polygon's patterns: shapes on its field, and the image of mirrors that lights them.

The Polygon1000's DMD is 912 columns by 1140 rows of diamond mirrors: each row is offset by half a mirror from the
one before, and rows are half a mirror apart, so the field is 912 by 570 mirror widths (16:10), not 912 by 1140. A
shape is given in the field's own coordinates, ``u`` across and ``v`` down, each 0 to 1 (as PolyScan keeps its
regions), and lit where the centres of the mirrors fall inside it; a round spot is round on the slice.

A pattern is a list of steps, each a name and the shapes lit together: a full field, a spot, a numbered grid of
spots (1 at the top left, along the rows), a line of spots, an image file, a PolyScan sequence (``*.seq``: its
regions of interest, ROI1, ROI2...), or shapes drawn on the camera's picture.
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
    affine map from the camera's pixels (x across, y down) to the field (u, v): [u, v] = A @ [x, y, 1]."""
    on_value: int = 255
    light_tested: bool = False                                        # on_value seen to light, not assumed
    to_field: list[list[float]] = field(default_factory=list)       # 2 x 3; empty: not calibrated
    camera_size: tuple[int, int] = (0, 0)                             # (width, height) it was measured at
    residual_px: float = float("nan")

    @property
    def ready(self) -> bool:
        return len(self.to_field) == 2

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
        return Calibration(self.on_value, self.light_tested, a.tolist(), (int(w), int(h)), self.residual_px / sx)

    def to_dict(self) -> dict:
        return {"on_value": self.on_value, "light_tested": self.light_tested, "to_field": self.to_field,
                "camera_size": list(self.camera_size), "residual_px": self.residual_px}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(int(d.get("on_value", 255)), bool(d.get("light_tested", False)),
                   [list(map(float, r)) for r in d.get("to_field", [])],
                   tuple(int(x) for x in d.get("camera_size", (0, 0))), float(d.get("residual_px", float("nan"))))


CALIBRATION_SPOTS = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5))
CALIBRATION_DIAMETER = 0.06


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
