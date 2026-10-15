"""A table from a picture of a plot (B12: "any other image of a plot"). The
user clicks two known values on each axis; points are then clicked, or a
curve is traced by its colour. The result is an estimate of the plotted
numbers, as good as the picture's resolution, and is always labelled so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ESTIMATE = "digitized from a picture: an estimate, as good as the picture's resolution"


@dataclass
class Axis:
    """Two pixels with their known values; ``log`` for a logarithmic axis."""
    p1: float
    v1: float
    p2: float
    v2: float
    log: bool = False

    def value(self, px) -> np.ndarray:
        px = np.asarray(px, float)
        if self.log:
            a, b = np.log10(self.v1), np.log10(self.v2)
            return 10 ** (a + (px - self.p1) * (b - a) / (self.p2 - self.p1))
        return self.v1 + (px - self.p1) * (self.v2 - self.v1) / (self.p2 - self.p1)


@dataclass
class Calibration:
    x: Axis          # pixels are columns
    y: Axis          # pixels are rows (counted from the top)

    def to_data(self, cols, rows) -> tuple[np.ndarray, np.ndarray]:
        return self.x.value(cols), self.y.value(rows)


def load(path: str | Path) -> np.ndarray:
    """The picture as rows x columns x RGB, values 0-255."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=float)


def trace(image: np.ndarray, colour, tolerance: float = 60.0, box=None) -> tuple[np.ndarray, np.ndarray]:
    """A curve by its colour: in each column (within ``box`` = col0, row0, col1,
    row1), the middle of the pixels within ``tolerance`` (RGB distance) of
    ``colour``. Returns (columns, rows) at pixel centres."""
    img = np.asarray(image, float)[..., :3]
    c0, r0, c1, r1 = box if box is not None else (0, 0, img.shape[1], img.shape[0])
    sub = img[int(r0):int(r1), int(c0):int(c1)]
    near = np.linalg.norm(sub - np.asarray(colour, float)[None, None, :], axis=2) <= tolerance
    cols, rows = [], []
    for j in range(near.shape[1]):
        hit = np.flatnonzero(near[:, j])
        if hit.size:
            cols.append(c0 + j)
            rows.append(r0 + float(np.median(hit)))
    return np.asarray(cols, float), np.asarray(rows, float)


def table(cal: Calibration, cols, rows, x_name: str = "x", y_name: str = "y") -> pd.DataFrame:
    """The digitized points as a table, its source said to be an estimate."""
    x, y = cal.to_data(cols, rows)
    order = np.argsort(x)
    frame = pd.DataFrame({x_name: x[order], y_name: y[order]})
    frame.attrs["source"] = ESTIMATE
    return frame
