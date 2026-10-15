"""Baselines, in one place (B21, item 2).

A baseline is a named window on the trace, measured one way (mean, or median
when events sit inside it), and said so. Drift removal is a straight line
fitted over a stated window and subtracted, done only when asked for and
recorded in the ledger, never silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import ledger

Method = Literal["mean", "median"]


@dataclass(frozen=True)
class Window:
    """A named stretch of time on the recording's clock."""

    name: str
    start_s: float
    end_s: float

    def indices(self, t: np.ndarray) -> slice:
        a = int(np.searchsorted(t, self.start_s, side="left"))
        b = int(np.searchsorted(t, self.end_s, side="right"))
        return slice(a, max(b, a))

    def describe(self) -> str:
        return f"{self.name} {self.start_s * 1e3:g}-{self.end_s * 1e3:g} ms"


@dataclass(frozen=True)
class Baseline:
    """A baseline level and how it was taken."""

    level: float
    sd: float
    window: Window
    method: Method
    n: int

    def describe(self) -> str:
        return f"{self.method} of the {self.window.describe()} window ({self.n} samples)"


def measure(t: np.ndarray, y: np.ndarray, window: Window, method: Method = "mean",
            purpose: str = "") -> Baseline:
    """The level of ``y`` over ``window``, with the noise SD of the same samples."""
    seg = np.asarray(y, dtype=float)[window.indices(t)]
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return Baseline(float("nan"), float("nan"), window, method, 0)
    level = float(np.median(seg)) if method == "median" else float(np.mean(seg))
    out = Baseline(level, float(np.std(seg)), window, method, int(seg.size))
    ledger.note(f"baseline: {out.describe()}", purpose)
    return out


@dataclass(frozen=True)
class Drift:
    slope_per_s: float
    intercept: float
    window: Window

    def describe(self) -> str:
        return f"straight line over {self.window.describe()} (slope {self.slope_per_s:g} per s)"


def drift_line(t: np.ndarray, y: np.ndarray, window: Window) -> Drift:
    """A straight line fitted to ``y`` over ``window`` (least squares)."""
    idx = window.indices(t)
    tt, yy = np.asarray(t, float)[idx], np.asarray(y, float)[idx]
    ok = np.isfinite(yy)
    if ok.sum() < 2:
        return Drift(float("nan"), float("nan"), window)
    slope, intercept = np.polyfit(tt[ok], yy[ok], 1)
    return Drift(float(slope), float(intercept), window)


def remove_drift(t: np.ndarray, y: np.ndarray, window: Window, purpose: str = "") -> tuple[np.ndarray, Drift]:
    """``y`` with the line fitted over ``window`` subtracted (a new array), and
    the line. Recorded in the ledger: a trace with its drift removed must say so."""
    line = drift_line(t, y, window)
    if not np.isfinite(line.slope_per_s):
        return np.array(y, dtype=float), line
    out = np.asarray(y, dtype=float) - (line.slope_per_s * np.asarray(t, float) + line.intercept)
    ledger.note(f"drift removed: {line.describe()}", purpose)
    return out, line
