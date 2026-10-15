"""Averaging, in one place (B21, item 5).

An average says what it lined the sweeps or events up on (the stimulus, the
event onset, the peak, or the half-rise) and which ones went in, so a number
measured on it can be traced to its members. The alignment is what Clampfit
and Mini Analysis offer, by the same names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from . import ledger

Alignment = Literal["stimulus", "onset", "peak", "half_rise"]


@dataclass
class Average:
    """The averaged trace and what it is an average of."""

    time_s: np.ndarray              # relative to the alignment point
    trace: np.ndarray
    n: int
    alignment: Alignment
    members: list[int] = field(default_factory=list)      # sweep or event numbers that went in
    left_out: list[int] = field(default_factory=list)
    sd: np.ndarray | None = None

    def describe(self) -> str:
        what = {"stimulus": "the stimulus", "onset": "each event's onset", "peak": "each peak",
                "half_rise": "each half-rise"}[self.alignment]
        return f"average of {self.n} aligned on {what}"


def _half_rise_index(seg: np.ndarray, direction: int) -> int:
    dev = direction * (seg - seg[0])
    peak = int(np.argmax(dev))
    if peak == 0 or dev[peak] <= 0:
        return 0
    below = np.flatnonzero(dev[:peak + 1] < dev[peak] / 2)
    return int(below[-1]) + 1 if below.size else 0


def align(sweeps: np.ndarray, fs: float, anchor_samples: Sequence[int], before_s: float, after_s: float,
          *, alignment: Alignment = "stimulus", direction: int = 1,
          include: Sequence[bool] | None = None, purpose: str = "") -> Average:
    """Line ``sweeps`` (rows) up on ``anchor_samples`` (one per row) and average
    the window ``before_s`` to ``after_s`` around it.

    For ``peak`` and ``half_rise`` the anchor is refined within the window on
    each row (the largest deflection in ``direction``, or the point where it
    first passes half of it). Rows without the full window are left out and
    listed, never padded.
    """
    sweeps = np.atleast_2d(np.asarray(sweeps, dtype=float))
    before, after = int(round(before_s * fs)), int(round(after_s * fs))
    members, left_out, segs = [], [], []
    for i, row in enumerate(sweeps):
        if include is not None and not include[i]:
            left_out.append(i)
            continue
        a = int(anchor_samples[i])
        if alignment in ("peak", "half_rise"):
            lo, hi = max(a - before, 0), min(a + after, row.size)
            seg = row[lo:hi]
            if seg.size == 0:
                left_out.append(i)
                continue
            if alignment == "peak":
                a = lo + int(np.argmax(direction * (seg - seg[0])))
            else:
                a = lo + _half_rise_index(seg, direction)
        if a - before < 0 or a + after > row.size:
            left_out.append(i)
            continue
        segs.append(row[a - before:a + after])
        members.append(i)
    if not segs:
        return Average(np.arange(-before, after) / fs, np.full(before + after, np.nan), 0, alignment, [], left_out)
    stack = np.vstack(segs)
    out = Average(np.arange(-before, after) / fs, stack.mean(axis=0), len(segs), alignment, members, left_out,
                  stack.std(axis=0) if len(segs) > 1 else None)
    ledger.note(out.describe(), purpose)
    return out


def plain(sweeps: np.ndarray, include: Sequence[bool] | None = None, purpose: str = "") -> Average:
    """The sweep average as the window and the trains analysis take it: every
    included sweep on the shared sweep clock (aligned on the stimulus, which
    sits at the same time in every sweep)."""
    sweeps = np.atleast_2d(np.asarray(sweeps, dtype=float))
    rows = [i for i in range(sweeps.shape[0]) if include is None or include[i]]
    left_out = [i for i in range(sweeps.shape[0]) if i not in rows]
    if not rows:
        return Average(np.arange(sweeps.shape[1]), np.full(sweeps.shape[1], np.nan), 0, "stimulus", [], left_out)
    stack = sweeps[rows]
    out = Average(np.arange(sweeps.shape[1]), stack.mean(axis=0), len(rows), "stimulus", rows, left_out,
                  stack.std(axis=0) if len(rows) > 1 else None)
    ledger.note(out.describe(), purpose)
    return out
