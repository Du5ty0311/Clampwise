"""Min/max decimation for drawing long traces (B18, item 3).

A 60-minute recording at 20 kHz is 72 million samples; a screen is a few
thousand pixels wide. Drawing every sample is slow and shows nothing more
than the envelope: for each pixel the minimum and the maximum of the samples
under it, which is what an oscilloscope shows. The pyramid keeps that
envelope at successive halvings, built once per channel, so any zoom level
is a slice of the right level rather than a pass over the samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def minmax(y: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """(mins, maxs) of ``y`` in blocks of ``factor`` samples (the last block may be short)."""
    y = np.asarray(y)
    n = y.size // factor * factor
    head = y[:n].reshape(-1, factor)
    mins, maxs = head.min(axis=1), head.max(axis=1)
    if y.size > n:
        tail = y[n:]
        mins = np.append(mins, tail.min())
        maxs = np.append(maxs, tail.max())
    return mins, maxs


@dataclass
class Pyramid:
    """Envelopes of one trace at factors 2, 4, 8, ... down to ``smallest`` points."""

    fs: float
    n: int
    levels: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def build(cls, y: np.ndarray, fs: float, smallest: int = 2048) -> "Pyramid":
        y = np.asarray(y, dtype=np.float64).ravel()
        out = cls(fs, y.size)
        factor = 2
        mins, maxs = y, y
        while y.size // factor >= smallest // 2 and factor <= y.size:
            # each level halves the one before: the min of two mins, the max of two maxs
            mins, maxs = minmax(mins, 2)[0], minmax(maxs, 2)[1]
            out.levels[factor] = (mins, maxs)
            factor *= 2
        return out

    def envelope(self, start_s: float, end_s: float, pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """(times, mins, maxs, factor) covering ``start_s``-``end_s`` with about
        ``pixels`` points; factor 1 means the raw samples are needed (the
        caller draws them from the source)."""
        a = max(0, int(start_s * self.fs))
        b = min(self.n, int(np.ceil(end_s * self.fs)))
        span = max(b - a, 1)
        want = max(1, span // max(pixels, 1))
        factor = 1
        for f in sorted(self.levels):
            if f <= want:
                factor = f
        if factor == 1:
            return np.array([]), np.array([]), np.array([]), 1
        mins, maxs = self.levels[factor]
        i, j = a // factor, int(np.ceil(b / factor))
        t = (np.arange(i, j) * factor + factor / 2) / self.fs
        return t, mins[i:j], maxs[i:j], factor
