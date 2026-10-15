"""Kinetics, defined in one place (B21, item 4).

How a rise time, a decay start, an onset and a half-width are found, with the
words that define them (``DEFINITIONS``), which the export definitions and the
manual read, so every column means what this file says.

The functions are the ones the evoked-response code has used since the
methods audit (they moved here unchanged; ``analysis.response`` imports them
back), so no number moves.
"""

from __future__ import annotations

import numpy as np

from . import filters

# The rule from the audit: the headline decay tau is
# fitted from 80% of the peak, because from the peak the rising phase still
# pulls the trace and reads the tau long. The from-peak fit (Clampfit's
# convention) is kept beside it.
DECAY_START_FRACTION = 0.8
# the two starts' taus differing by more than this: the decay is not one exponential
DECAY_DISAGREEMENT = 0.25

DEFINITIONS: dict[str, str] = {
    "rise_10_90": "Time for the response to go from 10% to 90% of its peak, each level taken as the "
                  "last crossing before the peak (searched backwards), so baseline noise cannot start it.",
    "rise_20_80": "As the 10-90% rise, between 20% and 80% of the peak.",
    "decay_tau": f"Time constant of a single exponential fitted from {DECAY_START_FRACTION:.0%} of the peak "
                 "(where the rise no longer pulls the trace) to the end of the response window; the "
                 "from-peak fit is reported alongside it. Flagged when the fits from "
                 f"80% and 50% of the peak differ by more than {DECAY_DISAGREEMENT:.0%} (the decay is not "
                 "one exponential), when the fit failed, or when tau sits at the fit's limit.",
    "half_width": "Width of the response at half of its peak amplitude, between the last crossing before "
                  "the peak and the first crossing after it, interpolated between samples.",
    "latency": "Time from the stimulus to the onset: the first sample where the baseline-subtracted "
               "response exceeds the onset threshold (a multiple of the baseline noise) and stays above "
               "it for the sustain time. Reported per sweep (median and jitter) and on the average.",
    "onset": "Where the response begins: the first sustained crossing of the onset threshold after the "
             "stimulus (see latency).",
}


def interp_crossing(t: np.ndarray, y: np.ndarray, i_lo: int, i_hi: int, level: float) -> float:
    """Sub-sample time at which y crosses ``level`` between two samples."""
    y0, y1 = y[i_lo], y[i_hi]
    if y1 == y0:
        return float(t[i_hi])
    frac = (level - y0) / (y1 - y0)
    return float(t[i_lo] + frac * (t[i_hi] - t[i_lo]))


def smooth(y: np.ndarray, n: int) -> np.ndarray:
    """A running mean of ``n`` samples (``same`` length)."""
    if n <= 1:
        return y
    return filters.moving_average(y, n, mode="same")


def robust_noise(x: np.ndarray) -> float:
    """MAD-based SD estimate; resists the drift and artifacts that inflate plain SD."""
    if x.size == 0:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x))))
    return mad * 1.4826


def find_onset(dev: np.ndarray, start: int, stop: int, threshold: float, sustain: int) -> int | None:
    """First index in [start, stop) where ``dev`` exceeds threshold and stays there.

    The sustain requirement is what keeps single noisy samples from being read
    as a synaptic onset, which would otherwise wreck the jitter estimate.
    """
    if stop - start < sustain or sustain < 1:
        return None
    above = dev[start:stop] > threshold
    if above.size < sustain:
        return None
    kernel = np.ones(sustain, dtype=int)
    runs = np.convolve(above.astype(int), kernel, mode="valid")
    hits = np.flatnonzero(runs == sustain)
    if hits.size == 0:
        return None
    return int(start + hits[0])


def decay_start(dev: np.ndarray, peak_idx: int, stop_idx: int, sample_rate_hz: float,
                fraction: float = DECAY_START_FRACTION) -> int:
    """Where the decay fit starts: the first sample after the peak where the
    response (``dev``, baseline-subtracted and signed so it is positive) has
    fallen to ``fraction`` of its peak, on a 0.5 ms running mean so noise
    cannot end it early. The peak itself when that leaves too few samples."""
    peak = dev[peak_idx]
    if not (np.isfinite(peak) and peak > 0) or stop_idx - peak_idx <= 8:
        return peak_idx
    n = max(1, int(round(0.5e-3 * sample_rate_hz)))
    lo = max(0, peak_idx - n)
    seg = dev[lo:stop_idx + n]
    # a running mean over the real samples only: zero padding at an edge (a
    # segment that starts at its peak) read half the peak and ended it at once
    sm = smooth(seg, n) / smooth(np.ones_like(seg), n)
    sm = sm[peak_idx - lo:stop_idx - lo]
    below = np.flatnonzero(sm <= fraction * peak)
    start = peak_idx + int(below[0]) if below.size else peak_idx
    return start if stop_idx - start >= 8 else peak_idx


def rise_time(t: np.ndarray, dev: np.ndarray, start: int, peak_idx: int,
              lo_frac: float, hi_frac: float) -> float:
    """Time for the rising phase to go from lo_frac to hi_frac of the peak.

    Each level is taken as the last crossing before the peak, searching
    backwards from it, so baseline noise that briefly touches 10% before the
    response starts cannot set the start of the rise.
    """
    peak = dev[peak_idx]
    if peak <= 0 or peak_idx - start < 2:
        return float("nan")

    def crossing(level: float) -> float:
        seg = dev[start:peak_idx + 1]
        below = np.flatnonzero(seg < level)
        if not below.size:
            return float("nan")
        i = start + int(below[-1])  # last sample under the level
        return interp_crossing(t, dev, i, i + 1, level)

    t_lo, t_hi = crossing(lo_frac * peak), crossing(hi_frac * peak)
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)):
        return float("nan")
    return max(t_hi - t_lo, 0.0)


def half_width(t: np.ndarray, dev: np.ndarray, start: int, peak_idx: int, stop: int) -> float:
    """Width at half the peak: from the last crossing before the peak to the
    first after it, both interpolated between samples."""
    peak = dev[peak_idx]
    if not (np.isfinite(peak) and peak > 0):
        return float("nan")
    half = peak / 2
    before = dev[start:peak_idx + 1]
    after = dev[peak_idx:stop]
    b = np.flatnonzero(before < half)
    a = np.flatnonzero(after < half)
    if not b.size or not a.size:
        return float("nan")
    i = start + int(b[-1])
    j = peak_idx + int(a[0])
    return interp_crossing(t, dev, j - 1, j, half) - interp_crossing(t, dev, i, i + 1, half)
