"""Action potentials, found and measured as the Allen Institute's IPFX does (B21, B6).

Detection is IPFX 2.1.2's ``SpikeFeatureExtractor.process`` with its defaults
(Gouwens et al. 2019, *Nat Neurosci* 22:1182), step by step, so that a count
can be checked against IPFX sweep by sweep:

- dV/dt on a 10 kHz 4-pole Bessel-filtered copy of the trace (the raw trace when
  the sampling rate leaves no room for the filter); every voltage reported comes
  from the trace as recorded;
- onsets where dV/dt rises through 20 mV/ms inside the window; a crossing
  starts a new spike only when dV/dt fell below zero since the crossing before;
- the peak: the highest point before the next onset; where dV/dt never fell
  below zero between a peak and the next onset, the two are one spike (the
  first onset with the second peak); above -30 mV and 2 mV above the onset;
- the threshold: going back from the steepest point of the upstroke to where
  dV/dt first falls to 5% of the average upstroke; two spikes whose threshold
  and peak overlap are merged; a peak 5 ms or more after its threshold has its
  threshold, then its peak, looked for again, and the spike is dropped when
  neither is found;
- the trough, fast trough, downstroke and half-width as IPFX defines them.

One rule of the lab's own, applied afterwards (IPFX has no width limit):
narrower than 10 ms at half height from the 20 mV/ms threshold (5 ms unless the
spike is full-sized; Analysis Decisions 2026-09-26: late spikes heading for
depolarization block broaden to 6-9 ms, a 3 mV, 7.6 ms bump is no spike).

**Full or attenuated**: full is at least 20 mV from the 20 mV/ms threshold to
the peak and a peak above -20 mV. Every spike counts towards rates, as in IPFX;
rheobase and the first-spike measures come from full spikes only.

**Thresholds, both given**: IPFX's (above) and where dV/dt crosses 20 mV/ms
(the old app's headline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filters, ledger

DVDT_CUTOFF = 20.0          # mV/ms (IPFX dv_cutoff)
SLOPE_THRESHOLD = 5.0       # mV/ms: NeuroExpress's "spike slope threshold", a third threshold reported beside IPFX's
MIN_HEIGHT = 2.0            # mV (IPFX min_height)
MIN_PEAK = -30.0            # mV (IPFX min_peak)
THRESH_FRAC = 0.05          # of the average upstroke (IPFX thresh_frac)
MAX_INTERVAL_S = 0.005      # threshold to peak (IPFX max_interval)
CLIP_TOL = 1.0              # mV: the last spike must come back to its threshold + this, or it is clipped (IPFX tol)
TERM_FRAC = 0.01            # the fast trough: where the fall slows to this fraction of its steepest (IPFX term_frac)
FILTER_KHZ = 10.0           # IPFX's Bessel filter for dV/dt
FULL_AMPLITUDE = 20.0       # mV from threshold: a full-sized spike...
FULL_PEAK = -20.0           # ...with a peak above this
MAX_HALF_WIDTH_S = 0.010    # the lab's rule; 5 ms for a spike that is not full-sized
MAX_HALF_WIDTH_SMALL_S = 0.005
AHP_SEARCH_S = 0.050
CHUNK = 2_000_000           # samples per piece on long recordings
NAN = float("nan")


@dataclass(frozen=True)
class Spike:
    threshold_index: int        # where dV/dt crosses 20 mV/ms (the last sample below it)
    peak_index: int
    threshold_mv: float         # at the 20 mV/ms crossing
    peak_mv: float
    upstroke: float             # the steepest rise, mV/ms (IPFX)
    ipfx_threshold_index: int = -1
    ipfx_threshold_mv: float = NAN
    amplitude_mv: float = NAN   # peak minus the 20 mV/ms threshold
    half_width_ms: float = NAN  # IPFX's: at half the height from the fast trough
    downstroke: float = NAN     # the steepest fall from the peak to the trough, mV/ms (IPFX)
    trough_mv: float = NAN      # the lowest point after the peak, before the next spike, within 50 ms
    ahp_mv: float = NAN         # trough minus threshold
    full: bool = True
    half_width_threshold_ms: float = NAN    # at half the amplitude from the 20 mV/ms threshold (the old app's)
    fast_trough_mv: float = NAN             # IPFX's: where the fall slows to 1% of its steepest (the spike's end)
    clipped: bool = False                   # IPFX's: cut off by the window's end, no fast trough or width
    upstroke_index: int = -1                # where the rise is steepest
    downstroke_index: int = -1              # where the fall is steepest (before the trough)
    fast_trough_index: int = -1
    slope_threshold_index: int = -1         # where dV/dt crosses SLOPE_THRESHOLD (NeuroExpress's spike slope threshold)
    slope_threshold_mv: float = NAN

    @property
    def upstroke_downstroke_ratio(self) -> float:
        return self.upstroke / abs(self.downstroke) if np.isfinite(self.downstroke) and self.downstroke else NAN


def _filtered(fs: float) -> bool:
    return 0 < FILTER_KHZ / (fs / 2000.0) < 1.0           # IPFX refuses a cutoff at or above Nyquist


def dvdt(y: np.ndarray, fs: float) -> np.ndarray:
    """dV/dt in mV/ms on IPFX's 10 kHz Bessel-filtered trace (raw when there is no room)."""
    src = filters.bessel_low_pass_normalized(y, 4, FILTER_KHZ / (fs / 2000.0)) if _filtered(fs) else y
    # IPFX's own arithmetic, over a time base of sample / rate: on quantized
    # samples two upstrokes can be equal, and which is "steepest" then rests on
    # the last bit
    d = 1e-3 * np.diff(src) / np.diff(np.arange(src.size) / fs)
    return np.concatenate([d, d[-1:]]) if d.size else d


def _nearest(x_s: float, fs: float, n: int) -> int:
    """IPFX's find_time_index on an evenly sampled trace: the nearest sample, the earlier on a tie."""
    lo = int(np.floor(x_s * fs))
    i = lo if abs(lo / fs - x_s) <= abs((lo + 1) / fs - x_s) else lo + 1
    return min(max(i, 0), n - 1)


def _detect(v: np.ndarray, d: np.ndarray, fs: float, a: int, b: int):
    """IPFX's detection in the window [a, b]: (thresholds, peaks, upstrokes,
    clipped) as index arrays (detect_putative_spikes, find_peak_indexes,
    filter_putative_spikes, refine_threshold_indexes, check_thresholds_and_peaks)."""
    empty = (np.array([], int),) * 3 + (np.array([], bool),)
    w = d[a:b]
    c = np.flatnonzero(np.diff((w >= DVDT_CUTOFF).astype(np.int8)) == 1)
    if c.size > 1:
        c = np.array([c[0]] + [s for prev, s in zip(c[:-1], c[1:]) if np.any(w[prev:s] < 0)], dtype=int)
    if not c.size:
        return empty
    spk = c + a
    nxt = np.append(spk[1:], b)
    pk = np.array([int(np.argmax(v[x:y])) + x for x, y in zip(spk, nxt)])
    # a peak not followed by dV/dt below zero before the next onset: one spike
    keep = [bool(np.any(d[p:s] < 0)) for p, s in zip(pk[:-1], spk[1:])]
    pk, spk = pk[np.array(keep + [True])], spk[np.array([True] + keep)]
    ok = v[pk] >= MIN_PEAK
    spk, pk = spk[ok], pk[ok]
    ok = (v[pk] - v[spk]) >= MIN_HEIGHT
    spk, pk = spk[ok], pk[ok]
    if not spk.size:
        return empty
    up = np.array([int(np.argmax(d[x:p])) + x for x, p in zip(spk, pk)], dtype=int)
    target = float(np.mean(d[up])) * THRESH_FRAC
    th = []
    for u, prev in zip(up, np.r_[0, up[:-1]]):
        below = np.flatnonzero(d[u:prev:-1] <= target)
        th.append(int(u - below[0]) if below.size else int(prev))
    th = np.array(th, dtype=int)
    # thresholds and peaks that overlap: one spike
    over = np.flatnonzero(th[1:] <= pk[:-1] + 1)
    if over.size:
        m = np.ones(th.size, bool)
        m[over + 1] = False
        th = th[m]
        m = np.ones(pk.size, bool)
        m[over] = False
        pk, up = pk[m], up[m]
    # a peak too long after its threshold: look again for the threshold, then the peak
    too_long = [i for i in range(th.size) if pk[i] / fs - th[i] / fs >= MAX_INTERVAL_S]
    if too_long:
        target = float(np.mean(d[up])) * THRESH_FRAC
        drop = []
        for i in too_long:
            t0 = _nearest(pk[i] / fs - MAX_INTERVAL_S, fs, v.size)
            below = np.flatnonzero(d[up[i]:t0:-1] <= target)
            if below.size:
                th[i] = up[i] - below[0]
                continue
            t0 = _nearest(th[i] / fs + 2 * MAX_INTERVAL_S, fs, v.size)
            p = int(np.argmax(v[th[i]:t0])) + th[i] if t0 > th[i] else int(th[i])
            if p / fs - th[i] / fs < MAX_INTERVAL_S and (i == th.size - 1 or p < th[i + 1]):
                pk[i] = p
            else:
                drop.append(i)
        if drop:
            th, pk, up = np.delete(th, drop), np.delete(pk, drop), np.delete(up, drop)
    clipped = np.zeros(th.size, bool)
    if th.size and not np.any(v[pk[-1]:b + 1] <= v[th[-1]] + CLIP_TOL):
        clipped[-1] = True                            # never came back down before the window's end
    up = np.array([int(np.argmax(d[x:p])) + x if p > x else int(x) for x, p in zip(th, pk)], dtype=int)
    return th, pk, up, clipped


def _features(v: np.ndarray, d: np.ndarray, fs: float, th, pk, up, clipped, b: int) -> list[Spike]:
    """IPFX's trough, downstroke, fast trough and half-width (find_trough_indexes,
    find_downstroke_indexes, analyze_trough_details, find_widths), and the lab's
    measures from the 20 mV/ms threshold."""
    n = th.size
    trough = np.full(n, -1)
    for i in range(n):
        stop = th[i + 1] if i + 1 < n else b
        if not clipped[i] and stop > pk[i]:
            trough[i] = int(np.argmin(v[pk[i]:stop])) + pk[i]
    # the fast trough, IPFX's end of the spike: after the steepest fall, where the
    # fall has slowed to 1% of it; a spike where it never does is clipped
    fast = np.full(n, -1)
    clipped = clipped.copy()
    valid = np.flatnonzero(~clipped)
    for j, i in enumerate(valid):
        stop = th[valid[j + 1]] if j + 1 < valid.size else b
        if stop <= pk[i]:
            clipped[i] = True
            continue
        ds = int(np.argmin(d[pk[i]:stop])) + pk[i]
        done = np.flatnonzero(d[ds:stop] >= TERM_FRAC * d[ds])
        if done.size:
            fast[i] = ds + int(done[0])
        else:
            clipped[i] = True
    ahp_n = int(round(AHP_SEARCH_S * fs))
    out = []
    for i in range(n):
        p, t_ipfx = int(pk[i]), int(th[i])
        peak = float(v[p])
        # the 20 mV/ms threshold: the last sample below the cutoff before the upstroke,
        # looked for back to the previous spike's peak (IPFX's threshold, at 5% of the
        # average upstroke, can lie on either side of it)
        floor = int(pk[i - 1]) if i else -1
        below = np.flatnonzero(d[up[i]:floor if floor >= 0 else None:-1] < DVDT_CUTOFF)
        t20 = int(up[i] - below[0]) if below.size else t_ipfx
        below = np.flatnonzero(d[up[i]:floor if floor >= 0 else None:-1] < SLOPE_THRESHOLD)
        t5 = int(up[i] - below[0]) if below.size else t_ipfx
        thr = float(v[t20])
        amp = peak - thr
        down, ds_i = NAN, -1
        if trough[i] > p:
            ds_i = int(np.argmin(d[p:trough[i]])) + p
            down = float(d[ds_i])
        width = NAN
        ft = int(fast[i])
        if ft >= 0 and not clipped[i]:
            level = float(v[ft]) + (peak - float(v[ft])) / 2.0
            if level < v[t_ipfx]:
                level = float(v[t_ipfx]) + (peak - float(v[t_ipfx])) / 2.0
            back = np.flatnonzero(v[p:t_ipfx:-1] <= level)
            fwd = np.flatnonzero(v[p:ft] <= level)
            if back.size and fwd.size:
                width = ((p + fwd[0]) - (p - back[0])) / fs * 1e3
        # the lab's measures: the AHP trough within 50 ms, the half-width from the 20 mV/ms threshold
        nxt = int(th[i + 1]) if i + 1 < n else v.size
        end = min(v.size, nxt, p + ahp_n)
        low = float(np.min(v[p:end])) if end > p else peak
        half = thr + amp / 2.0
        ups = np.flatnonzero(v[t20:p + 1] >= half)
        downs = np.flatnonzero(v[p:end] <= half)
        width_t = (p + downs[0] - (t20 + ups[0])) / fs * 1e3 if ups.size and downs.size else NAN
        out.append(Spike(t20, p, thr, peak, float(d[up[i]]), t_ipfx, float(v[t_ipfx]), amp, width, down, low,
                         low - thr, amp >= FULL_AMPLITUDE and peak > FULL_PEAK, width_t,
                         float(v[ft]) if ft >= 0 and not clipped[i] else NAN, bool(clipped[i]), int(up[i]), ds_i,
                         ft if ft >= 0 and not clipped[i] else -1, t5, float(v[t5])))
    return out


def _lab_rule(x: Spike) -> bool:
    """The lab's width limit (IPFX has none): far too slow to be an action potential."""
    limit = (MAX_HALF_WIDTH_S if x.full else MAX_HALF_WIDTH_SMALL_S) * 1e3
    return not (np.isfinite(x.half_width_threshold_ms) and x.half_width_threshold_ms > limit)


def _find(y: np.ndarray, fs: float, a: int, b: int) -> list[Spike]:
    d = dvdt(y, fs)
    th, pk, up, clipped = _detect(y, d, fs, a, b)
    if not th.size:
        return []
    return [x for x in _features(y, d, fs, th, pk, up, clipped, b) if _lab_rule(x)]


def find(y: np.ndarray, fs: float, purpose: str = "", window: tuple[int, int] | None = None) -> list[Spike]:
    """Every action potential in ``y`` (mV), in order (see the module text).
    With ``window`` (sample indices of its start and end), as IPFX's extractor
    with a start and end: onsets inside it, peaks before its end."""
    from dataclasses import replace

    y = np.asarray(y, dtype=float)
    if y.size < 10:
        return []
    ledger.note("action potentials detected as dV/dt above 20 mV/ms" +
                (" on a 10 kHz Bessel-filtered trace" if _filtered(fs) else " (the raw trace: a 10 kHz filter is at "
                                                                         "or above Nyquist here)") +
                ", peak above -30 mV and 2 mV above its onset, repolarized before the next, threshold to peak under "
                "5 ms; narrower than 10 ms at half height (5 ms unless full-sized); threshold, trough "
                "and half-width as defined by Gouwens et al. 2019", purpose)
    a, b = window if window is not None else (0, y.size - 1)
    a, b = max(0, int(a)), min(y.size - 1, int(b))
    if b - a <= CHUNK:
        if b <= a or y[a:b + 1].max() < MIN_PEAK:
            return []
        return _find(y, fs, a, b)
    out: list[Spike] = []
    pad = int(round(0.06 * fs))
    for c in range(a, b, CHUNK):
        lo, hi = max(0, c - pad), min(y.size, c + CHUNK + pad)
        seg = y[lo:hi]
        if seg.max() < MIN_PEAK:
            continue
        for sp in _find(seg, fs, max(a, lo) - lo, min(b, hi - 1) - lo):
            g = lo + sp.peak_index
            if c <= g < c + CHUNK:
                shift = {k: lo + getattr(sp, k) for k in ("threshold_index", "ipfx_threshold_index", "upstroke_index",
                                                          "downstroke_index", "fast_trough_index",
                                                          "slope_threshold_index") if getattr(sp, k) >= 0}
                out.append(replace(sp, peak_index=g, **shift))
    return out
