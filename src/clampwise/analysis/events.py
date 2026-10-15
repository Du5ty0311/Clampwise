"""Spontaneous and miniature synaptic events (B4): found, measured, reviewed,
and their timing described.

Detection, on the trace with its slow baseline removed (each point minus the
running median of the blocks around it, so drift and holding changes are not
events), turned so the events point up:

- **Deconvolution** (Pernia-Andrade et al. 2012, *Biophys J* 103:1429; the
  default): the trace is deconvolved by the template in the frequency domain,
  Wiener-regularized so high-frequency noise is not amplified, band-passed
  from 1 Hz to a top that follows the template (100 / rise (ms) Hz, never
  below 200 Hz: exactly 200 Hz at the default 0.5 ms rise; Analysis
  Decisions, 2026-09-26), and an event is a peak above the threshold (4.5)
  times the robust SD of the deconvolved trace.
- **Template matching** (Clements & Bekkers 1997, *Biophys J* 73:220): the
  template is scaled and offset to fit the trace at every point by least
  squares; the criterion is the scale over its standard error, and an event
  is a local maximum above the threshold (3).

The template is (1 - exp(-t/rise)) exp(-t/decay), default rise 0.5 / decay 5
ms; or a Clampfit template file; or the experiment's own, built from its
recordings' averaged events and applied identically to every recording
(a template refined per recording would let
detection differ between the conditions compared, a confound). There is no
per-recording refinement for that reason.

Each event (Analysis Decisions, 2026-09-29): amplitude against the median of
the 3 to 0.5 ms before its onset, at the first maximum the smoothed event
clearly falls from, as a short mean there (the raw maximum is the largest
noise excursion on it); 10-90% rise on a 0.2 ms running mean, 10% searched
back from the peak, 90% forward from there; decay tau from 80% of the peak
over six of its own taus (variable projection), not fitted when the next
event comes within three taus, its r2 over the first 1.5 taus, fits below
0.8 left out of the median; charge to the end of that window; half-width;
the interval from the previous accepted event, never across a left-out
stretch.

The recording's numbers: the mean amplitude, charge and interval, the median
beside each (decision D2, 2026-09-30: Mini Analysis and Clampfit report the
mean); kinetics from the averaged event (D3), aligned on each event's rise
midpoint, with the per-event medians beside them. Events from one cell are
not independent: cells are compared by their per-cell numbers, not pooled
events (the event table says so).

Timing: the inter-event intervals' mean, median and CV; their local variation
LV (Shinomoto et al. 2003, *Neural Comput* 15:2823), which a slow
change of rate does not move; and whether the events look like a random
(Poisson) process at the rate they have in each stretch of time: the
intervals are rescaled by that rate (the time-rescaling theorem, Brown et al.
2002, *Neural Comput* 14:325; Kass, Eden & Brown, *Analysis of Neural Data*)
and compared with an exponential by a Kolmogorov-Smirnov test whose null is
simulated with the same rates and the detector's dead time. A random process
has a CV of 1 (Dayan & Abbott, *Theoretical Neuroscience*, ch. 1); bursts make
the rescaled intervals more variable, regular firing less. So a change of
frequency can be told from a change of burstiness.

Left out before detection, bridged by a straight line so the filters never
see them: stimuli (2 ms before to 100 ms after), DSI steps with the current's
settling after them, and in current clamp action potentials (3 ms before
threshold to 25 ms after the peak; Analysis Decisions, 2026-09-26).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter1d
from scipy.signal import fftconvolve

from ..core.recording import Recording
from .registry import Output, Setting, register
from .result import Drawable, Flag, Result
from .tools import filters, fits, kinetics, ledger, spikes

NAN = float("nan")
METHODS = ("deconvolution", "template matching")
DIRECTIONS = ("auto", "down", "up")
UPWARD_FROM_MV = -30.0          # voltage clamp held at or above this: events read outward (IPSCs at 0 to +20 mV)
PEAK_SEARCH_MS = 20.0
PEAK_PASSED = 0.7               # the peak is past once the smoothed event falls below this of its running maximum
DECAY_TAUS = 6.0
DECAY_MAX_MS = 300.0
RISE_SMOOTH_MS = 0.2
SLOW_EVENTS = 2.0               # an averaged decay this many times the template's: events may be split
CHUNK_S = 60.0                  # long recordings are detected in pieces of this length
CHARGE_BASELINE_S = 0.5
CHARGE_BLOCK_S = 0.01           # charge kept in 10 ms blocks for the cell's analyses
MIN_STRETCH_S = 1.0             # shorter stretches at a sweep's ends, cut off by a left-out part, are not searched
DSI_MIN_STEP_S = 0.5
SETTLE_MIN_S = 0.2
SETTLE_MAX_S = 2.0
EVOKED_EVENTS_MIN_S = 3.0
MONTE_CARLO = 200
MIN_INTERVALS = 30
SPONTANEOUS = frozenset({"vc_spontaneous", "cc_spontaneous"})


# ------------------------------------------------------------------ templates

def template_shape(fs: float, rise_ms: float, decay_ms: float) -> np.ndarray:
    """The analytic template, upward, peak 1, five decay constants long."""
    n = max(8, int(round(5 * decay_ms * 1e-3 * fs)))
    t = np.arange(n) / fs
    y = (1 - np.exp(-t / (rise_ms * 1e-3))) * np.exp(-t / (decay_ms * 1e-3))
    return y / y.max()


def read_atf(path: str | Path, fs: float) -> np.ndarray | None:
    """A Clampfit template (.atf: time in ms, then the waveform) at ``fs``,
    upward with its peak at 1, its pre-event baseline taken off and the flat
    lead trimmed to where the event starts; None when it cannot be read."""
    try:
        lines = Path(path).read_text(encoding="latin-1").splitlines()
        n_header = int(lines[1].split()[0])
        rows = [ln.split("\t") for ln in lines[3 + n_header:] if ln.strip()]
        data = np.array([[float(v) for v in r[:2]] for r in rows], dtype=float)
    except (OSError, ValueError, IndexError):
        return None
    if data.ndim != 2 or data.shape[0] < 10:
        return None
    t_ms, y = data[:, 0], data[:, 1]
    y = y - np.median(y[:max(3, y.size // 10)])
    peak = int(np.argmax(np.abs(y)))
    if y[peak] == 0:
        return None
    y = y / y[peak]                      # upward, whatever its sign in the file
    lead = np.flatnonzero(y[:peak + 1] >= 0.1)
    first = max(0, int(lead[0]) - 1) if lead.size else 0
    grid = np.arange(t_ms[first], t_ms[-1], 1e3 / fs)
    return np.interp(grid, t_ms[first:], y[first:])


def shape_kinetics(y: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    """(10-90% rise ms, decay tau ms from 80% of the peak, its r2, rise tau ms
    = 10-90% / ln 9) of an upward waveform."""
    y = np.asarray(y, float)
    peak = int(np.argmax(y))
    t = np.arange(y.size) / fs
    if y[peak] <= 0:
        return NAN, NAN, NAN, NAN
    rise = kinetics.rise_time(t, y, 0, peak, 0.1, 0.9) * 1e3
    start = kinetics.decay_start(y, peak, y.size, fs)
    fit = fits.exp_decay(t[start:], y[start:], purpose="the averaged event's decay")
    tau = fit["tau"] * 1e3 if fit.ok else NAN
    r2 = fit.r2 if fit.ok else NAN
    return float(rise), float(tau), float(r2), (float(rise / np.log(9)) if np.isfinite(rise) and rise > 0 else NAN)


# ------------------------------------------------------------------ detection

def remove_slow(y: np.ndarray, fs: float, block_ms: float) -> np.ndarray:
    """``y`` minus the running median of ``block_ms`` blocks, interpolated
    between block centres (quick on an hour of data)."""
    block = max(3, int(round(block_ms * 1e-3 * fs)))
    n = y.size // block
    if n < 3:
        return y - np.median(y)
    meds = np.empty(n)
    step = max(1, 1_000_000 // block)
    for a in range(0, n, step):
        b = min(n, a + step)
        meds[a:b] = np.median(y[a * block:b * block].reshape(b - a, block), axis=1)
    centres = (np.arange(n) + 0.5) * block
    return y - np.interp(np.arange(y.size), centres, meds)


def criterion(y: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """Clements & Bekkers' detection criterion (scale / its standard error)
    with the template starting at every point; zero where it cannot start."""
    n = tpl.size
    out = np.zeros(y.size)
    if y.size < n + 2:
        return out
    s_t, s_tt = tpl.sum(), (tpl ** 2).sum()
    c = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    s_y, s_yy = c[n:] - c[:-n], c2[n:] - c2[:-n]
    s_ty = fftconvolve(y, tpl[::-1], mode="valid")
    scale = (s_ty - s_t * s_y / n) / (s_tt - s_t * s_t / n)
    offset = (s_y - scale * s_t) / n
    sse = s_yy + scale ** 2 * s_tt + n * offset ** 2 - 2 * (scale * s_ty + offset * s_y - scale * offset * s_t)
    out[:s_y.size] = scale / np.sqrt(np.maximum(sse, 1e-30) / (n - 1))
    return out


def band_top_hz(rise_ms: float, set_hz: float = 0.0) -> float:
    """The deconvolved trace's band-pass top: as set, or 100 / rise (ms) Hz
    and never below 200 Hz (a deconvolved event is about as wide as its rise:
    a 200 Hz ceiling smears a 0.2 ms rise away)."""
    return float(set_hz) if set_hz > 0 else max(200.0, 100.0 / max(rise_ms, 1e-3))


def deconvolved(y: np.ndarray, tpl: np.ndarray, fs: float, top_hz: float) -> np.ndarray:
    """The deconvolved, band-passed trace (not yet in SD units), computed in
    pieces with margins so an hour costs no more memory than a minute."""
    out = np.zeros(y.size, dtype=np.float32)
    if y.size < 100:
        return out
    band = filters.band_pass(1.0, min(top_hz, fs / 2 * 0.9))
    piece = int(CHUNK_S * fs)
    margin = max(int(2.0 * fs), 40 * tpl.size)
    for a in range(0, y.size, piece):
        lo, hi_i = max(0, a - margin), min(y.size, a + piece + margin)
        # at the recording's ends the margin is the recording mirrored, so the 1 Hz
        # high-pass starts and ends on a continuous trace: no start-up transient on
        # the first events (filtering from the bare edge made 10-13 false events in
        # the first 80 ms) and no change of the noise near the ends, which sets the
        # threshold (zeros there read the SD 8% low)
        left, right = margin - (a - lo), margin - (hi_i - min(y.size, a + piece))
        seg = np.asarray(y[lo:hi_i], float)
        left, right = min(left, seg.size - 1), min(right, seg.size - 1)     # a mirror is at most the data
        if left or right:
            seg = np.pad(seg, (left, right), mode="reflect")
        n = next_fast_len(seg.size + 41 * tpl.size, real=True)
        spectrum = np.fft.rfft(tpl, n)
        power = np.abs(spectrum) ** 2
        d = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(spectrum) / (power + 0.01 * power.max()), n)[:seg.size]
        d = filters.apply(d, fs, band, purpose="event detection") if d.size > 30 else d
        start = left + (a - lo)
        out[a:min(a + piece, y.size)] = d[start:start + min(piece, y.size - a)]
    return out


def find_events(dev: np.ndarray, fs: float, tpl: np.ndarray, s: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Onset indices and scores in an upward, baseline-free trace, and how many
    events the same detection finds pointing the other way (both scores are
    linear in the trace, so the other direction's is the same score negated)."""
    if s["method"] == "deconvolution":
        d = deconvolved(dev, tpl, fs, band_top_hz(s["template_rise_ms"], s["band_top_hz"]))
        # the robust SD over the whole trace, bridged spikes included, as the old app
        # (and its checks against miniML and injected events). Taking it from the
        # searched stretches only raised the threshold on a firing cell (17819003:
        # 42% fewer events) with no ground truth to say which is right (2026-10-05)
        med = float(np.median(d))
        sd = 1.4826 * float(np.median(np.abs(d - med)))
        score = (d - med) / sd if sd > 0 else np.zeros_like(d)
        threshold = s["deconvolution_threshold"]
    else:
        score = criterion(dev, tpl)
        threshold = s["template_threshold"]
    if score.size == 0:
        return np.array([], int), np.array([]), 0
    gap = max(1, int(round(s["min_interval_ms"] * 1e-3 * fs)))
    if s["method"] == "deconvolution":
        flipped = -score
    else:
        flipped = criterion(-np.asarray(dev, float), tpl)
    n_other = int(((flipped >= threshold) & (flipped == maximum_filter1d(flipped, size=2 * gap + 1))).sum())
    peaks = np.flatnonzero((score >= threshold) & (score == maximum_filter1d(score, size=2 * gap + 1)))
    scores = score[peaks].astype(float)
    if s["method"] == "deconvolution" and peaks.size:
        # the deconvolved peak sits on the onset; step back to where the rise starts
        peaks = np.maximum(peaks - max(1, int(round(0.3 * s["template_rise_ms"] * 1e-3 * fs))), 0)
    keep: list[int] = []
    for j, i in enumerate(peaks):
        if not keep or i - peaks[keep[-1]] >= gap:
            keep.append(j)
    return peaks[keep].astype(int), scores[keep], n_other


# ------------------------------------------------------------------ one event

@dataclass
class Event:
    sweep: int
    onset_s: float                      # on the sweep's own clock (gap-free: the file's)
    peak_s: float = NAN
    amplitude: float = NAN              # signed, in the signal's units
    rise_10_90_ms: float = NAN
    decay_tau_ms: float = NAN
    decay_r2: float = NAN
    charge: float = NAN                 # pC (voltage clamp) or mV*ms (current clamp)
    half_width_ms: float = NAN
    iei_ms: float = NAN
    score: float = NAN
    mid_s: float = NAN                  # the rise's 50% point: what the averaged event is aligned on
    accepted: bool = True
    manual: bool = False
    end_s: float = NAN                  # where its measurement window ends
    why_not: str = ""                   # the acceptance criterion it failed (NeuroExpress's), when it did
    baseline: float = NAN               # the trace's level just before it (3 to 0.5 ms), in its units
    rise_tau_ms: float = NAN            # a fit of the whole event (when asked): its rise and decay taus
    decay_tau_whole_ms: float = NAN


def noise_sd(dev: np.ndarray) -> float:
    """The robust SD of a baseline-free trace (1.4826 x the median absolute deviation)."""
    x = dev[np.isfinite(dev)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else NAN


def acceptance(e: "Event", noise_pp: float, s: dict) -> str:
    """NeuroExpress's criteria, as its mEPSC module sets them (the more stringent
    ones): an event at least the noise's peak-to-peak, rising no slower
    than the maximum rise, decaying no slower than the maximum decay, inside the
    amplitude window when one is set. The reason it fails, or ""."""
    if e.manual:
        return ""
    a = abs(e.amplitude)
    if np.isfinite(noise_pp) and a < noise_pp:
        return f"below the noise ({a:.3g} < {noise_pp:.3g}, its peak-to-peak)"
    if np.isfinite(e.rise_10_90_ms) and e.rise_10_90_ms > s["max_rise_ms"]:
        return f"rises too slowly ({e.rise_10_90_ms:.2f} ms > {s['max_rise_ms']:g} ms, 10-90%)"
    if np.isfinite(e.decay_tau_ms) and np.isfinite(e.decay_r2) and e.decay_r2 >= s["min_decay_r2"] \
            and e.decay_tau_ms > s["max_decay_ms"]:
        return f"decays too slowly (tau {e.decay_tau_ms:.1f} ms > {s['max_decay_ms']:g} ms)"
    if s["amplitude_window"] and not (s["amplitude_min"] <= a <= s["amplitude_max"]):
        return f"outside the amplitude window ({s['amplitude_min']:g} to {s['amplitude_max']:g})"
    return ""


def _running_mean(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.astype(float)
    num = np.convolve(x, np.ones(k), mode="same")
    den = np.convolve(np.ones(x.size), np.ones(k), mode="same")
    return num / den


def measure_event(dev: np.ndarray, i: int, nxt: int | None, fs: float, decay_hint_ms: float) -> dict:
    """Amplitude, kinetics and charge of the upward event starting at ``i`` in
    ``dev`` (upward, baseline-free), the next event starting at ``nxt``. Works
    on the stretch around the event only. Indices returned are into ``dev``."""
    out = {"peak": -1, "amplitude": NAN, "rise": NAN, "tau": NAN, "r2": NAN, "charge": NAN, "half": NAN,
           "mid": NAN, "end": -1}
    ms = lambda v: int(round(v * 1e-3 * fs))          # noqa: E731
    lo = max(0, i - ms(3.0) - 1)
    hi = min(dev.size, i + ms(PEAK_SEARCH_MS + DECAY_MAX_MS) + 4)
    if nxt is not None:
        hi = min(hi, nxt + 1)
    d = np.asarray(dev[lo:hi], float)
    i0 = i - lo
    nx = None if nxt is None else nxt - lo
    b0, b1 = max(0, i0 - ms(3.0)), max(0, i0 - ms(0.5))
    base = float(np.median(d[b0:b1])) if b1 - b0 >= 2 else float(d[i0])
    d = d - base
    limit = d.size if nx is None else min(d.size, nx)
    search_end = min(limit, i0 + ms(PEAK_SEARCH_MS))
    if search_end - i0 < 2:
        return out
    # the peak: on a smoothed trace, the first maximum it clearly falls from,
    # counted once it stands above the baseline's noise; measured as a short raw mean
    k = max(1, ms(min(max(0.2 * decay_hint_ms + 0.2, 0.4), 1.2)))
    m = max(1, ms(min(max(0.06 * decay_hint_ms, 0.15), 0.5)))
    sm = _running_mean(d[i0:min(d.size, search_end + k)], k)[:search_end - i0]
    pre = d[b0:b1]
    noise = kinetics.robust_noise(pre) / math.sqrt(k) if pre.size >= 5 else 0.0
    run = np.maximum.accumulate(sm)
    past = np.flatnonzero((sm < PEAK_PASSED * run) & (run > 3.0 * noise))
    stop = int(past[0]) if past.size else sm.size
    peak = i0 + int(np.argmax(sm[:max(stop, 1)]))
    amp = float(np.mean(d[max(i0, peak - m // 2):peak + m // 2 + 1]))
    if not amp > 0:
        return out
    out["peak"], out["amplitude"] = lo + peak, amp
    t = np.arange(d.size) / fs
    # the rise, on a 0.2 ms running mean: 10% the last sample below it before the
    # peak (searched back, so noise before the event cannot set it), 90% and 50%
    # the first crossings after that
    kr = max(1, ms(RISE_SMOOTH_MS))
    r = _running_mean(d[b0:min(d.size, peak + 1 + kr)], kr)[:peak + 1 - b0]
    if r.size >= 2:
        below = np.flatnonzero(r < 0.1 * amp)
        j10 = int(below[-1]) if below.size else 0
        if j10 + 1 < r.size:
            def at(j: int, level: float) -> float:
                y0, y1 = r[j], r[j + 1]
                return float(t[b0 + j] + ((level * amp - y0) / (y1 - y0) if y1 != y0 else 1.0) / fs)

            def forward(level: float) -> float:
                up = np.flatnonzero(r[j10 + 1:] >= level * amp)
                return at(j10 + int(up[0]), level) if up.size else float(t[peak])

            t10 = at(j10, 0.1)
            out["rise"] = (forward(0.9) - t10) * 1e3
            out["mid"] = lo + forward(0.5) * fs
    # the decay: from 80% of the peak over six of its own (half-decay) taus
    after = _running_mean(d[peak:limit], k) if limit - peak > 1 else np.array([])
    half = np.flatnonzero(after <= amp / 2)
    tau_est = (half[0] / fs / np.log(2)) if half.size and half[0] > 0 else decay_hint_ms * 1e-3
    end = min(limit, peak + int(round(min(DECAY_TAUS * tau_est, DECAY_MAX_MS * 1e-3) * fs)))
    crowded = nx is not None and nx < peak + int(round(3 * tau_est * fs))
    if not crowded and end - peak > 5:
        start = kinetics.decay_start(d, peak, end, fs)
        fit = fits.exp_decay_profiled(t[start:end], d[start:end], purpose="each event's decay tau")
        if fit.ok and fit["tau"] > 0:
            out["tau"] = fit["tau"] * 1e3
            n = max(5, min(end - start, int(round(1.5 * fit["tau"] * fs))))
            yy = _running_mean(d[start:min(d.size, start + n + kr)], kr)[:n]
            tt = t[start:start + yy.size] - t[start]
            pred = fit["amp"] * np.exp(-tt / fit["tau"]) + fit["offset"]
            ss = float(np.sum((yy - yy.mean()) ** 2))
            out["r2"] = 1.0 - float(np.sum((yy - pred) ** 2)) / ss if ss > 0 else NAN
    if end > i0 + 1:
        out["charge"] = float(np.trapezoid(d[i0:end], dx=1.0 / fs))
        out["end"] = lo + end
    out["half"] = kinetics.half_width(t, d, max(b0, 0), peak, max(end, peak + 2)) * 1e3
    return out


# ------------------------------------------------------------------ what is left out

def _merge(spans: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for a, b, why in sorted(spans):
        if out and a <= out[-1][1]:
            pa, pb, pw = out[-1]
            out[-1] = (pa, max(pb, b), pw if why in pw else f"{pw}; {why}")
        else:
            out.append((a, b, why))
    return out


def dsi_steps(recording: Recording) -> list[tuple[int, float, float, float]]:
    """(sweep, start s, end s, level mV) of each DSI step: in voltage clamp, from
    a holding at or below -40 mV, a step of 0.5 s or longer to -20 mV or above
    (the rule that types a recording as DSI)."""
    if not recording.is_voltage_clamp:
        return []
    hold = recording.holding
    out = []
    for k, sweep in enumerate(recording.epochs):
        base = hold if np.isfinite(hold) else (sweep[0].level if sweep else NAN)
        if not (np.isfinite(base) and base <= -40.0):
            continue
        for e in sweep:
            if e.kind == "step" and e.end_s - e.start_s >= DSI_MIN_STEP_S and e.level >= -20.0:
                out.append((k, e.start_s, e.end_s, e.level))
    return out


def settle_s(y: np.ndarray, fs: float, start_s: float, end_s: float) -> float:
    """How long after a step the current takes to come back: the first 20 ms
    block, from 0.2 s on, whose median is within one noise SD of where the
    current sits 1-3 s later (at most 2 s). Within three SDs let a 7 pA tail
    into the window after the step (Analysis Decisions, 2026-09-29)."""
    i_end = int(end_s * fs)
    r0, r1 = int((end_s + 1.0) * fs), int((end_s + 3.0) * fs)
    if r1 > y.size or r1 - r0 < int(0.2 * fs):
        return SETTLE_MIN_S
    ref = float(np.median(y[r0:r1]))
    pre = y[max(0, int((start_s - 1.0) * fs)):int(start_s * fs)]
    d = np.diff(pre) if pre.size > 10 else np.diff(y[r0:r1])
    noise = kinetics.robust_noise(d) / math.sqrt(2)
    k = max(1, int(round(0.02 * fs)))
    seg = y[i_end:min(y.size, i_end + int(SETTLE_MAX_S * fs) + k)]
    n = seg.size // k
    if n == 0:
        return SETTLE_MIN_S
    meds = np.median(seg[:n * k].reshape(n, k), axis=1)
    ends = (np.arange(n) + 1) * k / fs
    ok = np.flatnonzero((np.abs(meds - ref) <= max(noise, 1e-9)) & (ends >= SETTLE_MIN_S))
    return float(ends[ok[0]]) if ok.size else SETTLE_MAX_S


def parse_stretch(text: str) -> tuple[float, float]:
    """'30-90' -> (30, 90) seconds; '30-' -> (30, inf); '' -> (0, inf)."""
    text = (text or "").strip()
    if not text:
        return 0.0, math.inf
    a, _, b = text.partition("-")
    lo = float(a) if a.strip() else 0.0
    hi = float(b) if b.strip() else math.inf
    if hi <= lo:
        raise ValueError(f"the stretch {text!r} ends before it starts")
    return lo, hi


def parse_stretches(text: str) -> list[tuple[float, float]]:
    """'10-12, 30-31.5' -> [(10, 12), (30, 31.5)] seconds."""
    out = []
    for piece in (text or "").replace(";", ",").split(","):
        if piece.strip():
            a, b = parse_stretch(piece)
            out.append((a, b))
    return out


def parse_regions(text: str) -> list[tuple[str, float, float]]:
    """'baseline: 0-60; drug: 60-180' -> [('baseline', 0, 60), ('drug', 60, 180)]."""
    out = []
    for j, piece in enumerate((text or "").split(";")):
        if not piece.strip():
            continue
        name, _, span = piece.rpartition(":")
        a, b = parse_stretch(span)
        out.append(((name.strip() or f"region {j + 1}"), a, b))
    return out


def parse_edits(text: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """(rejected, added) events, each (sweep, onset s), from their saved form."""
    if not text:
        return [], []
    try:
        data = json.loads(text)
    except ValueError:
        return [], []
    rej = [(int(s), float(t)) for s, t in data.get("rejected", [])]
    add = [(int(s), float(t)) for s, t in data.get("added", [])]
    return rej, add


def format_edits(rejected, added) -> str:
    if not rejected and not added:
        return ""
    return json.dumps({"rejected": sorted([int(s), round(float(t), 5)] for s, t in rejected),
                       "added": sorted([int(s), round(float(t), 5)] for s, t in added)})


REJECT_TOLERANCE_S = 0.001      # a rejection holds for the detected onset within 1 ms (a setting moved it a sample)


# ------------------------------------------------------------------ timing

def local_variation(ieis_by_run: list[np.ndarray]) -> float:
    """LV: 3/(n-1) sum ((I_i - I_i+1)/(I_i + I_i+1))^2 over consecutive
    intervals (1 for a Poisson process, below for regular, above for bursty)."""
    terms = []
    for run in ieis_by_run:
        if run.size >= 2:
            a, b = run[:-1], run[1:]
            terms.append(((a - b) / (a + b)) ** 2)
    if not terms:
        return NAN
    x = np.concatenate(terms)
    return float(3.0 * x.mean()) if x.size >= 2 else NAN


def _rescaled(times_by_seg: list[np.ndarray], segs: list[tuple[float, float]], bin_s: float,
              rates: list[np.ndarray] | None = None) -> np.ndarray:
    """Intervals rescaled by the rate in each bin of each segment: z = the
    integral of the rate between consecutive events, exponential(1) for a
    Poisson process at that rate."""
    out = []
    for k, ((a, b), times) in enumerate(zip(segs, times_by_seg)):
        if times.size < 2:
            continue
        edges = np.arange(a, b, bin_s)
        edges = np.append(edges, b)
        if rates is None:
            counts = np.histogram(times, edges)[0]
            lam = counts / np.maximum(np.diff(edges), 1e-12)
        else:
            lam = rates[k]
        cum = np.concatenate([[0.0], np.cumsum(lam * np.diff(edges))])
        big = np.interp(times, edges, cum)
        out.append(np.diff(big))
    return np.concatenate(out) if out else np.array([])


def _ks_exponential(z: np.ndarray) -> float:
    z = np.sort(z[np.isfinite(z) & (z >= 0)])
    if z.size == 0:
        return NAN
    f = 1.0 - np.exp(-z)
    n = z.size
    return float(max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n)))


def timing_test(times_by_seg: list[np.ndarray], segs: list[tuple[float, float]], bin_s: float,
                dead_s: float, seed: int = 0) -> dict:
    """The time-rescaled intervals against an exponential: KS distance, its p
    from a simulated null (an inhomogeneous Poisson process at the same rates
    in the same bins, through the same dead time), and the rescaled CV."""
    z = _rescaled(times_by_seg, segs, bin_s)
    out = {"p": NAN, "d": NAN, "cv_rescaled": NAN, "n": int(z.size)}
    if z.size < MIN_INTERVALS:
        return out
    d_obs = _ks_exponential(z)
    out["d"] = d_obs
    out["cv_rescaled"] = float(np.std(z) / np.mean(z)) if np.mean(z) > 0 else NAN
    rng = np.random.default_rng(seed)
    rate_list = []
    for (a, b), times in zip(segs, times_by_seg):
        edges = np.append(np.arange(a, b, bin_s), b)
        rate_list.append((edges, np.histogram(times, edges)[0] / np.maximum(np.diff(edges), 1e-12)))
    exceed = 0
    for _ in range(MONTE_CARLO):
        sim_times = []
        for edges, lam in rate_list:
            ts = []
            for j in range(lam.size):
                n = rng.poisson(lam[j] * (edges[j + 1] - edges[j]))
                ts.append(rng.uniform(edges[j], edges[j + 1], n))
            ts = np.sort(np.concatenate(ts)) if ts else np.array([])
            if dead_s > 0 and ts.size > 1:
                ts = ts[np.r_[True, np.diff(ts) >= dead_s]]
            sim_times.append(ts)
        zs = _rescaled(sim_times, segs, bin_s)
        if zs.size >= 2 and _ks_exponential(zs) >= d_obs:
            exceed += 1
    out["p"] = (exceed + 1) / (MONTE_CARLO + 1)
    return out


def timing_words(test: dict, n_intervals: int) -> str:
    if n_intervals < MIN_INTERVALS or not np.isfinite(test.get("p", NAN)):
        return f"too few intervals to tell (fewer than {MIN_INTERVALS})"
    if test["p"] >= 0.05:
        return "random (Poisson-like) at the rate it has"
    return "bursty" if test["cv_rescaled"] > 1 else "more regular than random"


# ------------------------------------------------------------------ the declaration

SETTINGS = (
    Setting("method", "deconvolution", "Detection method", "Deconvolution resolves overlapping events; template "
            "matching does not. On simulated 10 ms events, template matching detects 95% at 2 Hz, 83% at 5 Hz and 78% "
            "at 9 Hz; deconvolution detects 95% at all three rates.", choices=METHODS,
            source="Pernia-Andrade et al. 2012; Clements & Bekkers 1997"),
    Setting("direction", "auto", "Events point", "Auto: upward in current clamp; in voltage clamp downward, or upward "
            "when the cell is held at -30 mV or above (outward IPSCs).", choices=DIRECTIONS,
            source=""),
    Setting("template_rise_ms", 0.5, "Template rise tau (ms)", "The template's rise time constant. Use one template for "
            "the whole experiment, set from the experiment's averaged event.", "ms",
            "", minimum=0.02, maximum=50.0),
    Setting("template_decay_ms", 5.0, "Template decay tau (ms)", "The template's decay time constant.", "ms",
            "", minimum=0.2, maximum=500.0),
    Setting("deconvolution_threshold", 4.5, "Deconvolution threshold (SD)", "Events are peaks of the deconvolved "
            "trace exceeding this many robust SDs.", "SD", "Pernia-Andrade et al. 2012", minimum=1.0),
    Setting("template_threshold", 3.0, "Template-matching threshold", "Events are detected where the template's "
            "scale factor exceeds this many standard errors.", "", "Clements & Bekkers 1997", minimum=1.0),
    Setting("timecourse_bin_s", 10.0, "Events over time: bins (s)", "Frequency, amplitude and charge are reported in bins of "
            "this length; the timing test also estimates the rate over these bins.", "s", minimum=0.5),
    Setting("template_origin", "", "Where the template came from", "Recorded when the template is built from the "
            "experiment's averaged event; shown in Results and the export.", advanced=True),
    Setting("template_file", "", "Template file (.atf)", "Use a template saved as an Axon text file instead of "
            "the rise and decay above.", advanced=True),
    Setting("band_top_hz", 0.0, "Deconvolution band-pass top (Hz)", "0: 100 / rise (ms) Hz, with a minimum of 200 Hz "
            "(200 Hz at a 0.5 ms rise). A higher cut-off (e.g. 318 Hz) detects about 6% more small events at the cost "
            "of about one additional false event per minute.", "Hz", "", advanced=True, minimum=0.0),
    Setting("nex_criteria", False, "Acceptance criteria", "When on, an event is accepted only if its amplitude is at "
            "least the noise peak-to-peak, its rise is no slower than the maximum rise and its decay no slower than "
            "the maximum decay; rejected events remain in the event table with the criterion they failed. Off by "
            "default: on simulated recordings these criteria reject 15-25% of true events and 83% of slow sIPSCs. "
            "The number of events they would accept is reported in either case.", "", ""),
    Setting("noise_pp_sd", 5.15, "Noise peak-to-peak (robust SDs)", "Noise is measured as the baseline's "
            "peak-to-peak range; for Gaussian noise, 99% of samples fall within 5.15 SDs.", "SD",
            "", advanced=True, minimum=0.0),
    Setting("max_rise_ms", 2.0, "Maximum rise, 10-90% (ms)", "Events rising slower than this are rejected; "
            "a typical value is 2-3 times the observed rise time.", "ms", "", advanced=True,
            minimum=0.05),
    Setting("max_decay_ms", 15.0, "Maximum decay tau (ms)", "Events decaying slower than this are rejected; set it "
            "to the decay of the slowest genuine events.", "ms",
            "", advanced=True, minimum=0.5),
    Setting("amplitude_window", False, "Amplitude window", "Only events with amplitudes between the two limits are accepted.", "",
            "", advanced=True),
    Setting("amplitude_min", 1.0, "Amplitude window: from", "", "signal units", "", advanced=True),
    Setting("amplitude_max", 100.0, "Amplitude window: to", "", "signal units", "", advanced=True),
    Setting("exclude", "", "Excluded stretches (s)", "Stretches to exclude, e.g. '10-12, 30-31.5' (artifacts, test "
            "pulses); shown in grey.", "", "", advanced=True),
    Setting("regions", "", "Regions (name: from-to s)", "Named stretches analyzed separately, "
            "e.g. 'baseline: 0-300; drug: 300-900'.", "", "", advanced=True),
    Setting("smoothing", "none", "Smoothing before detection", "Moving average or Savitzky-Golay (better preserves fast "
            "rises). Off by default, since the band-pass filter already smooths the trace.", "", "", advanced=True,
            choices=("none", "moving average", "Savitzky-Golay")),
    Setting("smoothing_ms", 1.0, "Smoothing width (ms)", "", "ms", "", advanced=True, minimum=0.05),
    Setting("event_fit", "decay only", "Each event's fit", "The decay from 80% of the peak (default), or the whole "
            "event with rising and decaying exponentials.", "", "", advanced=True,
            choices=("decay only", "whole event")),
    Setting("iei_hist_max_s", 1.0, "Interval histogram: up to (s)", "", "s", "", advanced=True),
    Setting("iei_hist_bin_ms", 10.0, "Interval histogram: bins (ms)", "", "ms", "", advanced=True),
    Setting("amp_hist_max", 100.0, "Amplitude histogram: up to (0: automatic)", "", "signal units", "",
            advanced=True),
    Setting("amp_hist_bin", 1.0, "Amplitude histogram: bins", "", "signal units", "", advanced=True),
    Setting("stability_p", 0.01, "Evaluation: unstable below this p", "", "p", "Clampwise", advanced=True),
    Setting("overlap_share", 0.1, "Evaluation: possible overlap above this share of events", "", "",
            "Clampwise", advanced=True),
    Setting("drift_noise_sds", 5.0, "Evaluation: a drifting baseline beyond this many noise SDs", "", "SD",
            "Clampwise", advanced=True),
    Setting("min_interval_ms", 2.0, "Events at least this far apart (ms)", "", "ms", advanced=True, minimum=0.1),
    Setting("baseline_block_ms", 100.0, "Slow baseline: blocks of (ms)", "The running median taken off before "
            "detection; at least ten template decays when they are longer than 10 ms.", "ms", advanced=True,
            minimum=5.0),
    Setting("min_decay_r2", 0.8, "Per-event decay fits: r² at least", "Fits below this are flagged and left out "
            "of the median decay.", "", advanced=True, minimum=0.0, maximum=1.0),
    Setting("exclude_spikes", True, "Leave out action potentials (current clamp)", "From 3 ms before threshold to "
            "the time below after the peak.", advanced=True, source=""),
    Setting("spike_exclusion_ms", 25.0, "After each action potential, leave out (ms)", "", "ms", advanced=True,
            minimum=0.0),
    Setting("evoked_min_s", EVOKED_EVENTS_MIN_S, "Evoked recordings: events when at least (s) per sweep is "
            "free of stimuli", "Voltage-clamp sweeps this long or longer outside the stimulus windows get events too.",
            "s", advanced=True, minimum=0.0, source=""),
    Setting("edits", "", "Reviewed events", "Rejected and added events, saved by the event review.", advanced=True,
            hidden=True),
)

OUTPUTS = (
    Output("frequency", "Accepted events per second of the time analyzed (left-out stretches not counted).", "Hz",
           "", label="Frequency", headline=True),
    Output("n_events", "Accepted events.", "count", label="Events", headline=True),
    Output("amplitude_mean", "Mean amplitude of the accepted events, each against the median of the 3 to 0.5 ms "
           "before its onset.", "signal units", "",
           label="Amplitude (mean)", headline=True),
    Output("amplitude_median", "Median amplitude of the accepted events.", "signal units", label="Amplitude (median)"),
    Output("rise_10_90", "10-90% rise time of the averaged event (events aligned on their rise midpoint), used as the "
           "recording's kinetics.", "ms", "Pernia-Andrade et al. 2012",
           label="Rise 10-90% (averaged event)", headline=True),
    Output("decay_tau", "Decay time constant of the averaged event, a single exponential from 80% of its peak.", "ms",
           "", label="Decay tau (averaged event)", headline=True),
    Output("charge_mean", "Mean charge of the accepted events (voltage clamp, pC) or area (current clamp, mV*ms), "
           "from the onset to six decay taus.", "pC or mV*ms", label="Charge (mean)"),
    Output("charge_median", "Median charge of the accepted events.", "pC or mV*ms", label="Charge (median)"),
    Output("iei_mean", "Mean interval between consecutive accepted events (never across a left-out stretch or "
           "between sweeps).", "ms", label="Interval (mean)"),
    Output("iei_median", "Median interval between consecutive accepted events.", "ms", label="Interval (median)"),
    Output("iei_cv", "Coefficient of variation of the intervals: 1 for a random (Poisson) process at a steady rate.",
           "ratio", "Dayan & Abbott, ch. 1", label="Interval CV"),
    Output("iei_lv", "Local variation of consecutive intervals: 1 for a Poisson process, below 1 for regular, above "
           "for bursty; not moved by slow changes of rate.", "ratio", "Shinomoto et al. 2003",
           label="Interval LV"),
    Output("timing", "Whether the events look like a random process at the rate they have in each bin: the intervals "
           "rescaled by that rate, against an exponential (KS test, null simulated with the same rates and dead "
           "time); bursty or more regular when p < 0.05, by the rescaled CV.", "",
           "Brown et al. 2002 (time rescaling); Kass, Eden & Brown", label="Timing", headline=True),
    Output("timing_p", "p of the timing test (the time-rescaled intervals against an exponential).", "p",
           "Brown et al. 2002", label="Timing test p"),
    Output("iei_cv_rescaled", "CV of the time-rescaled intervals: above 1 bursty, below 1 regular.", "ratio",
           "Brown et al. 2002", label="Rescaled interval CV"),
    Output("evaluation", "Statements read from the events, each with its test and numbers (the evaluation "
           "table).", "", "", label="Evaluation"),
    Output("noise_sd", "Robust SD of the baseline-free trace (1.4826 x MAD).", "signal units", label="Noise (SD)"),
    Output("noise_pp", "The noise's peak-to-peak: the acceptance level for an event's amplitude.", "signal units",
           "", label="Noise (peak-to-peak)"),
    Output("n_failed_criteria", "Detections that fail the acceptance criteria (marked in the event table).", "count",
           "", label="Failing acceptance criteria"),
    Output("n_events_nex_criteria", "The number of events the acceptance criteria would accept.", "count", "",
           label="Events passing acceptance criteria"),
    Output("rise_median", "Median 10-90% rise of the accepted events.", "ms", label="Rise 10-90% (per-event median)"),
    Output("decay_median", "Median decay tau of the accepted events with a decay fit of r² at least the setting.",
           "ms", label="Decay tau (per-event median)"),
    Output("half_width_median", "Median half-width of the accepted events.", "ms", label="Half-width (median)"),
    Output("decay_r2", "r² of the averaged event's decay fit.", "ratio", label="Averaged event decay r²"),
    Output("average_n", "Events in the averaged event (those with room on both sides).", "count",
           label="Events averaged"),
    Output("charge_rate", "Charge carried per second by everything that is not baseline, against an event-free "
           "baseline (the median of the samples no event covers, per 0.5 s): independent of the detector.",
           "pC/s or mV", "", label="Charge per second"),
    Output("analyzed_s", "Time analyzed: the stretch chosen, minus what was left out.", "s", label="Time analyzed"),
    Output("n_rejected", "Events rejected in the review.", "count", label="Rejected in review"),
    Output("n_unmeasured", "Detections with no clear peak to measure, not counted (in the event table, not "
           "accepted).", "count", label="Detections not measurable"),
    Output("n_added", "Events added by hand in the review.", "count", label="Added in review"),
    Output("n_other_direction", "Events the same detection finds pointing the other way: evidence on the direction "
           "(flagged when more than twice the events found and at least 20).", "count",
           label="Events the other way"),
    Output("template_rise_ms", "The template's rise tau as used.", "ms", label="Template rise tau"),
    Output("template_decay_ms", "The template's decay tau as used.", "ms", label="Template decay tau"),
    Output("template", "The template used and where it came from.", "", label="Template"),
    Output("method", "The detection method used: deconvolution or template matching (its setting).", "",
           label="Detection method"),
    Output("threshold", "The detection threshold used, on the chosen method's own scale (its setting).", "",
           label="Detection threshold"),
    Output("units", "The units of the amplitudes: the recorded signal's.", "", label="Amplitude units"),
    Output("integral_units", "The units of the charge: pC in voltage clamp, mV*ms in current clamp.", "",
           label="Charge units"),
    Output("direction", "Which way the events point, and why.", "", label="Direction"),
    Output("event_amplitude", "One event's amplitude.", "signal units", level="event"),
    Output("event_interval", "The interval from the previous accepted event.", "ms", level="event"),
)


def _direction(recording: Recording, chosen: str) -> tuple[int, str]:
    if chosen == "down":
        return -1, "down (set)"
    if chosen == "up":
        return 1, "up (set)"
    if not recording.is_voltage_clamp:
        return 1, "up: current clamp"
    hold = recording.holding
    cmd = recording.command
    if not np.isfinite(hold) and cmd is not None and cmd.size:
        hold = float(np.median(cmd[:, : max(1, cmd.shape[1] // 20)]))
    if np.isfinite(hold) and hold >= UPWARD_FROM_MV:
        return 1, f"up: voltage clamp held at {hold:g} mV, at or above {UPWARD_FROM_MV:g} mV (outward IPSCs)"
    return -1, ("down: voltage clamp" + (f" held at {hold:g} mV" if np.isfinite(hold) else ", holding unknown"))


def _template(s: dict, fs: float) -> tuple[np.ndarray, dict, str]:
    """The template (upward), the settings with its kinetics as used, and where it came from."""
    if s["template_file"]:
        loaded = read_atf(s["template_file"], fs)
        if loaded is not None:
            rise, decay, _, rise_tau = shape_kinetics(loaded, fs)
            s = dict(s, template_rise_ms=rise_tau if np.isfinite(rise_tau) else s["template_rise_ms"],
                     template_decay_ms=decay if np.isfinite(decay) else s["template_decay_ms"])
            return loaded, s, (f"{Path(s['template_file']).name} (rise tau {s['template_rise_ms']:.2f} / decay "
                               f"{s['template_decay_ms']:.2f} ms)")
        source = f"analytic ({Path(s['template_file']).name} could not be read)"
    else:
        source = "analytic"
    source += f", rise {s['template_rise_ms']:g} / decay {s['template_decay_ms']:g} ms"
    if s["template_origin"]:
        source += f", from {s['template_origin']}"
    return template_shape(fs, s["template_rise_ms"], s["template_decay_ms"]), s, source


@register("events", "Spontaneous events", version=2, types=SPONTANEOUS | {"vc_evoked", "dsi"}, settings=SETTINGS,
          outputs=OUTPUTS, requires=("baseline",),
          description="Spontaneous and miniature synaptic events: detection by deconvolution or template matching, "
                      "per-event measurements with event-by-event review, averaged-event kinetics, interevent "
                      "intervals, a test of random versus bursty timing, and event time courses.",
          source="Pernia-Andrade et al. 2012; Clements & Bekkers 1997; Brown et al. 2002")
def run(recording: Recording, context, s) -> Result | None:
    out = Result("events", 2)
    fs = recording.sample_rate_hz
    direction, why_dir = _direction(recording, s["direction"])
    tpl, s, source = _template(dict(s), fs)
    out.choose("direction", "up" if direction > 0 else "down", why_dir)
    out.choose("template", source, "one template per experiment, applied identically to every recording; it is not "
               "refined per recording, which would let detection differ between conditions")
    out.choose("recording's numbers", "means, with medians alongside; kinetics from the averaged event",
               "the averaged event is less affected by noise than single-event fits")
    if recording.header.get("event_driven"):
        return _event_driven(recording, s, direction, tpl, source, out)
    lo_s, hi_s = parse_stretch(str(context.shared.get("stretch", "") or ""))
    sweeps = range(recording.n_sweeps)
    # what is left out, per sweep, on the sweep's own clock
    left: dict[int, list[tuple[float, float, str]]] = {k: [] for k in sweeps}
    stim = context.stimulus
    if stim is not None and len(stim):
        for k in sweeps:
            if k < len(stim.times_s):
                widths = stim.widths_s[k] if k < len(stim.widths_s) else np.zeros(len(stim.times_s[k]))
                for t0, w in zip(stim.times_s[k], widths):
                    left[k].append((float(t0) - 0.002, float(t0) + float(w) + 0.1, "stimulus"))
    if context.decision.recording_type == "vc_evoked":
        free = [recording.sweep_duration_s - sum(b - a for a, b, _ in _merge(left[k])) for k in sweeps]
        if not free or min(free) < s["evoked_min_s"]:
            return None
        out.notes.append(f"an evoked recording with at least {min(free):.1f} s per sweep free of stimuli: events are "
                         "detected outside the stimulus windows (2 ms before to 100 ms after each)")
    steps = dsi_steps(recording)
    step_rows = []
    for k, a, b, level in steps:
        settle = settle_s(recording.sweep(k), fs, a, b)
        left[k].append((a - 0.005, b + settle, f"step to {level:g} mV and {settle:.2f} s settling"))
        step_rows.append({"sweep": k, "start_s": a, "end_s": b, "level_mv": level, "settle_s": settle})
    if steps:
        out.notes.append(f"{len(steps)} depolarizing step(s) left out, each with the current's settling after it "
                         f"(at least {SETTLE_MIN_S:g} s)")
    n_spikes = 0
    other_way = 0
    if not recording.is_voltage_clamp and s["exclude_spikes"]:
        for k in sweeps:
            for sp in spikes.find(recording.sweep(k), fs, purpose="action potentials left out of event detection"):
                left[k].append((sp.threshold_index / fs - 0.003, sp.peak_index / fs + s["spike_exclusion_ms"] * 1e-3,
                                "action potential"))
                n_spikes += 1
    for a, b in parse_stretches(s["exclude"]):
        for k in sweeps:
            left[k].append((a, b, "left out by hand"))
    rejected, added = parse_edits(s["edits"])
    block = s["baseline_block_ms"] if s["template_decay_ms"] <= 10.0 else max(s["baseline_block_ms"],
                                                                                10.0 * s["template_decay_ms"])
    ledger.note(f"slow baseline: the running median of {block:g} ms blocks taken off", "event detection")
    if s["method"] == "deconvolution":
        top = band_top_hz(s["template_rise_ms"], s["band_top_hz"])
        ledger.note(f"deconvolution by the template (Wiener-regularized), band-pass 1-{min(top, fs / 2 * 0.9):g} Hz, "
                    f"peaks above {s['deconvolution_threshold']:g} robust SD (Pernia-Andrade et al. 2012)",
                    "event detection")
    else:
        ledger.note(f"template matching, criterion above {s['template_threshold']:g} (Clements & Bekkers 1997)",
                    "event detection")
    events: list[Event] = []
    noise_rows: list[tuple[int, float, float]] = []
    drift_rows: list[tuple[int, float]] = []
    traces: dict[int, tuple[float, np.ndarray]] = {}       # sweep -> (time of its first sample, upward trace)
    segs: list[tuple[int, float, float]] = []            # (sweep, start, end) analyzed, left-out stretches removed
    excluded_rows = []
    analyzed = 0.0
    total = 0.0
    used: dict[int, list] = {}          # each sweep's left-out stretches as detection used them
    for k in sweeps:
        raw_full = recording.sweep(k)
        n_all = raw_full.size
        i_lo = int(np.clip(math.floor(lo_s * fs), 0, n_all))
        i_hi = n_all if not math.isfinite(hi_s) else int(np.clip(math.ceil(hi_s * fs), i_lo, n_all))
        if i_hi - i_lo < int(0.05 * fs):
            continue
        t0 = i_lo / fs
        t1 = i_hi / fs
        raw = np.array(raw_full[i_lo:i_hi], dtype=np.float64)
        spans = [(max(a, t0), min(b, t1), w) for a, b, w in _merge(left[k]) if b > t0 and a < t1]
        # a stretch at either end of the sweep shorter than MIN_STRETCH_S, cut off by
        # a left-out part, is left out too: the detector's 1 Hz filter has nothing
        # before (or after) it to start from (the 60 ms before a DSI step file's step
        # gave 75 "events" a second, 2026-10-04). Between left-out parts the trace is
        # bridged and continuous, so short stretches there are searched as before.
        if spans:
            if 0 < spans[0][0] - t0 < MIN_STRETCH_S:
                spans = _merge(spans + [(t0, spans[0][0], f"shorter than {MIN_STRETCH_S:g} s at the sweep's start")])
            if 0 < t1 - spans[-1][1] < MIN_STRETCH_S:
                spans = _merge(spans + [(spans[-1][1], t1, f"shorter than {MIN_STRETCH_S:g} s at the sweep's end")])
        used[k] = spans
        for a, b, w in spans:
            excluded_rows.append({"sweep": k, "start_s": a, "end_s": b, "why": w})
            # bridged from 2 ms into the stretch, so an onset placed just before its corner still falls inside
            i0 = int(np.clip(round((a + 0.002 - t0) * fs), 1, raw.size - 1))
            i1 = int(np.clip(round((b - t0) * fs), 1, raw.size - 1))
            if i1 > i0:
                raw[i0:i1] = np.linspace(raw[i0 - 1], raw[i1], i1 - i0, endpoint=False)
        if s["smoothing"] == "moving average":
            raw = filters.moving_average(raw, max(1, int(round(s["smoothing_ms"] * 1e-3 * fs))))
        elif s["smoothing"] == "Savitzky-Golay":
            raw = filters.savitzky_golay(raw, fs, s["smoothing_ms"] * 1e-3, purpose="event detection")
        flat = remove_slow(raw, fs, block)
        slow = raw - flat
        drift_rows.append((k, float(np.percentile(slow, 95) - np.percentile(slow, 5)) if slow.size else NAN))
        dev = direction * flat
        traces[k] = (t0, dev)
        total += t1 - t0
        cut = sum(b - a for a, b, _ in spans)
        analyzed += (t1 - t0) - cut
        free_at = t0
        for a, b, _ in spans:
            if a > free_at:
                segs.append((k, free_at, a))
            free_at = max(free_at, b)
        if free_at < t1:
            segs.append((k, free_at, t1))
        idx, score, n_other = find_events(dev, fs, tpl, s)
        other_way += n_other
        sd = noise_sd(dev)
        noise_pp = s["noise_pp_sd"] * sd
        noise_rows.append((k, sd, noise_pp))
        onsets = []
        for i, sc in zip(idx, score):
            ts = t0 + i / fs
            if any(a <= ts <= b for a, b, _ in spans):
                continue
            onsets.append((int(i), float(sc), False))
        onsets += [(int(round((ts - t0) * fs)), NAN, True) for sw, ts in added
                   if sw == k and t0 <= ts < t1]
        onsets.sort()
        previous = None
        for j, (i, sc, manual) in enumerate(onsets):
            nxt = onsets[j + 1][0] if j + 1 < len(onsets) else None
            m = measure_event(dev, i, nxt, fs, s["template_decay_ms"])
            e = Event(k, t0 + i / fs, score=sc, manual=manual)
            b0, b1 = max(0, i - int(0.003 * fs)), max(1, i - int(0.0005 * fs))
            e.baseline = float(np.median(raw[b0:b1])) if b1 > b0 else NAN
            if m["peak"] >= 0 and s["event_fit"] == "whole event" and m["end"] > i + 8:
                tt = np.arange(i, m["end"]) / fs
                w = fits.synaptic_waveform(tt, dev[i:m["end"]], purpose="each event, rise and decay together",
                                           rise0=m["rise"] / 2.2e3 if np.isfinite(m["rise"]) else None,
                                           decay0=m["tau"] / 1e3 if np.isfinite(m["tau"]) else None,
                                           rise_max=max(s["max_rise_ms"], 2 * (m["rise"] if np.isfinite(m["rise"])
                                                                               else 1.0)) / 1e3)
                if w.ok and w["tau_rise"] < w["tau_decay"]:
                    e.rise_tau_ms, e.decay_tau_whole_ms = w["tau_rise"] * 1e3, w["tau_decay"] * 1e3
            if m["peak"] >= 0:
                e.peak_s = t0 + m["peak"] / fs
                e.amplitude = direction * m["amplitude"]
                e.rise_10_90_ms, e.decay_tau_ms, e.decay_r2 = m["rise"], m["tau"], m["r2"]
                e.charge = direction * m["charge"] * (1.0 if recording.is_voltage_clamp else 1e3)
                e.half_width_ms = m["half"]
                e.mid_s = t0 + m["mid"] / fs if np.isfinite(m["mid"]) else NAN
                e.end_s = t0 + m["end"] / fs if m["end"] >= 0 else NAN
            reject = any(sw == k and abs(ts - e.onset_s) <= REJECT_TOLERANCE_S for sw, ts in rejected)
            e.accepted = not reject and (manual or np.isfinite(e.amplitude))
            if e.accepted:
                e.why_not = acceptance(e, noise_pp, s)          # what NeuroExpress's criteria say of it
                e.accepted = not (s["nex_criteria"] and e.why_not)
            if e.accepted:
                # an interval across a left-out stretch was not observed: events may have happened in it
                if previous is not None and not any(a < e.onset_s and b > previous for a, b, _ in spans):
                    e.iei_ms = (e.onset_s - previous) * 1e3
                previous = e.onset_s
            events.append(e)
    if not traces:
        out.notes.append("nothing to analyze: the stretch chosen holds less than 50 ms")
        return out
    _summarize(out, recording, s, events, traces, segs, analyzed, direction, tpl, source, fs)
    out.tables["excluded"] = excluded_rows
    # what a cell's analyses (DSI from events) read: the stretches analyzed and the steps
    out.tables["analyzed"] = [{"sweep": k, "start_s": a, "end_s": b} for k, a, b in segs]
    out.tables["steps"] = step_rows
    if not recording.is_voltage_clamp and total > 0 and analyzed < 0.5 * total and n_spikes:
        out.flags.append(Flag("mostly firing", f"The cell fired through most of the recording: {100 * (1 - analyzed / total):.0f}% "
                              "of the time was left out around action potentials, and what is found between spikes "
                              "is mostly their after-hyperpolarization. These event numbers are not meaningful.",
                              "at least half the time left out", ""))
    if math.isfinite(hi_s) or lo_s > 0:
        out.choose("stretch", f"{lo_s:g} s to {'the end' if not math.isfinite(hi_s) else f'{hi_s:g} s'}",
                   "chosen for this recording; events only from there")
    found = sum(1 for e in events if not e.manual)
    out.values["n_other_direction"] = other_way
    if other_way >= 20 and other_way > 2 * found:
        out.flags.append(Flag("direction?", f"The detector finds {other_way} events pointing "
                              f"{'down' if direction > 0 else 'up'} against {found} pointing "
                              f"{'up' if direction > 0 else 'down'}, the direction used ({why_dir}). The events may "
                              "point the other way: a holding set on the amplifier rather than in Clampex is not in "
                              "the file. Set 'Events point' if so.", "the other direction finds more than twice as many",
                              ""))
    _count_review(out, events, rejected)
    out.values["n_added"] = sum(1 for e in events if e.manual)
    _criteria(out, events, noise_rows, s)
    _neuroexpress(out, recording, events, segs, drift_rows, s)
    _charge_rate(out, recording, s, events, used, direction, lo_s, hi_s)
    return out


def _criteria(out: Result, events: list[Event], noise_rows, s: dict) -> None:
    """What NeuroExpress's acceptance criteria left out, by criterion."""
    if noise_rows:
        out.values["noise_sd"] = float(np.median([r[1] for r in noise_rows]))
        out.values["noise_pp"] = float(np.median([r[2] for r in noise_rows]))
    failed = [e for e in events if e.why_not]
    out.values["n_failed_criteria"] = len(failed)
    counted = [e for e in events if e.accepted and not e.manual]
    out.values["n_events_nex_criteria"] = sum(1 for e in counted if not e.why_not) + sum(1 for e in events
                                                                                          if e.manual and e.accepted)
    by = {}
    for e in failed:
        key = e.why_not.split(" (")[0]
        by[key] = by.get(key, 0) + 1
    rule = (f"at least the noise's peak-to-peak ({out.values.get('noise_pp', NAN):.3g}, {s['noise_pp_sd']:g} robust "
            f"SDs), 10-90% rise at most {s['max_rise_ms']:g} ms, decay tau at most {s['max_decay_ms']:g} ms")
    reasons = ", ".join(f"{n} {k}" for k, n in by.items()) or "none"
    if s["nex_criteria"]:
        out.choose("acceptance", f"acceptance criteria: {rule}", f"set for this analysis; left out: {reasons} "
                   "(in the event table, not accepted)")
    else:
        out.notes.append(f"The acceptance criteria ({rule}) would accept {out.values['n_events_nex_criteria']} of these "
                         f"events (rejected: {reasons}); off by default, as on simulated "
                         "recordings they reject 15-25% of true events and 83% of slow sIPSCs")


def _summarize(out: Result, recording: Recording, s: dict, events: list[Event],
               traces: dict[int, tuple[float, np.ndarray]],
               segs, analyzed: float, direction: int, tpl, source: str, fs: float,
               window: tuple[int, int] | None = None) -> None:
    acc = [e for e in events if e.accepted]
    vc = recording.is_voltage_clamp

    def vals(name: str, good=None) -> np.ndarray:
        v = np.array([getattr(e, name) for e in acc if good is None or good(e)], float)
        return v[np.isfinite(v)]

    def mean(v):
        return float(np.mean(v)) if v.size else NAN

    def median(v):
        return float(np.median(v)) if v.size else NAN

    amp, charge, iei = vals("amplitude"), vals("charge"), vals("iei_ms")
    good_fit = lambda e: np.isfinite(e.decay_r2) and e.decay_r2 >= s["min_decay_r2"]      # noqa: E731
    out.values.update({
        "units": recording.signal_units, "integral_units": "pC" if vc else "mV*ms",
        "frequency": len(acc) / analyzed if analyzed > 0 else NAN, "n_events": len(acc),
        "amplitude_mean": mean(amp), "amplitude_median": median(amp),
        "charge_mean": mean(charge), "charge_median": median(charge),
        "iei_mean": mean(iei), "iei_median": median(iei),
        "iei_cv": float(np.std(iei) / np.mean(iei)) if iei.size >= 2 and np.mean(iei) > 0 else NAN,
        "rise_median": median(vals("rise_10_90_ms")), "decay_median": median(vals("decay_tau_ms", good_fit)),
        "half_width_median": median(vals("half_width_ms")), "analyzed_s": analyzed,
        "template_rise_ms": s["template_rise_ms"], "template_decay_ms": s["template_decay_ms"],
        "template": f"{s['method']}, {source}, threshold "
                    f"{s['deconvolution_threshold'] if s['method'] == 'deconvolution' else s['template_threshold']:g}",
        "direction": "up" if direction > 0 else "down",
        "method": s["method"],
        "threshold": s["deconvolution_threshold"] if s["method"] == "deconvolution" else s["template_threshold"],
    })
    poor = sum(1 for e in acc if np.isfinite(e.decay_r2) and e.decay_r2 < s["min_decay_r2"])
    if poor:
        out.notes.append(f"{poor} of {len(acc)} events' decay fits have r² below {s['min_decay_r2']:g}; they are "
                         "flagged in the event table and left out of the median decay")
    # timing: runs of consecutive intervals for LV; the rescaling test on the analyzed segments
    runs, run_now = [], []
    for e in events:
        if not e.accepted:
            continue
        if np.isfinite(e.iei_ms):
            run_now.append(e.iei_ms)
        else:
            if len(run_now) >= 2:
                runs.append(np.asarray(run_now))
            run_now = []
    if len(run_now) >= 2:
        runs.append(np.asarray(run_now))
    out.values["iei_lv"] = local_variation(runs)
    seg_times = []
    seg_spans = []
    for k, a, b in segs:
        ts = np.array([e.onset_s for e in acc if e.sweep == k and a <= e.onset_s < b])
        if b - a > 0:
            seg_times.append(ts)
            seg_spans.append((a, b))
    test = timing_test(seg_times, seg_spans, s["timecourse_bin_s"], s["min_interval_ms"] * 1e-3,
                       seed=len(acc))
    out.values["timing_p"] = test["p"]
    out.values["iei_cv_rescaled"] = test["cv_rescaled"]
    out.values["timing"] = timing_words(test, test["n"])
    if test["n"] >= MIN_INTERVALS:
        ledger.note(f"timing: intervals rescaled by the rate in each {s['timecourse_bin_s']:g} s bin, KS against an "
                    f"exponential, null from {MONTE_CARLO} simulated Poisson processes at the same rates through "
                    f"the {s['min_interval_ms']:g} ms dead time (Brown et al. 2002)", "event timing")
    # the averaged event, aligned on each event's rise midpoint; its window follows
    # the events' own decay (the template's can be half of it), and only events
    # with no other event inside the window go in (in a burst a neighbour's
    # current would be averaged in as decay)
    pre = int(round(0.005 * fs))
    decays = vals("decay_tau_ms", good_fit)
    span_ms = max(5 * s["template_decay_ms"], 5 * (float(np.median(decays)) if decays.size else 0.0), 20.0)
    post = int(round(min(span_ms, DECAY_MAX_MS) * 1e-3 * fs))
    if window is not None:                  # an event-driven file: what its short sweeps hold
        pre, post = window
        span_ms = post / fs * 1e3
    found = {}
    for e in events:
        found.setdefault(e.sweep, []).append(e.onset_s)
    found = {k: np.sort(v) for k, v in found.items()}
    candidates, isolated = [], []
    for e in acc:
        if e.sweep not in traces:
            continue
        t0, y = traces[e.sweep]
        anchor = e.mid_s if np.isfinite(e.mid_s) else e.onset_s
        i = int(round((anchor - t0) * fs))
        if i - pre < 0 or i + post >= y.size:
            continue
        seg = np.asarray(y[i - pre:i + post], float)
        seg = seg - np.median(seg[:max(2, pre // 2)])
        candidates.append(seg)
        others = found[e.sweep]
        lo_t, hi_t = anchor - pre / fs, anchor + post / fs
        n_in = int(np.searchsorted(others, hi_t) - np.searchsorted(others, lo_t))
        if n_in <= 1:
            isolated.append(seg)
    snippets = isolated if len(isolated) >= 5 else candidates
    if snippets is candidates and candidates:
        out.notes.append(f"fewer than 5 events stand alone in the averaged event's window ({span_ms:.0f} ms): every "
                         "event was averaged, so neighbouring events lengthen its decay")
    out.values["average_n"] = len(snippets)
    if snippets:
        avg = np.mean(snippets, axis=0)
        ledger.note("averaged event: the accepted events with no other event within 5 ms before to five decay taus "
                    "after, aligned on each one's rise midpoint (50% of its amplitude)", "event kinetics")
        rise, tau, r2, _ = shape_kinetics(avg, fs)
        out.values.update({"rise_10_90": rise, "decay_tau": tau, "decay_r2": r2})
        out.drawables.append(Drawable("curve", "averaged event", x=(np.arange(avg.size) - pre) / fs * 1e3,
                                      y=direction * avg, domain="event"))
        clean = np.isfinite(r2) and r2 >= 0.9 and 0.3 <= tau <= 200.0 and np.isfinite(rise) and 0 < rise < tau
        if clean and not s["template_file"] and tau > SLOW_EVENTS * s["template_decay_ms"]:
            out.flags.append(Flag("slow events", f"The averaged event decays in {tau:.0f} ms, more than "
                                  f"{SLOW_EVENTS:g} times the template's {s['template_decay_ms']:g} ms, so slow events "
                                  "may be counted as several (on simulated 30 ms decays a 5 ms template found 40% too "
                                  "many). Make one template for the experiment from its averaged event and analyze "
                                  "every recording with it.", f"averaged decay above {SLOW_EVENTS:g} x the template's",
                                  ""))
        if not clean:
            # a value from a fit that measured nothing is flagged, never blanked (2026-09-26)
            out.flags.append(Flag("average not clean", f"The averaged event is not a clean single decay (fit r² "
                                  f"{r2:.2f}, decay {tau:.3g} ms, rise {rise:.3g} ms), so its kinetics are not a "
                                  "measurement: overlapping or mixed events, or few that stand alone. The per-event "
                                  f"medians are beside it (decay {out.values['decay_median']:.3g} ms).",
                                  "r² at least 0.9, decay 0.3-200 ms, rise shorter than decay",
                                  ""))
    else:
        out.values.update({"rise_10_90": NAN, "decay_tau": NAN, "decay_r2": NAN})
    if len(acc) < 20:
        out.notes.append(f"only {len(acc)} events: the averaged event and the interval statistics are rough")
    # events over time
    bin_s = s["timecourse_bin_s"]
    rows = []
    by_bin: dict[tuple[int, int], dict] = {}
    for k, a, b in segs:
        j = int(a // bin_s)
        while j * bin_s < b:
            lo, hi = max(a, j * bin_s), min(b, (j + 1) * bin_s)
            if hi > lo:
                row = by_bin.setdefault((k, j), {"sweep": k, "bin_start_s": j * bin_s, "analyzed_s": 0.0,
                                                 "n_events": 0, "amps": []})
                row["analyzed_s"] += hi - lo
            j += 1
    for e in acc:
        key = (e.sweep, int(e.onset_s // bin_s))
        if key in by_bin:
            by_bin[key]["n_events"] += 1
            if np.isfinite(e.amplitude):
                by_bin[key]["amps"].append(e.amplitude)
    for key in sorted(by_bin):
        r = by_bin[key]
        rows.append({"sweep": r["sweep"], "bin_start_s": r["bin_start_s"], "analyzed_s": r["analyzed_s"],
                     "n_events": r["n_events"],
                     "frequency_hz": r["n_events"] / r["analyzed_s"] if r["analyzed_s"] > 0 else NAN,
                     "amplitude_mean": float(np.mean(r["amps"])) if r["amps"] else NAN})
    out.tables["timecourse"] = rows
    out.tables["events"] = [
        {"sweep": e.sweep, "event": i + 1, "onset_s": e.onset_s, "peak_s": e.peak_s, "amplitude": e.amplitude,
         "rise_10_90_ms": e.rise_10_90_ms, "decay_tau_ms": e.decay_tau_ms, "decay_r2": e.decay_r2,
         "decay_poor_fit": bool(np.isfinite(e.decay_r2) and e.decay_r2 < s["min_decay_r2"]),
         "charge": e.charge, "half_width_ms": e.half_width_ms, "iei_ms": e.iei_ms, "score": e.score,
         "accepted": e.accepted, "manual": e.manual, "end_s": e.end_s, "why_not": e.why_not,
         "baseline_level": e.baseline,
         "local_frequency_hz": 1e3 / e.iei_ms if np.isfinite(e.iei_ms) and e.iei_ms > 0 else NAN,
         "area_per_amplitude_ms": (abs(e.charge / e.amplitude) * (1e3 if vc else 1.0)
                                   if np.isfinite(e.charge) and np.isfinite(e.amplitude) and e.amplitude else NAN),
         "rise_tau_ms": e.rise_tau_ms, "decay_tau_whole_ms": e.decay_tau_whole_ms}
        for i, e in enumerate(events)]
    out.tables = {"events": out.tables["events"], "timecourse": out.tables["timecourse"],
                  **{k: v for k, v in out.tables.items() if k not in ("events", "timecourse")}}
    if out.tables["events"]:
        out.notes.append("Events from one cell are not independent: compare cells by their per-cell numbers, not "
                         "pooled events (pooling inflates n).")


def histogram(x: np.ndarray, top: float, width: float) -> list[dict]:
    """Counts in bins of ``width`` up to ``top`` (0: the 99th percentile), the
    cumulative fraction, and a Gaussian kernel (Parzen) estimate scaled to counts
    (Scott's bandwidth; Parzen 1962)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2 or width <= 0:
        return []
    top = top if top > 0 else float(np.percentile(x, 99)) or float(x.max())
    edges = np.arange(0.0, top + width, width)
    if edges.size < 2:
        return []
    counts, _ = np.histogram(np.clip(x, 0, edges[-1]), edges)
    centers = (edges[:-1] + edges[1:]) / 2
    kde = np.full(centers.size, NAN)
    if np.unique(x).size >= 2:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(x)(centers) * x.size * width
    cum = np.cumsum(counts) / x.size
    return [{"from": float(a), "to": float(b), "count": int(c), "cumulative": float(f), "kernel": float(k)}
            for a, b, c, f, k in zip(edges[:-1], edges[1:], counts, cum, kde)]


def _neuroexpress(out: Result, recording: Recording, events: list[Event], segs, drift_rows, s: dict) -> None:
    """NeuroExpress's mEPSC additions: histograms with their
    cumulative and kernel estimate, named regions with their own statistics, and
    the evaluation (counts and amplitudes stable or not, overlapping events, a
    drifting baseline), each statement with its test and numbers."""
    from scipy import stats

    acc = [e for e in events if e.accepted and np.isfinite(e.amplitude)]
    out.tables["histogram_intervals"] = histogram(np.array([e.iei_ms for e in acc]) / 1e3, s["iei_hist_max_s"],
                                                  s["iei_hist_bin_ms"] / 1e3)
    out.tables["histogram_amplitudes"] = histogram(np.abs([e.amplitude for e in acc]), s["amp_hist_max"],
                                                   s["amp_hist_bin"])
    out.tables["histogram_charge"] = histogram(np.abs([e.charge for e in acc]), 0.0,
                                               max(1e-9, float(np.nanpercentile(np.abs([e.charge for e in acc]), 99))
                                                   / 50) if acc else 0.0)
    # regions
    regions = parse_regions(s["regions"])
    if regions:
        rows = []
        for name, a, b in regions:
            t = sum(max(0.0, min(hi, b) - max(lo, a)) for _, lo, hi in segs)
            inside = [e for e in acc if a <= e.onset_s < b]
            amp = np.array([e.amplitude for e in inside], float)
            ch = np.array([e.charge for e in inside], float)
            iei = np.array([e.iei_ms for e in inside if np.isfinite(e.iei_ms)], float)
            rows.append({"region": name, "from_s": a, "to_s": b, "analyzed_s": t, "n_events": len(inside),
                         "frequency_hz": len(inside) / t if t > 0 else NAN,
                         "amplitude_mean": float(np.nanmean(amp)) if amp.size else NAN,
                         "amplitude_median": float(np.nanmedian(amp)) if amp.size else NAN,
                         "charge_mean": float(np.nanmean(ch)) if ch.size else NAN,
                         "iei_cv": float(np.std(iei) / iei.mean()) if iei.size >= 2 and iei.mean() > 0 else NAN})
        out.tables["regions"] = rows
        out.choose("regions", "; ".join(f"{r['region']} {r['from_s']:g}-{r['to_s']:g} s" for r in rows),
                   "set for this recording; each region's events and time only")
    # the evaluation
    say = []
    p_lim = s["stability_p"]
    bins = [r for r in out.tables.get("timecourse", []) if r["analyzed_s"] >= 0.5 * s["timecourse_bin_s"]]
    if len(bins) >= 4 and sum(r["n_events"] for r in bins) >= 20:
        c = np.array([r["n_events"] for r in bins], float)
        t = np.array([r["analyzed_s"] for r in bins], float)
        lam = c.sum() / t.sum()
        chi2 = float(np.sum((c - lam * t) ** 2 / (lam * t)))
        p_disp = float(stats.chi2.sf(chi2, len(c) - 1))
        third = max(1, len(c) // 3)
        first, last = c[:third].sum() / t[:third].sum(), c[-third:].sum() / t[-third:].sum()
        ratio = last / first if first > 0 else NAN
        unstable = p_disp < p_lim or (np.isfinite(ratio) and ratio < 0.5)
        say.append(("Event counts fluctuate or drop" if unstable else "Event counts are stable",
                    f"over {len(c)} bins of {s['timecourse_bin_s']:g} s: more variable than a steady random rate "
                    f"p = {p_disp:.2g} (the rule: p < {p_lim:g}); the last third's rate {100 * ratio:.0f}% of the "
                    "first's (the rule: under 50%)", bool(unstable)))
    if len(acc) >= 20:
        tt = np.array([e.onset_s for e in acc])
        aa = np.abs([e.amplitude for e in acc])
        rho, p_amp = stats.spearmanr(tt, aa)
        third = max(1, len(acc) // 3)
        change = float(np.median(aa[-third:]) / np.median(aa[:third])) if np.median(aa[:third]) > 0 else NAN
        unstable = p_amp < p_lim and np.isfinite(change) and abs(change - 1) > 0.2
        say.append(("Event amplitudes fluctuate or drop" if unstable else "Event amplitudes are stable",
                    f"amplitude against time: Spearman rho {rho:.2f}, p = {p_amp:.2g}; the last third's median "
                    f"{100 * change:.0f}% of the first's (the rule: p < {p_lim:g} and a change of more than 20%)",
                    bool(unstable)))
        tau = out.values.get("decay_tau", NAN)
        tau = tau if np.isfinite(tau) and tau > 0 else s["template_decay_ms"]
        close = sum(1 for a, b in zip(acc[:-1], acc[1:]) if a.sweep == b.sweep and (b.onset_s - a.onset_s) * 1e3
                    < 3 * tau)
        share = close / (len(acc) - 1)
        if share > s["overlap_share"]:
            say.append(("Possible overlap of events", f"{100 * share:.0f}% of the events start within 3 decay taus "
                        f"({3 * tau:.1f} ms) of the one before (the rule: more than {100 * s['overlap_share']:.0f}%)",
                        True))
    drift = [d for _, d in drift_rows if np.isfinite(d)]
    sd = out.values.get("noise_sd", NAN)
    if drift and np.isfinite(sd) and sd > 0 and max(drift) > s["drift_noise_sds"] * sd:
        say.append(("The baseline drifts", f"its slow level moves by {max(drift):.3g} {recording.signal_units} "
                    f"(5th to 95th percentile), {max(drift) / sd:.0f} noise SDs (the rule: more than "
                    f"{s['drift_noise_sds']:g})", True))
    out.tables["evaluation"] = [{"statement": a, "why": b, "warning": w} for a, b, w in say]
    out.values["evaluation"] = "; ".join(a for a, _, _ in say) or "too few events to judge"
    for a, b, w in say:
        if w:
            out.flags.append(Flag(a, b, source="", severity="note"))


def _charge_rate(out: Result, recording: Recording, s: dict, events: list[Event], left, direction: int,
                 lo_s: float, hi_s: float) -> None:
    """Charge per second against an event-free baseline, overall and per bin."""
    fs = recording.sample_rate_hz
    acc = [e for e in events if e.accepted]
    bin_s = s["timecourse_bin_s"]
    total_q, total_t = 0.0, 0.0
    per_bin: dict[tuple[int, int], float] = {}
    for k in range(recording.n_sweeps):
        raw_full = recording.sweep(k)
        i_lo = int(np.clip(math.floor(lo_s * fs), 0, raw_full.size))
        i_hi = raw_full.size if not math.isfinite(hi_s) else int(np.clip(math.ceil(hi_s * fs), i_lo, raw_full.size))
        if i_hi - i_lo < int(0.05 * fs):
            continue
        raw = np.asarray(raw_full[i_lo:i_hi], float)
        t0 = i_lo / fs
        masked = np.zeros(raw.size, bool)
        for e in acc:
            if e.sweep != k:
                continue
            tau = e.decay_tau_ms if np.isfinite(e.decay_tau_ms) and 0 < e.decay_tau_ms < 200 else s["template_decay_ms"]
            a = int(round((e.onset_s - t0) * fs)) - int(round(0.001 * fs))
            end = e.peak_s if np.isfinite(e.peak_s) else e.onset_s
            b = int(round((end - t0) * fs + 5 * tau * 1e-3 * fs))
            masked[max(a, 0):max(min(b, raw.size), 0)] = True
        out_mask = np.zeros(raw.size, bool)
        for a, b, _ in _merge(left.get(k, [])):
            i0, i1 = (int(np.clip(round((x - t0) * fs), 0, raw.size)) for x in (a, b))
            out_mask[i0:i1] = True
        blk = max(3, int(round(CHARGE_BASELINE_S * fs)))
        centres, levels = [], []
        for a in range(0, raw.size, blk):
            b = min(raw.size, a + blk)
            free = raw[a:b][~(masked[a:b] | out_mask[a:b])]
            if free.size >= 0.2 * (b - a):
                centres.append((a + b) / 2)
                levels.append(float(np.median(free)))
        base = np.interp(np.arange(raw.size), centres, levels) if levels else np.full(raw.size, np.median(raw))
        dev = direction * (raw - base)
        dev[out_mask] = 0.0
        kb = max(1, int(round(CHARGE_BLOCK_S * fs)))
        nblk = raw.size // kb
        if nblk:
            # charge in the events' direction per 10 ms block, and the seconds of it analyzed
            out.arrays[f"charge_{k}"] = (dev[:nblk * kb].reshape(nblk, kb).sum(axis=1) / fs).astype(np.float32)
            out.arrays[f"covered_{k}"] = ((~out_mask[:nblk * kb]).reshape(nblk, kb).sum(axis=1) / fs).astype(np.float32)
            out.arrays[f"charge_t0_{k}"] = np.array([t0, kb / fs])
        q = float(dev.sum() / fs)                       # pA s = pC (voltage clamp); mV s (current clamp)
        tt = float((~out_mask).sum() / fs)
        total_q += q
        total_t += tt
        nb = int(math.ceil(raw.size / (bin_s * fs)))
        for j in range(nb):
            a = int(j * bin_s * fs)
            b = min(raw.size, int((j + 1) * bin_s * fs))
            key = (k, int((t0 + a / fs) // bin_s))
            per_bin[key] = per_bin.get(key, 0.0) + float(dev[a:b].sum() / fs)
    ledger.note(f"charge per second against an event-free baseline (median of the samples no event covers, per "
                f"{CHARGE_BASELINE_S:g} s)", "charge, independent of the detector")
    out.values["charge_rate"] = direction * total_q / total_t if total_t > 0 else NAN
    for row in out.tables.get("timecourse", []):
        q = per_bin.get((row["sweep"], int(row["bin_start_s"] // bin_s)))
        row["charge_rate"] = direction * q / row["analyzed_s"] if q is not None and row["analyzed_s"] > 0 else NAN


def _count_review(out: Result, events: list[Event], rejected) -> None:
    """Rejected in the review, and detections that could not be measured (said, not dropped silently)."""
    def is_rejected(e: Event) -> bool:
        return any(sw == e.sweep and abs(ts - e.onset_s) <= REJECT_TOLERANCE_S for sw, ts in rejected)

    out.values["n_rejected"] = sum(1 for e in events if not e.manual and is_rejected(e))
    unmeasured = sum(1 for e in events if not e.manual and not is_rejected(e) and not np.isfinite(e.amplitude))
    out.values["n_unmeasured"] = unmeasured
    if unmeasured:
        out.notes.append(f"{unmeasured} detection(s) had no clear peak to measure (the trace did not rise above the "
                         "level before them: usually the tail of a larger event or noise) and are not counted; they "
                         "are in the event table, not accepted")


def _event_driven(recording: Recording, s: dict, direction: int, tpl, source: str, out: Result) -> Result:
    """One event per sweep (Clampex's event-driven mode, or events saved from
    Clampfit): nothing to detect, each sweep's event is measured."""
    fs = recording.sample_rate_hz
    rejected, _ = parse_edits(s["edits"])
    recorded = bool(recording.header.get("event_times_recorded"))
    starts = np.asarray(recording.sweep_start_times_s, float)
    events, mids = [], []
    k_s = max(1, int(round(0.0003 * fs)))
    other_larger = 0
    for k in range(recording.n_sweeps):
        y = np.asarray(recording.sweep(k), float)
        y = y - np.median(y[:max(3, y.size // 6)])
        dev = direction * y
        sm = _running_mean(dev, k_s)
        other_larger += int(-sm.min() > sm.max())
        peak = int(np.argmax(sm))
        below = np.flatnonzero(sm[:peak] < 0.1 * sm[peak])
        onset = int(below[-1]) if below.size else max(0, peak - int(0.001 * fs))
        m = measure_event(dev, onset, None, fs, s["template_decay_ms"])
        e = Event(k, onset / fs)
        if m["peak"] >= 0:
            e.peak_s, e.amplitude = m["peak"] / fs, direction * m["amplitude"]
            e.rise_10_90_ms, e.decay_tau_ms, e.decay_r2 = m["rise"], m["tau"], m["r2"]
            e.charge = direction * m["charge"] * (1.0 if recording.is_voltage_clamp else 1e3)
            e.half_width_ms, e.mid_s = m["half"], (m["mid"] / fs if np.isfinite(m["mid"]) else NAN)
        e.accepted = not any(sw == k and abs(ts - e.onset_s) <= REJECT_TOLERANCE_S for sw, ts in rejected) \
            and np.isfinite(e.amplitude)
        events.append(e)
    acc = [e for e in events if e.accepted]
    if recorded and len(acc) >= 2:
        abs_t = np.array([starts[e.sweep] + e.onset_s for e in acc])
        for e, gap in zip(acc[1:], np.diff(abs_t)):
            e.iei_ms = gap * 1e3
        span = float(abs_t[-1] - abs_t[0])
        analyzed = span * len(acc) / max(len(acc) - 1, 1)
    else:
        analyzed = 0.0
    traces = {}
    for e in acc:
        y = np.asarray(recording.sweep(e.sweep), float)
        traces[e.sweep] = (0.0, direction * (y - np.median(y[:max(3, y.size // 6)])))
    mids = [int(round(e.mid_s * fs)) for e in acc if np.isfinite(e.mid_s)]
    window = None
    if mids:
        # a window every event but the earliest and latest tenth fits in
        window = (max(1, int(np.percentile(mids, 10))), max(2, recording.n_samples - int(np.percentile(mids, 90)) - 1))
    _summarize(out, recording, s, events, traces, [], analyzed, direction, tpl,
               "none: one event per sweep (an event-driven file)", fs, window)
    out.values["template"] = "none: one event per sweep (an event-driven file)"
    if not recorded:
        out.values["frequency"] = NAN
        out.values["timing"] = "not in the file (the events are stored back to back)"
        out.notes.append("Event-driven file with sweeps stored back to back: the times between events are not "
                         "recorded, so frequency, intervals and timing are left blank.")
    else:
        out.notes.append("An event-driven file: frequency from the first to the last event's time in the file's "
                         "synch array.")
    _count_review(out, events, rejected)
    out.values["n_added"] = 0
    out.values["n_other_direction"] = other_larger
    if recording.n_sweeps >= 10 and other_larger > 2 * recording.n_sweeps / 3:
        why = out.choices.get("direction", "").split(": ", 1)[-1]
        out.flags.append(Flag("direction?", f"In {other_larger} of {recording.n_sweeps} sweeps the largest deflection "
                              f"points {'down' if direction > 0 else 'up'}, against the direction used ({why}). The "
                              "events may point the other way: a holding set on the amplifier rather than in Clampex "
                              "is not in the file. Set 'Events point' if so.",
                              "the other direction larger in two thirds of the sweeps",
                              ""))
    out.values["charge_rate"] = NAN
    return out


# ------------------------------------------------------------------ one template per experiment

@dataclass
class ExperimentTemplate:
    rise_tau_ms: float = NAN
    decay_tau_ms: float = NAN
    rise_10_90_ms: float = NAN
    decay_r2: float = NAN
    n_events: int = 0
    n_recordings: int = 0
    t_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    average: np.ndarray = field(default_factory=lambda: np.array([]))
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.rise_tau_ms) and np.isfinite(self.decay_tau_ms))

    def origin(self) -> str:
        return (f"the experiment's averaged event ({self.n_events} events in {self.n_recordings} recordings; "
                f"10-90% rise {self.rise_10_90_ms:.2f} ms, decay {self.decay_tau_ms:.1f} ms)")

    def settings(self) -> dict:
        return {"template_rise_ms": round(self.rise_tau_ms, 3), "template_decay_ms": round(self.decay_tau_ms, 2),
                "template_origin": self.origin(), "template_file": ""}


def experiment_template(results) -> ExperimentTemplate:
    """One template for an experiment from its recordings' averaged events
    (``results``: events Results). Each recording's averaged event is scaled
    to a peak of 1 so a large cell does not set the shape, put on a common
    0.02 ms grid and averaged weighted by its number of events; rise tau =
    10-90% / ln 9 and the decay from 80% of the peak. Refused when the pooled
    average is not a clean single decay (r2 >= 0.9, 0.3-200 ms, rise under
    decay): on the lab's files an unguarded refinement fitted 5-750,000 ms
    "decays" and then found nothing."""
    out = ExperimentTemplate()
    parts = []
    for r in results:
        if r is None:
            continue
        curve = next((d for d in r.drawables if d.kind == "curve" and d.label == "averaged event"), None)
        n = int(r.values.get("average_n", 0) or 0)
        if curve is None or n < 5:
            continue
        y = np.asarray(curve.y, float) * (1 if r.values.get("direction") == "up" else -1)
        peak = float(np.max(y))
        if np.isfinite(peak) and peak > 0:
            parts.append((np.asarray(curve.x, float), y / peak, n))
    if not parts:
        out.note = "no recording has an averaged event of at least 5 events"
        return out
    lo, hi = max(t[0] for t, _, _ in parts), min(t[-1] for t, _, _ in parts)
    if hi - lo < 5:
        out.note = "the recordings' averaged events share less than 5 ms"
        return out
    grid = np.arange(lo, hi, 0.02)
    w = np.array([n for _, _, n in parts], float)
    avg = np.average(np.array([np.interp(grid, t, y) for t, y, _ in parts]), axis=0, weights=w)
    out.n_events, out.n_recordings, out.t_ms, out.average = int(w.sum()), len(parts), grid, avg
    rise, tau, r2, rise_tau = shape_kinetics(avg, 1e3 / 0.02)
    out.rise_10_90_ms, out.decay_r2 = rise, r2
    clean = np.isfinite(r2) and r2 >= 0.9 and 0.3 <= tau <= 200.0 and np.isfinite(rise) and 0 < rise < tau
    if not clean:
        out.note = (f"the pooled averaged event is not a clean single decay (r² {r2:.2f}, decay {tau:.1f} ms), so no "
                    "template was made from it")
        return out
    out.decay_tau_ms, out.rise_tau_ms = tau, max(rise_tau, 0.05)
    return out


def template_mismatch(named_results) -> str:
    """A warning when recordings compared together had their events found with
    different templates, methods or thresholds; "" when they agree.
    ``named_results``: (name, events Result) pairs."""
    used: dict[tuple, list[str]] = {}
    for name, r in named_results:
        if r is None or "template" not in r.values:
            continue
        key = (r.values.get("method"), r.values.get("template"), r.values.get("threshold"))
        used.setdefault(key, []).append(name)
    if len(used) <= 1:
        return ""
    parts = [f"{tpl} ({len(names)}: {', '.join(names[:3])}{' ...' if len(names) > 3 else ''})"
             for (_, tpl, _), names in sorted(used.items(), key=lambda kv: -len(kv[1]))]
    return ("Events in these recordings were found with different templates or thresholds: " + "; ".join(parts) +
            ". Detection sensitivity then differs between recordings, a confound in any comparison of event "
            "frequency, amplitude or kinetics. Make one template for the experiment and analyze every recording "
            "with it.")
