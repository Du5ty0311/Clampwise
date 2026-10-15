"""Field potentials and LTP (B7): the fEPSP's slope and amplitude, the fibre
volley, the population spike, paired-pulse facilitation, the volley-to-slope
input-output, and the time course as % of the baseline before the induction.

- **Where it is measured is decided once, on the average** of the sweeps (each
  pulse's own pulse-aligned average), and then used on every sweep: the end of
  the stimulus artifact (evoked's rule: a transient at least four times faster
  than the response, Analysis Decisions 2026-09-24), the fibre volley (the
  most prominent early deflection before the fEPSP peak), the notch where the
  synaptic limb begins, and the peak. Found sweep by sweep the window wandered
  with the noise (on WinLTP recordings the onset jumped between 1.75 and 3.1
  ms), more than the change in slope being measured.
- **The slope** is WinLTP's Maximum Slope (W.W. Anderson, WinLTP 3.01 manual
  4.11.4.1): a least-squares line over a short window slid across the rise in
  0.1 ms steps, the steepest kept; searched from the notch to the peak, so the
  volley's tail is never in it. The 20-80% line of the same rise is the
  alternative; one method per dataset (the brief, B7). Which one for new data
  is an open question.
- **The volley** is read on each sweep at the latency found on the average
  (a volley is small next to one sweep's noise).
- **The population spike**: a narrow trough (at most 5 ms wide, at least 5
  noise SD and 10% of the fEPSP deep) below the line joining its shoulders.
- **Paired pulses**: pulse 2 over pulse 1 on the averages, the ratio of the
  means (Kim & Alger 2001).
- **Input-output**: when the volley changes at least two-fold across sweeps (its
  upper quartile twice its lower: a stimulus-intensity series, not noise), the fEPSP slope against the volley, a straight
  line: the synaptic output per fibre recruited.
- **The time course**: each sweep's slope as % of the mean slope of the
  baseline: the ``baseline_min`` minutes before the induction (decision D4,
  2026-09-30), or the first ``baseline_min`` minutes when no induction is
  entered. The induction is typed (Analyze, or a drug named "induction"),
  never inferred from the files. The last ``late_min`` minutes, % of the
  baseline, is the LTP. A baseline whose slope drifts
  more than ``drift_pct`` per 10 minutes is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .evoked import artifact_end
from .registry import Output, Setting, register
from .result import Drawable, Flag, Result
from .tools import filters, fits
from .tools.kinetics import robust_noise

NAN = float("nan")
METHODS = ("Maximum Slope", "20-80%")
POLARITY = ("auto", "negative", "positive")
SWEEP_END_MARGIN_S = 0.001      # WinLTP's low-pass bends the last samples of a filtered (.FP0) sweep
MAX_SLOPE_STEP_S = 0.0001       # WinLTP slides its slope window in 0.1 ms steps


@dataclass
class Landmarks:
    """Seconds after the pulse, found on the average."""

    artifact_end_s: float
    volley_s: float = NAN
    notch_s: float = NAN
    peak_s: float = NAN
    artifact_seen: bool = False


def _smooth(y: np.ndarray, fs: float) -> np.ndarray:
    return filters.moving_average(y, max(1, int(round(0.0002 * fs))), mode="edge")


def find_landmarks(t: np.ndarray, avg: np.ndarray, fs: float, s: dict, sign: int) -> tuple[Landmarks, str]:
    """The artifact's end, the volley, the notch and the peak on a pulse-aligned
    average (``t`` = 0 at the pulse); a note when the artifact was refused."""
    i_p = int(np.searchsorted(t, 0.0))
    base = float(np.median(avg[:i_p]))
    blank, seen, note = artifact_end(t, avg, 0.0, fs, base, s["blank_ms"] / 1e3, s["response_ms"] / 1e3)
    out = Landmarks(blank, artifact_seen=seen)
    dev = sign * (_smooth(avg, fs) - base)
    r0, r1 = int(np.searchsorted(t, blank)), t.size
    if r1 - r0 < 5:
        return out, "the response window is too short to find the fEPSP"
    peak = r0 + int(np.argmax(dev[r0:r1]))
    out.peak_s = float(t[peak])
    noise = robust_noise(avg[:i_p]) if i_p >= 8 else 0.0
    notch = r0
    fv_end = min(int(np.searchsorted(t, s["volley_ms"] / 1e3)), peak)
    if fv_end > r0 + 2:
        cand, props = find_peaks(dev[r0:fv_end], prominence=max(3.0 * noise, 1e-9))
        if cand.size:
            fv = r0 + int(cand[int(np.argmax(props["prominences"]))])
            out.volley_s = float(t[fv])
            notch = fv + int(np.argmin(dev[fv:peak])) if peak > fv + 1 else fv
    if notch == r0:
        # no volley: the last local maximum before the peak still starts the synaptic limb
        tops, _ = find_peaks(-dev[r0:peak], prominence=0.02 * max(dev[peak], 1e-12))
        if tops.size:
            notch = r0 + int(tops[-1])
    out.notch_s = float(t[notch])
    return out, note


def max_slope(t: np.ndarray, y: np.ndarray, i_from: int, i_to: int, n_points: int, step: int,
              sign: int) -> tuple[float, int, int]:
    """The steepest least-squares slope of ``n_points`` samples within
    [i_from, i_to], only slopes in the response's direction competing
    (units per second, the winning window's first and last index)."""
    n_points = max(3, min(n_points, i_to - i_from + 1))
    starts = np.arange(i_from, i_to - n_points + 2, max(step, 1))
    if starts.size == 0:
        return NAN, i_from, i_from
    idx = starts[:, None] + np.arange(n_points)[None, :]
    x, v = t[idx], y[idx]
    xc = x - x.mean(axis=1, keepdims=True)
    slopes = np.sum(xc * (v - v.mean(axis=1, keepdims=True)), axis=1) / np.sum(xc * xc, axis=1)
    k = int(np.argmax(sign * slopes))
    return float(slopes[k]), int(starts[k]), int(starts[k] + n_points - 1)


def _population_spike(t, y, r0, r1, fs, min_prominence, max_width_s) -> tuple[float, float]:
    """Depth of a narrow trough below the line joining its flanking peaks, and
    its time. The width limit is what tells it from the fEPSP itself."""
    seg = y[r0:r1]
    if seg.size < 5:
        return NAN, NAN
    troughs, props = find_peaks(-seg, prominence=min_prominence, width=(None, max(2.0, max_width_s * fs)))
    if troughs.size == 0:
        return NAN, NAN
    best = int(troughs[int(np.argmax(props["prominences"]))])
    if best <= 0 or best >= seg.size - 1:
        return NAN, NAN
    left, right = int(np.argmax(seg[:best])), best + int(np.argmax(seg[best:]))
    if right <= left:
        return NAN, NAN
    shoulder = seg[left] + (best - left) / (right - left) * (seg[right] - seg[left])
    return float(shoulder - seg[best]), float(t[r0 + best])


def measure(t: np.ndarray, y: np.ndarray, fs: float, lm: Landmarks, s: dict, sign: int) -> dict:
    """One pulse-aligned trace measured at the landmarks: signed values in the
    signal's units (mV), slopes per ms, times in ms after the pulse."""
    i_p = int(np.searchsorted(t, 0.0))
    base, noise = float(np.median(y[:i_p])), robust_noise(y[:i_p])
    row = {k: NAN for k in ("slope", "slope_r2", "amplitude_mv", "onset_ms", "peak_ms", "volley_mv", "volley_ms",
                            "area_mv_ms", "pop_spike_mv", "pop_spike_ms", "snr", "slope_from_ms", "slope_to_ms")}
    if not np.isfinite(lm.notch_s) or not noise > 0:
        return row
    idx = lambda when: int(np.clip(np.searchsorted(t, when), 0, t.size - 1))     # noqa: E731
    dev = sign * (y - base)
    r0, r1, notch = idx(lm.artifact_end_s), t.size, idx(lm.notch_s)
    peak = notch + int(np.argmax(dev[notch:r1])) if r1 > notch else notch
    row.update(amplitude_mv=sign * float(dev[peak]), peak_ms=1e3 * float(t[peak]), snr=float(dev[peak] / noise))
    if np.isfinite(lm.volley_s):
        v, half = idx(lm.volley_s), max(1, int(round(0.0001 * fs)))
        row.update(volley_mv=sign * float(np.mean(dev[max(v - half, r0):v + half + 1])), volley_ms=1e3 * lm.volley_s)
    above = np.flatnonzero(dev[notch:peak] > 3.0 * noise)
    if above.size:
        row["onset_ms"] = 1e3 * float(t[notch + int(above[0])])
    s0, s1 = notch, idx(lm.peak_s)
    a = b = -1
    if s["slope_method"] == "Maximum Slope" and s1 > s0 + 2:
        _, a, b = max_slope(t, y, s0, s1, int(round(s["max_slope_ms"] / 1e3 * fs)) + 1,
                            int(round(MAX_SLOPE_STEP_S * fs)), sign)
    elif s1 > s0 + 2:
        end = max(peak, s0 + 3)
        span = dev[end] - dev[s0]
        seg = dev[s0:end + 1]
        lo, hi = np.flatnonzero(seg >= dev[s0] + 0.2 * span), np.flatnonzero(seg >= dev[s0] + 0.8 * span)
        if span > 0 and lo.size and hi.size and hi[0] > lo[0] + 1:
            a, b = s0 + int(lo[0]), s0 + int(hi[0])
    if b > a + 1:
        fit = fits.linear(t[a:b + 1], y[a:b + 1])
        row.update(slope=fit["slope"] / 1e3, slope_r2=fit.r2, slope_from_ms=1e3 * float(t[a]),
                   slope_to_ms=1e3 * float(t[b]))
    row["area_mv_ms"] = 1e3 * float(np.trapezoid(y[r0:r1] - base, t[r0:r1]))
    if r1 > notch + 4:
        depth, when = _population_spike(t, y, notch, r1, fs, max(5.0 * noise, 0.10 * abs(row["amplitude_mv"])),
                                        0.005)
        row.update(pop_spike_mv=depth, pop_spike_ms=1e3 * when)
    return row


def _segments(recording, rows: list[int], onsets: np.ndarray, pulse: int, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each of ``rows`` (sweeps) around its own onset of ``pulse`` (sweeps x samples),
    from ``baseline_ms`` before to the response window's end, the next pulse
    (less 0.5 ms) or the sweep's end, whichever comes first; and the time axis."""
    fs = recording.sample_rate_hz
    pre = min(int(round(s["baseline_ms"] / 1e3 * fs)), int(onsets[:, pulse].min() * fs))
    n = recording.signal.shape[1]
    room = [n - int(round(SWEEP_END_MARGIN_S * fs)) - int(round(o * fs)) for o in onsets[:, pulse]]
    if pulse + 1 < onsets.shape[1]:
        room += list(((onsets[:, pulse + 1] - onsets[:, pulse] - 0.0005) * fs).astype(int))
    post = int(min([int(round(s["response_ms"] / 1e3 * fs))] + room))
    starts = np.maximum(np.round(onsets[:, pulse] * fs).astype(int) - pre, 0)
    sig = recording.signal
    segs = np.vstack([sig[k, a:a + pre + post] for k, a in zip(rows, starts)]) if post > 0 and pre >= 8 \
        else np.empty((0, 0))
    return segs, (np.arange(pre + post) - pre) / fs


SETTINGS = (
    Setting("slope_method", "Maximum Slope", "Slope method", "Maximum Slope: the steepest line over a short window "
            "slid across the rise. 20-80%: a single line from 20% to 80% of the rise. Use one method for a "
            "whole dataset.", "", "", choices=METHODS),
    Setting("max_slope_ms", 1.5, "Maximum Slope window (ms)", "The length of the line slid across the rise.", "ms", "",
            minimum=0.3),
    Setting("induction_min", NAN, "Induction at (min)", "Time of the tetanus or theta-burst induction, in minutes "
            "from the first sweep. Leave blank if none (a drug named 'induction' is used when present). The "
            "induction time is never inferred from the files.", "min", ""),
    Setting("baseline_min", 10.0, "Baseline: minutes before the induction", "The mean slope over these minutes "
            "defines 100%; with no induction entered, the first this many minutes are used.", "min", "",
            minimum=0.5),
    Setting("late_min", 5.0, "LTP: the last minutes", "LTP is the mean slope over this many final minutes of "
            "the recording, as % of baseline.", "min", "", minimum=0.5),
    Setting("drift_pct", 5.0, "Baseline drift allowed (% per 10 min)", "A linear fit through the baseline "
            "slopes; a change greater than this per 10 minutes is flagged.", "%", "Clampwise", minimum=0.0),
    Setting("polarity", "auto", "fEPSP direction", "Negative in the dendritic layer, positive near the cell "
            "bodies. Auto: the larger deflection after the artifact on the average.", "", "Clampwise",
            choices=POLARITY, advanced=True),
    Setting("baseline_ms", 10.0, "Pre-pulse baseline (ms)", "The reference level for each trace's measurements.", "ms",
            "Clampwise", minimum=1.0, advanced=True),
    Setting("blank_ms", 1.0, "Artifact blank, at least (ms)", "The stimulus artifact duration is measured; this "
            "is the minimum blanking applied.", "ms", "", minimum=0.0, advanced=True),
    Setting("volley_ms", 5.0, "Fibre volley: look up to (ms)", "The fibre volley is searched for between the end of "
            "the artifact and this time, before the fEPSP peak.", "ms", "Clampwise", minimum=1.0, advanced=True),
    Setting("response_ms", 50.0, "Response window (ms)", "How long after the pulse the fEPSP is searched for.", "ms",
            "Clampwise", minimum=5.0, advanced=True),
    Setting("detect_sd", 4.0, "A response: at least (noise SD)", "Pulse 1's fEPSP peak on the average must exceed this "
            "many baseline noise SDs; otherwise it is flagged as no response.", "SD", "Clampwise", minimum=1.0,
            advanced=True),
)

_ROW = (
    ("slope", "The fEPSP slope (the set method), signed: negative for a negative-going fEPSP.", "mV/ms"),
    ("slope_r2", "R squared of the slope's line.", ""),
    ("amplitude_mv", "The fEPSP peak from the pre-pulse baseline, signed.", "mV"),
    ("onset_ms", "Where the fEPSP first passes 3 noise SD after the notch.", "ms"),
    ("peak_ms", "When the fEPSP peaks, after the pulse.", "ms"),
    ("volley_mv", "The fibre volley at the latency found on the average (0.2 ms mean), signed.", "mV"),
    ("volley_ms", "The fibre volley's latency, found on the average.", "ms"),
    ("area_mv_ms", "The area between the trace and the baseline from the artifact's end to the window's end.",
     "mV*ms"),
    ("pop_spike_mv", "The population spike: a narrow trough's depth below the line joining its shoulders.", "mV"),
    ("pop_spike_ms", "When the population spike's trough is, after the pulse.", "ms"),
    ("snr", "The fEPSP peak over the baseline noise SD.", "SD"),
    ("slope_from_ms", "Where the slope's line starts, after the pulse.", "ms"),
    ("slope_to_ms", "Where the slope's line ends, after the pulse.", "ms"),
)

OUTPUTS = (
    Output("fepsp_slope", "Pulse 1's fEPSP slope on the average of the baseline sweeps (the set method).", "mV/ms",
           label="fEPSP slope", headline=True),
    Output("fepsp_amplitude", "Pulse 1's fEPSP peak on the baseline average, signed.", "mV", label="fEPSP amplitude",
           headline=True),
    Output("fiber_volley", "Pulse 1's fibre volley on the baseline average, signed.", "mV", label="Fibre volley",
           headline=True),
    Output("fiber_volley_ms", "The fibre volley's latency after the pulse.", "ms", label="Volley latency"),
    Output("fepsp_onset_ms", "Where the fEPSP first passes 3 noise SD after the notch.", "ms", label="fEPSP onset"),
    Output("fepsp_peak_ms", "When the fEPSP peaks after the pulse.", "ms", label="fEPSP peak time"),
    Output("pop_spike", "Pulse 1's population spike on the baseline average (blank: none).", "mV",
           label="Population spike"),
    Output("snr", "Pulse 1's fEPSP peak over the baseline noise SD.", "SD", label="Signal to noise"),
    Output("slope_per_volley", "The input-output: fEPSP slope against fibre volley across the sweeps, a straight "
           "line's slope; only when the volley changes at least two-fold (its upper quartile twice its lower: an "
           "intensity series).", "1/ms",
           label="Slope per volley (I-O)", headline=True),
    Output("io_r2", "R squared of the input-output line.", "", label="I-O R squared"),
    Output("ppf_slope", "Paired pulses: pulse 2's slope over pulse 1's, on the averages (ratio of the means).",
           "ratio", "Kim & Alger 2001", label="Paired-pulse ratio (slope)", headline=True),
    Output("ppf_amplitude", "Paired pulses: pulse 2's amplitude over pulse 1's, on the averages.", "ratio",
           "Kim & Alger 2001", label="Paired-pulse ratio (amplitude)"),
    Output("isi_ms", "The interval between pulse 1 and pulse 2.", "ms", label="Pulse interval"),
    Output("induction_min", "When the induction was given (typed), minutes from the first sweep.", "min",
           label="Induction at"),
    Output("baseline_slope", "The mean pulse-1 slope over the baseline minutes: 100%.", "mV/ms",
           label="Baseline slope"),
    Output("baseline_from_min", "Where the baseline starts, minutes from the first sweep.", "min",
           label="Baseline from"),
    Output("baseline_to_min", "Where the baseline ends (the induction, when entered).", "min", label="Baseline to"),
    Output("baseline_drift", "A straight line through the baseline's slopes: its change per 10 minutes, % of their "
           "mean.", "%/10 min", label="Baseline drift"),
    Output("late_pct", "The mean pulse-1 slope over the last late_min minutes, % of the baseline: the LTP when an "
           "induction came before it.", "%", label="LTP (last minutes, % baseline)",
           headline=True),
    Output("n_sweeps", "Sweeps measured.", "count", label="Sweeps"),
    Output("pulse", "Which pulse, counted from 1.", level="pulse"),
    *(Output(k, d, *rest, level="pulse") for k, d, *rest in _ROW),
    Output("time_min", "The sweep's start, minutes from the first sweep.", "min", level="sweep"),
    Output("slope_pct", "The sweep's pulse-1 slope, % of the baseline slope.", "%", level="sweep"),
    Output("amplitude_pct", "The sweep's pulse-1 amplitude, % of the baseline amplitude.", "%", level="sweep"),
    Output("ppf", "The sweep's pulse 2 slope over its pulse 1 slope.", "ratio", level="sweep"),
    Output("in_baseline", "Whether the sweep is one of the baseline's.", level="sweep"),
)


def _induction(recording, s) -> float:
    if np.isfinite(s["induction_min"]):
        return float(s["induction_min"])
    d = recording.metadata.drug_on("induction") if recording.metadata is not None else None
    return d.on_s / 60.0 if d is not None else NAN


@register("field", "Field potentials and LTP", version=1, types={"field_evoked"}, settings=SETTINGS,
          outputs=OUTPUTS, modality="field",
          description="The fEPSP's slope and amplitude, the fibre volley, the population spike, paired pulses, "
                      "the volley-to-slope input-output, and the time course as % of the baseline before the "
                      "induction.",
          source="Bliss & Lomo 1973; Kim & Alger 2001 (paired pulses)")
def run(recording, context, s) -> Result:
    out = Result("field", 1)
    st = context.stimulus
    fs = recording.sample_rate_hz
    counts = [st.pulses_in(k) for k in range(recording.n_sweeps)]
    if not any(counts):
        out.notes.append("no stimulus found: nothing to measure")
        return out
    n_pulses = int(np.bincount([c for c in counts if c]).argmax())
    sweeps = [k for k, c in enumerate(counts) if c == n_pulses]
    if len(sweeps) < recording.n_sweeps:
        out.notes.append(f"{recording.n_sweeps - len(sweeps)} sweep(s) with a different number of pulses left out")
    onsets = np.vstack([np.asarray(st.times_s[k][:n_pulses], float) for k in sweeps])
    # when each sweep began, and the baseline (D4)
    starts = np.asarray(recording.sweep_start_times_s, float)
    minutes = (starts[sweeps] - starts[sweeps[0]]) / 60.0 if starts.size == recording.n_sweeps else \
        np.full(len(sweeps), NAN)
    timed = bool(np.all(np.isfinite(minutes)) and (len(sweeps) < 2 or np.all(np.diff(minutes) > 0)))
    induction = _induction(recording, s)
    span = s["baseline_min"]
    if not timed:
        b_from, b_to = NAN, NAN
        in_base = np.ones(len(sweeps), bool)
        out.notes.append("the sweeps' start times are not known: every sweep is the baseline")
    elif np.isfinite(induction):
        b_from, b_to = max(0.0, induction - span), induction
        in_base = (minutes >= b_from) & (minutes < b_to)
    else:
        b_from, b_to = 0.0, span
        in_base = minutes < b_to
        if minutes[-1] >= span:
            out.notes.append(f"no induction entered: the first {span:g} min are the baseline (Analyze, Field "
                             "potentials and LTP, Induction at)")
    if not in_base.any():
        in_base[:1] = True
        out.notes.append("no sweep falls in the baseline minutes: the first sweep is the baseline")
    if in_base.all() and len(sweeps) > 1 and timed and minutes[-1] >= span:
        out.notes.append("every sweep is in the baseline, so the time course is 100% by construction")
    # each pulse: landmarks on the average of every sweep, values on the baseline average
    rows, per_sweep, sign = [], [], 0
    for j in range(n_pulses):
        segs, t = _segments(recording, sweeps, onsets, j, s)
        if segs.shape[0] == 0 or t.size < 10:
            out.notes.append(f"pulse {j + 1}: too close to the sweep's edge or the next pulse to measure")
            break
        avg = segs.mean(axis=0)
        if j == 0:
            sign = {"negative": -1, "positive": 1}.get(s["polarity"], 0)
            if not sign:
                i0 = int(np.searchsorted(t, max(s["blank_ms"] / 1e3, 0.0015)))
                d = _smooth(avg, fs)[i0:] - np.median(avg[t < 0])
                sign = -1 if -d.min() >= d.max() else 1
            out.choose("direction", "negative-going" if sign < 0 else "positive-going",
                       "set" if s["polarity"] != "auto" else "the larger deflection after the artifact on the average")
        lm, note = find_landmarks(t, avg, fs, s, sign)
        if note:
            out.notes.append(f"pulse {j + 1}: {note}")
        row = {"pulse": j + 1, **measure(t, segs[in_base[:segs.shape[0]]].mean(axis=0), fs, lm, s, sign)}
        rows.append(row)
        per_sweep.append([measure(t, segs[k], fs, lm, s, sign) for k in range(segs.shape[0])])
        on = float(np.median(onsets[:, j]))
        out.drawables.append(Drawable("window", f"pulse {j + 1}: artifact", start_s=on, end_s=on + lm.artifact_end_s,
                                      domain="time"))
        if np.isfinite(row["slope_from_ms"]):
            out.drawables.append(Drawable("window", f"pulse {j + 1}: slope ({s['slope_method']})",
                                          start_s=on + row["slope_from_ms"] / 1e3,
                                          end_s=on + row["slope_to_ms"] / 1e3, domain="time"))
        for what, ms in (("fibre volley", row["volley_ms"]), ("fEPSP peak", row["peak_ms"])):
            if np.isfinite(ms):
                at = avg[int(np.clip(np.searchsorted(t, ms / 1e3), 0, t.size - 1))]
                out.drawables.append(Drawable("marker", f"pulse {j + 1}: {what}", x=np.array([on + ms / 1e3]),
                                              y=np.array([at]), domain="time"))
        if j == 0 and lm.artifact_seen and lm.artifact_end_s > s["blank_ms"] / 1e3 + 1e-9:
            out.notes.append(f"the stimulus artifact lasts {lm.artifact_end_s * 1e3:.2f} ms; measuring starts after it")
    if not rows:
        return out
    out.tables["pulses"] = rows
    p1 = rows[0]
    out.values.update({"fepsp_slope": p1["slope"], "fepsp_amplitude": p1["amplitude_mv"],
                       "fiber_volley": p1["volley_mv"], "fiber_volley_ms": p1["volley_ms"],
                       "fepsp_onset_ms": p1["onset_ms"], "fepsp_peak_ms": p1["peak_ms"],
                       "pop_spike": p1["pop_spike_mv"], "snr": p1["snr"], "n_sweeps": len(per_sweep[0])})
    if not p1["snr"] >= s["detect_sd"]:
        out.flags.append(Flag("no response", f"pulse 1's fEPSP peak is {p1['snr']:.1f} noise SD on the baseline "
                              "average", f"at least {s['detect_sd']:g} SD", "Clampwise"))
    out.choose("slope", s["slope_method"], "the steepest 1.5 ms line from the notch to the peak"
               if s["slope_method"] == "Maximum Slope" else "one line from 20% to 80% of the rise after the notch")
    # paired pulses, ratio of the means
    ppf = {"ppf_slope": NAN, "ppf_amplitude": NAN, "isi_ms": NAN}
    if len(rows) >= 2:
        ppf.update(ppf_slope=rows[1]["slope"] / p1["slope"] if p1["slope"] else NAN,
                   ppf_amplitude=rows[1]["amplitude_mv"] / p1["amplitude_mv"] if p1["amplitude_mv"] else NAN,
                   isi_ms=1e3 * float(np.median(onsets[:, 1] - onsets[:, 0])))
    out.values.update(ppf)
    # the time course
    n = len(per_sweep[0])
    slopes = np.array([r["slope"] for r in per_sweep[0]])
    amps = np.array([r["amplitude_mv"] for r in per_sweep[0]])
    base = in_base[:n]
    b_slope = float(np.nanmean(slopes[base])) if np.isfinite(slopes[base]).any() else NAN
    b_amp = float(np.nanmean(amps[base])) if np.isfinite(amps[base]).any() else NAN
    out.values.update({"induction_min": induction, "baseline_slope": b_slope, "baseline_from_min": b_from,
                       "baseline_to_min": b_to, "baseline_drift": NAN, "late_pct": NAN})
    pct = 100.0 * slopes / b_slope if b_slope else np.full(n, NAN)
    sweep_rows = []
    for k in range(n):
        r = {"sweep": sweeps[k], "time_min": float(minutes[k]), "slope": slopes[k], "slope_pct": float(pct[k]),
             "amplitude_mv": amps[k], "amplitude_pct": 100.0 * amps[k] / b_amp if b_amp else NAN,
             "volley_mv": per_sweep[0][k]["volley_mv"], "pop_spike_mv": per_sweep[0][k]["pop_spike_mv"],
             "in_baseline": "yes" if base[k] else ""}
        if len(per_sweep) >= 2:
            r["ppf"] = per_sweep[1][k]["slope"] / slopes[k] if slopes[k] else NAN
        sweep_rows.append(r)
    out.tables["sweeps"] = sweep_rows
    if timed and base.sum() >= 3:
        line = fits.linear(minutes[:n][base], slopes[base], purpose="baseline slopes against time")
        if line.ok and b_slope:
            drift = 100.0 * line["slope"] * 10.0 / b_slope
            out.values["baseline_drift"] = drift
            if abs(drift) > s["drift_pct"]:
                out.flags.append(Flag("baseline drift", f"the baseline's slope changes {drift:+.1f}% per 10 min",
                                      f"{s['drift_pct']:g}% per 10 min", "Clampwise"))
    if timed and n > 1:
        late = minutes[:n] > minutes[n - 1] - s["late_min"]
        late &= ~base if (~base).any() else late
        if late.any() and (~base).any():
            out.values["late_pct"] = float(np.nanmean(pct[late]))
    # the input-output: slope against volley, when the volley changes two-fold
    vol = np.array([r["volley_mv"] for r in per_sweep[0]])
    ok = np.isfinite(vol) & np.isfinite(slopes) & (np.abs(vol) > 0)
    out.values.update({"slope_per_volley": NAN, "io_r2": NAN})
    q1, q3 = np.percentile(np.abs(vol[ok]), (25, 75)) if ok.sum() >= 4 else (NAN, NAN)
    if q1 > 0 and q3 >= 2.0 * q1:
        io = fits.linear(vol[ok], slopes[ok], purpose="input-output: fEPSP slope against fibre volley")
        if io.ok:
            out.values.update({"slope_per_volley": io["slope"], "io_r2": io.r2})
            out.choose("input-output", f"{int(ok.sum())} sweeps, volley {np.abs(vol[ok]).min():.3g} to "
                       f"{np.abs(vol[ok]).max():.3g} mV", io.describe())
    return out
