"""Evoked responses: single pulses, paired pulses and trains, in current and
voltage clamp (B3). Every pulse is measured and shown, not only the first.

The method, and where each choice comes from:

- **Average, then measure** (Clampfit: average the traces, then measure;
  methods sections: "an average of 10-30 consecutive sweeps"). Each pulse is
  measured on the average of the sweeps used. When the pulses move from sweep
  to sweep (an interval series), each pulse is measured on a pulse-aligned
  average: every sweep lined up on its own onset of that pulse, contributing
  only until its own next pulse.
- **Later pulses after earlier tails are taken off.** Pulse k's decay is
  fitted on the trace with the earlier pulses' tails already removed, and its
  extrapolated tail is subtracted before pulse k+1 is measured (Analysis
  Decisions, 2026-09-26). Every pulse is measured against pulse 1's baseline.
  Per sweep the same is done with the average's tail shapes scaled to the
  sweep's own responses (decision D1, 2026-09-30).
- **Paired-pulse ratio as the ratio of the means**, P2 / P1 of the averages
  (Kim & Alger 2001, J Neurosci 21:9608: the mean of per-sweep ratios "is
  biased in favor of high values"). Pn / P1 against pulse number for trains;
  the steady state as the mean of the last pulses over P1.
- **Latency from the median of per-sweep onsets**, jitter their SD (Analysis
  Decisions: averaging jittered responses biases the average's onset early);
  the average's own onset is reported beside it. After light, an onset sooner
  than 1.5 ms is not synaptic and is not measured (2026-09-28).
- **Rise 10-90% and 20-80%; decay tau from 80% of the peak**, the from-peak
  fit beside it, and a flag when fits from 80% and 50% disagree by more than
  25%; see ``tools/kinetics.py``.
- **Failures, potency, CV and 1/CV^2.** A sweep's amplitude is the mean over
  1 ms at the average's peak time (the same time in every sweep, so a failure
  reads near zero); a failure is below 3 SD of the same measurement taken in
  the baseline. 1/CV^2 is corrected for the baseline noise (Faber & Korn 1991,
  Biophys J 60:1288).
- **The artifact is measured, not assumed** (Analysis Decisions, 2026-09-24):
  a transient at least four times faster than the response is over when the
  trace is back within a quarter of its deflection and no longer moving
  faster than the response, held for 0.1 ms; the set blank is a floor; longer
  than 5 ms is refused.
- **A stimulus with nothing time-locked to it is flagged** (2026-09-30): the
  average's largest deflection after pulse 1 is compared with the same after
  sham times in the same sweeps (a shuffled-trigger test); p > 0.05 is
  flagged. Every value is still reported.
- **Current clamp: a sweep with an action potential in a response window is
  left out of the PSP measures** and counted as that pulse's spike
  probability; never mixed silently with subthreshold PSPs.
- **The headline** (the recording's number in the summaries and the cell
  table) is pulse 1 unless chosen otherwise, and says what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..core.recording import Recording
from .registry import EVOKED, Output, Setting, register
from .result import Drawable, Flag, Result
from .tools import filters, fits, kinetics, ledger

NAN = float("nan")
HEADLINES = ("pulse 1", "a chosen pulse", "steady state", "train mean")
SPIKE_DVDT = 20.0            # mV/ms: IPFX's upstroke criterion
SPIKE_HEIGHT = 20.0          # mV above the baseline


# ------------------------------------------------------------------ the artifact

def artifact_end(t: np.ndarray, y: np.ndarray, t0: float, fs: float, baseline: float, floor_s: float,
                 limit_s: float) -> tuple[float, bool, str]:
    """(seconds after ``t0`` where the stimulus artifact is over, whether one
    was seen, a note). The configured blank ``floor_s`` is the shortest it
    can be; a transient longer than 5 ms is refused (the floor is used)."""
    i0 = int(np.searchsorted(t, t0))
    i_lim = min(t.size, i0 + int(limit_s * fs))
    if i_lim - i0 < 10:
        return floor_s, False, ""
    seg = y[i0:i_lim] - baseline
    slope = np.abs(np.gradient(kinetics.smooth(seg, max(1, int(round(0.1e-3 * fs))))) * fs)
    early = max(2, int(round(0.5e-3 * fs)))
    late0 = int(round(2e-3 * fs))
    if late0 >= slope.size - 2:
        return floor_s, False, ""
    fast = float(slope[:early].max())
    phys = float(np.percentile(slope[late0:], 99))
    if not (np.isfinite(fast) and phys > 0 and fast >= 4 * phys):
        return floor_s, False, ""
    deflection = np.abs(seg)
    peak_i = int(np.argmax(deflection[:early]))
    peak_def = float(deflection[peak_i])
    hold = max(1, int(round(0.1e-3 * fs)))
    calm = (deflection < 0.25 * peak_def + 1e-12) & (slope <= phys)
    run = np.convolve(calm.astype(int), np.ones(hold, int), mode="valid") == hold
    after = np.flatnonzero(run[peak_i:])
    if not after.size:
        return floor_s, True, "the stimulus artifact did not settle within the response window"
    end_s = (peak_i + int(after[0])) / fs
    if end_s > 5e-3:
        return floor_s, True, (f"the stimulus artifact seemed to last {end_s * 1e3:.1f} ms, longer than the 5 ms "
                               "allowed; the set blank was used")
    return max(end_s, floor_s), True, ""


# ------------------------------------------------------------------ one response

@dataclass
class Pulse:
    number: int
    time_s: float
    amplitude: float = NAN                  # baseline-subtracted, signed (the measurement)
    amplitude_local: float = NAN            # from the level just before this pulse, on the plain average
    baseline: float = NAN
    noise_sd: float = NAN
    peak_time_s: float = NAN                # relative to the pulse
    latency_avg_s: float = NAN              # onset on the average
    latency_median_s: float = NAN           # median of the per-sweep onsets
    jitter_s: float = NAN                   # SD of the per-sweep onsets
    rise_10_90_s: float = NAN
    rise_20_80_s: float = NAN
    decay_tau_s: float = NAN                # from 80% of the peak
    decay_r2: float = NAN
    decay_tau_from_peak_s: float = NAN
    half_width_s: float = NAN
    integral: float = NAN                   # pC (voltage clamp) or mV*ms (current clamp)
    snr: float = NAN
    detected: bool = False
    success_rate: float = NAN
    potency: float = NAN
    cv: float = NAN
    inv_cv2: float = NAN
    spike_probability: float = NAN
    n_sweeps: int = 0
    tail: tuple | None = None               # (amp, tau, t_start) of the fitted decay, for the next pulse
    flags: list[str] = field(default_factory=list)


def _measure(t: np.ndarray, y: np.ndarray, t0: float, fs: float, s: dict, *, baseline: float | None,
             polarity: int, next_t: float, light: bool) -> Pulse:
    """One response on a trace ``y`` (an average with earlier tails removed)."""
    p = Pulse(0, t0)
    b0, b1 = np.searchsorted(t, [t0 + s["baseline_start_ms"] / 1e3, t0 + s["baseline_end_ms"] / 1e3])
    seg = y[b0:b1]
    if seg.size < 3:
        p.flags.append("the baseline window is too short")
        return p
    p.baseline = float(np.median(seg)) if baseline is None else float(baseline)
    p.noise_sd = kinetics.robust_noise(seg - np.median(seg)) or float(np.std(seg))
    if not np.isfinite(p.noise_sd) or p.noise_sd <= 0:
        p.noise_sd = float(np.std(seg)) or NAN
    blank, seen, note = artifact_end(t, y, t0, fs, p.baseline, s["blank_ms"] / 1e3, s["response_end_ms"] / 1e3)
    if note:
        p.flags.append(note)
    elif seen and blank > s["blank_ms"] / 1e3 + 1e-9:
        p.flags.append(f"stimulus artifact lasts {blank * 1e3:.2f} ms, longer than the {s['blank_ms']:g} ms blank; "
                       "measured after it")
    limit = min(next_t - t0, s["follow_to_ms"] / 1e3) if np.isfinite(next_t) else s["follow_to_ms"] / 1e3
    r0 = int(np.searchsorted(t, t0 + blank))
    r1 = int(np.searchsorted(t, t0 + min(s["response_end_ms"] / 1e3, limit)))
    if r1 - r0 < 3:
        p.flags.append("the response window is too short (the next pulse comes too soon)")
        return p
    ys = kinetics.smooth(y, max(1, int(round(0.2e-3 * fs))))
    dev = polarity * (ys - p.baseline)
    peak = int(r0 + np.argmax(dev[r0:r1]))
    r_lim = int(np.searchsorted(t, t0 + limit)) if s["follow_slow"] else r1
    if s["follow_slow"] and peak >= r1 - max(2, int(fs * 1e-3)) and r_lim > r1:
        wider = int(r0 + np.argmax(dev[r0:r_lim]))
        if wider > peak:
            p.flags.append(f"still rising at the end of the {s['response_end_ms']:g} ms window; its peak was found at "
                           f"{(t[wider] - t0) * 1e3:.0f} ms")
            peak, r1 = wider, max(r1, wider + 1)
    p.amplitude = float(polarity * dev[peak])
    p.peak_time_s = float(t[peak] - t0)
    p.snr = float(dev[peak] / p.noise_sd) if p.noise_sd > 0 else NAN
    p.detected = bool(np.isfinite(p.snr) and p.snr >= s["detect_sd"])
    onset = kinetics.find_onset(dev, r0, peak + 1, 3.0 * p.noise_sd, max(1, int(round(0.5e-3 * fs))))
    if onset is not None and onset < peak:
        lat = float(t[onset] - t0)
        if light and lat < s["light_min_latency_ms"] / 1e3:
            p.flags.append(f"onset {lat * 1e3:.2f} ms after the light, sooner than a synapse can answer "
                           f"({s['light_min_latency_ms']:g} ms): the light itself or noise; latency not measured")
        else:
            p.latency_avg_s = lat
    peak_dev = dev[peak]
    end, returned = r1, True
    if s["follow_slow"] and peak_dev > 0 and r_lim > peak + 1:
        sm = kinetics.smooth(dev, max(1, int(round(1e-3 * fs))))
        back = np.flatnonzero(sm[peak:r_lim] <= max(p.noise_sd, 0.05 * peak_dev))
        end = peak + int(back[0]) if back.size else r_lim
        returned = bool(back.size)
        if not returned:
            p.flags.append(f"not back to baseline by {(t[min(r_lim, t.size) - 1] - t0) * 1e3:.0f} ms (the next pulse "
                           "or the limit): charge and decay stop there")
    if p.detected and peak_dev > 0:
        p.rise_10_90_s = kinetics.rise_time(t, dev, r0, peak, 0.1, 0.9)
        p.rise_20_80_s = kinetics.rise_time(t, dev, r0, peak, 0.2, 0.8)
        p.half_width_s = kinetics.half_width(t, dev, r0, peak, max(end, r1))
    decay_stop = max(int(np.searchsorted(t, t0 + min(s["decay_window_ms"] / 1e3, limit))), end)
    decay_stop = min(decay_stop, r_lim if s["follow_slow"] else decay_stop, t.size)
    if p.detected and decay_stop > peak + 8:
        p.decay_tau_from_peak_s = _decay(t, ys, peak, decay_stop, p.baseline)[0]
        start = kinetics.decay_start(dev, peak, decay_stop, fs)
        tau, r2, amp = _decay(t, ys, start, decay_stop, p.baseline, purpose="evoked decay tau, from 80% of the peak")
        p.decay_tau_s, p.decay_r2 = tau, r2
        if np.isfinite(tau):
            p.tail = (amp, tau, float(t[start]))
            start50 = kinetics.decay_start(dev, peak, decay_stop, fs, fraction=0.5)
            if 0 < tau < 9.9 and start50 > start:
                b = _decay(t, ys, start50, decay_stop, p.baseline)[0]
                if np.isfinite(b) and abs(b - tau) / tau > kinetics.DECAY_DISAGREEMENT:
                    p.flags.append(f"the decay is not one exponential: tau from 80% of the peak ({tau * 1e3:.1f} ms) "
                                   f"and from 50% ({b * 1e3:.1f} ms) differ by {abs(b - tau) / tau:.0%}")
            if not np.isfinite(r2) or r2 < 0.8:
                p.flags.append(f"decay fit poor (r2 = {r2:.2f}); tau is unreliable")
            if tau >= 9.9 or tau < 2.0 / fs:
                p.flags.append("decay fit stopped at its limit; tau is not a measurement")
    c0 = int(np.searchsorted(t, t0 + blank))
    c1 = max(int(np.searchsorted(t, t0 + min(s["charge_end_ms"] / 1e3, limit))), end if s["follow_slow"] else 0)
    c1 = min(c1, t.size)
    if c1 - c0 > 2:
        p.integral = float(np.trapezoid(y[c0:c1] - p.baseline, t[c0:c1]) * 1e3)    # pA*ms (fC), turned into pC below; mV*ms as is
    return p


def _decay(t, y, i0, i1, baseline, purpose: str = "") -> tuple[float, float, float]:
    """(tau_s, r2, amplitude) of amp exp(-(t - t0)/tau) + offset over [i0, i1)."""
    tt, yy = t[i0:i1], y[i0:i1]
    if tt.size < 8:
        return NAN, NAN, NAN
    t0 = tt[0]
    amp0 = yy[0] - baseline
    if amp0 == 0:
        return NAN, NAN, NAN
    target = baseline + amp0 / np.e
    crossed = np.flatnonzero((yy - target) * np.sign(amp0) <= 0)
    tau0 = max(float(tt[crossed[0]] - t0) if crossed.size else float(tt[-1] - t0) / 3.0, 1e-4)

    def model(x, a, tau, off):
        return a * np.exp(-(x - t0) / tau) + off

    def jac(x, a, tau, off):
        e = np.exp(-(x - t0) / tau)
        return np.column_stack([e, a * e * (x - t0) / tau ** 2, np.ones_like(x)])

    fit = fits.nonlinear(model, tt, yy, [amp0, tau0, baseline], ("amp", "tau", "offset"), label="exp1",
                         bounds=([-np.inf, 1e-5, -np.inf], [np.inf, 10.0, np.inf]), maxfev=5000, jac=jac,
                         clean=False, purpose=purpose)
    if not fit.ok:
        return NAN, NAN, NAN
    return fit["tau"], fit.r2, fit["amp"]


def _tail(t: np.ndarray, pulse: Pulse, shift_s: float = 0.0) -> np.ndarray:
    """The fitted decay of ``pulse`` extrapolated over ``t``, from where its
    decay fit begins (shifted by ``shift_s`` for a sweep whose pulse comes at
    another time), zero before it: what to take off before measuring the next
    pulse. Before the decay starts the exponential does not describe the
    trace (the response is still rising or at its peak), so nothing is taken
    off there."""
    out = np.zeros_like(t)
    if pulse.tail is None:
        return out
    amp, tau, ts = pulse.tail
    ts = ts + shift_s
    m = t >= ts
    x = np.clip((t[m] - ts) / tau, 0, 745)                 # exp(-746) is 0 in float64
    out[m] = amp * np.exp(-x)
    return out


# ------------------------------------------------------------------ spikes in current clamp

def spike_onset(t: np.ndarray, y: np.ndarray, fs: float, t0: float, t1: float, baseline: float) -> float:
    """The time (s) of the first action potential between ``t0`` and ``t1`` (where
    dV/dt first crosses 20 mV/ms, on IPFX's 10 kHz Bessel-filtered derivative,
    on a rise of 20 mV or more above the baseline), or NaN for none."""
    a, b = np.searchsorted(t, [t0, t1])
    if b - a < 4:
        return NAN
    lead = min(a, 10)
    seg = y[a - lead:b]
    coefficient = 10.0 / (fs / 2000.0)
    if 0 < coefficient < 0.95:
        src = filters.bessel_low_pass_normalized(seg, 4, coefficient, purpose="dV/dt for spike detection")
    else:
        # at 20 kHz or below IPFX's 10 kHz filter sits at or above Nyquist: the raw derivative is used, as IPFX does
        ledger.note("dV/dt on the unfiltered trace (a 10 kHz filter is at or above Nyquist at this rate)",
                    "spike detection")
        src = seg
    dvdt = np.diff(src) * fs / 1e3
    if not (dvdt.max() >= SPIKE_DVDT and (y[a:b].max() - baseline) >= SPIKE_HEIGHT):
        return NAN
    cross = np.flatnonzero(dvdt[lead:] >= SPIKE_DVDT)
    return float(t[a + int(cross[0])]) if cross.size else NAN


def spiked(t: np.ndarray, y: np.ndarray, fs: float, t0: float, t1: float, baseline: float) -> bool:
    return bool(np.isfinite(spike_onset(t, y, fs, t0, t1, baseline)))


# ------------------------------------------------------------------ the time-locked test

def time_locked_p(t: np.ndarray, avg_rows: np.ndarray, t0: float, fs: float, s: dict, stim_times: np.ndarray,
                  n_sham: int = 60, avoid: list[tuple[float, float]] = ()) -> tuple[float, int]:
    """p of the average's largest deflection after the first pulse against the
    same at sham times, in the same sweeps (a shuffled-trigger surrogate);
    (NaN, n) when the sweeps are too short for 40 shams."""
    win = s["response_end_ms"] / 1e3
    base = (s["baseline_start_ms"] / 1e3, s["baseline_end_ms"] / 1e3)
    avg = avg_rows.mean(axis=0)

    def deflection(at: float) -> float:
        b0, b1 = np.searchsorted(t, [at + base[0], at + base[1]])
        r0, r1 = np.searchsorted(t, [at + s["blank_ms"] / 1e3, at + win])
        if b1 - b0 < 3 or r1 - r0 < 3:
            return NAN
        return float(np.max(np.abs(avg[r0:r1] - np.median(avg[b0:b1]))))

    real = deflection(t0)
    span_lo, span_hi = -base[0] + 1e-3, t[-1] - win - 1e-3
    candidates = np.linspace(span_lo, span_hi, 400)
    # no sham where a measurement would take in a stimulus, a command change or a
    # test pulse (such an artifact is not "no response")
    keep = [c for c in candidates if np.all(np.abs(stim_times - c) > win + abs(base[0]))
            and not any(a <= c + win and c + base[0] <= b for a, b in avoid)]
    if len(keep) < 40:
        return NAN, len(keep)
    shams = np.array(keep)[np.linspace(0, len(keep) - 1, min(n_sham, len(keep))).astype(int)]
    values = np.array([deflection(c) for c in shams])
    values = values[np.isfinite(values)]
    if not np.isfinite(real) or values.size < 40:
        return NAN, int(values.size)
    return float((1 + np.sum(values >= real)) / (1 + values.size)), int(values.size)


FLAG_WORDS = (
    ("limit", "fit at its limit"), ("not one exponential", "two-component decay"), ("fit poor", "poor decay fit"),
    ("sooner than a synapse", "onset too soon"), ("not back to baseline", "not back to baseline"),
    ("artifact", "long artifact"), ("still rising", "slow peak"), ("exceed what the trace", "tail too large"),
    ("too short", "window too short"), ("every sweep had fired", "all sweeps fired"),
    ("no decay to take off", "no tail"), ("tail fit failed", "tail fit failed"), ("comes before", "pulses too close"),
)


def short_code(text: str) -> str:
    """The badge's word for a flag (the full text is its tooltip)."""
    for phrase, word in FLAG_WORDS:
        if phrase in text:
            return word
    return "note"


def _pulse_row(p: Pulse, p1_amplitude: float, vc: bool, spike_at: np.ndarray) -> dict:
    """One row of the pulse table; every column present, NaN where not measured."""
    at = spike_at[np.isfinite(spike_at)] if not vc else np.array([])
    finite = np.isfinite(p.amplitude) and np.isfinite(p1_amplitude) and p1_amplitude != 0
    return {
        "pulse": p.number, "time_s": p.time_s, "amplitude": p.amplitude, "amplitude_local": p.amplitude_local,
        "pn_p1": float(p.amplitude / p1_amplitude) if finite else NAN,
        "latency_median_ms": p.latency_median_s * 1e3, "jitter_ms": p.jitter_s * 1e3,
        "latency_average_ms": p.latency_avg_s * 1e3, "time_to_peak_ms": p.peak_time_s * 1e3,
        "rise_10_90_ms": p.rise_10_90_s * 1e3, "rise_20_80_ms": p.rise_20_80_s * 1e3,
        "decay_tau_ms": p.decay_tau_s * 1e3, "decay_r2": p.decay_r2,
        "decay_tau_from_peak_ms": p.decay_tau_from_peak_s * 1e3, "half_width_ms": p.half_width_s * 1e3,
        "integral": p.integral / (1e3 if vc else 1.0), "snr": p.snr, "detected": p.detected,
        "success_rate": p.success_rate, "potency": p.potency, "cv": p.cv, "inv_cv2": p.inv_cv2,
        "spike_probability": p.spike_probability,
        "spike_latency_ms": float(np.median(at)) * 1e3 if at.size else NAN,
        "spike_jitter_ms": float(np.std(at, ddof=1)) * 1e3 if at.size > 1 else NAN,
        "n_sweeps": p.n_sweeps, "flags": " | ".join(p.flags)}


# ------------------------------------------------------------------ the analysis

SETTINGS = (
    Setting("baseline_start_ms", -20.0, "Baseline from (ms)", "Start of the baseline, before each pulse.", "ms",
            "the convention: a 20 ms baseline just before the stimulus"),
    Setting("baseline_end_ms", -1.0, "Baseline to (ms)", "End of the baseline, before each pulse.", "ms"),
    Setting("response_end_ms", 50.0, "Look for the peak until (ms)", "How long after each pulse the peak is "
            "looked for (never past the next pulse).", "ms"),
    Setting("headline", "pulse 1", "The recording's number", "Which pulse or summary stands for the recording in "
            "the summaries and the cell table; it is printed with every result.", choices=HEADLINES,
            source="pulse 1: the only response not shaped by the pulses before it"),
    Setting("headline_pulse", 2, "Chosen pulse", "Used when the recording's number is a chosen pulse.",
            minimum=1),
    Setting("steady_state_pulses", 5, "Steady state: last pulses", "The steady state of a train is the mean of "
            "this many last pulses, over pulse 1.", minimum=1),
    Setting("exclude_spiking", True, "Leave out sweeps with a spike (current clamp)", "A PSP with an action "
            "potential on it is not a PSP measurement; those sweeps are counted as the pulse's spike probability."),
    Setting("charge_end_ms", 50.0, "Charge until (ms)", "Charge (voltage clamp) or area (current clamp) is "
            "integrated from the end of the artifact to here, or to where a slow response ends.", "ms",
            advanced=True),
    Setting("decay_window_ms", 60.0, "Decay fit until (ms)", "The decay is fitted from 80% of the peak to here, "
            "or to where a slow response ends.", "ms", advanced=True),
    Setting("blank_ms", 0.3, "Artifact blank, at least (ms)", "Skipped after each pulse; a longer stimulus "
            "artifact is measured and skipped too.", "ms", advanced=True, minimum=0.0),
    Setting("follow_slow", True, "Follow slow responses", "A peak at the window's end is looked for further, and "
            "charge and decay run until the response ends (never past the next pulse).", advanced=True),
    Setting("follow_to_ms", 500.0, "Follow at most (ms)", "", "ms", advanced=True),
    Setting("detect_sd", 4.0, "Response above (noise SD)", "A response smaller than this on the average is "
            "reported, flagged as not detected.", advanced=True),
    Setting("failure_sd", 3.0, "Failure below (noise SD)", "A sweep's response smaller than this is a failure.",
            source="the common 2-3 x noise threshold", advanced=True),
    Setting("mean_window_ms", 1.0, "Sweep amplitude: mean over (ms)", "Each sweep's amplitude is the mean over "
            "this window at the average's peak time.", "ms", advanced=True),
    Setting("light_min_latency_ms", 1.5, "After light, onsets sooner than (ms)", "are the light itself or noise, "
            "not a synapse; 0 for direct photocurrents.", "ms", advanced=True),
    Setting("polarity", "auto", "Response direction", "From pulse 1 of the average, or set.",
            choices=("auto", "up", "down"), advanced=True),
)

OUTPUTS = (
    Output("amplitude", "The recording's headline response (see headline_is), baseline-subtracted, measured on the "
           "average of the sweeps used.", "pA or mV"),
    Output("headline_is", "A description of the headline: which pulse or summary, and the number of sweeps "
           "averaged.", ""),
    Output("p1_amplitude", "Pulse 1: the response to the first stimulus of each sweep, on the average of the sweeps "
           "used.", "pA or mV"),
    Output("p1_latency_median", "Median of the per-sweep onsets of pulse 1 (onset: the first sustained crossing of "
           "3 noise SD).", "ms"),
    Output("p1_jitter", "SD of the per-sweep onsets of pulse 1.", "ms"),
    Output("p1_latency_average", "Onset of pulse 1 on the average trace (biased early when jitter is high).", "ms"),
    Output("p1_rise_10_90", kinetics.DEFINITIONS["rise_10_90"], "ms", ""),
    Output("p1_rise_20_80", kinetics.DEFINITIONS["rise_20_80"], "ms"),
    Output("p1_decay_tau", kinetics.DEFINITIONS["decay_tau"], "ms"),
    Output("p1_decay_tau_from_peak", "Decay tau fitted from the peak.", "ms", ""),
    Output("p1_half_width", kinetics.DEFINITIONS["half_width"], "ms"),
    Output("p1_integral", "Charge (voltage clamp, pC) or area (current clamp, mV*ms) of pulse 1, from the end of the "
           "artifact to the end of the charge window or of a slow response.", "pC or mV*ms"),
    Output("p1_success_rate", "Fraction of sweeps whose pulse-1 response is above the failure threshold.", "fraction",
           "failures below 3 SD of the noise"),
    Output("p1_potency", "Mean pulse-1 amplitude of the successes.", "pA or mV"),
    Output("p1_cv", "Coefficient of variation of the per-sweep pulse-1 amplitudes.", "ratio",
           "Faber & Korn 1991"),
    Output("p1_inv_cv2", "1/CV^2 of pulse 1, corrected for the baseline noise: mean^2 / (variance - noise "
           "variance).", "ratio", "Faber & Korn 1991"),
    Output("ppr", "Paired-pulse ratio: pulse 2 / pulse 1 on the averages, pulse 2 measured after pulse 1's tail is "
           "taken off (the ratio of means).", "ratio", "Kim & Alger 2001"),
    Output("ppr_raw", "Pulse 2 against its own pre-pulse baseline, over pulse 1 (no tail correction).", "ratio"),
    Output("steady_state_ratio", "Mean of the last pulses (set) over pulse 1.", "ratio"),
    Output("n_pulses", "Pulses per sweep.", "count"),
    Output("units", "The units of the amplitudes: the recorded signal's.", "", label="Amplitude units"),
    Output("integral_units", "The units of the integral: pC in voltage clamp, mV*ms in current clamp.", "",
           label="Integral units"),
    Output("n_sweeps", "Sweeps used (after sweeps with a spike in a response window are left out).", "count"),
    Output("time_locked_p", "p of pulse 1's response against sham times in the same sweeps; above 0.05: nothing "
           "time-locked to the stimulus.", "p", "shuffled-trigger surrogate"),
    Output("pulse", "One pulse on the average (pulse table).", "", level="pulse"),
    Output("amplitude_local", "A later pulse's peak against the level just before it on the plain average (no tail "
           "model): the rise the trace shows. The corrected amplitude should not be smaller while earlier "
           "responses decay; when it is, the pulse is flagged.", "pA or mV", level="pulse"),
    Output("spike_probability", "Current clamp: fraction of sweeps that fired an action potential on this pulse "
           "(fidelity).", "fraction", level="pulse"),
    Output("spike_latency", "Current clamp: median time from the pulse to its action potential (dV/dt crossing "
           "20 mV/ms), over the sweeps that fired.", "ms", level="pulse"),
    Output("spike_jitter", "Current clamp: SD of that spike latency.", "ms", level="pulse"),
    Output("sweep_amplitude", "One sweep's response to one pulse: the mean over 1 ms at the average's peak time, "
           "earlier tails taken off, against the sweep's own pulse-1 baseline.", "pA or mV", level="sweep"),
)


LABELS = {
    "amplitude": ("The recording's number", True), "headline_is": ("What that number is", True),
    "p1_amplitude": ("Pulse 1 amplitude", False), "p1_latency_median": ("Pulse 1 latency (median of sweeps)", True),
    "p1_jitter": ("Pulse 1 jitter (SD of sweep onsets)", False),
    "p1_latency_average": ("Pulse 1 latency on the average", False),
    "p1_rise_10_90": ("Pulse 1 rise, 10-90%", True), "p1_rise_20_80": ("Pulse 1 rise, 20-80%", False),
    "p1_decay_tau": ("Pulse 1 decay tau (from 80% of the peak)", True),
    "p1_decay_tau_from_peak": ("Pulse 1 decay tau (from the peak)", False),
    "p1_half_width": ("Pulse 1 half-width", False), "p1_integral": ("Pulse 1 charge or area", False),
    "p1_success_rate": ("Pulse 1 success rate", True), "p1_potency": ("Pulse 1 potency", False),
    "p1_cv": ("Pulse 1 CV", False), "p1_inv_cv2": ("Pulse 1 1/CV\u00b2", False),
    "ppr": ("Paired-pulse ratio (P2/P1, tail taken off)", True), "ppr_raw": ("Paired-pulse ratio, raw", False),
    "steady_state_ratio": ("Steady state / pulse 1", False), "n_pulses": ("Pulses per sweep", False),
    "n_sweeps": ("Sweeps averaged", True), "time_locked_p": ("Time-locked to the stimulus (p)", False),
}
OUTPUTS = tuple(replace(o, label=LABELS[o.key][0], headline=LABELS[o.key][1]) if o.key in LABELS else o
                for o in OUTPUTS)


@register("evoked", "Evoked responses, every pulse", version=2, types=EVOKED, settings=SETTINGS, outputs=OUTPUTS,
          description="Single pulses, paired pulses and trains: each pulse's amplitude, latency, kinetics, charge "
                      "and reliability, measured on the average and sweep by sweep.",
          source="Kim & Alger 2001 (PPR); Faber & Korn 1991 (CV)", requires=("baseline",))
def run(recording: Recording, context, s) -> Result:
    out = Result("evoked", 1)
    st = context.stimulus
    fs = recording.sample_rate_hz
    t = recording.time_s
    vc = recording.is_voltage_clamp
    units = recording.signal_units
    light = st.kind == "light"
    counts = [int(x.size) for x in st.times_s]
    if not counts or max(counts) == 0:
        out.notes.append("no stimulus in any sweep")
        return out
    n_pulses = int(np.bincount([c for c in counts if c]).argmax())     # the commonest count among sweeps with any
    sweeps = [i for i, c in enumerate(counts) if c == n_pulses]
    if len(sweeps) < recording.n_sweeps:
        out.notes.append(f"{recording.n_sweeps - len(sweeps)} sweep(s) with a different number of pulses left out "
                         f"(the recording's {n_pulses}-pulse sweeps are measured)")
    selection = context.shared.get("sweep_indices")
    if selection:
        sweeps = [i for i in sweeps if i in set(selection)]
    if not sweeps:
        out.notes.append("no chosen sweep carries the stimulus")
        return out
    onsets = np.vstack([st.times_s[i][:n_pulses] for i in sweeps])          # (sweeps, pulses)
    moving = bool(np.any(np.ptp(onsets, axis=0) > 1.5 / fs))
    data = recording.read(recording.signal_channel, sweeps)
    base_pre = s["baseline_start_ms"] / 1e3

    # ---- current clamp: which sweeps fire on which pulse
    # pulse 1's window reaches 10 ms before its baseline: a spike just before
    # still pulls the baseline through its afterhyperpolarization
    spike_table = np.zeros((len(sweeps), n_pulses), dtype=bool)
    spike_at = np.full((len(sweeps), n_pulses), np.nan)          # spike latency from its pulse, s
    if not vc:
        for k in range(len(sweeps)):
            row = data[k]
            b0, b1 = np.searchsorted(t, [onsets[k, 0] + base_pre, onsets[k, 0] + s["baseline_end_ms"] / 1e3])
            base = float(np.median(row[b0:b1])) if b1 > b0 else float(np.median(row[:100]))
            for j in range(n_pulses):
                start = onsets[k, 0] + base_pre - 0.010 if j == 0 else onsets[k, j]
                nxt = onsets[k, j + 1] if j + 1 < n_pulses else np.inf
                # as far as this pulse's measurement can reach: its window, or a slow
                # response followed to the next pulse (the last pulse: the limit)
                reach = s["follow_to_ms"] if s["follow_slow"] else s["response_end_ms"]
                at = spike_onset(t, row, fs, start, min(nxt, onsets[k, j] + reach / 1e3), base)
                spike_table[k, j] = bool(np.isfinite(at))
                spike_at[k, j] = at - onsets[k, j] if np.isfinite(at) else NAN
    # a sweep leaves the PSP measures from the first pulse it fires on
    if not vc and s["exclude_spiking"]:
        fired_by = np.cumsum(spike_table, axis=1) > 0
        member = ~fired_by
    else:
        member = np.ones((len(sweeps), n_pulses), dtype=bool)
    if not vc and spike_table.any():
        n_fired = int(spike_table.any(axis=1).sum())
        out.flags.append(Flag("spikes", f"{n_fired} of {len(sweeps)} sweep(s) fired an action potential on a pulse; "
                              + ("each is left out of the PSP measures from that pulse on and counted as that "
                                 "pulse's spike probability" if s["exclude_spiking"] else
                                 "they are kept (Analyze: leave out sweeps with a spike is off)"), severity="note"))
    out.values["n_pulses"] = n_pulses
    out.values["n_sweeps"] = int(member[:, 0].sum())
    if not member[:, 0].any():
        out.notes.append("every sweep fired on pulse 1: no PSP is measured; the spike probability of each pulse is "
                         "in the pulse table")
        out.tables["pulses"] = [_pulse_row(Pulse(j + 1, float(np.median(onsets[:, j])),
                                                 spike_probability=float(spike_table[:, j].mean())), NAN, vc,
                                           spike_at[:, j]) for j in range(n_pulses)]
        for key in ("amplitude", "p1_amplitude", "ppr"):
            out.values[key] = NAN
        return out
    used, rows, ons = sweeps, data, onsets

    fixed = ons[0]
    ledger.note("average of the sweeps aligned on the stimulus" +
                (" (pulse-aligned per pulse: the pulses move between sweeps)" if moving else ""),
                "evoked responses")
    out.choose("sweeps averaged", f"{int(member[:, 0].sum())} of {recording.n_sweeps} for pulse 1",
               "every sweep with the recording's number of pulses" +
               (", and from the first pulse a sweep fires on, it is left out" if spike_table.any() else ""))
    pol_setting = s["polarity"]
    polarity = 0 if pol_setting == "auto" else (1 if pol_setting == "up" else -1)
    work = rows.astype(float).copy()   # each sweep with its earlier tails taken off (scaled to it, D1)
    avg_tails = np.zeros(t.size)       # the earlier tails on the average, as fitted there

    def sweep_base(k: int) -> float:
        b0, b1 = np.searchsorted(t, [ons[k, 0] + base_pre, ons[k, 0] + s["baseline_end_ms"] / 1e3])
        return float(np.median(rows[k, b0:b1]))

    bases = np.array([sweep_base(k) for k in range(len(used))])

    def aligned(j: int, members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(t_rel, average) of pulse j aligned on each member sweep's own onset of
        it, each contributing until its own next pulse (its earlier tails already
        taken off in ``work``)."""
        import warnings

        pre = int(round(-base_pre * fs)) + 2
        post = int(round(s["follow_to_ms"] / 1e3 * fs))
        stack = np.full((int(members.sum()), pre + post), np.nan)
        for row_i, k in enumerate(np.flatnonzero(members)):
            i0 = int(round(ons[k, j] * fs))
            stop = int(round(ons[k, j + 1] * fs)) if j + 1 < n_pulses else rows.shape[1]
            a, b = max(i0 - pre, 0), min(stop, i0 + post, rows.shape[1])
            stack[row_i, (a - (i0 - pre)):(b - (i0 - pre))] = work[k, a:b]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # columns no sweep reaches: dropped below
            mean = np.nanmean(stack, axis=0)
        trel = (np.arange(pre + post) - pre) / fs
        good = np.isfinite(mean)
        return trel[good], mean[good]

    def aligned_plain(j: int, members: np.ndarray) -> np.ndarray:
        saved = work.copy()
        work[:] = rows
        try:
            return aligned(j, members)[1]
        finally:
            work[:] = saved

    half = max(1, int(round(s["mean_window_ms"] / 2e3 * fs)))
    per_sweep = np.full((len(used), n_pulses), np.nan)
    pulses: list[Pulse] = []
    p1_base = None
    for j in range(n_pulses):
        members = member[:, j]
        nxt_rel = float(np.median(ons[:, j + 1] - ons[:, j])) if j + 1 < n_pulses else np.inf
        if not members.any():
            p = Pulse(j + 1, float(np.median(ons[:, j])))
            p.flags.append("every sweep had fired by this pulse: not measured")
            pulses.append(p)
            continue
        if moving:
            tt, yy = aligned(j, members)
            t0, nxt = 0.0, nxt_rel
        else:
            tt, t0 = t, float(fixed[j])
            yy = work[members].mean(axis=0) if j == 0 else rows[members].mean(axis=0) - avg_tails
            nxt = float(fixed[j + 1]) if j + 1 < n_pulses else np.inf
        if j == 0 and polarity == 0:
            up = _measure(tt, yy, t0, fs, s, baseline=None, polarity=1, next_t=nxt, light=light)
            down = _measure(tt, yy, t0, fs, s, baseline=None, polarity=-1, next_t=nxt, light=light)
            polarity = 1 if abs(up.amplitude) >= abs(down.amplitude) else -1
            out.choose("response direction", "up" if polarity > 0 else "down",
                       "the larger deflection of pulse 1 on the average")
        p = _measure(tt, yy, t0, fs, s, baseline=None if (j == 0 or moving) else p1_base, polarity=polarity,
                     next_t=nxt, light=light)
        p.number, p.time_s = j + 1, float(np.median(ons[:, j]))
        if j == 0:
            p1_base = p.baseline
            p.amplitude_local = p.amplitude
        elif np.isfinite(p.peak_time_s):
            # the same peak against the level just before this pulse, on the plain
            # average: what the trace shows without any tail model
            plain_t, plain = (tt, aligned_plain(j, members)) if moving else (t, rows[members].mean(axis=0))
            pre0, pre1 = np.searchsorted(plain_t, [t0 - 0.002, t0 - 0.0002])
            pk = int(np.searchsorted(plain_t, t0 + p.peak_time_s))
            if pre1 > pre0 and pk < plain.size:
                local_base = float(np.median(plain[pre0:pre1]))
                p.amplitude_local = float(np.mean(plain[max(pk - 2, 0):pk + 3]) - local_base)
                # with earlier responses still decaying, the corrected amplitude is at
                # least the rise from the trough; less means the tails taken off were too big
                if polarity * p.amplitude < polarity * p.amplitude_local - 2 * (p.noise_sd or 0.0):
                    p.flags.append(f"the earlier tails taken off ({p.amplitude_local - p.amplitude:+.2f} {units} at its "
                                   f"peak) exceed what the trace allows: the rise from the trough just before it is "
                                   f"{p.amplitude_local:.2f} {units}, more than the corrected {p.amplitude:.2f}; read "
                                   "the corrected amplitude as a lower estimate (slow responses summing over a train)")
        pulses.append(p)
        # each sweep's response to this pulse (its earlier tails already off, against its own pulse-1 baseline)
        if np.isfinite(p.peak_time_s):
            for k in np.flatnonzero(members):
                c = int(round((ons[k, j] + p.peak_time_s) * fs))
                seg = work[k, max(c - half, 0):c + half + 1]
                per_sweep[k, j] = float(seg.mean() - bases[k]) if seg.size else NAN
        # this pulse's tail, taken off before the next: a decay to the baseline,
        # fitted from 80% of the peak through the next stimulus plus 80% of the
        # latency (where the next response begins)
        if j + 1 < n_pulses and np.isfinite(p.peak_time_s) and np.isfinite(p.amplitude) and p.amplitude != 0:
            base = p.baseline if (j == 0 or moving) else p1_base
            lag = 0.8 * p.latency_avg_s if np.isfinite(p.latency_avg_s) else 0.0
            ip, ie = np.searchsorted(tt, [t0 + p.peak_time_s, t0 + nxt_rel + lag])     # nxt_rel: from this pulse
            dev = polarity * (kinetics.smooth(yy, max(1, int(round(0.2e-3 * fs)))) - base)
            i0 = kinetics.decay_start(dev, int(ip), int(ie), fs)
            tail = None
            if ie - i0 >= 5:
                fit = fits.exp_decay(tt[i0:ie] - tt[i0], yy[i0:ie] - base, offset=0.0,
                                     purpose="tail of each pulse, taken off before the next")
                if fit.ok and fit["tau"] > 0 and np.sign(fit["amp"]) == np.sign(p.amplitude):
                    tail = (fit["amp"], fit["tau"], float(tt[i0] - t0))
            if tail is None and np.isfinite(p.decay_tau_s):
                tail = (p.amplitude, p.decay_tau_s, p.peak_time_s)
                p.flags.append("the tail fit failed; the decay tau from the response window was used for the tail")
            if tail is None:
                p.flags.append(f"no decay to take off before pulse {j + 2}")
                continue
            p.tail = tail
            amp, tau, ts_rel = tail
            if not moving:
                shape = np.zeros(t.size)
                m = t >= t0 + ts_rel
                shape[m] = np.exp(-np.clip((t[m] - (t0 + ts_rel)) / tau, 0, 745))
                avg_tails += amp * shape
            for k in range(len(used)):
                a_k = per_sweep[k, j]
                scale = float(np.clip(a_k / p.amplitude, 0.0, 3.0)) if np.isfinite(a_k) and p.amplitude else 0.0
                if not members[k]:
                    scale = 1.0           # a sweep that fired: its tail as the average's (it is out of the PSP measures)
                start = ons[k, j] + ts_rel
                m = t >= start
                work[k, m] -= scale * amp * np.exp(-np.clip((t[m] - start) / tau, 0, 745))

    # ---- per sweep: failures, latency
    onsets_sweep = np.full((len(used), n_pulses), np.nan)
    # the failure threshold of each pulse: the same 1 ms measurement taken just
    # before that pulse, on the same corrected sweep (in a train the residue of
    # earlier responses adds to the noise there, and to the amplitude alike)
    noise_meas = np.full((len(used), n_pulses), np.nan)
    for k in range(len(used)):
        row = work[k]
        b0, b1 = np.searchsorted(t, [ons[k, 0] + base_pre, ons[k, 0] + s["baseline_end_ms"] / 1e3])
        for j in range(n_pulses):
            nb = int(np.searchsorted(t, ons[k, j] + s["baseline_end_ms"] / 1e3)) - half
            noise_meas[k, j] = float(row[max(nb - half, 0):nb + half + 1].mean() - bases[k])
        sm = kinetics.smooth(row, max(1, int(round(0.2e-3 * fs))))
        noise = kinetics.robust_noise(row[b0:b1] - bases[k])
        dev = polarity * (sm - bases[k])
        for j, p in enumerate(pulses):
            if not member[k, j] or not np.isfinite(p.peak_time_s):
                continue
            blank = max(s["blank_ms"] / 1e3, 0.0)
            r0 = int(np.searchsorted(t, ons[k, j] + blank))
            pk = int(np.searchsorted(t, ons[k, j] + p.peak_time_s)) + 1
            # the onset is a departure from the level just before this pulse (a
            # later pulse can start on a residue its tail model did not take off)
            pre0 = int(np.searchsorted(t, ons[k, j] - 0.002))
            local = float(np.median(dev[pre0:max(r0 - 1, pre0 + 1)])) if j > 0 else 0.0
            on = kinetics.find_onset(dev - local, r0, pk, 3.0 * noise, max(1, int(round(0.5e-3 * fs))))
            if on is not None:
                lat = float(t[on] - ons[k, j])
                if not (light and lat < s["light_min_latency_ms"] / 1e3):
                    onsets_sweep[k, j] = lat
    out.choose("sweep amplitude", f"mean over {s['mean_window_ms']:g} ms at the average's peak time",
               "the same time in every sweep, so a failure reads near zero")
    thresholds = []
    for j, p in enumerate(pulses):
        nm = noise_meas[member[:, j], j]
        nm = nm[np.isfinite(nm)]
        noise_sd = float(np.std(nm, ddof=1)) if nm.size > 1 else NAN
        threshold = s["failure_sd"] * noise_sd
        thresholds.append(threshold)
        a = polarity * per_sweep[member[:, j], j]
        finite = a[np.isfinite(a)]
        if finite.size:
            success = finite > threshold if np.isfinite(threshold) else np.ones(finite.size, dtype=bool)
            p.success_rate = float(success.mean())
            p.potency = float(polarity * finite[success].mean()) if success.any() else NAN
            mean = float(finite.mean())
            if finite.size > 2 and mean != 0:
                var = float(np.var(finite, ddof=1))
                p.cv = float(np.sqrt(var) / abs(mean))
                corrected_var = var - (noise_sd ** 2 if np.isfinite(noise_sd) else 0.0)
                p.inv_cv2 = float(mean ** 2 / corrected_var) if corrected_var > 0 else NAN
            col = polarity * per_sweep[:, j]
            ok = member[:, j] & np.isfinite(onsets_sweep[:, j]) & (col > threshold if np.isfinite(threshold) else True)
            lats = onsets_sweep[ok, j]
            if lats.size:
                p.latency_median_s = float(np.median(lats))
                p.jitter_s = float(np.std(lats, ddof=1)) if lats.size > 1 else NAN
        p.n_sweeps = int(finite.size)
        p.spike_probability = float(spike_table[:, j].mean()) if not vc else NAN

    # ---- the time-locked test on pulse 1
    arts = getattr(context, "artifacts", [])
    avoid = [(a.start_s, a.end_s) for a in arts if a.kind != "stimulus artifact"]
    p_locked, n_sham = time_locked_p(t, rows[member[:, 0]], float(fixed[0]), fs, s,
                                     np.concatenate([st.times_s[i] for i in used]), avoid=avoid)
    if avoid:
        out.notes.append("sham times keep clear of " + "; ".join(
            f"{a.kind} {a.start_s * 1e3:.1f}-{a.end_s * 1e3:.1f} ms" for a in arts
            if a.kind != "stimulus artifact"))
    out.values["time_locked_p"] = p_locked
    if np.isfinite(p_locked) and p_locked > 0.05:
        out.flags.append(Flag("no response", f"nothing time-locked to the stimulus: pulse 1's response is no larger "
                              f"than the same measurement at {n_sham} sham times in the same sweeps (p = {p_locked:.2f})",
                              "p > 0.05", "shuffled-trigger surrogate"))
    elif not np.isfinite(p_locked):
        if not pulses[0].detected:
            out.flags.append(Flag("no response", f"pulse 1 is below {s['detect_sd']:g} noise SD on the average "
                                  "(the sweeps are too short for the sham test)", f"{s['detect_sd']:g} SD"))

    # ---- what stands for the recording
    p1 = pulses[0]
    ratio = lambda a, b: float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else NAN  # noqa: E731
    p2 = pulses[1] if n_pulses > 1 else None
    out.values.update({
        "p1_amplitude": p1.amplitude, "p1_latency_median": p1.latency_median_s * 1e3,
        "p1_jitter": p1.jitter_s * 1e3, "p1_latency_average": p1.latency_avg_s * 1e3,
        "p1_rise_10_90": p1.rise_10_90_s * 1e3, "p1_rise_20_80": p1.rise_20_80_s * 1e3,
        "p1_decay_tau": p1.decay_tau_s * 1e3, "p1_decay_tau_from_peak": p1.decay_tau_from_peak_s * 1e3,
        "p1_half_width": p1.half_width_s * 1e3, "p1_integral": p1.integral / (1e3 if vc else 1.0),
        "p1_success_rate": p1.success_rate, "p1_potency": p1.potency, "p1_cv": p1.cv, "p1_inv_cv2": p1.inv_cv2,
        "ppr": ratio(p2.amplitude, p1.amplitude) if p2 else NAN,
        "units": units, "integral_units": "pC" if vc else "mV*ms",
    })
    if p2 is not None and not moving:
        # pulse 2 against its own pre-pulse baseline on the plain average: the raw ratio, for comparison
        raw = _measure(t, rows[member[:, 1]].mean(axis=0), float(fixed[1]), fs, s, baseline=None, polarity=polarity,
                       next_t=float(fixed[2]) if n_pulses > 2 else np.inf, light=light)
        out.values["ppr_raw"] = ratio(raw.amplitude, p1.amplitude)
    else:
        out.values["ppr_raw"] = NAN
    k_ss = min(s["steady_state_pulses"], n_pulses)
    ss = float(np.nanmean([p.amplitude for p in pulses[-k_ss:]])) if n_pulses > 2 else NAN
    out.values["steady_state_ratio"] = ratio(ss, p1.amplitude) if n_pulses > 2 else NAN
    choice = s["headline"]
    n_avg = int(member[:, 0].sum())
    if choice == "a chosen pulse" and s["headline_pulse"] <= n_pulses:
        h = pulses[s["headline_pulse"] - 1].amplitude
        what = f"Pulse {s['headline_pulse']} amplitude (the response to stimulus {s['headline_pulse']} of each sweep, " \
               f"earlier tails taken off, on the average of {n_avg} sweeps)"
    elif choice == "steady state" and n_pulses > 2:
        h = ss
        what = f"Steady-state amplitude (mean of the last {k_ss} pulses, on the average of {n_avg} sweeps)"
    elif choice == "train mean" and n_pulses > 1:
        h = float(np.nanmean([p.amplitude for p in pulses]))
        what = f"Train mean amplitude (mean of all {n_pulses} pulses, on the average of {n_avg} sweeps)"
    else:
        h = p1.amplitude
        what = f"Pulse 1 amplitude (first stimulus of each sweep, measured on the average of {n_avg} sweeps)"
        if choice != "pulse 1":
            out.notes.append(f"'{choice}' cannot be the headline of a {n_pulses}-pulse recording; pulse 1 is used")
    out.values["amplitude"] = h
    out.values["headline_is"] = what
    out.choose("headline", what, "set in Analyze (the recording's number)")

    # ---- the tables
    out.tables["pulses"] = [_pulse_row(p, p1.amplitude, vc, spike_at[:, j]) for j, p in enumerate(pulses)]
    out.tables["sweeps"] = [{"sweep": used[k], "pulse": j + 1, "amplitude": per_sweep[k, j],
                             "in_psp_measures": bool(member[k, j]), "spiked": bool(spike_table[k, j]),
                             "failure": bool(np.isfinite(per_sweep[k, j]) and np.isfinite(thresholds[j])
                                             and polarity * per_sweep[k, j] <= thresholds[j]),
                             "spike_latency_ms": spike_at[k, j] * 1e3 if not vc else NAN,
                             "latency_ms": onsets_sweep[k, j] * 1e3, "onset_s": ons[k, j]}
                            for k in range(len(used)) for j in range(n_pulses)]
    if not vc:
        for k, i in enumerate(sweeps):
            for j in range(n_pulses):
                if spike_table[k, j]:
                    out.tables.setdefault("spikes", []).append({"sweep": i, "pulse": j + 1})
    # pulse 1's flags that say "not a measurement" are QC flags; the rest are notes
    for p in pulses:
        for text in p.flags:
            qc = p.number == 1 and ("limit" in text or "sooner than a synapse" in text)
            out.flags.append(Flag(short_code(text), text, level="pulse", where=f"pulse {p.number}",
                                  severity="flag" if qc else "note"))
    if moving:
        out.notes.append("the pulses move between sweeps: each pulse is measured on its own pulse-aligned average")
    # what to draw: each pulse's peak and the fitted decay of pulse 1
    for p in pulses:
        if np.isfinite(p.peak_time_s):
            out.drawables.append(Drawable("marker", f"pulse {p.number} peak", x=np.array([p.time_s + p.peak_time_s]),
                                          y=np.array([(p1_base if p1_base is not None else 0) + p.amplitude]),
                                          domain="time"))
    return out
