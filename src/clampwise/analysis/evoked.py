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
- **Sweeps that differ under the stimulus are never averaged together**
  (2026-11-09, the lab's photostim protocols). When the command under pulse 1
  changes from sweep to sweep (a current or voltage step family with the
  stimulus inside the step), or the stimulus's size does (an intensity
  series), each level is measured on its own (tables by_command,
  by_intensity). The recording's numbers are the level at the holding (no
  current injected), or the strongest stimulus. Each level's response is also read at one fixed latency
  (the peak time of the largest level's response) and a straight line through
  it against the potential before the stimulus gives the reversal potential:
  the fixed-latency method of Martina, Royer & Paré 2001 (J Neurosci
  21:2878) and the linear fit of Ahmed, Headley & Paré 2021 (J Neurophysiol
  126:1234).
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


def light_artifact(t: np.ndarray, rows: np.ndarray, ons: np.ndarray, widths: np.ndarray, fs: float,
                   s: dict) -> tuple[float, str]:
    """(how long after each light pulse's onset to blank, why) when the light itself marks the trace: a square
    deflection held while the light is on and gone when it goes off (a photoelectric potential at the electrode,
    or pickup from the LED driver), or a sharp jump at its onset and another at its end; either one the same in
    every sweep whatever the cell is doing. (0, "") when there is none. Judged on pulse 1 of every sweep
    (2026-11-10, the lab's connectivity experiments). A photocurrent or a synaptic response is not one: it goes on
    after the light ends."""
    width = float(np.nanmedian(widths)) if np.isfinite(widths).any() else NAN
    if not np.isfinite(width) or width <= 0 or rows.shape[0] == 0:
        return 0.0, ""
    base_pre, base_end = s["baseline_start_ms"] / 1e3, s["baseline_end_ms"] / 1e3
    inside, after, edges, noises = [], [], [], []
    two = max(2, int(round(0.1e-3 * fs)))
    for k in range(rows.shape[0]):
        on = float(ons[k, 0])
        off = on + width
        y = rows[k]
        b0, b1 = np.searchsorted(t, [on + base_pre, on + base_end])
        i0, i1 = np.searchsorted(t, [on + 0.3e-3, off - 0.1e-3])
        a0, a1 = np.searchsorted(t, [off + 0.5e-3, off + 1.5e-3])
        if b1 - b0 < 3 or i1 - i0 < 2 or a1 - a0 < 2 or a1 > y.size:
            continue
        base = float(np.median(y[b0:b1]))
        noises.append(kinetics.robust_noise(y[b0:b1] - base) or float(np.std(y[b0:b1])) or 1e-9)
        inside.append(float(np.mean(y[i0:i1]) - base))
        after.append(float(np.mean(y[a0:a1]) - base))
        j_on, j_off = int(np.searchsorted(t, on)), int(np.searchsorted(t, off))
        edges.append((float(np.max(np.abs(np.diff(y[max(j_on - two, 0):j_on + two + 1])))),
                      float(np.max(np.abs(np.diff(y[max(j_off - two, 0):j_off + two + 1]))))))
    if not inside:
        return 0.0, ""
    inside, after, noises, edges = (np.asarray(x, float) for x in (inside, after, noises, edges))
    noise = float(np.median(noises))
    level = float(np.median(inside))
    spread = 1.4826 * float(np.median(np.abs(inside - level))) if inside.size >= 3 else 0.0
    same = spread <= 0.35 * abs(level) + noise
    square = abs(level) >= 4 * noise and float(np.median(np.abs(after))) <= 0.35 * abs(level)
    blank = width + 0.5e-3
    if square and same:
        return blank, (f"a {level:+.2f} deflection held while the light is on ({width * 1e3:.1f} ms) and gone within "
                       f"0.5 ms of its end, the same in all {inside.size} sweep(s): the light's own mark on the "
                       "electrode (photoelectric) or pickup from the LED driver, not the cell. The light's pulse is "
                       "blanked and the response measured after it")
    # white noise's sample-to-sample change has about 1.4 x the noise SD; a jump of 8 x is not noise
    if edges.size and float(np.median(edges[:, 0])) >= 8 * noise and float(np.median(edges[:, 1])) >= 8 * noise:
        return blank, (f"a sharp jump at the light's onset ({float(np.median(edges[:, 0])) / noise:.0f} x the noise) "
                       f"and another at its end ({float(np.median(edges[:, 1])) / noise:.0f} x), in every sweep: the "
                       "light's own mark on the electrode, not the cell. The light's pulse is blanked and the "
                       "response measured after it")
    return 0.0, ""


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
    onset_raw_s: float = NAN                # the onset on the average whatever the light rule says
    at_light_end: float = NAN               # the deflection (signed as the response) when the light goes off
    time_to_half_peak_s: float = NAN        # from the stimulus to half the peak (Turesson 2013's rise)
    decay_80_20_s: float = NAN              # the decay from 80% to 20% of the peak (Ahmed & Pare 2023)
    initial_slope: float = NAN              # units per ms over the first half of the rise (Popescu 2007)
    onset_noise_s: float = NAN              # the onset on the average by each definition (kinetics.ONSET_...)
    onset_10pct_s: float = NAN
    onset_foot_s: float = NAN
    decay_tau_weighted_s: float = NAN       # two exponentials: (a1 tau1 + a2 tau2) / (a1 + a2)
    decay_two_exponentials: bool = False    # whether two fit the decay better than one (F test, p < 0.01)
    flags: list[str] = field(default_factory=list)


def _measure(t: np.ndarray, y: np.ndarray, t0: float, fs: float, s: dict, *, baseline: float | None,
             polarity: int, next_t: float, light: bool, light_width_s: float = NAN,
             light_blank_s: float = 0.0) -> Pulse:
    """One response on a trace ``y`` (an average with earlier tails removed). ``light_blank_s``: a light
    artifact recognised on the sweeps, blanked from the pulse's onset; ``light_width_s``: how long the light
    stays on, to read the deflection when it goes off (a direct photocurrent is still at its height there)."""
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
    if light_blank_s > blank:
        blank, seen, note = light_blank_s, False, ""
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
    if np.isfinite(light_width_s) and light_width_s > 0:
        e0, e1 = np.searchsorted(t, [t0 + light_width_s - 0.3e-3, t0 + light_width_s])
        if e1 > e0 and e1 <= dev.size:
            p.at_light_end = float(np.mean(dev[e0:e1]))
    onset = kinetics.find_onset(dev, r0, peak + 1, s["onset_noise_sd"] * p.noise_sd, max(1, int(round(0.5e-3 * fs))))
    # the onset by the field's other two definitions, beside the lab's (whichever is chosen is the latency)
    if p.detected:
        p.onset_10pct_s = kinetics.onset_fraction(t, dev, r0, peak) - t0
        p.onset_foot_s = kinetics.onset_foot(t, dev, r0, peak) - t0
    if onset is not None and onset < peak:
        lat = float(t[onset] - t0)
        p.onset_raw_s = p.onset_noise_s = lat
        chosen = s.get("onset_definition", kinetics.ONSET_NOISE)
        if chosen == kinetics.ONSET_FRACTION:
            lat = p.onset_10pct_s
        elif chosen == kinetics.ONSET_FOOT:
            lat = p.onset_foot_s
        # the initial slope (the LTP readout: Popescu et al. 2007): a line over the first half of the rise,
        # from the onset to half the peak, in units per ms
        half_i = onset + int(np.argmax(dev[onset:peak + 1] >= 0.5 * dev[peak])) if peak > onset else onset
        if half_i - onset >= 2:
            fit = fits.linear(t[onset:half_i + 1] * 1e3, y[onset:half_i + 1], purpose="initial slope of the rise")
            p.initial_slope = float(fit["slope"])
        if light_blank_s > 0 and onset <= r0 + max(1, int(round(0.1e-3 * fs))):
            p.flags.append(f"the response is already under way where the light's blank ends ({blank * 1e3:.1f} ms): "
                           "its onset lies inside the blank and is not known")
        if light and p.onset_raw_s < s["light_min_latency_ms"] / 1e3:
            p.flags.append(f"onset {p.onset_raw_s * 1e3:.2f} ms after the light, sooner than a synapse can answer "
                           f"({s['light_min_latency_ms']:g} ms): the light itself or noise; latency not measured")
        elif np.isfinite(lat):
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
        # the lab's rise and decay times: time to half the peak (Turesson 2013), from the stimulus; and the
        # time the decay takes from 80% to 20% of the peak (Ahmed & Pare 2023)
        seg = dev[r0:peak + 1]
        below = np.flatnonzero(seg < 0.5 * peak_dev)
        if below.size and r0 + int(below[-1]) + 1 <= peak:
            i = r0 + int(below[-1])
            p.time_to_half_peak_s = kinetics.interp_crossing(t, dev, i, i + 1, 0.5 * peak_dev) - t0
        stop = max(end, r1)
        after = dev[peak:stop]
        i80 = np.flatnonzero(after <= 0.8 * peak_dev)
        i20 = np.flatnonzero(after <= 0.2 * peak_dev)
        if i80.size and i20.size and i20[0] > i80[0]:
            a, b = peak + int(i80[0]), peak + int(i20[0])
            p.decay_80_20_s = (kinetics.interp_crossing(t, dev, b - 1, b, 0.2 * peak_dev)
                               - kinetics.interp_crossing(t, dev, a - 1, a, 0.8 * peak_dev))
    decay_stop = max(int(np.searchsorted(t, t0 + min(s["decay_window_ms"] / 1e3, limit))), end)
    decay_stop = min(decay_stop, r_lim if s["follow_slow"] else decay_stop, t.size)
    if p.detected and decay_stop > peak + 8:
        p.decay_tau_from_peak_s = _decay(t, ys, peak, decay_stop, p.baseline)[0]
        start = kinetics.decay_start(dev, peak, decay_stop, fs)
        tau, r2, amp = _decay(t, ys, start, decay_stop, p.baseline, purpose="evoked decay tau, from 80% of the peak")
        p.decay_tau_s, p.decay_r2 = tau, r2
        p.decay_tau_weighted_s, p.decay_two_exponentials = _two_exponentials(t, ys, start, decay_stop, p.baseline,
                                                                             polarity, tau)
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


def _two_exponentials(t, y, i0, i1, baseline, polarity, tau_one) -> tuple[float, bool]:
    """(the amplitude-weighted tau of a two-exponential fit of the decay over [i0, i1), whether two exponentials
    fit it better than one). Better: both components carry at least 10% of the amplitude, their taus differ at
    least twofold, the slow one lies inside the window, and an F test of the two fits gives p < 0.01. The
    weighted tau, (a1 tau1 + a2 tau2) / (a1 + a2), is what a decay with a fast and a slow part is reported by;
    for a decay that is one exponential it equals that one's tau."""
    from scipy import stats as sps

    n = i1 - i0
    if n < 20 or not np.isfinite(tau_one) or tau_one <= 0:
        return NAN, False
    x = t[i0:i1] - t[i0]
    z = polarity * (np.asarray(y[i0:i1], float) - baseline)
    try:
        two = fits.biexp_decay(x, z, tau_mono=tau_one, purpose="evoked decay, two exponentials")
    except Exception:
        return float(tau_one), False
    if not two.ok:                                   # no second component to find: one exponential it is
        return float(tau_one), False
    tau_w = float(two["tau_w"])
    one = fits.exp_decay(x, z, purpose="evoked decay, one exponential (against two)")
    if not one.ok or not np.isfinite(tau_w) or tau_w <= 0:
        return float(tau_one), False
    rss1 = float(np.sum((z - one.curve(x)) ** 2))
    rss2 = float(np.sum((z - two.curve(x)) ** 2))
    # samples of a filtered trace are not independent: count one per 0.5 ms for the test
    n_eff = max(8.0, (x[-1] - x[0]) / 0.5e-3)
    better = False
    if rss2 > 0 and rss1 > rss2 and n_eff > 6:
        f = ((rss1 - rss2) / 2.0) / (rss2 / (n_eff - 5.0))
        fast = float(two["fast_fraction"])
        better = bool(sps.f.sf(f, 2, n_eff - 5.0) < 0.01 and 0.1 <= fast <= 0.9
                      and two["tau2"] >= 2.0 * two["tau1"] and not two.params.get("slow_beyond_window", False))
    return (tau_w if better else float(tau_one)), better


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
    # the upstroke holds 20 mV/ms for 0.1 ms at least; a single sample over it is noise (0.2 mV of noise at
    # 20 kHz gives dV/dt a 6 mV/ms spread, and one sample in a few hundred crosses 20)
    above = dvdt[lead:] >= SPIKE_DVDT
    hold = max(2, int(round(0.1e-3 * fs)))
    if above.size < hold:
        return NAN
    runs = np.convolve(above.astype(int), np.ones(hold, int), mode="valid") == hold
    cross = np.flatnonzero(runs)
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
    ("inside the blank", "onset in the blank"),
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
        "latency_average_ms": p.latency_avg_s * 1e3, "latency_noise_ms": p.onset_noise_s * 1e3,
        "latency_10pct_ms": p.onset_10pct_s * 1e3, "latency_foot_ms": p.onset_foot_s * 1e3,
        "time_to_peak_ms": p.peak_time_s * 1e3,
        "rise_10_90_ms": p.rise_10_90_s * 1e3, "rise_20_80_ms": p.rise_20_80_s * 1e3,
        "decay_tau_ms": p.decay_tau_s * 1e3, "decay_r2": p.decay_r2,
        "decay_tau_from_peak_ms": p.decay_tau_from_peak_s * 1e3,
        "decay_tau_weighted_ms": p.decay_tau_weighted_s * 1e3, "decay_two_exponentials": p.decay_two_exponentials,
        "half_width_ms": p.half_width_s * 1e3,
        "time_to_half_peak_ms": p.time_to_half_peak_s * 1e3, "decay_80_20_ms": p.decay_80_20_s * 1e3,
        "initial_slope": p.initial_slope,
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
    Setting("onset_noise_sd", 2.0, "Onset: departure past (noise SD)", "The response's onset is the first departure "
            "from the baseline past this many times its noise that stays there for 0.5 ms (the lab's criterion: "
            "twice the noise, then growing without returning to baseline; 3 was Clampwise's).", "",
            "Ahmed, Headley & Pare 2021", minimum=1.0),
    Setting("onset_definition", kinetics.ONSET_NOISE, "Onset (latency) read as", "Which definition the latency "
            "follows; the other two are reported beside it on the average. The lab's: the first departure past the "
            "noise multiple below that holds for 0.5 ms. 10% of the peak: the last time the rise is under a tenth "
            "of the peak. The foot: a line through 20% and 80% of the rise, read where it meets the baseline; it "
            "ignores the noise and reads a little late on a response that begins gently. On single sweeps the last "
            "two need a clean peak: with small responses the lab's is the steadier.", "",
            "Ahmed, Headley & Pare 2021; Jonas, Major & Sakmann 1993 (the foot)", choices=kinetics.ONSET_DEFINITIONS),
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
    Output("p1_latency_median", "Median of the per-sweep onsets of pulse 1 (onset: the first sustained departure "
           "past the set number of noise SDs, twice the noise by default, the lab's criterion).", "ms",
           "Ahmed, Headley & Pare 2021"),
    Output("p1_jitter", "SD of the per-sweep onsets of pulse 1.", "ms"),
    Output("p1_latency_average", "Onset of pulse 1 on the average trace (biased early when jitter is high).", "ms"),
    Output("p1_rise_10_90", kinetics.DEFINITIONS["rise_10_90"], "ms", ""),
    Output("p1_rise_20_80", kinetics.DEFINITIONS["rise_20_80"], "ms"),
    Output("p1_decay_tau", kinetics.DEFINITIONS["decay_tau"], "ms"),
    Output("p1_decay_tau_from_peak", "Decay tau fitted from the peak.", "ms", ""),
    Output("p1_decay_tau_weighted", "Pulse 1's decay as two exponentials when two fit it better than one (each "
           "at least 10% of the amplitude, taus at least twofold apart, F test p < 0.01): the amplitude-weighted "
           "tau, (a1 tau1 + a2 tau2) / (a1 + a2). Equal to the single tau when one exponential is enough.", "ms",
           "the weighted time constant of a double-exponential decay", label="Pulse 1 decay tau (weighted)"),
    Output("p1_decay_two_exponentials", "yes when pulse 1's decay is two exponentials by that test, no when one "
           "is enough.", "", label="Pulse 1 decay: two exponentials"),
    Output("p1_latency_noise", "Pulse 1's onset on the average by the lab's rule: the first departure past the "
           "noise multiple that holds for 0.5 ms.", "ms", "Ahmed, Headley & Pare 2021",
           label="Pulse 1 onset (noise rule, average)"),
    Output("p1_latency_10pct", "Pulse 1's onset on the average as the last time the rise is under 10% of the "
           "peak.", "ms", "the fraction-of-peak onset", label="Pulse 1 onset (10% of the peak, average)"),
    Output("p1_latency_foot", "Pulse 1's onset on the average by the extrapolated foot: a line through 20% and 80% "
           "of the rise, read at the baseline.", "ms", "Jonas, Major & Sakmann 1993",
           label="Pulse 1 onset (foot, average)"),
    Output("ppr_mean_of_ratios", "The mean over sweeps of each sweep's own pulse 2 / pulse 1 (sweeps whose pulse 1 "
           "is a failure left out). Shown beside the paired-pulse ratio for comparison only: a ratio of two noisy "
           "numbers is skewed upward, so this reads higher than the ratio of the means and can show facilitation "
           "that is not there.", "ratio", "Kim & Alger 2001", label="Paired-pulse ratio, mean of ratios (biased)"),
    Output("p1_half_width", kinetics.DEFINITIONS["half_width"], "ms"),
    Output("p1_time_to_half_peak", "Pulse 1: from the stimulus to half the peak on the rise (the lab's rise "
           "time).", "ms", "Turesson, Rodriguez-Sierra & Pare 2013", label="Pulse 1 time to half peak"),
    Output("p1_decay_80_20", "Pulse 1: the time the decay takes from 80% to 20% of the peak.", "ms",
           "Ahmed & Pare 2023", label="Pulse 1 decay, 80-20%"),
    Output("p1_initial_slope", "Pulse 1: the initial slope of the rise, a line from the onset to half the peak "
           "(the first half of the rising phase: the LTP readout).", "signal units per ms",
           "Popescu, Saghyan & Pare 2007", label="Pulse 1 initial slope"),
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
    Output("condition", "What differs between the sweeps under the stimulus, when something does: the command "
           "(a step family played under the stimulus) or the stimulus's size. Each level is then measured on its own "
           "(table by_command or by_intensity) and the recording's numbers are one level's (see 'sweeps measured').", "",
           label="Sweeps differ by"),
    Output("n_levels", "How many levels of that condition the sweeps hold.", "count", label="Levels"),
    Output("reversal", "The response's reversal potential: a straight line through the response at a fixed latency "
           "(the peak time of the largest level's response) against the membrane potential before the stimulus "
           "(current clamp) or the command (voltage clamp), read where it crosses zero. Liquid junction potential "
           "not corrected.", "mV", "the fixed-latency method (Martina, Royer & Pare 2001); a linear fit of the "
           "response against Vm (Ahmed, Headley & Pare 2021)", label="Reversal potential"),
    Output("reversal_r2", "r squared of that line.", "ratio", label="Reversal fit r squared"),
    Output("reversal_extrapolated", "yes when the reversal lies outside the potentials recorded (read off the line "
           "beyond the data).", "", label="Reversal extrapolated"),
    Output("reversal_slope", "The line's slope: the response's change per mV of membrane potential (current "
           "clamp, mV/mV; voltage clamp, pA/mV = nS, the synaptic conductance at that latency).", "per mV",
           label="Response per mV"),
    Output("intensity_threshold", "The weakest stimulus (its channel's size during pulse 1: the light command, "
           "or an isolator's monitor) from which every stronger level carries a response time-locked to it "
           "(p < 0.05 against sham times) and above the detection level on its average.", "stimulus channel units",
           label="Stimulus threshold"),
    Output("direct_photocurrent", "yes when the recorded cell itself answers the light: the response begins "
           "sooner than a synapse can (under the light minimum latency), is still at half its peak or more when "
           "the light goes off, and peaks there or within 3 ms after. Such a cell expresses the opsin or has its "
           "membrane lit; the response says nothing about a connection.", "",
           "Turesson, Rodriguez-Sierra & Pare 2013", label="Direct photocurrent"),
    Output("light_artifact_blank_ms", "When the light itself marks the trace (a square deflection held while it "
           "is on and gone when it goes off, or sharp jumps at its onset and end, the same in every sweep), how "
           "long after each light pulse's onset the trace is blanked before anything is measured.", "ms",
           "Clampwise", label="Light artifact blanked"),
    Output("region_known", "Whether the region the patterned light lit on each sweep is known (the Polygon's "
           "log kept with the recording). A mapping protocol whose regions are unknown gets no averaged number.",
           "", "Clampwise", label="Region lit known"),
    Output("pattern", "The pattern the patterned light showed on every sweep, when its log says one lit them all.",
           "", "Clampwise", label="Pattern lit"),
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


MAPPING_WORDS = ("grid", "map", "connectivity")      # protocols that light a different region on each sweep


def is_mapping(recording: Recording) -> bool:
    """A protocol that lights one region a sweep (a grid map, a cell-by-cell connectivity run): its sweeps mean
    nothing averaged together unless the pattern lit on each is known."""
    text = f"{recording.protocol} {recording.source.name}".lower()
    return any(w in text for w in MAPPING_WORDS)


def _by_pattern(recording: Recording, context, s, sweeps: list[int], out: Result) -> str:
    """Patterned light (the Polygon's log kept with the recording): the sweeps measured again for each pattern lit
    (each region of interest, the full field, a control region), into the table ``by_pattern``; sweeps whose
    pattern is not sure are left out of it and counted. Nothing when one pattern lit them all. Returns what is
    known of the patterns: "no log" (nothing records them), "one pattern", "by pattern" (grouped), "not sure"
    (a log, but no sweep's pattern is sure)."""
    import copy

    from ..rig import light_record

    try:
        matched = light_record.for_recording(recording, write_it=False, stimulus=context.stimulus)
    except Exception:                                 # a log that cannot be read: the recording as a whole
        return "no log"
    if not matched:
        return "no log"
    per = light_record.for_sweeps(recording, matched)
    # stepped by the trigger: each pulse its own pattern, the same in every sweep (pulse j, step j)
    by_pulse: dict[int, set] = {}
    for i in sweeps:
        m = per[i] if i < len(per) else None
        for j, p in enumerate((m or {}).get("pulses") or []):
            if p.get("by_trigger"):
                by_pulse.setdefault(j, set()).add(p["step"] if p.get("sure") else None)
    if by_pulse:
        for j, names in sorted(by_pulse.items()):
            name = next(iter(names)) if len(names) == 1 and None not in names else None
            if j < len(out.tables.get("pulses", [])):
                out.tables["pulses"][j]["pattern"] = name or "not sure"
        out.notes.append("patterned light stepped on each trigger: " + ", ".join(
            f"pulse {j + 1} {next(iter(n)) if len(n) == 1 and None not in n else 'not sure'}"
            for j, n in sorted(by_pulse.items())) + " (each pulse's pattern assumes one trigger a pulse)")
    groups: dict[str, list[int]] = {}
    unsure = 0
    for i in sweeps:
        m = per[i] if i < len(per) else None
        if m is None or not m.get("sure"):
            unsure += 1
        elif m.get("lit"):
            groups.setdefault(m["step"], []).append(i)
    if len(groups) < 2 and not unsure:
        if groups:
            out.notes.append(f"patterned light: every sweep lit by {next(iter(groups))}")
            out.values["pattern"] = next(iter(groups))
            return "one pattern"
        return "no log"
    if not groups:
        out.notes.append(f"patterned light: no sweep's pattern is sure ({unsure} sweep(s) not sure)")
        return "not sure"
    rows = []
    for name, idx in groups.items():
        sub = copy.copy(context)
        sub.shared = {**context.shared, "sweep_indices": idx, "by_pattern": True}
        try:
            r = run(recording, sub, s)
        except Exception as exc:                      # one pattern never fails the rest
            out.notes.append(f"patterned light: {name} not measured ({exc})")
            continue
        first = per[idx[0]]
        v = r.values
        rows.append({"pattern": name, "sweeps": len(idx), "sweep_numbers": ", ".join(str(i + 1) for i in idx),
                     "area_um2": first.get("area_um2"), "field_fraction": first.get("field_fraction"),
                     "p1_amplitude": v.get("p1_amplitude", NAN), "p1_latency_median": v.get("p1_latency_median", NAN),
                     "p1_success_rate": v.get("p1_success_rate", NAN), "ppr": v.get("ppr", NAN)})
    if rows:
        out.tables["by_pattern"] = rows
        out.notes.append("patterned light: the sweeps measured again for each pattern lit (" +
                         ", ".join(f"{r['pattern']}: {r['sweeps']}" for r in rows) + "), in the table by_pattern")
    if unsure:
        out.notes.append(f"patterned light: {unsure} sweep(s) whose pattern is not sure (a change near or during "
                         "them, or the patterned light not held by Clampwise) are left out of by_pattern")
    for note in light_record.dose_notes([m for m in per if m]):
        out.notes.append("patterned light: " + note)
    return "by pattern"


@dataclass
class Conditions:
    """What differs between the sweeps under pulse 1, and the sweeps at each level."""

    kind: str                                   # "command" or "intensity"
    words: str                                  # "the command under the stimulus", "the stimulus's size"
    units: str
    levels: list[tuple[float, list[int]]]       # (level, sweeps), in order of level
    headline: tuple[float, list[int]]
    why: str


def _command_under(recording: Recording, sweep: int, t_s: float) -> float:
    """The epoch table's level at ``t_s`` in a sweep (NaN without a table)."""
    if sweep >= len(recording.epochs):
        return NAN
    e = next((e for e in recording.epochs[sweep] if e.start_s <= t_s < e.end_s), None)
    return float(e.level) if e is not None else NAN


def _stimulus_size(recording: Recording, st, sweeps: list[int]) -> dict[int, float]:
    """The stimulus channel's size during pulse 1 of each sweep (its peak against the 5 ms before), when the
    stimulus is a recorded channel (a light command, an isolator's monitor); empty for an output or a typed
    stimulus, which say when, not how much."""
    ch = next((c for c in recording.channels if c.name == st.channel and c.recorded), None)
    if ch is None:
        return {}
    fs = recording.sample_rate_hz
    rows = recording.read(ch, sweeps)
    out = {}
    for k, i in enumerate(sweeps):
        t0 = float(st.times_s[i][0])
        w = float(st.widths_s[i][0]) if i < len(st.widths_s) and len(st.widths_s[i]) else 0.001
        a = int(round(t0 * fs))
        b = a + max(1, int(round(w * fs)))
        pre = rows[k, max(0, a - int(0.005 * fs)):a]
        out[i] = float(np.max(rows[k, a:b]) - (np.median(pre) if pre.size else 0.0))
    return out


def _levels(values: dict[int, float], tolerance: float) -> list[tuple[float, list[int]]]:
    """Sweeps grouped by level: a new level where the sorted values jump by more than ``tolerance``."""
    items = sorted((v, i) for i, v in values.items() if np.isfinite(v))
    groups: list[list[tuple[float, int]]] = []
    for v, i in items:
        if groups and v - groups[-1][-1][0] <= tolerance:
            groups[-1].append((v, i))
        else:
            groups.append([(v, i)])
    return [(float(np.mean([v for v, _ in g])), sorted(i for _, i in g)) for g in groups]


def conditions(recording: Recording, st, sweeps: list[int]) -> Conditions | None:
    """The command under pulse 1, else the stimulus's size, when it differs between the sweeps."""
    holding = recording.holding if np.isfinite(recording.holding) else 0.0
    cmd = next((c for c in recording.channels if c.role.name == "COMMAND"), None)
    under = {i: _command_under(recording, i, float(st.times_s[i][0])) for i in sweeps}
    levels = _levels(under, 1e-6)
    if len(levels) >= 2:
        head = min(levels, key=lambda lv: (abs(lv[0] - holding), -len(lv[1])))
        units = cmd.units if cmd is not None else ""
        return Conditions("command", "the command under the stimulus", units, levels, head,
                          f"the command under pulse 1 differs between sweeps ({len(levels)} levels, "
                          f"{levels[0][0]:g} to {levels[-1][0]:g} {units}): the recording's numbers are the "
                          f"{len(head[1])} sweep(s) at {head[0]:g} {units}, the level nearest the holding "
                          f"({holding:g} {units}); every level is measured on its own in the table by_command")
    sizes = _stimulus_size(recording, st, sweeps)
    if sizes:
        top = max(sizes.values())
        levels = _levels(sizes, 0.05 * abs(top)) if top > 0 else []
        if len(levels) >= 2:
            head = levels[-1]
            units = next((c.units for c in recording.channels if c.name == st.channel), "")
            what = "light" if st.kind == "light" else "stimulus"
            return Conditions("intensity", f"the {what}'s size", units, levels, head,
                              f"the {what}'s size differs between sweeps ({len(levels)} levels, {levels[0][0]:.3g} "
                              f"to {levels[-1][0]:.3g} {units} on {st.channel}): the recording's numbers are the "
                              f"{len(head[1])} sweep(s) at the strongest {what}; every level is measured on its "
                              "own in the table by_intensity")
    return None


def _by_condition(recording: Recording, context, s, cond: Conditions, out: Result) -> None:
    """Each level measured on its own; each level's response also read at one fixed latency; with the command
    as the condition, the reversal potential from a line through those against the potential."""
    import copy

    fs = recording.sample_rate_hz
    t = recording.time_s
    vc = recording.is_voltage_clamp
    st = context.stimulus
    runs = []
    for level, idx in cond.levels:
        sub = copy.copy(context)
        sub.shared = {**context.shared, "sweep_indices": idx, "by_condition": True}
        try:
            runs.append((level, idx, run(recording, sub, s)))
        except Exception as exc:                          # one level never fails the rest
            out.notes.append(f"{cond.words} at {level:g} {cond.units}: not measured ({exc})")
    if not runs:
        return

    def responds(r: Result) -> bool:
        p = r.values.get("time_locked_p", NAN)
        return bool(np.isfinite(p) and p < 0.05)

    def peak_s(r: Result) -> float:
        rows = r.tables.get("pulses") or [{}]
        return rows[0].get("time_to_peak_ms", NAN) / 1e3

    # the fixed latency: the peak time of the largest response among the levels that respond
    pool = [x for x in runs if responds(x[2]) and np.isfinite(peak_s(x[2]))] or \
        [x for x in runs if np.isfinite(peak_s(x[2]))]
    ref = max(pool, key=lambda x: abs(x[2].values.get("p1_amplitude", 0.0) or 0.0), default=None)
    t_fix = peak_s(ref[2]) if ref is not None else NAN
    half = max(1, int(round(s["mean_window_ms"] / 2e3 * fs)))
    rows = []
    for level, idx, r in runs:
        fired = {row["sweep"] for row in r.tables.get("sweeps", []) if row.get("pulse") == 1 and row.get("spiked")}
        keep = [i for i in idx if i not in fired]
        data = recording.read(recording.signal_channel, keep) if keep else np.empty((0, t.size))
        bases, fixed = [], []
        for k, i in enumerate(keep):
            t0 = float(st.times_s[i][0])
            b0, b1 = np.searchsorted(t, [t0 + s["baseline_start_ms"] / 1e3, t0 + s["baseline_end_ms"] / 1e3])
            base = float(np.median(data[k, b0:b1])) if b1 > b0 else NAN
            bases.append(base)
            if np.isfinite(t_fix):
                c = int(np.searchsorted(t, t0 + t_fix))
                fixed.append(float(np.mean(data[k, max(0, c - half):c + half + 1])) - base)
        pulses = r.tables.get("pulses") or [{}]
        rows.append({"level": level, "sweeps": len(idx), "sweep_numbers": ", ".join(str(i + 1) for i in idx),
                     "baseline": float(np.mean(bases)) if bases else NAN,
                     "fixed_latency_amplitude": float(np.mean(fixed)) if fixed else NAN,
                     "p1_amplitude": r.values.get("p1_amplitude", NAN),
                     "p1_latency_median": r.values.get("p1_latency_median", NAN),
                     "p1_success_rate": r.values.get("p1_success_rate", NAN),
                     "spike_probability": pulses[0].get("spike_probability", NAN),
                     "spike_latency_ms": pulses[0].get("spike_latency_ms", NAN),
                     "time_locked_p": r.values.get("time_locked_p", NAN),
                     "detected": bool(pulses[0].get("detected", False)),
                     "direct": r.values.get("direct_photocurrent", "no"),
                     "ppr": r.values.get("ppr", NAN)})
    n_direct = sum(row["direct"] == "yes" for row in rows)
    answering = sum(row["direct"] == "yes" or row["detected"] or (np.isfinite(row["spike_probability"])
                                                                   and row["spike_probability"] >= 0.5)
                    for row in rows)
    if n_direct:
        out.notes.append(f"a direct photocurrent (the recorded cell answering the light itself) at {n_direct} of "
                         f"{len(rows)} levels")
        if out.values.get("direct_photocurrent") != "yes" and n_direct >= max(1, answering / 2):
            # the headline level did not show it (too small there, or the cell fired), but most levels that
            # answer the light answer it directly: the cell is light-sensitive
            out.values["direct_photocurrent"] = "yes"
            where = [f"{row['level']:g} {cond.units}" for row in rows if row["direct"] == "yes"]
            out.flags.append(Flag("direct photocurrent", "the recorded cell itself answers the light at "
                                  f"{n_direct} of the {answering} level(s) that answer it ({', '.join(where[:6])}"
                                  f"{', ...' if len(where) > 6 else ''}): the response begins sooner than a synapse "
                                  "can answer and is at its height when the light goes off, or the cell fires "
                                  "within the light's first milliseconds. This cell expresses the opsin; its "
                                  "responses say nothing about a connection", "",
                                  "Turesson, Rodriguez-Sierra & Pare 2013"))
    table = "by_command" if cond.kind == "command" else "by_intensity"
    out.tables[table] = rows
    out.values["n_levels"] = len(rows)
    if np.isfinite(t_fix):
        out.choose("fixed latency", f"{t_fix * 1e3:.2f} ms after pulse 1",
                   f"the peak time of the largest response, at {ref[0]:g} {cond.units} (sweeps that fired on pulse 1 "
                   "left out of each level's average)")
    if cond.kind == "intensity":
        # the weakest level from which every stronger level responds too (time-locked, p < 0.05 against sham
        # times, and above the detection level on its average): a weak level passing the sham test by chance
        # (1 in 20) does not set it
        lit = [bool(np.isfinite(row["time_locked_p"]) and row["time_locked_p"] < 0.05 and row["detected"])
               for row in rows]
        k = len(lit)
        while k > 0 and lit[k - 1]:
            k -= 1
        out.values["intensity_threshold"] = rows[k]["level"] if k < len(rows) else NAN
        if k < len(rows):
            out.choose("stimulus threshold", f"{rows[k]['level']:.3g} {cond.units}",
                       "the weakest level from which every stronger level carries a time-locked response "
                       "(p < 0.05 against sham times) above the detection level")
        return
    # the reversal: the response at the fixed latency against the potential before the stimulus
    if vc:
        cmd = next((c for c in recording.channels if c.role.name == "COMMAND"), None)
        if cmd is None or cmd.units != "mV":
            out.notes.append("no reversal potential: the command is not in mV (" +
                             (f"it is in {cmd.units}: a current-clamp protocol played in voltage clamp, so the "
                              "potential at each level is not known)" if cmd is not None else "no command)"))
            return
        x = np.array([row["level"] for row in rows])
    else:
        x = np.array([row["baseline"] for row in rows])
    y = np.array([row["fixed_latency_amplitude"] for row in rows])
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        out.notes.append(f"no reversal potential: {int(good.sum())} level(s) measured at the fixed latency (3 needed)")
        return
    fit = fits.linear(x[good], y[good], purpose="reversal: the response at a fixed latency against the potential")
    if fit is None or not fit.ok or fit["slope"] == 0:
        return
    slope, icpt = fit["slope"], fit["intercept"]
    lo, hi = fit.ci.get("slope", (NAN, NAN))
    out.values["reversal_slope"] = slope
    out.values["reversal_r2"] = fit.r2
    if np.isfinite(lo) and lo <= 0 <= hi:
        out.notes.append(f"no reversal potential: the response does not change with the potential (slope {slope:.3g} "
                         f"per mV, 95% interval {lo:.3g} to {hi:.3g})")
        return
    rev = -icpt / slope
    out.values["reversal"] = rev
    outside = not (x[good].min() <= rev <= x[good].max())
    out.values["reversal_extrapolated"] = "yes" if outside else "no"
    for f in out.flags:
        if f.code == "direct photocurrent":
            f.text += (f". Its size follows the membrane potential (reversal {rev:.0f} mV, r squared {fit.r2:.2f}): "
                       "a conductance in the recorded cell, not an artifact of the light on the electrode")
    out.choose("reversal", f"{rev:.1f} mV" + (" (extrapolated)" if outside else ""),
               f"a straight line through the response {t_fix * 1e3:.2f} ms after pulse 1 at {int(good.sum())} levels "
               f"({x[good].min():.1f} to {x[good].max():.1f} mV), r squared {fit.r2:.2f}; liquid junction potential "
               "not corrected")
    if outside:
        out.flags.append(Flag("reversal extrapolated", f"the reversal ({rev:.1f} mV) lies outside the potentials "
                              f"recorded ({x[good].min():.1f} to {x[good].max():.1f} mV): read off the line beyond the "
                              "data", severity="note"))


@register("evoked", "Evoked responses, every pulse", version=5, types=EVOKED, settings=SETTINGS, outputs=OUTPUTS,
          description="Single pulses, paired pulses and trains: each pulse's amplitude, latency, kinetics, charge "
                      "and reliability, measured on the average and sweep by sweep. Sweeps that differ under the "
                      "stimulus (a step family played under it, a rising light) are measured level by level, with "
                      "the response's reversal potential or the stimulus threshold. The light's own mark on the "
                      "trace is recognised and blanked; a direct photocurrent (the recorded cell answering the "
                      "light itself) is told from a synaptic response; a mapping protocol whose regions are "
                      "unknown is not averaged.",
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
    all_sweeps = list(sweeps)
    cond = None
    if selection is None and not context.shared.get("by_pattern"):
        cond = conditions(recording, st, sweeps)
        if cond is not None:
            selection = cond.headline[1]
            out.values["condition"] = cond.words
            out.choose("sweeps measured", f"{len(selection)} sweep(s) at {cond.headline[0]:.4g} {cond.units}", cond.why)
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
    # a spike before pulse 1 (in the 30 ms its window reaches back) takes the sweep out of the PSP measures
    # (its afterhyperpolarization pulls the baseline) but is not the pulse's doing: the spike probability
    # counts spikes after the pulse only (a cell that fires on its depolarizing steps before every light pulse)
    fired_after = spike_table & (np.nan_to_num(spike_at, nan=-1.0) >= 0)
    n_before = int((spike_table & ~fired_after).sum())
    if n_before:
        out.notes.append(f"{n_before} sweep(s) fired before pulse 1 (within the 30 ms before it): left out of the "
                         "PSP measures, not counted as firing on the pulse")
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
                                                 spike_probability=float(fired_after[:, j].mean())), NAN, vc,
                                           np.where(fired_after[:, j], spike_at[:, j], NAN)) for j in range(n_pulses)]
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
    # the light's own mark on the trace, if any: blanked before anything is measured
    widths = np.array([float(st.widths_s[i][0]) if i < len(st.widths_s) and len(st.widths_s[i]) else NAN
                       for i in used])
    light_width = float(np.nanmedian(widths)) if light and np.isfinite(widths).any() else NAN
    light_blank, why_blank = light_artifact(t, rows, ons, widths, fs, s) if light else (0.0, "")
    out.values["light_artifact_blank_ms"] = light_blank * 1e3 if light_blank else NAN
    if light_blank:
        out.flags.append(Flag("light artifact", why_blank, source="Clampwise", severity="note"))
        out.choose("light blank", f"{light_blank * 1e3:.1f} ms after each light pulse", why_blank.split(":")[0])
    extra = {"light_width_s": light_width, "light_blank_s": light_blank}

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
            up = _measure(tt, yy, t0, fs, s, baseline=None, polarity=1, next_t=nxt, light=light, **extra)
            down = _measure(tt, yy, t0, fs, s, baseline=None, polarity=-1, next_t=nxt, light=light, **extra)
            polarity = 1 if abs(up.amplitude) >= abs(down.amplitude) else -1
            out.choose("response direction", "up" if polarity > 0 else "down",
                       "the larger deflection of pulse 1 on the average")
        p = _measure(tt, yy, t0, fs, s, baseline=None if (j == 0 or moving) else p1_base, polarity=polarity,
                     next_t=nxt, light=light, **extra)
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
            blank = max(s["blank_ms"] / 1e3, light_blank, 0.0)
            r0 = int(np.searchsorted(t, ons[k, j] + blank))
            pk = int(np.searchsorted(t, ons[k, j] + p.peak_time_s)) + 1
            # the onset is a departure from the level just before this pulse (a
            # later pulse can start on a residue its tail model did not take off)
            pre0 = int(np.searchsorted(t, ons[k, j] - 0.002))
            local = float(np.median(dev[pre0:max(r0 - 1, pre0 + 1)])) if j > 0 else 0.0
            sustain = max(1, int(round(0.5e-3 * fs)))
            on = kinetics.find_onset(dev - local, r0, pk, s["onset_noise_sd"] * noise, sustain)
            if on is not None:
                lat = float(t[on] - ons[k, j])
                if not (light and lat < s["light_min_latency_ms"] / 1e3):
                    chosen = s.get("onset_definition", kinetics.ONSET_NOISE)
                    if chosen != kinetics.ONSET_NOISE and pk - 1 > r0:
                        # the sweep's own peak near the average's, then the chosen rule on this sweep
                        peak_k = r0 + int(np.argmax((dev - local)[r0:pk]))
                        lat = kinetics.onset_by(chosen, t, dev - local, r0, peak_k, 0.0, sustain) - ons[k, j]
                    if np.isfinite(lat):
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
        p.spike_probability = float(fired_after[:, j].mean()) if not vc else NAN

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
        "p1_decay_tau_weighted": p1.decay_tau_weighted_s * 1e3,
        "p1_decay_two_exponentials": ("yes" if p1.decay_two_exponentials else "no") if np.isfinite(
            p1.decay_tau_weighted_s) else "",
        "p1_latency_noise": p1.onset_noise_s * 1e3, "p1_latency_10pct": p1.onset_10pct_s * 1e3,
        "p1_latency_foot": p1.onset_foot_s * 1e3,
        "p1_half_width": p1.half_width_s * 1e3, "p1_integral": p1.integral / (1e3 if vc else 1.0),
        "p1_time_to_half_peak": p1.time_to_half_peak_s * 1e3, "p1_decay_80_20": p1.decay_80_20_s * 1e3,
        "p1_initial_slope": p1.initial_slope,
        "p1_success_rate": p1.success_rate, "p1_potency": p1.potency, "p1_cv": p1.cv, "p1_inv_cv2": p1.inv_cv2,
        "ppr": ratio(p2.amplitude, p1.amplitude) if p2 else NAN,
        "units": units, "integral_units": "pC" if vc else "mV*ms",
    })
    # the mean of the sweeps' own ratios, beside the ratio of the means it is biased above (Kim & Alger 2001)
    out.values["ppr_mean_of_ratios"] = NAN
    if p2 is not None:
        both = member[:, 0] & member[:, 1] & np.isfinite(per_sweep[:, 0]) & np.isfinite(per_sweep[:, 1])
        a1, a2 = polarity * per_sweep[both, 0], polarity * per_sweep[both, 1]
        ok = a1 > (thresholds[0] if np.isfinite(thresholds[0]) else 0.0)
        if int(ok.sum()) >= 3:
            out.values["ppr_mean_of_ratios"] = float(np.mean(a2[ok] / a1[ok]))
    if p2 is not None and not moving:
        # pulse 2 against its own pre-pulse baseline on the plain average: the raw ratio, for comparison
        raw = _measure(t, rows[member[:, 1]].mean(axis=0), float(fixed[1]), fs, s, baseline=None, polarity=polarity,
                       next_t=float(fixed[2]) if n_pulses > 2 else np.inf, light=light, **extra)
        out.values["ppr_raw"] = ratio(raw.amplitude, p1.amplitude)
    else:
        out.values["ppr_raw"] = NAN
    # a direct photocurrent: the recorded cell itself answers the light (it expresses the opsin, or its membrane
    # is lit): the response begins sooner than any synapse can answer, is still at its height when the light goes
    # off, and peaks there or just after (Turesson et al. 2013: direct responses have a short latency, rise
    # during the light and decay from its offset). Not a synaptic response, and not a light artifact (one of
    # those is gone when the light goes off; see light_artifact)
    direct = bool(light and p1.detected and np.isfinite(light_width) and np.isfinite(p1.onset_raw_s)
                  and p1.onset_raw_s < s["light_min_latency_ms"] / 1e3
                  and np.isfinite(p1.at_light_end) and np.isfinite(p1.peak_time_s)
                  and p1.peak_time_s <= light_width + 3e-3
                  and p1.at_light_end >= 0.5 * abs(p1.amplitude))
    # or the light makes the cell fire sooner than a synapse could: the opsin's own depolarization reaching
    # threshold (a cell that fires 1 ms after every pulse at its depolarized steps)
    fired_at = spike_at[fired_after[:, 0], 0] if not vc else np.array([])
    direct_spikes = bool(light and fired_at.size >= max(1, len(sweeps) // 2)
                         and float(np.median(fired_at)) < s["light_min_latency_ms"] / 1e3)
    out.values["direct_photocurrent"] = "yes" if (direct or direct_spikes) else "no"
    if direct:
        out.flags.append(Flag("direct photocurrent", f"the recorded cell itself answers the light: the response "
                              f"begins {p1.onset_raw_s * 1e3:.2f} ms after the light comes on (sooner than a synapse "
                              f"can answer), is still at {p1.at_light_end / abs(p1.amplitude):.0%} of its peak when the "
                              f"light goes off ({light_width * 1e3:.0f} ms) and peaks {p1.peak_time_s * 1e3:.1f} ms "
                              "after the light's onset. This cell expresses the opsin, or its own membrane is lit; "
                              "the response is not synaptic, so it says nothing about a connection", "",
                              "Turesson, Rodriguez-Sierra & Pare 2013"))
    elif direct_spikes:
        out.flags.append(Flag("direct photocurrent", f"the light makes the recorded cell fire "
                              f"{float(np.median(fired_at)) * 1e3:.2f} ms after it comes on (median of {fired_at.size} "
                              f"sweep(s)), sooner than a synapse can answer ({s['light_min_latency_ms']:g} ms): the "
                              "cell's own photocurrent reaches threshold. This cell expresses the opsin; its firing "
                              "says nothing about a connection", "", "Turesson, Rodriguez-Sierra & Pare 2013"))
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
    out.tables["pulses"] = [_pulse_row(p, p1.amplitude, vc, np.where(fired_after[:, j], spike_at[:, j], NAN))
                            for j, p in enumerate(pulses)]
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
    if cond is not None:
        _by_condition(recording, context, s, cond, out)
    if light and not context.shared.get("by_pattern") and not context.shared.get("by_condition"):
        known = _by_pattern(recording, context, s, all_sweeps if cond is None else sweeps, out)
        out.values["region_known"] = {"no log": "no", "not sure": "no", "one pattern": "yes",
                                      "by pattern": "yes"}.get(known, "no")
        if known in ("no log", "not sure") and is_mapping(recording) and recording.n_sweeps > 1:
            # a grid map or a connectivity run lights a different region on each sweep: averaged together the
            # sweeps mean nothing, and no log says which region each one saw (2026-11-10, a grid map run)
            why = ("no Polygon log covers this recording" if known == "no log"
                   else "the Polygon's log does not say for sure")
            out.flags.append(Flag("region unknown", f"which region lit each sweep is unknown ({why}): a "
                                  f"{recording.protocol or 'mapping'} protocol lights a different region on each "
                                  f"sweep, so its {recording.n_sweeps} sweeps are not averaged into one number. "
                                  "Each sweep's own response is in the sweeps table; record the light's log with "
                                  "the recording (Clampwise's patterned light writes polygon_log.csv beside it) "
                                  "or type each sweep's region", source="Clampwise"))
            for key in ("amplitude", "p1_amplitude", "p1_latency_median", "p1_jitter", "p1_latency_average",
                        "p1_success_rate", "p1_potency", "ppr", "ppr_raw", "p1_integral", "p1_cv", "p1_inv_cv2"):
                out.values[key] = NAN
            out.values["headline_is"] = ("not given: which region lit each sweep is unknown, so the sweeps are not "
                                         "averaged (see the flag)")
            out.choose("headline", out.values["headline_is"], "a mapping protocol with no record of the regions lit")
    # what to draw: each pulse's peak and the fitted decay of pulse 1
    for p in pulses:
        if np.isfinite(p.peak_time_s):
            out.drawables.append(Drawable("marker", f"pulse {p.number} peak", x=np.array([p.time_s + p.peak_time_s]),
                                          y=np.array([(p1_base if p1_base is not None else 0) + p.amplitude]),
                                          domain="time"))
    return out
