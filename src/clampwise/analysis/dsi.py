"""DSI and DSE from evoked responses (B5): a response evoked again and again,
and between them the cell stepped to about 0 mV for seconds.

Every evoked response is measured on its own sweep, on the recording's time
axis, with the evoked analysis's single-response measurement (baseline before
the pulse, peak after the artifact). Responses during a step, or before the
holding current has settled after it (at least 0.2 s; Analysis Decisions,
2026-09-29), are left out.

For each step (an "episode"), after Pitler & Alger (1992, *J Neurosci*
12:4122; to verify):

- baseline: the mean of the last 5 responses before the step;
- normalized responses: each over that baseline;
- suppression: 100 (1 - the mean of the first 3 normalized responses after);
- time to recovery: from the step's end until a 3-response running mean is
  back at 90% of baseline, each mean at its responses' mean time and the
  crossing interpolated (the window's last response read two intervals late);
- recovery tau: 1 - S exp(-t / tau) fitted to the normalized responses. The
  recording's tau and time to recovery average only episodes suppressed by at
  least 20%: a dip the size of the noise cannot be timed (an 11% dip in 5%
  noise fitted 213 s for a true 15 s).

When the step's duration (or level) changes between episodes, suppression is
given per duration and the **threshold** is the shortest duration whose
suppression is significant: the first responses after each step against the
baseline responses (each normalized to its own episode's baseline), one-sided
Mann-Whitney, p < 0.05, **and** at least 20% (with
one trial a 5% dip on a quiet cell already separates 3 responses from 5). The
threshold read from every duration at once is given beside it: each
duration's mean suppression made to rise with duration (pool-adjacent-
violators) and where it crosses 20%, on log duration.

**The time course**: every response from the baseline
through the first ``recovery_responses`` after the step, as % of its own episode's
baseline, one line per step; and their average, response by response, with
its SEM (the "% average eIPSC over time").

**Fourier transform** (a standard one),
beside the peak: the discrete Fourier transform of each response's window,
as it is (no taper, no zero padding, no averaging of segments), the same
length N for every response so every spectrum has the same frequencies
(fs / N apart). The window is the response itself (from the end of its own
stimulus artifact, else the blank, for ``power_ms``, its pre-pulse level
taken off) or the stretch
between responses (the ongoing activity, its mean taken off); N is the
median length available, and a response with less is left out. Kept for
every response at every frequency up to ``spectrum_to_hz``: the amplitude
(single-sided, 2|X|/N, in the signal's units), the power spectral density
(2|X|^2 / (fs N)) and the phase. The band power (``power_low_hz`` to
``power_high_hz``: the density summed times fs / N) is each response's
number, as % of its episode's baseline like the peak. The mean spectra of the
baseline, the suppressed responses and the end of the recovery are drawn
together.

The same analysis measures DSE (an EPSC suppressed instead of an IPSC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.recording import ClampMode, Recording
from . import evoked
from .events import SETTLE_MIN_S, dsi_steps, settle_s
from .registry import Output, Setting, load, register
from .result import Flag, Result
from .tools import fits, ledger

NAN = float("nan")
RULES = ("significance", "percent")
POWER_WINDOWS = ("response", "between responses")


# ------------------------------------------------------------------ shared with DSI from events

@dataclass
class Trial:
    """One step and the responses (or event rates) around it, normalized to its own baseline."""
    group: float                         # its duration (s) or level (mV)
    suppression_pct: float
    post_normalized: np.ndarray          # the measure after the step, over the baseline
    baseline_normalized: np.ndarray      # the baseline's own values, over the baseline (centred on 1)
    baseline_cv: float = NAN


@dataclass
class GroupTest:
    group: float
    suppression_pct: float
    p: float
    trials: int
    n_baseline: int
    n_post: int
    significant: bool


def group_tests(trials: list[Trial], *, rule: str, alpha: float, min_suppression_pct: float,
                threshold_pct: float = 20.0, cv_multiple: float = 2.0) -> list[GroupTest]:
    """Suppression and its significance for each duration (or level)."""
    from scipy.stats import mannwhitneyu

    groups: dict[float, list[Trial]] = {}
    for tr in trials:
        if np.isfinite(tr.suppression_pct):
            groups.setdefault(tr.group, []).append(tr)
    out = []
    for g in sorted(groups):
        trs = groups[g]
        mean = float(np.mean([t.suppression_pct for t in trs]))
        base = np.concatenate([t.baseline_normalized for t in trs]) if trs else np.array([])
        post = np.concatenate([t.post_normalized for t in trs]) if trs else np.array([])
        base, post = base[np.isfinite(base)], post[np.isfinite(post)]
        p = float(mannwhitneyu(post, base, alternative="less").pvalue) if base.size >= 2 and post.size >= 2 else NAN
        if rule == "percent":
            cvs = [t.baseline_cv for t in trs if np.isfinite(t.baseline_cv)]
            floor = max(threshold_pct, cv_multiple * 100 * (float(np.mean(cvs)) if cvs else 0.0))
            significant = bool(mean > floor)
        else:
            significant = bool(np.isfinite(p) and p < alpha and mean > min_suppression_pct)
        out.append(GroupTest(g, mean, p, len(trs), int(base.size), int(post.size), significant))
    return out


def rule_words(rule: str, key: str, alpha: float, min_suppression_pct: float, what: str,
               threshold_pct: float = 20.0, cv_multiple: float = 2.0) -> str:
    if rule == "percent":
        return f"the shortest {key} with suppression above {threshold_pct:g}% and {cv_multiple:g} x the baseline CV"
    floor = f" and above {min_suppression_pct:g}%" if min_suppression_pct > 0 else ""
    return f"the shortest {key} with significant suppression (one-sided Mann-Whitney, {what}, p < {alpha:g}{floor})"


def rising(values) -> np.ndarray:
    """The closest non-decreasing sequence (pool adjacent violators)."""
    blocks: list[list[float]] = []
    for v in map(float, values):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([[v] * int(n) for v, n in blocks]) if blocks else np.array([])


def curve_threshold(tests: list[GroupTest], floor_pct: float) -> tuple[float, float]:
    """(the duration where the rising suppression curve crosses ``floor_pct``,
    interpolated on log duration; the longest duration's suppression). The
    first duration when it starts above the floor; NaN when it never reaches it.
    The significance threshold turns on three responses at one duration (on a
    simulated cell it ranged 0.5-4 s across seeds); this one spread 0.7 s."""
    pts = [(t.group, t.suppression_pct) for t in tests if np.isfinite(t.suppression_pct) and t.group > 0]
    longest = pts[-1][1] if pts else NAN
    if len(pts) < 2:
        return NAN, longest
    d = np.array([p[0] for p in pts], float)
    r = rising([p[1] for p in pts])
    if r[0] >= floor_pct:
        return float(d[0]), longest
    for k in range(1, d.size):
        if r[k - 1] < floor_pct <= r[k]:
            frac = (floor_pct - r[k - 1]) / (r[k] - r[k - 1])
            return float(np.exp(np.log(d[k - 1]) + frac * np.log(d[k] / d[k - 1]))), longest
    return NAN, longest


def recovery_time(times_s: np.ndarray, normalized: np.ndarray, fraction: float, k: int) -> float:
    """Seconds from the step's end until a ``k``-value running mean is back at
    ``fraction`` of baseline: each mean at its values' mean time, the crossing
    interpolated between them."""
    if normalized.size < k:
        return NAN
    avg = np.convolve(normalized, np.ones(k) / k, mode="valid")
    centres = np.convolve(times_s, np.ones(k) / k, mode="valid")
    back = np.flatnonzero(avg >= fraction)
    if not back.size:
        return NAN
    i = int(back[0])
    if i > 0 and avg[i] > avg[i - 1]:
        frac = (fraction - avg[i - 1]) / (avg[i] - avg[i - 1])
        return float(centres[i - 1] + frac * (centres[i] - centres[i - 1]))
    return float(centres[i])


THRESHOLD_SETTINGS = (
    Setting("alpha", 0.05, "Significance level", "Suppression at the threshold duration must be significant at this "
            "level (one-sided Mann-Whitney test).", "", "", minimum=0.0001, maximum=0.5),
    Setting("min_suppression_pct", 20.0, "Suppression at least (%)", "Also required for the threshold and for timing "
            "the recovery (0 disables it).", "%", "", minimum=0.0, maximum=100.0),
    Setting("threshold_rule", "significance", "Threshold rule", "Significance (default), or a percentage "
            "rule (suppression above a percentage and above twice the baseline CV).", choices=RULES,
            advanced=True),
    Setting("threshold_pct", 20.0, "Percentage rule: above (%)", "", "%", advanced=True, minimum=0.0),
    Setting("threshold_cv_multiple", 2.0, "Percentage rule: and above this many baseline CVs", "", advanced=True,
            minimum=0.0),
)


# ------------------------------------------------------------------ evoked DSI

@dataclass
class Episode:
    sweep: int
    start_s: float                       # on the recording's time axis
    end_s: float
    level_mv: float
    settle_s: float = NAN
    baseline: float = NAN
    baseline_cv: float = NAN
    suppression_pct: float = NAN
    recovery_time_s: float = NAN
    tau_s: float = NAN
    fit_s: float = NAN
    fit_r2: float = NAN
    power_suppression_pct: float = NAN
    recovered_pct: float = NAN
    baseline_gap_s: float = NAN          # from the last baseline response to the step
    measurable: bool = False
    times_s: np.ndarray = field(default_factory=lambda: np.array([]))
    normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_normalized: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


SETTINGS = (
    Setting("baseline_responses", 5, "Baseline: last responses before the step", "", "", "Pitler & Alger 1992",
            minimum=1),
    Setting("post_responses", 3, "Suppression: first responses after the step", "", "", "Pitler & Alger 1992",
            minimum=1),
    *THRESHOLD_SETTINGS[:2],
    Setting("direction", "auto", "Response direction", "Auto: the sign of the responses' charge before the first "
            "step (charge is used because a stimulus artifact can exceed the response in height but not in area).", "", "Clampwise",
            choices=("auto", "inward", "outward")),
    Setting("recovery_responses", 10, "Time course: responses after the step", "Each step's responses, as % of "
            "its baseline, are plotted and averaged from the baseline to this many responses after the step.", "",
            "", minimum=1),
    Setting("power_window", "response", "Power: of which window", "The response itself (from the end of the artifact "
            "blank to the set length after the pulse), or the interval between responses (ongoing activity).", "",
            "Clampwise", choices=POWER_WINDOWS),
    Setting("power_low_hz", 1.0, "Power band from (Hz)", "", "Hz", "Clampwise", minimum=0.0),
    Setting("power_high_hz", 100.0, "Power band to (Hz)", "", "Hz", "Clampwise", minimum=1.0),
    Setting("power_ms", 100.0, "Fourier: response window (ms)", "Length of the window after the pulse used for the "
            "Fourier transform; frequency resolution is 1000 / this value (10 Hz for 100 ms).", "ms", "Clampwise", minimum=5.0,
            maximum=1000.0),
    Setting("spectrum_to_hz", 500.0, "Fourier: keep frequencies up to (Hz)", "Each response's amplitude, density "
            "and phase are stored, plotted and exported up to this frequency.", "Hz", "Clampwise", advanced=True,
            minimum=1.0),
    Setting("recovery_fraction", 0.9, "Recovered at (fraction of baseline)", "", "", advanced=True, minimum=0.1,
            maximum=1.0),
    Setting("recovery_average", 3, "Recovery: running mean of responses", "", "", advanced=True, minimum=1),
    *THRESHOLD_SETTINGS[2:],
)

OUTPUTS = (
    Output("suppression_pct", "Mean suppression over the episodes: 100 (1 - the mean of the first responses after "
           "each step over the mean of the last before it).", "%", "Pitler & Alger 1992",
           label="Suppression", headline=True),
    Output("recovery_tau_s", "Mean recovery time constant (1 - S exp(-t/tau) on the normalized responses), over the "
           "episodes suppressed by at least the minimum.", "s", label="Recovery tau", headline=True),
    Output("time_to_recovery_s", "Mean time from the step's end until a running mean of responses is back at 90% "
           "of baseline, over the episodes suppressed by at least the minimum.", "s", label="Time to recovery"),
    Output("threshold", "The shortest step duration (or level) whose suppression is significant and at least the "
           "minimum; blank when none is.", "s or mV", "", label="Threshold",
           headline=True),
    Output("threshold_interpolated_s", "Where each duration's mean suppression, made to rise with duration, "
           "crosses the minimum suppression, on log duration.", "s", label="Threshold from the curve"),
    Output("suppression_longest_pct", "Suppression at the longest step duration (the classic 5 s DSI).", "%",
           label="Suppression at the longest step"),
    Output("no_threshold", "1 when steps of several durations were tried and none suppressed significantly: the "
           "threshold is longer than the longest tried.", "", label="No threshold reached"),
    Output("power_suppression_pct", "Mean over the episodes of 100 (1 - the band power of the first responses after "
           "the step over that of the last before it), from the Fourier transform of the set window.", "%",
           label="Suppression (power)", headline=True),
    Output("recovered_pct", "Mean over the episodes of the last baseline-count responses of the time course, % of "
           "the baseline: how far the response has come back by then.", "%",
           label="Recovered by the window's end"),
    Output("n_episodes", "Steps measured.", "count", label="Episodes", headline=True),
    Output("n_responses", "Evoked responses measured.", "count", label="Responses"),
)


def _responses(recording: Recording, stim, steps_settle, s_evoked: dict, polarity: int) -> list[tuple]:
    """(sweep, pulse, time on the recording's axis, amplitude, time in the sweep, the next pulse, charge, where
    an opposite-going stimulus artifact ends, the peak's time; both after the pulse, s) of every response outside
    the steps. The charge is the area from the pulse to the response window's end, whatever the direction: what
    tells an IPSC from a taller but brief stimulus artifact."""
    t = recording.time_s
    fs = recording.sample_rate_hz
    starts = np.asarray(recording.sweep_start_times_s, float)
    out = []
    for sw in range(min(recording.n_sweeps, len(stim.times_s))):
        times = np.asarray(stim.times_s[sw], float)
        if not times.size:
            continue
        y = recording.sweep(sw)
        light = stim.kind == "light"
        for k, t0 in enumerate(times):
            if any(ss == sw and a - 0.01 <= t0 <= b + settle for ss, a, b, _, settle in steps_settle):
                continue
            nxt = float(times[k + 1]) if k + 1 < times.size else np.inf
            p = evoked._measure(t, y, float(t0), fs, s_evoked, baseline=None, polarity=polarity, next_t=nxt,
                                light=light)
            i0 = int(t0 * fs)
            i1 = min(y.size, int((t0 + min(s_evoked["response_end_ms"] / 1e3, nxt - t0)) * fs))
            dev = y[i0:i1] - p.baseline if np.isfinite(p.baseline) else np.array([])
            charge = float(dev.sum() / fs) if dev.size else NAN
            artifact = NAN
            if dev.size and np.isfinite(p.peak_time_s):
                # an artifact going the other way from the response: where it is last on that side before the peak
                pk = min(int(p.peak_time_s * fs), dev.size)
                other = np.flatnonzero(polarity * dev[:pk] < -3.0 * p.noise_sd)
                if other.size:
                    artifact = float((other[-1] + 1) / fs)
            out.append((sw, k, float(starts[sw] if sw < starts.size else 0.0) + float(t0), p.amplitude, float(t0),
                        nxt, charge, artifact, p.peak_time_s))
    return out


def _direction(rows: list[tuple], before_s: float, setting: str) -> tuple[int, str]:
    """The responses' direction: as set, else the sign of the median charge of the responses before the first
    step (a stimulus artifact can be taller than the response, but it is brief)."""
    if setting in ("inward", "outward"):
        return (-1 if setting == "inward" else 1), "set"
    q = [r[6] for r in rows if r[2] < before_s and np.isfinite(r[6])]
    return (1 if q and np.median(q) > 0 else -1), "the sign of the median charge of the responses before the " \
        "first step (not their tallest point: a stimulus artifact can be taller, but it is brief)"


@dataclass
class Fourier:
    """The transform of every response's window (rows: responses; NaN where a
    response's window was too short)."""
    n: int                               # samples transformed
    freqs: np.ndarray                    # kept frequencies, fs / n apart
    amplitude: np.ndarray                # single-sided, signal units
    density: np.ndarray                  # signal units^2 / Hz
    phase_deg: np.ndarray
    band_power: np.ndarray               # signal units^2
    left_out: int = 0


def fourier(items: list[tuple], s: dict, s_ev: dict) -> Fourier:
    """The standard (discrete) Fourier transform of each response's window, the
    same length for all. ``items``: (recording, response row, its steps) in
    time order, so one sweep is read at a time; the recordings may differ (a
    cell's files)."""
    fs = items[0][0].sample_rate_hz if items else 1.0
    windows, here, y = [], None, None
    for rec, (sw, _, _, _, t0, nxt, *more), steps_settle in items:
        if here != (id(rec), sw):
            here, y = (id(rec), sw), np.asarray(rec.sweep(sw), float)
        if s["power_window"] == "response":
            # from where the response's own stimulus artifact ends, else the blank
            art = more[1] if len(more) > 1 and np.isfinite(more[1]) else 0.0
            a = t0 + max(s_ev["blank_ms"] / 1e3, art)
            b = min(a + s["power_ms"] / 1e3, nxt - 0.001)
        else:
            later_steps = [st for ss, st, _, _, _ in steps_settle if ss == sw and st > t0]
            a, b = t0 + s_ev["response_end_ms"] / 1e3, min([nxt - 0.001, y.size / fs] + later_steps)
        ia = int(round(a * fs))
        windows.append((rec, sw, ia, max(0, min(int(b * fs), y.size) - ia), t0))
    usable = [w[3] for w in windows if w[3] >= int(0.01 * fs)]
    n = int(np.median(usable)) if usable else 0
    f = np.fft.rfftfreq(max(n, 2), 1.0 / fs)
    keep = f <= s["spectrum_to_hz"]
    band = (f >= s["power_low_hz"]) & (f <= s["power_high_hz"])
    shape = (len(items), int(keep.sum()))
    out = Fourier(n, f[keep], np.full(shape, NAN), np.full(shape, NAN), np.full(shape, NAN), np.full(len(items), NAN))
    one_sided = np.full(f.size, 2.0)
    one_sided[0] = 1.0
    if n % 2 == 0:
        one_sided[-1] = 1.0                                 # the Nyquist bin has no mirror
    here = None
    for k, (rec, sw, ia, length, t0) in enumerate(windows):
        if n == 0 or length < n:
            out.left_out += 1
            continue
        if here != (id(rec), sw):
            here, y = (id(rec), sw), np.asarray(rec.sweep(sw), float)
        seg = y[ia:ia + n]
        if s["power_window"] == "response":
            r0, r1 = int((t0 + s_ev["baseline_start_ms"] / 1e3) * fs), int((t0 + s_ev["baseline_end_ms"] / 1e3) * fs)
            seg = seg - float(np.median(y[max(r0, 0):max(r1, r0 + 1)]))
        else:
            seg = seg - seg.mean()
        x = np.fft.rfft(seg)
        density = one_sided * np.abs(x) ** 2 / (fs * n)
        out.amplitude[k] = (one_sided * np.abs(x) / n)[keep]
        out.density[k] = density[keep]
        out.phase_deg[k] = np.degrees(np.angle(x))[keep]
        out.band_power[k] = float(density[band].sum() * fs / n)
    return out


def band_bins(ft: Fourier, s: dict) -> int:
    return int(np.sum((ft.freqs >= s["power_low_hz"]) & (ft.freqs <= s["power_high_hz"])))


def _mean_sem(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else NAN, float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else NAN)


@register("dsi", "DSI / DSE from evoked responses", version=5, types={"dsi"}, settings=SETTINGS, outputs=OUTPUTS,
          requires=("baseline",),
          description="Suppression of the evoked response after each depolarizing step, its recovery, and the "
                      "shortest step that suppresses it.",
          source="Pitler & Alger 1992")
def run(recording: Recording, context, s) -> Result | None:
    stim = context.stimulus
    steps = dsi_steps(recording)
    if not steps or stim is None or not len(stim):
        return None
    out = Result("dsi", 1)
    fs = recording.sample_rate_hz
    s_ev, _ = load()["evoked"].resolve({})
    s_ev = dict(s_ev, follow_slow=False)
    steps_settle = [(sw, a, b, lv, settle_s(recording.sweep(sw), fs, a, b)) for sw, a, b, lv in steps]
    starts = np.asarray(recording.sweep_start_times_s, float)
    first_step = float(starts[steps[0][0]] if steps[0][0] < starts.size else 0.0) + steps[0][1]
    rows = _responses(recording, stim, steps_settle, s_ev, -1)
    polarity, why = _direction(rows, first_step, s["direction"])
    out.choose("response direction", "outward (up)" if polarity > 0 else "inward (down)", why)
    if polarity > 0:
        rows = _responses(recording, stim, steps_settle, s_ev, 1)
    rows.sort(key=lambda r: r[2])
    items =[(recording, r, steps_settle) for r in rows]
    absolute = sorted((float(starts[sw] if sw < starts.size else 0.0) + a, float(starts[sw] if sw < starts.size else 0.0)
                       + b, sw, lv, st) for sw, a, b, lv, st in steps_settle)
    summarize(out, rows, items, absolute, s, s_ev, recording.sample_rate_hz, recording.signal_units)
    return out


def summarize(out: Result, rows: list[tuple], items: list[tuple], absolute: list[tuple], s: dict, s_ev: dict,
              fs: float, units: str, files: list[str] | None = None) -> None:
    """Every episode from the responses (``rows``: sweep, pulse, time on one
    clock, amplitude, time in the sweep, next pulse) and the steps on the same
    clock (``absolute``: start, end, sweep or file, level, settling): the
    suppression, the recovery, the time course, the Fourier transform, the
    threshold. ``files``: each response's file, for a cell."""
    def where(k: int, r: tuple) -> dict:
        return {"file": files[k], "sweep": r[0]} if files else {"sweep": r[0]}

    times = np.array([r[2] for r in rows])
    amps = np.array([r[3] for r in rows], float)
    ft = fourier(items, s, s_ev)
    pw = ft.band_power
    out.choose("fourier", f"{s['power_window']} windows of {ft.n} samples ({1e3 * ft.n / fs:.4g} ms), "
               f"frequencies {fs / max(ft.n, 1):.4g} Hz apart; band power {s['power_low_hz']:g}-"
               f"{s['power_high_hz']:g} Hz", "the discrete Fourier transform of each window as it is (no taper, "
               "no padding), " + ("from the end of each response's stimulus artifact, its pre-pulse level taken off"
                                  if s["power_window"] == "response"
                                  else "its mean taken off"))
    if ft.left_out:
        out.notes.append(f"{ft.left_out} response(s) with a window shorter than the others' ({ft.n} samples) have "
                         "no Fourier transform")
    if ft.freqs.size and band_bins(ft, s) == 0:
        out.flags.append(Flag("empty band", f"no frequency of the transform ({fs / max(ft.n, 1):.4g} Hz apart) falls "
                              f"in {s['power_low_hz']:g}-{s['power_high_hz']:g} Hz", "", "", severity="note"))
    course, phases = [], {"baseline": [], "suppressed": [], "recovered": []}
    ledger.note("each evoked response measured on its own sweep (the evoked analysis's single-response "
                "measurement); none during a step or before the current settled after it", "DSI")
    episodes = []
    for i, (a, b, sw, lv, st) in enumerate(absolute):  # sw: the sweep, or the step's file for a cell
        e = Episode(sw, a, b, lv, st)
        prev_end = absolute[i - 1][1] if i else -np.inf
        next_start = absolute[i + 1][0] if i + 1 < len(absolute) else np.inf
        pre = (times < a) & (times > prev_end) & np.isfinite(amps)
        base = amps[pre][-s["baseline_responses"]:]
        post = (times > b) & (times < next_start) & np.isfinite(amps)
        if base.size == 0 or not post.any() or np.mean(base) == 0:
            episodes.append(e)
            continue
        e.baseline = float(np.mean(base))
        e.baseline_gap_s = float(a - times[pre][-1])
        e.baseline_normalized = base / e.baseline
        e.baseline_cv = float(np.std(base, ddof=1) / abs(e.baseline)) if base.size > 1 else NAN
        e.times_s = times[post] - b
        e.normalized = amps[post] / e.baseline
        e.suppression_pct = 100.0 * (1.0 - float(np.mean(e.normalized[:s["post_responses"]])))
        e.measurable = bool(e.suppression_pct >= s["min_suppression_pct"])
        e.recovery_time_s = recovery_time(e.times_s, e.normalized, s["recovery_fraction"], s["recovery_average"])
        # the time course, peak and power, each as % of this episode's baseline
        nb = s["baseline_responses"]
        before = np.flatnonzero(pre)[-nb:]
        after = np.flatnonzero(post)[:s["recovery_responses"]]
        bp = float(np.nanmean(pw[before])) if np.isfinite(pw[before]).any() else NAN
        first = after[:s["post_responses"]]
        e.power_suppression_pct = 100.0 * (1.0 - float(np.nanmean(pw[first])) / bp) \
            if bp and np.isfinite(pw[first]).any() else NAN
        e.recovered_pct = 100.0 * float(np.mean(amps[after[-nb:]])) / e.baseline if after.size else NAN
        for j, q in enumerate(np.r_[before, after]):
            course.append({"episode": len(episodes) + 1, "duration_s": e.duration_s,
                           "response": j - before.size if j < before.size else j - before.size + 1,
                           "from_step_s": times[q] - (a if j < before.size else b), "amplitude": amps[q],
                           "pct": 100.0 * amps[q] / e.baseline, "power": pw[q],
                           "power_pct": 100.0 * pw[q] / bp if bp else NAN})
        phases["baseline"] += list(before)
        phases["suppressed"] += list(first)
        phases["recovered"] += list(after[-nb:])
        fit = fits.suppression_recovery(e.times_s, e.normalized, purpose="DSI recovery")
        if fit.ok:
            e.tau_s, e.fit_s, e.fit_r2 = fit["tau"], fit["s"], fit.r2
        episodes.append(e)
    sup = [e.suppression_pct for e in episodes if np.isfinite(e.suppression_pct)]
    timed = [e for e in episodes if e.measurable]
    out.values.update({
        "suppression_pct": float(np.mean(sup)) if sup else NAN,
        "recovery_tau_s": float(np.mean([e.tau_s for e in timed if np.isfinite(e.tau_s)])) if any(
            np.isfinite(e.tau_s) for e in timed) else NAN,
        "time_to_recovery_s": float(np.mean([e.recovery_time_s for e in timed if np.isfinite(e.recovery_time_s)]))
        if any(np.isfinite(e.recovery_time_s) for e in timed) else NAN,
        "n_episodes": len(episodes), "n_responses": len(rows), "units": units,
    })
    for key in ("power_suppression_pct", "recovered_pct"):
        out.values[key] = _mean_sem(np.array([getattr(e, key) for e in episodes]))[0]
    mean_rows = []
    for k in sorted({r["response"] for r in course}):
        at = [r for r in course if r["response"] == k]
        pct, pct_sem = _mean_sem(np.array([r["pct"] for r in at]))
        ppct, ppct_sem = _mean_sem(np.array([r["power_pct"] for r in at]))
        mean_rows.append({"response": k, "from_step_s": float(np.mean([r["from_step_s"] for r in at])), "pct": pct,
                          "pct_sem": pct_sem, "power_pct": ppct, "power_pct_sem": ppct_sem, "n": len(at)})
    out.tables["course"] = course
    out.tables["course_mean"] = mean_rows
    if ft.freqs.size and phases["baseline"]:
        def mean(a, v):
            return np.nanmean(a[v], axis=0) if v and np.isfinite(a[v]).any() else np.full(ft.freqs.size, NAN)
        means = {f"{what}_{k}": mean(a, v) for what, a in (("psd", ft.density), ("amp", ft.amplitude))
                 for k, v in phases.items()}
        out.tables["spectra"] = [{"frequency_hz": float(fr), **{k: float(m[i]) for k, m in means.items()}}
                                 for i, fr in enumerate(ft.freqs)]
    out.tables["fourier"] = [{**where(k, r), "pulse": r[1] + 1, "time_s": r[2], "frequency_hz": float(fr),
                              "fft_amplitude": float(ft.amplitude[k, i]), "fft_density": float(ft.density[k, i]),
                              "fft_phase_deg": float(ft.phase_deg[k, i])}
                             for k, r in enumerate(rows) if np.isfinite(ft.band_power[k])
                             for i, fr in enumerate(ft.freqs)]
    # every episode is timed, suppressed too little to time, or not measured: the counts sum to the episodes
    if len(timed) < len(sup):
        out.notes.append(f"{len(sup) - len(timed)} episode(s) suppressed by less than "
                         f"{s['min_suppression_pct']:g}%: their recovery cannot be timed and is left out of the "
                         "recording's tau and time to recovery (each episode keeps its own)")
    if len(sup) < len(episodes):
        out.notes.append(f"{len(episodes) - len(sup)} episode(s) have no response to measure before or after their "
                         "step (or a baseline of exactly zero): no suppression is read from them")
    durations = sorted({round(e.duration_s, 2) for e in episodes})
    levels = sorted({round(e.level_mv) for e in episodes})
    key = "duration" if len(durations) > 1 else ("level" if len(levels) > 1 else "")
    tests: list[GroupTest] = []
    if key:
        trials = [Trial(round(e.duration_s, 2) if key == "duration" else round(e.level_mv), e.suppression_pct,
                        e.normalized[:s["post_responses"]], e.baseline_normalized, e.baseline_cv)
                  for e in episodes if np.isfinite(e.suppression_pct)]
        tests = group_tests(trials, rule=s["threshold_rule"], alpha=s["alpha"],
                            min_suppression_pct=s["min_suppression_pct"], threshold_pct=s["threshold_pct"],
                            cv_multiple=s["threshold_cv_multiple"])
        threshold = next((t.group for t in tests if t.significant), NAN)
        words = rule_words(s["threshold_rule"], key, s["alpha"], s["min_suppression_pct"],
                           f"the first {s['post_responses']} responses after the step against the last "
                           f"{s['baseline_responses']} before it", s["threshold_pct"], s["threshold_cv_multiple"])
        out.choose("threshold", f"{threshold:g} {'s' if key == 'duration' else 'mV'}" if np.isfinite(threshold)
                   else "none reached", words)
        interp, longest = curve_threshold(tests, s["min_suppression_pct"] or s["threshold_pct"]) \
            if key == "duration" else (NAN, NAN)
        out.values.update({"threshold": threshold, "threshold_interpolated_s": interp,
                           "suppression_longest_pct": longest, "no_threshold": float(not np.isfinite(threshold)),
                           "threshold_key": key})
    else:
        out.values.update({"threshold": NAN, "threshold_interpolated_s": NAN, "no_threshold": NAN,
                           "suppression_longest_pct": out.values["suppression_pct"]})
        out.notes.append("every step has the same duration and level: no threshold to find")
    out.tables["episodes"] = [
        {"episode": i + 1, ("step_file" if files else "sweep"): e.sweep, "step_start_s": e.start_s, "step_duration_s": e.duration_s,
         "step_level_mv": e.level_mv, "settle_s": e.settle_s, "baseline": e.baseline, "baseline_cv": e.baseline_cv,
         "suppression_pct": e.suppression_pct, "time_to_recovery_s": e.recovery_time_s, "recovery_tau_s": e.tau_s,
         "fit_S": e.fit_s, "fit_r2": e.fit_r2, "timed": e.measurable, "power_suppression_pct": e.power_suppression_pct,
         "recovered_pct": e.recovered_pct, "baseline_to_step_s": e.baseline_gap_s} for i, e in enumerate(episodes)]
    out.tables["durations"] = [{"group": t.group, "suppression_pct": t.suppression_pct, "p": t.p, "trials": t.trials,
                                "significant": t.significant} for t in tests]
    out.tables["responses"] = [{**where(k, r), "pulse": r[1] + 1, "time_s": r[2], "amplitude": r[3], "power": pw[k],
                                "charge": r[6], "artifact_end_ms": 1e3 * r[7], "peak_ms": 1e3 * r[8]}
                               for k, r in enumerate(rows)]
    # a stimulus artifact going the other way, and peaks it may hide
    art = np.array([r[7] for r in rows], float)
    seen = np.isfinite(art)
    if seen.any():
        hidden = seen & (np.array([r[8] for r in rows], float) - art < 1e-3)
        out.choose("stimulus artifact", f"in {int(seen.sum())} of {len(rows)} responses, going the other way from "
                   f"them, over {1e3 * np.median(art[seen]):.1f} ms after the pulse (median; {1e3 * art[seen].min():.1f}"
                   f" to {1e3 * art[seen].max():.1f} ms)", "where the trace is last more than 3 noise SD on the "
                   "artifact's side before the response's peak; the peak is looked for after it")
        if hidden.any():
            out.flags.append(Flag("artifact over the peak", f"{int(hidden.sum())} of {len(rows)} responses peak "
                                  "within 1 ms of the end of the stimulus artifact: their true peak may be under it, "
                                  "and their amplitude too small", "1 ms", "Clampwise"))
        if np.ptp(art[seen]) > 2e-3:
            out.flags.append(Flag("artifact length varies", f"the stimulus artifact ends between "
                                  f"{1e3 * art[seen].min():.1f} and {1e3 * art[seen].max():.1f} ms after the pulse "
                                  "from response to response", "2 ms", "Clampwise", severity="note"))


# ------------------------------------------------------------------ evoked DSI over a cell's files

@register("dsi_cell", "DSI / DSE from evoked responses, over a cell's files", version=3, types={"vc_evoked", "dsi"},
          settings=SETTINGS, outputs=OUTPUTS, level="cell",
          description="DSI sessions recorded across several files (evoked responses and depolarizing steps in "
                      "separate files), aligned on the cell's clock; each step is then analyzed as for a single "
                      "recording (baseline, suppression, recovery, time course, Fourier transform, threshold).",
          source="Pitler & Alger 1992")
def run_cell(cell, s) -> Result | None:
    from .stimulus import find_stimulus

    steps, evoked = [], []
    for m, offset, sweep_t in cell.placed():
        rec = m.recording
        if rec is None:
            continue
        sweep_t = np.asarray(sweep_t, float)
        found = dsi_steps(rec)
        stim = find_stimulus(rec)
        if found and not len(stim):           # a file holding only the step (a DSI_Step file)
            steps += [(float(sweep_t[sw] if sw < sweep_t.size else offset) + a,
                       float(sweep_t[sw] if sw < sweep_t.size else offset) + b, m.name, lv, SETTLE_MIN_S)
                      for sw, a, b, lv in found]
        elif len(stim) and not found and rec.clamp_mode is ClampMode.VOLTAGE_CLAMP:
            evoked.append((m.name, rec, stim, sweep_t))
    if not steps or not evoked:
        return None
    out = Result("dsi_cell", 1)
    s_ev, _ = load()["evoked"].resolve({})
    s_ev = dict(s_ev, follow_slow=False)
    absolute = sorted(steps)
    def measured(pol: int) -> list[tuple]:
        return sorted(((float(sweep_t[r[0]] if r[0] < sweep_t.size else 0.0) + r[4], name, rec, r)
                       for name, rec, stim, sweep_t in evoked for r in _responses(rec, stim, [], s_ev, pol)),
                      key=lambda x: x[0])

    picked = measured(-1)
    polarity, why = _direction([(*r[:2], t, *r[3:]) for t, _, _, r in picked], absolute[0][0], s["direction"])
    out.choose("response direction", "outward (up)" if polarity > 0 else "inward (down)", why)
    if polarity > 0:
        picked = measured(1)
    rows = [(r[0], r[1], t, *r[3:]) for t, _, _, r in picked]
    items = [(rec, r, []) for _, _, rec, r in picked]
    files = [name for _, name, _, _ in picked]
    fs = evoked[0][1].sample_rate_hz
    if any(abs(rec.sample_rate_hz - fs) > 1e-6 for _, rec, _, _ in evoked):
        out.flags.append(Flag("sample rates differ", "the cell's evoked files were not all recorded at the same "
                              "sample rate: their Fourier transforms are not on one set of frequencies", level="cell"))
    out.choose("files", f"{len(evoked)} file(s) of evoked responses, {len(steps)} step(s) from "
               f"{len({st[2] for st in steps})} step file(s)", "on the cell's clock (each file's start time plus its "
               f"sweeps'); no response is read until {SETTLE_MIN_S:g} s after a step")
    ledger.note("a cell's files on one clock; each evoked response measured on its own sweep (the evoked analysis's "
                "single-response measurement); each step from its own file", "DSI over a cell's files")
    summarize(out, rows, items, absolute, s, s_ev, fs, evoked[0][1].signal_units, files)
    return out
