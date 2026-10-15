"""DSI from spontaneous events, over a cell's files on one clock (B5).

A DSI experiment is a gap-free file of sIPSCs, a file
holding the step to 0 mV, and a file recorded after it; only together are
they a trial. The events analysis has already found each file's events, the
stretches it analyzed, its steps with the current's settling, and its charge
in 10 ms blocks; here they are put on the cell's clock (each sweep at its
file's start time plus its own) and measured around each step.

**Each trial** (Makara et al. 2007, *J
Neurosci* 27:10211 for the windows and the charge):

- before: the 10 s immediately before the step;
- after: the first 5 s from the end of the settling (the holding current back
  within noise of where it sits 1-3 s later, at least 0.2 s after the step);
- suppression by frequency (events per second, **the headline**: the mentor's
  request and Heinbockel et al. 2005) and by charge (against an event-free
  baseline, so independent of the detector), each 100 (1 - after / before);
- both windows are fixed to the step however much of them was recorded: a
  thinly recorded window still gives its value with a note saying how much was
  recorded and when the first data after the step began (a window slid to the
  next data read 24125234, recorded 4 min after its step, as -89%);
- the time course in 1 s bins, normalized to the rate before, and
  1 - S exp(-t/tau) fitted to the bins after the step when at least 20% was
  suppressed.

**By step duration**: the threshold with the evoked DSI's rule (one-sided
Mann-Whitney of the 1 s bins after the step against those before, each over
its trial's rate before, p < 0.05 and at least 20%), and the curve threshold.

**Heinbockel et al. 2005** (*J Neurosci* 25:9449), per duration, the trials
pooled (events summed over trials, divided by the seconds recorded in each
bin): 1 s bins; the onset in 200 ms bins; the minimum as the mean 2-3 s after
the stimulus **ends** (their 100 ms stimulus left 0-3 s recordable; after a
0.5-5 s step nothing is recorded until it has ended and settled); values
normalized between control (100%) and that minimum (0%); the onset as a
delayed exponential, whose start is t_DSI (extrapolated, and said so, when the
first bins were not recorded); the recovery tau; suppression their way (32 s
before, 20 s after) beside the headline. Built on it, onset and recovery fitted
together, which does not need the minimum (when recovery has begun by 2-3 s
their normalization squeezes the onset: tau 0.40 s read 0.14 s with a 6 s
recovery on simulated cells).

One trial cannot separate DSI from the slow swings of carbachol-driven
activity (1-17 events/s over 20-30 s in 24125234): the published method
averages 2-4 trials per duration, 90-130 s apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsi as dsi_mod
from .cells import Cell
from .events import SETTLE_MIN_S, dsi_steps
from .registry import Output, Setting, register
from .result import Flag, Result
from .tools import fits, ledger

NAN = float("nan")
MIN_COVERAGE = 0.5           # a bin less than half recorded gives no rate
ONSET_BIN_S = 0.2
ONSET_SPAN_S = 3.0           # after the stimulus ends
MINIMUM_S = (2.0, 3.0)       # after the stimulus ends
PAPER_PRE_S, PAPER_POST_S = 32.0, 20.0


# ------------------------------------------------------------------ the cell's clock

@dataclass
class Step:
    start_s: float
    end_s: float
    level_mv: float
    settle_s: float
    file: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Timeline:
    events: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))   # time, |amplitude|, decay (ok), file index
    spans: list[tuple[float, float]] = field(default_factory=list)          # analyzed, on the clock
    charge: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)   # block starts, charge, s
    steps: list[Step] = field(default_factory=list)
    units: str = "pA"
    notes: list[str] = field(default_factory=list)

    def analyzed(self, a: float, b: float) -> float:
        return float(sum(max(0.0, min(e, b) - max(s, a)) for s, e in self.spans))

    def window(self, a: float, b: float, min_coverage: float = MIN_COVERAGE) -> dict:
        covered = self.analyzed(a, b)
        sel = (self.events[:, 0] >= a) & (self.events[:, 0] < b) if self.events.size else np.zeros(0, bool)
        n = int(sel.sum())
        enough = covered > 0 and covered >= min_coverage * (b - a)
        q = qcov = 0.0
        for starts, charge, cov in self.charge:
            m = (starts >= a) & (starts < b)
            if m.any():
                q += float(charge[m].sum())
                qcov += float(cov[m].sum())
        amps = self.events[sel, 1] if n else np.array([])
        return {"analyzed_s": covered, "events": n, "rate_hz": n / covered if enough else NAN,
                "charge_per_s": q / qcov if enough and qcov > 0 else NAN,
                "amplitude_median": float(np.median(amps)) if amps.size else NAN}


def timeline(cell: Cell) -> Timeline:
    tl = Timeline()
    rows = []
    for fi, (m, offset, sweep_t) in enumerate(cell.placed()):
        sweep_t = np.asarray(sweep_t, float)

        def at(sweep: int, t: float) -> float:
            return float(sweep_t[sweep] if sweep < sweep_t.size else offset) + t

        ev = m.result("events")
        if ev is not None:
            tl.units = m.recording.signal_units
            for r in ev.tables.get("events", []):
                if r["accepted"]:
                    good = np.isfinite(r["decay_r2"]) and r["decay_r2"] >= ev.settings.get("min_decay_r2", 0.8)
                    rows.append((at(r["sweep"], r["onset_s"]), abs(r["amplitude"]),
                                 r["decay_tau_ms"] if good else NAN, fi))
            for r in ev.tables.get("analyzed", []):
                tl.spans.append((at(r["sweep"], r["start_s"]), at(r["sweep"], r["end_s"])))
            for key, charge in ev.arrays.items():
                if key.startswith("charge_") and not key.startswith("charge_t0_"):
                    k = int(key.split("_")[1])
                    t0, width = ev.arrays[f"charge_t0_{k}"]
                    starts = at(k, float(t0)) + np.arange(charge.size) * float(width)
                    tl.charge.append((starts, np.asarray(charge, float), np.asarray(ev.arrays[f"covered_{k}"], float)))
            for r in ev.tables.get("steps", []):
                tl.steps.append(Step(at(r["sweep"], r["start_s"]), at(r["sweep"], r["end_s"]), r["level_mv"],
                                     r["settle_s"], m.name))
        else:
            # a file holding only the step (a DSI_Step file): its step on the clock
            for sw, a, b, level in dsi_steps(m.recording):
                tl.steps.append(Step(at(sw, a), at(sw, b), level, SETTLE_MIN_S, m.name))
    tl.events = np.array(sorted(rows), float) if rows else np.zeros((0, 4))
    tl.spans.sort()
    tl.steps.sort(key=lambda st: st.start_s)
    # the same step seen in two files (it never is in Clampex, but a copy could be): kept once
    unique: list[Step] = []
    for st in tl.steps:
        if not unique or abs(st.start_s - unique[-1].start_s) > 0.05:
            unique.append(st)
    tl.steps = unique
    return tl


def bins(tl: Timeline, start: float, end: float, width: float) -> list[dict]:
    edges = np.arange(start, end + 1e-9, width)
    return [{"bin_start_s": float(a), **tl.window(a, b)} for a, b in zip(edges[:-1], edges[1:])]


# ------------------------------------------------------------------ trials

@dataclass
class TrialResult:
    step: Step
    pre: dict
    post: dict
    first_data_after_s: float
    suppression_pct: float = NAN
    charge_suppression_pct: float = NAN
    tau_s: float = NAN
    fit_s: float = NAN
    fit_r2: float = NAN
    course: list[dict] = field(default_factory=list)
    post_normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    baseline_cv: float = NAN
    notes: list[str] = field(default_factory=list)


def trials(tl: Timeline, s: dict) -> list[TrialResult]:
    out = []
    w = s["bin_s"]
    for k, st in enumerate(tl.steps):
        prev_end = tl.steps[k - 1].end_s if k else -np.inf
        nxt = tl.steps[k + 1].start_s if k + 1 < len(tl.steps) else np.inf
        pre = tl.window(max(st.start_s - s["pre_s"], prev_end), st.start_s, min_coverage=0.0)
        after = [x for x in tl.spans if x[1] > st.end_s]
        first = min((max(x[0], st.end_s) for x in after), default=NAN)
        gap = first - st.end_s if np.isfinite(first) else NAN
        post_a = st.end_s + st.settle_s
        post_b = min(post_a + s["post_s"], nxt)
        post = tl.window(post_a, post_b, min_coverage=0.0)
        tr = TrialResult(st, pre, post, gap)
        if pre["analyzed_s"] < MIN_COVERAGE * s["pre_s"]:
            tr.notes.append(f"only {pre['analyzed_s']:.1f} s of spontaneous activity recorded in the {s['pre_s']:g} s "
                            "before the step")
        if not np.isfinite(gap):
            tr.notes.append("no spontaneous activity recorded after the step")
        elif gap > st.settle_s + 0.5:
            tr.notes.append(f"the first spontaneous activity after the step was recorded {gap:.1f} s after it: DSI "
                            "peaks at once and is largely over within 10-20 s")
        if post["analyzed_s"] < MIN_COVERAGE * s["post_s"]:
            tr.notes.append(f"only {post['analyzed_s']:.1f} s of spontaneous activity recorded in the {s['post_s']:g} s "
                            "after the step")
        r0, r1 = pre["events"] / pre["analyzed_s"] if pre["analyzed_s"] > 0 else NAN, \
            post["events"] / post["analyzed_s"] if post["analyzed_s"] > 0 else NAN
        if np.isfinite(r0) and np.isfinite(r1) and r0 > 0:
            tr.suppression_pct = 100.0 * (1.0 - r1 / r0)
        q0, q1 = pre["charge_per_s"], post["charge_per_s"]
        if np.isfinite(q0) and np.isfinite(q1) and q0 > 0:
            tr.charge_suppression_pct = 100.0 * (1.0 - q1 / q0)
        before = bins(tl, st.start_s - np.floor(s["pre_s"] / w) * w, st.start_s, w)
        course_end = min(post_a + s["course_s"], nxt)
        after_bins = bins(tl, post_a, course_end, w) if course_end > post_a + w else []
        if np.isfinite(r0) and r0 > 0:
            for row in before + after_bins:
                row["rate_normalized"] = row["rate_hz"] / r0
                row["from_step_end_s"] = row["bin_start_s"] - st.end_s
            tr.course = before + after_bins
            pb = np.array([r["rate_normalized"] for r in before], float)
            tr.baseline_normalized = pb[np.isfinite(pb)]
            rates = np.array([r["rate_hz"] for r in before], float)
            rates = rates[np.isfinite(rates)]
            if rates.size > 1:
                tr.baseline_cv = float(np.std(rates, ddof=1) / r0)
            post_rows = [r for r in after_bins if r["bin_start_s"] < post_b - 1e-9]
            pn = np.array([r["rate_normalized"] for r in post_rows], float)
            tr.post_normalized = pn[np.isfinite(pn)]
            if np.isfinite(tr.suppression_pct) and tr.suppression_pct >= s["min_suppression_pct"] and after_bins:
                ok = [r for r in after_bins if np.isfinite(r["rate_normalized"])]
                fit = fits.suppression_recovery(np.array([r["from_step_end_s"] + w / 2 for r in ok]),
                                                np.array([r["rate_normalized"] for r in ok]),
                                                purpose="DSI recovery (events per second, 1 s bins)")
                if fit.ok:
                    tr.tau_s, tr.fit_s, tr.fit_r2 = fit["tau"], fit["s"], fit.r2
        out.append(tr)
    return out


# ------------------------------------------------------------------ Heinbockel et al. 2005, per duration

@dataclass
class Kinetics:
    duration_s: float
    n_trials: int
    pre_rate_hz: float = NAN
    min_rate_hz: float = NAN
    suppression_pct: float = NAN               # the mean of the trials' headline (10 s / 5 s)
    suppression_at_min_pct: float = NAN
    suppression_32_20_pct: float = NAN
    t_dsi_s: float = NAN
    onset_tau_s: float = NAN
    onset_r2: float = NAN
    recovery_tau_s: float = NAN
    recovery_s: float = NAN
    recovery_r2: float = NAN
    joint_t_dsi_s: float = NAN
    joint_tau_onset_s: float = NAN
    joint_s: float = NAN
    joint_tau_recovery_s: float = NAN
    joint_r2: float = NAN
    first_data_after_end_s: float = NAN
    course: list[dict] = field(default_factory=list)       # 1 s bins, pooled
    onset: list[dict] = field(default_factory=list)        # 200 ms bins, pooled
    notes: list[str] = field(default_factory=list)


def _pooled_bins(tl: Timeline, trs: list[TrialResult], width: float, before_s: float, after_end_s: float) -> list[dict]:
    """Events and recorded seconds summed over the trials in bins on the step's
    clock, from ``before_s`` before its onset to ``after_end_s`` after its end;
    never past the next trial's step."""
    dur = float(np.median([t.step.duration_s for t in trs]))
    edges = np.arange(-before_s, dur + after_end_s + 1e-9, width)
    ev = np.zeros(edges.size - 1)
    cov = np.zeros(edges.size - 1)
    for tr in trs:
        start = tr.step.start_s
        nxt = min((x.start_s for x in tl.steps if x.start_s > start + 1e-6), default=np.inf)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            lo, hi = start + a, min(start + b, nxt)
            if hi > lo:
                m = tl.window(lo, hi, 0.0)
                ev[i] += m["events"]
                cov[i] += m["analyzed_s"]
    rows = []
    for i, a in enumerate(edges[:-1]):
        enough = cov[i] >= MIN_COVERAGE * width * len(trs) and cov[i] > 0
        rows.append({"from_step_start_s": float(a), "from_step_end_s": float(a - dur), "bin_width_s": width,
                     "events": int(ev[i]), "recorded_s": float(cov[i]), "trials": len(trs),
                     "rate_hz": ev[i] / cov[i] if enough else NAN})
    return rows


def _pooled_rate(tl: Timeline, trs: list[TrialResult], a: float, b: float, ref: str) -> tuple[float, float]:
    events, cov = 0, 0.0
    for tr in trs:
        r = tr.step.start_s if ref == "start" else tr.step.end_s
        m = tl.window(r + a, r + b, 0.0)
        events += m["events"]
        cov += m["analyzed_s"]
    return (events / cov if cov > 0 else NAN), cov


def kinetics(tl: Timeline, trs: list[TrialResult], s: dict) -> Kinetics:
    dur = float(np.median([t.step.duration_s for t in trs]))
    k = Kinetics(round(dur, 3), len(trs))
    sup = [t.suppression_pct for t in trs if np.isfinite(t.suppression_pct)]
    k.suppression_pct = float(np.mean(sup)) if sup else NAN
    k.pre_rate_hz, _ = _pooled_rate(tl, trs, -s["pre_s"], 0.0, "start")
    k.min_rate_hz, min_cov = _pooled_rate(tl, trs, MINIMUM_S[0], MINIMUM_S[1], "end")
    if min_cov < MIN_COVERAGE * (MINIMUM_S[1] - MINIMUM_S[0]) * len(trs):
        k.min_rate_hz = NAN
        k.notes.append(f"the {MINIMUM_S[0]:g}-{MINIMUM_S[1]:g} s after the step (the minimum) were not recorded")
    if np.isfinite(k.pre_rate_hz) and k.pre_rate_hz > 0 and np.isfinite(k.min_rate_hz):
        k.suppression_at_min_pct = 100.0 * (1.0 - k.min_rate_hz / k.pre_rate_hz)
    settle = float(np.median([t.step.settle_s for t in trs]))
    paper_pre, _ = _pooled_rate(tl, trs, -PAPER_PRE_S, 0.0, "start")
    paper_post, _ = _pooled_rate(tl, trs, settle, settle + PAPER_POST_S, "end")
    if np.isfinite(paper_pre) and paper_pre > 0 and np.isfinite(paper_post):
        k.suppression_32_20_pct = 100.0 * (1.0 - paper_post / paper_pre)
    gaps = [t.first_data_after_s for t in trs if np.isfinite(t.first_data_after_s)]
    k.first_data_after_end_s = float(np.median(gaps)) if gaps else NAN
    k.course = _pooled_bins(tl, trs, s["bin_s"], s["pre_s"], s["course_s"])
    k.onset = _pooled_bins(tl, trs, ONSET_BIN_S, 0.0, ONSET_SPAN_S)
    pre, mn = k.pre_rate_hz, k.min_rate_hz
    for rows in (k.course, k.onset):
        for r in rows:
            r["pct_of_control"] = 100.0 * r["rate_hz"] / pre if np.isfinite(pre) and pre > 0 else NAN
            span = pre - mn if np.isfinite(pre) and np.isfinite(mn) else NAN
            r["pct_between_control_and_min"] = (100.0 * (r["rate_hz"] - mn) / span
                                                if np.isfinite(span) and span > 0 else NAN)
    if not (np.isfinite(pre) and pre > 0):
        k.notes.append("no rate before the step")
        return k
    onset = [r for r in k.onset if r["from_step_start_s"] >= 0 and np.isfinite(r["pct_between_control_and_min"])]
    if np.isfinite(mn) and pre - mn > 0 and len(onset) >= 4:
        t = np.array([r["from_step_start_s"] + ONSET_BIN_S / 2 for r in onset])
        fit = fits.delayed_exponential(t, np.array([r["pct_between_control_and_min"] for r in onset]),
                                       purpose="DSI onset (Heinbockel et al. 2005)")
        if fit.ok:
            k.t_dsi_s, k.onset_tau_s, k.onset_r2 = fit["t0"], fit["tau"], fit.r2
            first = float(min(r["from_step_start_s"] for r in onset))
            if first > k.t_dsi_s:
                k.notes.append(f"t_DSI extrapolated: nothing was recorded until {first:.1f} s after the step's onset "
                               f"(the step lasts {dur:g} s); Heinbockel et al. timed the onset after a 100 ms stimulus")
            if not (np.isfinite(k.onset_r2) and k.onset_r2 >= 0.5):
                k.notes.append(f"the onset fit is poor (r² {k.onset_r2:.2f}): its t_DSI and tau are not measurements")
        else:
            k.notes.append("the onset could not be fitted (less than 30% suppression in the 200 ms bins)")
    else:
        k.notes.append("no suppression to time the onset of" if len(onset) >= 4 else "too few 200 ms bins recorded")
    after = [r for r in k.course if r["from_step_end_s"] >= settle and np.isfinite(r["rate_hz"])]
    big = np.isfinite(k.suppression_pct) and k.suppression_pct >= s["min_suppression_pct"]
    if big and len(after) >= 3:
        fit = fits.suppression_recovery(np.array([r["from_step_end_s"] + s["bin_s"] / 2 for r in after]),
                                        np.array([r["rate_hz"] / pre for r in after]),
                                        purpose="DSI recovery (pooled 1 s bins)")
        if fit.ok:
            k.recovery_tau_s, k.recovery_s, k.recovery_r2 = fit["tau"], fit["s"], fit.r2
    if big:
        t = [r["from_step_start_s"] + ONSET_BIN_S / 2 for r in k.onset
             if r["from_step_start_s"] >= 0 and np.isfinite(r["pct_of_control"])]
        y = [r["pct_of_control"] for r in k.onset if r["from_step_start_s"] >= 0 and np.isfinite(r["pct_of_control"])]
        late = [r for r in k.course if r["from_step_end_s"] >= ONSET_SPAN_S and np.isfinite(r["pct_of_control"])]
        t += [r["from_step_start_s"] + s["bin_s"] / 2 for r in late]
        y += [r["pct_of_control"] for r in late]
        fit = fits.onset_recovery(np.array(t), np.array(y), dur, purpose="DSI onset and recovery together")
        if fit.ok:
            k.joint_t_dsi_s, k.joint_tau_onset_s = fit["t0"], fit["tau_on"]
            k.joint_s, k.joint_tau_recovery_s, k.joint_r2 = fit["s"], fit["tau_rec"], fit.r2
            if t and min(t) - ONSET_BIN_S / 2 > k.joint_t_dsi_s:
                k.notes.append(f"joint fit: t_DSI and the onset tau are extrapolated (nothing recorded until "
                               f"{min(t) - ONSET_BIN_S / 2:.1f} s after the step's onset)")
    return k


# ------------------------------------------------------------------ the declaration

SETTINGS = (
    Setting("pre_s", 10.0, "Before the step (s)", "Event rate over this many seconds immediately before each "
            "step.", "s", "Makara et al. 2007", minimum=1.0),
    Setting("post_s", 5.0, "After the step (s)", "Event rate over this many seconds after the holding current "
            "has settled.", "s", "Makara et al. 2007", minimum=0.5),
    *dsi_mod.THRESHOLD_SETTINGS[:2],
    Setting("bin_s", 1.0, "Time course: bins (s)", "", "s", "Heinbockel et al. 2005", advanced=True, minimum=0.1),
    Setting("course_s", 60.0, "Time course: after the step (s)", "", "s", advanced=True, minimum=5.0),
    *dsi_mod.THRESHOLD_SETTINGS[2:],
)

OUTPUTS = (
    Output("suppression_pct", "Mean over the trials of 100 (1 - events per second in the 5 s after the step's "
           "settling / in the 10 s before it).", "%", "Heinbockel et al. 2005",
           label="Suppression (events per second)", headline=True, level="cell"),
    Output("charge_suppression_pct", "The same with the charge per second against an event-free baseline: "
           "independent of the detector.", "%", "Makara et al. 2007", label="Suppression (charge)", headline=True,
           level="cell"),
    Output("threshold_s", "The shortest step duration whose suppression is significant (one-sided Mann-Whitney of "
           "the 1 s bins after against before, p < 0.05) and at least 20%.", "s",
           "", label="Threshold duration", headline=True, level="cell"),
    Output("threshold_interpolated_s", "Where the durations' mean suppression, made to rise with duration, crosses "
           "the minimum suppression, on log duration.", "s", label="Threshold from the curve", level="cell"),
    Output("no_threshold", "1 when several durations were tried and none suppressed significantly.", "",
           label="No threshold reached", level="cell"),
    Output("recovery_tau_s", "Mean recovery time constant of the trials suppressed by at least the minimum "
           "(1 - S exp(-t/tau) on the 1 s bins).", "s", label="Recovery tau", headline=True, level="cell"),
    Output("n_trials", "Steps with spontaneous activity recorded around them.", "count", label="Trials",
           headline=True, level="cell"),
    Output("units", "The units of the events' amplitudes: the recorded signal's.", "", label="Amplitude units",
           level="cell"),
    Output("n_durations", "Different step durations.", "count", label="Step durations", level="cell"),
)


@register("dsi_events", "DSI from spontaneous events", version=1, types={"vc_spontaneous", "dsi", "vc_evoked"},
          settings=SETTINGS, outputs=OUTPUTS, level="cell",
          description="For each step to 0 mV, the cell's spontaneous event rate (and charge) in the 10 s before "
                      "versus the 5 s after, across all of its files on one clock; grouped by step duration with the "
                      "threshold; and the onset, minimum and recovery of the suppression.",
          source="Makara et al. 2007; Heinbockel et al. 2005")
def run(cell: Cell, s) -> Result | None:
    if not any(m.result("events") is not None for m in cell.members):
        return None
    tl = timeline(cell)
    if not tl.steps:
        return None
    out = Result("dsi_events", 1)
    ledger.note(f"a cell's files on one clock (each sweep at its file's start time plus its own); per step, events "
                f"per second and charge in the {s['pre_s']:g} s before against the {s['post_s']:g} s after the "
                "settling", "DSI from spontaneous events")
    trs = trials(tl, s)
    durations: dict[float, list[TrialResult]] = {}
    for tr in trs:
        durations.setdefault(round(tr.step.duration_s, 2), []).append(tr)
    tests = dsi_mod.group_tests(
        [dsi_mod.Trial(round(t.step.duration_s, 2), t.suppression_pct, t.post_normalized, t.baseline_normalized,
                       t.baseline_cv) for t in trs],
        rule=s["threshold_rule"], alpha=s["alpha"], min_suppression_pct=s["min_suppression_pct"],
        threshold_pct=s["threshold_pct"], cv_multiple=s["threshold_cv_multiple"])
    by = {t.group: t for t in tests}
    threshold = next((t.group for t in tests if t.significant), NAN) if len(durations) > 1 else NAN
    interp, _ = dsi_mod.curve_threshold(tests, s["min_suppression_pct"] or s["threshold_pct"]) \
        if len(durations) > 1 else (NAN, NAN)
    words = dsi_mod.rule_words(s["threshold_rule"], "duration", s["alpha"], s["min_suppression_pct"],
                               f"the {s['bin_s']:g} s bins of events per second in the {s['post_s']:g} s after the "
                               f"step against the {s['pre_s']:g} s before it", s["threshold_pct"],
                               s["threshold_cv_multiple"])
    if len(durations) > 1:
        out.choose("threshold", f"{threshold:g} s" if np.isfinite(threshold) else "none reached", words)
    sup = [t.suppression_pct for t in trs if np.isfinite(t.suppression_pct)]
    qsup = [t.charge_suppression_pct for t in trs if np.isfinite(t.charge_suppression_pct)]
    taus = [t.tau_s for t in trs if np.isfinite(t.tau_s)]
    out.values.update({
        "suppression_pct": float(np.mean(sup)) if sup else NAN,
        "charge_suppression_pct": float(np.mean(qsup)) if qsup else NAN,
        "threshold_s": threshold, "threshold_interpolated_s": interp,
        "no_threshold": float(not np.isfinite(threshold)) if len(durations) > 1 else NAN,
        "recovery_tau_s": float(np.mean(taus)) if taus else NAN,
        "n_trials": len(trs), "n_durations": len(durations), "units": tl.units,
    })
    out.choose("headline", "events per second, charge beside it", "Frequency is the standard DSI measure "
               "(Heinbockel et al.); charge does not depend on the detector")
    kin = [kinetics(tl, g, s) for _, g in sorted(durations.items())]
    out.tables["trials"] = [
        {"trial": i + 1, "step_file": t.step.file, "step_start_s": t.step.start_s,
         "step_duration_s": round(t.step.duration_s, 3), "step_level_mv": t.step.level_mv, "settle_s": t.step.settle_s,
         "first_data_after_step_s": t.first_data_after_s, "pre_analyzed_s": t.pre["analyzed_s"],
         "pre_events": t.pre["events"], "post_analyzed_s": t.post["analyzed_s"], "post_events": t.post["events"],
         "suppression_pct": t.suppression_pct, "charge_suppression_pct": t.charge_suppression_pct,
         "pre_amplitude_median": t.pre["amplitude_median"], "post_amplitude_median": t.post["amplitude_median"],
         "recovery_tau_s": t.tau_s, "recovery_fit_r2": t.fit_r2, "notes": "; ".join(t.notes)}
        for i, t in enumerate(trs)]
    out.tables["durations"] = [
        {"step_duration_s": k.duration_s, "trials": k.n_trials,
         "suppression_pct": k.suppression_pct, "p": by[round(k.duration_s, 2)].p if round(k.duration_s, 2) in by else NAN,
         "significant": by[round(k.duration_s, 2)].significant if round(k.duration_s, 2) in by else False,
         "pre_rate_hz": k.pre_rate_hz, "min_rate_hz": k.min_rate_hz, "suppression_at_min_pct": k.suppression_at_min_pct,
         "suppression_32_20_pct": k.suppression_32_20_pct, "t_dsi_s": k.t_dsi_s, "onset_tau_s": k.onset_tau_s,
         "onset_fit_r2": k.onset_r2, "recovery_tau_s": k.recovery_tau_s, "recovery_S": k.recovery_s,
         "recovery_fit_r2": k.recovery_r2, "joint_t_dsi_s": k.joint_t_dsi_s,
         "joint_onset_tau_s": k.joint_tau_onset_s, "joint_S": k.joint_s,
         "joint_recovery_tau_s": k.joint_tau_recovery_s, "joint_fit_r2": k.joint_r2,
         "first_data_after_step_s": k.first_data_after_end_s, "notes": "; ".join(k.notes)} for k in kin]
    out.tables["timecourse"] = [
        {"step_duration_s": k.duration_s, "bins": label, **r} for k in kin
        for label, rows in (("1 s", k.course), ("200 ms onset", k.onset)) for r in rows]
    thin = [i + 1 for i, t in enumerate(trs) if t.notes]
    if thin:
        out.flags.append(Flag("thin window", f"Trial(s) {', '.join(map(str, thin))}: a window around the step was not "
                              "fully recorded (see the trials table's notes). Their values are kept, with what was "
                              "recorded said.", "half of each window recorded", "",
                              level="cell", severity="note"))
    if len(trs) < 2 * max(len(durations), 1):
        out.notes.append("Fewer than two trials per step duration: one trial cannot separate DSI from the slow swings "
                         "of spontaneous activity (the published method averages 2-4 trials per duration, 90-130 s "
                         "apart).")
    return out


# ------------------------------------------------------------------ cells combined (Heinbockel's Figs. 3D, 6)

def combine(per_cell: list[Result]) -> list[dict]:
    """Cells combined by step duration: each 1 s bin's mean and SEM of % of
    control across cells, and the recovery fitted to the combined bins."""
    by: dict[float, list[list[dict]]] = {}
    for r in per_cell:
        if r is None or r.error:
            continue
        rows = [x for x in r.tables.get("timecourse", []) if x["bins"] == "1 s"]
        for d in sorted({x["step_duration_s"] for x in rows}):
            by.setdefault(d, []).append([x for x in rows if x["step_duration_s"] == d])
    out = []
    for d, cells in sorted(by.items()):
        acc: dict[float, list[float]] = {}
        for rows in cells:
            for x in rows:
                if np.isfinite(x.get("pct_of_control", NAN)):
                    acc.setdefault(round(x["from_step_start_s"], 6), []).append(float(x["pct_of_control"]))
        for t in sorted(acc):
            v = np.array(acc[t])
            out.append({"step_duration_s": d, "from_step_start_s": t, "from_step_end_s": t - d,
                        "pct_of_control_mean": float(v.mean()),
                        "pct_of_control_sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else NAN,
                        "n_cells": int(v.size)})
    return out
