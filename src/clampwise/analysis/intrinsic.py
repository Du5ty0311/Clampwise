"""Intrinsic properties and firing from a current-step family (B6).

The standard is the Allen Institute's IPFX (Gouwens et al. 2019, *Nat
Neurosci* 22:1182), with the lab's recorded departures (Analysis Decisions,
2026-09-22; Measurement Methods):

- **Windows** (IPFX): baseline the 100 ms before the step (shorter when the
  command moves sooner, a test pulse), steady state the last 100 ms of the step
  (at most a quarter of it), the peak searched from the step's onset to there,
  averaged over 5 ms.
- **Input resistance** (NeuroExpress's): each step's
  steady-state resistance, a line through the steps from -70 to -10 pA read at
  0 pA (``iv_curves``). Beside it, the recorded IPFX departure: the slope of
  the steady-state voltage against the current over the subthreshold steps
  from -100 to 0 pA, every subthreshold hyperpolarizing step when fewer than
  two small ones exist (said); it stands in when too few steps lie near 0 pA.
- **Membrane tau**: a single exponential over the charging phase of each small
  hyperpolarizing step from 10% to **95%** of the way to the peak (IPFX: to the
  peak; on real cells with little sag the peak is the lowest noise dip of a long
  plateau and tau ranged 10-113 ms between steps of one cell), steps at least
  20 baseline-noise SDs deep (noise: SD after removing a line), the **median**
  across steps.
- **Sag** (IPFX): (V_peak - V_ss) / (V_peak - V_baseline) on the step whose peak
  is closest to -100 mV; the step and windows said.
- **Capacitance**: tau / the resistance over the same small steps; NeuroExpress's
  tau and capacitance at rest beside them.
- **Rheobase**: the smallest depolarizing step with a full-sized action
  potential (attenuated ones count towards rates, never towards rheobase).
- **Spikes** (``tools/spikes``, IPFX's detection): threshold both ways, peak,
  amplitude, half-width, upstroke, downstroke and their ratio, AHP (trough in
  50 ms), fast AHP (1-5 ms after repolarizing through threshold) and medium AHP
  (10-100 ms after the peak); the headline is the first full spike at rheobase.
- **Trains** (IPFX's definitions; Dayan & Abbott 2001 for ISI statistics): count,
  rate, first-spike latency, ISIs, ISI CV, adaptation index mean((ISI[n+1] -
  ISI[n]) / (ISI[n+1] + ISI[n])), last / first ISI; bursts (3 or more spikes at
  10 ms or less) and an initial doublet; rebound spikes after hyperpolarizing
  steps.
- **f-I curve**: rate against current; its gain is the slope from rheobase over
  the next 5 steps (Hz/pA).
- **Firing, NeuroExpress's way**: no pattern is named. The
  interval measures (adaptation index, ISI ratio, ISI CV) are read at IPFX's
  "hero" step (rheobase + 40 to 60 pA; the step with the most spikes when none
  is), every step's own in the steps table, and the evaluation's statements
  (stuttering among them) in ``iv_curves``.
- **Step by step, at rest, and the evaluation** (NeuroExpress's measures):
  ``iv_curves``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.recording import Recording
from .registry import Output, Setting, register
from .result import Drawable, Flag, Result
from . import iv_curves
from .tools import fits, ledger, spikes

NAN = float("nan")
PEAK_AVERAGE_S = 0.005
STEP_FLAT_FRACTION = 0.2    # a step's command varies by less than this share of its size (a recorded one is noisy)


# ------------------------------------------------------------------ the step and its windows

@dataclass
class Step:
    start_s: float
    end_s: float
    baseline: tuple[float, float]
    steady: tuple[float, float]
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def find_step(recording: Recording, baseline_s: float, steady_s: float, why: list[str] | None = None) -> Step | None:
    """The current step, from the command of the sweep that moves the most; its
    windows. None when there is no single step (the reason added to ``why``)."""
    why = [] if why is None else why
    cmd = recording.command
    if cmd is None or not cmd.size:
        why.append("no command recorded")
        return None
    fs = recording.sample_rate_hz
    hold = np.median(cmd[:, : max(1, cmd.shape[1] // 64)], axis=1, keepdims=True)
    dev = np.abs(cmd - hold)
    k = int(np.argmax(dev.max(axis=1)))
    if dev[k].max() <= 0:
        return None
    on = np.flatnonzero(dev[k] > 0.5 * dev[k].max())
    a, b = on[0] / fs, (on[-1] + 1) / fs
    if b - a < 0.02:
        why.append("the command's largest deflection lasts under 20 ms")
        return None
    # one level from start to end: a staircase or a ramp within the sweep is not a step family
    inside = cmd[k, on[0]:on[-1] + 1]
    spread = float(np.percentile(inside, 95) - np.percentile(inside, 5))
    if spread > STEP_FLAT_FRACTION * dev[k].max():
        why.append(f"the command changes level within the sweep (a staircase or ramp: {spread:.0f} pA across "
                   "its largest stretch), not one step per sweep")
        return None
    notes = []
    # nothing else in the baseline: stop at an earlier command change (a test pulse)
    moved = np.flatnonzero(np.abs(cmd[k, : on[0]] - hold[k, 0]) > max(0.5, 0.05 * dev[k].max()))
    previous = min((moved[-1] + 1) / fs + 0.002, a) if moved.size else 0.0
    base0 = max(previous, a - baseline_s)
    if a - base0 < baseline_s - 1e-9:
        notes.append(f"baseline {1e3 * (a - base0):.0f} ms (the command moves {1e3 * (a - previous):.0f} ms before "
                     "the step)" if moved.size else f"baseline {1e3 * (a - base0):.0f} ms (the sweep starts sooner)")
    ss = min(steady_s, (b - a) / 4)
    if ss < steady_s - 1e-9:
        notes.append(f"steady state the last {ss * 1e3:.0f} ms (a quarter of the step)")
    return Step(a, b, (base0, a), (b - ss, b), notes)


def _noise_sd(v: np.ndarray) -> float:
    """SD after removing a line: slow background counts, which is what spoils a small step's fit."""
    if v.size < 4:
        return 0.0
    x = np.arange(v.size, dtype=float)
    return float(np.std(v - np.polyval(np.polyfit(x, v, 1), x)))


def tau_fit(t: np.ndarray, v: np.ndarray, i0: int, i_peak: int, v_base: float, start_frac: float, end_frac: float,
            min_snr: float, noise_sd: float, fs: float):
    """(tau ms, the fit, the window) over the charging phase from ``start_frac`` to
    ``end_frac`` of the way to the peak, both crossings on a 1 ms running mean."""
    if i_peak - i0 < 10:
        return NAN, None, None
    defl = float(v[i_peak]) - v_base
    if defl == 0 or (min_snr and noise_sd > 0 and abs(defl) / noise_sd < min_snr):
        return NAN, None, None
    n = max(1, int(round(0.001 * fs)))
    seg = v[i0:i_peak + 1]
    # a running mean with the ends held at their own level (zero padding pulled them towards 0)
    sm = (np.convolve(np.pad(seg, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")
          if seg.size > n else seg)

    def cross(frac):
        level = v_base + frac * defl
        hits = np.flatnonzero(sm <= level) if defl < 0 else np.flatnonzero(sm >= level)
        return i0 + int(hits[0]) if hits.size else None

    a, b = cross(start_frac), cross(end_frac)
    if a is None:
        return NAN, None, None
    b = i_peak if b is None or b <= a else b
    if b - a < 10:
        return NAN, None, None
    fit = fits.exp_decay(t[a:b + 1], v[a:b + 1], purpose="membrane tau, 10-95% of the charging phase")
    if not fit.ok:
        return NAN, None, (a, b)
    return fit["tau"] * 1e3, fit, (a, b)


# ------------------------------------------------------------------ trains

@dataclass
class Train:
    count: int = 0
    attenuated: int = 0
    rate_hz: float = NAN
    latency_ms: float = NAN
    isi_mean_ms: float = NAN
    isi_first_ms: float = NAN
    isi_last_ms: float = NAN
    isi_cv: float = NAN
    adaptation_index: float = NAN
    isi_ratio: float = NAN                 # last / first
    bursts: int = 0
    burst_fraction: float = NAN
    doublet: bool = False
    max_gap_ratio: float = NAN             # the longest ISI over the median ISI


def train(sp: list[spikes.Spike], fs: float, start_s: float, end_s: float, s: dict) -> Train:
    inside = [x for x in sp if start_s <= x.peak_index / fs < end_s]
    out = Train(len(inside), sum(1 for x in inside if not x.full))
    if end_s > start_s:
        out.rate_hz = len(inside) / (end_s - start_s)
    if not inside:
        return out
    out.latency_ms = (inside[0].threshold_index / fs - start_s) * 1e3
    if len(inside) >= 2:
        isi = np.diff([x.peak_index for x in inside]) / fs * 1e3
        out.isi_mean_ms, out.isi_first_ms, out.isi_last_ms = float(isi.mean()), float(isi[0]), float(isi[-1])
        out.isi_ratio = float(isi[-1] / isi[0]) if isi[0] > 0 else NAN
        if isi.size >= 2:
            out.isi_cv = float(np.std(isi, ddof=1) / isi.mean())
            out.adaptation_index = float(np.mean((isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])))
            out.max_gap_ratio = float(isi.max() / np.median(isi))
        close = list(isi <= s["burst_isi_ms"]) + [False]
        run = inside_bursts = 0
        for c in close:
            if c:
                run += 1
            else:
                if run + 1 >= s["burst_min_spikes"]:
                    out.bursts += 1
                    inside_bursts += run + 1
                run = 0
        out.burst_fraction = inside_bursts / len(inside)
        later = isi[1:]
        out.doublet = bool(isi[0] <= s["doublet_isi_ms"] and later.size and isi[0] < 0.5 * later.mean())
    return out


def fast_medium_ahp(v: np.ndarray, fs: float, x: spikes.Spike, next_threshold: int | None) -> tuple[float, float]:
    """(fast AHP 1-5 ms after repolarizing through threshold, medium AHP 10-100 ms
    after the peak), each the minimum relative to the spike's threshold."""
    stop = v.size if next_threshold is None else next_threshold
    after = np.flatnonzero(v[x.peak_index:stop] < x.threshold_mv)
    fast = medium = NAN
    if after.size:
        r = x.peak_index + int(after[0])
        a, b = r + int(0.001 * fs), min(stop, r + int(0.005 * fs))
        if b > a:
            fast = float(v[a:b].min() - x.threshold_mv)
    a, b = x.peak_index + int(0.010 * fs), min(stop, x.peak_index + int(0.100 * fs))
    if b > a:
        medium = float(v[a:b].min() - x.threshold_mv)
    return fast, medium


# ------------------------------------------------------------------ the declaration

SETTINGS = (
    Setting("moved", "", "Points moved by hand", "Set on Check: a step's baseline, peak, steady state or rebound "
            "measured at the position placed by hand.", hidden=True),
    Setting("small_step_min_pa", -100.0, "Small steps from (pA)", "Input resistance and tau are measured on the "
            "subthreshold steps from this current to 0 pA, where the membrane response is approximately linear.", "pA",
            "Gouwens et al. 2019"),
    Setting("sag_target_mv", -100.0, "Sag measured at peak voltage (mV)", "Sag is measured on the step whose peak is "
            "closest to this voltage, since Ih activation depends on membrane potential rather than injected current.",
            "mV", "Gouwens et al. 2019"),
    Setting("tau_end_fraction", 0.95, "Tau fit end (fraction of the way to the peak)", "Ending the fit short of the "
            "peak keeps Ih-mediated sag out of the membrane time constant. 1.0 fits up to the peak.", "",
            "", minimum=0.5, maximum=1.0),
    Setting("stutter_gap_ratio", 3.0, "Evaluation: stuttering when an interval is at least this times the median",
            "", "", "", advanced=True),
    Setting("tau_start_fraction", 0.1, "Tau fit starts at (fraction of the way to the peak)", "", "", "Gouwens et al. 2019",
            advanced=True, minimum=0.0, maximum=0.5),
    Setting("tau_min_snr", 20.0, "Tau: steps deeper than (baseline noise SD)", "", "", "",
            advanced=True, minimum=1.0),
    Setting("baseline_s", 0.1, "Baseline before the step (s)", "", "s", "Gouwens et al. 2019", advanced=True, minimum=0.005),
    Setting("steady_s", 0.1, "Steady state: the last (s) of the step", "", "s", "Gouwens et al. 2019", advanced=True, minimum=0.005),
    Setting("gain_steps", 5, "f-I gain: steps from rheobase", "", "", "Clampwise", advanced=True, minimum=2),
    Setting("burst_isi_ms", 10.0, "Bursts: intervals of at most (ms)", "", "ms", advanced=True),
    Setting("burst_min_spikes", 3, "Bursts: at least this many spikes", "", "", advanced=True, minimum=2),
    Setting("doublet_isi_ms", 20.0, "Initial doublet: first interval at most (ms)", "", "ms", advanced=True),
    Setting("rebound_window_ms", 200.0, "Rebound: spikes this long after a hyperpolarizing step (ms)", "", "ms",
            advanced=True),
    # the per-step measures and the values extrapolated to 0 pA
    Setting("reject_range_pa", 10.0, "Current reject range (pA)", "Steps closer than this to 0 pA are excluded from "
            "resistance measures, where a small deflection divided by a small current is unreliable.", "pA", "",
            advanced=True, minimum=0.0),
    Setting("resistance_midpoint_pa", -70.0, "Resistance midpoint (pA)", "The resistance at rest is a linear fit over "
            "the steps from this current to minus the reject range, evaluated at 0 pA.", "pA", "", advanced=True),
    Setting("nex_baseline_s", 0.1, "Per-step measures: baseline before the step (s)", "", "s", "", advanced=True,
            minimum=0.005),
    Setting("nex_steady_s", 0.05, "Per-step measures: steady state, final portion of the step (s)", "", "s", "",
            advanced=True, minimum=0.005),
    Setting("io_spike_limit", 8, "I-O curve: linear up to this spike count", "", "", "", advanced=True,
            minimum=2),
    Setting("cumulative_limit_pa", 100.0, "Cumulative spike count up to (pA)", "", "pA", "", advanced=True),
    Setting("min_negative_steps", 3, "Evaluation: hyperpolarizing steps needed for the resistance at rest", "", "",
            "", advanced=True, minimum=2),
    Setting("rectification_fraction", 0.2, "Evaluation: rectification when the resistance at the extreme step is lower "
            "than at rest by", "", "", "", advanced=True),
    Setting("resistance_disagree_fraction", 0.3, "Evaluation: the peak resistance at rest exceeds the peak V-I "
            "resistance by more than", "", "", "", advanced=True),
    Setting("strong_sag_pct", 15.0, "Evaluation: a strong sag from (% of the peak deflection, the most "
            "hyperpolarizing step)", "", "%", "", advanced=True),
    Setting("strong_rebound_mv", 10.0, "Evaluation: a strong rebound from (mV above the baseline, spikes excluded)",
            "", "mV", "", advanced=True),
    Setting("rectification_min_steps", 3, "Evaluation: rectification assessed only with this many steps beyond the "
            "midpoint", "Families with few steps beyond the midpoint do not constrain the curvature of the V-I "
            "relation.", "", "", advanced=True, minimum=1),
    Setting("depolarized_rest_mv", -40.0, "Evaluation: a cell resting above this is considered unhealthy", "", "mV",
            "", advanced=True),
    Setting("single_spike_fraction", 0.5, "Evaluation: an initial single spike when this share of firing steps fire "
            "once", "", "", "", advanced=True),
    Setting("block_fraction", 0.5, "Evaluation: depolarization block when the largest step fires this share of the "
            "best or less", "", "", "", advanced=True),
    Setting("broad_spike_ms", 2.0, "Evaluation: broad spikes over (ms at half amplitude)", "", "ms",
            "", advanced=True),
)

OUTPUTS = (
    Output("v_rest", "The median over the sweeps of the mean voltage in the baseline before the step.", "mV", "Gouwens et al. 2019",
           label="Resting potential", headline=True),
    Output("r_input", "Input resistance at rest: the steady-state resistance of each step, (V - Vbaseline) / I, a "
           "linear fit over the steps from the midpoint (-70 pA) to -10 pA, evaluated at 0 pA; the slope over the small "
           "steps when too few steps are available.", "MOhm", "", label="Input resistance",
           headline=True),
    Output("r_input_slope", "The slope of the steady-state voltage against the current over the small subthreshold "
           "steps (-100 to 0 pA).", "MOhm", "Gouwens et al. 2019", label="Input resistance, slope over "
           "-100 to 0 pA"),
    Output("r_peak0", "The peak resistance of each step, a line read at 0 pA (as the input resistance).", "MOhm",
           "", label="Resistance at rest, peak"),
    Output("r_steady0", "The steady-state resistance of each step, a line read at 0 pA.", "MOhm", "",
           label="Resistance at rest, steady state"),
    Output("r_peak_slope0", "How the peak resistance changes with current near 0 pA.", "MOhm/nA", "",
           label="Peak resistance slope"),
    Output("r_steady_slope0", "How the steady-state resistance changes with current near 0 pA (positive: lower when "
           "hyperpolarized, inward rectification).", "MOhm/nA", "", label="Steady-state resistance slope"),
    Output("r_vi_peak0", "The slope at 0 pA of a quadratic through the peak voltage against current.", "MOhm",
           "", label="Resistance from the V-I fit, peak"),
    Output("r_vi_steady0", "The slope at 0 pA of a quadratic through the steady-state voltage against current.", "MOhm",
           "", label="Resistance from the V-I fit, steady state"),
    Output("v0_steady", "The steady-state V-I quadratic at 0 pA.", "mV", "", label="Potential at 0 pA"),
    Output("v0_peak", "The peak V-I quadratic at 0 pA.", "mV", "", label="Potential at 0 pA, peak"),
    Output("n_resistance_steps", "Steps in the line for the resistance at rest.", "count", label="Steps for R at rest"),
    Output("tau0", "Tau at the step's onset, a line through the steps near 0 pA read at 0 pA.", "ms", "",
           label="Tau at rest (onset)"),
    Output("tau_off0", "Tau as the voltage returns after the step, read at 0 pA.", "ms", "",
           label="Tau at rest (offset)"),
    Output("capacitance0", "Tau at rest / the resistance at rest.", "pF", "", label="Capacitance at rest"),
    Output("sag_pct0", "Relative sag of each step (% of its peak deflection), read at 0 pA.", "%", "",
           label="Relative sag at 0 pA"),
    Output("sag_slope", "How the sag (mV) grows with hyperpolarizing current.", "mV/nA", "",
           label="Sag slope"),
    Output("rebound_pct0", "Rebound after each step (% of its steady deflection), read at 0 pA.", "%", "",
           label="Relative rebound at 0 pA"),
    Output("rebound_slope", "How the rebound after the step (mV) grows with hyperpolarizing current.", "mV/nA",
           "", label="Rebound slope"),
    Output("io_gain", "Spikes per nA from rheobase until the count reaches the I-O limit (8).", "1/nA", "",
           label="I-O gain"),
    Output("io_rheobase", "Where the I-O line reaches zero spikes.", "pA", "", label="Rheobase from the I-O "
           "line"),
    Output("cumulative_spikes", "Spikes summed over the steps up to the counting limit (100 pA).", "count",
           "", label="Cumulative spikes"),
    Output("latency_1p3", "First-spike latency at 1.3 times rheobase (interpolated).", "ms", "",
           label="Latency at 1.3 x rheobase"),
    Output("mean_latency_to_limit", "Mean first-spike latency over the firing steps up to the counting limit.", "ms",
           "", label="Mean latency to the limit"),
    Output("min_vm_phase", "Where the lowest voltage between two spikes falls in their interval (0 to 1), averaged.",
           "ratio", "", label="Phase of the minimum between spikes"),
    Output("first_threshold_5_mv", "First spikes of the firing steps, averaged: threshold where dV/dt crosses 5 mV/ms.",
           "mV", "", label="1st spikes: threshold at 5 mV/ms"),
    Output("first_amplitude_5_mv", "First spikes: peak minus the 5 mV/ms threshold.", "mV", "",
           label="1st spikes: amplitude"),
    Output("first_half_width_5_ms", "First spikes: width at half that amplitude.", "ms", "",
           label="1st spikes: half-width"),
    Output("first_time_to_peak_ms", "First spikes: threshold to peak.", "ms", "", label="1st spikes: time "
           "to peak"),
    Output("first_fall_time_ms", "First spikes: peak back down to the threshold's voltage.", "ms", "",
           label="1st spikes: fall time"),
    Output("first_ahp_5_mv", "First spikes: the threshold's voltage minus the fast trough.", "mV", "",
           label="1st spikes: AHP"),
    Output("first_upstroke", "First spikes: the steepest rise.", "mV/ms", "", label="1st spikes: max rise"),
    Output("first_downstroke", "First spikes: the steepest fall.", "mV/ms", "",
           label="1st spikes: max fall"),
    Output("first_upstroke_phase", "First spikes: where the steepest rise comes, as a share of the amplitude.",
           "ratio", "", label="1st spikes: rise phase"),
    Output("first_downstroke_phase", "First spikes: where the steepest fall comes, as a share of the amplitude.",
           "ratio", "", label="1st spikes: fall phase"),
    Output("first_kink_max", "First spikes: the largest d2V/dt2 between the threshold and the steepest rise (the "
           "kink).", "mV/ms2", "", label="1st spikes: kink"),
    Output("first_time_to_kink_ms", "First spikes: threshold to the kink.", "ms", "",
           label="1st spikes: time to kink"),
    Output("first_kink_mv", "First spikes: the voltage at the kink.", "mV", "", label="1st spikes: kink "
           "voltage"),
    Output("first_initial_phase_slope", "First spikes: the slope of dV/dt against V at the threshold (onset "
           "rapidness).", "1/ms", "Naundorf, Wolf & Volgushev 2006",
           label="1st spikes: onset rapidness"),
    Output("first_kink_phase_slope", "First spikes: the slope of dV/dt against V at the kink.", "1/ms",
           "", label="1st spikes: phase slope at the kink"),
    Output("holding_current", "The holding current before the steps: the recorded current when the amplifier "
           "output is recorded, otherwise the command level.", "pA", "", label="Holding current"),
    Output("evaluation", "Statements derived from the data, each with the rule that produced it (the evaluation table).", "",
           "", label="Evaluation", headline=True),
    Output("tau_m", "Membrane time constant: a single exponential from 10% to 95% of the way to the peak of each "
           "small hyperpolarizing step, the median over steps.", "ms", "Gouwens et al. 2019",
           label="Membrane tau", headline=True),
    Output("capacitance", "Membrane capacitance: tau / the resistance over the same small steps (-100 to 0 pA).",
           "pF", "Johnston & Wu 1995", label="Capacitance"),
    Output("sag_ratio", "(V_peak - V_steady) / (V_peak - V_baseline) on the step whose peak is closest to the sag "
           "target, the peak averaged over 5 ms.", "ratio", "Gouwens et al. 2019", label="Sag ratio", headline=True),
    Output("sag_step_pa", "The step the sag was read on.", "pA", label="Sag: step"),
    Output("sag_amplitude", "V_steady - V_peak on the sag step: how far the voltage comes back from its peak.", "mV",
           label="Sag: amplitude"),
    Output("tau_n_steps", "Steps whose tau fit was kept; the membrane tau is their median.", "count",
           label="Membrane tau: steps"),
    Output("r_input_r2", "R squared of the line through the steady-state voltage against the current that gives the "
           "input resistance's slope.", "ratio", label="Input resistance slope: R squared"),
    Output("holding_current_source", "Where the holding current was read: the recorded current before the step, or "
           "the command when no current is recorded.", "", label="Holding current: from"),
    Output("rheobase", "The smallest depolarizing step with a full-sized action potential (its resolution is the "
           "protocol's step size).", "pA", "Gouwens et al. 2019", label="Rheobase", headline=True),
    Output("fi_gain", "f-I gain: the slope of the firing rate against the current from rheobase over the next steps.",
           "Hz/pA", "Dayan & Abbott 2001", label="f-I gain", headline=True),
    Output("max_rate", "The highest firing rate of a depolarizing step (spikes / step duration); none when no "
           "depolarizing step fires.", "Hz", label="Highest rate"),
    Output("ap_threshold", "Threshold of the first full spike at rheobase: searching back from its steepest rise "
           "to where dV/dt falls below 5% of that maximum.", "mV", "Gouwens et al. 2019", label="AP threshold (first spike at rheobase)",
           headline=True),
    Output("ap_threshold_20", "The same spike's voltage where dV/dt crosses 20 mV/ms (a fixed-rate threshold).",
           "mV", "", label="AP threshold at 20 mV/ms"),
    Output("ap_peak", "Peak of the first full spike at rheobase.", "mV", label="AP peak"),
    Output("ap_amplitude", "Peak minus threshold (20 mV/ms) of the first full spike at rheobase.", "mV",
           label="AP amplitude"),
    Output("ap_half_width", "Width of the first full spike at rheobase at half its height from the fast trough "
           "(from the threshold when that level lies below it).", "ms", "Gouwens et al. 2019", label="AP half-width"),
    Output("ap_half_width_threshold", "Its width at half the amplitude from the 20 mV/ms threshold to the peak.", "ms", label="AP half-width from threshold"),
    Output("ap_fast_trough", "The end of the spike: after its steepest fall, where the fall has slowed to 1% of it "
           "(the fast trough).", "mV", "Gouwens et al. 2019", label="Fast trough"),
    Output("ap_upstroke", "The steepest rise of the first full spike at rheobase.", "mV/ms", "Gouwens et al. 2019",
           label="AP max rise"),
    Output("ap_downstroke", "The steepest fall between its peak and the following trough (the downstroke).", "mV/ms", "Gouwens et al. 2019",
           label="AP max fall"),
    Output("ap_ahp", "The lowest voltage in the 50 ms after the peak (before the next spike), minus threshold.", "mV",
           label="AHP"),
    Output("ap_fast_ahp", "The lowest voltage 1-5 ms after repolarizing through threshold, minus threshold.", "mV",
           label="Fast AHP"),
    Output("ap_medium_ahp", "The lowest voltage 10-100 ms after the peak, minus threshold.", "mV", label="Medium AHP"),
    Output("first_latency", "Step onset to the threshold of the first spike at rheobase.", "ms", "Gouwens et al. 2019",
           label="First-spike latency"),
    Output("adaptation_index", "At the step the pattern is read on: mean((ISI[n+1] - ISI[n]) / (ISI[n+1] + ISI[n])).",
           "ratio", "Gouwens et al. 2019", label="Adaptation index"),
    Output("isi_ratio", "At the same step: the last interspike interval over the first.", "ratio",
           label="ISI ratio (last / first)"),
    Output("isi_cv", "At the same step: SD / mean of the interspike intervals.", "ratio", "Dayan & Abbott 2001",
           label="ISI CV"),
    Output("isi_step_pa", "The step on which the interval measures (adaptation index, ISI ratio, ISI CV) are "
           "made: the first step at rheobase + 40 to 60 pA, otherwise the step with the most spikes.", "pA", "Gouwens et al. 2019",
           label="Interval measures: step"),
    Output("attenuated_spikes", "Spikes that never reached full size (counted in rates, never in rheobase).", "count",
           label="Attenuated spikes"),
    Output("rebound_spikes", "Spikes in the window after the hyperpolarizing steps.", "count", label="Rebound spikes"),
)


@register("intrinsic", "Intrinsic properties and firing", version=2, types={"cc_steps"}, settings=SETTINGS,
          outputs=OUTPUTS, requires=("baseline",),
          description="Resting potential, input resistance, membrane tau, sag, capacitance, rheobase, the f-I curve, "
                      "action-potential shape and interspike-interval measures from a current-step family. Firing is "
                      "reported step by step rather than classified into a named pattern.",
          source="Gouwens et al. 2019; Dayan & Abbott 2001; Izhikevich 2007")
def run(recording: Recording, context, s) -> Result | None:
    out = Result("intrinsic", 1)
    if recording.is_voltage_clamp:
        return None
    why: list[str] = []
    step = find_step(recording, s["baseline_s"], s["steady_s"], why)
    if step is None:
        out.notes.append("no current step found in the command" + (f": {why[0]}" if why else ""))
        return out
    fs = recording.sample_rate_hz
    t = np.arange(recording.n_samples) / fs
    cmd = recording.command
    i0, i1 = int(round(step.start_s * fs)), int(round(step.end_s * fs))
    b0, b1 = int(round(step.baseline[0] * fs)), int(round(step.baseline[1] * fs))
    s0 = int(round(step.steady[0] * fs))
    out.notes += step.notes
    ledger.note(f"step {step.start_s * 1e3:.0f}-{step.end_s * 1e3:.0f} ms; baseline the {1e3 * (b1 - b0) / fs:.0f} ms "
                f"before, steady state the last {1e3 * (i1 - s0) / fs:.0f} ms", "intrinsic properties")
    rows, all_spikes, trains = [], [], []
    for k in range(recording.n_sweeps):
        v = np.asarray(recording.sweep(k), float)
        c = cmd[k]
        current = float(np.median(c[i0:i1]) - np.median(c[max(0, b0):b1])) if b1 > b0 else float(np.median(c[i0:i1]))
        sp = spikes.find(v, fs, purpose="action potentials in the step family", window=(i0, i1))
        tr = train(sp, fs, step.start_s, step.end_s, s)
        full = sum(1 for x in sp if step.start_s <= x.peak_index / fs < step.end_s and x.full)
        vb = float(np.mean(v[b0:b1])) if b1 > b0 else NAN
        vss = float(np.mean(v[s0:i1]))
        rows.append({"sweep": k, "current_pa": current, "v_baseline": vb, "v_steady": vss, "spikes": tr.count,
                     "full_spikes": full, "attenuated": tr.attenuated, "rate_hz": tr.rate_hz,
                     "latency_ms": tr.latency_ms, "isi_mean_ms": tr.isi_mean_ms, "isi_cv": tr.isi_cv,
                     "adaptation_index": tr.adaptation_index, "isi_ratio": tr.isi_ratio, "bursts": tr.bursts,
                     "doublet": tr.doublet})
        all_spikes.append(sp)
        trains.append(tr)
        if sp:                                  # each spike's threshold and peak, on the trace
            out.drawables += [Drawable("marker", what, x=np.array([getattr(x, idx) for x in sp]) / fs,
                                       y=np.array([v[getattr(x, idx)] for x in sp]), sweep=k, domain="time")
                              for what, idx in (("spike threshold", "threshold_index"), ("spike peak", "peak_index"))]
    cur = np.array([r["current_pa"] for r in rows])
    counts = np.array([r["spikes"] for r in rows])
    fulls = np.array([r["full_spikes"] for r in rows])
    vb = np.array([r["v_baseline"] for r in rows])
    vss = np.array([r["v_steady"] for r in rows])
    out.values["v_rest"] = float(np.nanmedian(vb)) if np.isfinite(vb).any() else NAN
    tol = 3.0                                   # a recorded command reads a pA or two off its nominal step
    sub = (counts == 0) & (cur <= tol)
    small = sub & (cur >= s["small_step_min_pa"] - tol)
    # input resistance: steady state against current over the small steps
    use = small if np.unique(np.round(cur[small], 3)).size >= 2 else sub
    if use is not small and np.unique(np.round(cur[use], 3)).size >= 2:
        out.notes.append(f"fewer than two steps between {s['small_step_min_pa']:g} and 0 pA: input resistance from every "
                         "subthreshold step (large steps recruit Ih and can lower it)")
    rin = NAN
    if np.unique(np.round(cur[use], 3)).size >= 2:
        fit = fits.linear(cur[use], vss[use], purpose="input resistance, steady-state V against I")
        rin = fit["slope"] * 1e3
        out.values["r_input_r2"] = fit.r2
        out.choose("input resistance", f"{int(use.sum())} steps from {cur[use].min():g} to {cur[use].max():g} pA",
                   "the steady state (the last 100 ms) against the current, over the small subthreshold steps")
        out.drawables.append(Drawable("curve", "v_steady: input resistance line", x=cur[use], y=fit.curve(cur[use]),
                                      domain="current"))
        if rin <= 0:
            out.flags.append(Flag("Rin not positive", "The steady-state voltage does not fall with the current: is the "
                                  "recorded channel the cell the steps went to?", source="Clampwise"))
    else:
        out.notes.append("not enough subthreshold steps for input resistance")
    # NeuroExpress's measures, step by step and at 0 pA: its
    # resistance at rest leads; the slope over the small steps stays beside it
    curves = iv_curves.analyze(recording, step, rows, all_spikes, s)
    if curves.moved:
        out.flags.append(Flag("moved by hand", f"{len(curves.moved)} point(s) read where they were put on Check: "
                              + "; ".join(curves.moved), source="",
                              severity="note"))
    out.values["r_input_slope"] = rin
    r0 = curves.values.get("r_steady0", NAN)
    if np.isfinite(r0) and r0 > 0:
        rin = r0
        slope = out.values["r_input_slope"]
        out.choose("input resistance", f"{r0:.1f} MOhm: the steady-state resistance of each step, (V - Vbaseline) / I, "
                   f"a line through the {curves.values['n_resistance_steps']} steps from "
                   f"{s['resistance_midpoint_pa']:g} to -{s['reject_range_pa']:g} pA read at 0 pA",
                   "Input resistance at rest; " + (f"the steady-state slope over {cur[use].min():g} to "
                                                   f"{cur[use].max():g} pA, for comparison: {slope:.1f} MOhm"
                                                   if np.isfinite(slope) else "no steady-state slope to compare it "
                                                   "with (fewer than two subthreshold steps)"))
    elif np.isfinite(rin):
        out.notes.append("too few hyperpolarizing steps near 0 pA for the resistance at rest: the input "
                         "resistance is the slope over the small steps")
    out.values["r_input"] = rin
    # sag and tau, on the hyperpolarizing subthreshold steps
    hyper = np.flatnonzero(sub & (cur < -tol))
    peaks = {}
    pend = max(s0, i0 + 1)
    for k in hyper:
        v = recording.sweep(int(k))
        i = i0 + int(np.argmin(v[i0:pend]))
        h = max(int(round(PEAK_AVERAGE_S / 2 * fs)), 0)
        peaks[int(k)] = (float(np.mean(v[max(i - h, i0):min(i + h + 1, pend)])), i)
    if peaks:
        k = min(peaks, key=lambda q: abs(peaks[q][0] - s["sag_target_mv"]))
        vp = peaks[k][0]
        den = vp - vb[k]
        out.values["sag_ratio"] = float((vp - vss[k]) / den) if den else NAN
        out.values["sag_step_pa"] = float(cur[k])
        out.values["sag_amplitude"] = float(vss[k] - vp)
        out.choose("sag", f"the {cur[k]:g} pA step (peak {vp:.1f} mV)",
                   f"its peak is the closest to {s['sag_target_mv']:g} mV; peak averaged over 5 ms between the step's "
                   f"onset and the steady state, steady state the last {1e3 * (i1 - s0) / fs:.0f} ms")
        if abs(den) < 2.0:
            out.flags.append(Flag("sag is noise", "The largest hyperpolarization is under 2 mV: the sag ratio is noise, "
                                  "not a measurement.", "2 mV", "", severity="note"))
        taus = []
        cand = [q for q in peaks if cur[q] >= s["small_step_min_pa"] - tol] or [max(peaks, key=lambda q: cur[q])]
        for q in cand:
            v = recording.sweep(q)
            tau, fit, win = tau_fit(t, v, i0, peaks[q][1], vb[q], s["tau_start_fraction"], s["tau_end_fraction"],
                                    s["tau_min_snr"], _noise_sd(v[b0:b1]), fs)
            if np.isfinite(tau):
                taus.append(tau)
                out.drawables.append(Drawable("curve", f"tau fit, {cur[q]:g} pA", x=t[win[0]:win[1] + 1],
                                              y=fit.curve(t[win[0]:win[1] + 1]), sweep=q, domain="time"))
        out.values["tau_m"] = float(np.median(taus)) if taus else NAN
        out.values["tau_n_steps"] = len(taus)
        if taus:
            out.choose("membrane tau", f"the median of {len(taus)} step(s)", f"single exponentials from "
                       f"{100 * s['tau_start_fraction']:.0f}% to {100 * s['tau_end_fraction']:.0f}% of the way to each "
                       "step's peak, steps deeper than 20 baseline-noise SDs")
            if out.values["tau_m"] >= 990:
                out.flags.append(Flag("tau at limit", "Tau reached the fit's limit: no charging curve to fit.",
                                      source="Clampwise"))
        else:
            out.notes.append("no hyperpolarizing step deep enough over the noise to fit tau")
    else:
        out.values.update({"sag_ratio": NAN, "tau_m": NAN})
        out.notes.append("no subthreshold hyperpolarizing step: no sag or tau")
    # tau and the resistance from the same small steps (the input resistance at rest
    # comes from other steps); NeuroExpress's capacitance at rest is beside it
    r_small = out.values.get("r_input_slope", NAN)
    out.values["capacitance"] = (out.values["tau_m"] / r_small * 1e3 if np.isfinite(out.values.get("tau_m", NAN))
                                 and np.isfinite(r_small) and r_small > 0 else NAN)
    # rheobase, the first spike, the f-I curve
    firing = np.flatnonzero((fulls > 0) & (cur > 0))
    attenuated = int(sum(r["attenuated"] for r in rows))
    out.values["attenuated_spikes"] = attenuated
    if attenuated:
        out.notes.append(f"{attenuated} spike(s) did not reach full size (amplitude accommodation or depolarization "
                         "block): counted in the rates, never in the rheobase")
    # the highest rate on a depolarizing step; none when none fires (spikes on a
    # hyperpolarizing step are rebound or spontaneous, not driven by the step)
    driven = [r["rate_hz"] for r in rows if r["current_pa"] > tol and r["spikes"] > 0]
    out.values["max_rate"] = float(np.nanmax(driven)) if driven else NAN
    order = np.argsort(cur)
    out.tables["fi"] = [{"current_pa": float(cur[k]), "rate_hz": rows[k]["rate_hz"], "spikes": int(counts[k])}
                        for k in order if cur[k] > 0]
    if firing.size:
        rk = int(firing[np.argmin(cur[firing])])
        out.values["rheobase"] = float(cur[rk])
        ev = [x for x in all_spikes[rk] if step.start_s <= x.peak_index / fs < step.end_s and x.full]
        x = ev[0]
        nxt = ev[1].threshold_index if len(ev) > 1 else None
        fast, medium = fast_medium_ahp(np.asarray(recording.sweep(rk), float), fs, x, nxt)
        out.values.update({"ap_threshold": x.ipfx_threshold_mv, "ap_threshold_20": x.threshold_mv,
                           "ap_peak": x.peak_mv, "ap_amplitude": x.amplitude_mv, "ap_half_width": x.half_width_ms,
                           "ap_half_width_threshold": x.half_width_threshold_ms, "ap_fast_trough": x.fast_trough_mv,
                           "ap_upstroke": x.upstroke, "ap_downstroke": x.downstroke, "ap_ahp": x.ahp_mv,
                           "ap_fast_ahp": fast, "ap_medium_ahp": medium, "first_latency": trains[rk].latency_ms})
        out.choose("first spike", f"the first full spike of the rheobase step ({cur[rk]:g} pA, sweep {rk + 1}) at "
                   f"{x.peak_index / fs * 1e3:.1f} ms", "the first full spike at rheobase")
        # gain: rate against current from rheobase over the next steps
        above = [k for k in order if cur[k] >= cur[rk]][: s["gain_steps"] + 1]
        if len(above) >= 2:
            g = fits.linear(cur[above], np.array([rows[k]["rate_hz"] for k in above]), purpose="f-I gain")
            out.values["fi_gain"] = g["slope"]
            out.choose("f-I gain", f"{len(above)} steps from rheobase ({cur[above[0]]:g} to {cur[above[-1]]:g} pA)",
                       "a straight line through rate against current")
        else:
            out.values["fi_gain"] = NAN
        # the step the pattern is read on: IPFX's hero step
        hero_c = [k for k in order if cur[rk] + 39 <= cur[k] <= cur[rk] + 61]
        hk = hero_c[0] if hero_c else int(np.argmax(counts))
        hero = trains[hk]
        out.values.update({"adaptation_index": hero.adaptation_index, "isi_ratio": hero.isi_ratio,
                           "isi_cv": hero.isi_cv, "isi_step_pa": float(cur[hk])})
        out.choose("interval measures", f"the {cur[hk]:g} pA step ({hero.count} spikes)",
                   "the first step at rheobase + 40 to 60 pA" if hero_c else "no step at rheobase + 40-60 pA; the step "
                   "with the most spikes")
    else:
        out.values.update({"rheobase": NAN, "fi_gain": NAN})
        out.notes.append("no depolarizing step produced a full-sized action potential: no rheobase"
                         + (" (only attenuated events)" if counts.any() else ""))
    # rebound spikes after the hyperpolarizing steps
    reb = 0
    i_reb = min(recording.n_samples, i1 + int(round(s["rebound_window_ms"] / 1e3 * fs)))
    for k in np.flatnonzero(cur < -tol):
        reb += len(spikes.find(recording.sweep(int(k)), fs, purpose="rebound spikes", window=(i1, i_reb)))
    out.values["rebound_spikes"] = reb
    for row, extra in zip(rows, curves.rows):
        row.update({k: v for k, v in extra.items() if k not in ("sweep", "current_pa")})
    out.values.update(curves.values)
    out.drawables += curves.drawables
    out.tables["evaluation"] = [{"statement": a, "why": b, "warning": w} for a, b, w in curves.statements]
    out.values["evaluation"] = "; ".join(a for a, _, _ in curves.statements) or "nothing to remark"
    for a, b, w in curves.statements:
        if w:
            out.flags.append(Flag(a, b, source="", severity="note"))
    out.tables["steps"] = rows
    out.tables["spikes"] = [
        {"sweep": k, "current_pa": float(cur[k]), "spike": j + 1, "time_ms": x.peak_index / fs * 1e3,
         "threshold_mv": x.ipfx_threshold_mv, "threshold_20_mv": x.threshold_mv, "peak_mv": x.peak_mv,
         "amplitude_mv": x.amplitude_mv, "half_width_ms": x.half_width_ms,
         "half_width_threshold_ms": x.half_width_threshold_ms, "fast_trough_mv": x.fast_trough_mv, "upstroke": x.upstroke,
         "downstroke": x.downstroke, "ahp_mv": x.ahp_mv, "full": x.full, "clipped": x.clipped}
        for k, sp in enumerate(all_spikes) for j, x in enumerate(sp)]
    return out
