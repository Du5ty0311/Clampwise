"""The manual workbench, the part with no window in it: what Clampfit's cursors, Statistics and fits do, as
functions on a recording, so a measure made by hand is made again from its saved description (B18: provenance).

A hand measurement is a selection of sweeps (overlaid, averaged, or the one nearest their mean), a stack of
transforms that never touch the raw data (a baseline by region or by a sloping line, Bessel or Gaussian low-pass,
high-pass, notch, blanking, subtracting a trace, scaling, an offset), named regions (baseline, measure, fit;
a region can repeat relative to every stimulus pulse), the region statistics of Clampfit's list, a fit on a
region (``tools/fits.py``), points set by a click (an onset, a peak, a threshold), sweeps excluded with a reason,
and a table of each measure per sweep (against sweep, time, command level or another measure for a plot). All of
it is one dictionary (``HandMeasurement.to_dict``), kept in the project and run again on demand; a template is
the same without the sweeps, run on every file of a protocol.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..core.recording import Recording, Role, parse_sweeps
from .tools import filters, fits, kinetics

NAN = float("nan")

MODES = {"each": "Every sweep", "average": "The average of the sweeps", "median": "The median of the sweeps",
         "typical": "The sweep closest to their mean"}
REGION_KINDS = ("baseline", "measure", "fit")
TRANSFORMS = {
    "baseline_region": "Baseline: the mean of a region taken off",
    "baseline_line": "Baseline: a sloping line through two regions taken off",
    "lowpass": "Low-pass (Bessel, 4th order)", "highpass": "High-pass (Butterworth, 2nd order)",
    "notch": "Notch (the mains and its harmonics)", "gaussian": "Gaussian smoothing",
    "blank": "Blank a stretch after each stimulus", "subtract": "Subtract a trace", "scale": "Scale",
    "offset": "Offset",
}
MEASURES = {
    "mean": ("Mean", "the mean of the samples in the region, less the baseline"),
    "sd": ("SD", "the standard deviation of the samples in the region"),
    "min": ("Minimum", "the lowest sample, less the baseline"),
    "max": ("Maximum", "the highest sample, less the baseline"),
    "peak": ("Peak", "the extreme in the response's direction (set, or the larger departure from the baseline), "
                     "on a running mean of the smoothing points, less the baseline"),
    "antipeak": ("Antipeak", "the extreme the other way, less the baseline"),
    "peak_to_peak": ("Peak to peak", "the maximum minus the minimum"),
    "time_of_peak": ("Time of peak", "when the peak comes, from the region's start (per pulse: from the pulse)"),
    "time_of_antipeak": ("Time of antipeak", "when the antipeak comes"),
    "area": ("Area", "the integral of the trace less the baseline over the region (units x ms)"),
    "half_width": ("Half-width", "the width at half the peak, between the last crossing before it and the first "
                                 "after, interpolated"),
    "rise_time": ("Rise time", "from the lower to the upper rise percentage of the peak (10-90% unless set), "
                               "the last crossings before the peak"),
    "decay_time": ("Decay time", "from the upper to the lower decay percentage of the peak after it (90-10% "
                                 "unless set)"),
    "max_rise_slope": ("Max rise slope", "the steepest rise in the region, per ms"),
    "time_of_max_rise": ("Time of max rise", "when the rise is steepest"),
    "max_decay_slope": ("Max decay slope", "the steepest fall in the region, per ms"),
    "time_of_max_decay": ("Time of max decay", "when the fall is steepest"),
    "onset_latency": ("Onset latency", "when the response begins: by threshold (a departure past the set noise "
                                       "multiple, sustained 0.5 ms), by the extrapolated foot (a line through the "
                                       "20-80% rise back to the baseline) or by maximal curvature before the peak"),
    "slope": ("Slope", "a straight line through the region, per ms"),
    "value_at": ("Value at a time", "the trace at the set time, less the baseline (the region's start unless set)"),
    "threshold_crossings": ("Threshold crossings", "how many times the trace crosses the set level upward"),
    "spike_count": ("Spike count", "action potentials in the region (an upstroke past 20 mV/ms rising 20 mV)"),
    "instantaneous_frequency": ("Instantaneous frequency", "1 over the mean interval between the spikes in the "
                                                           "region, Hz"),
}
# each measure's units: "u" stands for the measured channel's own (mV, pA)
MEASURE_UNITS = {"mean": "u", "sd": "u", "min": "u", "max": "u", "peak": "u", "antipeak": "u", "peak_to_peak": "u",
                 "time_of_peak": "ms", "time_of_antipeak": "ms", "area": "u·ms", "half_width": "ms", "rise_time": "ms",
                 "decay_time": "ms", "max_rise_slope": "u/ms", "time_of_max_rise": "ms", "max_decay_slope": "u/ms",
                 "time_of_max_decay": "ms", "onset_latency": "ms", "slope": "u/ms", "value_at": "u",
                 "threshold_crossings": "", "spike_count": "", "instantaneous_frequency": "Hz"}
FIT_MODELS = {
    "exp1": "One exponential (offset free)", "exp1_fixed": "One exponential (offset fixed at the baseline)",
    "exp2": "Two exponentials", "rise1": "Exponential rise", "synaptic": "Synaptic (rise and decay exponentials)",
    "alpha": "Alpha function", "linear": "Straight line", "poly2": "Polynomial, 2nd order",
    "poly3": "Polynomial, 3rd order", "power": "Power, a x^b", "gaussian": "Gaussian", "boltzmann": "Boltzmann",
    "hill": "Hill (logistic)", "custom": "Typed expression in t with a, b, c, d",
}
ONSET_METHODS = ("threshold", "foot", "curvature")


# ------------------------------------------------------------------ what a hand measurement is made of

@dataclass
class Region:
    name: str
    start_s: float
    end_s: float
    kind: str = "measure"                  # baseline | measure | fit
    per_pulse: bool = False                # start and end relative to each stimulus pulse
    pulse: int = 0                         # with per_pulse: 0 every pulse, else that pulse (from 1)

    def windows(self, recording: Recording, sweep: int) -> list[tuple[float, float, int]]:
        """(start_s, end_s, pulse number from 1 or 0) in ``sweep``: one, or one per pulse."""
        if not self.per_pulse:
            return [(self.start_s, self.end_s, 0)]
        times = recording.stimulus.times_s
        if sweep >= len(times) or not len(times[sweep]):
            return []
        out = []
        for j, on in enumerate(np.asarray(times[sweep], float)):
            if self.pulse and j + 1 != self.pulse:
                continue
            out.append((float(on) + self.start_s, float(on) + self.end_s, j + 1))
        return out


@dataclass
class Transform:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass
class Selection:
    sweeps: list[int] = field(default_factory=list)
    channel: str = ""                      # "" is the recorded signal
    mode: str = "each"

    def text(self) -> str:
        return ", ".join(str(k + 1) for k in self.sweeps)


@dataclass
class Override:
    sweep: int
    what: str                              # onset | peak | threshold
    time_s: float
    region: str = ""


@dataclass
class Exclusion:
    sweep: int
    reason: str


@dataclass
class Options:
    direction: str = "auto"                # auto | up | down
    smooth_points: int = 1
    rise_pct: tuple = (10.0, 90.0)
    decay_pct: tuple = (90.0, 10.0)
    onset_method: str = "threshold"
    onset_noise_sd: float = 2.0
    value_at_s: float = NAN
    threshold: float = NAN
    baseline_region: str = ""              # which baseline region the measures use ("" : the first one)


@dataclass
class HandMeasurement:
    """One hand measurement, with everything needed to make it again."""

    name: str
    file: str = ""
    selection: Selection = field(default_factory=Selection)
    transforms: list[Transform] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    measures: list[str] = field(default_factory=lambda: ["peak", "time_of_peak", "area"])
    fit_model: str = ""
    fit_region: str = ""
    fit_custom: str = ""
    options: Options = field(default_factory=Options)
    overrides: list[Override] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    made_at: str = ""
    app_version: str = ""
    from_template: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["options"]["rise_pct"] = list(self.options.rise_pct)
        d["options"]["decay_pct"] = list(self.options.decay_pct)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HandMeasurement":
        d = dict(d)
        d["selection"] = Selection(**d.get("selection", {}))
        d["transforms"] = [Transform(**x) for x in d.get("transforms", [])]
        d["regions"] = [Region(**x) for x in d.get("regions", [])]
        d["overrides"] = [Override(**x) for x in d.get("overrides", [])]
        d["exclusions"] = [Exclusion(**x) for x in d.get("exclusions", [])]
        o = dict(d.get("options", {}))
        o["rise_pct"] = tuple(o.get("rise_pct", (10.0, 90.0)))
        o["decay_pct"] = tuple(o.get("decay_pct", (90.0, 10.0)))
        d["options"] = Options(**{k: v for k, v in o.items() if k in Options.__dataclass_fields__})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        """How it was measured, in one paragraph (the provenance Results shows)."""
        parts = [f"measured by hand on sweeps {self.selection.text() or 'none'} "
                 f"({MODES.get(self.selection.mode, self.selection.mode).lower()}"
                 + (f", channel {self.selection.channel}" if self.selection.channel else "") + ")"]
        if self.transforms:
            parts.append("transforms: " + "; ".join(describe_transform(x) for x in self.transforms))
        if self.regions:
            parts.append("regions: " + "; ".join(
                f"{r.name} ({r.kind}) {r.start_s * 1e3:.1f}-{r.end_s * 1e3:.1f} ms"
                + (" from each pulse" if r.per_pulse and not r.pulse else f" from pulse {r.pulse}" if r.per_pulse else "")
                for r in self.regions))
        if self.measures:
            parts.append("measures: " + ", ".join(MEASURES.get(m, (m,))[0].lower() for m in self.measures))
        if self.fit_model:
            parts.append(f"fit: {FIT_MODELS.get(self.fit_model, self.fit_model)} on {self.fit_region or 'the fit region'}"
                         + (f" ({self.fit_custom})" if self.fit_model == "custom" else ""))
        if self.overrides:
            parts.append("set by hand: " + "; ".join(f"{o.what} of sweep {o.sweep + 1} at {o.time_s * 1e3:.2f} ms"
                                                      for o in self.overrides))
        if self.exclusions:
            parts.append("left out: " + "; ".join(f"sweep {e.sweep + 1} ({e.reason})" for e in self.exclusions))
        if self.from_template:
            parts.append(f"from the template '{self.from_template}'")
        return ". ".join(parts) + "."


@dataclass
class Template:
    """A hand measurement's regions, measures, fit and transforms, to run on every file of a protocol."""

    name: str
    protocol: str = "*"                    # a pattern on the protocol's name (fnmatch)
    measurement: dict = field(default_factory=dict)      # HandMeasurement.to_dict without sweeps and file

    @classmethod
    def from_measurement(cls, name: str, protocol: str, m: HandMeasurement) -> "Template":
        d = m.to_dict()
        d["selection"]["sweeps"] = []
        d.update({"file": "", "overrides": [], "exclusions": [], "made_at": "", "name": name,
                  "from_template": name})
        return cls(name, protocol, d)

    def matches(self, protocol: str) -> bool:
        import fnmatch

        return fnmatch.fnmatchcase((protocol or "").lower(), self.protocol.lower())

    def measurement_for(self, recording: Recording) -> HandMeasurement:
        m = HandMeasurement.from_dict(self.measurement)
        m.name = self.name
        m.file = str(recording.source)
        m.selection.sweeps = list(range(recording.n_sweeps))
        m.from_template = self.name
        return m


def describe_transform(x: Transform) -> str:
    p = x.params
    return {
        "baseline_region": lambda: f"baseline from the region {p.get('region', '')}",
        "baseline_line": lambda: f"a line through {p.get('region', '')} and {p.get('region2', '')} taken off",
        "lowpass": lambda: f"low-pass {p.get('hz', 0):g} Hz (Bessel)",
        "highpass": lambda: f"high-pass {p.get('hz', 0):g} Hz",
        "notch": lambda: f"notch {p.get('hz', 60):g} Hz and {p.get('harmonics', 1)} harmonic(s)",
        "gaussian": lambda: f"Gaussian smoothing, sigma {p.get('ms', 0):g} ms",
        "blank": lambda: f"{p.get('ms', 0):g} ms blanked after each stimulus",
        "subtract": lambda: f"minus {p.get('label', 'a trace')}" + (f" x {p['scale']:g}" if p.get("scale", 1) != 1 else ""),
        "scale": lambda: f"x {p.get('factor', 1):g}",
        "offset": lambda: f"{p.get('value', 0):+g} added",
    }.get(x.kind, lambda: x.kind)()


# ------------------------------------------------------------------ sweeps and traces

def parse_sweep_words(text: str, n_sweeps: int) -> list[int]:
    """Sweeps from words: ``1-10, 15``, ``every 3rd``, ``odd``, ``even``, ``all``, ``last 5``, ``first 5``."""
    text = (text or "").strip().lower()
    if not text or text == "all":
        return list(range(n_sweeps))
    m = re.match(r"every\s+(\d+)(?:st|nd|rd|th)?(?:\s+from\s+(\d+))?$", text)
    if m:
        k, start = int(m.group(1)), int(m.group(2) or 1) - 1
        return list(range(max(start, 0), n_sweeps, max(k, 1)))
    if text == "odd":
        return list(range(0, n_sweeps, 2))
    if text == "even":
        return list(range(1, n_sweeps, 2))
    m = re.match(r"(first|last)\s+(\d+)$", text)
    if m:
        k = min(int(m.group(2)), n_sweeps)
        return list(range(k)) if m.group(1) == "first" else list(range(n_sweeps - k, n_sweeps))
    return parse_sweeps(text, n_sweeps)


def channel_named(recording: Recording, name: str):
    if not name:
        return recording.signal_channel
    for c in recording.channels:
        if c.name == name:
            return c
    raise KeyError(f"no channel named {name!r}")


def raw(recording: Recording, sweep: int, channel: str = "") -> np.ndarray:
    return np.asarray(recording.sweep(sweep, channel_named(recording, channel)), float)


def _region_named(regions: list[Region], name: str) -> Region | None:
    return next((r for r in regions if r.name == name), None)


def _level_in(t: np.ndarray, y: np.ndarray, start_s: float, end_s: float) -> float:
    a, b = np.searchsorted(t, [start_s, end_s])
    return float(np.nanmean(y[a:b])) if b > a else NAN


def transform(recording: Recording, sweep: int, y: np.ndarray, stack: list[Transform], regions: list[Region],
              aux: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """``y`` (one sweep of a channel) with the stack applied in order; the raw data stay as they are. ``aux``
    holds the traces a 'subtract' names (another sweep, a file's average), by label."""
    fs = recording.sample_rate_hz
    t = recording.time_s
    out = np.array(y, float, copy=True)
    for x in stack:
        p = x.params
        if x.kind == "baseline_region":
            r = _region_named(regions, p.get("region", ""))
            if r is not None:
                wins = r.windows(recording, sweep)
                if wins:
                    out = out - _level_in(t, out, wins[0][0], wins[0][1])
        elif x.kind == "baseline_line":
            r1, r2 = _region_named(regions, p.get("region", "")), _region_named(regions, p.get("region2", ""))
            if r1 is not None and r2 is not None:
                w1, w2 = r1.windows(recording, sweep), r2.windows(recording, sweep)
                if w1 and w2:
                    x1, x2 = (w1[0][0] + w1[0][1]) / 2, (w2[0][0] + w2[0][1]) / 2
                    y1, y2 = _level_in(t, out, *w1[0][:2]), _level_in(t, out, *w2[0][:2])
                    if x2 != x1 and np.isfinite(y1) and np.isfinite(y2):
                        out = out - (y1 + (y2 - y1) * (t - x1) / (x2 - x1))
        elif x.kind == "lowpass":
            hz = float(p.get("hz", 1000.0))
            if 0 < hz < fs / 2:
                out = filters.apply(out, fs, filters.bessel_low_pass(hz), purpose="workbench low-pass")
        elif x.kind == "highpass":
            hz = float(p.get("hz", 1.0))
            if 0 < hz < fs / 2:
                out = filters.apply(out, fs, filters.high_pass(hz), purpose="workbench high-pass")
        elif x.kind == "notch":
            out = filters.notch(out, fs, float(p.get("hz", 60.0)), int(p.get("harmonics", 1)))
        elif x.kind == "gaussian":
            out = filters.gaussian(out, fs, float(p.get("ms", 0.5)) / 1e3)
        elif x.kind == "blank":
            ms = float(p.get("ms", 0.0))
            times = recording.stimulus.times_s
            if ms > 0 and sweep < len(times):
                for on in np.asarray(times[sweep], float):
                    a, b = np.searchsorted(t, [on, on + ms / 1e3])
                    if b > a:
                        out[a:b] = np.nan
        elif x.kind == "subtract":
            other = (aux or {}).get(p.get("label", ""))
            if other is not None:
                other = np.asarray(other, float)
                if other.size != out.size:                    # another file's rate: resampled onto this sweep
                    other = np.interp(t, np.arange(other.size) / float(p.get("fs", fs)), other)
                out = out - float(p.get("scale", 1.0)) * other
        elif x.kind == "scale":
            out = out * float(p.get("factor", 1.0))
        elif x.kind == "offset":
            out = out + float(p.get("value", 0.0))
    return out


def traces(recording: Recording, m: HandMeasurement, aux: dict[str, np.ndarray] | None = None
           ) -> list[tuple[str, int | None, np.ndarray]]:
    """(label, sweep or None for a combined trace, y) for the selection: every sweep (the excluded ones left
    out), with the average / median / typical trace when the mode asks for one."""
    excluded = {e.sweep for e in m.exclusions}
    sweeps = [k for k in m.selection.sweeps if 0 <= k < recording.n_sweeps and k not in excluded]
    rows = [(f"sweep {k + 1}", k, transform(recording, k, raw(recording, k, m.selection.channel), m.transforms,
                                           m.regions, aux)) for k in sweeps]
    if m.selection.mode == "each" or not rows:
        return rows
    stack = np.vstack([y for _, _, y in rows])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if m.selection.mode == "average":
            return rows + [(f"average of {len(rows)}", None, np.nanmean(stack, axis=0))]
        if m.selection.mode == "median":
            return rows + [(f"median of {len(rows)}", None, np.nanmedian(stack, axis=0))]
        mean = np.nanmean(stack, axis=0)
        k = int(np.nanargmin(np.nanmean((stack - mean) ** 2, axis=1)))
    return rows + [(f"typical: {rows[k][0]}", rows[k][1], rows[k][2])]


def combine(a: np.ndarray, b: np.ndarray, op: str, fs_a: float = 1.0, fs_b: float = 1.0) -> np.ndarray:
    """Trace arithmetic: a - b, a + b, a * b, a / b, b resampled onto a when their rates differ."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size != b.size or fs_a != fs_b:
        b = np.interp(np.arange(a.size) / fs_a, np.arange(b.size) / fs_b, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"-": a - b, "+": a + b, "*": a * b, "/": a / b}[op]


# ------------------------------------------------------------------ the statistics of a region

def _spikes_in(y: np.ndarray, fs: float) -> list[int]:
    """Upstrokes past 20 mV/ms held 0.1 ms that rise 20 units above the level before them: their first samples."""
    if y.size < 4:
        return []
    d = np.diff(y) * fs / 1e3
    above = d >= 20.0
    hold = max(2, int(round(0.1e-3 * fs)))
    runs = np.convolve(above.astype(int), np.ones(hold, int), mode="valid") == hold
    starts = np.flatnonzero(np.diff(np.r_[0, runs.astype(int)]) == 1)
    out = []
    for s in starts:
        if out and s - out[-1] < int(0.002 * fs):
            continue
        peak = y[s:s + int(0.005 * fs)].max() if s + 2 < y.size else y[s]
        if peak - y[max(s - int(0.001 * fs), 0)] >= 20.0:
            out.append(int(s))
    return out


def statistics(t: np.ndarray, y: np.ndarray, start_s: float, end_s: float, fs: float, *, baseline: float = 0.0,
               noise_sd: float = NAN, options: Options | None = None, origin_s: float | None = None,
               override: dict[str, float] | None = None) -> dict[str, float]:
    """Clampfit's Statistics on the stretch [start_s, end_s) of ``y`` (NaN where blanked), less ``baseline``.
    Times are from ``origin_s`` (the region's start unless given: a pulse's time for a region per pulse).
    ``override``: a time (s) set by a click for 'onset' or 'peak', used instead of the search."""
    o = options or Options()
    override = override or {}
    a, b = np.searchsorted(t, [start_s, end_s])
    out = {k: NAN for k in MEASURES}
    if b - a < 2:
        return out
    seg = np.asarray(y[a:b], float) - baseline
    tt = t[a:b]
    ok = np.isfinite(seg)
    if ok.sum() < 2:
        return out
    org = start_s if origin_s is None else origin_s
    n = max(1, int(o.smooth_points))
    sm = kinetics.smooth(np.where(ok, seg, np.nanmean(seg)), n) if n > 1 else seg
    out["mean"] = float(np.nanmean(seg))
    out["sd"] = float(np.nanstd(seg))
    out["min"], out["max"] = float(np.nanmin(seg)), float(np.nanmax(seg))
    out["peak_to_peak"] = out["max"] - out["min"]
    direction = o.direction
    if direction == "auto":
        direction = "up" if abs(float(np.nanmax(sm))) >= abs(float(np.nanmin(sm))) else "down"
    sign = 1.0 if direction == "up" else -1.0
    dev = np.where(ok, sign * sm, -np.inf)
    i_pk = int(np.argmax(dev))
    if "peak" in override and np.isfinite(override["peak"]):
        j = int(np.clip(np.searchsorted(tt, override["peak"]), 0, tt.size - 1))
        if np.isfinite(sm[j]):
            i_pk = j
    peak = float(sign * sm[i_pk])
    out["peak"] = peak
    out["time_of_peak"] = (float(tt[i_pk]) - org) * 1e3
    anti = np.where(ok, -sign * sm, -np.inf)
    i_an = int(np.argmax(anti))
    out["antipeak"] = float(sign * sm[i_an])
    out["time_of_antipeak"] = (float(tt[i_an]) - org) * 1e3
    good = np.where(ok, seg, 0.0)
    out["area"] = float(np.trapezoid(good, tt)) * 1e3
    d = np.gradient(np.where(ok, sm, np.nan), tt) / 1e3            # per ms
    if np.isfinite(d).any():
        i_r, i_f = int(np.nanargmax(d)), int(np.nanargmin(d))
        out["max_rise_slope"], out["time_of_max_rise"] = float(d[i_r]), (float(tt[i_r]) - org) * 1e3
        out["max_decay_slope"], out["time_of_max_decay"] = float(d[i_f]), (float(tt[i_f]) - org) * 1e3
    devp = sign * sm                                                 # the response positive
    if peak and sign * peak > 0:
        pk = sign * peak
        lo, hi = o.rise_pct[0] / 100.0, o.rise_pct[1] / 100.0
        out["rise_time"] = kinetics.rise_time(tt, np.nan_to_num(devp, nan=-np.inf), 0, i_pk, lo, hi) * 1e3
        out["half_width"] = kinetics.half_width(tt, np.nan_to_num(devp, nan=-np.inf), 0, i_pk, devp.size) * 1e3
        after = np.nan_to_num(devp[i_pk:], nan=np.inf)
        hi_d, lo_d = o.decay_pct[0] / 100.0, o.decay_pct[1] / 100.0
        i_hi = np.flatnonzero(after <= hi_d * pk)
        i_lo = np.flatnonzero(after <= lo_d * pk)
        if i_hi.size and i_lo.size and i_lo[0] > i_hi[0]:
            h, l = i_pk + int(i_hi[0]), i_pk + int(i_lo[0])
            out["decay_time"] = (kinetics.interp_crossing(tt, devp, l - 1, l, lo_d * pk)
                                 - kinetics.interp_crossing(tt, devp, h - 1, h, hi_d * pk)) * 1e3
        # the onset
        if "onset" in override and np.isfinite(override["onset"]):
            out["onset_latency"] = (override["onset"] - org) * 1e3
        else:
            noise = noise_sd if np.isfinite(noise_sd) and noise_sd > 0 else (kinetics.robust_noise(seg[:max(3, (b - a) // 10)]) or out["sd"] or 1e-9)
            if o.onset_method == "threshold":
                on = kinetics.find_onset(np.nan_to_num(devp, nan=-np.inf), 0, i_pk + 1, o.onset_noise_sd * noise,
                                         max(1, int(round(0.5e-3 * fs))))
                if on is not None:
                    out["onset_latency"] = (float(tt[on]) - org) * 1e3
            elif o.onset_method == "foot":
                rise = devp[:i_pk + 1]
                lo_i = np.flatnonzero(rise < 0.2 * pk)
                hi_i = np.flatnonzero(rise < 0.8 * pk)
                if lo_i.size and hi_i.size and hi_i[-1] > lo_i[-1]:
                    x0, x1 = int(lo_i[-1]), int(hi_i[-1])
                    fit = fits.linear(tt[x0:x1 + 1], devp[x0:x1 + 1], purpose="onset by the extrapolated foot")
                    if fit.ok and fit["slope"] > 0:
                        out["onset_latency"] = (-fit["intercept"] / fit["slope"] - org) * 1e3
            else:
                d2 = np.gradient(np.gradient(np.nan_to_num(sm, nan=0.0), tt), tt)
                seg2 = d2[:i_pk + 1] * sign
                if seg2.size > 3:
                    out["onset_latency"] = (float(tt[int(np.argmax(seg2))]) - org) * 1e3
    fit = fits.linear(tt[ok], seg[ok], purpose="slope of a region") if ok.sum() >= 2 else None
    if fit is not None and fit.ok:
        out["slope"] = fit["slope"] / 1e3
    at = o.value_at_s if np.isfinite(o.value_at_s) else start_s
    j = int(np.clip(np.searchsorted(t, at), 0, t.size - 1))
    out["value_at"] = float(y[j] - baseline) if np.isfinite(y[j]) else NAN
    if np.isfinite(o.threshold):
        above = np.nan_to_num(seg, nan=-np.inf) > o.threshold
        out["threshold_crossings"] = float(np.sum(np.diff(above.astype(int)) == 1))
    spikes = _spikes_in(np.nan_to_num(seg, nan=np.nanmean(seg)), fs)
    out["spike_count"] = float(len(spikes))
    if len(spikes) >= 2:
        out["instantaneous_frequency"] = float(fs / np.mean(np.diff(spikes)))
    elif spikes:
        out["instantaneous_frequency"] = 0.0
    return out


# ------------------------------------------------------------------ fits on a region

def fit_region(t: np.ndarray, y: np.ndarray, start_s: float, end_s: float, model: str, *, baseline: float = 0.0,
               custom: str = "") -> dict:
    """A fit on [start_s, end_s): the model, its parameters with 95% intervals, r2, the curve to draw (t, y)
    and the residuals. ``t`` in the fit is ms from the region's start."""
    a, b = np.searchsorted(t, [start_s, end_s])
    seg = np.asarray(y[a:b], float) - baseline
    tt = (t[a:b] - t[a]) * 1e3 if b > a else np.array([])
    ok = np.isfinite(seg)
    out = {"model": model, "ok": False, "params": {}, "ci": {}, "r2": NAN, "n": int(ok.sum()), "curve_t": np.array([]),
           "curve_y": np.array([]), "residuals": np.array([]), "words": ""}
    if ok.sum() < 5:
        out["words"] = "too few points to fit"
        return out
    x, z = tt[ok], seg[ok]
    fit = None
    pred = None
    try:
        if model in ("exp1", "exp1_fixed"):
            fit = fits.exp_decay(x, z, offset=None if model == "exp1" else 0.0, purpose="workbench fit")
        elif model == "exp2":
            fit = fits.biexp_decay(x, z, purpose="workbench fit")
        elif model == "rise1":
            fit = fits.exp_rise(x, z, purpose="workbench fit")
        elif model == "synaptic":
            fit = fits.synaptic_waveform(x, z, purpose="workbench fit")
        elif model == "alpha":
            def alpha(tx, amp, tau, t0):
                u = np.maximum(tx - t0, 0.0) / tau
                return amp * u * np.exp(1 - u)
            i_pk = int(np.argmax(np.abs(z)))
            fit = fits.nonlinear(alpha, x, z, [z[i_pk], max(x[i_pk], 0.1), 0.0], ("amp", "tau", "t0"), label="alpha",
                                 purpose="workbench fit", maxfev=5000)
        elif model == "linear":
            fit = fits.linear(x, z, purpose="workbench fit")
        elif model in ("poly2", "poly3"):
            deg = 2 if model == "poly2" else 3
            coef = np.polyfit(x, z, deg)
            pred = np.polyval(coef, x)
            out.update({"ok": True, "params": {f"c{deg - i}": float(c) for i, c in enumerate(coef)},
                        "r2": float(1 - np.sum((z - pred) ** 2) / max(np.sum((z - z.mean()) ** 2), 1e-12))})
        elif model == "power":
            def power(tx, amp, b_):
                return amp * np.power(np.maximum(tx, 1e-9), b_)
            fit = fits.nonlinear(power, x, z, [z[-1] if z[-1] else 1.0, 1.0], ("a", "b"), label="power",
                                 purpose="workbench fit", maxfev=5000)
        elif model == "gaussian":
            i_pk = int(np.argmax(np.abs(z)))
            fit = fits.gaussian(x, z, (z[i_pk], x[i_pk], max((x[-1] - x[0]) / 4, 1e-3)), purpose="workbench fit")
        elif model == "boltzmann":
            fit = fits.boltzmann(x, z, purpose="workbench fit")
        elif model == "hill":
            fit = fits.logistic(x, z, purpose="workbench fit")
        elif model == "custom":
            expr = (custom or "").strip()
            if not expr:
                out["words"] = "type an expression in t with a, b, c, d"
                return out
            code = compile(expr, "<custom fit>", "eval")
            names = {"np": np, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
                     "tanh": np.tanh, "abs": np.abs, "pi": np.pi, "e": np.e}
            used = [p for p in ("a", "b", "c", "d") if re.search(rf"\b{p}\b", expr)]
            if not used:
                out["words"] = "the expression names none of a, b, c, d"
                return out

            def model_fn(tx, *params):
                env = dict(names, t=tx, **dict(zip(used, params)))
                return np.asarray(eval(code, {"__builtins__": {}}, env), float) * np.ones_like(tx)

            fit = fits.nonlinear(model_fn, x, z, [1.0] * len(used), tuple(used), label="custom",
                                 purpose="workbench fit", maxfev=5000)
        else:
            out["words"] = f"no model named {model}"
            return out
    except Exception as exc:                                           # a fit never raises into the screen
        out["words"] = f"the fit failed: {type(exc).__name__}: {exc}"
        return out
    if fit is not None:
        out.update({"ok": bool(fit.ok), "params": dict(fit.params), "ci": dict(getattr(fit, "ci", {}) or {}),
                    "r2": float(fit.r2)})
        if fit.ok:
            try:
                pred = np.asarray(fit.curve(x), float)
            except Exception:
                pred = None
        if not fit.ok:
            out["words"] = "the fit did not converge" + (f" ({'; '.join(fit.flags)})" if getattr(fit, "flags", None) else "")
    if pred is not None and pred.size == x.size:
        out["curve_t"] = t[a:b][ok]
        out["curve_y"] = pred + baseline
        out["residuals"] = z - pred
        out["words"] = ", ".join(f"{k} = {v:.4g}" + (f" [{out['ci'][k][0]:.3g}, {out['ci'][k][1]:.3g}]"
                                                      if k in out["ci"] and np.all(np.isfinite(out["ci"][k])) else "")
                                 for k, v in out["params"].items()) + f"; r2 {out['r2']:.3f}"
    return out


# ------------------------------------------------------------------ the table: every measure, per sweep

def _noise_of(t: np.ndarray, y: np.ndarray, base_win: tuple[float, float] | None) -> float:
    if base_win is None:
        return NAN
    a, b = np.searchsorted(t, base_win[:2])
    seg = y[a:b]
    seg = seg[np.isfinite(seg)]
    return kinetics.robust_noise(seg - np.median(seg)) if seg.size > 3 else NAN


def command_level(recording: Recording, sweep: int, at_s: float) -> float:
    cmd = recording.channel(Role.COMMAND)
    if cmd is None:
        return NAN
    c = recording.read(cmd, [sweep])[0]
    i = int(np.clip(round(at_s * recording.sample_rate_hz), 0, c.size - 1))
    return float(c[i])


def measure(recording: Recording, m: HandMeasurement, aux: dict[str, np.ndarray] | None = None) -> dict:
    """Everything the workbench shows for a hand measurement: the traces, the rows of the table (one per sweep
    and region window: the measures), the fit, and a summary over sweeps."""
    t = recording.time_s
    fs = recording.sample_rate_hz
    rows: list[dict] = []
    tr = traces(recording, m, aux)
    per_sweep = [(label, k, y) for label, k, y in tr if k is not None and not label.startswith("typical")]
    combined = [(label, k, y) for label, k, y in tr if k is None or label.startswith("typical")]
    base_region = _region_named(m.regions, m.options.baseline_region) or next(
        (r for r in m.regions if r.kind == "baseline"), None)
    measured = per_sweep if m.selection.mode == "each" else combined
    if m.selection.mode != "each" and not combined:
        measured = per_sweep
    overrides = {(o.sweep, o.region, o.what): o.time_s for o in m.overrides}
    excluded = {e.sweep: e.reason for e in m.exclusions}
    for label, k, y in measured:
        sweep_for_windows = k if k is not None else (m.selection.sweeps[0] if m.selection.sweeps else 0)
        base_wins = base_region.windows(recording, sweep_for_windows) if base_region is not None else []
        for r in m.regions:
            if r.kind == "baseline":
                continue
            for start_s, end_s, pulse in r.windows(recording, sweep_for_windows):
                bw = None
                if base_wins:
                    bw = next((w for w in base_wins if w[2] == pulse), base_wins[0]) if r.per_pulse else base_wins[0]
                base = _level_in(t, y, bw[0], bw[1]) if bw is not None else 0.0
                noise = _noise_of(t, y, bw)
                origin = start_s if not r.per_pulse else start_s - r.start_s
                ov = {w: tm for (sw, rg, w), tm in overrides.items() if sw == k and rg in ("", r.name)}
                stats = statistics(t, y, start_s, end_s, fs, baseline=base if np.isfinite(base) else 0.0,
                                   noise_sd=noise, options=m.options, origin_s=origin, override=ov)
                row = {"trace": label, "sweep": k, "region": r.name, "pulse": pulse or NAN,
                       "time_s": float(recording.sweep_start_times_s[k]) if k is not None and k < recording.sweep_start_times_s.size else NAN,
                       "command": command_level(recording, sweep_for_windows, start_s), "baseline": base,
                       "start_ms": start_s * 1e3, "end_ms": end_s * 1e3,
                       "excluded": excluded.get(k, "") if k is not None else ""}
                for key in m.measures:
                    row[key] = stats.get(key, NAN)
                rows.append(row)
    fit = None
    if m.fit_model:
        r = _region_named(m.regions, m.fit_region) or next((r for r in m.regions if r.kind == "fit"), None)
        if r is not None and measured:
            label, k, y = measured[-1] if m.selection.mode != "each" else measured[0]
            sweep_for_windows = k if k is not None else (m.selection.sweeps[0] if m.selection.sweeps else 0)
            wins = r.windows(recording, sweep_for_windows)
            if wins:
                base_wins = base_region.windows(recording, sweep_for_windows) if base_region is not None else []
                base = _level_in(t, y, base_wins[0][0], base_wins[0][1]) if base_wins else 0.0
                fit = fit_region(t, y, wins[0][0], wins[0][1], m.fit_model, baseline=base if np.isfinite(base) else 0.0,
                                 custom=m.fit_custom)
                fit["trace"] = label
                fit["region"] = r.name
    # the summary: each measure over the sweeps, region by region (a fit region's rows are not a measure region's)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for r in m.regions:
        if r.kind == "baseline":
            continue
        summary[r.name] = {}
        for key in m.measures:
            vals = np.array([row[key] for row in rows if row["region"] == r.name and not row["excluded"]], float)
            vals = vals[np.isfinite(vals)]
            summary[r.name][key] = {"mean": float(vals.mean()) if vals.size else NAN,
                                    "sd": float(vals.std(ddof=1)) if vals.size > 1 else NAN,
                                    "median": float(np.median(vals)) if vals.size else NAN, "n": int(vals.size)}
    return {"traces": tr, "rows": rows, "fit": fit, "summary": summary}


def plot_series(rows: list[dict], y_key: str, x_key: str, region: str = "") -> tuple[np.ndarray, np.ndarray, list]:
    """A measure against sweep, time, command or another measure, from the table's rows (one region)."""
    picked = [r for r in rows if (not region or r["region"] == region) and not r["excluded"]]
    if x_key == "sweep":
        x = np.array([(r["sweep"] + 1) if r["sweep"] is not None else NAN for r in picked], float)
    elif x_key == "time":
        x = np.array([r["time_s"] / 60.0 for r in picked], float)
    elif x_key == "command":
        x = np.array([r["command"] for r in picked], float)
    else:
        x = np.array([r.get(x_key, NAN) for r in picked], float)
    y = np.array([r.get(y_key, NAN) for r in picked], float)
    return x, y, picked


def units_of(measure: str, signal_units: str = "") -> str:
    """A measure's units on a channel recorded in ``signal_units`` ("mV·ms" for an area in mV)."""
    return MEASURE_UNITS.get(measure, "").replace("u", signal_units or "units")


def headings(measures: list[str], signal_units: str = "", command_units: str = "") -> dict[str, str]:
    """The table's column names: each measure by its name with its units, and the columns every row has."""
    out = {"trace": "Trace", "sweep": "Sweep", "region": "Region", "pulse": "Pulse",
           "time_s": "Sweep starts at (s)", "command": f"Command ({command_units})" if command_units else "Command",
           "baseline": f"Baseline ({signal_units})" if signal_units else "Baseline", "excluded": "Left out"}
    for key in measures:
        u = units_of(key, signal_units)
        out[key] = MEASURES.get(key, (key,))[0] + (f" ({u})" if u else "")
    return out


def summary_lines(m: HandMeasurement, result: dict, signal_units: str = "") -> list[str]:
    """The Summary tab's lines (HTML): each measure's mean, SD, median and n over the sweeps, with units."""
    lines = []
    for region, measures in result["summary"].items():
        for key, s in measures.items():
            name = f"<b>{region}: {MEASURES.get(key, (key,))[0]}</b>"
            u = units_of(key, signal_units)
            u = f" {u}" if u else ""
            if not s["n"]:
                lines.append(f"{name}: nothing measured (n = 0)")
                continue
            lines.append(f"{name}: mean {s['mean']:.4g}{u}"
                         + (f", SD {s['sd']:.3g}" if np.isfinite(s["sd"]) else "")
                         + (f", median {s['median']:.4g}" if np.isfinite(s["median"]) else "") + f" (n = {s['n']})")
    fit = result.get("fit")
    if fit and fit.get("ok"):
        lines.append(f"<b>Fit, {FIT_MODELS.get(m.fit_model, m.fit_model)}</b> on {fit.get('region', '')}: "
                     f"{fit['words']}")
    return lines


def stamp(m: HandMeasurement, version: str = "") -> HandMeasurement:
    m.made_at = dt.datetime.now().isoformat(timespec="seconds")
    m.app_version = version
    return m


def summary_rows(m: HandMeasurement, result: dict) -> list[dict]:
    """The lines Results shows for a hand measurement: each measure's mean, SD, median and n over the sweeps."""
    out = []
    for region, measures in result["summary"].items():
        for key, s in measures.items():
            out.append({"measure": f"{region}: {MEASURES.get(key, (key,))[0]}", "mean": s["mean"], "sd": s["sd"],
                        "median": s["median"], "n": s["n"], "measured": "by hand"})
    if result.get("fit") and result["fit"].get("ok"):
        for k, v in result["fit"]["params"].items():
            out.append({"measure": f"fit {FIT_MODELS.get(m.fit_model, m.fit_model)}: {k}", "mean": v, "sd": NAN,
                        "median": NAN, "n": result["fit"]["n"], "measured": "by hand (fit)"})
    return out
