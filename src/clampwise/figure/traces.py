"""Raw traces for a figure, picked by hand.

A figure's example panel is a list of named trace sets ("Before the step",
"After the step", "Control", "Drug"...). Each set is a list of sweeps (a file,
a sweep, and the stimulus pulse to align on; pulse 0 aligns on the sweep's
start) and is drawn as their average, as every sweep, or as the one sweep
closest to their mean. Every trace is cut from the same window around its
alignment point, less its own level before that point, with the stimulus
artifact blanked over one shared stretch, so the sets compare like with like.

For DSI the responses are labelled by where they sit around each step
("episode 2, baseline 3 of 4", "episode 2, 1st after the step"), so pre and
post sets are picked by role rather than by sweep number.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

MODES = {"average": "Average of the sweeps", "each": "Every sweep", "typical": "The sweep closest to their mean"}


@dataclass
class TraceSet:
    name: str
    mode: str = "average"
    items: list = field(default_factory=list)       # [file, sweep, pulse]: file a path or a name, pulse 0 = sweep start;
    #                                                 [file, sweep, 0, at_s]: aligned at at_s (an event's onset)


@dataclass
class Window:
    before_ms: float = 5.0
    after_ms: float = 60.0
    baseline: bool = True                           # each trace less its median before the alignment point
    blank_ms: float = 0.0                           # the stimulus artifact, blanked from 0 to this (0: none)

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def stimulus_s(recording, sweep: int, pulse: int) -> float:
    """When pulse ``pulse`` (from 1) of ``sweep`` came, in s from the sweep's start; 0 for pulse 0."""
    if pulse <= 0:
        return 0.0
    if not recording.stimulus.times_s:              # found as the analyses find it, and kept with the recording
        from ..analysis.stimulus import find_stimulus

        recording.stimulus = find_stimulus(recording)
    times = recording.stimulus.times_s
    if sweep < len(times) and pulse - 1 < len(times[sweep]):
        return float(times[sweep][pulse - 1])
    return np.nan


def cut(recording, sweep: int, at_s: float, w: Window) -> tuple[np.ndarray, np.ndarray]:
    """``w.before_ms`` before to ``w.after_ms`` after ``at_s`` in ``sweep``: time in ms from that point, the
    signal (less its median before it when asked), NaN over the artifact."""
    fs = recording.sample_rate_hz
    y = np.asarray(recording.sweep(sweep), float)
    i = int(round(at_s * fs))
    a, b = max(0, i - int(w.before_ms * 1e-3 * fs)), min(y.size, i + int(w.after_ms * 1e-3 * fs))
    t = (np.arange(a, b) - i) / fs * 1e3
    seg = y[a:b] - np.median(y[a:i]) if (w.baseline and i > a) else y[a:b].copy()
    if w.blank_ms > 0:
        seg = np.where((t >= 0) & (t <= w.blank_ms), np.nan, seg)
    return t, seg


def closest_to_mean(traces: list[np.ndarray]) -> int:
    """The index of the trace nearest the mean of them all (NaN ignored)."""
    m = np.array(traces, float)
    with warnings.catch_warnings():                 # a blanked stretch is NaN in every trace
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(m, axis=0)
        return int(np.nanargmin(np.nanmean((m - mean) ** 2, axis=1)))


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def thin(t: np.ndarray, y: np.ndarray, most: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """A long trace as each stretch's lowest and highest sample, in time order, so spikes and events keep their
    peaks when drawn; a short one as it is."""
    if t.size <= most:
        return t, y
    k = int(np.ceil(2 * t.size / most))
    n = t.size // k * k
    tb, yb = t[:n].reshape(-1, k), y[:n].reshape(-1, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanargmin(np.nan_to_num(yb, nan=np.inf), axis=1), np.nanargmax(np.nan_to_num(yb, nan=-np.inf),
                                                                                   axis=1)
    pick = np.sort(np.stack([lo, hi], axis=1), axis=1)
    rows = np.arange(len(tb))[:, None]
    return tb[rows, pick].ravel(), yb[rows, pick].ravel()


def kinetics(t: np.ndarray, y: np.ndarray, start_ms: float = 0.0) -> tuple[list[pd.DataFrame], list[str]]:
    """The 10-90% rise and a single-exponential decay measured on a drawn response (``t`` in ms): the rise's two
    points and the fitted curve as extra lines, and the caption's sentence of what was measured on it."""
    from ..analysis.tools import fits
    from ..analysis.tools.kinetics import decay_start, interp_crossing

    ok = np.isfinite(y) & (t > max(start_ms, 0.0))  # the peak: after the alignment point and any artifact
    if ok.sum() < 10:
        return [], []
    i0 = int(np.flatnonzero(ok)[0])
    seg = np.where(ok, y, np.nan)
    sign = 1.0 if np.nanmax(seg) > -np.nanmin(seg) else -1.0
    dev = np.nan_to_num(sign * y, nan=0.0)
    peak = i0 + int(np.argmax(dev[i0:]))
    if dev[peak] <= 0:
        return [], []

    def crossing(frac: float) -> float:              # the last crossing before the peak, from the trace's start
        below = np.flatnonzero(dev[:peak + 1] < frac * dev[peak])
        return interp_crossing(t, dev, int(below[-1]), int(below[-1]) + 1, frac * dev[peak]) \
            if below.size else np.nan

    out, said = [], []
    t10, t90 = crossing(0.1), crossing(0.9)
    if np.isfinite(t10) and np.isfinite(t90):
        out.append(pd.DataFrame({"t": [t10, t90], "y": [sign * 0.1 * dev[peak], sign * 0.9 * dev[peak]],
                                 "Trace": "10-90% rise", "Sweep": "rise"}))
        said.append(f"10-90% rise {t90 - t10:.2f} ms")
    fs = 1e3 / float(np.median(np.diff(t)))
    a = decay_start(dev, peak, t.size - 1, fs)
    fit = fits.exp_decay(t[a:] - t[a], dev[a:])
    if fit.ok:
        out.append(pd.DataFrame({"t": t[a:], "y": sign * fit.curve(t[a:] - t[a]), "Trace": "decay fit",
                                 "Sweep": "fit"}))
        said.append(f"decay τ {fit['tau']:.1f} ms (single exponential from {t[a]:.1f} ms, r² {fit.r2:.2f})")
    return out, ([f"Measured on the trace drawn: {', '.join(said)}."] if said else [])


def build(sets: list[TraceSet], load: Callable[[str], object], w: Window,
          overlay: str | dict | None = None) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """The table an example panel draws (Time, the signal, Trace, Sweep), the caption's notes and the artifact's
    marker. A sweep that cannot be cut (no such pulse) is left out and said. ``overlay`` draws over the first
    set: "kinetics" its rise and decay fit, {"name", "at_ms"} points marked on it (detected events)."""
    rows, notes, units, missing, first = [], [], "", 0, None
    for s in sets:
        cuts, names = [], []
        for file, sweep, pulse, *at_s in s.items:
            rec = load(file)
            at = (float(at_s[0]) if at_s else stimulus_s(rec, int(sweep), int(pulse))) if rec is not None else np.nan
            if rec is None or not np.isfinite(at) or not 0 <= int(sweep) < rec.n_sweeps:
                missing += 1
                continue
            units = units or rec.signal_units
            cuts.append(cut(rec, int(sweep), at, w))
            name = f"{(str(file).replace(chr(92), '/').rsplit('/', 1)[-1] + ' ') if file else ''}"                 f"sweep {int(sweep) + 1}" + (f", pulse {int(pulse)}" if int(pulse) > 1 else "") + \
                (f" at {float(at_s[0]):.3f} s" if at_s else "")
            names.append(name if name not in names else f"{name} ({len(names) + 1})")     # each its own line
        if not cuts:
            continue
        n = min(len(t) for t, _ in cuts)
        t = cuts[0][0][:n]
        stack = np.array([y[:n] for _, y in cuts])
        unit = "events" if all(len(i) > 3 for i in s.items) else "sweeps"     # aligned on events' onsets
        if s.mode == "each":
            for k, y in enumerate(stack):
                rows.append(pd.DataFrame(dict(zip("ty", thin(t, y)))).assign(Trace=s.name, Sweep=names[k]))
            notes.append(f"{s.name}: {len(cuts)} {unit}, each drawn.")
            continue
        if s.mode == "typical":
            k = closest_to_mean(list(stack))
            y, who = stack[k], names[k]
            notes.append(f"{s.name}: {names[k]}, the one of {len(cuts)} closest to their mean.")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y = np.nanmean(stack, axis=0)
            who = names[0] if len(cuts) == 1 else f"{len(cuts)} {unit} averaged"
            notes.append(f"{s.name}: " + (names[0] if len(cuts) == 1 else f"the average of {len(cuts)} {unit}") + ".")
        first = first or (t, y)
        rows.append(pd.DataFrame(dict(zip("ty", thin(t, y)))).assign(Trace=s.name, Sweep=who))
    if first and overlay == "kinetics":
        extra, said = kinetics(*first, w.blank_ms)
        rows += extra
        notes += said
    elif first and isinstance(overlay, dict):
        at = np.asarray(overlay.get("at_ms", []), float)
        at = at[(at >= first[0][0]) & (at <= first[0][-1])]
        rows.append(pd.DataFrame({"t": at, "y": np.interp(at, *first), "Trace": overlay.get("name", "marked"),
                                  "Sweep": "marks"}))
        notes.append(f"{overlay.get('name', 'marked').capitalize()}: {len(at)}, at {overlay.get('where', 'their times')}.")
    name = f"{'Current' if units.lower() in ('pa', 'na') else 'Voltage'} ({units or 'signal'})"
    frame = pd.concat(rows, ignore_index=True).rename(columns={"t": "Time (ms)", "y": name}) if rows else \
        pd.DataFrame(columns=["Time (ms)", name, "Trace", "Sweep"])
    if w.baseline:
        notes.append("Each trace is less its own level before the alignment point.")
    markers = []
    if w.blank_ms > 0:
        notes.append(f"The stimulus artifact (0 to {w.blank_ms:.1f} ms) is blanked and shaded.")
        markers.append({"at": 0.0, "end": w.blank_ms, "label": "artifact"})
    if missing:
        notes.append(f"{missing} picked sweep(s) could not be read (file not found, or no such stimulus pulse) "
                     "and are left out.")
    return frame, notes, markers


def source(sets: list[TraceSet], w: Window, table: str, overlay: str | dict | None = None) -> dict:
    """What a recipe keeps so the picker reopens with these picks (and what is drawn over them)."""
    return {"kind": "traces", "table": table, "sets": [asdict(s) for s in sets], "window": asdict(w),
            **({"overlay": overlay} if overlay else {})}


def from_source(src: dict) -> tuple[list[TraceSet], Window]:
    sets = [TraceSet(d.get("name", "Trace"), d.get("mode", "average"), [list(i) for i in d.get("items", [])])
            for d in src.get("sets", [])]
    return sets, Window.from_dict(src.get("window", {}))


# ---------------------------------------------------------------- DSI: responses by their place around the steps

def dsi_roles(result) -> list[dict]:
    """Every evoked response of a DSI result with where it sits around the steps: ``roles``, a list of (episode,
    position), position negative for a baseline response counted back from its step (-1 the last before it) and
    positive after it (1 the first). A response between two steps is both "after" one and "baseline" of the
    next."""
    out = []
    for r in result.tables.get("responses", []):
        out.append({"file": str(r.get("file", "")), "sweep": int(r["sweep"]), "pulse": int(r["pulse"]),
                    "time_s": float(r["time_s"]), "roles": [], "label": "",
                    "artifact_end_ms": float(r.get("artifact_end_ms", np.nan))})
    times = np.array([o["time_s"] for o in out])
    order = list(np.argsort(times))
    episodes = sorted(result.tables.get("episodes", []), key=lambda e: float(e["step_start_s"]))
    for n, e in enumerate(episodes):
        k = int(e["episode"])
        start = float(e["step_start_s"])
        end = start + float(e["step_duration_s"]) + float(e.get("settle_s", 0.0) or 0.0)
        prev = float(episodes[n - 1]["step_start_s"]) + float(episodes[n - 1]["step_duration_s"]) if n else -np.inf
        nxt = float(episodes[n + 1]["step_start_s"]) if n + 1 < len(episodes) else np.inf
        pre = [i for i in order if prev < times[i] < start]
        post = [i for i in order if end < times[i] < nxt]
        for j, i in enumerate(reversed(pre)):
            out[i]["roles"].append((k, -(j + 1)))
        for j, i in enumerate(post):
            out[i]["roles"].append((k, j + 1))
    for o in out:
        o["label"] = "; ".join(f"episode {k}, {_ordinal(abs(p))} {'before' if p < 0 else 'after'} the step"
                               for k, p in o["roles"])
    return out


def dsi_indices(roles: list[dict], episode: int, before: int = 0, after: int = 0) -> list[int]:
    """Which of ``roles`` are, in ``episode`` (0: every episode), among its last ``before`` baseline responses
    or its first ``after`` responses after the step."""
    def wanted(k, p):
        return (episode == 0 or k == episode) and bool((before and -before <= p < 0) or (after and 0 < p <= after))
    return [i for i, r in enumerate(roles) if any(wanted(k, p) for k, p in r["roles"])]


def dsi_pick(roles: list[dict], episode: int, before: int = 0, after: int = 0) -> list[list]:
    """The items (file, sweep, pulse) of ``dsi_indices``."""
    return [[roles[i]["file"], roles[i]["sweep"], roles[i]["pulse"]] for i in dsi_indices(roles, episode, before,
                                                                                         after)]


def dsi_blank(roles: list[dict]) -> float:
    """The artifact blank to start from: the median end of the responses' artifacts (the longest would hide the
    peak of most responses when the artifact's end drifts)."""
    art = np.array([r["artifact_end_ms"] for r in roles], float)
    return float(np.nanmedian(art)) if np.isfinite(art).any() else 0.0
