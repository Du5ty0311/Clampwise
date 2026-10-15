"""The baseline of every sweep: its level, its noise, and how it moves across
the recording. Every other analysis reads these; the quality checks (B10)
judge them.

Method: the pre-protocol stretch of each sweep (from the start of the sweep
to the first stimulus or the first command change, whichever comes first; a
gap-free file is cut into pieces) gives a level (the median, so one event in
the stretch cannot pull it) and a noise SD (after removing a straight line,
so slow drift is not counted as noise). Drift across sweeps is the span of
the levels (the 2.5th to 97.5th percentile when there are 40 sweeps or more,
so a few sweeps with an event in their baseline do not set it); drift within
a recording is the slope of the levels against time.
"""

from __future__ import annotations

import weakref

import numpy as np

from ..core.recording import Recording
from .registry import Output, Setting, TYPES, register
from .result import Result
from .tools import baselines, ledger

PIECE_S = 10.0          # a gap-free file is judged in pieces of this length


def first_command_change_s(recording: Recording, sweep: int) -> float:
    """When the command first moves in ``sweep``: the first sample that differs
    from the sweep's first command sample by more than the command's own
    noise (a recorded command jitters by a pA every sample) and by at least
    half a unit. Clampex's epoch table is not used for this: it lists epochs
    that sit at the holding level before the real step. Without a command,
    the first epoch whose level differs from the first epoch's; else the end
    of the sweep."""
    cmd = recording.command
    if cmd is not None:
        c = cmd[sweep]
        noise = float(np.std(np.diff(c[: max(int(0.01 * recording.sample_rate_hz), 2)]))) if c.size > 2 else 0.0
        tol = max(0.5, 5.0 * noise)
        moved = np.flatnonzero(np.abs(c - c[0]) > tol)
        if moved.size:
            return float(moved[0] / recording.sample_rate_hz)
        return recording.sweep_duration_s
    if sweep < len(recording.epochs) and recording.epochs[sweep]:
        first = recording.epochs[sweep][0].level
        for e in recording.epochs[sweep][1:]:
            if np.isfinite(e.level) and abs(e.level - first) > 0.5:
                return e.start_s
    return recording.sweep_duration_s


_EARLIEST: dict[int, tuple[weakref.ref, float]] = {}


def earliest_command_change_s(recording: Recording) -> float:
    """The family's earliest command change over every sweep, worked out once per
    recording: asked for in every sweep of a 3,699-sweep seal test it took 340 s
    (13.7 million sweep reads; the lab corpus, 2026-10-09)."""
    hit = _EARLIEST.get(id(recording))
    if hit is not None and hit[0]() is recording:
        return hit[1]
    value = min(first_command_change_s(recording, s) for s in range(recording.n_sweeps))
    _EARLIEST[id(recording)] = (weakref.ref(recording), value)
    return value


def pre_protocol_end_s(recording: Recording, stimulus, sweep: int, floor_s: float = 0.002,
                       max_baseline_s: float = 0.1) -> float:
    """Where the baseline stretch of ``sweep`` ends: the first stimulus or the
    first command change, whichever is earlier; never shorter than ``floor_s``.
    When nothing happens in the sweep (no stimulus, a flat command) the
    baseline is the start of the sweep, not its end."""
    # The family's earliest command change, the same in every sweep: a step of
    # 0 pA has no change of its own, and one window for every sweep is what
    # makes the per-sweep levels comparable as a drift measure.
    end = earliest_command_change_s(recording)
    if stimulus is not None and sweep < len(stimulus.times_s) and stimulus.times_s[sweep].size:
        end = min(end, float(stimulus.times_s[sweep][0]))
    if end >= recording.sweep_duration_s:
        end = min(max_baseline_s, recording.sweep_duration_s)
    return max(end, floor_s)


@register("baseline", "Baseline of each sweep", version=1, types=tuple(t for t in TYPES if t != "spike_times"),
          description="The level and noise of each sweep's pre-protocol baseline, and the drift of that level "
                      "across the recording. The median is used so that a single event cannot set the level.",
          source="Gouwens et al. 2019",
          settings=(
              Setting("max_baseline_s", 0.1, "Baseline length (s)", "Up to this much of each sweep before the "
                      "protocol is used; less when a stimulus or command change occurs sooner.", "s",
                      "Gouwens et al. 2019", minimum=0.001),
              Setting("piece_s", PIECE_S, "Gap-free piece (s)", "Gap-free recordings are assessed in segments of this "
                      "length, each treated as one sweep.", "s", advanced=True, minimum=1.0),
          ),
          outputs=(
              Output("level", "The baseline level: the median of the 100 ms baseline interval before the first "
                     "stimulus or command change of each sweep, then the median across sweeps (resting potential in "
                     "current clamp, holding current in voltage clamp).", "signal units",
                     "Gouwens et al. 2019", label="Baseline level", headline=True),
              Output("noise_sd", "The baseline noise: SD of the baseline stretch after removing a straight line, median "
                     "across sweeps.", "signal units", "Clampwise", label="Baseline noise (SD)"),
              Output("drift_range", "How far the baseline level moves across sweeps: the span of the per-sweep levels "
                     "(2.5th-97.5th percentile with 40 sweeps or more, else the full range).", "signal units",
                     "Clampwise", label="Baseline drift across sweeps"),
              Output("drift_trend", "The steady drift: the median of the last three sweeps' levels minus the first "
                     "three.", "signal units", "Clampwise", label="Steady drift (last 3 sweeps minus first 3)"),
              Output("baseline_s", "How long the baseline stretch is, per sweep (median).", "s", "Clampwise"),
              Output("sweep_level", "The baseline level of one sweep.", "signal units", "Clampwise", "sweep"),
              Output("sweep_noise_sd", "The baseline noise of one sweep.", "signal units", "Clampwise", "sweep"),
          ))
def run(recording: Recording, context, settings) -> Result:
    out = Result("baseline", 1)
    fs = recording.sample_rate_hz
    t = recording.time_s
    rows = []
    if recording.is_gap_free:
        y = recording.sweep(0)
        piece = max(int(settings["piece_s"] * fs), 1)
        starts = range(0, y.size - piece + 1, piece) if y.size >= piece else [0]
        for k, a in enumerate(starts):
            seg = y[a:a + piece]
            level = float(np.median(seg))
            x = np.arange(seg.size)
            resid = seg - np.polyval(np.polyfit(x, seg, 1), x) if seg.size > 3 else seg - level
            rows.append({"sweep": k, "sweep_level": level, "sweep_noise_sd": float(np.std(resid)),
                         "baseline_s": seg.size / fs, "time_s": a / fs})
        ledger.note(f"baseline: median of each {settings['piece_s']:g} s piece, noise SD after removing a line",
                    "baseline of a gap-free file")
    else:
        for s in range(recording.n_sweeps):
            end = pre_protocol_end_s(recording, context.stimulus, s, max_baseline_s=settings["max_baseline_s"])
            # the stretch immediately before the protocol (IPFX's baseline interval), not the sweep's first samples
            w = baselines.Window("pre-protocol", max(end - settings["max_baseline_s"], 0.0), end)
            seg = recording.sweep(s)[w.indices(t)]
            if seg.size == 0:
                continue
            level = float(np.median(seg))
            x = np.arange(seg.size)
            resid = seg - np.polyval(np.polyfit(x, seg, 1), x) if seg.size > 3 else seg - level
            rows.append({"sweep": s, "sweep_level": level, "sweep_noise_sd": float(np.std(resid)),
                         "baseline_s": w.end_s - w.start_s, "time_s": float(recording.sweep_start_times_s[s])})
        ledger.note(f"baseline: median of the {settings['max_baseline_s'] * 1e3:g} ms before the first stimulus or "
                    "command change of each sweep, noise SD after removing a line", "baseline per sweep")
    if not rows:
        out.notes.append("no sweep has a baseline stretch before its protocol")
        return out
    levels = np.array([r["sweep_level"] for r in rows])
    noise = np.array([r["sweep_noise_sd"] for r in rows])
    if levels.size >= 40:
        lo, hi = np.percentile(levels, [2.5, 97.5])
        span_how = "2.5th-97.5th percentile of the sweep levels"
    else:
        lo, hi = levels.min(), levels.max()
        span_how = "range of the sweep levels"
    k = min(3, levels.size)
    trend = float(np.median(levels[-k:]) - np.median(levels[:k])) if levels.size >= 2 else 0.0
    out.values.update({"level": float(np.median(levels)), "noise_sd": float(np.median(noise)),
                       "drift_range": float(hi - lo), "drift_trend": trend,
                       "baseline_s": float(np.median([r["baseline_s"] for r in rows]))})
    out.tables["sweeps"] = rows
    out.choose("drift measure", span_how, f"{levels.size} sweep(s)" + (": the full range would be set by a few sweeps "
               "with an event in their baseline" if levels.size >= 40 else ""))
    return out
