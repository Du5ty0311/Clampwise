"""Paper-figure layouts and the example-trace picker (B12).

The picker: of a set of responses, the one closest to their mean (least
squares), so the example shown is typical rather than the best-looking; each
response is its own baseline taken off (the median before the stimulus), and a
stimulus artifact is blanked and marked, never drawn as a response.

The DSI layout, one click, double column: (A) example responses before the
step and just after it, scale bars; (B) the peak as % of its own baseline
against time from the step, every episode; (C) the suppression against the
step's duration.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import traces
from .recipe import Recipe, Spec, pack

BEFORE_MS, AFTER_MS = 5.0, 60.0
closest_to_mean = traces.closest_to_mean


def cut(recording, sweep: int, at_s: float, blank_ms: float = np.nan) -> tuple[np.ndarray, np.ndarray]:
    """``BEFORE_MS`` before to ``AFTER_MS`` after ``at_s`` in ``sweep`` (see ``traces.cut``)."""
    return traces.cut(recording, sweep, at_s, traces.Window(BEFORE_MS, AFTER_MS, True,
                                                            blank_ms if np.isfinite(blank_ms) else 0.0))


def _artifact_ms(results: dict) -> float:
    art = results.get("artifacts")
    spans = [r["end_ms"] - r["start_ms"] for r in (art.tables.get("artifacts", []) if art is not None else [])
             if r["kind"] == "stimulus artifact"]
    return float(max(spans)) if spans else 0.0


def typical_sets(recording, results: dict) -> tuple[list[traces.TraceSet], traces.Window] | None:
    """Every sweep's response to its first stimulus, drawn as the one closest to their mean."""
    sweeps = [k for k in range(recording.n_sweeps) if np.isfinite(traces.stimulus_s(recording, k, 1))]
    if not sweeps:
        return None
    return ([traces.TraceSet("Typical response", "typical", [["", k, 1] for k in sweeps])],
            traces.Window(BEFORE_MS, AFTER_MS, True, _artifact_ms(results)))


def typical_response(recording, results: dict) -> tuple[pd.DataFrame, list[str], list[dict]] | None:
    """Of every sweep's response to its first stimulus, the one closest to their
    mean: the table to draw, the caption's notes and the artifact's marker."""
    made = typical_sets(recording, results)
    if made is None:
        return None
    return traces.build(*made[:1], lambda _f: recording, made[1])


def dsi_sets(result, before: int = 4, after: int = 1, mode: str = "average") -> tuple[list[traces.TraceSet],
                                                                                 traces.Window]:
    """The DSI figure's examples: the last ``before`` responses before each step and the first ``after`` after it,
    every episode, averaged; the artifact blanked over the median of the responses' artifacts."""
    roles = traces.dsi_roles(result)
    return ([traces.TraceSet("before the step", mode, traces.dsi_pick(roles, 0, before=before)),
             traces.TraceSet("just after the step", mode, traces.dsi_pick(roles, 0, after=after))],
            traces.Window(BEFORE_MS, AFTER_MS, True, traces.dsi_blank(roles)))


# ---------------------------------------------------------------- the presets' trace rules
# Each takes the selection's results (analysis -> Result) and its recording, and gives the trace sets, their
# window, what is drawn over them and the caption's sentence of how they were chosen; or why it cannot.

Made = tuple[list[traces.TraceSet], traces.Window, object, list[str]]


def _rows(results: dict, key: str, table: str) -> pd.DataFrame:
    res = results.get(key)
    return pd.DataFrame(res.tables.get(table, []) if res is not None else [])


def _accepted(ev: pd.DataFrame) -> pd.DataFrame:
    return ev[ev["accepted"].fillna(True).astype(bool)] if "accepted" in ev else ev


def step_examples(results: dict, recording) -> Made | str:
    """The steps a paper shows of a family, each drawn whole with the command under it: the most hyperpolarizing
    step (the sag and the rebound), rheobase (the first full spike) and the step nearest twice rheobase (the
    firing pattern). A cell that fires on no step has no rheobase to show: its whole family is drawn."""
    steps = _rows(results, "intrinsic", "steps")
    if steps.empty or recording is None:
        return "Needs the intrinsic properties of a current-step recording."
    steps = steps.sort_values("current_pa").reset_index(drop=True)
    spikes = steps["full_spikes"] if "full_spikes" in steps else steps["spikes"]
    fired = list(steps.index[(spikes.fillna(0) > 0) & (steps["current_pa"] > 0)])
    long_ms = 1e3 * recording.sweep(0).size / recording.sample_rate_hz
    w = traces.Window(0.0, long_ms, False, 0.0)
    if not fired:
        items = [["", int(row.sweep), 0] for row in steps.itertuples()]
        lo, hi = float(steps["current_pa"].min()), float(steps["current_pa"].max())
        return [traces.TraceSet(f"{lo:+g} to {hi:+g} pA", "each", items)], w, None, \
            [f"The cell fired on no depolarizing step: all {len(items)} steps are drawn ({lo:+g} to {hi:+g} pA), "
             "each sweep as recorded."]
    rheo = fired[0]
    twice = int((steps["current_pa"] - 2 * steps["current_pa"].iloc[rheo]).abs().idxmin())
    if twice <= rheo:
        twice = min(rheo + 1, len(steps) - 1)
    picks = {0: "the most hyperpolarizing step", rheo: "rheobase (the smallest step with a full spike)"}
    picks.setdefault(twice, "the step nearest twice rheobase")
    sets = []
    for i, why in picks.items():
        row = steps.iloc[i]
        sets.append(traces.TraceSet(f"{row.current_pa:+g} pA" + (" (rheobase)" if why.startswith("rheobase") else ""),
                                    "average", [["", int(row.sweep), 0]]))
    return sets, w, None, [f"Steps drawn: {'; '.join(picks.values())}; each sweep as recorded."]


def stimulus_window(recording) -> tuple[list[int], traces.Window, int] | None:
    """The sweeps that hold the usual number of stimulus pulses, and the stretch around the first pulse that
    holds them all (20 ms before it to 80 ms after the last, 800 ms at most); None without a stimulus."""
    if recording is None or recording.n_sweeps == 0:
        return None
    counts = [int(np.isfinite(traces.stimulus_s(recording, k, 1))) and len(recording.stimulus.times_s[k])
              for k in range(recording.n_sweeps)]
    if not any(counts):
        return None
    n_pulses = int(np.bincount([c for c in counts if c]).argmax())
    sweeps = [k for k, c in enumerate(counts) if c == n_pulses]
    ons = np.asarray(recording.stimulus.times_s[sweeps[0]][:n_pulses], float)
    return sweeps, traces.Window(20.0, float(min((ons[-1] - ons[0]) * 1e3 + 80.0, 800.0)), True, 0.0), n_pulses


def pattern_examples(results: dict, recording) -> Made | str:
    """A cell's light (or stimulus) files side by side: each file's sweeps averaged around the first pulse, one
    trace per pattern lit (the targeted cell, each region, the whole field), in the files' order. One file alone:
    its sweeps and their average."""
    found = stimulus_window(recording)
    if found is None:
        return "Needs a recording with a stimulus."
    sweeps, w, _n = found
    cell = results.get("connectivity_cell")
    files = (cell.tables.get("files") if cell is not None else None) or []
    wins = (cell.tables.get("windows") if cell is not None else None) or []
    if len(files) >= 2 and wins:
        sets = []
        for f in files:
            mine = sorted({int(x["sweep"]) for x in wins if x["file"] == f["file"]})
            if mine:
                sets.append(traces.TraceSet(str(f["pattern"] or f["file"]), "average", [[f["file"], k, 1] for k in mine]))
        if sets:
            return sets, w, None, ["Each trace: the average of that file's sweeps around the first pulse, each sweep "
                                   "against its own level before it."]
    items = [["", int(k), 1] for k in sweeps]
    sets = ([traces.TraceSet("sweeps", "each", items)] if len(items) > 1 else []) + [
        traces.TraceSet("average" if len(items) > 1 else "sweep", "average", items)]
    return sets, w, None, ["Every sweep around the first pulse, each against its own level before it, and their "
                           "average."]


def evoked_average(results: dict, recording) -> Made | str:
    """Every sweep's response to its first stimulus, averaged, with its rise and decay measured on it."""
    if recording is None or "evoked" not in results:
        return "Needs the evoked responses of this recording."
    made = typical_sets(recording, results)
    if made is None:
        return "Needs a stimulus to align the responses on."
    sets, w = made
    sets[0].name, sets[0].mode = "averaged response", "average"
    nxt = [traces.stimulus_s(recording, k, 2) - traces.stimulus_s(recording, k, 1)
           for _f, k, _p in sets[0].items]
    nxt = [v for v in nxt if np.isfinite(v)]
    if nxt and 1e3 * min(nxt) - 1.0 < w.after_ms:   # the response to pulse 1 alone, its decay fitted before pulse 2
        w.after_ms = 1e3 * min(nxt) - 1.0
        return sets, w, "kinetics", ["Pulse 1 only: the window ends 1 ms before pulse 2."]
    return sets, w, "kinetics", []


def interval_examples(results: dict, recording) -> Made | str:
    """One averaged trace per interval between pulse 1 and pulse 2 (an interval series); the average of every
    sweep when the interval does not change."""
    sw = _rows(results, "evoked", "sweeps")
    if sw.empty or recording is None:
        return "Needs the evoked responses of this recording."
    on = sw.pivot_table(index="sweep", columns="pulse", values="onset_s", aggfunc="first")
    made = typical_sets(recording, results)
    if 2 not in on:
        return (made[0], made[1], None, []) if made else "Needs a stimulus to align the responses on."
    iv = ((on[2] - on[1]) * 1e3).round(0).dropna()
    last = float(((on.max(axis=1) - on[1]) * 1e3).max())
    sets = [traces.TraceSet(f"{v:g} ms", "average", [["", int(k), 1] for k in iv.index[iv == v]])
            for v in sorted(iv.unique())]
    return sets, traces.Window(BEFORE_MS, last + AFTER_MS, True, _artifact_ms(results)), None, \
        ["One trace per interval between pulse 1 and pulse 2, each the average of its sweeps."
         if len(sets) > 1 else "Every sweep averaged."]


STRETCH_S = 2.0


def event_stretch(results: dict, recording) -> Made | str:
    """A stretch of the recording with its detected events marked: of the 2 s stretches, the one whose count is
    closest to the mean count (typical, not the busiest)."""
    ev = _accepted(_rows(results, "events", "events"))
    if ev.empty or recording is None:
        return "Needs detected events (spontaneous or miniature)."
    fs = recording.sample_rate_hz
    spans = [(k, float(a)) for k in range(recording.n_sweeps)
             for a in np.arange(0.0, recording.sweep(k).size / fs - STRETCH_S + 1e-9, STRETCH_S)]
    if not spans:
        return "Needs a sweep at least 2 s long."
    counts = np.array([((ev.sweep == k) & (ev.peak_s >= a) & (ev.peak_s < a + STRETCH_S)).sum() for k, a in spans])
    k, a = spans[int(np.argmin(np.abs(counts - counts.mean())))]
    mine = ev[(ev.sweep == k) & (ev.peak_s >= a) & (ev.peak_s < a + STRETCH_S)]
    return [traces.TraceSet("recording", "average", [["", int(k), 0, a]])], \
        traces.Window(0.0, STRETCH_S * 1e3, False, 0.0), \
        {"name": "detected events", "at_ms": [float(v) for v in (mine.peak_s - a) * 1e3], "where": "their peaks"}, \
        [f"Sweep {k + 1}, {a:g} to {a + STRETCH_S:g} s: of the {STRETCH_S:g} s stretches, the one whose event count "
         f"({len(mine)}) is closest to the mean ({counts.mean():.1f})."]


def event_average(results: dict, recording) -> Made | str:
    """Every accepted event aligned on its onset and averaged, its rise and decay measured on the average."""
    ev = _accepted(_rows(results, "events", "events"))
    if ev.empty:
        return "Needs detected events (spontaneous or miniature)."
    return [traces.TraceSet("averaged event", "average", [["", int(r.sweep), 0, float(r.onset_s)]
                                                          for r in ev.itertuples()])], \
        traces.Window(5.0, 50.0, True, 0.0), "kinetics", ["Every accepted event, aligned on its onset."]


def dsi_examples(results: dict, recording) -> Made | str:
    res = results.get("dsi_cell") or results.get("dsi")
    if res is None or not res.tables.get("responses"):
        return "Needs a DSI analysis (a DSI recording, or a cell's step and response files)."
    sets, w = dsi_sets(res)
    return sets, w, None, ["Before the step: the last 4 baseline responses of every episode; after it: the first "
                           "response after every step."]


RULES = {"steps": step_examples, "evoked": evoked_average, "intervals": interval_examples,
         "event_stretch": event_stretch, "event_average": event_average, "dsi": dsi_examples,
         "patterns": pattern_examples}


def dress(r: Recipe, frame: pd.DataFrame, extras: dict, fresh: bool = True) -> Recipe:
    """A trace panel given what the field draws with example traces, read from the recordings: the stimulus as
    bars over the traces at its time, the potential each trace sat at (or was held at) written at its start, and
    the command under the traces when it changes within them (a step). ``fresh`` False keeps the bars and labels
    a person has already changed, and only follows the traces."""
    cmd = traces.command_of(frame)
    r.y2 = cmd
    steps = False
    if cmd:                                          # a command that moves within a trace is worth drawing
        for _, part in frame.groupby(["Trace", "Sweep"], sort=False):
            v = part[cmd].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size and float(v.max() - v.min()) > 0:
                steps = True
                break
    if fresh:
        r.stim_bars = [dict(bar) for bar in extras.get("stim_bars", [])]
        r.vm_labels = dict(extras.get("vm_labels", {}))
        r.command = {"show": steps}
    else:
        r.vm_labels = {**extras.get("vm_labels", {}), **{k: v for k, v in r.vm_labels.items()
                                                       if k in extras.get("vm_labels", {})}}
        if not cmd:
            r.command = {**r.command, "show": False}
    if r.stim_bars and not any("stimul" in n.lower() or "light" in n.lower() for n in r.notes):
        light = any(bar.get("nm") for bar in r.stim_bars)
        r.notes = list(r.notes) + [("Bars over the traces: the light." if light else
                                    "Ticks over the traces: the stimulus.")]
    return r


def example_recipe(sets, w, load, table: str, title: str = "Example responses", overlay=None) -> Recipe:
    """An example panel drawn from trace sets; its source keeps the picks (and what is drawn over them) so they
    can be changed."""
    frame, notes, markers = traces.build(sets, load, w, overlay)
    yname = frame.columns[1]
    r = Recipe(source=traces.source(sets, w, table, overlay), data=pack(frame), question="example",
               plot="trace",
               x="Time (ms)", y=yname, group="Trace", cell="Sweep", level="row", scale_bars=True, title=title,
               notes=notes, markers=markers)
    return dress(r, frame, traces.extras(sets, load, w))


def dsi_figure(result, load: Callable[[str], object], key: str = "dsi") -> Spec | None:
    """The DSI paper figure from a DSI result (``dsi`` or ``dsi_cell``): the DSI preset. ``load`` gives a
    response's recording from its file name ("" for the run's own file)."""
    from . import presets

    if not (result.tables.get("responses") and result.tables.get("episodes") and result.tables.get("course")):
        return None
    return presets.fill(presets.PRESETS["dsi"], presets.Selection(results={key: result}, load=load))
