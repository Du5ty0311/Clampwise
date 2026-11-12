"""Figure presets: the figures patch-clamp papers use, each filled from the
selected recording (or cell, or a table of cells) in one step.

A preset is data: its name, what it needs, and its panels. Each panel names
where its numbers come from (a result table of the selection, a table of
cells, a derived table, or a trace rule of ``layouts.RULES``), the question,
the plot and the columns by their result keys. One function, ``fill``, turns a
preset and a selection into an ordinary ``Spec``: every panel is a normal
``Recipe``, its table and its source kept, so it edits, undoes, saves and
reopens like any other figure. A panel that cannot be filled says what it
needs; the others are drawn.

Colours follow the lab's convention: before and after, control and drug, in
black and grey.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import data as fdata
from . import layouts
from .recipe import Recipe, Spec, pack

PAIR = ("#000000", "#999999")                     # before / after, control / drug
OVERLAY_LOOK = {"decay fit": {"colour": "#D55E00", "line": "--"},
                "10-90% rise": {"colour": "#D55E00", "line": "", "marker": "o"},
                "detected events": {"colour": "#D55E00", "line": "", "marker": "v"}}
OPS = {"==": operator.eq, "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}
SINGLE, ACROSS = "Single recording", "Across cells"


@dataclass(frozen=True)
class Panel:
    title: str
    source: str              # "analysis.table" ("a|b.table": the first present), "traces:<rule>", "derive:<name>",
    #                          or "cells" (the table of cells given)
    question: str = "relationship"
    plot: str = "line"
    x: str = ""              # result keys (or a derived table's own column names)
    y: str = ""
    group: str = ""
    cell: str = ""
    keep: tuple = ()         # (key, operator, value): the rows drawn
    long: tuple = ()         # ((key, name), ...): several columns drawn as groups of one, named in y
    clock: str = ""          # a time-in-sweep key: adds "Time (min)" on the recording's clock
    mean_line: bool = False  # a reference line at the mean of y
    note: str = ""           # the caption's sentence, filled from the analysis's values ({p1_jitter:.2f})
    look: dict = field(default_factory=dict)      # any other Recipe field
    least: tuple = ()        # (low, high, words): when every y is one value, the axis shows this range and the
    #                          caption says the words (a rate of 0 Hz on every step: 0 to 10 Hz)
    span: tuple = ()         # the panel's columns and rows in the figure's grid ((1, 1) unless given)
    letter: str = ""         # its letter when not the next of A, B, C ("-": none: it belongs to the panel before)


@dataclass(frozen=True)
class PresetDef:
    key: str
    name: str
    category: str
    line: str
    needs: tuple             # analyses, any one of which the selection must have; ("cells",) a table of cells
    panels: tuple
    columns: int = 2
    size: str = "double"
    unless: tuple = ()       # (analysis, value key, the value, why): the selection it is not for, and why
    height_mm: float = 0.0   # the page's height (0: as high as the rows need)
    row_ratios: tuple = ()   # the rows' shares of the height


@dataclass
class Selection:
    """What a preset is filled from."""
    results: dict = field(default_factory=dict)     # analysis -> Result: the recording's and its cell's
    load: Callable[[str], object] = lambda _f: None  # a file name or path -> Recording ("": the selected one)
    path: str = ""                                   # the selected recording, written into trace picks
    name: str = ""                                   # its name, in the panels' sources
    files: dict = field(default_factory=dict)        # file name -> path: a cell's other recordings
    results_of: Callable[[str], dict] = lambda _f: {}  # a file name -> that recording's results (a cell's files)
    table: pd.DataFrame | None = None                # across cells: the table of cells
    source: dict = field(default_factory=dict)       # where that table came from
    y: str = ""                                      # the measure compared (blank: the first headline measure)


NEEDS_WORDS = {"intrinsic": "a current-step recording analyzed for intrinsic properties",
               "connectivity": "a recording with a stimulus, tested for a connection (Connected or not)",
               "evoked": "an evoked-response recording", "events": "detected spontaneous or miniature events",
               "dsi": "a DSI recording", "dsi_cell": "a DSI cell", "cells": "a table of cells with groups"}

PRESETS = {p.key: p for p in (
    PresetDef("cc_steps", "Current-clamp step family", SINGLE,
              "Example steps, the f-I curve, V-I at the peak and steady state, and sag per step.", ("intrinsic",), (
                  Panel("Example traces", "traces:steps", "example", "trace"),
                  Panel("f-I curve", "intrinsic.fi", x="current_pa", y="rate_hz",
                        least=(0.0, 10.0, "The cell fired on no step: 0 Hz throughout."),
                        look={"y_from_zero": True},
                        note="Spikes on the rebound after hyperpolarizing steps ({rebound_spikes:g}) are not part "
                             "of the rate."),
                  Panel("V-I", "intrinsic.steps", x="current_pa", y="Membrane potential (mV)", group="Measure",
                        long=(("v_peak", "peak"), ("v_steady", "steady state")), keep=(("spikes", "==", 0),),
                        note="Steps without spikes."),
                  Panel("Sag per step", "intrinsic.steps", x="current_pa", y="sag_pct", keep=(("current_pa", "<", 0),),
                        note="Hyperpolarizing steps only."))),
    PresetDef("connectivity", "Light or stimulus responses: connected or not", SINGLE,
              "Example traces of each pattern lit with the light's bars, every response window against its sham "
              "threshold, the success rate with its 95% interval, the successes' latency and jitter, and the "
              "cell's health sweep by sweep.", ("connectivity",), (
                  Panel("", "traces:patterns", "example", "trace", look={"stack": True}, span=(2, 2)),
                  Panel("", "derive:windows", "compare", "points", y="Response / sham threshold", group="Pattern",
                        look={"summary": "median_iqr", "y_log": True, "x_label": " ", "x_rotate": 30,
                              "y_label": "Response / threshold", "row_unit": "response window",
                              "ref_lines": [{"axis": "y", "at": 1.0, "label": ""}]},
                        note="Each point: one response window, its largest departure from its own baseline over "
                             "its file's sham threshold (the 95% quantile of sham windows in the same sweeps; "
                             "dotted line); over 1 it is a success."),
                  Panel("", "derive:windows", "counts", "proportion", x="Outcome", group="Pattern",
                        look={"event": "success", "stats": True, "pairs": "all", "x_label": " ", "x_rotate": 30,
                              "y_label": "Successes (%)", "row_unit": "response window"}),
                  Panel("", "derive:latencies", "compare", "points", y="Latency (ms)", group="Pattern",
                        look={"summary": "mean_sd", "x_label": " ", "x_rotate": 30, "row_unit": "success"},
                        letter="-",
                        note="Latency of each success (its onset after the pulse); bars: mean ± SD, the SD being "
                             "the jitter."),
                  Panel("", "derive:health", plot="line", x="Sweep", y="Vm before the protocol (mV)", group="Pattern",
                        look={"ref_lines": [{"axis": "y", "at": -55.0, "label": "-55 mV"}], "legend": "none",
                              "y_label": "Vm before (mV)", "row_unit": "sweep"}, span=(3, 1),
                        note="The membrane potential before each sweep's protocol, in the colours of the traces; "
                             "a healthy cell rests below the dotted line.")),
              columns=5, height_mm=92.0, row_ratios=(1.25, 1.0)),
    PresetDef("evoked", "Evoked PSC", SINGLE,
              "The averaged response with its rise and decay, its amplitude over the run, and its latency.",
              ("evoked",), (
                  Panel("Averaged response", "traces:evoked", "example", "trace"),
                  Panel("Amplitude per sweep", "evoked.sweeps", plot="scatter", x="time_min", y="amplitude",
                        keep=(("pulse", "==", 1),), clock="onset_s", mean_line=True,
                        note="Pulse 1 of every sweep; the line is the run's mean."),
                  Panel("Latency", "evoked.sweeps", "distribution", "histogram", y="latency_ms",
                        keep=(("pulse", "==", 1),), note="Jitter (SD of the latency): {p1_jitter:.2f} ms.")),
              columns=3,
              unless=(("evoked", "condition", "the command under the stimulus",
                       "Its sweeps are a step family played under the stimulus, each at another level: there is "
                       "no one response to average. Use “Light or stimulus responses: connected or not”."),)),
    PresetDef("paired_pulses", "Paired pulses and trains", SINGLE,
              "Example traces at each interval, the paired-pulse ratio against interval, and Pn/P1 by pulse.",
              ("evoked",), (
                  Panel("Example traces", "traces:intervals", "example", "trace"),
                  Panel("Paired-pulse ratio", "derive:ppr", x="Interval (ms)", y="PPR (P2/P1)",
                        look={"ref_lines": [{"axis": "y", "at": 1.0, "label": ""}]}),
                  Panel("Pn / P1", "evoked.pulses", x="pulse", y="pn_p1",
                        look={"ref_lines": [{"axis": "y", "at": 1.0, "label": ""}]})),
              columns=3),
    PresetDef("events", "Spontaneous and miniature events", SINGLE,
              "A stretch with its events marked, the averaged event, cumulative amplitudes and intervals, and the "
              "frequency over time.", ("events",), (
                  Panel("Recording", "traces:event_stretch", "example", "trace"),
                  Panel("Averaged event", "traces:event_average", "example", "trace"),
                  Panel("Amplitude", "events.events", "distribution", "cumulative", y="amplitude",
                        keep=(("accepted", "==", True),)),
                  Panel("Inter-event interval", "events.events", "distribution", "cumulative", y="iei_ms",
                        keep=(("accepted", "==", True),)),
                  Panel("Frequency over time", "events.timecourse", x="time_min", y="frequency_hz",
                        clock="bin_start_s")),
              columns=3),
    PresetDef("dsi", "DSI", SINGLE,
              "Responses before and after the step, % of baseline over time from the step, and the suppression "
              "by step duration.", ("dsi", "dsi_cell"), (
                  Panel("Example responses", "traces:dsi", "example", "trace"),
                  Panel("Every episode", "derive:dsi_course", "time", "lines", x="Time from the step (s)",
                        y="Peak (% of baseline)", cell="Episode",
                        look={"unit": "episode", "markers": [{"at": 0.0, "end": None, "label": "step"}]}),
                  Panel("Suppression by step duration", "dsi_cell|dsi.episodes", plot="xy_error",
                        x="step_duration_s", y="suppression_pct", cell="episode", look={"unit": "episode"})),
              columns=3),
    PresetDef("groups", "Group comparison", ACROSS,
              "Every cell with the groups' mean ± SEM, and the difference between groups with its 95% CI.",
              ("cells",), (
                  Panel("Every cell", "cells", "compare", "points"),
                  Panel("Difference between groups", "cells", "compare", "estimation", look={"stats": True})),
              columns=2),
)}


# ---------------------------------------------------------------- derived tables (name -> frame, notes)

def _ppr(sel: Selection) -> tuple[pd.DataFrame | None, list[str]]:
    sw = layouts._rows(sel.results, "evoked", "sweeps")
    if sw.empty or not {1, 2} <= set(sw.pulse):
        return None, []
    amp = sw.pivot_table(index="sweep", columns="pulse", values="amplitude", aggfunc="first")
    on = sw.pivot_table(index="sweep", columns="pulse", values="onset_s", aggfunc="first")
    iv = ((on[2] - on[1]) * 1e3).round(0)
    rows = [{"Interval (ms)": v, "PPR (P2/P1)": amp.loc[iv == v, 2].mean() / amp.loc[iv == v, 1].mean(),
             "Sweeps": int((iv == v).sum())} for v in sorted(iv.dropna().unique())]
    return pd.DataFrame(rows), ["At each interval, the mean pulse 2 amplitude over the mean pulse 1 amplitude of "
                                "its sweeps (the ratio of the means)."]


def _dsi_course(sel: Selection) -> tuple[pd.DataFrame | None, list[str]]:
    course = layouts._rows(sel.results, "dsi_cell" if "dsi_cell" in sel.results else "dsi", "course")
    if course.empty:
        return None, []
    per = course.groupby("response")["from_step_s"].mean()
    return pd.DataFrame({"Episode": course["episode"].astype(str), "Time from the step (s)": course["response"].map(per),
                         "Peak (% of baseline)": course["pct"], "Step duration (s)": course["duration_s"]}), \
        ["Each response at its position's mean time from the step."]


def _p_words(p: float) -> str:
    return "p n/a" if not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def _windows(sel: Selection) -> tuple[pd.DataFrame | None, list[str]]:
    """Every response window of the cell's stimulus files (of the one file, without a cell), named by the
    pattern lit; and each file's verdict in words."""
    cell, one = sel.results.get("connectivity_cell"), sel.results.get("connectivity")
    rows = (cell.tables.get("windows") if cell is not None else None) or []
    files = (cell.tables.get("files") if cell is not None else None) or []
    if not rows and one is not None:
        name = str(one.values.get("pattern") or sel.name or "this recording")
        rows = [{"file": sel.name, "pattern": name, "sweep": w["sweep"], "pulse": w["pulse"],
                 "outcome": "success" if w["success"] else "failure", "over_threshold": w.get("over_threshold", np.nan),
                 "latency_ms": w["latency_ms"], "amplitude": w["amplitude"]} for w in one.tables.get("windows", [])]
        files = [{"pattern": name, **{k: one.values.get(k, np.nan) for k in ("connected", "p", "successes",
                                                                             "n_windows")}}]
    if not rows:
        return None, []
    units = next((f.get("units") for f in files if f.get("units")), "") or (one.values.get("units", "") if one else "")
    f = pd.DataFrame([{"Pattern": w["pattern"], "File": w["file"], "Sweep": int(w["sweep"]) + 1,
                       "Pulse": int(w["pulse"]), "Outcome": w["outcome"],
                       "Response / sham threshold": w["over_threshold"], "Latency (ms)": w["latency_ms"],
                       f"Amplitude ({units})" if units else "Amplitude": w["amplitude"]} for w in rows])
    words = {"yes": "connected", "no": "not connected"}
    said = "; ".join(f"{x['pattern'] or x.get('file', '')}: {words.get(str(x['connected']), str(x['connected']))}"
                     + (f" ({int(x['successes'])} of {int(x['n_windows'])} windows, {_p_words(float(x['p']))})"
                        if isinstance(x.get("p"), float) and np.isfinite(x["p"]) else "") for x in files)
    return f, [f"The windows pooled against sham windows in the same sweeps: {said}."]


def _latencies(sel: Selection) -> tuple[pd.DataFrame | None, list[str]]:
    f, _notes = _windows(sel)
    if f is None:
        return None, []
    f = f[(f["Outcome"] == "success") & np.isfinite(pd.to_numeric(f["Latency (ms)"], errors="coerce"))]
    return (f.reset_index(drop=True) if len(f) else None), []


def _health(sel: Selection) -> tuple[pd.DataFrame | None, list[str]]:
    """The potential before each sweep's protocol and each sweep's input resistance, for every stimulus file of
    the cell (the selected one alone, without a cell)."""
    cell = sel.results.get("connectivity_cell")
    files = (cell.tables.get("files") if cell is not None else None) or []
    todo = [(str(x["pattern"] or x["file"]), x["file"]) for x in files] or [(sel.name or "this recording", "")]
    rows, said = [], []
    for pattern, name in todo:
        results = sel.results if (not name or name == sel.name) else sel.results_of(name)
        h = results.get("sweep_health") if results else None
        if h is None:
            continue
        for s in h.tables.get("sweeps", []):
            rows.append({"Pattern": pattern, "Sweep": int(s["sweep"]) + 1,
                         "Vm before the protocol (mV)": s.get("v_before", np.nan),
                         "Input resistance (MOhm)": s.get("rin_mohm", np.nan),
                         "Healthy": "yes" if s.get("healthy") is not False else "no"})
        n, bad = h.values.get("n_sweeps"), h.values.get("unhealthy_sweeps")
        if isinstance(n, (int, float)) and isinstance(bad, (int, float)) and n:
            said.append(f"{pattern}: {int(bad)} of {int(n)} sweeps unhealthy"
                        + (f" ({h.values['words']})" if bad and h.values.get("words") else ""))
    return (pd.DataFrame(rows) if rows else None), ([f"{'; '.join(said)}."] if said else [])


DERIVED_WORDS = {"windows": "every response window", "latencies": "the latency of each success",
                 "health": "the cell's health sweep by sweep"}
DERIVED = {"ppr": _ppr, "dsi_course": _dsi_course, "windows": _windows, "latencies": _latencies, "health": _health}


# ---------------------------------------------------------------- filling a preset

def needs(preset: PresetDef, sel: Selection) -> str:
    """Why ``preset`` does not apply to ``sel`` ("" when it does)."""
    if preset.needs == ("cells",):
        t = sel.table
        ok = t is not None and fdata.guess(t)["group"] and fdata.numeric_columns(t)
        return "" if ok else "Needs " + NEEDS_WORDS["cells"] + "."
    if any(k in sel.results for k in preset.needs):
        for key, value, wrong, why in preset.unless:
            if key in sel.results and sel.results[key].values.get(value) == wrong:
                return why
        return ""
    return "Needs " + " or ".join(NEEDS_WORDS.get(k, k) for k in preset.needs) + "."


def _label(key: str) -> str:
    from ..analysis import registry

    a = registry.load().get(key)
    return a.label if a else key


def _named(key: str, col: str, units: str) -> str:
    from ..analysis import registry

    if col == "time_min":
        return "Time (min)"
    o = registry.column(key, col)
    return fdata._named(o.title if o else col.replace("_", " ").capitalize(), o.units if o else "", units)


def _run_table(p: Panel, sel: Selection) -> tuple[pd.DataFrame | None, dict, str, dict]:
    """A result table of the selection, its rows kept, its columns named as the workbook names them: the frame,
    its source, the analysis it came from, and the result keys' new names."""
    keys, table = p.source.split(".")
    key = next((k for k in keys.split("|") if k in sel.results and sel.results[k].tables.get(table)), "")
    if not key:
        return None, {}, keys.split("|")[0], {}
    f = pd.DataFrame(sel.results[key].tables[table])
    for col, op, v in p.keep:
        if col in f:
            f = f[OPS[op](f[col].fillna(v if op == "==" else np.nan), v)]
    rec = sel.load("")
    if p.clock and rec is not None and "sweep" in f:
        starts = np.asarray(rec.sweep_start_times_s, float)
        f = f.assign(time_min=(starts[f["sweep"].astype(int).to_numpy()] + f[p.clock].to_numpy(float)) / 60.0)
    units = getattr(rec, "signal_units", "") if rec is not None else ""
    names = {c: _named(key, c, units) for c in f.columns}
    f = f.rename(columns=names).reset_index(drop=True)
    src = {"kind": "project", "table": f"{_label(key)}: {table}", **({"recording": sel.name} if sel.name else {})}
    return f, src, key, names


def _empty(p: Panel, why: str) -> Recipe:
    return Recipe(source={"kind": "preset", "needs": why}, question=p.question, plot=p.plot, title=p.title)


def _traces(p: Panel, sel: Selection) -> Recipe:
    made = layouts.RULES[p.source.split(":", 1)[1]](sel.results, sel.load(""))
    if isinstance(made, str):
        return _empty(p, made)
    sets, w, overlay, notes = made
    for s in sets:                                   # "" (the selection's own file) and file names made paths
        s.items = [[sel.files.get(f, f) if f else sel.path, *rest] for f, *rest in s.items] if sel.path else s.items
    r = layouts.example_recipe(sets, w, sel.load, f"{sel.name or 'Recording'}: {(p.title or 'example traces').lower()}",
                               p.title, overlay)
    if r.frame() is None or r.frame().empty:
        return _empty(p, "None of its sweeps could be cut around the chosen point.")
    r.notes = notes + r.notes
    r.series = {k: v for k, v in OVERLAY_LOOK.items() if k in set(r.frame()["Trace"])}
    for k, v in p.look.items():
        setattr(r, k, v)
    return r


def _cells(p: Panel, sel: Selection) -> Recipe:
    from ..analysis import registry

    f = sel.table
    if f is None:
        return _empty(p, "Needs " + NEEDS_WORDS["cells"] + ".")
    g = fdata.guess(f)
    nums = [c for c in fdata.numeric_columns(f) if c not in fdata.IDENTITY and c not in (g["cell"], g["animal"])]
    heads = [o.title for a in registry.load().values() for o in a.outputs if o.headline]
    y = sel.y if sel.y in f else next((c for c in nums if any(str(c).split(" (")[0] == h for h in heads)),
                                      nums[0] if nums else "")
    if not (g["group"] and y):
        return _empty(p, "Needs " + NEEDS_WORDS["cells"] + ".")
    animal = g["animal"] if g["animal"] and f[g["animal"]].notna().any() else ""
    plot = "superplot" if p.plot == "estimation" and animal and g["cell"] and f[g["cell"]].duplicated().any() \
        else p.plot
    return Recipe(source=dict(sel.source), data=pack(f), question=p.question, plot=plot, y=y, group=g["group"],
                  cell=g["cell"], animal=animal, level="cell" if g["cell"] else "row",
                  title="Cells and their rows" if plot != p.plot else p.title,
                  **{k: v for k, v in p.look.items()})


def panel(p: Panel, sel: Selection) -> Recipe:
    """One panel of a preset, filled from ``sel`` (or saying what it needs)."""
    if p.source.startswith("traces:"):
        return _traces(p, sel)
    if p.source == "cells":
        r = _cells(p, sel)
    else:
        if p.source.startswith("derive:"):
            f, notes = DERIVED[p.source.split(":", 1)[1]](sel)
            key, names = "", {}
            what = p.title or DERIVED_WORDS.get(p.source.split(":", 1)[1], "a derived table")
            src = {"kind": "project", "table": f"{sel.name or 'Recording'}: {what.lower()}"}
        else:
            f, src, key, names = _run_table(p, sel)
            notes = []
        if f is None or f.empty:
            return _empty(p, "Needs " + NEEDS_WORDS.get(key, f"the {key} analysis") + "." if key else
                          "Its table could not be made from this recording.")
        x, y, group, cell = (names.get(v, v) for v in (p.x, p.y, p.group, p.cell))
        if p.long:
            cols = [names.get(k, k) for k, _ in p.long]
            f = fdata.to_long(f, cols, group or "Measure", y, [n for _, n in p.long])
        values = sel.results[key].values if key in sel.results else {}
        try:
            notes = notes + ([p.note.format(**values)] if p.note else [])
        except (KeyError, ValueError, TypeError):
            notes = notes + ([p.note.split("{")[0].rstrip(" :(") + " not measured."] if "{" in p.note else [])
        r = Recipe(source=src, data=pack(f), question=p.question, plot=p.plot, x=x, y=y, group=group, cell=cell,
                   level="cell" if cell else "row", title=p.title, notes=notes, **dict(p.look))
        if p.mean_line and y in f:
            r.ref_lines = r.ref_lines + [{"axis": "y", "at": float(f[y].mean()), "label": "mean"}]
    frame = r.frame()
    groups = list(dict.fromkeys(frame[r.group].dropna().astype(str))) if r.group and frame is not None else []
    if len(groups) == 2:
        r.series = {**{g: {"colour": c} for g, c in zip(groups, PAIR)}, **r.series}
    if len(p.least) == 3 and frame is not None and r.y in frame:
        v = pd.to_numeric(frame[r.y], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        if not v.size or (v.min() >= p.least[0] and v.max() < p.least[1] and v.max() - v.min() == 0):
            r.y_range = [float(p.least[0]), float(p.least[1])]      # nothing to scale the axis by: a plain one
            r.notes = list(r.notes) + [p.least[2]]
    return r


def fill(preset: PresetDef, sel: Selection) -> Spec:
    """``preset`` filled from ``sel``: an ordinary figure, every panel editable."""
    panels = [panel(p, sel) for p in preset.panels]
    for p, r in zip(preset.panels, panels):
        r.span = list(p.span) if p.span else r.span
        r.letter = p.letter or r.letter
    for r in panels:                                 # trace panels with two sets: black and grey
        f = r.frame()
        if r.question == "example" and f is not None:
            sets = [t for t in dict.fromkeys(f["Trace"]) if t not in OVERLAY_LOOK]
            if len(sets) == 2:
                r.series = {**{g: {"colour": c} for g, c in zip(sets, PAIR)}, **r.series}
    return Spec(panels, columns=min(preset.columns, len(panels)), size=preset.size,
                height_mm=float(preset.height_mm), row_ratios=list(preset.row_ratios))
