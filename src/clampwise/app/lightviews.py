"""The Figures screen's graphs for the lab's stimulus experiments (light through the pattern illuminator, or an
electrode): what a person looks at first after such a recording, drawn from the results and, for the traces, from
the recording itself.

- the sweeps around the stimulus, each against its own level before it, with their average and the light's bars;
- the test itself: the response windows against the sham windows they are judged by, with the threshold and the
  verdict in words (connected or not, at what false-positive rate);
- each pulse's success rate with its 95% interval beside what chance gives, and each success's latency;
- the cell's health sweep by sweep (the potential before the protocol, the input resistance);
- a cell's files side by side, pattern by pattern (the targeted cell, each region, the whole field);
- for any recording, its sweeps as recorded.

Each graph opens in the figure builder (its Edit button) as a figure made for it: example traces with the light's
bars and scale bars; a proportion with its interval and Fisher's exact test, drawn from the table of every window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from ..figure import layouts, traces
from ..figure.elements import wavelength_colour
from ..figure.recipe import Recipe, Spec, pack
from . import theme

MOST_SWEEPS = 60          # sweeps drawn in one graph
VERDICTS = {"yes": "Connected", "no": "Not connected", "direct photocurrent": "The recorded cell answers the light "
            "itself (a direct photocurrent): no connection can be read", "region unknown": "The region lit in each "
            "sweep is unknown", "too few windows": "Too few windows for a verdict", "not tested": "Not tested"}


def _view(key, title, units, draw, figure=None, group=""):
    from .panels import View

    return View(key, title, units, draw, figure=figure, group=group)


def _stimulus(rec):
    st = rec.stimulus
    if not st.times_s:
        from ..analysis.stimulus import find_stimulus

        rec.stimulus = st = find_stimulus(rec)
    return st


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(v))


def fmt_p(p: float) -> str:
    return "p n/a" if not _finite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def verdict_words(v: dict) -> str:
    """The connectivity verdict in a line: connected or not, how many windows, against what chance gives."""
    said = VERDICTS.get(str(v.get("connected", "")), str(v.get("connected", "")))
    if _finite(v.get("successes")) and _finite(v.get("n_windows")) and v.get("n_windows"):
        said += (f": {int(v['successes'])} of {int(v['n_windows'])} response windows beat the sham threshold "
                 f"(chance alone gives {v.get('chance_successes', float('nan')):.1f})")
    if _finite(v.get("p")):
        said += f"; the windows pooled against sham windows, {fmt_p(v['p'])}"
    if _finite(v.get("false_positive_rate")):
        said += f"; false-positive rate {v['false_positive_rate']:.0%}"
    return said + "."


# ---------------------------------------------------------------- the sweeps around the stimulus

def example(run) -> list:
    """Every sweep around its first stimulus pulse (each against its own level before it), their average, and
    the stimulus as bars: the first thing to look at."""
    rec = getattr(run, "recording", None)
    if rec is None or rec.n_sweeps == 0:
        return []
    st = _stimulus(rec)
    counts = [len(x) for x in st.times_s]
    if not any(counts):
        return []
    n_pulses = int(np.bincount([c for c in counts if c]).argmax())
    sweeps = [i for i, c in enumerate(counts) if c == n_pulses][:MOST_SWEEPS]
    ons = np.asarray(st.times_s[sweeps[0]][:n_pulses], float)
    widths = np.asarray(st.widths_s[sweeps[0]][:n_pulses], float) if sweeps[0] < len(st.widths_s) else np.zeros(n_pulses)
    light = st.kind == "light"
    what = "light" if light else "stimulus"
    after = float(min((ons[-1] - ons[0]) * 1e3 + 80.0, 800.0))
    w = traces.Window(20.0, after, True, 0.0)
    units = rec.signal_units
    conn = run.results.get("connectivity") if getattr(run, "results", None) else None

    def draw(item, pal) -> str:
        cuts = [traces.cut(rec, k, float(st.times_s[k][0]), w) for k in sweeps]
        n = min(len(t) for t, _y in cuts)
        t = cuts[0][0][:n]
        stack = np.vstack([y[:n] for _t, y in cuts])
        for y in stack:
            item.plot(t, y, pen=pg.mkPen(pal.sweep, width=0.8))
        if len(stack) > 1:
            item.plot(t, np.nanmean(stack, axis=0), pen=pg.mkPen(theme.series(pal)[0], width=2),
                      name=f"average of {len(stack)} sweeps")
        colour = pg.mkColor(wavelength_colour(traces.LIGHT_NM) if light else pal.text)
        for j, on in enumerate(ons):
            start = float((on - ons[0]) * 1e3)
            wide = float(widths[j] * 1e3) if j < widths.size and np.isfinite(widths[j]) else 0.0
            if wide > 0:
                tint = pg.mkColor(colour)
                tint.setAlpha(60)
                band = pg.LinearRegionItem([start, start + wide], movable=False, brush=pg.mkBrush(tint),
                                           pen=pg.mkPen(None))
                band.setZValue(-10)
                item.addItem(band)
            else:
                item.addItem(pg.InfiniteLine(start, angle=90, pen=pg.mkPen(colour, width=1, style=QtCore.Qt.DashLine)))
        item.setLabel("bottom", f"Time from the first {what} pulse (ms)")
        item.setLabel("left", units)
        wide = float(np.nanmedian(widths) * 1e3) if widths.size and np.isfinite(widths).any() else 0.0
        said = (f"{len(stack)} sweep{'s' if len(stack) != 1 else ''}, each against its own level before the {what}; "
                f"{'shaded' if wide > 0 else 'dashed'}: the {what}, {n_pulses} pulse{'s' if n_pulses != 1 else ''}"
                + (f" of {wide:g} ms" if wide > 0 else "") + ".")
        if conn is not None and conn.values.get("connected"):
            said += " " + verdict_words(conn.values)
        return said

    def figure(source: str = ""):
        path = str(getattr(rec, "source", "") or source)
        items = [[path, int(k), 1] for k in sweeps]
        sets = ([traces.TraceSet("sweeps", "each", items)] if len(items) > 1 else []) + [
            traces.TraceSet("average" if len(items) > 1 else "sweep", "average", items)]
        r = layouts.example_recipe(sets, w, lambda _f: rec, f"Sweeps around the {what}", title="")
        if len(items) > 1:
            r.series = {"sweeps": {"colour": "#9A9A9A", "width": 0.5, "legend": False}, "average": {"legend": False}}
        r.source["recording"] = source
        return Spec(panels=[r])

    return [_view("light.example", f"Sweeps around the {what}", f"[{units or '-'} vs. ms]", draw, figure,
                  "Example traces")]


def sweeps_view(run) -> list:
    """The recording's sweeps as recorded, whole: for any recording, and all there is to show of one whose kind
    Clampwise could not tell."""
    rec = getattr(run, "recording", None)
    if rec is None or rec.n_sweeps == 0:
        return []
    units = rec.signal_units
    sweeps = list(range(min(rec.n_sweeps, MOST_SWEEPS)))
    fs = rec.sample_rate_hz

    def draw(item, pal) -> str:
        for k in sweeps:
            y = np.asarray(rec.sweep(k), float)
            t = np.arange(y.size) / fs
            curve = item.plot(t, y, pen=pg.mkPen(pal.sweep if len(sweeps) > 1 else theme.series(pal)[0], width=0.8))
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
        item.setLabel("bottom", "Time (s)")
        item.setLabel("left", units)
        more = f" (the first {len(sweeps)} of {rec.n_sweeps})" if rec.n_sweeps > len(sweeps) else ""
        return f"{len(sweeps)} sweep{'s' if len(sweeps) != 1 else ''}{more}, as recorded."

    def figure(source: str = ""):
        path = str(getattr(rec, "source", "") or source)
        length = rec.n_samples / fs * 1e3
        items = [[path, int(k), 0] for k in sweeps[:20]]
        r = layouts.example_recipe([traces.TraceSet("sweeps", "each", items)], traces.Window(0.0, length, False, 0.0),
                                   lambda _f: rec, "The recording's sweeps", title="")
        r.series = {"sweeps": {"legend": False}}
        r.source["recording"] = source
        return Spec(panels=[r])

    return [_view("recording.sweeps", "The recording's sweeps", f"[{units or '-'} vs. s]", draw, figure, "Traces")]


# ---------------------------------------------------------------- connected or not: one recording

def connectivity(run, group: str = "Connected or not") -> list:
    res = run.results.get("connectivity") if getattr(run, "results", None) else None
    if res is None or getattr(res, "error", None) or not res.tables.get("windows"):
        return []
    return [v for v in (_against_sham(res, group), _success(res, group), _latency(res, group)) if v is not None]


def _against_sham(res, group: str):
    wins, shams, v = res.tables.get("windows") or [], res.tables.get("shams") or [], res.values
    thr = v.get("threshold_z")
    if not wins or not shams or not _finite(thr):
        return None

    def draw(item, pal) -> str:
        # departures over the noise are skewed (most sham windows near 1, a spike in a window in the hundreds):
        # drawn on a logarithmic axis, so the sham windows, the threshold and the responses are all told apart
        z_s = np.log10(np.clip(np.array([s["z"] for s in shams], float), 0.1, None))
        z_w = np.log10(np.clip(np.array([w["z"] for w in wins], float), 0.1, None))
        ok = np.array([bool(w["success"]) for w in wins])
        lo_z = float(min(z_s.min(), z_w.min(), np.log10(thr))) - 0.05
        hi_z = float(max(z_s.max(), z_w.max(), np.log10(thr))) + 0.05
        edges = np.linspace(lo_z, hi_z, 31)
        share = np.histogram(z_s, edges)[0] / max(z_s.size, 1) * 100.0
        item.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=share, brush=pg.mkBrush(pal.sweep), pen=None))
        row = float(share.max()) * 1.12
        lift = np.random.default_rng(0).uniform(-0.04, 0.04, z_w.size) * float(share.max())
        colour = theme.series(pal)[0]
        if ok.any():
            item.plot(z_w[ok], row + lift[ok], pen=None, symbol="o", symbolSize=9, symbolBrush=colour, symbolPen=None,
                      name="response windows: successes")
        if (~ok).any():
            item.plot(z_w[~ok], row + lift[~ok], pen=None, symbol="o", symbolSize=9,
                      symbolBrush=pg.mkBrush(pal.plot_background), symbolPen=pg.mkPen(colour, width=1.2),
                      name="response windows: failures")
        item.addItem(pg.InfiniteLine(float(np.log10(thr)), angle=90, label="sham threshold",
                                     pen=pg.mkPen(pal.text, width=1, style=QtCore.Qt.DashLine),
                                     labelOpts={"position": 0.22, "color": pal.text, "movable": False}))
        marks = [m for m in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                 if lo_z <= np.log10(m) <= hi_z]
        item.getAxis("bottom").setTicks([[(float(np.log10(m)), f"{m:g}") for m in marks]])
        item.setLabel("bottom", "Largest departure in the window (× baseline noise, log scale)")
        item.setLabel("left", "Sham windows (%)")
        if getattr(item, "legend", None) is not None:          # beside the bars, not over the windows' marks
            item.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 34))
        return (f"{verdict_words(v)} Bars: {z_s.size} sham windows in the same sweeps; "
                f"{v.get('false_positive_rate', 0.05):.0%} of them pass the threshold. Circles: the response "
                "windows (filled: successes).")

    def figure(source: str = ""):
        name = "Largest departure (× baseline noise)"
        frame = pd.concat([
            pd.DataFrame({"Window": "sham windows", "Sweep": [int(s["sweep"]) + 1 for s in shams],
                          name: [float(s["z"]) for s in shams]}),
            pd.DataFrame({"Window": "response windows", "Sweep": [int(w["sweep"]) + 1 for w in wins],
                          name: [float(w["z"]) for w in wins]})], ignore_index=True)
        r = Recipe(source={"kind": "graph", "table": "Response windows against sham windows", "recording": source},
                   data=pack(frame), question="distribution", plot="cumulative", y=name, group="Window", level="row",
                   x_log=True, ref_lines=[{"axis": "x", "at": float(thr), "label": "sham threshold"}],
                   notes=[verdict_words(v)])
        return Spec(panels=[r])

    return _view("connectivity.sham", "Response windows against sham windows", "[% vs. × noise]", draw, figure, group)


def _per_pulse(res) -> str:
    said = []
    for r in res.tables.get("pulses") or []:
        one = (f"pulse {int(r['pulse'])}: {int(r['successes'])} of {int(r['n_windows'])} windows "
               f"({r['success_rate']:.0%}, 95% interval {r['success_rate_low']:.0%} to {r['success_rate_high']:.0%})")
        if _finite(r.get("latency_median_ms")):
            one += f", latency {r['latency_median_ms']:.1f} ms"
        if _finite(r.get("jitter_ms")):
            one += f", jitter {r['jitter_ms']:.1f} ms"
        said.append(one)
    return "; ".join(said)


def _success(res, group: str):
    rows = res.tables.get("pulses") or []
    if not rows:
        return None
    alpha = res.values.get("false_positive_rate", 0.05)

    def draw(item, pal) -> str:
        x = np.array([float(r["pulse"]) for r in rows])
        rate = np.array([r["success_rate"] for r in rows], float) * 100
        lo = np.array([r["success_rate_low"] for r in rows], float) * 100
        hi = np.array([r["success_rate_high"] for r in rows], float) * 100
        colour = theme.series(pal)[0]
        item.addItem(pg.BarGraphItem(x=x, height=rate, width=0.55, brush=pg.mkBrush(colour), pen=None))
        item.addItem(pg.ErrorBarItem(x=x, y=rate, top=hi - rate, bottom=rate - lo, beam=0.18,
                                     pen=pg.mkPen(pal.text, width=1.2)))
        item.addItem(pg.InfiniteLine(float(alpha) * 100, angle=0, label="chance",
                                     pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DashLine),
                                     labelOpts={"position": 0.94, "color": pal.text_muted, "movable": False}))
        item.getAxis("bottom").setTicks([[(float(v), f"{int(v)}") for v in x]])
        item.setLabel("bottom", "Pulse")
        item.setLabel("left", "Successes (% of windows)")
        item.getViewBox().setRange(xRange=(float(x.min()) - 0.7, float(x.max()) + 0.7),
                                   yRange=(0, max(100.0 if hi.max() > 60 else 0.0, float(hi.max()) * 1.15)), padding=0)
        return _per_pulse(res) + ". Bars: Wilson's 95% interval; dashed: what chance alone gives."

    def figure(source: str = ""):
        units = res.values.get("units", "")
        frame = pd.DataFrame([{"Pulse": int(w["pulse"]), "Sweep": int(w["sweep"]) + 1,
                               "Outcome": "success" if w["success"] else "failure",
                               "Response / sham threshold": w.get("over_threshold", np.nan),
                               "Latency (ms)": w["latency_ms"], f"Amplitude ({units})" if units else "Amplitude":
                               w["amplitude"]} for w in res.tables["windows"]])
        frame["Pulse"] = "pulse " + frame["Pulse"].astype(str)
        r = Recipe(source={"kind": "graph", "table": "Every response window", "recording": source},
                   data=pack(frame), question="counts", plot="proportion", x="Outcome", group="Pulse", level="row",
                   event="success", y_label="Successes (% of windows)", x_label="",
                   ref_lines=[{"axis": "y", "at": float(alpha) * 100, "label": "chance"}],
                   notes=[f"A success: a response window whose largest departure beats the sham threshold ("
                          f"{float(alpha):.0%} of sham windows do)."])
        return Spec(panels=[r])

    return _view("connectivity.success", "Success rate of each pulse", "[% vs. pulse]", draw, figure, group)


def _latency(res, group: str):
    wins = [w for w in res.tables.get("windows") or [] if _finite(w.get("latency_ms"))]
    if not wins:
        return None

    def draw(item, pal) -> str:
        colour = theme.series(pal)[0]
        rng = np.random.default_rng(0)
        pulses = sorted({int(w["pulse"]) for w in wins})
        for p in pulses:
            y = np.array([w["latency_ms"] for w in wins if int(w["pulse"]) == p], float)
            item.plot(p + rng.uniform(-0.12, 0.12, y.size), y, pen=None, symbol="o", symbolSize=8, symbolBrush=colour,
                      symbolPen=None)
            item.plot([p - 0.25, p + 0.25], [float(np.median(y))] * 2, pen=pg.mkPen(pal.text, width=2))
        item.getAxis("bottom").setTicks([[(float(p), f"{p}") for p in pulses]])
        item.setLabel("bottom", "Pulse")
        item.setLabel("left", "Latency of each success (ms)")
        item.getViewBox().setXRange(min(pulses) - 0.7, max(pulses) + 0.7, padding=0)
        return _per_pulse(res) + ". Line: the median."

    def figure(source: str = ""):
        frame = pd.DataFrame([{"Pulse": f"pulse {int(w['pulse'])}", "Sweep": int(w["sweep"]) + 1,
                               "Latency (ms)": w["latency_ms"]} for w in wins])
        r = Recipe(source={"kind": "graph", "table": "Latency of each success", "recording": source},
                   data=pack(frame), question="compare", plot="points", y="Latency (ms)", group="Pulse", level="row",
                   summary="median_iqr", x_label="")
        return Spec(panels=[r])

    return _view("connectivity.latency", "Latency of each success, by pulse", "[ms vs. pulse]", draw, figure, group)


# ---------------------------------------------------------------- the cell's health, sweep by sweep

def health(run, group: str = "Cell health, sweep by sweep") -> list:
    res = run.results.get("sweep_health") if getattr(run, "results", None) else None
    rows = (res.tables.get("sweeps") if res is not None and not getattr(res, "error", None) else None) or []
    if not rows:
        return []
    from ..analysis.sweep_health import SETTINGS

    limits = {s.key: s.default for s in SETTINGS}
    words = str(res.values.get("words") or "")
    n_bad = sum(1 for r in rows if r.get("healthy") is False)

    def make(key: str, title: str, units: str, column: str, ylabel: str, bounds):
        if not any(_finite(r.get(column)) for r in rows):
            return None

        def draw(item, pal) -> str:
            x = np.array([r["sweep"] + 1 for r in rows], float)
            y = np.array([r.get(column) if _finite(r.get(column)) else np.nan for r in rows], float)
            good = np.array([r.get("healthy") is not False for r in rows])
            fine = np.isfinite(y)
            if (good & fine).any():
                item.plot(x[good & fine], y[good & fine], pen=None, symbol="o", symbolSize=8,
                          symbolBrush=theme.series(pal)[0], symbolPen=None, name="healthy sweeps")
            if (~good & fine).any():
                item.plot(x[~good & fine], y[~good & fine], pen=None, symbol="o", symbolSize=8, symbolBrush=pal.warn,
                          symbolPen=None, name="unhealthy sweeps")
            for at, text in bounds(y[fine]):
                item.addItem(pg.InfiniteLine(float(at), angle=0, label=text,
                                             pen=pg.mkPen(pal.warn, width=1, style=QtCore.Qt.DashLine),
                                             labelOpts={"position": 0.5, "color": pal.warn, "movable": False}))
            item.setLabel("bottom", "Sweep")
            item.setLabel("left", ylabel)
            said = (f"Unhealthy in {n_bad} of {len(rows)} sweeps: {words}." if n_bad else
                    f"Healthy in all {len(rows)} sweeps.")
            v = res.values
            if _finite(v.get("rin_median")):
                said += f" Input resistance {v['rin_median']:.0f} MOhm (median)"
                said += f", {v['rin_change']:+.0%} over the recording." if _finite(v.get("rin_change")) else "."
            return said

        return _view(key, title, units, draw, None, group)

    def rest(_y):
        return [(limits["rest_high_mv"], f"limit {limits['rest_high_mv']:g} mV"),
                (limits["rest_low_mv"], f"limit {limits['rest_low_mv']:g} mV")]

    def rin(y):
        if not y.size:
            return []
        first = float(np.median(y[:int(limits["first_sweeps"])]))
        tol = float(limits["max_rin_change"])
        return [(first * (1 + tol), f"+{tol:.0%}"), (first * (1 - tol), f"-{tol:.0%}")]

    made = [make("sweep_health.v_before", "Vm before each sweep's protocol", "[mV vs. sweep]",
                 "v_before", "Vm before the protocol (mV)", rest),
            make("sweep_health.rin", "Input resistance of each sweep", "[MOhm vs. sweep]", "rin_mohm",
                 "Input resistance (MOhm)", rin)]
    return [v for v in made if v is not None]


# ---------------------------------------------------------------- a cell's files, pattern by pattern

def cell(run, group: str = "The cell's files, pattern by pattern") -> list:
    res = run.results.get("connectivity_cell") if getattr(run, "results", None) else None
    files = (res.tables.get("files") if res is not None and not getattr(res, "error", None) else None) or []
    if len(files) < 2:
        return []
    wins = res.tables.get("windows") or []
    units = next((f.get("units") for f in files if f.get("units")), "")
    one = run.results.get("connectivity")
    alpha = one.values.get("false_positive_rate", 0.05) if one is not None else 0.05
    names = [f["pattern"] or f["file"] for f in files]

    def table() -> pd.DataFrame:
        amp = f"Amplitude ({units})" if units else "Amplitude"
        return pd.DataFrame([{"Pattern": w["pattern"], "File": w["file"], "Sweep": int(w["sweep"]) + 1,
                              "Pulse": int(w["pulse"]), "Outcome": w["outcome"],
                              "Response / sham threshold": w["over_threshold"], "Latency (ms)": w["latency_ms"],
                              amp: w["amplitude"]} for w in wins])

    def said() -> str:
        return "; ".join(x for x in (str(res.values.get("summary") or ""), str(res.values.get("comparison") or ""))
                         if x) + "."

    def ticks(item) -> None:
        item.getAxis("bottom").setTicks([[(float(i), n) for i, n in enumerate(names)]])
        item.setLabel("bottom", "Pattern lit")
        item.getViewBox().setXRange(-0.7, len(names) - 0.3, padding=0)

    def rates(item, pal) -> str:
        x = np.arange(len(files), dtype=float)
        rate = np.array([f["success_rate"] if _finite(f["success_rate"]) else np.nan for f in files], float) * 100
        lo = np.array([f["success_rate_low"] if _finite(f["success_rate_low"]) else np.nan for f in files], float) * 100
        hi = np.array([f["success_rate_high"] if _finite(f["success_rate_high"]) else np.nan for f in files], float) * 100
        yes = np.array([f["connected"] == "yes" for f in files])
        fine = np.isfinite(rate)
        for mask, brush in ((yes & fine, theme.series(pal)[0]), (~yes & fine, pal.sweep)):
            if mask.any():
                item.addItem(pg.BarGraphItem(x=x[mask], height=rate[mask], width=0.55, brush=pg.mkBrush(brush),
                                             pen=None))
        if fine.any():
            item.addItem(pg.ErrorBarItem(x=x[fine], y=rate[fine], top=(hi - rate)[fine], bottom=(rate - lo)[fine],
                                         beam=0.18, pen=pg.mkPen(pal.text, width=1.2)))
        item.addItem(pg.InfiniteLine(float(alpha) * 100, angle=0, label="chance",
                                     pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DashLine),
                                     labelOpts={"position": 0.94, "color": pal.text_muted, "movable": False}))
        ticks(item)
        item.setLabel("left", "Successes (% of windows)")
        top = float(np.nanmax(hi)) if fine.any() else 100.0
        item.getViewBox().setYRange(0, max(100.0 if top > 60 else 0.0, top * 1.15), padding=0)
        return said() + " In colour: connected; grey: not. Bars: Wilson's 95% interval; dashed: chance alone."

    def rates_figure(source: str = ""):
        if not wins:
            return None
        r = Recipe(source={"kind": "graph", "table": "Every response window of the cell's files", "recording": source},
                   data=pack(table()), question="counts", plot="proportion", x="Outcome", group="Pattern", level="row",
                   event="success", stats=True, pairs="all", y_label="Successes (% of windows)", x_label="",
                   ref_lines=[{"axis": "y", "at": float(alpha) * 100, "label": "chance"}],
                   notes=["A success: a response window whose largest departure beats its own file's sham "
                          f"threshold ({float(alpha):.0%} of sham windows do)."])
        return Spec(panels=[r])

    def dots(column: str, ylabel: str, line=None):
        def draw(item, pal) -> str:
            rng = np.random.default_rng(0)
            colour = theme.series(pal)[0]
            for i, n in enumerate(names):
                mine = [w for w in wins if w["pattern"] == n and _finite(w.get(column))]
                if not mine:
                    continue
                y = np.array([w[column] for w in mine], float)
                ok = np.array([w["outcome"] == "success" for w in mine])
                xs = i + rng.uniform(-0.16, 0.16, y.size)
                if ok.any():
                    item.plot(xs[ok], y[ok], pen=None, symbol="o", symbolSize=8, symbolBrush=colour, symbolPen=None)
                if (~ok).any():
                    item.plot(xs[~ok], y[~ok], pen=None, symbol="o", symbolSize=8,
                              symbolBrush=pg.mkBrush(pal.plot_background), symbolPen=pg.mkPen(colour, width=1.2))
            if line is not None:
                item.addItem(pg.InfiniteLine(line[0], angle=0, label=line[1],
                                             pen=pg.mkPen(pal.text, width=1, style=QtCore.Qt.DashLine),
                                             labelOpts={"position": 0.94, "color": pal.text, "movable": False}))
            ticks(item)
            item.setLabel("left", ylabel)
            return said() + " Filled: successes; open: failures."
        return draw

    def dots_figure(y: str, title: str, line=None, only_success: bool = False):
        def figure(source: str = ""):
            f = table()
            if only_success:
                f = f[f["Outcome"] == "success"]
            f = f[np.isfinite(pd.to_numeric(f[y], errors="coerce"))]
            if f.empty:
                return None
            r = Recipe(source={"kind": "graph", "table": title, "recording": source}, data=pack(f),
                       question="compare", plot="points", y=y, group="Pattern", level="row", summary="median_iqr",
                       x_label="", ref_lines=[{"axis": "y", "at": line[0], "label": line[1]}] if line else [])
            return Spec(panels=[r])
        return figure

    out = [_view("connectivity_cell.patterns", "Success rate of each pattern", "[% vs. pattern]", rates,
                 rates_figure, group)]
    if wins:
        thr = (1.0, "sham threshold")
        out.append(_view("connectivity_cell.response", "Every response window, by pattern", "[× threshold vs. pattern]",
                         dots("over_threshold", "Response / sham threshold", thr),
                         dots_figure("Response / sham threshold", "Every response window, by pattern", thr), group))
        if any(_finite(w.get("latency_ms")) for w in wins):
            out.append(_view("connectivity_cell.latency", "Latency of each success, by pattern", "[ms vs. pattern]",
                             dots("latency_ms", "Latency of each success (ms)"),
                             dots_figure("Latency (ms)", "Latency of each success, by pattern", None, True), group))
    return out
