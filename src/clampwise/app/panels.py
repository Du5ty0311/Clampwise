"""One plot panel for every graph (B11, NeuroExpress's look): a title with the
units in it ("Resistance vs. current [MOhm vs. pA]"), a menu choosing what it
shows, a marker on every measured point, a light grid, and a one-line readout
under it.

What a panel can show comes from the results, nothing written per measure:
every numeric column of every result table against that table's x (current for
a step family, time for events and spikes), columns that share a name and
units on one graph (R peak and R steady, tau on and tau off), histogram tables
as bars, and the analyses' drawables in the same terms (the lines to 0 pA on
the resistance, the V-I lines, the averaged event).

Heat maps (``analysis/heatmaps.py``) draw through one path, ``draw_heatmap``:
the matrix with its axes in real units, a colour bar saying what the colour
means and its unit, the markers, and the scale rules; each panel's Colour
menu chooses the map, the limits, dB, and keeps a map's limits for every
recording compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..analysis import heatmaps as hmaps
from ..analysis import registry
from . import theme
from .widgets import box, label

UNITS = ((r"_pa$", "pA"), (r"_mv$", "mV"), (r"_ms$", "ms"), (r"_s$", "s"), (r"_hz$", "Hz"), (r"_pct$", "%"),
         (r"_pf$", "pF"), (r"^r_", "MOhm"), (r"^v_", "mV"), (r"^spikes$|^count$|^bursts$", "count"),
         (r"^amplitude|^baseline_level$|^sweep_level$|^sweep_noise_sd$", "signal units"), (r"^charge$", "charge units"))
OWN_VIEWS = {("dsi", "fourier"), ("dsi_cell", "fourier")}                         # tables drawn by their own view below
POSITIONS = {"peak_s", "end_s", "baseline_s"}             # where something is, not a measure of it
# a table's x, in this order of preference; the rest are its y
X_COLUMNS = ("onset_s", "time_ms", "time_s", "time_min", "current_pa", "v_mv", "amplitude_pa", "bin_start_s", "start_s", "lag_s", "isi_s",
             "frequency_hz", "step_duration_s", "sweep", "pulse", "event", "burst")
INDEX = {"sweep", "pulse", "event", "burst", "spike"}
X_WORDS = {"onset_s": "time", "time_ms": "time", "time_s": "time", "time_min": "time", "current_pa": "current", "bin_start_s": "time",
           "start_s": "time", "lag_s": "lag", "isi_s": "interval", "frequency_hz": "frequency",
           "amplitude_pa": "pulse size", "v_mv": "voltage", "step_duration_s": "step duration"}


def _output(analysis: str, column: str, table: bool):
    """The definition of that name: a table's column (registry.column), or a recording's number."""
    if table:
        return registry.column(analysis, column)
    a = registry.load().get(analysis)
    return a.output(column) if a else None


def units_of(analysis: str, column: str, table: bool = True, signal: str = "") -> str:
    o = _output(analysis, column, table)
    u = o.units if o is not None and o.units else next((u for pat, u in UNITS if re.search(pat, column)), "")
    vc = signal.strip().lower() in ("pa", "na")
    return u.replace("signal units", signal or "signal units").replace("charge units", "pC" if vc else "mV*ms") \
        if signal else u


def title_of(analysis: str, column: str, table: bool = True) -> str:
    o = _output(analysis, column, table)
    if o is not None and o.label:
        return o.label
    words = re.sub(r"_(pa|mv|ms|s|hz|pct|pf)$", "", column).replace("_", " ")
    words = re.sub(r"^r ", "R ", re.sub(r"^v ", "V ", words))
    return words[:1].upper() + words[1:]


def _numeric(rows: list[dict], key: str) -> np.ndarray | None:
    vals = [r.get(key) for r in rows]
    if not all(v is None or isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
               for v in vals):
        return None
    a = np.array([np.nan if v is None else float(v) for v in vals])
    return a if np.isfinite(a).any() else None


@dataclass
class View:
    key: str                            # stable across recordings of a type: analysis.table.columns
    title: str
    units: str
    draw: Callable | None = None        # (PlotItem, palette) -> readout
    heatmap: Callable | None = None     # () -> Heatmap, built the first time it is shown
    _built: list = field(default_factory=list, repr=False)

    def built(self):
        if not self._built:
            self._built.append(self.heatmap())
        return self._built[0]


# ---------------------------------------------------------------- heat maps
COLOUR_MAPS = {"sequential": ("cividis", "viridis", "magma", "inferno"), "diverging": ("CET-CBD1", "CET-D1A", "CET-D1")}
LIMITS = {"percentile": "1st to 99th percentile", "full": "full range", "fixed": "fixed"}
SHARED: dict[str, tuple[float, float]] = {}      # a map's limits kept for every recording, by view key


def heat_options() -> dict:
    return {"cmap": "", "mode": "percentile", "fixed": None, "db": False}


def clear_heatmap(item) -> None:
    """Take a heat map's colour bar and axis labels off ``item`` before it draws anything else."""
    bar = getattr(item, "_heat_bar", None)
    if bar is not None:
        item.layout.removeItem(bar)
        if bar.scene() is not None:
            bar.scene().removeItem(bar)
        item._heat_bar = None
    for name in ("left", "bottom"):
        item.getAxis(name).setTicks(None)
    item.getViewBox().setLimits(yMin=None, yMax=None, maxYRange=None)


def shown_heatmap(hm, opts: dict):
    """The map as drawn: in its own units, in dB only when asked (dB an option, never the
    default)."""
    db = hm.power is not None and bool(opts["db"])
    return hmaps.in_db(hm) if db else hm


def draw_heatmap(item, pal, hm, opts: dict, key: str = "") -> str:
    clear_heatmap(item)
    if hm is None:
        return "Nothing to draw: this recording's results have no values for this map."
    shown = shown_heatmap(hm, opts)
    if key in SHARED:
        lv, how = SHARED[key], "kept for every recording"
    else:
        lv = hmaps.levels(shown, opts["mode"], opts["fixed"])
        how = "the measure's own range" if shown.fixed is not None else LIMITS[opts["mode"]]
    names = COLOUR_MAPS[shown.scale]
    cmap = pg.colormap.get(opts["cmap"] if opts["cmap"] in names else names[0])
    img = pg.ImageItem(np.asarray(shown.matrix, np.float32), axisOrder="row-major")
    x0, x1, y0, y1 = shown.extent()
    img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
    item.addItem(img)
    bar = pg.ColorBarItem(values=lv, colorMap=cmap, label=shown.colour_label, interactive=False, width=14,
                          pen=pg.mkPen(pal.plot_axis), colorMapMenu=False)
    bar.setImageItem(img, insert_in=item)
    bar.axis.setPen(pg.mkPen(pal.plot_axis))
    bar.axis.setTextPen(pg.mkPen(pal.plot_text))
    item._heat_bar = bar
    said: set[str] = set()
    for mk in shown.markers:
        text = mk.label if mk.label not in said else ""
        said.add(mk.label)
        item.addItem(pg.InfiniteLine(mk.at, angle=90 if mk.axis == "x" else 0, label=text or None,
                                     pen=pg.mkPen(pal.text, width=1, style=QtCore.Qt.DashLine),
                                     labelOpts={"position": 0.96, "color": pal.text, "movable": False}))
    if shown.mask is not None and shown.mask.any():
        r, c = np.nonzero(shown.mask)
        item.addItem(pg.ScatterPlotItem(shown.x[c], shown.y[r], size=5, pen=None, brush=pg.mkBrush(pal.text)))
    for name, labels, at in (("left", shown.row_labels, shown.y), ("bottom", shown.col_labels, shown.x)):
        if labels is None and 1 <= at.size <= 60 and np.allclose(at, np.round(at)) and np.allclose(np.diff(at), 1):
            labels = [f"{v:g}" for v in at]            # counted rows (sweeps, minutes, pulses): whole numbers
        if labels:
            every = int(np.ceil(len(labels) / 20))      # at most about 20 labels, so none overlap
            item.getAxis(name).setTicks([[(float(v), t) for v, t in list(zip(at, labels))[::every]]])
    item.setLabel("bottom", f"{shown.x_label} ({shown.x_unit})" if shown.x_unit else shown.x_label)
    item.setLabel("left", f"{shown.y_label} ({shown.y_unit})" if shown.y_unit else shown.y_label)
    vb = item.getViewBox()
    vb.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1, maxXRange=x1 - x0, maxYRange=y1 - y0)
    vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
    centre = f", centred on {shown.center:g}" if shown.scale == "diverging" else ""
    return "; ".join([f"Colour: {shown.colour_label}, {lv[0]:.3g} to {lv[1]:.3g} ({how}{centre}); blank: no value"]
                     + shown.notes)


def _heat_views(run) -> list[View]:
    return [View(key, title, "[heat map]", heatmap=build) for key, title, build in hmaps.catalogue(run)]


def views(run) -> list[View]:
    """Every graph the results of ``run`` can give."""
    out: list[View] = []
    signal = run.recording.signal_units if getattr(run, "recording", None) is not None else ""
    for akey in registry.load():
        res = run.results.get(akey)
        if res is None:
            continue
        current = [d for d in res.drawables if getattr(d, "domain", "") == "current"]
        for tname, rows in res.tables.items():
            if not rows or (akey, tname) in OWN_VIEWS:
                continue
            cols = list(dict.fromkeys(k for r in rows for k in r))
            if {"from", "to", "count"} <= set(cols):
                out.append(_histogram(akey, tname, rows))
                continue
            x = next((c for c in X_COLUMNS if c in cols and _numeric(rows, c) is not None), None)
            if x is None:                       # a list (checks, artifacts), not something against something
                continue
            ys = [c for c in cols if c != x and c not in INDEX | POSITIONS and _numeric(rows, c) is not None]
            groups: dict[tuple[str, str], list[str]] = {}
            for c in ys:
                groups.setdefault((c.split("_")[0], units_of(akey, c, signal=signal)), []).append(c)
            for (_, u), members in groups.items():
                out.append(_scatter(res, akey, tname, rows, x, members, u, signal,
                                    [d for d in current if x in ("current_pa", "v_mv")
                                     and d.label.split(":")[0] in members]))
        if akey == "intrinsic":
            out += _first_spikes(run, res)
        if akey == "events":
            out += _overlay(run, res)
        if akey in ("dsi", "dsi_cell"):
            out += _dsi_course(res, akey)
        for d in res.drawables:
            # results kept before drawables said their domain: the averaged event by its name
            if getattr(d, "domain", "") == "event" or (not getattr(d, "domain", "") and d.label == "averaged event"):
                out.append(_event_view(res, akey, d, run))
    return out + _heat_views(run)


def _evenly(x: np.ndarray) -> bool:
    d = np.diff(x[np.isfinite(x)])
    return d.size > 2 and bool(np.all(d > 0)) and np.allclose(d, d[0], rtol=1e-3)


def _scatter(res, akey, tname, rows, x, members, u, signal, lines) -> View:
    xs = _numeric(rows, x) if x else np.arange(len(rows), dtype=float)
    xu = units_of(akey, x, signal=signal) if x else ""
    xt = X_WORDS.get(x, title_of(akey, x).lower()) if x else "row"
    names = [title_of(akey, c) for c in members]
    title = (" and ".join(names) if len(names) <= 3 else f"{names[0]} and {len(names) - 1} more") + f" vs. {xt}"

    def draw(item, pal) -> str:
        colours = theme.series(pal)
        dense = len(rows) > 400
        for k, c in enumerate(members):
            ys = _numeric(rows, c)
            colour = colours[k % len(colours)]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if _evenly(xs[ok]):                 # a series (a density, a correlogram, a step family): joined
                item.plot(xs[ok], ys[ok], pen=pg.mkPen(colour, width=1.2), symbol=None if dense else "o",
                          symbolSize=5, symbolBrush=colour, symbolPen=None, name=names[k])
            else:                               # one point per event, spike or burst: dots
                item.plot(xs[ok], ys[ok], pen=None, symbol="o", symbolSize=4 if dense else 6, symbolBrush=colour,
                          symbolPen=None, name=names[k])
        for d in lines:
            k = members.index(d.label.split(":")[0])
            item.plot(d.x, d.y, pen=pg.mkPen(colours[k % len(colours)], width=1, style=QtCore.Qt.DashLine))
        if x == "current_pa" or u == "pA" and x == "v_mv":
            item.addItem(pg.InfiniteLine(0, angle=0 if x == "v_mv" else 90, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DotLine)))
        item.setLabel("bottom", f"{xt[:1].upper() + xt[1:]}" + (f" ({xu})" if xu else ""))
        item.setLabel("left", u)
        return _readout(res, akey, members, len(rows), signal)

    return View(f"{akey}.{tname}.{'+'.join(members)}", title, f"[{u or '-'} vs. {xu or '-'}]", draw)


def _readout(res, akey, members, n, signal) -> str:
    """The recording's numbers named for these columns (R peak -> its value at 0 pA), else the count."""
    a = registry.load().get(akey)
    said = []
    for c in members:
        for o in (a.outputs if a else ()):
            v = res.values.get(o.key)
            if o.key != c and o.key.startswith(c) and isinstance(v, (int, float)) and np.isfinite(v) \
                    and o.units == units_of(akey, c):
                said.append(f"{o.title}: {v:.4g} {units_of(akey, c, signal=signal)}")
    return " | ".join(said[:4]) if said else f"{n} rows"


def _histogram(akey, tname, rows) -> View:
    lo, hi = _numeric(rows, "from"), _numeric(rows, "to")
    count = _numeric(rows, "count")
    title = title_of(akey, tname)

    def draw(item, pal) -> str:
        ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(count)
        item.addItem(pg.BarGraphItem(x0=lo[ok], x1=hi[ok], height=count[ok], brush=theme.series(pal)[0], pen=None))
        kernel = _numeric(rows, "kernel")
        if kernel is not None:
            mid = (lo + hi) / 2
            item.plot(mid[ok], kernel[ok], pen=pg.mkPen(theme.series(pal)[1], width=1.5))
        item.setLabel("left", "count")
        item.setLabel("bottom", title.replace("Histogram ", "").capitalize())
        return f"{int(np.nansum(count))} in {int(ok.sum())} bins"

    return View(f"{akey}.{tname}", title, "[count]", draw)


def _first_spikes(run, res) -> list[View]:
    """NeuroExpress's spike panel: each step's first full action potential, as V,
    dV/dt and d2V/dt2 against time from its peak, and as a phase plot."""
    rec = getattr(run, "recording", None)
    rows = [r for r in res.tables.get("spikes", []) if r.get("spike") == 1 and r.get("full")]
    if rec is None or not rows:
        return []
    fs = rec.sample_rate_hz
    pre, post = int(0.003 * fs), int(0.006 * fs)
    cuts = []
    for r in rows[:40]:
        p = int(round(r["time_ms"] / 1e3 * fs))
        if p - pre < 0 or p + post > rec.n_samples:
            continue
        v = np.asarray(rec.sweep(r["sweep"]), float)[p - pre:p + post]
        dv = np.gradient(v) * fs / 1e3                       # mV/ms
        cuts.append((r["current_pa"], v, dv, np.gradient(dv) * fs / 1e3))
    if not cuts:
        return []
    t = (np.arange(pre + post) - pre) / fs * 1e3
    span = f"the first spike of {len(cuts)} step(s), {cuts[0][0]:g} to {cuts[-1][0]:g} pA"

    def make(key, title, units, xy, xlabel, ylabel):
        def draw(item, pal) -> str:
            colours = theme.series(pal)
            for k, c in enumerate(cuts):
                x, y = xy(c)
                item.plot(x, y, pen=pg.mkPen(colours[k % len(colours)], width=1.2))
            item.setLabel("bottom", xlabel)
            item.setLabel("left", ylabel)
            return span
        return View(f"intrinsic.first spikes.{key}", title, units, draw)

    return [make("v", "First spikes", "[mV vs. ms]", lambda c: (t, c[1]), "Time from the peak (ms)", "mV"),
            make("dvdt", "First spikes: dV/dt", "[mV/ms vs. ms]", lambda c: (t, c[2]), "Time from the peak (ms)",
                 "mV/ms"),
            make("d2vdt2", "First spikes: d2V/dt2", "[mV/ms2 vs. ms]", lambda c: (t, c[3]), "Time from the peak (ms)",
                 "mV/ms2"),
            make("phase", "First spikes: phase plot", "[mV/ms vs. mV]", lambda c: (c[1], c[2]), "V (mV)", "dV/dt (mV/ms)")]


def _overlay(run, res) -> list[View]:
    """NeuroExpress's event overlay: the first 20 accepted events, from 5 ms before
    their onset to 25 ms after, and their average."""
    rec = getattr(run, "recording", None)
    rows = [r for r in res.tables.get("events", []) if r.get("accepted") and np.isfinite(r.get("onset_s", np.nan))]
    if rec is None or not rows:
        return []
    fs = rec.sample_rate_hz
    pre, post = int(0.005 * fs), int(0.025 * fs)
    sweeps: dict[int, np.ndarray] = {}
    cuts = []
    for r in rows:
        if len(cuts) == 20:
            break
        k = int(r.get("sweep", 0) or 0)
        if k not in sweeps:
            sweeps[k] = np.asarray(rec.sweep(k), float)
        i = int(round(r["onset_s"] * fs))
        if i - pre >= 0 and i + post <= sweeps[k].size:
            y = sweeps[k][i - pre:i + post]
            cuts.append(y - np.median(y[:pre]))
    if not cuts:
        return []
    t = (np.arange(pre + post) - pre) / fs * 1e3
    units = rec.signal_units

    def draw(item, pal) -> str:
        for y in cuts:
            item.plot(t, y, pen=pg.mkPen(pal.sweep, width=0.8))
        item.plot(t, np.mean(cuts, axis=0), pen=pg.mkPen(theme.series(pal)[0], width=2))
        item.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DotLine)))
        item.setLabel("bottom", "Time from onset (ms)")
        item.setLabel("left", units)
        return f"{len(cuts)} of {len(rows)} accepted events, each against its own baseline, and their average"

    return [View("events.overlay", "Events and their average", f"[{units} vs. ms]", draw)]


def _dsi_course(res, akey: str = "dsi") -> list[View]:
    """DSI's time course: each step's responses as % of its own baseline (thin
    lines), and their mean with its SEM, one per step duration when they differ."""
    rows = res.tables.get("course", [])
    if not rows:
        return []
    durations = sorted({round(r["duration_s"], 2) for r in rows})

    def make(col: str, what: str) -> View:
        def draw(item, pal) -> str:
            colours = theme.series(pal)
            for ep in sorted({r["episode"] for r in rows}):
                mine = sorted((r for r in rows if r["episode"] == ep), key=lambda r: r["response"])
                item.plot([r["from_step_s"] for r in mine], [r[col] for r in mine], pen=pg.mkPen(pal.sweep, width=0.8))
            for k, d in enumerate(durations):
                mine = [r for r in rows if round(r["duration_s"], 2) == d]
                xs, ys, es = [], [], []
                for n in sorted({r["response"] for r in mine}):
                    at = np.array([r[col] for r in mine if r["response"] == n], float)
                    at = at[np.isfinite(at)]
                    if at.size:
                        xs.append(np.mean([r["from_step_s"] for r in mine if r["response"] == n]))
                        ys.append(at.mean())
                        es.append(at.std(ddof=1) / np.sqrt(at.size) if at.size > 1 else 0.0)
                colour = colours[k % len(colours)]
                item.addItem(pg.ErrorBarItem(x=np.array(xs), y=np.array(ys), height=2 * np.array(es),
                                             pen=pg.mkPen(colour, width=1)))
                item.plot(xs, ys, pen=pg.mkPen(colour, width=2), symbol="o", symbolSize=6, symbolBrush=colour,
                          symbolPen=None, name=f"{d:g} s step, mean ± SEM" if len(durations) > 1 else "mean ± SEM")
            item.addItem(pg.InfiniteLine(100, angle=0, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DotLine)))
            item.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DotLine)))
            item.setLabel("bottom", "Time from the step (s)")
            item.setLabel("left", "% of baseline")
            n_ep = len({r["episode"] for r in rows})
            return f"{n_ep} step(s), each as % of its own baseline responses; before 0 from the step's start, " \
                   "after it from its end"

        return View(f"{akey}.time course.{col}", f"{what}, % of baseline vs. time from the step", "[% vs. s]", draw)

    views_ = [make("pct", "Peak"), make("power_pct", "Power (Fourier)")]
    spectra = res.tables.get("fourier", [])
    if spectra:
        by_response: dict[tuple, list] = {}
        for r in spectra:
            by_response.setdefault((r.get("file", ""), r["sweep"], r["pulse"]), []).append(r)
        means = res.tables.get("spectra", [])
        units = res.values.get("units", "")

        def draw(item, pal) -> str:
            for rs in by_response.values():
                item.plot([r["frequency_hz"] for r in rs], [r["fft_amplitude"] for r in rs],
                          pen=pg.mkPen(pal.sweep, width=0.6))
            for k, (col, name) in enumerate((("amp_baseline", "baseline"), ("amp_suppressed", "suppressed"),
                                             ("amp_recovered", "end of recovery"))):
                if means and col in means[0]:
                    item.plot([m["frequency_hz"] for m in means], [m[col] for m in means], name=name,
                              pen=pg.mkPen(theme.series(pal)[k], width=2))
            item.setLabel("bottom", "Frequency (Hz)")
            item.setLabel("left", f"Fourier amplitude ({units})")
            return f"{len(by_response)} responses' Fourier transforms (thin), and the means by phase"

        views_.append(View(f"{akey}.fourier", "Fourier amplitude of every response vs. frequency",
                           f"[{units} vs. Hz]", draw))
    return views_


def _event_view(res, akey, d, run) -> View:
    u = run.recording.signal_units if run.recording is not None else ""

    def draw(item, pal) -> str:
        item.plot(d.x, d.y, pen=pg.mkPen(theme.series(pal)[0], width=1.8))
        item.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DotLine)))
        item.setLabel("bottom", "Time from onset (ms)")
        item.setLabel("left", u)
        v = res.values
        return " | ".join(f"{k}: {v[x]:.3g} ms" for k, x in (("rise 10-90", "rise_10_90"), ("decay tau", "decay_tau"))
                          if isinstance(v.get(x), float) and np.isfinite(v[x])) or d.label

    return View(f"{akey}.{d.label}", d.label[:1].upper() + d.label[1:], f"[{u or '-'} vs. ms]", draw)


def corner_buttons(owner, maximize: bool = True, detach: bool = True) -> list[QtWidgets.QToolButton]:
    """A graph's edit-as-figure, maximize (and restore) and own-window buttons; they emit the
    owner's ``edit_requested``, ``maximize_requested`` and ``detach_requested``."""
    edit = QtWidgets.QToolButton()
    edit.setText("Edit")
    edit.setToolTip("Edit this graph as a publication figure in the figure builder: its data, colours, axes and "
                    "labels, then export")
    edit.clicked.connect(lambda: owner.edit_requested.emit(owner))
    out = [edit]
    if not detach:
        return out
    if maximize:
        owner.max_button = QtWidgets.QToolButton()
        owner.max_button.setText("⤢")
        owner.max_button.setToolTip("Fill the Figures area with this graph (double-click its title); Esc restores")
        owner.max_button.clicked.connect(lambda: owner.maximize_requested.emit(owner))
        out.append(owner.max_button)
    own = QtWidgets.QToolButton()
    own.setText("⧉")
    own.setToolTip("Open this graph in its own window (for a second monitor)")
    own.clicked.connect(lambda: owner.detach_requested.emit(owner))
    return out + [own]


class Panel(QtWidgets.QFrame):
    """A titled plot with its menu and its readout. ``tools``: the maximize and
    own-window buttons (a graph already in its own window has none)."""

    maximize_requested = QtCore.Signal(object)
    detach_requested = QtCore.Signal(object)
    edit_requested = QtCore.Signal(object)

    def __init__(self, palette, wanted: tuple[str, ...] = (), tools: bool = True) -> None:
        super().__init__()
        self.setObjectName("surface")
        self.palette, self.wanted = palette, wanted
        self._views: list[View] = []
        self._key = ""
        lay = box("v", "xs", "s")
        head = box("h", "s")
        self.title = label("")                  # the title in bold, then its units, on one line
        self.title.setTextFormat(QtCore.Qt.RichText)
        self.readout = label("", "muted")
        for w in (self.title, self.readout):    # a long line is cut, never widens the grid
            w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.menu_button = QtWidgets.QToolButton()
        self.menu_button.setText("Show")
        self.menu_button.setToolTip("What this graph shows")
        self.menu_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.menu = QtWidgets.QMenu(self.menu_button)
        self.menu_button.setMenu(self.menu)
        self.colour_button = QtWidgets.QToolButton()
        self.colour_button.setText("Colour")
        self.colour_button.setToolTip("The heat map's colours: the map, the limits, dB, and limits kept for every "
                                      "recording compared")
        self.colour_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.colour_menu = QtWidgets.QMenu(self.colour_button)
        self.colour_menu.aboutToShow.connect(self._fill_colour_menu)
        self.colour_button.setMenu(self.colour_menu)
        self.colour_button.setVisible(False)
        self.heat = heat_options()
        head.addWidget(self.title, 1)
        head.addWidget(self.colour_button)
        head.addWidget(self.menu_button)
        for b in corner_buttons(self, maximize=tools, detach=tools):
            head.addWidget(b)
        self.tools = tools
        lay.addLayout(head)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(160)
        theme.style_plot(self.plot, palette)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.readout)
        self.setLayout(lay)

    def set_views(self, all_views: list[View], taken: set[str] = frozenset()) -> str:
        """Offer ``all_views``; show the one shown before, else the first wanted,
        else the first not ``taken`` by another panel. Returns the key shown."""
        self._views = all_views
        self.menu.clear()
        for v in all_views:
            self.menu.addAction(f"{v.title} {v.units}", lambda k=v.key: self.show_view(k))
        keys = [v.key for v in all_views]
        if self._key not in keys:
            self._key = next((k for w in self.wanted for k in keys if k.startswith(w)), "") or next(
                (k for k in keys if k not in taken), keys[0] if keys else "")
        self.show_view(self._key)
        return self._key

    def show_view(self, key: str) -> None:
        self._key = key
        self.plot.clear()
        item = self.plot.getPlotItem()
        view = next((v for v in self._views if v.key == key), None)
        import html

        self.title.setText(f"<b>{html.escape(view.title)}</b> {html.escape(view.units)}" if view else
                           "Nothing to show")
        self.title.setToolTip(f"{view.title} {view.units}" if view else "")
        legend = item.addLegend(offset=(10, 10))
        legend.clear()
        clear_heatmap(item)
        heat = view is not None and view.heatmap is not None
        self.colour_button.setVisible(heat)
        if heat:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                text = draw_heatmap(item, self.palette, view.built(), self.heat, view.key)
            except Exception as exc:            # a map that cannot be made says why; the panel stays usable
                text = f"Could not make this map: {exc}"
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        elif view is not None:
            text = view.draw(item, self.palette)
        else:
            text = ""
            item.setLabel("bottom", "")
            item.setLabel("left", "")
        legend.setLabelTextColor(self.palette.plot_text)
        back = pg.mkColor(self.palette.plot_background)
        back.setAlpha(220)
        legend.setBrush(pg.mkBrush(back))
        legend.setVisible(len(legend.items) > 1)
        if not heat:
            theme.keep_in_frame(item)
        self.readout.setText(text)
        self.readout.setToolTip(text)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tools and event.pos().y() < self.plot.geometry().top():      # on the title line
            self.maximize_requested.emit(self)
        else:
            super().mouseDoubleClickEvent(event)

    def as_figure(self, source: str = ""):
        """The graph shown, as a figure the builder can edit (None when nothing is drawn)."""
        from . import editable

        view = next((v for v in self._views if v.key == self._key), None)
        if view is None:
            return None
        if view.heatmap is not None:
            hm = view.built()
            return editable.from_heatmap(shown_heatmap(hm, self.heat) if hm is not None else None, view.title,
                                         source)
        return editable.from_plot(self.plot.getPlotItem(), view.title, source)

    def copy(self) -> "Panel":
        """The same graph, for a window of its own."""
        p = Panel(self.palette, tools=False)
        p._key, p.heat = self._key, dict(self.heat)
        p.set_views(self._views)
        return p

    def _fill_colour_menu(self) -> None:
        """The heat map's choices, as they stand for the map shown."""
        m = self.colour_menu
        m.clear()
        view = next((v for v in self._views if v.key == self._key), None)
        hm = view.built() if view is not None and view.heatmap is not None else None
        if hm is None:
            return
        shown = shown_heatmap(hm, self.heat)

        def choose(**kw):
            self.heat.update(kw)
            self.show_view(self._key)

        m.addSection("Colour map")
        names = COLOUR_MAPS[shown.scale]
        for name in names:
            a = m.addAction(name, lambda n=name: choose(cmap=n))
            a.setCheckable(True)
            a.setChecked(name == (self.heat["cmap"] if self.heat["cmap"] in names else names[0]))
        m.addSection("Limits")
        for mode, words in LIMITS.items():
            a = m.addAction(words[:1].upper() + words[1:] + ("..." if mode == "fixed" else ""),
                            lambda md=mode: self._fixed_limits(shown) if md == "fixed" else choose(mode=md))
            a.setCheckable(True)
            a.setChecked(self.heat["mode"] == mode and self._key not in SHARED)
        if hm.power is not None:
            m.addSection("Values")
            a = m.addAction("Show in dB", lambda: choose(db=not self.heat["db"]))
            a.setCheckable(True)
            a.setChecked(bool(self.heat["db"]))
        m.addSeparator()
        keep = m.addAction("Keep these limits for this map on every recording", self._share)
        keep.setCheckable(True)
        keep.setChecked(self._key in SHARED)
        keep.setToolTip("Recordings compared side by side then share one colour scale")

    def _fixed_limits(self, shown) -> None:
        lo, hi = hmaps.levels(shown, self.heat["mode"], self.heat["fixed"])
        text, ok = QtWidgets.QInputDialog.getText(self, "Fixed limits", f"Lowest and highest {shown.colour_label}, "
                                                  "separated by a comma:", text=f"{lo:.4g}, {hi:.4g}")
        if not ok:
            return
        try:
            a, b = (float(v) for v in text.split(","))
        except ValueError:
            return
        if b > a:
            self.heat.update(mode="fixed", fixed=(a, b))
            SHARED.pop(self._key, None)
            self.show_view(self._key)

    def _share(self) -> None:
        if self._key in SHARED:
            SHARED.pop(self._key)
        else:
            view = next((v for v in self._views if v.key == self._key), None)
            if view is not None and view.built() is not None:
                SHARED[self._key] = hmaps.levels(shown_heatmap(view.built(), self.heat), self.heat["mode"],
                                                 self.heat["fixed"])
        self.show_view(self._key)

    def on_theme(self, palette) -> None:
        self.palette = palette
        theme.style_plot(self.plot, palette)
        self.show_view(self._key)
