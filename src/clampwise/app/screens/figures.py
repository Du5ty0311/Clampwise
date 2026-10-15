"""Figures (B11, NeuroExpress's way): the selected recording's graphs, four at a
time, each with its own menu of everything the results can show (B11: a small
set of the most useful graphs for the type, the rest one click away). The
figure builder (B12) comes later.

Any graph fills the Figures area (its ⤢ button or a double-click on its
title; Esc or the button again restores the grid) or opens in a window of
its own for a second monitor."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...analysis import registry
from .. import theme, widgets
from ..panels import Panel, View, clear_heatmap, corner_buttons, draw_heatmap, heat_options, views
from ..widgets import box, label
from .base import Screen

# what a recording can be grouped by: (its words, where it comes from)
CATEGORIES = (("Condition", "identity", "condition"), ("Animal", "identity", "animal"),
              ("Cell type", "metadata", "cell_type"), ("Genotype", "metadata", "genotype"),
              ("Sex", "metadata", "sex"), ("Region", "metadata", "region"), ("Age", "metadata", "age"))


def _number(r, key) -> float:
    res = r.results.get(key[0])
    v = res.values.get(key[1]) if res is not None else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan


class Across(QtWidgets.QWidget):
    """Every analyzed recording of a type, one point each: any recording-level
    number against any other, coloured by a category (NeuroExpress's cross-cell
    scatter). Read from the project's store; nothing is opened or re-analyzed."""

    maximize_requested = QtCore.Signal(object)
    detach_requested = QtCore.Signal(object)
    edit_requested = QtCore.Signal(object)

    def __init__(self, window, tools: bool = True) -> None:
        super().__init__()
        self.window = window
        lay = box("v", "s")
        row = box("h", "s")
        self.kind, self.x, self.y, self.colour = (QtWidgets.QComboBox() for _ in range(4))
        for text, combo in (("Recordings", self.kind), ("X", self.x), ("Y", self.y), ("Colour by", self.colour)):
            row.addWidget(label(text, "muted"))
            row.addWidget(combo, 1)
        self.correlations = QtWidgets.QToolButton()
        self.correlations.setText("Correlations")
        self.correlations.setCheckable(True)
        self.correlations.setToolTip("A heat map of the Spearman correlations between the type's headline "
                                     "numbers, instead of X against Y")
        self.correlations.toggled.connect(self._draw)
        row.addWidget(self.correlations)
        for b in corner_buttons(self, maximize=tools, detach=tools):
            row.addWidget(b)
        lay.addLayout(row)
        self.plot = pg.PlotWidget()
        theme.style_plot(self.plot, window.palette)
        self.plot.addLegend(offset=(10, 10))
        lay.addWidget(self.plot, 1)
        self.readout = label("", "muted", wrap=True)
        lay.addWidget(self.readout)
        self.setLayout(lay)
        for text, *_ in CATEGORIES:
            self.colour.addItem(text)
        self.kind.currentIndexChanged.connect(self._outputs)
        for combo in (self.x, self.y, self.colour):
            combo.currentIndexChanged.connect(self._draw)
        self._rows = []
        self._hm = None

    def as_figure(self, source: str = ""):
        """The comparison shown, as a figure the builder can edit (None when nothing is drawn)."""
        from .. import editable

        t = registry.TYPE_WORDS.get(self.kind.currentData(), self.kind.currentData() or "")
        if self.correlations.isVisible() and self.correlations.isChecked():
            return editable.from_heatmap(self._hm, f"Correlations between the headline measures: {t}", source)
        return editable.from_plot(self.plot.getPlotItem(), f"{self.y.currentText()} vs. {self.x.currentText()}"
                                  f" ({t})", source)

    def refresh(self) -> None:
        from ... import export

        p = self.window.project
        self._rows = export.collect(p) if p is not None else []
        here = [t for t in registry.TYPES if any(r.recording_type == t for r in self._rows)]
        keep = self.kind.currentData()
        self.kind.blockSignals(True)
        self.kind.clear()
        for t in here:
            self.kind.addItem(f"{registry.TYPE_WORDS.get(t, t)} ({sum(r.recording_type == t for r in self._rows)})", t)
        if keep in here:
            self.kind.setCurrentIndex(here.index(keep))
        self.kind.blockSignals(False)
        self._outputs()

    def _outputs(self) -> None:
        t = self.kind.currentData()
        rows = [r for r in self._rows if r.recording_type == t]
        found = []
        for key, a in registry.load().items():
            for o in a.outputs:
                if o.level == "recording" and any(np.isfinite(_number(r, (key, o.key))) for r in rows):
                    found.append((f"{o.title} ({o.units})" if o.units else o.title, (key, o.key), o.headline))
        # it opens on the type's own headline numbers, not the checks every recording has
        support = ("baseline.", "bridge.", "quality.", "artifacts.", "rs_error.")
        heads = [k for k, (_, data, h) in enumerate(found) if h and not f"{data[0]}.".startswith(support)] or \
            [k for k, (_, _, h) in enumerate(found) if h]
        self._heads = [found[k][:2] for k in heads]
        for combo, nth in ((self.x, 0), (self.y, 1)):
            combo.blockSignals(True)
            combo.clear()
            for text, data, _ in found:
                combo.addItem(text, data)
            if found:
                combo.setCurrentIndex(heads[nth] if len(heads) > nth else min(nth, len(found) - 1))
            combo.blockSignals(False)
        self._draw()

    def _draw(self) -> None:
        self.plot.clear()
        item = self.plot.getPlotItem()
        clear_heatmap(item)
        heat = self.window.project is not None and self.window.project.features()["heat_maps"]
        self.correlations.setVisible(heat)
        for combo in (self.x, self.y, self.colour):
            combo.setEnabled(not (heat and self.correlations.isChecked()))
        if heat and self.correlations.isChecked():
            from ...analysis import heatmaps

            t = self.kind.currentData()
            rows = [r for r in self._rows if r.recording_type == t]
            hm = heatmaps.correlation({text: np.array([_number(r, data) for r in rows]) for text, data in
                                       getattr(self, "_heads", [])})
            self._hm = hm
            text = draw_heatmap(item, self.window.palette, hm, heat_options(), "across.correlation")
            self.readout.setText(text if hm is not None else "Correlations need at least 4 recordings with two "
                                 "headline numbers that vary.")
            return
        t, xk, yk = self.kind.currentData(), self.x.currentData(), self.y.currentData()
        if not (t and xk and yk):
            self.readout.setText("Analyze recordings to compare them here.")
            return
        _, where, attr = CATEGORIES[max(0, self.colour.currentIndex())]
        groups: dict[str, list] = {}
        for r in (r for r in self._rows if r.recording_type == t):
            x, y = _number(r, xk), _number(r, yk)
            if np.isfinite(x) and np.isfinite(y):
                holder = getattr(r, where)
                cat = str(getattr(holder, attr, "") or "(none)") if holder is not None else "(none)"
                groups.setdefault(cat, []).append((x, y, r.name))
        colours = theme.series(self.window.palette)
        for k, (cat, pts) in enumerate(sorted(groups.items())):
            xs, ys, names = zip(*pts)
            self.plot.addItem(pg.ScatterPlotItem(list(xs), list(ys), size=9, pen=None, brush=colours[k % len(colours)],
                                                 name=cat, data=list(names), hoverable=True,
                                                 tip=lambda x, y, data: f"{data}: {x:.4g}, {y:.4g}"))
        self.plot.getPlotItem().setLabel("bottom", self.x.currentText())
        self.plot.getPlotItem().setLabel("left", self.y.currentText())
        theme.keep_in_frame(self.plot)
        n = sum(len(v) for v in groups.values())
        self.readout.setText(f"{n} recording(s) with both numbers, in {len(groups)} group(s) by "
                             f"{self.colour.currentText().lower()}; hover a point for its file.")

    def copy(self) -> "Across":
        """The same comparison, for a window of its own (its selectors its own from then on)."""
        a = Across(self.window, tools=False)
        a.refresh()
        for mine, theirs in ((self.kind, a.kind), (self.x, a.x), (self.y, a.y), (self.colour, a.colour)):
            theirs.setCurrentIndex(max(0, theirs.findText(mine.currentText())))
        return a

    def on_theme(self, palette) -> None:
        theme.style_plot(self.plot, palette)
        self._draw()


# the graphs a recording type opens on (prefixes of the view keys, panels.views)
DEFAULTS = {
    "cc_steps": (("intrinsic.steps.v_",), ("intrinsic.steps.r_peak",), ("intrinsic.first spikes.phase",
                                                                        "intrinsic.steps.sag_mv"),
                 ("intrinsic.fi.rate_hz", "intrinsic.steps.rate_hz")),
    "vc_spontaneous": (("events.overlay", "events.averaged event"), ("events.events.iei_ms",),
                       ("events.events.decay_tau_ms",), ("events.histogram_amplitudes",)),
    "cc_short_square": (("short_square.sweeps.spikes",), ("short_square.sweeps.latency_ms",),
                        ("short_square.sweeps.threshold_mv",), ("artifacts",)),
    "vc_steps": (("vc_iv.steps.i_",), ("vc_iv.steps.g_ns", "vc_iv.steps.r_slope"), ("vc_iv.steps.r_slope",),
                 ("artifacts",)),
    "cc_ramp": (("ramp.ramps.threshold_pa",), ("ramp.ramps.latency_s",), ("ramp.ramps.spikes",), ("baseline",)),
    "spike_times": (("spike_train.spike_density.",), ("spike_train.isi_histogram",),
                    ("spike_train.return_map.next_isi_s",), ("spike_train.autocorrelogram.count",)),
    "dsi": (("dsi.time course.pct",), ("dsi.time course.power_pct",), ("dsi.fourier",),
            ("dsi.episodes.suppression_pct",)),
    "vc_evoked": (("evoked.pulses.amplitude",), ("evoked.sweeps.",), ("evoked.pulses.latency",),
                  ("evoked.pulses.decay",)),
}
DEFAULTS["dsi_cell"] = tuple(tuple(w.replace("dsi.", "dsi_cell.", 1) for w in ws) for ws in DEFAULTS["dsi"])
DEFAULTS["cc_spontaneous"] = DEFAULTS["vc_spontaneous"]
DEFAULTS["cc_evoked"] = DEFAULTS["vc_evoked"]


class FiguresScreen(Screen):
    key = "figures"
    title = "Figures"
    purpose = "Plots of the selected recording, four at a time; each plot's Show menu lists every view its " \
              "results support."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.across_button = widgets.button("Across recordings", self._toggle,
                                            "Compare every analyzed recording of a type, one point each")
        self.across_button.setCheckable(True)
        self.add_action(self.across_button)
        self.add_action(widgets.primary("Make a figure", self.open_builder,
                                        "The figure builder: from what you want to show to a publication figure. "
                                        "To start from a graph below, use its Edit button."))
        self.builder = None
        self.stack = QtWidgets.QStackedWidget()
        self.empty = widgets.EmptyState("Select an analyzed recording", "Pick a recording in Files; once analyzed, its "
                                        "graphs show here.", [("Go to Files", lambda: window.go("files"), False)])
        self.stack.addWidget(self.empty)
        grid_page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(theme.SPACE["s"])
        grid.setContentsMargins(*[theme.SPACE["xs"]] * 4)
        self.panels = [Panel(window.palette) for _ in range(4)]
        for k, p in enumerate(self.panels):
            grid.addWidget(p, k // 2, k % 2)
        grid_page.setLayout(grid)
        self.stack.addWidget(grid_page)
        self.across = Across(window)
        self.stack.addWidget(self.across)
        self.body(self.stack)
        self._pending = None
        self._type = ""
        self.maximized = None             # the graph filling the area, if one does
        self.windows: list[QtWidgets.QWidget] = []
        for g in self.panels + [self.across]:
            g.maximize_requested.connect(self.maximize)
            g.detach_requested.connect(self.detach)
            g.edit_requested.connect(self.edit_graph)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, lambda: self.maximized is not None and self.maximize(
            self.maximized), context=QtCore.Qt.WidgetWithChildrenShortcut)

    def maximize(self, graph) -> None:
        """Fill the area with ``graph``, or, when it already does, restore the grid."""
        restore = self.maximized is graph
        self.maximized = None if restore else graph
        if graph is self.across:              # its selectors and readout give way to the plot
            for w in self.across.findChildren(QtWidgets.QComboBox) + self.across.findChildren(QtWidgets.QLabel):
                w.setVisible(restore)
        else:
            for p in self.panels:
                p.setVisible(restore or p is graph)
        graph.max_button.setText("⤢" if restore else "⤡")
        graph.max_button.setToolTip("Fill the Figures area with this graph (double-click its title); Esc restores"
                                    if restore else "Back to every graph (Esc)")
        graph.setFocus()

    def detach(self, graph) -> None:
        """The graph in a window of its own, drawn from the same results; it stays when Figures moves on."""
        win = QtWidgets.QWidget(None, QtCore.Qt.Window)
        win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        lay = box("v", "s", "s")
        copy = graph.copy()
        copy.edit_requested.connect(self.edit_graph)
        lay.addWidget(copy)
        win.setLayout(lay)
        what = "Across recordings" if graph is self.across else copy.title.toolTip() or "Graph"
        name = self.window.selected.name if self.window.selected is not None and graph is not self.across else ""
        win.setWindowTitle(" – ".join(x for x in (what, name, "Clampwise") if x))
        win.resize(900, 600)
        win.graph = copy
        self.windows.append(win)
        win.destroyed.connect(lambda _=None, w=win: self.windows.remove(w) if w in self.windows else None)
        win.show()

    def on_shown(self) -> None:
        if self.across_button.isChecked():
            self.across.refresh()
            return
        self.on_selection(self._pending if self._pending is not None else self.window.selected)

    def _toggle(self) -> None:
        if self.across_button.isChecked():
            self.across.refresh()
            self.stack.setCurrentWidget(self.across)
        else:
            self.on_selection(self.window.selected)

    def on_project(self) -> None:
        if self.window.project is None:
            self.stack.setCurrentWidget(self.empty)

    def on_selection(self, path) -> None:
        if self.across_button.isChecked():
            return
        if path is not None and not self.is_current():
            self._pending = path
            return
        self._pending = None
        run = None
        if path is not None and self.window.project is not None:
            try:
                run = self.window.project.last_run(path, self.window.recording(path))
            except Exception as exc:
                self.window.say(f"Could not read the results of {path.name}: {exc}")
        kind = run.recording_type if run is not None else ""
        if run is not None and run.results:
            # a cell's results over its files (DSI with the step in a file of its own) beside the file's own
            cell = self.window.project.cell_result(path)
            if cell is not None and cell.results:
                from types import SimpleNamespace

                run = SimpleNamespace(results={**run.results, **cell.results}, recording=run.recording,
                                      recording_type=run.recording_type)
                kind = "dsi_cell" if "dsi_cell" in cell.results else kind
        found = views(run) if run is not None and run.results else []
        if self.window.project is not None and not self.window.project.features()["heat_maps"]:
            found = [v for v in found if v.heatmap is None]
        if found and path is not None:
            found += self._health_views(path)
        if not found:
            self.stack.setCurrentWidget(self.empty)
            return
        if kind != self._type:          # a new type opens on its own graphs
            self._type = kind
            for p, wanted in zip(self.panels, DEFAULTS.get(kind, ((), (), (), ()))):
                p.wanted, p._key = wanted, ""
        taken: set[str] = set()
        for p in self.panels:
            taken.add(p.set_views(found, taken))
        self.stack.setCurrentIndex(1)

    def _health_views(self, path) -> list:
        """The cell health timeline: Rs, holding and noise across the cell's files on one clock."""
        from ...project import health

        p = self.window.project
        key = p.identity(path).cell_key()
        paths = [r.path for r in p.recordings()
                 if (p.identity(r.path).cell_key() == key if any(key) else r.path.parent == path.parent)]
        if len(paths) < 2:
            return []
        cache: dict = {}
        try:
            signal = self.window.recording(path).signal_units
        except Exception:
            signal = ""

        def timeline():
            if "t" not in cache:
                cache["t"] = health.timeline(p, paths, self.window.recording)
            return cache["t"]

        def make(measure: str) -> View:
            name, unit = health.MEASURES[measure]

            def draw(item, pal) -> str:
                t = timeline()
                rows = t.rows[t.rows.measure == measure]
                units = unit or t.units
                colours = theme.series(pal)
                for k, (_f, g) in enumerate(rows.groupby("file", sort=False)):
                    c = colours[k % len(colours)]
                    item.plot(g.minute.to_numpy(), g.value.to_numpy(), pen=None, symbol="o", symbolSize=4,
                              symbolBrush=c, symbolPen=None)
                if len(rows) > 1:
                    first = float(rows.value.iloc[:3].median())
                    lim = health.limits(p)
                    vc = t.units.lower() in ("pa", "na")
                    bounds = {"rs": (first * (1 - lim["rs_change_pct"] / 100), first * (1 + lim["rs_change_pct"] / 100)),
                              "holding": (first - (lim["holding_change"] if vc else 5.0),
                                          first + (lim["holding_change"] if vc else 5.0)),
                              "noise": (None, first * lim["noise_ratio"])}[measure]
                    for b in bounds:
                        if b is not None:
                            item.addItem(pg.InfiniteLine(b, angle=0, pen=pg.mkPen(pal.warn, width=1,
                                                                                  style=QtCore.Qt.DashLine)))
                item.setLabel("bottom", "Time on the cell's clock (min)")
                item.setLabel("left", f"{name} ({units})" if units else name)
                said = [w for w in t.warnings if w.lower().startswith({"rs": "rs", "holding": "the holding",
                                                                       "noise": "the noise"}[measure])]
                if rows.empty:
                    return {"rs": "No Rs: these files have no test step; type Clampex's Membrane Test Rs under "
                                  "Files, Experiment details"}.get(measure, "Nothing measured")
                return " ".join(f"⚠ {w}" for w in said) or (f"{rows.file.nunique()} files, {len(rows)} points; "
                                                            "dashed lines: drift limits")

            return View(f"health.{measure}", f"Cell health: {name.lower()} over the cell's files",
                        f"[{unit or signal or '-'} vs. min]", draw)

        return [make(m) for m in ("rs", "holding", "noise")]

    def edit_graph(self, graph):
        """``graph`` (a panel or the comparison across recordings) opened in the figure builder, editable."""
        name = self.window.selected.name if self.window.selected is not None and graph is not self.across else ""
        try:
            spec = graph.as_figure(name)
        except Exception as exc:                # a graph that cannot be read back says why
            self.window.say(f"Could not open this graph as a figure: {exc}")
            return None
        if spec is None:
            self.window.say("This graph has nothing drawn to edit.")
            return None
        b = self.open_builder(spec)
        b.statusBar().showMessage("Opened from the Figures screen: change the data, the look and the labels, then "
                                  "Export or Save figure.", 15000)
        return b

    def open_builder(self, spec=None):
        """The figure builder, in a window of its own (one at a time, kept open)."""
        from ..builder import Builder

        if self.builder is None or spec is not None:
            self.builder = Builder(self.window, spec)
        self.builder.show()
        self.builder.raise_()
        self.builder.activateWindow()
        return self.builder

    def on_theme(self, palette) -> None:
        for p in self.panels:
            p.on_theme(palette)
        self.across.on_theme(palette)
        for w in self.windows:
            w.graph.on_theme(palette)
