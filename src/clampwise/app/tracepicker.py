"""The figure builder's trace picker: which raw traces an example panel shows.

Left, the sweeps of one recording (or, for DSI, every evoked response with
where it sits around the steps), selected by hand or by a rule: a time range,
or an episode's last baseline responses and first responses after the step.
Each selection becomes a named trace, drawn as the average of its sweeps, as
every sweep, or as the sweep closest to their mean. Right, the traces of the
figure and a live preview; under them the window cut around each sweep's
stimulus, the baseline subtraction and the artifact blanking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import traces
from . import theme, widgets
from .widgets import box, label


def _menu_on(widget, fill) -> None:
    """A right-click menu on ``widget``: ``fill(menu, pos)`` adds its actions."""
    widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def show(pos):
        m = QtWidgets.QMenu(widget)
        fill(m, pos)
        if not m.isEmpty():
            m.exec_(widget.mapToGlobal(pos))

    widget.customContextMenuRequested.connect(show)


class TracePicker(QtWidgets.QDialog):
    def __init__(self, parent, window, load, sets=None, w: traces.Window | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose traces")
        self.window_, self.load = window, load
        self.sets: list[traces.TraceSet] = [traces.TraceSet(s.name, s.mode, [list(i) for i in s.items])
                                            for s in (sets or [])]
        self.w = w or traces.Window()
        self._rows: list[list] = []          # the item of each row of the sweeps table
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._preview)
        self._build()
        self._sources()
        self._fill_sets()
        self._preview()

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        outer = box("v", "m", "m")
        intro = label("Pick sweeps on the left and add them as a named trace (for example 'Before DSI' and 'After "
                      "DSI'); each trace is drawn as the average of its sweeps, every sweep, or the most typical "
                      "sweep. The preview shows the figure's traces as they will be drawn.", "muted", wrap=True)
        outer.addWidget(intro)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # left: where the sweeps come from, the sweeps, the quick selections, add as a trace
        left = QtWidgets.QWidget()
        ll = box("v", "s")
        ll.addWidget(label("1. Sweeps", "heading"))
        self.source = QtWidgets.QComboBox()
        self.source.currentIndexChanged.connect(self._fill_sweeps)
        ll.addWidget(widgets.field("Pick sweeps from", self.source, "A recording of this project, or the DSI "
                                   "responses of the selected recording or cell", inline=True))
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._said_selection)
        _menu_on(self.table, self._sweeps_menu)
        ll.addWidget(self.table, 1)
        self.picked = label("", "muted")
        ll.addWidget(self.picked)
        # quick selections: by time for a recording, by episode for DSI
        self.by_time = QtWidgets.QWidget()
        bt = box("h", "s")
        self.t_from, self.t_to = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for s in (self.t_from, self.t_to):
            s.setRange(0, 1e6)
            s.setDecimals(1)
            s.setSuffix(" s")
        bt.addWidget(label("Select sweeps starting from", "muted"))
        bt.addWidget(self.t_from)
        bt.addWidget(label("to", "muted"))
        bt.addWidget(self.t_to)
        bt.addWidget(widgets.button("Select", self._select_time, "Select the sweeps that start in this time range"))
        bt.addStretch(1)
        self.by_time.setLayout(bt)
        ll.addWidget(self.by_time)
        self.by_episode = QtWidgets.QWidget()
        be = box("h", "s")
        self.episode = QtWidgets.QComboBox()
        self.episode.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.which = QtWidgets.QComboBox()
        self.which.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.which.addItem("baseline responses before the step", "before")
        self.which.addItem("responses after the step", "after")
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 100)
        self.count.setValue(4)
        for w_ in (label("Select in", "muted"), self.episode, label("the", "muted"), self.which,
                   label("count", "muted"), self.count):
            be.addWidget(w_)
        be.addWidget(widgets.button("Select", self._select_episode, "Select those responses in the table"))
        be.addStretch(1)
        self.by_episode.setLayout(be)
        ll.addWidget(self.by_episode)
        self.align = QtWidgets.QComboBox()
        self.align.addItem("the stimulus (pulse 1)", 1)
        self.align.addItem("the stimulus (pulse 2)", 2)
        self.align.addItem("the start of the sweep", 0)
        self.align_field = widgets.field("Align each sweep on", self.align, "Time 0 of each trace", inline=True)
        ll.addWidget(self.align_field)
        ll.addWidget(label("2. Add them as a trace", "heading"))
        add = box("h", "s")
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("Name, e.g. Before DSI")
        self.mode = QtWidgets.QComboBox()
        for k, text in traces.MODES.items():
            self.mode.addItem(text, k)
        add.addWidget(self.name, 1)
        add.addWidget(self.mode)
        add.addWidget(widgets.primary("Add as a trace", self._add, "Add the selected sweeps as one named trace"))
        ll.addLayout(add)
        left.setLayout(ll)
        split.addWidget(left)

        # right: the traces, the preview, the window
        right = QtWidgets.QWidget()
        rl = box("v", "s")
        rl.addWidget(label("3. The figure's traces", "heading"))
        self.list = QtWidgets.QTableWidget(0, 3)
        self.list.setHorizontalHeaderLabels(["Trace", "Drawn as", "Sweeps"])
        self.list.verticalHeader().setVisible(False)
        self.list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.list.setMaximumHeight(170)
        self.list.itemChanged.connect(self._renamed)
        _menu_on(self.list, self._traces_menu)
        QtWidgets.QShortcut(QtGui.QKeySequence.Delete, self.list, self._remove,
                            context=QtCore.Qt.WidgetWithChildrenShortcut)
        rl.addWidget(self.list)
        row = box("h", "s")
        for text, tip, slot in (("Remove trace", "Remove the selected trace (Delete)", self._remove),
                                ("Up", "Draw it earlier", lambda: self._move(-1)),
                                ("Down", "Draw it later", lambda: self._move(1)),
                                ("Show its sweeps", "Select the selected trace's sweeps on the left",
                                 self._show_sweeps)):
            row.addWidget(widgets.button(text, slot, tip, quiet=True))
        row.addStretch(1)
        rl.addLayout(row)
        row = box("h", "s")
        for text, tip, slot in (("+ Add selected sweeps to it", "Add the sweeps selected on the left to the selected "
                                 "trace", lambda: self._change_items(True)),
                                ("Remove selected sweeps from it", "Take the sweeps selected on the left out of the "
                                 "selected trace", lambda: self._change_items(False))):
            row.addWidget(widgets.button(text, slot, tip, quiet=True))
        row.addStretch(1)
        rl.addLayout(row)
        self.plot = pg.PlotWidget()
        theme.style_plot(self.plot, self.window_.palette)
        self.plot.addLegend(offset=(10, 10))
        rl.addWidget(self.plot, 1)
        win = box("h", "s")
        self.before, self.after, self.blank = (QtWidgets.QDoubleSpinBox() for _ in range(3))
        for s, lo, hi, v, suffix in ((self.before, 0, 1e5, self.w.before_ms, " ms"),
                                     (self.after, 1, 1e6, self.w.after_ms, " ms"),
                                     (self.blank, 0, 1e3, self.w.blank_ms, " ms")):
            s.setRange(lo, hi)
            s.setDecimals(1)
            s.setValue(v)
            s.setSuffix(suffix)
            s.valueChanged.connect(self._window_changed)
        self.blank.setSpecialValueText("none")
        self.baseline = QtWidgets.QCheckBox("Subtract each trace's baseline")
        self.baseline.setChecked(self.w.baseline)
        self.baseline.toggled.connect(self._window_changed)
        for w_ in (label("Show from", "muted"), self.before, label("before to", "muted"), self.after,
                   label("after time 0; blank the artifact for", "muted"), self.blank):
            win.addWidget(w_)
        win.addStretch(1)
        rl.addLayout(win)
        rl.addWidget(self.baseline)
        right.setLayout(rl)
        split.addWidget(right)
        split.setSizes([620, 620])
        outer.addWidget(split, 1)
        buttons = box("h", "s")
        buttons.addStretch(1)
        buttons.addWidget(widgets.button("Cancel", self.reject))
        self.ok = widgets.primary("Use these traces", self.accept, "Draw these traces in the figure's panel")
        buttons.addWidget(self.ok)
        outer.addLayout(buttons)
        widgets.resizable(self, outer, "trace_picker", (1250, 780))

    # ---------------------------------------------------------------- the sweeps
    def _sources(self) -> None:
        w = self.window_
        p = w.project
        self.source.blockSignals(True)
        self.source.clear()
        selected = getattr(w, "selected", None)
        if p is not None and selected is not None:
            cell = p.cell_result(selected)
            run = p.last_run(selected, w.recording(selected))
            for key, res in ((("dsi_cell", cell.results.get("dsi_cell")) if cell is not None else ("", None)),
                             ("dsi", run.results.get("dsi") if run is not None else None)):
                if res is not None and res.tables.get("responses"):
                    where = "this cell's files" if key == "dsi_cell" else selected.name
                    self.source.addItem(f"DSI responses: {where}", ("dsi", str(selected), key))
        for r in (p.recordings() if p is not None else []):
            self.source.addItem(r.name, ("file", str(r.path), ""))
        if selected is not None:
            k = next((i for i in range(self.source.count()) if self.source.itemData(i)[0] == "dsi"), -1)
            if k < 0:
                k = next((i for i in range(self.source.count()) if self.source.itemData(i)[1] == str(selected)), 0)
            self.source.setCurrentIndex(max(0, k))
        self.source.blockSignals(False)
        self._fill_sweeps()

    def _fill_sweeps(self, *_args) -> None:
        data = self.source.currentData()
        self.table.setRowCount(0)
        self._rows = []
        self._roles = []
        if not data:
            return
        kind, path, key = data
        dsi = kind == "dsi"
        self.by_time.setVisible(not dsi)
        self.by_episode.setVisible(dsi)
        self.align_field.setVisible(not dsi)
        w = self.window_
        if dsi:
            sel = Path(path)
            res = (w.project.cell_result(sel).results.get("dsi_cell") if key == "dsi_cell" else
                   w.project.last_run(sel, w.recording(sel)).results.get("dsi"))
            self._roles = traces.dsi_roles(res)
            files = {r.name: str(r.path) for r in w.project.recordings()}
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(["File", "Sweep", "Pulse", "Time (s)", "Where"])
            self.table.setRowCount(len(self._roles))
            for i, r in enumerate(self._roles):
                file = files.get(r["file"], r["file"]) if r["file"] else path
                self._rows.append([file, r["sweep"], r["pulse"]])
                for j, v in enumerate((Path(file).name, r["sweep"] + 1, r["pulse"], f"{r['time_s']:.1f}",
                                       r["label"] or "not around a step")):
                    cell = QtWidgets.QTableWidgetItem(str(v))
                    cell.setToolTip(r["label"] or "")
                    self.table.setItem(i, j, cell)
            episodes = sorted({k for r in self._roles for k, _ in r["roles"]})
            self.episode.clear()
            self.episode.addItem("every episode", 0)
            for k in episodes:
                self.episode.addItem(f"episode {k}", k)
            if not self.blank.value():
                self.blank.setValue(traces.dsi_blank(self._roles))
        else:
            rec = w.recording(Path(path))
            starts = list(getattr(rec, "sweep_start_times_s", []) or [])
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["Sweep", "Starts at (s)", "Stimulus"])
            self.table.setRowCount(rec.n_sweeps)
            for k in range(rec.n_sweeps):
                self._rows.append([path, k, None])
                at = traces.stimulus_s(rec, k, 1)
                start = starts[k] if k < len(starts) else np.nan
                for j, v in enumerate((k + 1, f"{start:.1f}" if np.isfinite(start) else "",
                                       f"pulse 1 at {1e3 * at:.1f} ms" if np.isfinite(at) else "none found")):
                    self.table.setItem(k, j, QtWidgets.QTableWidgetItem(str(v)))
            if starts:
                self.t_from.setValue(0.0)
                self.t_to.setValue(float(max(starts)))
            has_stim = any(np.isfinite(traces.stimulus_s(rec, k, 1)) for k in range(min(rec.n_sweeps, 5)))
            self.align.setCurrentIndex(self.align.findData(1 if has_stim else 0))
        self.table.resizeColumnsToContents()
        self._said_selection()

    def _selected_items(self) -> list[list]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        out = []
        for r in rows:
            f, sweep, pulse = self._rows[r]
            out.append([f, sweep, self.align.currentData() if pulse is None else pulse])
        return out

    def _said_selection(self) -> None:
        n = len({i.row() for i in self.table.selectedIndexes()})
        self.picked.setText(f"{n} sweep{'s' if n != 1 else ''} selected (Ctrl or Shift + click to select several)")

    def _select_rows(self, rows: list[int]) -> None:
        self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for r in rows:
            self.table.selectRow(r)
        self.table.setSelectionMode(mode)
        if rows:
            self.table.scrollToItem(self.table.item(rows[0], 0))

    def _select_time(self) -> None:
        a, b = self.t_from.value(), self.t_to.value()
        rec = self.window_.recording(Path(self._rows[0][0])) if self._rows else None
        starts = list(getattr(rec, "sweep_start_times_s", []) or []) if rec is not None else []
        self._select_rows([k for k, s in enumerate(starts) if a <= s <= b])

    def _select_episode(self) -> None:
        n = self.count.value()
        before, after = (n, 0) if self.which.currentData() == "before" else (0, n)
        self._select_rows(traces.dsi_indices(self._roles, self.episode.currentData() or 0, before, after))
        if not self.name.text().strip():
            ep = self.episode.currentText()
            self.name.setText(("Before the step" if before else "After the step") +
                              ("" if self.episode.currentData() == 0 else f" ({ep})"))

    # ---------------------------------------------------------------- the traces
    def _add(self) -> None:
        items = self._selected_items()
        if not items:
            QtWidgets.QMessageBox.information(self, "No sweeps selected", "Select one or more sweeps in the table "
                                              "first (Ctrl or Shift + click, or a Select button).")
            return
        name = self.name.text().strip() or f"Trace {len(self.sets) + 1}"
        self.sets.append(traces.TraceSet(name, self.mode.currentData(), items))
        self.name.clear()
        self._fill_sets()
        self.timer.start()

    def _fill_sets(self) -> None:
        self.list.blockSignals(True)
        self.list.setRowCount(len(self.sets))
        for i, s in enumerate(self.sets):
            self.list.setItem(i, 0, QtWidgets.QTableWidgetItem(s.name))
            mode = QtWidgets.QComboBox()
            for k, text in traces.MODES.items():
                mode.addItem(text, k)
            mode.setCurrentIndex(max(0, mode.findData(s.mode)))
            mode.currentIndexChanged.connect(lambda _i, s=s, c=mode: self._mode(s, c))
            self.list.setCellWidget(i, 1, mode)
            n = QtWidgets.QTableWidgetItem(str(len(s.items)))
            n.setFlags(n.flags() & ~QtCore.Qt.ItemIsEditable)
            self.list.setItem(i, 2, n)
        self.list.blockSignals(False)
        self.ok.setEnabled(bool(self.sets))

    def _mode(self, s: traces.TraceSet, combo) -> None:
        s.mode = combo.currentData()
        self.timer.start()

    def _renamed(self, item) -> None:
        if item.column() == 0 and 0 <= item.row() < len(self.sets):
            self.sets[item.row()].name = item.text().strip() or self.sets[item.row()].name
            self.timer.start()

    def _remove(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self.sets):
            del self.sets[i]
            self._fill_sets()
            self.timer.start()

    def _move(self, d: int) -> None:
        i = self.list.currentRow()
        j = i + d
        if 0 <= i < len(self.sets) and 0 <= j < len(self.sets):
            self.sets[i], self.sets[j] = self.sets[j], self.sets[i]
            self._fill_sets()
            self.list.selectRow(j)
            self.timer.start()

    def _change_items(self, add: bool, i: int | None = None) -> None:
        """The sweeps selected on the left added to, or taken out of, trace ``i`` (the selected one)."""
        i = self.list.currentRow() if i is None else i
        if not 0 <= i < len(self.sets):
            QtWidgets.QMessageBox.information(self, "No trace selected", "Select a trace in the list first.")
            return
        picked = self._selected_items()
        if not picked:
            QtWidgets.QMessageBox.information(self, "No sweeps selected", "Select sweeps in the table on the left.")
            return
        s = self.sets[i]
        key = lambda it: (str(it[0]), int(it[1]), int(it[2]))     # noqa: E731
        have = {key(it) for it in s.items}
        if add:
            s.items += [it for it in picked if key(it) not in have]
        else:
            gone = {key(it) for it in picked}
            s.items = [it for it in s.items if key(it) not in gone]
        if not s.items:
            del self.sets[i]
        self._fill_sets()
        self.timer.start()

    def _traces_menu(self, m: QtWidgets.QMenu, pos) -> None:
        i = self.list.rowAt(pos.y())
        if i < 0:
            return
        self.list.selectRow(i)
        m.addAction("Rename", lambda: self.list.editItem(self.list.item(i, 0)))
        drawn = m.addMenu("Draw as")
        for k, text in traces.MODES.items():
            a = drawn.addAction(text, lambda k=k: self.list.cellWidget(i, 1).setCurrentIndex(
                self.list.cellWidget(i, 1).findData(k)))
            a.setCheckable(True)
            a.setChecked(self.sets[i].mode == k)
        m.addAction("Show its sweeps", self._show_sweeps)
        m.addAction("Add the selected sweeps to it", lambda: self._change_items(True, i))
        m.addAction("Remove the selected sweeps from it", lambda: self._change_items(False, i))
        m.addSeparator()
        m.addAction("Move up", lambda: self._move(-1))
        m.addAction("Move down", lambda: self._move(1))
        m.addAction("Remove this trace", self._remove)

    def _sweeps_menu(self, m: QtWidgets.QMenu, pos) -> None:
        r = self.table.rowAt(pos.y())
        if r >= 0 and r not in {i.row() for i in self.table.selectedIndexes()}:
            self.table.selectRow(r)
        m.addAction("Add the selected sweeps as a new trace", self._add)
        if self.sets:
            to, out = m.addMenu("Add them to the trace"), m.addMenu("Remove them from the trace")
            for k, s in enumerate(self.sets):
                to.addAction(s.name, lambda k=k: self._change_items(True, k))
                out.addAction(s.name, lambda k=k: self._change_items(False, k))
        m.addSeparator()
        m.addAction("Select every sweep", self.table.selectAll)
        m.addAction("Clear the selection", self.table.clearSelection)

    def _show_sweeps(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self.sets):
            want = {(str(f), int(s)) for f, s, *_ in self.sets[i].items}
            self._select_rows([k for k, (f, s, _p) in enumerate(self._rows) if (str(f), int(s)) in want
                               or ("", int(s)) in want])

    # ---------------------------------------------------------------- the window and the preview
    def trace_window(self) -> traces.Window:
        return traces.Window(self.before.value(), self.after.value(), self.baseline.isChecked(), self.blank.value())

    def _window_changed(self, *_args) -> None:
        self.timer.start()

    def _preview(self) -> None:
        self.plot.clear()
        item = self.plot.getPlotItem()
        legend = item.addLegend(offset=(10, 10))
        legend.clear()
        if not self.sets:
            item.setLabel("bottom", "")
            item.setLabel("left", "")
            return
        frame, _notes, _markers = traces.build(self.sets, self.load, self.trace_window())
        colours = theme.series(self.window_.palette)
        for k, s in enumerate(self.sets):
            mine = frame[frame.Trace == s.name]
            for j, (_, c) in enumerate(mine.groupby("Sweep", sort=False)):
                many = mine.Sweep.nunique() > 1
                item.plot(c["Time (ms)"].to_numpy(float), c.iloc[:, 1].to_numpy(float), connect="finite",
                          pen=pg.mkPen(colours[k % len(colours)], width=0.8 if many else 1.6),
                          name=s.name if j == 0 else None)
        item.setLabel("bottom", "Time (ms)")
        item.setLabel("left", frame.columns[1])
        theme.keep_in_frame(item)
