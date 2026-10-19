"""The figure builder's window (B12): start from the question, and every control
says what it does.

Above, in groups: where the data come from, undo and redo, the panels and the
figure's size, styles, layouts, the Methods text, and the one primary action
(Export). A graph of the Figures screen opens here through its Edit button
(``editable``), so the automatic graphs are edited like any other figure.
Left: "What do you want to show?", then tabs: Data (which table -> which rows
-> which time window -> bins -> baseline), Plot (every chart type; any can be
tried), Groups (each group's colour, marker, line, name and order), Axes,
Theme (a whole look in one click, then every part of it) and Marks; only what
the question needs is open. Right, the largest: the figure as it will export
(matplotlib), and under it "What this figure shows", the warnings and the
statistics. Everything a control sets lives in the figure's ``Spec``
(``clampwise.figure``), so undo, saving and reopening are the same thing.
"""

from __future__ import annotations

import copy
import html
import json
from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import data as fdata
from ..figure import layouts, methods, presets, render, traces
from ..figure.prepare import OPERATORS
from ..figure.recipe import (CONTROLS, ERROR_STYLES, FITS, LINES, MARKERS, PALETTES, PLOT_QUESTION, PLOTS,
                             QUESTION, QUESTIONS, SIZES, SUMMARIES, THEMES, Recipe, Spec, Style, pack)
from . import theme, widgets
from .widgets import box, label

SCREEN_DPI = 110
LEGENDS = ("best", "upper left", "upper right", "lower left", "lower right", "outside", "none")
BARS = ("bar", "hbar", "grouped_bar", "count_bar", "stacked_bar")
UNITS = ("cell", "episode", "slice", "animal", "recording", "sweep")
EVERY = {q[0] for q in QUESTIONS}
# the controls each question needs open (the rest are hidden: B0 item 8)
SHOWN = {
    "x": {"time", "pulses", "relationship", "example", "counts"}, "y": EVERY - {"counts"}, "y2": {"relationship"},
    "show_points": {"compare"}, "percent": {"counts"},
    "group": EVERY, "cell": EVERY - {"example"}, "unit": EVERY - {"example"}, "animal": EVERY - {"example"},
    "panel_by": EVERY - {"example"},
    "filters": EVERY, "window": {"time", "pulses", "relationship"}, "bin": {"time"}, "normalize": {"time"},
    "level": EVERY - {"example"}, "summary": {"compare", "time", "pulses", "relationship"},
    "paired": {"compare"}, "fit": {"relationship"}, "stats": {"compare"}, "scale_bars": {"example"},
    "err": {"relationship"},
}


def _vline() -> QtWidgets.QFrame:
    """A thin divider between groups of the toolbar."""
    f = QtWidgets.QFrame()
    f.setObjectName("vline")
    f.setFrameShape(QtWidgets.QFrame.VLine)
    return f


def _float(text: str):
    try:
        return float(text.strip().replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _pair(text: str) -> list:
    """'0, 10' or '0 to 10' -> [0.0, 10.0]; anything else -> []."""
    parts = [p for p in text.replace(" to ", ",").replace(";", ",").split(",") if p.strip()]
    vals = [_float(p) for p in parts]
    return vals if len(vals) == 2 and None not in vals else []


def _pair_text(v: list) -> str:
    return f"{v[0]:g}, {v[1]:g}" if len(v) == 2 else ""


def context_menu(widget: QtWidgets.QWidget, fill) -> None:
    """A right-click menu on ``widget``: ``fill(menu, pos)`` adds its actions."""
    widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def show(pos):
        m = QtWidgets.QMenu(widget)
        fill(m, pos)
        if not m.isEmpty():
            m.exec_(widget.mapToGlobal(pos))

    widget.customContextMenuRequested.connect(show)


class ItemTable(QtWidgets.QWidget):
    """A list the figure is given, one row each (reference lines, marks along x, text on the figure): added with
    "+ Add", removed with "Remove selected", the right-click menu or the Delete key, edited in place.

    ``columns``: (key, header, kind), kind "number", "optional number" (blank allowed), "text", or a dict of
    choices {value: words}. ``new``: the row an Add starts from."""

    changed = QtCore.Signal()

    def __init__(self, columns: list[tuple], noun: str, new: dict) -> None:
        super().__init__()
        self.columns, self.noun, self.new = columns, noun, new
        self._loading = False
        lay = box("v", "xs")
        self.table = QtWidgets.QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels([h for _k, h, _t in columns])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(90)
        self.table.setMaximumHeight(170)
        self.table.itemChanged.connect(self._emit)
        lay.addWidget(self.table)
        row = box("h", "s")
        self.add_button = widgets.button(f"+ Add {noun}", self.add, f"Add a {noun} to the figure", quiet=True)
        self.remove_button = widgets.button("Remove selected", self.remove_selected,
                                            f"Remove the selected {noun}(s) from the figure (Delete)", quiet=True)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        lay.addLayout(row)
        self.setLayout(lay)
        QtWidgets.QShortcut(QtGui.QKeySequence.Delete, self.table, self.remove_selected,
                            context=QtCore.Qt.WidgetWithChildrenShortcut)
        context_menu(self.table, self._menu)
        self._enable()

    def _menu(self, m: QtWidgets.QMenu, pos) -> None:
        r = self.table.rowAt(pos.y())
        if r >= 0 and r not in self._selected():
            self.table.selectRow(r)
        m.addAction(f"Add a {self.noun}", self.add)
        if self._selected():
            m.addAction(f"Duplicate this {self.noun}", self.duplicate)
            m.addAction(f"Remove the selected {self.noun}(s)", self.remove_selected)
        if self.table.rowCount():
            m.addSeparator()
            m.addAction(f"Remove every {self.noun}", self.clear)

    def _selected(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _enable(self) -> None:
        self.remove_button.setEnabled(self.table.rowCount() > 0)

    def _emit(self, *_args) -> None:
        if not self._loading:
            self._enable()
            self.changed.emit()

    def _put(self, i: int, d: dict) -> None:
        for j, (key, _h, kind) in enumerate(self.columns):
            v = d.get(key)
            if isinstance(kind, dict):
                c = QtWidgets.QComboBox()
                for k, words in kind.items():
                    c.addItem(words, k)
                c.setCurrentIndex(max(0, c.findData(v)))
                c.currentIndexChanged.connect(self._emit)
                self.table.setCellWidget(i, j, c)
            else:
                text = "" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(text))

    def set_value(self, rows: list[dict]) -> None:
        self._loading = True
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))
        for i, d in enumerate(rows):
            self._put(i, d)
        self._loading = False
        self._enable()

    def value(self) -> list[dict]:
        """The rows complete enough to draw (a row missing a required number is left out)."""
        return [d for d, ok in self._all() if ok]

    def _all(self) -> list[tuple[dict, bool]]:
        out = []
        for i in range(self.table.rowCount()):
            d, ok = {}, True
            for j, (key, _h, kind) in enumerate(self.columns):
                if isinstance(kind, dict):
                    d[key] = self.table.cellWidget(i, j).currentData()
                    continue
                text = self.table.item(i, j).text() if self.table.item(i, j) else ""
                if kind == "text":
                    d[key] = text.strip()
                else:
                    v = _float(text)
                    if v is None and kind == "number":
                        ok = False
                    d[key] = v
            out.append((d, ok))
        return out

    def add(self) -> None:
        rows = [d for d, _ok in self._all()] + [dict(self.new)]
        self.set_value(rows)
        self.table.selectRow(len(rows) - 1)
        self.table.editItem(self.table.item(len(rows) - 1, next(
            (j for j, (_k, _h, kind) in enumerate(self.columns) if not isinstance(kind, dict)), 0)))
        self.changed.emit()

    def duplicate(self) -> None:
        rows = [d for d, _ok in self._all()]
        keep = [rows[i] for i in self._selected() if i < len(rows)]
        self.set_value(rows + [dict(d) for d in keep])
        self.changed.emit()

    def remove_selected(self) -> None:
        gone = set(self._selected())
        if not gone:
            return
        self.set_value([d for i, (d, _ok) in enumerate(self._all()) if i not in gone])
        self.changed.emit()

    def clear(self) -> None:
        self.set_value([])
        self.changed.emit()


class ColourButton(QtWidgets.QPushButton):
    """A colour, shown as a swatch with its hex code; a click opens the colour chooser."""

    changed = QtCore.Signal()

    def __init__(self, colour: str = "", compact: bool = False) -> None:
        super().__init__()
        self.colour = ""
        self.compact = compact                  # a swatch only (in a table), its code in the tooltip
        self.clicked.connect(self._pick)
        self.set_colour(colour)

    def set_colour(self, colour: str) -> None:
        self.colour = colour or ""
        pix = QtGui.QPixmap(14, 14)
        pix.fill(QtGui.QColor(self.colour) if self.colour else QtGui.QColor(0, 0, 0, 0))
        self.setIcon(QtGui.QIcon(pix))
        self.setText("" if self.compact else (self.colour or "the theme's"))
        self.setToolTip(f"{self.colour or 'The theme' + chr(39) + 's colour'}: click to choose another")

    def _pick(self) -> None:
        start = QtGui.QColor(self.colour) if self.colour else QtGui.QColor(QtCore.Qt.black)
        c = QtWidgets.QColorDialog.getColor(start, self, "Choose a colour")
        if c.isValid():
            self.set_colour(c.name())
            self.changed.emit()


class SeriesTable(QtWidgets.QWidget):
    """Each group's look (colour, marker, line, width, filled, its name in the legend,
    shown or not) and their order. Only what differs from the theme is kept, so a new
    theme still recolours whatever was left to it."""

    changed = QtCore.Signal()
    HEAD = ("Group", "Show", "Colour", "Marker", "Line", "Width", "Filled", "Legend name")

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []
        self._defaults: dict = {}
        self._loading = False
        lay = box("v", "s")
        self.table = QtWidgets.QTableWidget(0, len(self.HEAD))
        self.table.setHorizontalHeaderLabels(self.HEAD)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(180)
        context_menu(self.table, self._menu)
        lay.addWidget(self.table)
        row = box("h", "s")
        for text, tip, slot in (("Up", "Draw the selected group earlier", lambda: self._move(-1)),
                                ("Down", "Draw it later", lambda: self._move(1)),
                                ("Hide from the figure", "Leave the selected group out of the figure (and its "
                                 "numbers); Show brings it back", lambda: self._show(False)),
                                ("Show", "Put the selected group back in the figure", lambda: self._show(True)),
                                ("Reset to the theme", "Every group's look from the theme again", self._reset)):
            row.addWidget(widgets.button(text, slot, tip, quiet=True))
        row.addStretch(1)
        lay.addLayout(row)
        self.setLayout(lay)

    def set_rows(self, names: list[str], series: dict, defaults: dict) -> None:
        self._loading = True
        self._names, self._defaults = list(names), defaults
        shown = {"colour", "marker", "line", "width", "filled", "label", "hidden"}
        self._extra = {n: {k: v for k, v in series.get(n, {}).items() if k not in shown} for n in names}
        self.table.setRowCount(len(names))
        for i, name in enumerate(names):
            d, look = series.get(name, {}), defaults.get(name)
            item = QtWidgets.QTableWidgetItem(name or "(all)")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(i, 0, item)
            colour = ColourButton(d.get("colour") or (look.colour if look else ""), compact=True)
            colour.changed.connect(self._emit)
            marker, line = QtWidgets.QComboBox(), QtWidgets.QComboBox()
            for c in (marker, line):
                c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
                c.setMinimumContentsLength(5)
            for k, text in MARKERS.items():
                marker.addItem(text, k)
            for k, text in LINES.items():
                line.addItem(text, k)
            marker.setCurrentIndex(max(0, marker.findData(d.get("marker") or (look.marker if look else "o"))))
            line.setCurrentIndex(max(0, line.findData(d.get("line") if d.get("line") is not None else
                                                     (look.line if look else "-"))))
            width = QtWidgets.QDoubleSpinBox()
            width.setMaximumWidth(64)
            colour.setMaximumWidth(40)
            width.setRange(0.2, 6.0)
            width.setSingleStep(0.1)
            width.setValue(float(d.get("width") or (look.width if look else 1.0)))
            filled, shown = QtWidgets.QCheckBox(), QtWidgets.QCheckBox()
            filled.setChecked(bool(d.get("filled", True)))
            shown.setChecked(not d.get("hidden"))
            name_edit = QtWidgets.QLineEdit(d.get("label", ""))
            name_edit.setPlaceholderText(name or "(no name)")
            for w, sig in ((marker, marker.currentIndexChanged), (line, line.currentIndexChanged),
                           (width, width.valueChanged), (filled, filled.toggled), (shown, shown.toggled),
                           (name_edit, name_edit.editingFinished)):
                sig.connect(self._emit)
            for j, w in enumerate((shown, colour, marker, line, width, filled, name_edit), start=1):
                self.table.setCellWidget(i, j, w)
        self._loading = False

    def value(self) -> tuple[dict, list]:
        series = {}
        for i, name in enumerate(self._names):
            look = self._defaults.get(name)
            shown, colour, marker, line, width, filled, name_edit = (self.table.cellWidget(i, j) for j in range(1, 8))
            d = {}
            if colour.colour and (look is None or colour.colour.lower() != look.colour.lower()):
                d["colour"] = colour.colour
            if look is None or marker.currentData() != look.marker:
                d["marker"] = marker.currentData()
            if look is None or line.currentData() != look.line:
                d["line"] = line.currentData()
            if look is None or abs(width.value() - look.width) > 1e-9:
                d["width"] = width.value()
            if not filled.isChecked():
                d["filled"] = False
            if name_edit.text().strip():
                d["label"] = name_edit.text().strip()
            if not shown.isChecked():
                d["hidden"] = True
            d.update(getattr(self, "_extra", {}).get(name, {}))     # what this table does not show (bars) is kept
            if d:
                series[name] = d
        return series, list(self._names)

    def _emit(self, *_args) -> None:
        if not self._loading:
            self.changed.emit()

    def _show(self, on: bool, row: int | None = None) -> None:
        i = self.table.currentRow() if row is None else row
        if 0 <= i < len(self._names):
            self.table.cellWidget(i, 1).setChecked(on)

    def _menu(self, m: QtWidgets.QMenu, pos) -> None:
        i = self.table.rowAt(pos.y())
        if i < 0:
            return
        self.table.selectRow(i)
        shown = self.table.cellWidget(i, 1).isChecked()
        m.addAction("Hide from the figure" if shown else "Show in the figure", lambda: self._show(not shown, i))
        m.addAction("Move up", lambda: self._move(-1))
        m.addAction("Move down", lambda: self._move(1))
        m.addSeparator()
        m.addAction("Reset every group's look", self._reset)

    def _move(self, d: int) -> None:
        i = self.table.currentRow()
        j = i + d
        if i < 0 or not 0 <= j < len(self._names):
            return
        series, _ = self.value()
        self._names[i], self._names[j] = self._names[j], self._names[i]
        self.set_rows(self._names, series, self._defaults)
        self.table.selectRow(j)
        self.changed.emit()

    def _reset(self) -> None:
        self.set_rows(self._names, {n: dict(e) for n, e in self._extra.items() if e}, self._defaults)
        self.changed.emit()


class FilterList(QtWidgets.QWidget):
    """Keep only rows where...: one line per condition, all of them must hold."""

    changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple] = []
        self.columns: list[str] = []
        self.lay = box("v", "xs")
        self.add_button = widgets.button("Add a condition", lambda: self.add(), "Another condition the rows must meet",
                                         quiet=True)
        self.lay.addWidget(self.add_button)
        self.setLayout(self.lay)

    def set_columns(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def add(self, col: str = "", op: str = "is", value: str = "", emit: bool = True) -> None:
        row = QtWidgets.QWidget()
        h = box("h", "xs")
        c, o, v = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QLineEdit()
        c.addItems(self.columns)
        c.setCurrentText(col)
        o.addItems(OPERATORS)
        o.setCurrentText(op)
        v.setText(str(value))
        v.setPlaceholderText("value")
        x = QtWidgets.QToolButton()
        x.setText("✕")
        x.setToolTip("Remove this condition")
        for w in (c, o):
            w.currentIndexChanged.connect(self.changed)
        v.editingFinished.connect(self.changed)
        x.clicked.connect(lambda _=False, r=row: self._remove(r))
        for w in (c, o, v, x):
            h.addWidget(w, 1 if w in (c, v) else 0)
        row.setLayout(h)
        context_menu(row, lambda m, _pos, r=row: (m.addAction("Remove this condition", lambda: self._remove(r)),
                                                   m.addAction("Add a condition", lambda: self.add()),
                                                   m.addAction("Remove every condition", self.clear)))
        self.rows.append((row, c, o, v))
        self.lay.insertWidget(self.lay.count() - 1, row)
        if emit:
            self.changed.emit()

    def _remove(self, row) -> None:
        self.rows = [r for r in self.rows if r[0] is not row]
        row.deleteLater()
        self.changed.emit()

    def clear(self) -> None:
        for row, *_ in self.rows:
            row.deleteLater()
        self.rows = []
        self.changed.emit()

    def value(self) -> list:
        return [[c.currentText(), o.currentText(), v.text()] for _, c, o, v in self.rows if c.currentText()]

    def set_value(self, filters: list) -> None:
        for row, *_ in self.rows:
            row.deleteLater()
        self.rows = []
        for col, op, value in filters:
            self.add(col, op, value, emit=False)


class DigitizeDialog(QtWidgets.QDialog):
    """A table from a picture of a plot: two known values on each axis, then a
    curve traced by its colour or points clicked. An estimate, and said so."""

    STEPS = (("x", 1, "Click a point on the x axis whose value you know"),
             ("x", 2, "Click a second point on the x axis, far from the first"),
             ("y", 1, "Click a point on the y axis whose value you know"),
             ("y", 2, "Click a second point on the y axis, far from the first"))

    def __init__(self, parent, path: Path) -> None:
        super().__init__(parent)
        from ..figure import digitize

        self.dz = digitize
        pal = getattr(getattr(parent, "window_", None), "palette", None) or theme.LIGHT
        self.calibration_pen, self.point_pen = QtGui.QPen(QtGui.QColor(pal.accent)), QtGui.QPen(QtGui.QColor(pal.warn))
        self.setWindowTitle(f"Digitize {path.name}")
        self.image = digitize.load(path)
        self.clicks: list[tuple[float, float, float]] = []      # (column, row, value)
        self.points: list[tuple[float, float]] = []
        self._click_marks: list = []                # the scene's dots, so each can be taken off again
        self._point_marks: list = []
        self.mode = "calibrate"
        lay = box("v", "s", "m")
        self.say = label(self.STEPS[0][2], "heading", wrap=True)
        lay.addWidget(self.say)
        lay.addWidget(label("The table this makes is an estimate of the plotted numbers, as good as the picture.",
                            "muted", wrap=True))
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap(str(path)))
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.viewport().installEventFilter(self)
        lay.addWidget(self.view, 1)
        row = box("h", "s")
        self.log_x, self.log_y = QtWidgets.QCheckBox("Log x axis"), QtWidgets.QCheckBox("Log y axis")
        self.trace_button = widgets.button("Trace a curve by its colour", lambda: self._mode("trace"),
                                           "Then click on the curve: every column of that colour is read")
        self.click_button = widgets.button("Click points", lambda: self._mode("points"),
                                           "Then click each point to read it")
        self.names = QtWidgets.QLineEdit("x, y")
        self.names.setToolTip("The two columns' names, with units: e.g. Time (min), Slope (%)")
        for w in (self.log_x, self.log_y, self.trace_button, self.click_button, label("Columns:", "muted"),
                  self.names):
            row.addWidget(w)
        row.addStretch(1)
        self.done = widgets.primary("Use this table", self.accept, "Make the figure from the digitized points")
        row.addWidget(self.done)
        lay.addLayout(row)
        fix = box("h", "s")
        for text, tip, slot in (("Undo last click", "Take off the last point or calibration click (Ctrl+Z)",
                                 self.undo_last),
                                ("Clear points", "Take off every point read so far (the calibration stays)",
                                 self.clear_points),
                                ("Calibrate again", "Start over: take off the calibration and every point",
                                 self.restart)):
            fix.addWidget(widgets.button(text, slot, tip, quiet=True))
        fix.addWidget(label("Right-click a point to remove it.", "hint"))
        fix.addStretch(1)
        lay.addLayout(fix)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, self.undo_last)
        widgets.resizable(self, lay, "digitize", (900, 700))
        self._mode("calibrate")

    def _mode(self, mode: str) -> None:
        self.mode = mode
        ready = len(self.clicks) == 4
        self.trace_button.setEnabled(ready)
        self.click_button.setEnabled(ready)
        self.done.setEnabled(bool(self.points))
        if mode == "trace":
            self.say.setText("Click on the curve: the columns of its colour are read")
        elif mode == "points":
            self.say.setText("Click each point; they are read as you go")

    def calibration(self):
        (cx1, _, vx1), (cx2, _, vx2), (_, ry1, vy1), (_, ry2, vy2) = self.clicks
        return self.dz.Calibration(self.dz.Axis(cx1, vx1, cx2, vx2, self.log_x.isChecked()),
                                   self.dz.Axis(ry1, vy1, ry2, vy2, self.log_y.isChecked()))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonPress:
            p = self.view.mapToScene(event.pos())
            if event.button() == QtCore.Qt.RightButton:
                self._menu(p.x(), p.y(), event.globalPos())
            else:
                self.clicked(p.x(), p.y())
            return True
        return super().eventFilter(obj, event)

    def _menu(self, col: float, row: float, where) -> None:
        m = QtWidgets.QMenu(self)
        a = m.addAction("Remove the point nearest here", lambda: self.remove_nearest(col, row))
        a.setEnabled(bool(self.points))
        a = m.addAction("Undo the last click", self.undo_last)
        a.setEnabled(bool(self.points or self.clicks))
        a = m.addAction("Clear every point", self.clear_points)
        a.setEnabled(bool(self.points))
        m.addAction("Calibrate again", self.restart)
        m.exec_(where)

    def _drop(self, items) -> None:
        for it in items:
            if it.scene() is not None:
                self.scene.removeItem(it)

    def remove_nearest(self, col: float, row: float) -> None:
        if not self.points:
            return
        k = min(range(len(self.points)), key=lambda i: (self.points[i][0] - col) ** 2 + (self.points[i][1] - row) ** 2)
        del self.points[k]
        self._redraw_points()

    def _redraw_points(self) -> None:
        """The points' dots drawn again (a traced curve: every 4th column)."""
        self._drop(self._point_marks)
        every = 4 if self.mode == "trace" else 1
        self._point_marks = [self.scene.addEllipse(a - 2, b - 2, 4, 4, self.point_pen) for a, b in self.points[::every]]
        self.done.setEnabled(bool(self.points))

    def undo_last(self) -> None:
        if self.points:
            if self.mode == "trace":                # a traced curve comes off as a whole
                self.clear_points()
            else:
                self.points.pop()
                self._redraw_points()
        elif self.clicks:
            self.clicks.pop()
            self._drop([self._click_marks.pop()])
            self.mode = "calibrate"
            self.say.setText(self.STEPS[len(self.clicks)][2])
            self._mode("calibrate")
        self.done.setEnabled(bool(self.points))

    def clear_points(self) -> None:
        self.points = []
        self._drop(self._point_marks)
        self._point_marks = []
        self.done.setEnabled(False)

    def restart(self) -> None:
        self.clear_points()
        self.clicks = []
        self._drop(self._click_marks)
        self._click_marks = []
        self.say.setText(self.STEPS[0][2])
        self._mode("calibrate")

    def clicked(self, col: float, row: float, value: float | None = None) -> None:
        if self.mode == "calibrate":
            axis, k, _ = self.STEPS[len(self.clicks)]
            if value is None:
                value, ok = QtWidgets.QInputDialog.getDouble(self, "Known value", f"The value at this point of the "
                                                             f"{axis} axis:", 0.0, -1e12, 1e12, 6)
                if not ok:
                    return
            self.clicks.append((col, row, value))
            self._click_marks.append(self.scene.addEllipse(col - 3, row - 3, 6, 6, self.calibration_pen))
            if len(self.clicks) < 4:
                self.say.setText(self.STEPS[len(self.clicks)][2])
            else:
                self.say.setText("Calibrated: trace a curve by its colour, or click points")
                self._mode("ready")
        elif self.mode == "points":
            self.points.append((col, row))
            self._point_marks.append(self.scene.addEllipse(col - 2, row - 2, 4, 4, self.point_pen))
        elif self.mode == "trace":
            r, c = int(row), int(col)
            if 0 <= r < self.image.shape[0] and 0 <= c < self.image.shape[1]:
                cols, rows = self.dz.trace(self.image, self.image[r, c])
                self.clear_points()
                self.points = list(zip(cols, rows))
                self._point_marks = [self.scene.addEllipse(a - 1, b - 1, 2, 2, self.point_pen)
                                     for a, b in self.points[::4]]
                self.say.setText(f"{len(self.points)} columns read from the curve")
        self.done.setEnabled(bool(self.points))

    def table(self):
        names = [n.strip() for n in self.names.text().split(",")] + ["x", "y"]
        cols, rows = zip(*self.points)
        return self.dz.table(self.calibration(), cols, rows, names[0] or "x", names[1] or "y")


class Builder(QtWidgets.QMainWindow):
    def __init__(self, window, spec: Spec | None = None) -> None:
        super().__init__(window)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle("Figure builder – Clampwise")
        self.window_ = window
        self.spec = spec or Spec()
        self.current = 0
        self._loading = True                 # no edits fire while the controls are made
        self._history: list[str] = []
        self._future: list[str] = []
        self._drawn: list = []
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.redraw)
        self.typing = QtCore.QTimer(self)            # text boxes: applied a moment after typing stops
        self.typing.setSingleShot(True)
        self.typing.setInterval(700)
        self.typing.timeout.connect(self._edited)
        self.fields: dict[str, QtWidgets.QWidget] = {}
        self._sections: list[tuple[QtWidgets.QLabel, list[str]]] = []
        self._build()
        self._load_controls()
        self._snapshot()
        self.redraw()
        self.resize(1400, 900)

    # ---------------------------------------------------------------- the layout
    def _features(self) -> dict:
        p = getattr(self.window_, "project", None)
        return p.features() if p is not None else {}

    def _on(self, key: str) -> bool:
        return self._features().get(key, True)

    def _tool(self, text: str, tip: str, actions=None, slot=None) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        if actions is not None:
            b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            m = QtWidgets.QMenu(b)
            if callable(actions):
                m.aboutToShow.connect(lambda m=m: actions(m))
            else:
                for t, s in actions:
                    m.addAction(t, s)
            b.setMenu(m)
        if slot is not None:
            b.clicked.connect(slot)
        return b

    def _build(self) -> None:
        central = QtWidgets.QWidget()
        outer = box("v", "s")
        outer.setContentsMargins(0, 0, 0, theme.SPACE["s"])
        bar = QtWidgets.QWidget()
        bar.setObjectName("toolbar")
        top = box("h", "xs", "s")
        top.addWidget(self._tool("Open data", "Where the numbers come from", [
            ("Raw traces from the recordings (pick sweeps, average them)...", self.choose_traces),
            ("A table of this project...", self.open_project_table),
            ("A CSV or Excel file...", self.open_file),
            ("A saved figure (to edit it again)...", self.open_figure),
            ("A picture of a plot (digitize it)...", self.open_picture),
            ("Remove this panel's data", self.clear_data)]))
        top.addWidget(_vline())
        self.undo_button = self._tool("Undo", "Undo the last change (Ctrl+Z)", slot=self.undo)
        self.redo_button = self._tool("Redo", "Redo it (Ctrl+Y)", slot=self.redo)
        top.addWidget(self.undo_button)
        top.addWidget(self.redo_button)
        top.addWidget(_vline())
        top.addWidget(self._tool("Panels", "How many panels, and in how many columns", self._panels_menu))
        self.size = QtWidgets.QComboBox()
        self.size.setToolTip(CONTROLS["size"][1])
        self.size.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        for k, (w, h) in SIZES.items():
            self.size.addItem(f"{k[:1].upper() + k[1:]} column ({w:g} mm)", k)
        self.size.currentIndexChanged.connect(self._edited)
        top.addWidget(self.size)
        top.addWidget(_vline())
        top.addWidget(self._tool("Styles", "Save this look, or use a saved one", self._styles_menu))
        self.layouts_button = self._tool("Presets", "A whole paper figure from the selected recording or cells", self._layouts_menu)
        top.addWidget(self.layouts_button)
        self.methods_button = self._tool("Methods text", "A Methods paragraph for this figure", slot=self.show_methods)
        top.addWidget(self.methods_button)
        top.addStretch(1)
        top.addWidget(widgets.button("Save figure...", self.save_figure, "Keep it to edit again (.cwfig)"))
        top.addWidget(widgets.primary("Export...", self.export, "PDF, SVG, EPS, PNG or TIFF, at a journal's size"))
        bar.setLayout(top)
        outer.addWidget(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._controls())
        split.addWidget(self._preview())
        split.setStretchFactor(1, 3)
        split.setSizes([520, 1000])
        body = box("h", "s")
        body.setContentsMargins(theme.SPACE["m"], 0, theme.SPACE["m"], 0)
        body.addWidget(split)
        outer.addLayout(body, 1)
        central.setLayout(outer)
        self.setCentralWidget(central)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo),
                           ("Ctrl+S", self.save_figure), ("Ctrl+E", self.export), ("Ctrl+Return", self.apply)):
            QtWidgets.QShortcut(QtGui.QKeySequence(keys), self, slot)

    def _preview(self) -> QtWidgets.QWidget:
        """The figure as it will export, on a desk, with its size and a zoom; under it, the caption and warnings."""
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setChildrenCollapsible(False)
        top = QtWidgets.QWidget()
        tl = box("v", "xs")
        head = box("h", "s")
        head.addWidget(label("Preview", "heading"))
        self.size_note = label("", "muted")
        head.addWidget(self.size_note)
        head.addStretch(1)
        head.addWidget(label("Zoom", "muted"))
        self.zoom = QtWidgets.QComboBox()
        for text, f in (("Fit to window", 0.0), ("Print size", 1.0), ("150%", 1.5), ("200%", 2.0)):
            self.zoom.addItem(text, f)
        self.zoom.setToolTip("Fit the figure to the window, or show it at its printed size (and larger) to judge "
                             "the font sizes and line widths")
        self.zoom.currentIndexChanged.connect(lambda _i: self._fit_canvas())
        head.addWidget(self.zoom)
        tl.addLayout(head)
        self.fig = Figure(dpi=SCREEN_DPI)
        self.canvas = FigureCanvasQTAgg(self.fig)
        context_menu(self.canvas, self._canvas_menu)
        self.holder = QtWidgets.QScrollArea()
        self.holder.setObjectName("desk")
        self.holder.setAlignment(QtCore.Qt.AlignCenter)
        self.holder.setWidget(self.canvas)
        self.holder.setWidgetResizable(False)
        tl.addWidget(self.holder, 1)
        top.setLayout(tl)
        right.addWidget(top)
        bottom = QtWidgets.QWidget()
        bl = box("v", "xs")
        cap_head = box("h", "s")
        cap_head.addWidget(label("What this figure shows", "heading"))
        cap_head.addStretch(1)
        cap_head.addWidget(widgets.button("Copy caption", self.copy_caption, "Copy the caption for a figure legend",
                                          quiet=True))
        bl.addLayout(cap_head)
        self.caption = QtWidgets.QTextBrowser()
        self.caption.setMinimumHeight(60)
        bl.addWidget(self.caption, 1)
        self.warnings = QtWidgets.QWidget()
        self.warnings.setLayout(box("v", "xs"))
        bl.addWidget(self.warnings)
        bottom.setLayout(bl)
        right.addWidget(bottom)
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)
        right.setSizes([640, 150])
        return right

    def _f(self, key: str, control: QtWidgets.QWidget, into) -> QtWidgets.QWidget:
        caption, line, help_text = CONTROLS[key]
        wide = isinstance(control, (SeriesTable, FilterList, ItemTable))
        w = widgets.field(caption, control, line, caption, help_text, manual_key="figures", inline=not wide)
        self.fields[key] = w
        into.add(w) if isinstance(into, widgets.Advanced) else into.addWidget(w)
        if self._sections:
            self._sections[-1][1].append(key)
        return w

    def _section(self, lay: QtWidgets.QVBoxLayout, title: str) -> None:
        """A heading over the controls that follow it in a tab; hidden when all of them are."""
        head = label(title.upper(), "section")
        lay.addWidget(head)
        self._sections.append((head, []))

    def _combo(self) -> QtWidgets.QComboBox:
        c = QtWidgets.QComboBox()
        c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(16)              # long choices are cut, never widen the panel
        c.currentIndexChanged.connect(self._edited)
        return c

    def _line(self, placeholder: str = "") -> QtWidgets.QLineEdit:
        """A text box: the figure follows as you type (after a short pause), and at Enter."""
        e = QtWidgets.QLineEdit()
        e.setPlaceholderText(placeholder)
        e.editingFinished.connect(self._edited)
        e.textEdited.connect(lambda _t: self.typing.start())
        return e

    def apply(self) -> None:
        """Every control read now and the figure drawn now (the Apply button)."""
        self.typing.stop()
        self._edited()
        self.timer.stop()
        self.redraw()
        self.statusBar().showMessage("Applied.", 2000)

    def _check(self, text: str) -> QtWidgets.QCheckBox:
        c = QtWidgets.QCheckBox(text)
        c.toggled.connect(self._edited)
        return c

    def _tab(self, name: str) -> QtWidgets.QVBoxLayout:
        """A tab of controls, scrolling when long."""
        page = QtWidgets.QWidget()
        lay = box("v", "s", "m")
        page.setLayout(lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        self.sections.addTab(scroll, name)
        return lay

    def _spin(self, lo: float, hi: float, step: float, decimals: int = 1) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.valueChanged.connect(self._edited)
        return s

    def _choice(self, items: dict) -> QtWidgets.QComboBox:
        c = self._combo()
        for k, text in items.items():
            c.addItem(text, k)
        return c

    def _controls(self) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        outer = box("v", "s")
        outer.setContentsMargins(0, theme.SPACE["s"], theme.SPACE["s"], 0)
        outer.addWidget(label("What do you want to show?", "heading"))
        self.question_box = QtWidgets.QComboBox()
        for key, title, why, *_rest in QUESTIONS:
            self.question_box.addItem(title, key)
        self.question_box.setToolTip("The question sets up the standard plot for it; any other plot can be tried "
                                     "on the Plot tab")
        self.question_box.activated.connect(self._question_chosen)
        outer.addWidget(self.question_box)
        self.why = label("", "hint", wrap=True)
        outer.addWidget(self.why)
        strip = box("h", "s")
        strip.addWidget(label("Editing panel", "muted"))
        self.tabs = QtWidgets.QTabBar()
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setElideMode(QtCore.Qt.ElideNone)
        self.tabs.setToolTip("The panel the controls below change")
        self.tabs.currentChanged.connect(self._select_panel)
        strip.addWidget(self.tabs)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.remove_panel)
        context_menu(self.tabs, self._tab_menu)
        self.add_panel = widgets.button("+ Add panel", lambda: self.set_panels(len(self.spec.panels) + 1),
                                        "Add a panel, starting as a copy of this one", quiet=True)
        strip.addWidget(self.add_panel)
        self.remove_panel_button = widgets.button("Remove panel", lambda: self.remove_panel(self.current),
                                                  "Remove the panel being edited (its × does the same)", quiet=True)
        strip.addWidget(self.remove_panel_button)
        strip.addStretch(1)
        outer.addLayout(strip)
        self.sections = QtWidgets.QTabWidget()
        self.sections.setDocumentMode(True)
        outer.addWidget(self.sections, 1)
        foot = box("h", "s")
        foot.addWidget(label("Changes apply as you type, or with Enter.", "hint"))
        foot.addStretch(1)
        foot.addWidget(widgets.primary("Apply", self.apply, "Apply every change now and redraw the figure "
                                       "(Ctrl+Enter)"))
        outer.addLayout(foot)
        holder.setLayout(outer)
        holder.setMinimumWidth(480)            # the tabs and the Groups table whole

        # Data: which table -> which rows -> which time window -> bins -> baseline
        lay = self._tab("Data")
        self.source = label("No table yet: Open data, above.", "muted", wrap=True)
        lay.addWidget(self.source)
        self.file_box = QtWidgets.QWidget()
        fb = box("h", "s")
        self.sheet = QtWidgets.QComboBox()
        self.sheet_label = label("Sheet", "muted")
        self.header = QtWidgets.QSpinBox()
        self.skip = QtWidgets.QSpinBox()
        for w, tip in ((self.sheet, "The sheet"), (self.header, "The row holding the column names (from 1)"),
                       (self.skip, "Rows under the names to skip (units)")):
            w.setToolTip(tip)
        self.header.setRange(1, 100)
        self.skip.setRange(0, 20)
        for lab, w in ((self.sheet_label, self.sheet), (label("Names in row", "muted"), self.header),
                       (label("Rows to skip", "muted"), self.skip)):
            fb.addWidget(lab)
            fb.addWidget(w)
        fb.addStretch(1)
        self.sheet.currentIndexChanged.connect(self._reread)
        self.header.valueChanged.connect(self._reread)
        self.skip.valueChanged.connect(self._reread)
        self.file_box.setLayout(fb)
        lay.addWidget(self.file_box)
        tools = box("h", "s")
        self.reshape_button = widgets.button("Conditions are in separate columns...", self.reshape,
                                             "Make the columns of a wide table (one per condition) one column of "
                                             "values and one of conditions", quiet=True)
        tools.addWidget(self.reshape_button)
        self.example_button = widgets.button("Use the selected recording's typical response", self.use_example,
                                             "The sweep closest to the mean, artifact blanked, with scale bars")
        tools.addWidget(self.example_button)
        tools.addStretch(1)
        lay.addLayout(tools)
        self.traces_box = QtWidgets.QWidget()
        tb = box("h", "s")
        self.traces_said = label("", "", wrap=True)
        tb.addWidget(self.traces_said, 1)
        self.traces_button = widgets.primary("Choose traces...", self.choose_traces,
                                             "Pick the recording and the sweeps to show (for example the responses "
                                             "before and after a DSI step), and whether each is averaged")
        tb.addWidget(self.traces_button, 0, QtCore.Qt.AlignTop)
        self.traces_box.setLayout(tb)
        lay.addWidget(self.traces_box)
        self._section(lay, "Columns")
        self.roles = {}
        for key in ("y", "x", "group", "cell", "animal", "y2", "err"):
            self.roles[key] = self._combo()
            self._f(key, self.roles[key], lay)
            if key == "cell":
                self.unit = self._combo()
                self.unit.addItems(UNITS)
                self._f("unit", self.unit, lay)
        self._section(lay, "Rows")
        self.filters = FilterList()
        self.filters.changed.connect(self._edited)
        self._f("filters", self.filters, lay)
        self.window_edit = self._line("from, to")
        self._f("window", self.window_edit, lay)
        self._section(lay, "Averaging")
        self.bin_edit = self._line("e.g. 1")
        self._f("bin", self.bin_edit, lay)
        self.normalize_edit = self._line("from, to: e.g. 0, 10")
        self._f("normalize", self.normalize_edit, lay)
        self.level = self._combo()
        self.level.addItem("A cell (its rows averaged first)", "cell")
        self.level.addItem("A row of the table", "row")
        self._f("level", self.level, lay)
        lay.addStretch(1)

        # Plot: every chart type, the summary and what is drawn
        lay = self._tab("Plot")
        self._section(lay, "Chart")
        self.plot = self._combo()
        model = self.plot.model()
        for key, title, *_rest, plots in QUESTIONS:
            self.plot.addItem(f"— {title} —")
            model.item(self.plot.count() - 1).setEnabled(False)
            for p in plots:
                self.plot.addItem(f"{PLOTS[p][0]}: {PLOTS[p][1]}", p)
        self._f("plot", self.plot, lay)
        self.summary = self._choice({k: f"{t}: {m}" for k, (t, m) in SUMMARIES.items()})
        self._f("summary", self.summary, lay)
        self.error_style = self._choice(ERROR_STYLES)
        self._f("error_style", self.error_style, lay)
        self.show_points = self._check("Show them")
        self._f("show_points", self.show_points, lay)
        self.paired = self._check("Join each cell's values")
        self._f("paired", self.paired, lay)
        self._section(lay, "Analysis")
        self.fit = self._choice(FITS)
        self._f("fit", self.fit, lay)
        self.size_by = self._combo()
        self._f("size_by", self.size_by, lay)
        self.percent = self._check("Percentages")
        self._f("percent", self.percent, lay)
        self.stats = self._check("Compare the groups")
        self._f("stats", self.stats, lay)
        self._section(lay, "Layout")
        self.panel_by = self._combo()
        self._f("panel_by", self.panel_by, lay)
        self.scale_bars = self._check("Scale bars")
        self._f("scale_bars", self.scale_bars, lay)
        lay.addStretch(1)

        # Groups: each group's own look and their order
        lay = self._tab("Groups")
        self.series_table = SeriesTable()
        self.series_table.changed.connect(self._series_edited)
        self._f("series", self.series_table, lay)
        lay.addStretch(1)

        # Axes: titles, ranges, scales, legend, reference lines
        lay = self._tab("Axes")
        self._section(lay, "Titles")
        self.title_edit, self.x_label, self.y_label = self._line(), self._line(), self._line()
        self.x_range, self.y_range = self._line("from, to"), self._line("from, to")
        for key, w in (("title", self.title_edit), ("x_label", self.x_label), ("y_label", self.y_label)):
            self._f(key, w, lay)
        self._section(lay, "Scales")
        for key, w in (("x_range", self.x_range), ("y_range", self.y_range)):
            self._f(key, w, lay)
        logs = QtWidgets.QWidget()
        lg = box("h", "s")
        self.x_log, self.y_log = self._check("x"), self._check("y")
        lg.addWidget(self.x_log)
        lg.addWidget(self.y_log)
        lg.addStretch(1)
        logs.setLayout(lg)
        self._f("log", logs, lay)
        self.y_from_zero = self._check("From zero")
        self._f("y_from_zero", self.y_from_zero, lay)
        self.x_rotate = self._spin(0, 90, 15, 0)
        self._f("x_rotate", self.x_rotate, lay)
        self._section(lay, "Legend and reference lines")
        self.legend = self._combo()
        self.legend.addItems(LEGENDS)
        self._f("legend", self.legend, lay)
        self.ref_lines = ItemTable([("axis", "Line", {"y": "Horizontal, at y", "x": "Vertical, at x"}),
                                    ("at", "At", "number"), ("label", "Label", "text")], "reference line",
                                   {"axis": "y", "at": 0.0, "label": ""})
        self.ref_lines.changed.connect(self._edited)
        self._f("ref_lines", self.ref_lines, lay)
        lay.addStretch(1)

        # Theme: a whole look in one click, then every part of it
        lay = self._tab("Theme")
        self._section(lay, "Theme")
        self.theme_box = QtWidgets.QComboBox()
        self.theme_box.addItems(list(THEMES))
        self.theme_box.activated.connect(self._theme_chosen)
        self._f("theme", self.theme_box, lay)
        self.palette_box = self._combo()
        self.palette_box.addItems(list(PALETTES))
        self._f("palette", self.palette_box, lay)
        self.spins, self.flags, self.colours = {}, {}, {}

        def spin(key, lo, hi, step, dec):
            self.spins[key] = self._spin(lo, hi, step, dec)
            self._f(key, self.spins[key], lay)

        def flag(key, text):
            self.flags[key] = self._check(text)
            self._f(key, self.flags[key], lay)

        self._section(lay, "Text")
        self.font = QtWidgets.QFontComboBox()
        self.font.currentFontChanged.connect(self._edited)
        self._f("font", self.font, lay)
        spin("font_size", 5, 28, 0.5, 1)
        flag("title_bold", "Bold")
        self._section(lay, "Points and lines")
        for key, lo, hi, step, dec in (("marker_size", 1, 20, 0.5, 1), ("line_width", 0.2, 6, 0.1, 1),
                                       ("point_alpha", 0.05, 1, 0.05, 2), ("jitter", 0, 0.4, 0.02, 2),
                                       ("cap_size", 0, 10, 0.5, 1)):
            spin(key, lo, hi, step, dec)
        flag("vary_markers", "Vary markers")
        self._section(lay, "Axes and background")
        self.choices = {"spines": self._choice({"open": "Open: left and bottom", "box": "A box", "left-bottom-light":
                                                "Light, left and bottom", "none": "None"}),
                        "grid": self._choice({"none": "None", "y": "Along y", "x": "Along x", "both": "Both"}),
                        "tick_dir": self._choice({"out": "Outside", "in": "Inside", "none": "None"}),
                        "label_style": self._choice({"A": "A", "a": "a", "(A)": "(A)", "A.": "A."})}
        for key in ("spines", "grid", "tick_dir"):
            self._f(key, self.choices[key], lay)
        flag("minor_ticks", "Minor ticks")
        for key in ("background", "ink"):
            self.colours[key] = ColourButton()
            self.colours[key].changed.connect(self._edited)
            self._f(key, self.colours[key], lay)
        self.transparent = self._check("Transparent background")
        lay.addWidget(self.transparent)
        self._section(lay, "Legend and panel letters")
        spin("legend_cols", 1, 6, 1, 0)
        flag("legend_frame", "Frame")
        self._f("label_style", self.choices["label_style"], lay)
        lay.addStretch(1)

        # Marks: what happened along x, and words on the figure
        lay = self._tab("Marks")
        self.marks = ItemTable([("at", "From x", "number"), ("end", "To x (optional)", "optional number"),
                                ("label", "Label", "text")], "mark", {"at": 0.0, "end": None, "label": ""})
        self.marks.changed.connect(self._edited)
        lay.addWidget(widgets.field("Marks along x", self.marks, "A line at a value, or a shaded span from one value "
                                    "to another, with its name (the induction, a drug's wash-in, the step)",
                                    "Marks along x", "Lines and spans for what happened: the induction, a drug's "
                                    "wash-in, the step."))
        self.notes = ItemTable([("kind", "Kind", {"text": "Text", "arrow": "Arrow with text"}), ("x", "x", "number"),
                                ("y", "y", "number"), ("text", "Text", "text"),
                                ("x2", "Arrow from x", "optional number"), ("y2", "Arrow from y", "optional number")],
                               "text", {"kind": "text", "x": 0.0, "y": 0.0, "text": "n = ", "x2": None, "y2": None})
        self.notes.changed.connect(self._edited)
        lay.addWidget(widgets.field("Text on the figure", self.notes, "Words at x, y in the data's units; an arrow "
                                    "points at x, y from where its text sits", "Text",
                                    "Words placed on the panel at the given position."))
        lay.addStretch(1)
        return holder

    # ---------------------------------------------------------------- the recipe <-> the controls
    @property
    def recipe(self) -> Recipe:
        return self.spec.panels[self.current]

    def _columns(self) -> list[str]:
        f = self.recipe.frame()
        return [str(c) for c in f.columns] if f is not None else []

    def _load_controls(self) -> None:
        """The controls set from the current panel's recipe (no edits fire)."""
        self._loading = True
        try:
            r, s = self.recipe, self.spec.style
            self.tabs.blockSignals(True)
            while self.tabs.count():
                self.tabs.removeTab(0)
            for k in range(len(self.spec.panels)):
                self.tabs.addTab(f"Panel {chr(ord('A') + k)}")
            self.tabs.setCurrentIndex(self.current)
            self.add_panel.setEnabled(len(self.spec.panels) < 6)
            self.remove_panel_button.setEnabled(len(self.spec.panels) > 1)
            self.tabs.setTabsClosable(len(self.spec.panels) > 1)
            self.tabs.blockSignals(False)
            q = [k for k, *_ in QUESTIONS].index(r.question) if r.question in QUESTION else 0
            self.question_box.setCurrentIndex(q)
            self.why.setText(QUESTIONS[q][2])
            cols = self._columns()
            nums = [c for c in cols if c in set(map(str, fdata.numeric_columns(r.frame())))] if cols else []
            for key, combo in self.roles.items():
                combo.clear()
                combo.addItem("(none)", "")
                numeric_only = key in ("y", "y2", "err") or (key == "x" and r.question not in ("counts", "compare"))
                for c in (nums if numeric_only else cols):
                    combo.addItem(c, c)
                combo.setCurrentIndex(max(0, combo.findData(getattr(r, key))))
            for combo, value, blank in ((self.panel_by, r.panel_by, "(one panel)"), (self.size_by, r.size_by,
                                                                                      "(the same size)")):
                combo.clear()
                combo.addItem(blank, "")
                for c in (cols if combo is self.panel_by else nums):
                    combo.addItem(c, c)
                combo.setCurrentIndex(max(0, combo.findData(value)))
            self.filters.set_columns(cols)
            self.filters.set_value(r.filters)
            self.window_edit.setText(_pair_text(r.window))
            self.bin_edit.setText(f"{r.bin:g}" if r.bin else "")
            self.normalize_edit.setText(_pair_text(r.normalize))
            self.level.setCurrentIndex(max(0, self.level.findData(r.level)))
            self.unit.setCurrentText(r.unit if r.unit in UNITS else "cell")
            self.plot.setCurrentIndex(max(0, self.plot.findData(r.plot)))
            self.summary.setCurrentIndex(max(0, self.summary.findData(r.summary)))
            self.error_style.setCurrentIndex(max(0, self.error_style.findData(r.error_style)))
            self.show_points.setChecked(r.show_points)
            self.paired.setChecked(r.paired)
            self.fit.setCurrentIndex(max(0, self.fit.findData(r.fit)))
            self.percent.setChecked(r.percent)
            self.stats.setChecked(r.stats)
            self.scale_bars.setChecked(r.scale_bars)
            for w, v in ((self.title_edit, r.title), (self.x_label, r.x_label), (self.y_label, r.y_label)):
                w.setText(v)
            self.x_range.setText(_pair_text(r.x_range))
            self.y_range.setText(_pair_text(r.y_range))
            self.x_log.setChecked(r.x_log)
            self.y_log.setChecked(r.y_log)
            self.y_from_zero.setChecked(bool(r.y_from_zero if r.y_from_zero is not None else r.plot in BARS))
            self.x_rotate.setValue(r.x_rotate)
            self.legend.setCurrentText(r.legend)
            self.ref_lines.set_value(r.ref_lines)
            self.marks.set_value(r.markers)
            self.notes.set_value(r.annotations)
            self.size.setCurrentIndex(max(0, self.size.findData(self.spec.size)))
            self.theme_box.setCurrentText(s.theme)
            self.palette_box.setCurrentText(s.palette)
            self.font.setCurrentFont(QtGui.QFont(s.font))
            for key, spin in self.spins.items():
                spin.setValue(getattr(s, key))
            for key, combo in self.choices.items():
                combo.setCurrentIndex(max(0, combo.findData(getattr(s, key))))
            self.transparent.setChecked(s.background == "transparent")
            self.colours["background"].set_colour("" if s.background == "transparent" else s.background)
            self.colours["ink"].set_colour(s.ink)
            for key, check in self.flags.items():
                check.setChecked(bool(getattr(s, key)))
            self._source_text()
            self._refresh_series()
            self._visibility()
        finally:
            self._loading = False

    def _source_text(self) -> None:
        r = self.recipe
        f = r.frame()
        kind = r.source.get("kind", "")
        where = {"project": f"This project's table “{r.source.get('table', '')}”",
                 "file": f"{Path(r.source.get('path', '')).name}" + (f", sheet {r.source.get('sheet')}"
                                                                      if r.source.get("sheet") else ""),
                 "digitized": f"Digitized from {Path(r.source.get('path', '')).name}: an estimate",
                 "example": r.source.get("table", ""),
                 "graph": f"The graph “{r.source.get('table', '')}”" + (f" of {r.source['recording']}"
                                                                     if r.source.get("recording") else ""),
                 "traces": "Raw traces"
                 }.get(kind, r.source.get("table", ""))
        self.source.setText(f"{where}: {len(f)} rows, {f.shape[1]} columns." if f is not None and where else
                            "No table yet: Open data, above.")
        self.file_box.setVisible(kind == "file" and Path(r.source.get("path", "")).suffix.lower() in
                                 (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt"))
        if kind == "file":
            self.sheet.blockSignals(True)
            self.sheet.clear()
            self.sheet.addItems(fdata.sheets(r.source["path"]))
            self.sheet.setCurrentText(r.source.get("sheet") or "")
            self.sheet.setVisible(self.sheet.count() > 0)
            self.sheet_label.setVisible(self.sheet.count() > 0)
            self.sheet.blockSignals(False)
            for w, v in ((self.header, r.source.get("header", 0) + 1), (self.skip, r.source.get("skip", 0))):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

    def _visibility(self) -> None:
        r = self.recipe
        q = r.question
        for key, w in self.fields.items():
            if key in SHOWN:
                w.setVisible(q in SHOWN[key])
        self.fields["x"].setVisible(q in SHOWN["x"] or (q == "compare" and r.plot == "grouped_bar"))
        self.fields["summary"].setVisible(q in ("compare", "time", "pulses") or r.plot == "xy_error")
        self.fields["error_style"].setVisible(q in ("compare", "time", "pulses") or r.plot == "xy_error")
        self.fields["size_by"].setVisible(r.plot == "bubble")
        self.fields["stats"].setVisible(q == "compare" and self._on("figure_statistics"))
        self.example_button.setVisible(q == "example" and self._on("example_traces")
                                       and getattr(self.window_, "selected", None) is not None)
        self.reshape_button.setVisible(r.frame() is not None and q != "example")
        picked = r.source.get("kind") == "traces"
        self.traces_box.setVisible(q == "example" or picked)
        names = [f"{s['name']} ({traces.MODES.get(s.get('mode'), '').lower()}, {len(s.get('items', []))} sweep"
                 f"{'s' if len(s.get('items', [])) != 1 else ''})" for s in r.source.get("sets", [])] if picked else []
        self.traces_said.setText("Showing: " + "; ".join(names) if names else
                                 "Choose the recording and the sweeps this panel shows.")
        self.layouts_button.setVisible(self._on("figure_layouts"))
        self.methods_button.setVisible(self._on("methods_text"))
        for head, keys in self._sections:
            head.setVisible(any(not self.fields[k].isHidden() for k in keys))

    def _read_controls(self) -> None:
        r, s = self.recipe, self.spec.style
        for key, combo in self.roles.items():
            setattr(r, key, combo.currentData() or "")
        r.panel_by = self.panel_by.currentData() or ""
        r.size_by = self.size_by.currentData() or ""
        r.filters = self.filters.value()
        r.window = _pair(self.window_edit.text())
        r.bin = _float(self.bin_edit.text()) or 0.0
        r.normalize = _pair(self.normalize_edit.text())
        r.level = self.level.currentData() or "cell"
        r.unit = self.unit.currentText() or "cell"
        r.plot = self.plot.currentData() or r.plot
        r.summary = self.summary.currentData() or "mean_sem"
        r.error_style = self.error_style.currentData() or "bar"
        r.show_points = self.show_points.isChecked()
        r.paired = self.paired.isChecked()
        r.fit = self.fit.currentData() or ""
        r.percent = self.percent.isChecked()
        r.stats = self.stats.isChecked() and self._on("figure_statistics")
        r.scale_bars = self.scale_bars.isChecked()
        r.title, r.x_label, r.y_label = self.title_edit.text(), self.x_label.text(), self.y_label.text()
        r.x_range, r.y_range = _pair(self.x_range.text()), _pair(self.y_range.text())
        r.x_log, r.y_log = self.x_log.isChecked(), self.y_log.isChecked()
        r.y_from_zero = self.y_from_zero.isChecked()
        r.x_rotate = self.x_rotate.value()
        r.legend = self.legend.currentText()
        r.ref_lines = self.ref_lines.value()
        r.markers = self.marks.value()
        r.annotations = [{k: v for k, v in a.items() if v is not None} for a in self.notes.value()]
        self.spec.size = self.size.currentData() or "single"
        s.palette = self.palette_box.currentText()
        s.font = self.font.currentFont().family()
        for key, spin in self.spins.items():
            setattr(s, key, int(spin.value()) if key == "legend_cols" else spin.value())
        for key, combo in self.choices.items():
            setattr(s, key, combo.currentData())
        plain = Style()                         # the figure's own defaults, when no colour is chosen
        s.background = "transparent" if self.transparent.isChecked() else (self.colours["background"].colour
                                                                           or plain.background)
        s.ink = self.colours["ink"].colour or plain.ink
        for key, check in self.flags.items():
            setattr(s, key, check.isChecked())

    def _edited(self, *_args) -> None:
        if self._loading:
            return
        before = self.recipe.plot
        self._read_controls()
        r = self.recipe
        if r.plot != before:
            if PLOT_QUESTION.get(r.plot, r.question) != r.question:      # any plot can be tried: its question follows
                r.question = PLOT_QUESTION[r.plot]
                if r.question in ("distribution", "example"):
                    r.level = "row"
            if r.plot in BARS:                                           # a bar's height means something from zero
                r.y_from_zero = True
            self._load_controls()
        self._visibility()
        self._changed()

    def _changed(self) -> None:
        self._snapshot()
        self.timer.start()

    def _series_edited(self) -> None:
        if self._loading:
            return
        self.recipe.series, self.recipe.order = self.series_table.value()
        self._changed()

    def _refresh_series(self) -> None:
        """The Groups tab lists the groups (and, for counts, the categories) the panel draws."""
        drawn = self._drawn[self.current] if self.current < len(self._drawn) else None
        names = []
        if drawn is not None and drawn.prepared is not None:
            pts = drawn.prepared.points
            names = render.ordered(list(dict.fromkeys(pts.group)), self.recipe)
            if self.recipe.question == "counts" and "xcat" in pts:
                names += [c for c in dict.fromkeys(pts.xcat) if c not in names]
        names += [g for g, look in self.recipe.series.items() if look.get("hidden") and g not in names]
        defaults = render.looks(self.recipe, self.spec.style, names)
        self.series_table.set_rows(names, self.recipe.series, defaults)

    def _theme_chosen(self, _index: int = 0) -> None:
        self._read_controls()
        self.spec.style.apply_theme(self.theme_box.currentText())
        self._load_controls()
        self._changed()

    def _question_chosen(self, index: int) -> None:
        if self._loading:
            return
        self._read_controls()
        key, _title, why, plot, _plots = QUESTIONS[index]
        r = self.recipe
        r.question, r.plot = key, plot
        if key == "distribution":
            r.level = "row"                  # every event's value; the curves are per cell
        elif key == "example":
            r.level, r.scale_bars = "row", True
        elif r.level == "row" and r.cell:
            r.level = "cell"
        if key == "counts" and r.y and not r.x:
            r.x = ""
        self._load_controls()
        self._changed()

    # ---------------------------------------------------------------- undo, redo, panels
    def _snapshot(self) -> None:
        text = self.spec.to_json()
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-40]
            self._future.clear()
        self.undo_button.setEnabled(len(self._history) > 1)
        self.redo_button.setEnabled(bool(self._future))

    def _restore(self, text: str) -> None:
        self.spec = Spec.from_json(text)
        self.current = min(self.current, len(self.spec.panels) - 1)
        self._load_controls()
        self.undo_button.setEnabled(len(self._history) > 1)
        self.redo_button.setEnabled(bool(self._future))
        self.redraw()

    def undo(self) -> None:
        if len(self._history) > 1:
            self._future.append(self._history.pop())
            self._restore(self._history[-1])

    def redo(self) -> None:
        if self._future:
            self._history.append(self._future.pop())
            self._restore(self._history[-1])

    def _select_panel(self, k: int) -> None:
        if 0 <= k < len(self.spec.panels) and not self._loading:
            self.current = k
            self._load_controls()

    def set_panels(self, n: int, columns: int | None = None) -> None:
        panels = self.spec.panels
        while len(panels) < n:              # a new panel starts as a copy of the current one
            panels.append(copy.deepcopy(self.recipe))
        del panels[n:]
        self.spec.columns = columns or min(n, 3)
        self.current = min(self.current, n - 1)
        self._load_controls()
        self._changed()

    def _panels_menu(self, m: QtWidgets.QMenu) -> None:
        m.clear()
        for n in (1, 2, 3, 4, 6):
            a = m.addAction(f"{n} panel{'s' if n > 1 else ''}", lambda n=n: self.set_panels(n))
            a.setCheckable(True)
            a.setChecked(len(self.spec.panels) == n)
        m.addSeparator()
        for c in (1, 2, 3):
            a = m.addAction(f"In {c} column{'s' if c > 1 else ''}", lambda c=c: self.set_panels(len(self.spec.panels), c))
            a.setCheckable(True)
            a.setChecked(self.spec.columns == c)
        lab = m.addAction("Label the panels A, B, C", self._toggle_labels)
        lab.setCheckable(True)
        lab.setChecked(self.spec.labels)

    def remove_panel(self, k: int) -> None:
        """Panel ``k`` out of the figure (undo brings it back); the last panel stays."""
        if len(self.spec.panels) < 2 or not 0 <= k < len(self.spec.panels):
            return
        del self.spec.panels[k]
        self.spec.columns = max(1, min(self.spec.columns, len(self.spec.panels)))
        self.current = min(self.current if self.current < k else max(0, self.current - 1), len(self.spec.panels) - 1)
        self._load_controls()
        self._changed()
        self.statusBar().showMessage(f"Panel {chr(ord('A') + k)} removed (Undo, Ctrl+Z, brings it back).", 5000)

    def duplicate_panel(self, k: int) -> None:
        if len(self.spec.panels) >= 6:
            return
        self.spec.panels.insert(k + 1, copy.deepcopy(self.spec.panels[k]))
        self.spec.columns = min(max(self.spec.columns, 1), 3)
        self.current = k + 1
        self._load_controls()
        self._changed()

    def move_panel(self, k: int, d: int) -> None:
        j = k + d
        if 0 <= k < len(self.spec.panels) and 0 <= j < len(self.spec.panels):
            p = self.spec.panels
            p[k], p[j] = p[j], p[k]
            self.current = j
            self._load_controls()
            self._changed()

    def _panel_actions(self, m: QtWidgets.QMenu, k: int) -> None:
        n = len(self.spec.panels)
        letter = chr(ord("A") + k)
        m.addAction(f"Edit panel {letter}", lambda: self.tabs.setCurrentIndex(k))
        a = m.addAction(f"Duplicate panel {letter}", lambda: self.duplicate_panel(k))
        a.setEnabled(n < 6)
        a = m.addAction(f"Remove panel {letter}", lambda: self.remove_panel(k))
        a.setEnabled(n > 1)
        a = m.addAction("Move it left", lambda: self.move_panel(k, -1))
        a.setEnabled(k > 0)
        a = m.addAction("Move it right", lambda: self.move_panel(k, 1))
        a.setEnabled(k < n - 1)
        m.addSeparator()
        a = m.addAction("Add a new panel", lambda: self.set_panels(n + 1))
        a.setEnabled(n < 6)

    def _tab_menu(self, m: QtWidgets.QMenu, pos) -> None:
        k = self.tabs.tabAt(pos)
        self._panel_actions(m, k if k >= 0 else self.current)

    def _canvas_menu(self, m: QtWidgets.QMenu, pos) -> None:
        """Right-click on the preview: the panel under the pointer, and the figure's own actions."""
        x, y = self.canvas.mouseEventCoords(pos)
        k = None
        cols = max(1, self.spec.columns)
        for ax in self.fig.axes:
            ss = ax.get_subplotspec()
            if ss is not None and ax.bbox.contains(x, y):
                top = ss.get_topmost_subplotspec()
                k = top.rowspan.start * cols + top.colspan.start
                break
        if k is not None and k < len(self.spec.panels):
            self._panel_actions(m, k)
        else:
            a = m.addAction("Add a new panel", lambda: self.set_panels(len(self.spec.panels) + 1))
            a.setEnabled(len(self.spec.panels) < 6)
        m.addSeparator()
        m.addAction("Copy the figure as a picture", self.copy_picture)
        m.addAction("Export...", self.export)

    def copy_picture(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())
        self.statusBar().showMessage("The figure is on the clipboard.", 3000)

    def clear_data(self) -> None:
        """This panel without its table (its look kept); undo brings the table back."""
        r = self.recipe
        r.data, r.source = "", {}
        r.x = r.y = r.y2 = r.err = r.group = r.cell = r.animal = r.panel_by = r.size_by = ""
        r.filters, r.notes, r.markers = [], [], []
        self._load_controls()
        self._changed()

    def _toggle_labels(self) -> None:
        self.spec.labels = not self.spec.labels
        self._changed()

    # ---------------------------------------------------------------- data in
    def set_table(self, frame, source: dict) -> None:
        """Give the current panel a table; the columns that look like cells, animals,
        groups and time are taken for those roles (and can be changed)."""
        r = self.recipe
        r.data, r.source = pack(frame), dict(source)
        r.markers, r.notes, r.annotations = [], [], []          # they belonged to the table before
        r.title = r.x_label = r.y_label = ""
        r.x_range, r.y_range = [], []
        g = fdata.guess(frame)
        nums = fdata.numeric_columns(frame)
        r.cell, r.animal, r.group = g["cell"], g["animal"], g["group"]
        r.x = g["x"] if r.question in ("time", "pulses", "relationship", "example") else ""
        r.y = r.y if r.y in frame.columns else next((c for c in nums if c not in (r.x, r.cell, r.animal)), "")
        r.filters = [f for f in r.filters if f[0] in frame.columns]
        self._load_controls()
        self._changed()

    def open_project_table(self) -> None:
        p = getattr(self.window_, "project", None)
        if p is None:
            QtWidgets.QMessageBox.information(self, "No project", "Open a project (Files) to use its tables.")
            return
        tables = fdata.project_tables(p)
        if not tables:
            QtWidgets.QMessageBox.information(self, "Nothing analyzed", "Analyze the recordings first.")
            return
        name, ok = QtWidgets.QInputDialog.getItem(self, "A table of this project", "Table:", list(tables), 0, False)
        if ok:
            self.set_table(tables[name](), {"kind": "project", "table": name})

    def open_file(self, path: str | None = None, sheet: str | None = None) -> None:
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "A CSV or Excel file", "",
                                                            "Tables (*.csv *.tsv *.txt *.xlsx *.xlsm *.xls)")
        if not path:
            return
        names = fdata.sheets(path)
        if names and sheet is None:
            sheet, ok = QtWidgets.QInputDialog.getItem(self, "Which sheet", "Sheet:", names, 0, False)
            if not ok:
                return
        header, skip = fdata.sniff(path, sheet)
        try:
            frame = fdata.read_table(path, sheet, header, skip)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not read the file", str(exc))
            return
        self.set_table(frame, {"kind": "file", "path": str(path), "sheet": sheet or "", "header": header,
                               "skip": skip})

    def _reread(self, *_args) -> None:
        if self._loading:
            return
        src = dict(self.recipe.source)
        src.update(sheet=self.sheet.currentText() or src.get("sheet", ""), header=self.header.value() - 1,
                   skip=self.skip.value())
        try:
            frame = fdata.read_table(src["path"], src["sheet"] or None, src["header"], src["skip"])
        except Exception as exc:
            self.source.setText(f"Could not read it that way: {exc}")
            return
        self.set_table(frame, src)

    def open_figure(self, path: str | None = None) -> None:
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "A figure saved by Clampwise", "", "Clampwise figures "
                                                            "(*.cwfig *.pdf *.svg *.eps *.png *.tif *.tiff)")
        if not path:
            return
        try:
            spec = render.read_spec(path)
        except Exception as exc:
            QtWidgets.QMessageBox.information(self, "Not a Clampwise figure", str(exc))
            return
        self.spec, self.current = spec, 0
        self._load_controls()
        self._changed()

    def open_picture(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "A picture of a plot", "", "Pictures (*.png *.jpg "
                                                        "*.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        dlg = DigitizeDialog(self, Path(path))
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.points:
            frame = dlg.table()
            self.recipe.question, self.recipe.plot, self.recipe.level = "relationship", "line", "row"
            self.set_table(frame, {"kind": "digitized", "path": path})
            self.recipe.x, self.recipe.y = frame.columns[0], frame.columns[1]
            self._load_controls()
            self._changed()

    def reshape(self) -> None:
        f = self.recipe.frame()
        if f is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Conditions in separate columns")
        lay = box("v", "s", "m")
        lay.addWidget(label("Pick the columns that each hold one condition's values; they become one column of "
                            "values and one naming the condition.", "", wrap=True))
        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        lst.addItems([str(c) for c in fdata.numeric_columns(f)])
        lay.addWidget(lst)
        name = QtWidgets.QLineEdit("Condition")
        value = QtWidgets.QLineEdit("Value")
        labels = QtWidgets.QLineEdit()
        labels.setPlaceholderText("e.g. control, drug")
        lay.addWidget(widgets.field("Each picked column's condition", labels, "In the columns' order, separated by "
                                    "commas (blank: the columns' own names)"))
        lay.addWidget(widgets.field("Name of the condition column", name))
        lay.addWidget(widgets.field("Name of the values column (with units)", value))
        ok = widgets.primary("Reshape", dlg.accept)
        lay.addWidget(ok)
        widgets.resizable(dlg, lay, "reshape")
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            cols = [i.text() for i in sorted(lst.selectedItems(), key=lst.row)]
            if len(cols) >= 2:
                self.apply_reshape(cols, [t.strip() for t in labels.text().split(",") if t.strip()],
                                   name.text() or "Condition", value.text() or "Value")

    def apply_reshape(self, columns: list[str], labels: list[str], name: str, value: str) -> None:
        """Conditions in separate columns -> one column of values (``value``) and one
        naming the condition (``name``), each column's condition from ``labels``."""
        long = fdata.to_long(self.recipe.frame(), columns, name, value, labels)
        src = {**self.recipe.source, "reshaped": f"{', '.join(columns)} -> {name}"}
        self.set_table(long, src)
        self.recipe.group, self.recipe.y = name, value
        self._load_controls()
        self._changed()

    def use_example(self) -> None:
        w = self.window_
        path = getattr(w, "selected", None)
        if path is None or w.project is None:
            return
        rec = w.recording(path)
        run = w.project.last_run(path, rec)
        made = layouts.typical_sets(rec, run.results if run is not None else {})
        if made is None:
            QtWidgets.QMessageBox.information(self, "No stimulus", "This recording has no stimulus to align on.")
            return
        sets, win = made
        for s in sets:
            s.items = [[str(path), k, p] for _f, k, p in s.items]
        self.set_traces(sets, win, f"{path.name}: the typical response")

    # ---------------------------------------------------------------- raw traces
    def _load(self, file: str):
        """A trace's recording: a path, a file name of this project, or "" for the selected recording."""
        w = self.window_
        if not file:
            return w.recording(w.selected) if getattr(w, "selected", None) is not None else None
        p = Path(file)
        if not p.exists() and w.project is not None:
            p = next((r.path for r in w.project.recordings() if r.name == file), p)
        return w.recording(p) if p.exists() else None

    def choose_traces(self) -> None:
        """The trace picker, opened on this panel's picks when it has them."""
        from .tracepicker import TracePicker

        if getattr(self.window_, "project", None) is None:
            QtWidgets.QMessageBox.information(self, "No project", "Open a project (Files) to pick its traces.")
            return
        src = self.recipe.source
        sets, win = traces.from_source(src) if src.get("kind") == "traces" else ([], None)
        dlg = TracePicker(self, self.window_, self._load, sets, win)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.sets:
            self.set_traces(dlg.sets, dlg.trace_window(), src.get("table") if src.get("kind") == "traces" else
                            "Raw traces")

    def set_traces(self, sets, win, table: str) -> None:
        """This panel shows ``sets`` (raw traces), cut by ``win``; its look, titles and what is drawn over the
        traces (a rise and decay fit, marked events) are kept."""
        overlay = self.recipe.source.get("overlay") if self.recipe.source.get("kind") == "traces" else None
        frame, notes, markers = traces.build(sets, self._load, win, overlay)
        if frame.empty:
            QtWidgets.QMessageBox.information(self, "Nothing to draw", "None of the picked sweeps could be cut "
                                              "around the chosen point (no such stimulus pulse).")
            return
        r = self.recipe
        if r.question != "example":                 # a new example panel starts with scale bars
            r.scale_bars = True
        r.data, r.source = pack(frame), traces.source(sets, win, table, overlay)
        r.question, r.plot, r.level = "example", "trace", "row"
        r.x, r.y, r.group, r.cell = frame.columns[0], frame.columns[1], "Trace", "Sweep"
        r.notes, r.markers, r.filters = notes, markers, []
        self._load_controls()
        self._changed()

    # ---------------------------------------------------------------- layouts, styles, methods
    def _layouts_menu(self, m: QtWidgets.QMenu) -> None:
        """The presets by category; one that does not apply is shown greyed with what it needs."""
        m.clear()
        sel = self.selection()
        for cat in (presets.SINGLE, presets.ACROSS):
            m.addSection(cat)
            for p in presets.PRESETS.values():
                if p.category != cat:
                    continue
                why = presets.needs(p, sel)
                a = m.addAction(f"{p.name}: {p.line}" + (f"  ({why})" if why else ""),
                                lambda k=p.key: self.apply_preset(k))
                a.setEnabled(not why)
                a.setToolTip(why or p.line)

    def selection(self) -> "presets.Selection":
        """What a preset is filled from: the selected recording's results and its cell's, and, across cells, this
        panel's table when it names groups (else the project's table of the selected recording's type)."""
        from ..analysis import registry

        w = self.window_
        path, project = getattr(w, "selected", None), getattr(w, "project", None)
        sel = presets.Selection(load=self._load, y=self.recipe.y)
        f = self.recipe.frame()
        if f is not None and self.recipe.question != "example" and fdata.guess(f)["group"]:
            sel.table, sel.source = f, dict(self.recipe.source)
        if project is None or path is None:
            return sel
        rec = w.recording(path)
        run = project.last_run(path, rec)
        cell = project.cell_result(path)
        sel.results = {**(run.results if run is not None else {}), **(cell.results if cell is not None else {})}
        sel.path, sel.name = str(path), Path(path).name
        sel.files = {r.name: str(r.path) for r in project.recordings()}
        if sel.table is None and run is not None:
            name = f"Recordings: {registry.TYPE_WORDS.get(run.recording_type, run.recording_type)}"
            tables = fdata.project_tables(project)
            if name in tables:
                t = tables[name]()
                if fdata.guess(t)["group"]:
                    sel.table, sel.source = t, {"kind": "project", "table": name}
        return sel

    def apply_preset(self, key: str) -> None:
        """The figure becomes preset ``key`` filled from the selection; the look is kept, undo brings back the
        figure before."""
        spec = presets.fill(presets.PRESETS[key], self.selection())
        spec.style = self.spec.style
        self.spec, self.current = spec, 0
        self._load_controls()
        self._changed()

    def dsi_layout(self) -> None:
        self.apply_preset("dsi")

    def _styles(self) -> dict:
        try:
            return json.loads(self.window_._settings().value("figure_styles", "{}") or "{}")
        except Exception:
            return {}

    def _styles_menu(self, m: QtWidgets.QMenu) -> None:
        m.clear()
        m.addAction("Save this look as a style...", self.save_style)
        m.addAction("Export the saved styles to a file...", self.export_styles)
        m.addAction("Import styles from a file...", self.import_styles)
        saved = self._styles()
        if saved:
            m.addSeparator()
        for name, d in saved.items():
            m.addAction(f"Use “{name}”", lambda d=d: self.apply_style(d))
        if saved:
            gone = m.addMenu("Delete a saved style")
            for name in saved:
                gone.addAction(f"“{name}”", lambda n=name: self.delete_style(n))

    def delete_style(self, name: str) -> None:
        if QtWidgets.QMessageBox.question(self, "Delete the style", f"Delete the saved style “{name}”?") != \
                QtWidgets.QMessageBox.Yes:
            return
        saved = self._styles()
        saved.pop(name, None)
        self.window_._settings().setValue("figure_styles", json.dumps(saved))
        self.statusBar().showMessage(f"Style “{name}” deleted.", 3000)

    def save_style(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save the look", "Name:")
        if ok and name.strip():
            saved = self._styles()
            from dataclasses import asdict

            saved[name.strip()] = asdict(self.spec.style)
            self.window_._settings().setValue("figure_styles", json.dumps(saved))

    def export_styles(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the styles", "figure styles.cwstyles",
                                                            "Clampwise figure styles (*.cwstyles)")
        if path:
            Path(path).write_text(json.dumps({"format": "clampwise-figure-styles", "styles": self._styles()},
                                             indent=1), encoding="utf-8")

    def import_styles(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import styles", "",
                                                            "Clampwise figure styles (*.cwstyles *.json)")
        if not path:
            return
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "clampwise-figure-styles":
            QtWidgets.QMessageBox.information(self, "Not figure styles", f"{Path(path).name} holds no figure styles.")
            return
        saved = {**self._styles(), **d.get("styles", {})}
        self.window_._settings().setValue("figure_styles", json.dumps(saved))

    def apply_style(self, d: dict) -> None:
        self.spec.style = Style(**{k: v for k, v in d.items() if k in Style.__dataclass_fields__})
        self._load_controls()
        self._changed()

    def methods_text(self) -> str:
        return methods.paragraph(self._drawn, getattr(self.window_, "project", None))

    def show_methods(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Methods text")
        lay = box("v", "s", "m")
        lay.addWidget(label("Written from the analyses, their settings and sources, and this figure: edit it into "
                            "your paper.", "muted", wrap=True))
        text = QtWidgets.QPlainTextEdit(self.methods_text())
        lay.addWidget(text, 1)
        lay.addWidget(widgets.button("Copy", lambda: QtWidgets.QApplication.clipboard().setText(text.toPlainText())))
        widgets.resizable(dlg, lay, "methods_text", (700, 420))
        dlg.exec_()

    # ---------------------------------------------------------------- drawing and saving
    def redraw(self) -> None:
        try:
            self.fig, self._drawn = render.build(self.spec, self.fig)
        except Exception as exc:                     # a combination that cannot be drawn says why
            self.fig.clear()
            self.fig.text(0.5, 0.5, f"Cannot draw this: {exc}", ha="center", va="center", wrap=True)
            self._drawn = []
        self._fit_canvas()
        self._describe()
        self._refresh_series()              # the Groups tab follows the groups the panel now draws

    def _fit_canvas(self) -> None:
        """The figure as large as the area allows (or at its printed size, zoomed), its proportions as it will
        print."""
        w, h = self.fig.get_size_inches()
        zoom = self.zoom.currentData() if hasattr(self, "zoom") else 0.0
        if zoom:
            dpi = float(self.logicalDpiX() or 96) * zoom
        else:
            room = self.holder.viewport().size()
            dpi = min(max(room.width() - 24, 200) / w, max(room.height() - 24, 150) / h) if room.width() > 50 \
                else SCREEN_DPI
            dpi = float(min(max(dpi, 60.0), 220.0))
        wmm, hmm = self.spec.size_mm()
        n = len(self.spec.panels)
        self.size_note.setText(f"· {wmm:.0f} × {hmm:.0f} mm, {n} panel{'s' if n > 1 else ''}")
        self.fig.set_dpi(dpi)
        self.canvas.setFixedSize(int(w * dpi), int(h * dpi))
        self.canvas.draw_idle()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "fig"):
            self._fit_canvas()

    def _describe(self) -> None:
        parts, warns = [], []
        many = len(self._drawn) > 1
        for k, d in enumerate(self._drawn):
            tag = f"({chr(ord('A') + k)}) " if many else ""
            if d.prepared is None:
                continue
            parts.append(f"<p>{html.escape(tag + d.prepared.caption)}</p>")
            for c in d.comparisons:
                if d.recipe.stats or d.recipe.plot == "estimation":
                    from ..figure.render import _units

                    parts.append(f"<p><i>{html.escape(tag + c.sentence(_units(d.recipe.y)))}</i></p>")
            warns += [f"{tag}{w}" for w in d.prepared.warnings]
        self.caption.setHtml("".join(parts) or "<p>Choose a table and what to show.</p>")
        lay = self.warnings.layout()
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().setParent(None)        # gone now, not at the next event loop
        for w in warns:
            b = widgets.Badge("warn", w, w)
            b.setWordWrap(True)
            lay.addWidget(b)

    def caption_text(self) -> str:
        return self.caption.toPlainText()

    def copy_caption(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.caption_text())

    def export(self, path: str | None = None, dpi: int | None = None) -> Path | None:
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the figure", "figure.pdf",
                                                            "PDF (*.pdf);;SVG (*.svg);;EPS (*.eps);;PNG (*.png);;"
                                                            "TIFF (*.tif *.tiff)")
        if not path:
            return None
        if dpi is None and Path(path).suffix.lower() in (".png", ".tif", ".tiff"):
            dpi, ok = QtWidgets.QInputDialog.getInt(self, "Resolution", "Dots per inch (300 to 600 for print):", 600,
                                                    72, 1200)
            if not ok:
                return None
        out = render.export(self.spec, path, dpi or 600)
        self.statusBar().showMessage(f"Saved {out.name}; it opens here again, editable (Open data, a saved figure).")
        return out

    def save_figure(self, path: str | None = None) -> Path | None:
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the figure to edit later", "figure.cwfig",
                                                            "Clampwise figure (*.cwfig)")
        if not path:
            return None
        out = render.save_spec(self.spec, path)
        self.statusBar().showMessage(f"Saved {out.name}.")
        return out

