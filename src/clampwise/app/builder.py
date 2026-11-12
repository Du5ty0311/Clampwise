"""The figure builder's window: the figure itself to work on, with what it is drawn from beside it.

In the middle, the largest, the figure as it will export (matplotlib), on a desk at its printed proportions.
Anything on it is selected by a click, moved by a drag, rewritten by a double-click, and has its own actions on
a right-click (``figcanvas``). Beside it, three tabs: **Selected**, the few properties of whatever is selected
(``inspector``: a panel, an axis, a group, the legend, the scale bars, a stimulus bar, a text, a bracket); **Data
and statistics**, the table the current panel is drawn from, rows that can be left out by hand, the test and its
result (``dataview``); and **All settings**, every control of the current panel and of the look in full, for
what the first two do not reach. Under the figure, its legend's draft and what a reader should be warned of.

A figure starts as an empty page: **Add panel** asks what it shows (traces, a graph of the Figures screen, a
kept Workbench measurement, a table, a file) and then how it is drawn, from pictures (``addpanel``). The old
"what do you want to show?" questions remain as starters under All settings. Everything a control sets lives
in the figure's ``Spec`` (``clampwise.figure``), so undo, saving and reopening are the same thing.
"""

from __future__ import annotations

import copy
import html
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import data as fdata
from ..figure import elements, journals, layouts, methods, page, presets, render, traces
from ..figure.elements import Tag
from ..figure.prepare import OPERATORS
from ..figure.recipe import (CONTROLS, ERROR_STYLES, FITS, LINES, MARKERS, PALETTES, PLOT_QUESTION, PLOTS,
                             QUESTION, QUESTIONS, SIZES, SUMMARIES, THEMES, Recipe, Spec, Style, pack)
from . import theme, widgets
from .widgets import box, label

SCREEN_DPI = 110
MOST_PANELS = 12
LEGENDS = ("best", "upper left", "upper right", "lower left", "lower right", "outside", "none")
# what a person may drag on the figure, and whose words they may rewrite in place
MOVABLE = ("letter", "legend", "scalebar", "stimbar", "annotation", "bracket", "figtext", "panel")
PLACE_FIELDS = ("span", "rect", "inset_of", "inset_rect", "letter", "letter_offset")
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
        on = self.table.rowCount() > 0
        self.remove_button.setEnabled(on)
        self.remove_button.setToolTip(f"Remove the selected {self.noun}(s) from the figure (Delete)" if on else
                                      f"There is no {self.noun} to remove.")

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
        self.buttons = {}
        for text, tip, slot in (("Up", "Draw the selected group earlier", lambda: self._move(-1)),
                                ("Down", "Draw the selected group later", lambda: self._move(1)),
                                ("Hide from the figure", "Leave the selected group out of the figure (and its "
                                 "numbers); Show brings it back", lambda: self._show(False)),
                                ("Show", "Put the selected group back in the figure", lambda: self._show(True)),
                                ("Reset to the theme", "Every group's look from the theme again", self._reset)):
            self.buttons[text] = (widgets.button(text, slot, tip, quiet=True), tip)
            row.addWidget(self.buttons[text][0])
        row.addStretch(1)
        lay.addLayout(row)
        self.setLayout(lay)
        self.table.itemSelectionChanged.connect(self._enable)
        self._enable()

    def _enable(self) -> None:
        """Each button on when it would change something; off, its tooltip saying why, when it would not."""
        i, n = self.table.currentRow() if self.table.selectedIndexes() else -1, len(self._names)
        shown = 0 <= i < n and self.table.cellWidget(i, 1) is not None and self.table.cellWidget(i, 1).isChecked()
        none = "This panel draws no groups yet." if not n else "Select a group in the table first."
        series, _order = self.value() if n else ({}, [])
        own = any(k not in getattr(self, "_extra", {}).get(g, {}) for g, d in series.items() for k in d)
        for text, on, why in (("Up", i > 0, none if i < 0 else "The selected group is already drawn first."),
                              ("Down", 0 <= i < n - 1, none if i < 0 else "The selected group is already drawn last."),
                              ("Hide from the figure", shown, none if i < 0 else "The selected group is already "
                                                                                  "hidden."),
                              ("Show", 0 <= i < n and not shown, none if i < 0 else "The selected group is already "
                                                                                    "shown."),
                              ("Reset to the theme", own, "Every group's look already comes from the theme.")):
            b, tip = self.buttons[text]
            b.setEnabled(bool(on))
            b.setToolTip(tip if on else why)

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
            for w, words in ((shown, "shown in the figure"), (colour, "colour"), (marker, "marker"), (line, "line"),
                             (width, "line width"), (filled, "filled markers"), (name_edit, "name in the legend")):
                w.setAccessibleName(f"A group's {words}")
                if not w.toolTip():
                    w.setToolTip(f"{name or 'The data'}: {words}")
        self._loading = False
        if hasattr(self, "buttons"):
            self._enable()

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
            self._enable()
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
        for w in (self.trace_button, self.click_button):
            w.setCheckable(True)
            w.setProperty("oneOfSeveral", True)        # the way of reading in use stays pressed
        for w in (self.log_x, self.log_y):
            w.setToolTip("The axis is logarithmic: the two known values are read on a log scale")
            w.toggled.connect(lambda _on: self._said_table())
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
        self.range_said = label("", "muted", wrap=True)
        lay.addWidget(self.range_said)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, self.undo_last)
        widgets.resizable(self, lay, "digitize", (900, 700))
        self._mode("calibrate")

    def _mode(self, mode: str) -> None:
        self.mode = mode
        ready = len(self.clicks) == 4
        need = "Calibrate first: click two known values on each axis."
        self.trace_button.setEnabled(ready)
        self.trace_button.setToolTip("Then click on the curve: every column of that colour is read" if ready else need)
        self.click_button.setEnabled(ready)
        self.click_button.setToolTip("Then click each point to read it" if ready else need)
        self.trace_button.setChecked(mode == "trace")
        self.click_button.setChecked(mode == "points")
        self._can_finish()
        if mode == "trace":
            self.say.setText("Click on the curve: the columns of its colour are read")
        elif mode == "points":
            self.say.setText("Click each point; they are read as you go")

    def _can_finish(self) -> None:
        on = bool(self.points) and len(self.clicks) == 4
        self.done.setEnabled(on)
        self.done.setToolTip("Make the figure from the digitized points" if on else
                             "No point is read yet: calibrate, then trace a curve or click points.")
        self._said_table()

    def _said_table(self) -> None:
        """The range the points read now span, so a wrong calibration or a log axis is seen at once."""
        if not hasattr(self, "range_said"):
            return
        if not self.points or len(self.clicks) < 4:
            self.range_said.setText("")
            return
        try:
            t = self.table()
            x, y = t.iloc[:, 0], t.iloc[:, 1]
            self.range_said.setText(f"{len(t)} points read: {t.columns[0]} from {x.min():.4g} to {x.max():.4g}, "
                                    f"{t.columns[1]} from {y.min():.4g} to {y.max():.4g}")
        except Exception as exc:
            self.range_said.setText(f"The points cannot be read with this calibration: {exc}")

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
        self._can_finish()

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
        self._can_finish()

    def clear_points(self) -> None:
        self.points = []
        self._drop(self._point_marks)
        self._point_marks = []
        self._can_finish()

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
        self._can_finish()

    def table(self):
        names = [n.strip() for n in self.names.text().split(",")] + ["x", "y"]
        cols, rows = zip(*self.points)
        return self.dz.table(self.calibration(), cols, rows, names[0] or "x", names[1] or "y")


class ExportDialog(QtWidgets.QDialog):
    """How the figure is written: the format (PDF for Illustrator, with its text kept as text), the resolution
    of a pixel format, text as shapes for a journal that asks for it; and, first, what the journal's rules (or
    plain legibility) say about the figure as it stands."""

    FORMATS = (("pdf", "PDF: vector, text stays editable (Illustrator, Inkscape)"),
               ("svg", "SVG: vector, text stays editable (Inkscape, the web)"),
               ("eps", "EPS: vector, for journals that ask for it"),
               ("tiff", "TIFF: pixels, for journals that ask for it"),
               ("png", "PNG: pixels, for slides and messages"))

    def __init__(self, parent, spec: Spec, findings: list) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export the figure")
        self.controls_name = "Export"
        lay = box("v", "s", "m")
        w, h = spec.size_mm()
        j = journals.JOURNALS.get(spec.journal)
        lay.addWidget(label(f"{w:.1f} × {h:.1f} mm" + (f", {j.name}'s sizes" if j else ""), "heading"))
        lay.addWidget(label("The file is written at exactly this size, the figure itself inside it: it opens "
                            "here again, editable.", "muted", wrap=True))
        self.findings = QtWidgets.QListWidget()
        self.findings.setAccessibleName("What the check found")
        self.findings.setMaximumHeight(120)
        for f in findings:
            item = QtWidgets.QListWidgetItem(("Fix: " if f.serious else "Note: ") + (f"panel {f.where}: " if f.where
                                                                                     else "") + f.text)
            self.findings.addItem(item)
        if not findings:
            self.findings.addItem("Nothing to fix: the sizes, text and lines are fine" +
                                  (f" for {j.name}." if j else " for print."))
        lay.addWidget(self.findings)
        for n in (j.notes if j else ()):
            lay.addWidget(label(f"{j.name}: {n}", "hint", wrap=True))
        self.format = QtWidgets.QComboBox()
        self.format.setAccessibleName("Format")
        for key, words in self.FORMATS:
            self.format.addItem(words, key)
        prefer = next((f for f in (j.formats if j else ()) if f in ("pdf", "eps", "tiff")), "pdf")
        self.format.setCurrentIndex(max(0, self.format.findData("pdf" if j is None else prefer)))
        self.dpi = QtWidgets.QSpinBox()
        self.dpi.setAccessibleName("Resolution")
        self.dpi.setRange(72, 2400)
        self.dpi.setSingleStep(100)
        self.dpi.setSuffix(" dpi")
        self.dpi.setValue(render.export_dpi(spec))
        self.outline = QtWidgets.QCheckBox("Turn the text into shapes")
        self.outline.setChecked(False)
        self.outline.setToolTip("The letters become drawn shapes: no font is needed to print them, and they can no "
                                "longer be edited as text. Only for a journal that asks for outlined fonts"
                                + (f" ({j.name} does, in EPS files)." if j is not None and j.outline_text else "."))
        self._outline_tip = self.outline.toolTip()
        lay.addWidget(widgets.field("Format", self.format))
        self.dpi_field = widgets.field("Resolution", self.dpi, "For TIFF and PNG: 600 to 1200 dpi keeps text and "
                                       "lines sharp in print")
        lay.addWidget(self.dpi_field)
        lay.addWidget(self.outline)
        row = box("h", "s")
        row.addStretch(1)
        row.addWidget(widgets.button("Cancel", self.reject, "Do not export"))
        row.addWidget(widgets.primary("Choose where to save...", self.accept, "Then the file is written"))
        lay.addLayout(row)
        self.format.currentIndexChanged.connect(lambda _i: self._follow())
        self._follow()
        widgets.resizable(self, lay, "export_figure", (560, 420))

    def _follow(self) -> None:
        pixels = self.format.currentData() in ("tiff", "png")
        self.dpi.setEnabled(pixels)
        self.dpi.setToolTip("Dots per inch at the figure's printed size" if pixels else
                            "A vector format has no resolution: it is sharp at any size.")
        self.outline.setEnabled(not pixels)
        self.outline.setToolTip("A pixel format holds no text to turn into shapes." if pixels else self._outline_tip)


class Builder(QtWidgets.QMainWindow):
    MOST = MOST_PANELS

    def __init__(self, window, spec: Spec | None = None) -> None:
        super().__init__(window)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle("Figure builder – Clampwise")
        self.window_ = window
        if not isinstance(spec, Spec):       # a new figure: an empty page, Arial 7 pt, no journal chosen
            spec = Spec()
            spec.style.font_size = 7.0
        self.spec = spec
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
        self.dataview.refresh()              # its buttons say from the start what they need
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

    def _data_menu(self, m: QtWidgets.QMenu) -> None:
        m.clear()
        m.setToolTipsVisible(True)
        m.addAction("A saved figure, to work on it again...", self.open_figure)
        m.addSeparator()
        m.addAction("Data for the selected panel...", lambda: self.choose_data(self.current))
        for text, slot in (("Raw traces from the recordings (pick sweeps, average them)...", self.choose_traces),
                           ("A table of this project...", self.open_project_table),
                           ("A CSV or Excel file...", self.open_file),
                           ("A picture of a plot (digitize it)...", self.open_picture)):
            m.addAction(text, slot)
        m.addSeparator()
        a = m.addAction("Remove this panel's data", self.clear_data)
        has = bool(self.recipe.data)
        a.setEnabled(has)
        a.setToolTip("This panel without its table, its look kept (Undo brings the table back)" if has else
                     "This panel has no data to remove.")

    # ---------------------------------------------------------------- what a panel shows: one chooser
    def _sources(self) -> tuple[dict, dict]:
        """What the data chooser offers now: ({source: why it cannot be used}, {source: [(words, what to use)]})."""
        from . import panels as figpanels

        w = self.window_
        project, path = getattr(w, "project", None), getattr(w, "selected", None)
        no_project = "open a project in Files first"
        no_rec = "select a recording in Files first"
        cannot: dict[str, str] = {}
        extra: dict[str, list] = {}
        if project is None:
            for key in ("traces", "typical", "graph", "workbench", "project"):
                cannot[key] = no_project
            return cannot, extra
        run = None
        if path is None:
            cannot["typical"] = cannot["graph"] = no_rec
        else:
            try:
                run = project.last_run(path, w.recording(path))
            except Exception:
                run = None
            if run is None or not run.results:
                cannot["graph"] = "analyze the selected recording first"
            else:
                found = [v for v in figpanels.views(run) if v.heatmap is None]
                extra["graph"] = [(f"{v.title} {v.units}", v) for v in found]
                if not found:
                    cannot["graph"] = "its results give no graph"
            try:
                rec = w.recording(path)
                if not any(np.isfinite(traces.stimulus_s(rec, k, 1)) for k in range(min(rec.n_sweeps, 5))):
                    cannot["typical"] = "the selected recording has no stimulus to align on"
            except Exception:
                cannot["typical"] = "the selected recording could not be read"
        hands = fdata.project_hands(project)
        if hands:
            extra["workbench"] = [(words, use) for words, use in hands]
        else:
            cannot["workbench"] = "nothing is kept yet: Workbench, Keep this measurement"
        tables = fdata.project_tables(project)
        if tables:
            extra["project"] = [(name, make) for name, make in tables.items()]
        else:
            cannot["project"] = "analyze the recordings first"
        return cannot, extra

    def add_panel_dialog(self, k: int | None = None) -> None:
        """Add panel: ask what it shows, put it in the first empty panel (else a new one), then ask how it is
        drawn when that is a choice. With ``k``: the same for panel ``k``'s data."""
        from .addpanel import DataChooser

        cannot, extra = self._sources()
        dlg = DataChooser(self, cannot, extra, "Add a panel" if k is None else "Choose the panel's data")
        if dlg.exec_() != QtWidgets.QDialog.Accepted or dlg.choice is None:
            return
        self.use_source(*dlg.choice, k=k)

    def choose_data(self, k: int) -> None:
        self.add_panel_dialog(k)

    def _target(self, k: int | None) -> int:
        """The panel new data go into: ``k``, else the first empty one, else a new one (None when full)."""
        if k is not None:
            return k
        empty = next((i for i, r in enumerate(self.spec.panels) if r.frame() is None), None)
        if empty is not None:
            return empty
        if len(self.spec.panels) >= self.MOST:
            QtWidgets.QMessageBox.information(self, "The figure is full", f"A figure holds {self.MOST} panels at "
                                              "most: remove one, or change a panel's data.")
            return -1
        self.spec.panels.append(Recipe())
        if self.spec.columns == 1 and len(self.spec.panels) == 2:
            self.spec.columns = 2                     # the second panel goes beside the first
            self._widen()
        return len(self.spec.panels) - 1

    def _widen(self) -> None:
        """A one-column page that now holds panels side by side becomes a two-column one (said, and undone with
        the step that caused it)."""
        if self.spec.size == "single" and self.spec.columns > 1:
            self.spec.size = "double"
            self._fill_sizes()
            self.statusBar().showMessage(f"The page is now two columns wide ({self.spec.width_mm():g} mm) to hold "
                                         "panels side by side; the width is in the toolbar.", 7000)

    def use_source(self, key: str, use=None, k: int | None = None) -> None:
        """Panel ``k`` (or the next free one) shows the source ``key`` (``addpanel.SOURCES``)."""
        if key == "figure":
            return self.open_figure(append=True)
        target = self._target(k)
        if target < 0:
            return
        before = len(self.spec.panels)
        self.current = target
        self._load_controls()
        had = self.recipe.data
        if key == "traces":
            self.choose_traces()
        elif key == "typical":
            self.use_example()
        elif key == "graph":
            self.use_graph(use)
        elif key == "workbench":
            self.use_hand(*use)
        elif key == "project":
            name = next((n for n, make in fdata.project_tables(self.window_.project).items() if make is use), "table")
            self._table_then_plot(use(), {"kind": "project", "table": name})
        elif key == "file":
            self.open_file()
            if self.recipe.data and self.recipe.data != had:
                self.choose_plot(self.current)
        elif key == "picture":
            self.open_picture()
        if not self.recipe.data and k is None and len(self.spec.panels) == before and before > 1 and target == before - 1 \
                and not had:
            pass                                      # nothing was chosen: the empty panel stays to be filled
        self._changed()
        self.redraw_now()
        self.canvas.select(Tag("panel", self.current))

    def _table_then_plot(self, frame, source: dict) -> None:
        r = self.recipe
        if r.question == "example":                   # a table is not a trace: the usual start for one
            r.question, r.plot, r.level, r.scale_bars = "compare", "points", "cell", False
        self.set_table(frame, source)
        self.choose_plot(self.current)

    def choose_plot(self, k: int) -> None:
        """How panel ``k`` is drawn, picked from pictures of every kind of plot."""
        from .addpanel import PlotChooser

        if not 0 <= k < len(self.spec.panels):
            return
        self.current = k
        self._load_controls()
        dlg = PlotChooser(self, self.recipe.plot)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.plot and dlg.plot != self.recipe.plot:
            self.set_plot(dlg.plot)

    def set_plot(self, plot: str) -> None:
        """The current panel drawn as ``plot`` (its question follows, with the columns that plot needs)."""
        r = self.recipe
        self.side.blockSignals(True)
        i = self.plot.findData(plot)
        if i >= 0:
            self.plot.setCurrentIndex(i)              # the same path as the Plot pick list: its question follows
        self.side.blockSignals(False)
        f = r.frame()
        if f is not None:                             # the columns the new plot needs, when not yet named
            g = fdata.guess(f)
            nums = [c for c in map(str, fdata.numeric_columns(f))]
            if r.question in ("time", "pulses", "relationship") and not r.x:
                r.x = g["x"] or next((c for c in nums if c != r.y), "")
            if r.question == "counts" and not r.x:
                r.x = next((c for c in map(str, fdata.category_columns(f)) if c not in (r.group, r.cell, r.animal)),
                           r.group)
            if plot == "raster":
                r.level = "row"
                r.x = r.x or (nums[0] if nums else "")
            self._load_controls()
            self._changed()

    def add_recipe(self, recipe: Recipe, k: int | None = None) -> int:
        """A ready panel (from the Workbench, from a Figures graph) added to the figure: into the first empty
        panel, else as a new one. Returns where it went (-1: the figure is full)."""
        target = self._target(k)
        if target < 0:
            return -1
        old = self.spec.panels[target]
        for f in PLACE_FIELDS:                        # it takes the place of the panel it fills
            setattr(recipe, f, copy.deepcopy(getattr(old, f)))
        self.spec.panels[target] = recipe
        self.current = target
        self._load_controls()
        self._changed()
        self.redraw_now()
        self.canvas.select(Tag("panel", target))
        return target

    def use_graph(self, view) -> None:
        """A graph of the Figures screen as the current panel: drawn out of sight, read back, its look kept."""
        import pyqtgraph as pg

        from . import editable

        w = self.window_
        item = pg.PlotItem()
        pal = getattr(w, "palette", None) or theme.LIGHT
        view.draw(item, pal)
        name = Path(w.selected).name if getattr(w, "selected", None) is not None else ""
        spec = editable.from_plot(item, view.title, name)
        if spec is None:
            QtWidgets.QMessageBox.information(self, "Nothing drawn", f"“{view.title}” draws nothing for this "
                                              "recording, so there is nothing to put in the panel.")
            return
        self.add_recipe(spec.panels[0], self.current)

    def use_hand(self, path, hand: dict) -> None:
        """A measurement kept in the Workbench as the current panel: its table (every measure, sweep by sweep),
        the first measure against the sweep to start with."""
        try:
            frame = fdata.hand_table(self.window_.recording(Path(path)), hand)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not measure it again", f"{hand.get('name', '')}: {exc}")
            return
        if frame.empty:
            QtWidgets.QMessageBox.information(self, "Nothing measured", "This kept measurement has no rows.")
            return
        r = self.recipe
        r.question, r.plot, r.level, r.scale_bars = "relationship", "scatter", "row", False
        self.set_table(frame, {"kind": "workbench", "table": hand.get("name", "by hand"),
                               "recording": Path(path).name})
        measures = [c for c in frame.columns if not str(c).startswith(fdata.HAND_KEYS)]
        r = self.recipe
        r.x, r.y = "Sweep", (measures[0] if measures else "")
        r.group = "Region" if frame["Region"].nunique() > 1 else ("Pulse" if "Pulse" in frame and
                                                                  frame["Pulse"].nunique() > 1 else "")
        r.cell = r.animal = ""
        r.notes = [f"Measured by hand in the Workbench on {Path(path).name}."]
        self._load_controls()
        self._changed()

    def _build(self) -> None:
        central = QtWidgets.QWidget()
        outer = box("v", "s")
        outer.setContentsMargins(0, 0, 0, theme.SPACE["s"])
        bar = QtWidgets.QWidget()
        bar.setObjectName("toolbar")
        top = box("h", "xs", "s")
        self.add_button = widgets.primary("Add panel...", lambda *_a: self.add_panel_dialog())
        top.addWidget(self.add_button)
        top.addWidget(self._tool("Open", "A saved figure to work on again, or data for the selected panel",
                                 self._data_menu))
        top.addWidget(_vline())
        self.undo_button = self._tool("Undo", "Undo the last change (Ctrl+Z)", slot=self.undo)
        self.redo_button = self._tool("Redo", "Redo it (Ctrl+Y)", slot=self.redo)
        top.addWidget(self.undo_button)
        top.addWidget(self.redo_button)
        top.addWidget(_vline())
        top.addWidget(self._tool("Arrange", "Where the panels sit: a template, the grid's columns, free placement, "
                                 "aligning", self._arrange_menu))
        self.size = QtWidgets.QComboBox()
        self.size.setToolTip("The figure's width on the page (a journal's column widths when a journal is chosen: "
                             "click the page's background and see Journal)")
        self.size.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.size.currentIndexChanged.connect(self._size_chosen)
        self.size.setAccessibleName("Figure size")
        top.addWidget(self.size)
        top.addWidget(_vline())
        top.addWidget(self._tool("Styles", "Save this look, or use a saved one", self._styles_menu))
        self.layouts_button = self._tool("Starters", "A whole figure in one step from the selected recording or "
                                         "cells; every panel stays editable", self._layouts_menu)
        top.addWidget(self.layouts_button)
        self.methods_button = self._tool("Methods text", "A Methods paragraph for this figure", slot=self.show_methods)
        top.addWidget(self.methods_button)
        top.addStretch(1)
        top.addWidget(widgets.button("Save figure...", self.save_figure, "Keep it to edit again (.cwfig)"))
        top.addWidget(widgets.button("Export...", self.export, "PDF (text editable in Illustrator), SVG, EPS, TIFF "
                                     "or PNG, at the figure's exact size, checked first"))
        bar.setLayout(top)
        outer.addWidget(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._preview())
        split.addWidget(self._side())
        split.setStretchFactor(0, 3)
        split.setSizes([940, 480])
        body = box("h", "s")
        body.setContentsMargins(theme.SPACE["m"], 0, theme.SPACE["m"], 0)
        body.addWidget(split)
        outer.addLayout(body, 1)
        central.setLayout(outer)
        self.setCentralWidget(central)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo),
                           ("Ctrl+S", self.save_figure), ("Ctrl+E", self.export), ("Ctrl+Return", self.apply)):
            QtWidgets.QShortcut(QtGui.QKeySequence(keys), self, slot)

    def _side(self) -> QtWidgets.QWidget:
        """Beside the figure: what is selected, the current panel's data and statistics, and every setting."""
        from .dataview import DataView
        from .inspector import Inspector

        self.side = QtWidgets.QTabWidget()
        self.side.setDocumentMode(True)
        self.side.setMinimumWidth(430)
        self.inspector = Inspector(self)
        self.dataview = DataView(self)
        for w, name, tip in ((self.inspector, "Selected", "The properties of what is selected on the figure (the "
                              "figure's own when nothing is)"),
                             (self.dataview, "Data and statistics", "The table the current panel is drawn from, and "
                              "the statistics on it")):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            holder = QtWidgets.QWidget()
            hl = box("v", "s", "m")
            hl.addWidget(w)
            holder.setLayout(hl)
            scroll.setWidget(holder)
            self.side.addTab(scroll, name)
            self.side.setTabToolTip(self.side.count() - 1, tip)
        self.side.addTab(self._controls(), "All settings")
        self.side.setTabToolTip(2, "Every control of the current panel and of the figure's look, in full")
        self.side.currentChanged.connect(self._side_shown)
        return self.side

    def _side_shown(self, index: int) -> None:
        if self._loading:
            return
        if index == 1:
            self.dataview.refresh()
        elif index == 2:
            self._load_controls()

    def show_all_settings(self, tab: str = "") -> None:
        """The All settings tab, on one of its own tabs (Data, Plot, Groups, Axes, Theme, Marks)."""
        self.side.setCurrentIndex(2)
        for i in range(self.sections.count()):
            if self.sections.tabText(i) == tab:
                self.sections.setCurrentIndex(i)

    def _preview(self) -> QtWidgets.QWidget:
        """The figure as it will export, on a desk, with its size and a zoom; under it, its legend and warnings."""
        from .figcanvas import FigureCanvas

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setChildrenCollapsible(False)
        top = QtWidgets.QWidget()
        tl = box("v", "xs")
        head = box("h", "s")
        head.addWidget(label("Figure", "heading"))
        self.size_note = label("", "muted")
        head.addWidget(self.size_note)
        head.addStretch(1)
        self.how = label("Click to select · drag to move · double-click text to rewrite · right-click for more",
                         "hint")
        self.how.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        head.addWidget(label("Zoom", "muted"))
        self.zoom = QtWidgets.QComboBox()
        for text, f in (("Fit to window", 0.0), ("Print size", 1.0), ("150%", 1.5), ("200%", 2.0), ("300%", 3.0)):
            self.zoom.addItem(text, f)
        self.zoom.setToolTip("Fit the figure to the window, or show it at its printed size (and larger) to judge "
                             "the font sizes and line widths")
        self.zoom.currentIndexChanged.connect(lambda _i: self._fit_canvas())
        self.zoom.setAccessibleName("Zoom")
        head.addWidget(self.zoom)
        tl.addLayout(head)
        tl.addWidget(self.how)                # what the mouse does with the thing selected, on a line of its own
        self.fig = Figure(dpi=SCREEN_DPI)
        self.canvas = FigureCanvas(self.fig)
        pal = getattr(self.window_, "palette", None)
        if pal is not None:
            self.canvas.accent = QtGui.QColor(pal.accent)
        self.canvas.can_move = self._can_move
        self.canvas.can_resize = self._can_resize
        self.canvas.words_of = self.words_of
        self.canvas.selected.connect(self._on_selected)
        self.canvas.moved.connect(self._moved)
        self.canvas.resized.connect(self._resized)
        self.canvas.text_edited.connect(self.set_words)
        self.canvas.menu_requested.connect(self._thing_menu)
        self.canvas.activated.connect(self._activated)
        self.canvas.removed.connect(self.remove_thing)
        self.canvas.nudged.connect(self._nudged)
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
        cap_head.addWidget(label("Figure legend (draft)", "heading"))
        cap_head.addStretch(1)
        cap_head.addWidget(widgets.button("Copy legend", self.copy_caption, "Copy the legend's draft: what each panel "
                                          "shows, its n, its statistics", quiet=True))
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
        if not control.accessibleName():
            control.setAccessibleName(caption)          # what a screen reader (and the controls test) calls it
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
        outer.addWidget(label("Start from a question (optional)", "heading"))
        self.question_box = QtWidgets.QComboBox()
        for key, title, why, *_rest in QUESTIONS:
            self.question_box.addItem(title, key)
        self.question_box.setToolTip("The question sets up the standard plot for it; any other plot can be tried "
                                     "on the Plot tab")
        self.question_box.activated.connect(self._question_chosen)
        self.question_box.setAccessibleName("What do you want to show?")
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
        self.add_panel = widgets.button("+ Copy as a new panel", lambda: self.set_panels(len(self.spec.panels) + 1),
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
        holder.setMinimumWidth(420)            # the tabs and the Groups table whole

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
            w.setAccessibleName(tip.split(" (")[0])
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
        self.x_log.setAccessibleName("Log x axis")
        self.y_log.setAccessibleName("Log y axis")
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
        self.transparent.setToolTip("No background colour: a slide's own colour shows through (PNG, SVG, PDF)")
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
            names = render.letters(self.spec)
            for k in range(len(self.spec.panels)):
                self.tabs.addTab(f"Panel {names[k] or chr(ord('A') + k)}" + (" (inset)" if page.is_inset(
                    self.spec.panels, k) else ""))
            self.tabs.setCurrentIndex(self.current)
            self.add_panel.setEnabled(len(self.spec.panels) < self.MOST)
            self.add_panel.setToolTip("Add a panel, starting as a copy of this one" if len(self.spec.panels) <
                                      self.MOST else f"A figure holds {self.MOST} panels at most.")
            self.remove_panel_button.setEnabled(len(self.spec.panels) > 1)
            self.remove_panel_button.setToolTip("Remove the panel being edited (its × does the same; Undo brings it "
                                                "back)" if len(self.spec.panels) > 1 else
                                                "A figure keeps at least one panel.")
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
            self._fill_sizes()
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
        self._undo_buttons()

    def _undo_buttons(self) -> None:
        can_undo, can_redo = len(self._history) > 1, bool(self._future)
        self.undo_button.setEnabled(can_undo)
        self.undo_button.setToolTip("Undo the last change (Ctrl+Z)" if can_undo else "Nothing to undo yet.")
        self.redo_button.setEnabled(can_redo)
        self.redo_button.setToolTip("Redo the change just undone (Ctrl+Y)" if can_redo else
                                    "Nothing to redo: no change was undone.")

    def _restore(self, text: str) -> None:
        self.spec = Spec.from_json(text)
        self.current = min(self.current, len(self.spec.panels) - 1)
        self._load_controls()
        self._undo_buttons()
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
        while len(panels) < n:              # a new panel starts as a copy of the current one, in a cell of its own
            new = copy.deepcopy(self.recipe)
            new.span, new.rect, new.inset_of, new.inset_rect, new.letter, new.letter_offset = [], [], -1, [], "", []
            panels.append(new)
        del panels[n:]
        for r in panels:                    # an inset whose panel is gone is a panel again
            if r.inset_of >= len(panels):
                r.inset_of = -1
        self.spec.columns = columns or (self.spec.columns if self.spec.columns > 1 and n > self.spec.columns
                                        else min(n, 3))
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
        for c in (1, 2, 3, 4):
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
        for r in self.spec.panels:          # its insets become panels; the others' parents keep their numbers
            r.inset_of = -1 if r.inset_of == k else r.inset_of - 1 if r.inset_of > k else r.inset_of
        self.spec.columns = max(1, min(self.spec.columns, len(self.spec.panels)))
        self.canvas.select(None)
        self.current = min(self.current if self.current < k else max(0, self.current - 1), len(self.spec.panels) - 1)
        self._load_controls()
        self._changed()
        self.statusBar().showMessage(f"Panel {chr(ord('A') + k)} removed (Undo, Ctrl+Z, brings it back).", 5000)

    def duplicate_panel(self, k: int) -> None:
        if len(self.spec.panels) >= self.MOST:
            return
        new = copy.deepcopy(self.spec.panels[k])
        new.letter, new.letter_offset = "", []
        if len(new.rect) == 4:              # a freely placed copy sits a little down and right of its original
            new.rect = [new.rect[0] + 4.0, new.rect[1] + 4.0, new.rect[2], new.rect[3]]
        for r in self.spec.panels:
            if r.inset_of > k:
                r.inset_of += 1
        self.spec.panels.insert(k + 1, new)
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
        letter = self.letter_of(k) or chr(ord("A") + k)
        r = self.spec.panels[k]

        def add(text, slot, on=True, why=""):
            a = m.addAction(text, slot)
            a.setEnabled(bool(on))
            if not on:
                a.setToolTip(why)
            return a

        full = f"A figure holds {self.MOST} panels at most."
        has = r.frame() is not None
        add("Choose its data..." if not has else "Change its data...", lambda: self.choose_data(k))
        add("Choose how it is drawn...", lambda: self.choose_plot(k), has, "This panel has no data yet.")
        add("See its data and statistics", lambda: (self._select_panel(k), self.side.setCurrentIndex(1)), has,
            "This panel has no data yet.")
        m.addSeparator()
        add("Add text", lambda: self.add_annotation(k, "text"), has, "This panel has no data yet.")
        add("Add an arrow with text", lambda: self.add_annotation(k, "arrow"), has, "This panel has no data yet.")
        add("Add a stimulus bar", lambda: self.add_stim_bar(k), has, "This panel has no data yet.")
        if not page.is_inset(self.spec.panels, k):
            add("Add an inset", lambda: self.add_inset(k), n < self.MOST, full)
            add("Place the panels freely (drag and resize)" if len(r.rect) != 4 else "Put the panels back in the grid",
                lambda: self.place_freely(k, len(r.rect) != 4))
        m.addSeparator()
        add(f"Duplicate panel {letter}", lambda: self.duplicate_panel(k), n < self.MOST, full)
        add(f"Remove panel {letter}", lambda: self.remove_panel(k), n > 1, "A figure keeps at least one panel.")
        add("Move it earlier", lambda: self.move_panel(k, -1), k > 0, "It is the first panel already.")
        add("Move it later", lambda: self.move_panel(k, 1), k < n - 1, "It is the last panel already.")
        m.setToolTipsVisible(True)

    def _tab_menu(self, m: QtWidgets.QMenu, pos) -> None:
        k = self.tabs.tabAt(pos)
        self._panel_actions(m, k if k >= 0 else self.current)

    def _arrange_menu(self, m: QtWidgets.QMenu) -> None:
        """Where the panels sit: a template, the grid's columns, free placement, aligning the free ones."""
        m.clear()
        m.setToolTipsVisible(True)
        t = m.addMenu("A template")
        for key, tpl in page.TEMPLATES.items():
            a = t.addAction(f"{tpl.name} ({tpl.n} panel{'s' if tpl.n != 1 else ''})", lambda k=key: self.use_template(k))
            a.setToolTip(tpl.line)
        t.setToolTipsVisible(True)
        cols = m.addMenu("Columns of the grid")
        for c in (1, 2, 3, 4, 5, 6):
            a = cols.addAction(f"{c} column{'s' if c > 1 else ''}", lambda c=c: self.set_columns(c, True))
            a.setCheckable(True)
            a.setChecked(self.spec.columns == c)
        m.addSeparator()
        free = [k for k, r in enumerate(self.spec.panels) if len(r.rect) == 4]
        a = m.addAction("Place every panel freely", self.all_free)
        a.setEnabled(len(free) < len([k for k in range(len(self.spec.panels))
                                      if not page.is_inset(self.spec.panels, k)]))
        a.setToolTip("Each panel keeps where the grid has it, and can then be dragged and resized" if a.isEnabled()
                     else "Every panel is placed freely already.")
        a = m.addAction("Put every panel back in the grid", self.all_grid)
        a.setEnabled(bool(free))
        a.setToolTip("The panels flow into the grid's cells again" if free else "No panel is placed freely.")
        al = m.addMenu("Align and space the freely placed panels")
        for key, words in page.ALIGN.items():
            need = 3 if key.startswith("spread") else 2
            a = al.addAction(words, lambda k=key: self.arrange(k))
            a.setEnabled(len(free) >= need)
            a.setToolTip("The selected panel is the one the others follow" if len(free) >= need else
                         f"Needs {need} freely placed panels (Place every panel freely).")
        al.setToolTipsVisible(True)
        m.addSeparator()
        lab = m.addAction("Letter the panels A, B, C", self._toggle_labels)
        lab.setCheckable(True)
        lab.setChecked(self.spec.labels)

    # ---------------------------------------------------------------- the figure worked on directly
    def touch(self, what: str = "") -> None:
        """Something of the figure was changed (by the inspector, the data view, the canvas): kept for Undo and
        drawn again."""
        self._changed()

    def select(self, tag) -> None:
        self.canvas.select(tag)

    def select_panel_on_figure(self, k: int) -> None:
        """Panel ``k`` selected on the figure (what was just added, shown as the thing to work on)."""
        if 0 <= k < len(self.spec.panels):
            self.canvas.select(Tag("panel", k))

    def letter_of(self, k: int) -> str:
        names = render.letters(self.spec)
        return names[k] if 0 <= k < len(names) else ""

    def drawn_of(self, k: int):
        return self.canvas.panel_drawn(k)

    def look_of(self, k: int, group: str):
        d = self.drawn_of(k)
        if d is None or d.prepared is None:
            return None
        groups = render.ordered(list(dict.fromkeys(d.prepared.points.group if len(d.prepared.points) else
                                                   d.prepared.rows.group)), d.recipe)
        return render.looks(d.recipe, self.spec.style, groups).get(group)

    def source_words(self, k: int) -> str:
        r = self.spec.panels[k]
        f = r.frame()
        if f is None:
            return "No data yet: choose what this panel shows."
        kind = r.source.get("kind", "")
        where = {"project": f"This project's table “{r.source.get('table', '')}”",
                 "file": f"{Path(r.source.get('path', '')).name}" + (f", sheet {r.source.get('sheet')}"
                                                                      if r.source.get("sheet") else ""),
                 "digitized": f"Digitized from {Path(r.source.get('path', '')).name}: an estimate",
                 "example": r.source.get("table", ""),
                 "graph": f"The graph “{r.source.get('table', '')}”" + (f" of {r.source['recording']}"
                                                                     if r.source.get("recording") else ""),
                 "workbench": f"The Workbench measurement “{r.source.get('table', '')}”" + (
                     f" of {r.source['recording']}" if r.source.get("recording") else ""),
                 "traces": "Raw traces: " + "; ".join(
                     f"{s['name']} ({len(s.get('items', []))} sweep{'s' if len(s.get('items', [])) != 1 else ''})"
                     for s in r.source.get("sets", []))}.get(kind, r.source.get("table", "") or "A table")
        return f"{where}: {len(f)} rows, {f.shape[1]} columns."

    @staticmethod
    def pair(text: str) -> list:
        return _pair(text)

    @staticmethod
    def pair_text(v: list) -> str:
        return _pair_text(v)

    @staticmethod
    def set_dict(d: dict, key: str, value) -> None:
        """``d[key]`` set, or taken out when the value is blank (then it is automatic again)."""
        if value in (None, ""):
            d.pop(key, None)
        else:
            d[key] = value

    def units_of(self, r: Recipe, which: str) -> str:
        return render._units(getattr(r, which, "") or "")

    def default_label(self, r: Recipe, which: str) -> str:
        return (r.x if which == "x" else render._y_name(r)) or ""

    # -- what may be dragged, and whose words may be rewritten
    def _can_move(self, tag) -> bool:
        if tag is None or tag.kind not in MOVABLE:
            return False
        return not (tag.kind == "scalebar" and tag.key == "command")

    def _can_resize(self, tag) -> bool:
        if tag is None or tag.kind != "panel" or not 0 <= tag.panel < len(self.spec.panels):
            return False
        r = self.spec.panels[tag.panel]
        return len(r.rect) == 4 or page.is_inset(self.spec.panels, tag.panel)

    def bracket(self, k: int, key: str) -> dict:
        """The bracket between two groups as set by hand ({} when it is still automatic)."""
        a, _, c = str(key).partition("|")
        return next((x for x in self.spec.panels[k].brackets if (str(x.get("a")), str(x.get("b"))) == (a, c)), {})

    def set_bracket(self, tag, field_: str, value) -> None:
        r = self.spec.panels[tag.panel]
        a, _, c = str(tag.key).partition("|")
        own = dict(self.bracket(tag.panel, tag.key)) or {"a": a, "b": c}
        self.set_dict(own, field_, value)
        r.brackets = [x for x in r.brackets if (str(x.get("a")), str(x.get("b"))) != (a, c)]
        if set(own) - {"a", "b"}:
            r.brackets.append(own)

    def words_of(self, tag):
        """The words of a thing that can be rewritten in place (None: it has none)."""
        if tag is None:
            return None
        if tag.kind == "figtext":
            i = int(tag.key)
            return self.spec.texts[i].get("text", "") if 0 <= i < len(self.spec.texts) else None
        if not 0 <= tag.panel < len(self.spec.panels):
            return None
        r, kind = self.spec.panels[tag.panel], tag.kind
        try:
            if kind == "title":
                return r.title
            if kind == "letter":
                return r.letter or self.letter_of(tag.panel)
            if kind == "xlabel":
                return r.x_label or self.default_label(r, "x")
            if kind == "ylabel":
                return r.y_label or self.default_label(r, "y")
            if kind == "annotation":
                return r.annotations[int(tag.key)].get("text", "")
            if kind == "stimbar":
                return r.stim_bars[int(tag.key)].get("label", "")
            if kind == "bracket":
                own = self.bracket(tag.panel, tag.key).get("text")
                return own if own else self._bracket_text(tag)
            if kind == "series":
                return r.series.get(tag.key, {}).get("label") or tag.key
            if kind == "vmlabel":
                return r.vm_labels.get(tag.key, "")
            if kind == "refline":
                return r.ref_lines[int(tag.key)].get("label", "")
            if kind == "marker":
                return r.markers[int(tag.key)].get("label", "")
        except (IndexError, ValueError):
            return None
        return None

    def _bracket_text(self, tag) -> str:
        for t, a in render.things(self.fig):
            if (t.kind, t.panel, t.key, t.part) == ("bracket", tag.panel, tag.key, "text"):
                return a.get_text()
        return ""

    def set_words(self, tag, text: str) -> None:
        """A thing's words rewritten (in place on the figure, or from the inspector)."""
        if tag.kind == "figtext":
            self.spec.texts[int(tag.key)]["text"] = text
            return self.touch("text")
        r, kind = self.spec.panels[tag.panel], tag.kind
        if kind == "title":
            r.title = text
        elif kind == "letter":
            r.letter = text.strip()
        elif kind == "xlabel":
            r.x_label = text
        elif kind == "ylabel":
            r.y_label = text
        elif kind == "annotation":
            r.annotations[int(tag.key)]["text"] = text
        elif kind == "stimbar":
            r.stim_bars[int(tag.key)]["label"] = text.strip()
        elif kind == "bracket":
            self.set_bracket(tag, "text", text.strip())
        elif kind == "series":
            mine = dict(r.series.get(tag.key, {}))
            self.set_dict(mine, "label", text.strip() if text.strip() != tag.key else "")
            r.series = {**r.series, tag.key: mine}
        elif kind == "vmlabel":
            r.vm_labels = {**r.vm_labels, tag.key: text.strip()}
        elif kind == "refline":
            r.ref_lines[int(tag.key)]["label"] = text.strip()
        elif kind == "marker":
            r.markers[int(tag.key)]["label"] = text.strip()
        self.touch("text")

    # -- the canvas's reports
    def _on_selected(self, tag) -> None:
        if tag is not None and 0 <= tag.panel < len(self.spec.panels) and tag.panel != self.current:
            self.current = tag.panel
            if self.side.currentIndex() == 2:
                self._load_controls()
            else:
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(self.current)
                self.tabs.blockSignals(False)
        self.inspector.show_for(tag)
        if self.side.currentIndex() == 2 or (self.side.currentIndex() == 1 and tag is not None
                                            and tag.kind != "panel"):
            self.side.setCurrentIndex(0)
        if self.side.currentIndex() == 1:
            self.dataview.refresh()
        self.how.setText(self._how_words(tag))

    def _how_words(self, tag) -> str:
        if tag is None:
            return "Click to select · drag to move · double-click text to rewrite · right-click for more"
        words = elements.KIND_WORDS.get(tag.kind, tag.kind).lower()
        can = []
        if self._can_move(tag):
            can.append("drag or the arrow keys move it")
        if self.words_of(tag) is not None:
            can.append("double-click rewrites it")
        can.append("Delete removes it" if tag.kind not in ("xaxis", "yaxis", "xlabel", "ylabel") else
                   "its settings are on the right")
        return f"Selected: the {words}. " + "; ".join(can).capitalize() + "."

    def _activated(self, tag) -> None:
        """A double-click on a thing with no words of its own: a panel's data or plot; else its settings."""
        if tag.kind == "panel":
            r = self.spec.panels[tag.panel]
            self.choose_plot(tag.panel) if r.frame() is not None else self.choose_data(tag.panel)
        else:
            self.side.setCurrentIndex(0)

    def _nudged(self, tag, dx_mm: float, dy_mm: float) -> None:
        k = self.canvas.mm_per_pixel()
        self._moved(tag, dx_mm / k, dy_mm / k)

    def _moved(self, tag, dx: float, dy: float) -> None:
        """A thing dragged by (dx, dy) figure pixels (y up): its place in the figure's Spec follows."""
        kind = tag.kind
        mm = self.canvas.mm_per_pixel()
        if kind == "figtext":
            t = self.spec.texts[int(tag.key)]
            t["x"], t["y"] = round(float(t.get("x", 0)) + dx * mm, 2), round(float(t.get("y", 0)) - dy * mm, 2)
            return self.touch("move")
        if not 0 <= tag.panel < len(self.spec.panels):
            return
        r = self.spec.panels[tag.panel]
        ax = self.canvas.axes_of(tag.panel)
        if ax is None:
            return
        renderer = render._renderer(self.fig)
        to_axes = ax.transAxes.inverted().transform

        def shifted_axes(point):
            px = ax.transAxes.transform(point)
            return [round(float(v), 4) for v in to_axes((px[0] + dx, px[1] + dy))]

        if kind == "letter":
            off = (list(r.letter_offset) + [0.0, 0.0])[:2]
            r.letter_offset = [round(off[0] + dx * mm, 2), round(off[1] - dy * mm, 2)]
        elif kind == "legend" and ax.get_legend() is not None:
            bb = ax.get_legend().get_window_extent(renderer)
            r.legend_at = [round(float(v), 4) for v in to_axes((bb.x0 + dx, bb.y1 + dy))]
        elif kind == "scalebar":
            r.scale_bar = {**r.scale_bar, "at": shifted_axes(r.scale_bar.get("at") or (0.995, 0.02))}
        elif kind == "stimbar":
            bar = r.stim_bars[int(tag.key)]
            bar["y"] = float(round(float(bar.get("y", 1.02)) + dy / max(float(ax.bbox.height), 1.0), 3))
        elif kind == "annotation":
            a = r.annotations[int(tag.key)]
            kx, ky = ("x2", "y2") if a.get("kind") == "arrow" and a.get("x2") is not None else ("x", "y")
            px = ax.transData.transform((float(a[kx]), float(a[ky])))
            a[kx], a[ky] = (float(f"{v:.5g}") for v in ax.transData.inverted().transform((px[0] + dx, px[1] + dy)))
        elif kind == "bracket":
            line = next((a for t, a in render.things(self.fig)
                         if (t.kind, t.panel, t.key, t.part) == ("bracket", tag.panel, tag.key, "line")), None)
            if line is None:
                return
            px = ax.transData.transform((0.0, float(line.get_ydata()[1])))
            self.set_bracket(tag, "y", float(f"{ax.transData.inverted().transform((px[0], px[1] + dy))[1]:.5g}"))
        elif kind == "panel":
            self._panel_moved(tag.panel, dx, dy)
        else:
            return
        self.touch("move")

    def _panel_moved(self, k: int, dx: float, dy: float) -> None:
        """A panel dragged: an inset moves inside its panel, a freely placed one on the page (landing on the snap
        grid), one of the grid changes places with the panel it is dropped on."""
        spec, r, mm = self.spec, self.spec.panels[k], self.canvas.mm_per_pixel()
        w_mm, h_mm = spec.size_mm()
        if page.is_inset(spec.panels, k):
            parent = self.canvas.axes_of(r.inset_of)
            rect = list(r.inset_rect) if len(r.inset_rect) == 4 else page.inset_default()
            if parent is not None:
                rect[0] = round(rect[0] + dx / max(parent.bbox.width, 1.0), 3)
                rect[1] = round(rect[1] + dy / max(parent.bbox.height, 1.0), 3)
                r.inset_rect = rect
        elif len(r.rect) == 4:
            x, y, w, h = r.rect
            r.rect = [round(v, 2) for v in page.clamp(page.snap([x + dx * mm, y - dy * mm, w, h], spec.snap_mm),
                                                    w_mm, h_mm)]
        else:
            o = self.canvas.outline(Tag("panel", k))
            if o is None:
                return
            cx, cy = (o[0] + o[2]) / 2 + dx, (o[1] + o[3]) / 2 + dy
            for d in self.canvas.drawn:
                j = getattr(d.axes[0], "_cw_panel", -1) if d.axes else -1
                box_ = render.panel_outline(self.fig, d)
                if j not in (-1, k) and d.box and box_ and box_[0] <= cx <= box_[2] and box_[1] <= cy <= box_[3]:
                    self.swap_panels(k, j)
                    self.canvas.select(Tag("panel", j), tell=False)
                    self.current = j
                    return
            self.statusBar().showMessage("Dropped on no other panel: it stays in its cell. To put it anywhere, "
                                         "right-click it and Place the panels freely.", 6000)

    def swap_panels(self, i: int, j: int) -> None:
        """Two panels change places: what each shows goes to the other's place (cell, span, letter)."""
        p = self.spec.panels
        place = {k: {f: copy.deepcopy(getattr(p[k], f)) for f in PLACE_FIELDS} for k in (i, j)}
        p[i], p[j] = p[j], p[i]
        for k in (i, j):
            for f, v in place[k].items():
                setattr(p[k], f, v)

    def _resized(self, tag, handle: str, dx: float, dy: float) -> None:
        k = tag.panel
        spec, r, mm = self.spec, self.spec.panels[k], self.canvas.mm_per_pixel()
        if page.is_inset(spec.panels, k):
            parent = self.canvas.axes_of(r.inset_of)
            if parent is None:
                return
            x, y, w, h = list(r.inset_rect) if len(r.inset_rect) == 4 else page.inset_default()
            fx, fy = dx / max(parent.bbox.width, 1.0), dy / max(parent.bbox.height, 1.0)
            if "w" in handle:
                x, w = x + fx, w - fx
            if "e" in handle:
                w += fx
            if "s" in handle:
                y, h = y + fy, h - fy
            if "n" in handle:
                h += fy
            r.inset_rect = [round(x, 3), round(y, 3), round(max(w, 0.08), 3), round(max(h, 0.08), 3)]
        elif len(r.rect) == 4:
            x, y, w, h = r.rect
            ddx, ddy = dx * mm, -dy * mm                 # the page's y runs down
            if "w" in handle:
                x, w = x + ddx, w - ddx
            if "e" in handle:
                w += ddx
            if "n" in handle:
                y, h = y + ddy, h - ddy
            if "s" in handle:
                h += ddy
            r.rect = [round(v, 2) for v in page.clamp(page.snap([x, y, w, h], spec.snap_mm), *spec.size_mm())]
        else:
            return
        self.touch("resize")

    def remove_thing(self, tag) -> None:
        """What is selected taken off the figure (Delete, or its Remove): Undo brings it back."""
        if tag is None:
            return
        kind = tag.kind
        if kind == "figtext":
            del self.spec.texts[int(tag.key)]
        elif kind == "panel":
            if len(self.spec.panels) < 2:
                self.statusBar().showMessage("A figure keeps at least one panel; its data can be removed (Open).", 5000)
                return
            return self.remove_panel(tag.panel)
        else:
            r = self.spec.panels[tag.panel]
            if kind == "annotation":
                del r.annotations[int(tag.key)]
            elif kind == "stimbar":
                del r.stim_bars[int(tag.key)]
            elif kind == "refline":
                del r.ref_lines[int(tag.key)]
            elif kind == "marker":
                del r.markers[int(tag.key)]
            elif kind == "legend":
                r.legend, r.legend_at = "none", []
            elif kind == "title":
                r.title = ""
            elif kind == "letter":
                r.letter = "-"
            elif kind == "bracket":
                self.set_bracket(tag, "hidden", True)
            elif kind == "scalebar":
                if tag.key == "command":
                    r.command = {**r.command, "show": False}
                else:
                    r.scale_bars = False
            elif kind == "command":
                r.command = {**r.command, "show": False}
            elif kind == "vmlabel":
                r.show_vm = False
            elif kind == "series":
                r.series = {**r.series, tag.key: {**r.series.get(tag.key, {}), "hidden": True}}
            else:
                self.statusBar().showMessage(f"The {elements.KIND_WORDS.get(kind, kind).lower()} cannot be removed; "
                                             "its settings are on the right.", 5000)
                return
        self.canvas.select(None)
        self.touch("remove")

    def _thing_menu(self, tag, where) -> None:
        """Right-click: what can be done with the thing under the mouse (or with the page)."""
        m = QtWidgets.QMenu(self)
        m.setToolTipsVisible(True)
        if tag is None:
            a = m.addAction("Add a panel...", self.add_panel_dialog)
            a.setEnabled(len(self.spec.panels) < self.MOST or any(r.frame() is None for r in self.spec.panels))
            m.addAction("Add text on the page", self.add_page_text)
        elif tag.kind == "panel":
            self._panel_actions(m, tag.panel)
        else:
            words = elements.KIND_WORDS.get(tag.kind, tag.kind).lower()
            if self.words_of(tag) is not None:
                m.addAction(f"Rewrite the {words}...", lambda: self.canvas.edit_text(tag))
            if tag.kind not in ("xaxis", "yaxis", "xlabel", "ylabel"):
                gone = "Hide" if tag.kind in ("bracket", "series", "vmlabel") else "Remove"
                m.addAction(f"{gone} the {words}", lambda: self.remove_thing(tag))
            if 0 <= tag.panel < len(self.spec.panels):
                sub = m.addMenu(f"Panel {self.letter_of(tag.panel) or tag.panel + 1}")
                self._panel_actions(sub, tag.panel)
        m.addSeparator()
        m.addAction("Copy the figure as a picture", self.copy_picture)
        m.addAction("Export...", self.export)
        m.exec_(where)

    def _canvas_menu(self, m: QtWidgets.QMenu, pos) -> None:
        """The right-click menu at a point of the canvas, filled into ``m`` (kept for callers that build it)."""
        tag = self.canvas.hit(pos)
        if tag is not None and 0 <= tag.panel < len(self.spec.panels):
            self._panel_actions(m, tag.panel)
        else:
            a = m.addAction("Add a panel...", self.add_panel_dialog)
            a.setEnabled(len(self.spec.panels) < self.MOST)
        m.addSeparator()
        m.addAction("Copy the figure as a picture", self.copy_picture)
        m.addAction("Export...", self.export)

    def copy_picture(self) -> None:
        was = self.canvas.selection
        self.canvas.select(None, tell=False)          # the picture without the selection's outline
        self.canvas.repaint()
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())
        self.canvas.select(was, tell=False)
        self.statusBar().showMessage("The figure is on the clipboard.", 3000)

    # -- things added
    def add_annotation(self, k: int, kind: str = "text") -> None:
        """Words (or an arrow with words) in the middle of panel ``k``, selected, ready to drag and rewrite."""
        r = self.spec.panels[k]
        ax = self.canvas.axes_of(k)
        if ax is None:
            return
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        cx, cy = float(f"{(x0 + x1) / 2:.4g}"), float(f"{(y0 + y1) / 2:.4g}")
        a = {"kind": kind, "x": cx, "y": cy, "text": "text"}
        if kind == "arrow":
            a.update(x2=float(f"{cx + 0.15 * (x1 - x0):.4g}"), y2=float(f"{cy + 0.15 * (y1 - y0):.4g}"))
        r.annotations = list(r.annotations) + [a]
        self.touch("add text")
        self.redraw_now()
        self.canvas.select(Tag("annotation", k, str(len(r.annotations) - 1)))

    def add_page_text(self) -> None:
        w, h = self.spec.size_mm()
        self.spec.texts = list(self.spec.texts) + [{"x": round(w * 0.4, 1), "y": round(h * 0.03, 1), "text": "text"}]
        self.touch("add text")
        self.redraw_now()
        self.canvas.select(Tag("figtext", -1, str(len(self.spec.texts) - 1)))

    def add_stim_bar(self, k: int) -> None:
        """A stimulus bar over panel ``k``: over the middle fifth of x to start with (blue light), selected."""
        r = self.spec.panels[k]
        ax = self.canvas.axes_of(k)
        if ax is None:
            return
        x0, x1 = ax.get_xlim()
        last = r.stim_bars[-1] if r.stim_bars else {}
        bar = {"start": float(f"{x0 + 0.4 * (x1 - x0):.4g}"), "end": float(f"{x0 + 0.6 * (x1 - x0):.4g}"),
               **{key: last[key] for key in ("nm", "colour", "kind", "y", "height") if key in last}}
        if not last:
            bar["nm"] = traces.LIGHT_NM
        r.stim_bars = list(r.stim_bars) + [bar]
        self.touch("add a stimulus bar")
        self.redraw_now()
        self.canvas.select(Tag("stimbar", k, str(len(r.stim_bars) - 1)))

    @staticmethod
    def set_stim_light(bar: dict, choice: str) -> None:
        for key in ("nm", "colour", "kind"):
            bar.pop(key, None)
        if choice == "electrical":
            bar["kind"] = "electrical"
        elif choice:
            bar["nm"] = float(choice)
        else:
            bar["colour"] = "#555555"

    def match_stim_bars(self, k: int, i: int) -> None:
        r = self.spec.panels[k]
        keys = ("nm", "colour", "kind", "y", "height")
        ref = {key: r.stim_bars[i][key] for key in keys if key in r.stim_bars[i]}
        for bar in r.stim_bars:                       # (the one matched is among them: its values read first)
            for key in keys:
                bar.pop(key, None)
            bar.update(ref)
        self.touch("stimulus bars")

    def add_inset(self, k: int) -> None:
        """A small panel inside panel ``k``, empty, selected: choose its data next."""
        if len(self.spec.panels) >= self.MOST:
            return
        self.spec.panels.append(Recipe(inset_of=k, inset_rect=page.inset_default(), legend="none"))
        self.current = len(self.spec.panels) - 1
        self._load_controls()
        self.touch("add an inset")
        self.redraw_now()
        self.canvas.select(Tag("panel", self.current))

    def set_inset(self, k: int, parent: int) -> None:
        r = self.spec.panels[k]
        r.inset_of, r.inset_rect = parent, (r.inset_rect if parent >= 0 else [])
        self._load_controls()
        self.touch("inset")

    def set_inset_part(self, k: int, i: int, value: float) -> None:
        r = self.spec.panels[k]
        rect = list(r.inset_rect) if len(r.inset_rect) == 4 else page.inset_default()
        rect[i] = round(float(value), 3)
        r.inset_rect = rect

    def set_rect_part(self, k: int, i: int, value: float) -> None:
        r = self.spec.panels[k]
        rect = list(r.rect)
        rect[i] = float(value)
        r.rect = [round(v, 2) for v in page.clamp(rect, *self.spec.size_mm())]

    def set_span(self, k: int, columns: int | None, rows: int | None) -> None:
        r = self.spec.panels[k]
        c, rw = page.spans(self.spec.panels)[k]
        r.span = [int(columns or c), int(rows or rw)]

    def place_freely(self, k: int, on: bool) -> None:
        """Free placement is the whole figure's: every panel keeps where it is and can then be dragged (were one
        panel alone taken out of the grid, the others would flow into its cell)."""
        self.all_free() if on else self.all_grid()
        self.inspector.show_for(self.canvas.selection)

    def all_free(self) -> None:
        w, h = self.spec.size_mm()
        page.to_free(self.spec)
        if self.spec.size in SIZES:                   # the page keeps its height (no grid rows set it any more)
            self.spec.height_mm = round(h, 1)
        self.touch("placement")
        self.statusBar().showMessage("Every panel can now be dragged and resized (pull a corner); Arrange puts "
                                     "them back in the grid.", 7000)

    def all_grid(self) -> None:
        page.to_grid(self.spec)
        self.spec.height_mm = 0.0
        self.touch("placement")

    def arrange(self, how: str) -> None:
        """The freely placed panels aligned, sized or spaced; the selected one (else the first) leads."""
        free = [k for k, r in enumerate(self.spec.panels) if len(r.rect) == 4]
        sel = self.canvas.selection
        if sel is not None and sel.kind == "panel" and sel.panel in free:
            free.remove(sel.panel)
            free.insert(0, sel.panel)
        new = page.arrange([self.spec.panels[k].rect for k in free], how)
        for k, rect in zip(free, new):
            self.spec.panels[k].rect = [round(v, 2) for v in page.clamp(rect, *self.spec.size_mm())]
        self.touch("arrange")

    def use_template(self, key: str) -> None:
        """The panels in a template's arrangement; panels are added (empty) when the template has more places."""
        t = page.TEMPLATES[key]
        own = [k for k in range(len(self.spec.panels)) if not page.is_inset(self.spec.panels, k)]
        for _ in range(max(0, min(t.n, self.MOST) - len(own))):
            self.spec.panels.append(Recipe())
        page.apply_template(self.spec, key)
        self._load_controls()
        self.statusBar().showMessage(f"{t.name}: {t.line}", 6000)
        self._widen()
        self.touch("template")

    def set_columns(self, n: int, touch: bool = False) -> None:
        self.spec.columns = max(1, int(n))
        for r in self.spec.panels:
            if r.span and r.span[0] > self.spec.columns:
                r.span = [self.spec.columns, r.span[1] if len(r.span) > 1 else 1]
        if touch:
            self.touch("columns")

    # -- the page
    def width_of(self, size: str) -> float | None:
        if self.spec.journal:
            return journals.width_mm(self.spec.journal, size)
        return SIZES[size][0]

    def _fill_sizes(self) -> None:
        self.size.blockSignals(True)
        self.size.clear()
        for k in SIZES:
            w = self.width_of(k)
            if w:
                self.size.addItem(f"{k[:1].upper() + k[1:]} column ({w:g} mm)", k)
        if self.spec.size not in SIZES:
            w, h = self.spec.size_mm()
            self.size.addItem(f"{w:g} × {h:g} mm", self.spec.size)
        i = self.size.findData(self.spec.size)
        if i < 0:                                # the journal has no such column: its widest
            self.spec.size = self.size.itemData(self.size.count() - 1)
            i = self.size.count() - 1
        self.size.setCurrentIndex(i)
        self.size.blockSignals(False)

    def _size_chosen(self, _index: int = 0) -> None:
        if self._loading or not self.size.currentData():
            return
        self.spec.size = self.size.currentData()
        self._changed()

    def set_journal(self, key: str) -> None:
        """The journal's widths, font and text sizes; '' leaves the sizes as they are and checks nothing."""
        self.spec.journal = key or ""
        if key:
            self.spec.style.apply_journal(key)
            if self.spec.size in SIZES and not journals.width_mm(key, self.spec.size):
                self.spec.size = "double"
        self._fill_sizes()
        QtCore.QTimer.singleShot(0, lambda: self.inspector.show_for(self.inspector.tag))

    def set_width(self, size: str) -> None:
        if size == "custom":
            w, h = self.spec.size_mm()
            self.spec.size, self.spec.height_mm = f"{w:g}x{h:g}", 0.0
        else:
            self.spec.size = size
        self._fill_sizes()
        QtCore.QTimer.singleShot(0, lambda: self.inspector.show_for(self.inspector.tag))

    def set_page(self, width: float | None, height: float | None) -> None:
        w, h = self.spec.size_mm()
        if self.spec.size in SIZES:
            self.spec.height_mm = float(height or 0.0)
        else:
            self.spec.size = f"{float(width or w):g}x{float(height or h):g}"
        self._fill_sizes()

    def set_theme(self, name: str) -> None:
        self.spec.style.apply_theme(name)
        QtCore.QTimer.singleShot(0, lambda: self.inspector.show_for(self.inspector.tag))

    def set_legend_place(self, k: int, place: str) -> None:
        r = self.spec.panels[k]
        if place != "dragged":
            r.legend, r.legend_at = place, []
        if place == "none":
            self.canvas.select(None)

    def set_group_colour(self, group: str, colour: str) -> None:
        """A group's colour wherever it is drawn in this figure, and in the project's later figures."""
        s = self.spec.style
        s.group_colours = {**s.group_colours, group: colour}
        for r in self.spec.panels:                   # no panel keeps another colour for it
            if group in r.series and "colour" in r.series[group]:
                mine = {key: v for key, v in r.series[group].items() if key != "colour"}
                r.series = {**{g: v for g, v in r.series.items() if g != group}, **({group: mine} if mine else {})}
        self._remember_colours()

    def _remember_colours(self) -> None:
        p = getattr(self.window_, "project", None)
        if p is not None:
            try:
                known = json.loads(p.meta("figure_colours") or "{}")
                p.set_meta("figure_colours", json.dumps({**known, **self.spec.style.group_colours}))
            except Exception:
                pass

    def _known_colours(self) -> dict:
        p = getattr(self.window_, "project", None)
        try:
            return json.loads(p.meta("figure_colours") or "{}") if p is not None else {}
        except Exception:
            return {}

    def _settle_colours(self) -> bool:
        """Every group drawn with others gets one colour for the whole figure (the project's when it has met the
        group before, else the palette's next), so a condition is the same colour in every panel. True when a
        group was given one."""
        s = self.spec.style
        known = self._known_colours()
        seen: dict[str, list] = {}                   # group -> the panels that draw it (with other groups)
        for d in self._drawn:
            if d.prepared is None:
                continue
            src = d.prepared.points if len(d.prepared.points) else d.prepared.rows
            groups = render.ordered([g for g in dict.fromkeys(src.group) if g and g != "all"], d.recipe)
            if len(groups) >= 2:
                for g in groups:
                    seen.setdefault(g, []).append((d, groups))
        new = False
        for g, where in seen.items():
            # one colour for the figure once a group is in two panels, or the project has met it before
            if g in s.group_colours or not (len(where) > 1 or g in known):
                continue
            d, groups = where[0]
            s.group_colours = {**s.group_colours, g: known.get(g) or render.looks(d.recipe, s, groups)[g].colour}
            new = True
        if new:
            self._remember_colours()
        return new

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
        # what is measured: a numeric column that is not the time (or x), the cell or the animal
        r.y = r.y if r.y in frame.columns else next(
            (c for c in nums if c not in (r.x, g["x"], r.cell, r.animal)), next(iter(nums), ""))
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

    def open_figure(self, path: str | None = None, append: bool = False) -> None:
        """A figure saved by Clampwise opened in place of this one; ``append``: its panels added to this one."""
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
        if append:
            room = self.MOST - len(self.spec.panels)
            base = len(self.spec.panels)
            for r in spec.panels[:max(room, 0)]:
                r.rect, r.letter, r.letter_offset = [], "", []
                r.inset_of = r.inset_of + base if r.inset_of >= 0 else -1
                self.spec.panels.append(r)
            self.current = min(base, len(self.spec.panels) - 1)
            if len(spec.panels) > room:
                self.statusBar().showMessage(f"{room} of its {len(spec.panels)} panels added: a figure holds "
                                             f"{self.MOST} at most.", 6000)
        else:
            self.spec, self.current = spec, 0
        self.canvas.select(None)
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
        fresh = r.question != "example" or r.source.get("kind") != "traces"
        if r.question != "example":                 # a new example panel starts with scale bars
            r.scale_bars = True
        r.data, r.source = pack(frame), traces.source(sets, win, table, overlay)
        r.question, r.plot, r.level = "example", "trace", "row"
        r.x, r.y, r.group, r.cell = frame.columns[0], frame.columns[1], "Trace", "Sweep"
        r.notes, r.markers, r.filters = notes, markers, []
        layouts.dress(r, frame, traces.extras(sets, self._load, win), fresh)
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
        m.setToolTipsVisible(True)

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

        def results_of(name: str) -> dict:           # another file of the cell, by its name
            other = sel.files.get(name)
            try:
                found = project.last_run(other) if other else None
            except Exception:
                found = None
            return found.results if found is not None else {}

        sel.results_of = results_of
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
        spec.style, spec.journal = self.spec.style, self.spec.journal      # the look and the journal stay
        self.spec, self.current = spec, 0
        self._fill_sizes()
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
            m.addAction(f"Use “{name}”", lambda d=d, n=name: self.apply_style(d, n))
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
            self.statusBar().showMessage(f"This look is saved as the style “{name.strip()}” (Styles lists it).", 5000)

    def export_styles(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the styles", "figure styles.cwstyles",
                                                            "Clampwise figure styles (*.cwstyles)")
        if path:
            Path(path).write_text(json.dumps({"format": "clampwise-figure-styles", "styles": self._styles()},
                                             indent=1), encoding="utf-8")
            self.statusBar().showMessage(f"{len(self._styles())} saved style(s) written to {Path(path).name}.", 5000)

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
        self.statusBar().showMessage(f"{len(d.get('styles', {}))} style(s) read from {Path(path).name}; Styles lists "
                                     "them.", 5000)

    def apply_style(self, d: dict, name: str = "") -> None:
        before = self.spec.to_json()
        self.spec.style = Style(**{k: v for k, v in d.items() if k in Style.__dataclass_fields__})
        self._load_controls()
        self._changed()
        same = self.spec.to_json() == before
        self.statusBar().showMessage((f"The style “{name}”" if name else "The style") +
                                     (" is this figure's look already." if same else " is applied (Undo takes it "
                                                                                    "back)."), 5000)

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
        self.canvas.look = render.rc(self.spec.style)
        try:
            self.fig, self._drawn = render.build(self.spec, self.fig)
            if self._settle_colours():               # a group met for the first time: its colour, then once more
                self.fig, self._drawn = render.build(self.spec, self.fig)
        except Exception as exc:                     # a combination that cannot be drawn says why
            self.fig.clear()
            self.fig.text(0.5, 0.5, f"Cannot draw this: {exc}", ha="center", va="center", wrap=True)
            self._drawn = []
        self._fit_canvas()
        self.canvas.set_drawn(self._drawn)
        self._describe()
        self._refresh_series()              # the Groups tab follows the groups the panel now draws
        self.inspector.refresh()
        if self.side.currentIndex() == 1:
            self.dataview.refresh()
        any_data = any(r.frame() is not None for r in self.spec.panels)
        full = len(self.spec.panels) >= self.MOST and all(r.frame() is not None for r in self.spec.panels)
        self.add_button.setEnabled(not full)
        self.add_button.setToolTip(f"A figure holds {self.MOST} panels at most: remove one first." if full else
                                   "A new panel: choose what it shows (traces, a graph, a kept measurement, a "
                                   "table, a file), then how it is drawn" + ("" if any_data else
                                                                            ". The page is empty: start here."))

    def redraw_now(self) -> None:
        """Drawn at once (not after the short pause a typed change waits), so what was just added can be
        selected."""
        self.timer.stop()
        self.redraw()

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
        j = journals.JOURNALS.get(self.spec.journal)
        self.size_note.setText(f"· {wmm:.0f} × {hmm:.0f} mm, {n} panel{'s' if n > 1 else ''}"
                               + (f" · {j.name}" if j else ""))
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
        last = ""
        for k, d in enumerate(self._drawn):
            # a panel with no letter of its own belongs to the lettered one before it
            last = d.letter or last
            tag = (f"({d.letter}) " if d.letter else f"({last}, continued) " if last else
                   f"({chr(ord('A') + k)}) ") if many else ""
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
        try:                                         # the page against the journal's rules (or plain legibility)
            self.findings = render.check(self.spec, built=(self.fig, self._drawn))
        except Exception:
            self.findings = []
        warns += [(f"Panel {f.where}: " if f.where else "") + f.text for f in self.findings if f.serious]
        for w in warns:
            b = widgets.Badge("warn", w, w)
            b.setWordWrap(True)
            lay.addWidget(b)

    def caption_text(self) -> str:
        return self.caption.toPlainText()

    def copy_caption(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.caption_text())

    def export(self, path: str | None = None, dpi: int | None = None, outline: bool = False) -> Path | None:
        """The figure written to a file. With no ``path``: the export dialog first (what the check found, the
        format, the resolution, text as shapes), then where to save."""
        if not path:
            try:
                found = render.check(self.spec, built=(self.fig, self._drawn))
            except Exception:
                found = []
            dlg = ExportDialog(self, self.spec, found)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return None
            ext = dlg.format.currentData()
            dpi = dlg.dpi.value() if ext in ("tiff", "png") else None
            outline = dlg.outline.isChecked() and dlg.outline.isEnabled()
            filt = {"pdf": "PDF (*.pdf)", "svg": "SVG (*.svg)", "eps": "EPS (*.eps)", "tiff": "TIFF (*.tif *.tiff)",
                    "png": "PNG (*.png)"}[ext]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the figure", f"figure.{ext}", filt)
            if not path:
                return None
            if not Path(path).suffix:
                path = f"{path}.{ext}"
        try:
            out = render.export(self.spec, path, dpi, outline)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not export", f"{Path(path).name}: {exc}")
            return None
        w, h = self.spec.size_mm()
        self.statusBar().showMessage(f"Saved {out.name} at {w:.1f} × {h:.1f} mm; it opens here again, editable "
                                     "(Open, a saved figure).")
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

