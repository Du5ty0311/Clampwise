"""Choosing what a panel shows, in two plain steps: its data, then how the data are drawn.

``DataChooser`` lists everything Clampwise can draw a panel from, in one place: the selected recording's traces,
one of its graphs from the Figures screen, a measurement kept in the Workbench, a table of the project's results,
a spreadsheet, a picture of a plot to digitize, another saved figure. ``PlotChooser`` shows every kind of plot as a
small picture, grouped by what the panel is to show; the pictures are drawn by the code that draws the figure.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import thumbs
from ..figure.recipe import PLOTS
from . import theme, widgets
from .widgets import box, label

# (key, what it is, what happens next)
SOURCES = (
    ("traces", "Traces of a recording", "Pick the sweeps to draw (averaged, every one, or the most typical), with "
     "the light's bar, scale bars and the command under them"),
    ("typical", "The selected recording's typical response", "The sweep closest to the mean of its responses, "
     "aligned on the stimulus"),
    ("graph", "A graph of the selected recording", "Any graph the Figures screen draws for it, to edit"),
    ("workbench", "A measurement kept in the Workbench", "Its table: every measure, sweep by sweep"),
    ("project", "A table of this project's results", "Every recording's numbers, or one analysis's own table"),
    ("file", "A CSV or Excel file", "A spreadsheet of your own"),
    ("picture", "A picture of a plot (digitize it)", "Read the points off a published figure: an estimate"),
    ("figure", "A figure saved by Clampwise", "A .cwfig file, or any PDF, SVG or PNG exported from here: its panels "
     "are added to this figure"),
)


class DataChooser(QtWidgets.QDialog):
    """Where a panel's data come from. ``available``: {source key: why it cannot be used now} for the sources
    that cannot (they are listed greyed, with the reason)."""

    def __init__(self, parent, unavailable: dict[str, str], extra: dict[str, list[tuple[str, object]]],
                 title: str = "Choose the panel's data") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.controls_name = "Data chooser"
        self.choice: tuple[str, object] | None = None
        self.extra = extra                         # source key -> [(words, what to use)]: its own list to pick from
        lay = box("v", "s", "m")
        lay.addWidget(label("What does this panel show?", "heading"))
        split = box("h", "m")
        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Kind of data")
        self.list.setMinimumWidth(300)
        for key, words, _what in SOURCES:
            item = QtWidgets.QListWidgetItem(words)
            item.setData(QtCore.Qt.UserRole, key)
            why = unavailable.get(key, "")
            if why:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                item.setToolTip(why)
                item.setText(f"{words}  ({why})")
            self.list.addItem(item)
        split.addWidget(self.list, 1)
        right = box("v", "s")
        self.what = label("", "", wrap=True)
        right.addWidget(self.what)
        self.sub = QtWidgets.QListWidget()
        self.sub.setAccessibleName("Which one")
        self.sub.itemDoubleClicked.connect(lambda _i: self.accept())
        right.addWidget(self.sub, 1)
        holder = QtWidgets.QWidget()
        holder.setLayout(right)
        holder.setMinimumWidth(360)
        split.addWidget(holder, 1)
        lay.addLayout(split, 1)
        row = box("h", "s")
        row.addStretch(1)
        row.addWidget(widgets.button("Cancel", self.reject, "Leave the panel as it is"))
        self.ok = widgets.primary("Use it", self.accept)
        row.addWidget(self.ok)
        lay.addLayout(row)
        self.list.currentRowChanged.connect(self._picked)
        self.sub.currentRowChanged.connect(lambda _r: self._enable())
        self.list.itemDoubleClicked.connect(lambda _i: self.accept() if self.ok.isEnabled() else None)
        first = next((i for i in range(self.list.count())
                      if self.list.item(i).flags() & QtCore.Qt.ItemIsEnabled), 0)
        self.list.setCurrentRow(first)
        widgets.resizable(self, lay, "data_chooser", (760, 440))

    def _picked(self, row: int) -> None:
        key = self.list.item(row).data(QtCore.Qt.UserRole) if row >= 0 else ""
        self.what.setText(next((w for k, _n, w in SOURCES if k == key), ""))
        self.sub.clear()
        for words, _use in self.extra.get(key, []):
            self.sub.addItem(words)
        self.sub.setVisible(key in self.extra)
        if self.sub.count():
            self.sub.setCurrentRow(0)
        self._enable()

    def _enable(self) -> None:
        item = self.list.currentItem()
        key = item.data(QtCore.Qt.UserRole) if item is not None else ""
        usable = item is not None and bool(item.flags() & QtCore.Qt.ItemIsEnabled)
        needs = key in self.extra
        on = usable and (not needs or self.sub.currentRow() >= 0)
        self.ok.setEnabled(on)
        self.ok.setToolTip("Use this for the panel" if on else item.toolTip() if item is not None and not usable
                           else "Pick one in the list on the right." if needs else "Pick a kind of data.")

    def accept(self) -> None:
        item = self.list.currentItem()
        if item is None or not self.ok.isEnabled():
            return
        key = item.data(QtCore.Qt.UserRole)
        use = self.extra[key][self.sub.currentRow()][1] if key in self.extra else None
        self.choice = (key, use)
        super().accept()


class PlotChooser(QtWidgets.QDialog):
    """Every kind of plot as a small picture, grouped by what the panel is to show; one is picked by a click
    (double-click, or Use it)."""

    def __init__(self, parent, current: str = "", title: str = "How should it be drawn?") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.controls_name = "Plot chooser"
        self.plot = current
        lay = box("v", "s", "m")
        lay.addWidget(label("How should the numbers be drawn?", "heading"))
        self.says = label("", "muted", wrap=True)
        self.says.setMinimumHeight(34)
        lay.addWidget(self.says)
        page_ = QtWidgets.QWidget()
        inner = box("v", "s")
        self.buttons: dict[str, QtWidgets.QToolButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for _key, words, plots in thumbs.catalogue():
            inner.addWidget(label(words, "section"))
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(theme.SPACE["xs"])
            for i, (plot, name, what) in enumerate(plots):
                b = QtWidgets.QToolButton()
                b.setCheckable(True)
                b.setProperty("oneOfSeveral", True)
                b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
                b.setText(name if len(name) <= 24 else name[:22] + "…")
                b.setAccessibleName(f"Plot: {name}")
                b.setToolTip(f"{name}: {what}")
                pix = QtGui.QPixmap()
                try:
                    pix.loadFromData(thumbs.png(plot))
                except Exception:                    # a picture that cannot be made leaves its name
                    pass
                if not pix.isNull():
                    b.setIcon(QtGui.QIcon(pix))
                    b.setIconSize(pix.size())
                b.setFixedSize(150, 132)
                b.setChecked(plot == current)
                b.clicked.connect(lambda _=False, p=plot: self._pick(p))
                b.installEventFilter(self)
                b._plot = plot
                group.addButton(b)
                grid.addWidget(b, i // 5, i % 5)
                self.buttons[plot] = b
            holder = QtWidgets.QWidget()
            holder.setLayout(grid)
            inner.addWidget(holder, 0, QtCore.Qt.AlignLeft)
        inner.addStretch(1)
        page_.setLayout(inner)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(page_)
        lay.addWidget(scroll, 1)
        row = box("h", "s")
        row.addStretch(1)
        row.addWidget(widgets.button("Cancel", self.reject, "Leave the plot as it is"))
        self.ok = widgets.primary("Use it", self.accept)
        row.addWidget(self.ok)
        lay.addLayout(row)
        self._pick(current) if current in PLOTS else self._say("")
        widgets.resizable(self, lay, "plot_chooser", (860, 640))

    def _say(self, plot: str) -> None:
        self.says.setText(f"<b>{PLOTS[plot][0]}</b>: {PLOTS[plot][1]}." if plot in PLOTS else
                          "Click a picture; double-click to use it.")
        on = plot in PLOTS
        self.ok.setEnabled(on)
        self.ok.setToolTip("Draw the panel this way" if on else "Click a picture first.")

    def _pick(self, plot: str) -> None:
        self.plot = plot
        if plot in self.buttons:
            self.buttons[plot].setChecked(True)
        self._say(plot)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonDblClick and hasattr(obj, "_plot"):
            self._pick(obj._plot)
            self.accept()
            return True
        return super().eventFilter(obj, event)
