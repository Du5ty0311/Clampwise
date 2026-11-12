"""The table a panel is drawn from, and the statistics on it, beside the figure (as a data sheet and its
results sit beside a graph).

Top: where the table came from and which of its columns the panel uses, in words. Middle: the table itself,
every row; the columns in use are marked in their headings, rows can be left out of the figure by hand (and
put back), and the table can be copied or saved. Bottom, for a comparison of groups: which test, which groups
are compared, whether the result is drawn on the graph, and the result in a sentence. Nothing is computed here:
the choices are written into the panel's recipe and the figure is drawn again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import stats as fstats
from . import theme, widgets
from .widgets import box, label

ROLES = (("y", "Y"), ("x", "X"), ("group", "Groups"), ("cell", "Cell"), ("animal", "Animal"), ("y2", "Second Y"),
         ("err", "Error"), ("panel_by", "Split by"), ("size_by", "Size"))


class FrameModel(QtCore.QAbstractTableModel):
    """A pandas table for a view: numbers to four significant figures, the columns a panel uses named in their
    headings, rows left out by hand greyed and struck through."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.frame = pd.DataFrame()
        self.roles: dict[str, str] = {}             # column -> the role it plays ("Y", "Groups")
        self.excluded: set[int] = set()

    def set_frame(self, frame: pd.DataFrame | None, roles: dict[str, str], excluded) -> None:
        self.beginResetModel()
        self.frame = frame if frame is not None else pd.DataFrame()
        self.roles, self.excluded = dict(roles), set(int(i) for i in excluded or [])
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:          # noqa: N802 (Qt's name)
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:       # noqa: N802
        return 0 if parent.isValid() else self.frame.shape[1]

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):      # noqa: N802
        if orientation == QtCore.Qt.Horizontal and 0 <= section < self.frame.shape[1]:
            name = str(self.frame.columns[section])
            if role == QtCore.Qt.DisplayRole:
                return f"{name}\n{self.roles[name]}" if name in self.roles else name
            if role == QtCore.Qt.ToolTipRole:
                return f"{name}: the panel's {self.roles[name]}" if name in self.roles else \
                    f"{name}: not used by this panel"
            if role == QtCore.Qt.FontRole and name in self.roles:
                f = QtGui.QFont()
                f.setBold(True)
                return f
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole and 0 <= section < len(self.frame):
            return str(section + 1)
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.frame) and 0 <= index.column() < self.frame.shape[1]):
            return None
        v = self.frame.iat[index.row(), index.column()]
        numeric = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
        out = index.row() in self.excluded
        if role == QtCore.Qt.DisplayRole:
            if numeric:
                return "" if not np.isfinite(v) else f"{v:.4g}"
            return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        if role == QtCore.Qt.TextAlignmentRole and numeric:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if role == QtCore.Qt.FontRole:
            f = theme.numeric_font() if numeric else QtGui.QFont()
            f.setStrikeOut(out)
            return f
        if role == QtCore.Qt.ForegroundRole and out:           # greyed, as the theme greys what is off
            return QtGui.QBrush(QtWidgets.QApplication.palette().color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))
        if role == QtCore.Qt.ToolTipRole and out:
            return "Left out of the figure by hand (right-click to put it back)"
        return None


class DataView(QtWidgets.QWidget):
    def __init__(self, builder) -> None:
        super().__init__()
        self.b = builder
        self._filling = False
        lay = box("v", "s")
        self.heading = label("The panel's data", "heading")
        lay.addWidget(self.heading)
        self.where = label("", "muted", wrap=True)
        lay.addWidget(self.where)
        self.roles = label("", "", wrap=True)
        lay.addWidget(self.roles)
        row = widgets.Wrap(
            widgets.button("Change data...", lambda: self.b.choose_data(self.b.current),
                           "Another table, other traces, a kept Workbench measurement, a file", quiet=True),
            widgets.button("Columns and rows...", lambda: self.b.show_all_settings("Data"),
                           "Which column is Y, X, the groups and the cells; which rows are kept; bins and baselines",
                           quiet=True), spacing="xs")
        self.columns_button = row.parts[1]
        lay.addWidget(row)
        self.table = QtWidgets.QTableView()
        self.model = FrameModel(self.table)
        self.table.setModel(self.model)
        self.table.setAccessibleName("The panel's table")
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setDefaultSectionSize(92)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        self.table.selectionModel().selectionChanged.connect(lambda *_a: self._enable())
        lay.addWidget(self.table, 3)
        self.count = label("", "hint", wrap=True)
        lay.addWidget(self.count)
        self.out_button = widgets.button("Leave the selected rows out", lambda: self.exclude(True), quiet=True)
        self.in_button = widgets.button("Put them back", lambda: self.exclude(False), quiet=True)
        self.copy_button = widgets.button("Copy the table", self.copy, quiet=True)
        self.save_button = widgets.button("Save the table...", self.save, quiet=True)
        lay.addWidget(widgets.Wrap(self.out_button, self.in_button, spacing="xs"))
        lay.addWidget(widgets.Wrap(self.copy_button, self.save_button, spacing="xs"))
        lay.addWidget(widgets.hairline())
        # the statistics on it
        self.stats_head = label("Statistics", "heading")
        lay.addWidget(self.stats_head)
        self.stats_box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()
        none = 0
        form.setContentsMargins(none, none, none, none)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.test = self._combo({k: f"{name}" + (f" ({what})" if k else "") for k, (name, what, _p) in
                                 fstats.TESTS.items()}, "Test")
        self.event = self._combo({}, "Counted")
        self.pairs = self._combo({"first": "Each group with the first", "all": "Every pair of groups"}, "Compare")
        self.says = self._combo({"p": "The exact p value", "stars": "Stars", "both": "Stars and the p value"},
                                "Brackets say")
        self.on_graph = QtWidgets.QCheckBox("Draw the result on the graph (brackets)")
        self.on_graph.setAccessibleName("Draw the result on the graph")
        self.paired = QtWidgets.QCheckBox("The same cells are in both groups (paired)")
        self.paired.setAccessibleName("Paired design")
        for caption, w, tip in (
                ("Counted", self.event, "A proportion: which category each group's share is of (connected, a "
                 "success). The share has Wilson's 95% interval; groups are compared by Fisher's exact test."),
                ("Test", self.test, "Whose p value the brackets and the caption carry; the other test of its kind "
                 "is reported beside it. Automatic: Welch's t test, or the paired t test for a paired design."),
                ("Compare", self.pairs, "Which groups are compared; p values are Holm-corrected over the "
                 "comparisons made"),
                ("Brackets say", self.says, "Journals ask for exact p values; some do not accept asterisks alone")):
            w.setToolTip(tip)
            form.addRow(caption, w)
        self.paired.setToolTip("A line joins each cell's values across the groups, and the paired tests are used")
        self.on_graph.setToolTip("A bracket over each pair compared, with its p value; drag a bracket to move it, "
                                 "double-click its text to write your own result")
        form.addRow(self.paired)
        form.addRow(self.on_graph)
        self.stats_box.setLayout(form)
        self.form = form
        lay.addWidget(self.stats_box)
        self.result = QtWidgets.QTextBrowser()
        self.result.setAccessibleName("Statistics result")
        self.result.setMinimumHeight(70)
        lay.addWidget(self.result, 1)
        self.copy_result = widgets.button("Copy the result", lambda: QtWidgets.QApplication.clipboard().setText(
            self.result.toPlainText()), "The result as text, for a legend or a Results section", quiet=True)
        lay.addWidget(self.copy_result)
        self.setLayout(lay)
        for w, sig in ((self.test, self.test.currentIndexChanged), (self.pairs, self.pairs.currentIndexChanged),
                       (self.event, self.event.currentIndexChanged),
                       (self.says, self.says.currentIndexChanged), (self.on_graph, self.on_graph.toggled),
                       (self.paired, self.paired.toggled)):
            sig.connect(lambda *_a: self._edited())

    def _combo(self, items: dict, name: str) -> QtWidgets.QComboBox:
        c = QtWidgets.QComboBox()
        c.setAccessibleName(name)
        c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(16)
        for k, words in items.items():
            c.addItem(words, k)
        return c

    # ---- showing the current panel
    def refresh(self) -> None:
        b = self.b
        r = b.recipe
        f = r.frame()
        self._filling = True
        try:
            letter = b.letter_of(b.current)
            self.heading.setText(f"Panel {letter}'s data" if letter else "The panel's data")
            self.where.setText(b.source_words(b.current))
            used = {getattr(r, key): words for key, words in ROLES if getattr(r, key, "")}
            if f is None:
                self.roles.setText("")
            else:
                said = [f"<b>{words}</b>: {getattr(r, key)}" for key, words in ROLES if getattr(r, key, "")]
                said.append("each point is " + ("a cell (its rows averaged)" if r.level == "cell" and r.cell else
                                                "a row of the table"))
                self.roles.setText(" · ".join(said))
            self.model.set_frame(f, used, r.excluded)
            n = 0 if f is None else len(f)
            out = len([i for i in r.excluded if f is not None and i in f.index])
            self.count.setText("" if f is None else f"{n} rows, {f.shape[1]} columns" + (
                f"; {out} left out by hand (greyed)" if out else "") + ". Columns in bold are the ones drawn.")
            share = f is not None and r.question == "counts" and r.plot == "proportion"
            compare = f is not None and (share or (r.question == "compare" and r.plot != "grouped_bar"))
            self.stats_head.setVisible(f is not None)
            self.stats_box.setVisible(compare)
            for w, on in ((self.event, share), (self.test, compare and not share), (self.paired, compare and not share)):
                w.setVisible(on)                      # a proportion has one test (Fisher's exact) and no pairing
                lab = self.form.labelForField(w)
                if lab is not None:
                    lab.setVisible(on)
            self.event.clear()
            if share and r.x in f.columns:
                from ..figure.prepare import _categories

                cats = [c for c in dict.fromkeys(_categories(f[r.x])) if c != ""]
                for c in cats:
                    self.event.addItem(c, c)
                self.event.setCurrentIndex(max(0, self.event.findData(r.event)))
            self.test.setCurrentIndex(max(0, self.test.findData(r.test)))
            self.pairs.setCurrentIndex(max(0, self.pairs.findData(r.pairs)))
            self.says.setCurrentIndex(max(0, self.says.findData(r.stats_text)))
            self.on_graph.setChecked(bool(r.stats))
            self.paired.setChecked(bool(r.paired))
            self.result.setVisible(f is not None)
            self.copy_result.setVisible(f is not None)
            self.result.setHtml(self._result_words(compare))
        finally:
            self._filling = False
        self._enable()

    def _result_words(self, compare: bool) -> str:
        import html

        b = self.b
        d = b.drawn_of(b.current)
        r = b.recipe
        if d is None or d.prepared is None:
            return "<p>No data yet.</p>"
        parts = [f"<p>{html.escape(d.prepared.caption)}</p>"]
        if compare:
            from ..figure.render import _units, comparisons_of

            comps = d.comparisons or comparisons_of(r, d.prepared)

            if comps:
                parts += [f"<p>{html.escape(c.sentence(_units(r.y)))}</p>" for c in comps]
            else:
                parts.append("<p>One group only: nothing to compare. Name the column that holds the groups "
                             "(Columns and rows...).</p>")
        for w in d.prepared.warnings:
            parts.append(f"<p><b>Note.</b> {html.escape(w)}</p>")
        return "".join(parts)

    def _enable(self) -> None:
        r = self.b.recipe
        has = r.frame() is not None
        rows = self.rows()
        gone = set(r.excluded or [])
        none = "This panel has no table yet: Change data..."
        for w, on, does, why in (
                (self.out_button, has and any(i not in gone for i in rows), "The selected rows are left out of the "
                 "figure and of its numbers (the caption says how many); they stay in the table, greyed",
                 none if not has else "Select rows of the table first (rows already left out are greyed)."),
                (self.in_button, has and any(i in gone for i in rows), "The selected rows are drawn again",
                 none if not has else "None of the selected rows is left out." if rows else
                 "Select rows that were left out (greyed) first."),
                (self.copy_button, has, "The whole table on the clipboard, to paste into a spreadsheet", none),
                (self.save_button, has, "The whole table as a CSV file", none),
                (self.columns_button, has, "Which column is Y, X, the groups and the cells; which rows are kept; "
                 "bins and baselines", none)):
            w.setEnabled(bool(on))
            w.setToolTip(does if on else why)

    def rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()} |
                      {i.row() for i in self.table.selectedIndexes()})

    # ---- what the person does
    def _edited(self) -> None:
        if self._filling:
            return
        r = self.b.recipe
        r.test = self.test.currentData() or ""
        if self.event.isVisibleTo(self) and self.event.count():
            r.event = self.event.currentData() or ""
        r.pairs = self.pairs.currentData() or "first"
        r.stats_text = self.says.currentData() or "p"
        r.stats = self.on_graph.isChecked()
        r.paired = self.paired.isChecked() or bool(fstats.TESTS.get(r.test, ("", "", None))[2])
        self.b.touch("statistics")

    def exclude(self, out: bool) -> None:
        r = self.b.recipe
        rows = self.rows()
        if not rows:
            return
        now = set(int(i) for i in r.excluded or [])
        r.excluded = sorted(now | set(rows)) if out else sorted(now - set(rows))
        self.b.touch("rows left out" if out else "rows put back")
        self.refresh()

    def _menu(self, pos) -> None:
        m = QtWidgets.QMenu(self.table)
        row = self.table.rowAt(pos.y())
        if row >= 0 and row not in self.rows():
            self.table.selectRow(row)
        self._enable()
        for b in (self.out_button, self.in_button, self.copy_button):
            a = m.addAction(b.text(), b.click)
            a.setEnabled(b.isEnabled())
            if not b.isEnabled():
                a.setToolTip(b.toolTip())
        m.setToolTipsVisible(True)
        m.exec_(self.table.viewport().mapToGlobal(pos))

    def copy(self) -> None:
        f = self.b.recipe.frame()
        if f is not None:
            QtWidgets.QApplication.clipboard().setText(f.to_csv(sep="\t", index=False))
            self.b.statusBar().showMessage(f"The table ({len(f)} rows) is on the clipboard.", 4000)

    def save(self) -> None:
        f = self.b.recipe.frame()
        if f is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the panel's table", "panel table.csv",
                                                        "CSV (*.csv)")
        if path:
            f.to_csv(path, index=False)
            self.b.statusBar().showMessage(f"Saved {path}.", 4000)
