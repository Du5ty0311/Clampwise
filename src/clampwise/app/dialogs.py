"""Dialogs that edit something the project keeps (B1): the naming rules that
read a recording's identity from its folder and file names."""

from __future__ import annotations

import re
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ..core.identity import FIELDS, Rule, Rules
from . import widgets
from .widgets import box, label

WHERE = {"folders": "each folder above the file", "file": "the file's name"}


class RulesDialog(QtWidgets.QDialog):
    """The naming rules, editable, with what they read from a recording shown
    as they change. A rule is a regular expression with named groups:
    (?P<animal>...), (?P<slice>...), (?P<cell>...), (?P<condition>...)."""

    def __init__(self, parent, rules: Rules, examples: list[Path], project_folder: Path) -> None:
        super().__init__(parent)
        self.setWindowTitle("Naming rules")
        self.examples, self.folder = examples, project_folder
        lay = box("v", "m", "m")
        lay.addWidget(label("How Clampwise reads animal, slice, cell and condition from folder and file names. Each "
                            "rule is a pattern with named parts: (?P<animal>...), (?P<slice>...), (?P<cell>...), "
                            "(?P<condition>...). Folder rules are tried on each folder above the file, nearest first; "
                            "the folder above a slice or cell folder no rule names is the animal. What you type in "
                            "Recording identity always wins.", "muted", wrap=True))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Applies to", "Pattern"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        for r in rules.rules:
            self._add(r)
        self.table.itemChanged.connect(self._preview)
        lay.addWidget(self.table, 1)
        row = box("h", "s")
        row.addWidget(widgets.button("Add a rule", lambda: self._add(Rule("new rule", r"^(?P<cell>cell\s*\d+)$")),
                                     "A new folder rule to edit"))
        row.addWidget(widgets.button("Remove", self._remove, "Remove the selected rule"))
        row.addWidget(widgets.button("Defaults", self._defaults, "Put back the rules Clampwise starts with"))
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(label("What these rules read", "heading"))
        self.preview = label("", "", wrap=True)
        self.preview.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(self.preview)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        widgets.resizable(self, lay, "naming_rules", (820, 520))
        self._preview()

    def _add(self, rule: Rule) -> None:
        self.table.blockSignals(True)
        i = self.table.rowCount()
        self.table.insertRow(i)
        self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(rule.name))
        where = QtWidgets.QComboBox()
        where.addItems(list(WHERE))
        where.setCurrentText(rule.applies_to)
        where.setToolTip("; ".join(f"{k}: {v}" for k, v in WHERE.items()))
        where.currentIndexChanged.connect(self._preview)
        self.table.setCellWidget(i, 1, where)
        self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(rule.pattern))
        self.table.blockSignals(False)
        self._preview()

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._preview()

    def _defaults(self) -> None:
        self.table.setRowCount(0)
        for r in Rules().rules:
            self._add(r)

    def rules(self) -> tuple[list[Rule], str]:
        """(the rules in the table, the first problem with them or "")."""
        out, problem = [], ""
        for i in range(self.table.rowCount()):
            name = (self.table.item(i, 0).text() if self.table.item(i, 0) else "").strip() or f"rule {i + 1}"
            pattern = self.table.item(i, 2).text() if self.table.item(i, 2) else ""
            where = self.table.cellWidget(i, 1).currentText()
            try:
                groups = set(re.compile(pattern).groupindex)
            except re.error as exc:
                problem = problem or f"'{name}': the pattern is not valid ({exc})"
                continue
            if not groups & set(FIELDS):
                problem = problem or f"'{name}': it names no part (animal, slice, cell or condition)"
            out.append(Rule(name, pattern, where))
        return out, problem

    def _preview(self, *_) -> None:
        if not hasattr(self, "preview"):        # rows added before the preview exists
            return
        rules, problem = self.rules()
        parser = Rules(rules)
        lines = []
        for p in self.examples[:4]:
            ident = parser.parse(p, stop_at=(self.folder,))
            parts = [f"{k} <b>{getattr(ident, k)}</b>" for k in FIELDS if getattr(ident, k)]
            lines.append(f"{p.name}: " + (", ".join(parts) if parts else "<i>nothing</i>"))
        text = "<br>".join(lines) or "Select a recording in Files to see what the rules read from it."
        if problem:
            text = f"<b>{problem}</b><br>" + text
        self.preview.setText(text)

    def _accept(self) -> None:
        rules, problem = self.rules()
        if problem:
            QtWidgets.QMessageBox.warning(self, "Naming rules", problem)
            return
        self.result_rules = rules
        self.accept()
