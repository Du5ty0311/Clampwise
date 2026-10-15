"""Export (B14): the project's numbers to Excel or tidy CSV, every column defined
(the writing is in ``clampwise.export``, built from the registry)."""

from __future__ import annotations

import collections
import os
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...analysis.registry import TYPE_WORDS
from .. import widgets
from ..widgets import label
from .base import Screen


def open_path(path: Path) -> None:
    """Open a file or folder in the program Windows has for it (Excel for .xlsx)."""
    if hasattr(os, "startfile"):
        os.startfile(str(path))                                    # noqa: S606 (the user's own file)
    else:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


class ExportScreen(Screen):
    key = "export"
    title = "Export"
    purpose = "Export results as an Excel workbook or tidy CSV files, with every column defined and the quality " \
              "flags and analysis choices alongside the values."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.csv_button = widgets.button("Write CSV", self.write_csv, "Tidy CSV (one value per row), each table, the "
                                         "definitions and a manifest, in a folder beside the recordings")
        self.excel_button = widgets.primary("Send to Excel", self.send_to_excel,
                                            "Write the project's workbook beside its recordings and open it")
        for w in (self.csv_button, self.excel_button):
            self.add_action(w)
            w.setVisible(False)                 # the empty state's button is the one primary until there are results
        self.stack = QtWidgets.QStackedWidget()
        self.empty = widgets.EmptyState("Nothing to export yet", "Open a folder and analyze its recordings; their "
                                        "numbers, definitions and flags go to Excel from here.",
                                        [("Go to Files", lambda: window.go("files"), True)])
        self.stack.addWidget(self.empty)
        page = QtWidgets.QWidget()
        lay = widgets.box("v", "m")
        self.contents = label("", wrap=True)
        self.contents.setTextFormat(QtCore.Qt.RichText)
        self.contents.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        lay.addWidget(self.contents)
        self.last = label("", "muted", wrap=True)
        lay.addWidget(self.last)
        self.open_last = widgets.button("Open it again", lambda: self._last and open_path(self._last), quiet=True)
        self.open_last.setVisible(False)
        lay.addWidget(self.open_last, 0, QtCore.Qt.AlignLeft)
        lay.addStretch(1)
        page.setLayout(lay)
        self.stack.addWidget(page)
        self.body(self.stack)
        self._last: Path | None = None

    def on_shown(self) -> None:
        self.on_project()

    def on_project(self) -> None:
        if not self.is_current():
            return
        p = self.window.project
        rows = [r for r in p.summaries() if r["recording_type"]] if p is not None else []
        self.excel_button.setVisible(bool(rows))
        self.csv_button.setVisible(bool(rows))
        if not rows:
            self.stack.setCurrentWidget(self.empty)
            return
        types = collections.Counter(TYPE_WORDS.get(r["recording_type"], r["recording_type"]) for r in rows)
        sheets = "".join(f"<li><b>{t}</b>: {n} recording{'s' if n > 1 else ''}</li>" for t, n in types.items())
        self.contents.setText(
            f"<p>{len(rows)} analyzed recording{'s' if len(rows) > 1 else ''} of <b>{p.name}</b> go out, read from the "
            "project (nothing is re-analyzed).</p>"
            f"<p>The workbook has a sheet per recording type:</p><ul>{sheets}</ul>"
            "<p>Each holds who the recording is of, its type, quality and flags, then every number the analyses "
            "gave, with its units; hover a column's name in Excel to read its definition and source. Each analysis's "
            "choices (which pulse, window or baseline stands for the recording) sit beside its numbers. Then the "
            "cells' results, every result table on its own sheet (steps, pulses, events...), the Definitions and "
            "the Provenance (file hash, version, the settings changed).</p>"
            "<p>CSV holds the same numbers one value per row, with the tables, the definitions and a manifest.</p>")
        self.stack.setCurrentIndex(1)

    def _write(self, what: str, write) -> Path | None:
        if self.window.project is None:
            return None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            out = write(self.window.project)
        except Exception as exc:                  # the folder unwritable, the disk full: say so
            self.window.say(f"The {what} could not be written: {exc}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._last = out
        self.last.setText(f"Written: {out}")
        self.open_last.setVisible(True)
        self.window.say(f"{what[:1].upper() + what[1:]} written: {out.name}.")
        return out

    def send_to_excel(self) -> None:
        from ... import export

        out = self._write("workbook", export.write_workbook)
        if out is not None:
            open_path(out)

    def write_csv(self) -> None:
        from ... import export

        out = self._write("CSV", lambda p: export.write_csv(p)[-1].parent)
        if out is not None:
            open_path(out)
