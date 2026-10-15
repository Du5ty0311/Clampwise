"""Settings: the theme, the project's optional features, the folders (recent
projects, the demos, the log), the project's saved results, and a check of
this copy of Clampwise."""

from __future__ import annotations

import sys
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ... import __version__
from ...project.features import FEATURES
from .. import theme, widgets
from ..widgets import box, label
from .base import Screen


def _section(title: str) -> QtWidgets.QLabel:
    return label(title, "heading")


class SettingsScreen(Screen):
    key = "settings"
    title = "Settings"
    purpose = "Appearance, optional features, folders, the project's saved results, and an installation check."

    def __init__(self, window) -> None:
        super().__init__(window)
        lay = box("v", "m")
        # the theme
        lay.addWidget(_section("Theme"))
        row = box("h", "s")
        self.theme_buttons = {}
        for key, text in (("light", "Light"), ("dark", "Dark"), ("", "As Windows is set")):
            b = QtWidgets.QRadioButton(text)
            b.toggled.connect(lambda on, k=key: on and self._theme(k))
            self.theme_buttons[key] = b
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        # the optional features: on or off per project, printed in every export
        lay.addWidget(_section("Features"))
        self.features_note = label("", "muted", wrap=True)
        lay.addWidget(self.features_note)
        self.feature_boxes = {}
        for f in FEATURES:
            b = QtWidgets.QCheckBox(f.title)
            b.toggled.connect(lambda on, k=f.key: self._feature(k, on))
            self.feature_boxes[f.key] = b
            lay.addWidget(b)
            lay.addWidget(label(f.what, "hint", wrap=True))
        # folders
        lay.addWidget(_section("Folders"))
        self.recent = QtWidgets.QListWidget()
        self.recent.setMaximumHeight(140)
        self.recent.itemActivated.connect(lambda item: window.open_project(Path(item.text())))
        lay.addWidget(label("Recent projects (double-click to open):", "muted"))
        lay.addWidget(self.recent)
        row = box("h", "s")
        row.addWidget(widgets.button("Open the demos", window.open_demos, "Open and analyze the example recordings"))
        row.addWidget(widgets.button("Show the log folder", self._log_folder,
                                     "Where Clampwise writes its error log and the installation check's report"))
        row.addStretch(1)
        lay.addLayout(row)
        # saved results
        lay.addWidget(_section("Saved results"))
        self.cache = label("", "muted", wrap=True)
        lay.addWidget(self.cache)
        row = box("h", "s")
        self.clear = widgets.button("Clear this project's saved results", self._clear,
                                    "Every recording is analyzed anew next time (decisions and settings are kept)")
        row.addWidget(self.clear)
        row.addStretch(1)
        lay.addLayout(row)
        # about this copy
        lay.addWidget(_section("This copy of Clampwise"))
        lay.addWidget(label(f"Clampwise {__version__}; Python {sys.version.split()[0]}", "muted"))
        lay.addWidget(label("Copyright © 2026 Daniel Mercado. Clampwise is free software under the GNU General "
                            "Public License, version 3 (the LICENSE file beside the program; "
                            "github.com/Du5ty0311/Clampwise), and comes with ABSOLUTELY NO WARRANTY.", "muted",
                            wrap=True))
        row = box("h", "s")
        self.check = widgets.button("Check this installation", self._self_test,
                                    "Analyze the demos with every analysis and write a workbook: about ten seconds")
        row.addWidget(self.check)
        row.addStretch(1)
        lay.addLayout(row)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setVisible(False)
        lay.addWidget(self.report, 1)
        lay.addStretch(1)
        page = QtWidgets.QWidget()
        page.setLayout(lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(page)
        self.body(scroll)
        self._process = None

    def on_shown(self) -> None:
        saved = str(self.window._settings().value("theme", "") or "")
        self.theme_buttons.get(saved, self.theme_buttons[""]).blockSignals(True)
        self.theme_buttons.get(saved, self.theme_buttons[""]).setChecked(True)
        self.theme_buttons.get(saved, self.theme_buttons[""]).blockSignals(False)
        self.recent.clear()
        for f in self.window._settings().value("recent_projects", []) or []:
            self.recent.addItem(str(f))
        self.on_project()

    def on_project(self) -> None:
        p = self.window.project
        states = p.features() if p is not None else {}
        for k, b in self.feature_boxes.items():
            b.blockSignals(True)
            b.setChecked(states.get(k, False))
            b.setEnabled(p is not None)
            b.blockSignals(False)
        self.features_note.setText("Kept with the project and printed in its exports." if p is not None
                                   else "Open a project to choose its features.")
        if p is None:
            self.cache.setText("Open a project to see its saved results.")
            self.clear.setEnabled(False)
            return
        stats = p.cache_stats()
        self.cache.setText(f"{p.name}: {stats['results']} saved result(s), {stats['bytes'] / 1e6:.1f} MB, kept in "
                           f"{p.folder}. A recording is analyzed again only when its file, the settings or an "
                           "analysis's version change.")
        self.clear.setEnabled(stats["results"] > 0)

    def _feature(self, key: str, on: bool) -> None:
        if self.window.project is not None:
            self.window.project.set_feature(key, on)
            for s in self.window.screens.values():
                s.on_project()
            if key in ("notes", "session_grouping", "rig_presets"):
                self.window.organize()

    def _theme(self, key: str) -> None:
        app = QtWidgets.QApplication.instance()
        palette = theme.LIGHT if key == "light" else theme.DARK if key == "dark" else (
            theme.DARK if theme.system_prefers_dark(app) else theme.LIGHT)
        self.window.set_theme(palette)
        if self.window.persist:
            self.window._settings().setValue("theme", key)

    def _log_folder(self) -> None:
        from ..main import log_folder
        from .export import open_path

        open_path(log_folder())

    def _clear(self) -> None:
        p = self.window.project
        if p is None:
            return
        ok = QtWidgets.QMessageBox.question(self, "Clear saved results", f"Clear the saved results of {p.name}? Every "
                                            "recording is analyzed anew next time; who each recording is, the "
                                            "settings and your decisions are kept.")
        if ok == QtWidgets.QMessageBox.Yes:
            n = p.clear_cache()
            self.window.say(f"Cleared {n} saved result(s).")
            self.window.project_changed()

    def _self_test(self) -> None:
        """The same check the build runs, in its own process (it opens a window of its own, unseen)."""
        frozen = getattr(sys, "frozen", False)
        args = ["--self-test"] if frozen else ["-m", "clampwise.app", "--self-test"]
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._process.finished.connect(self._tested)
        self.check.setEnabled(False)
        self.report.setVisible(True)
        self.report.setPlainText("Checking: analyzing the demos...")
        self._process.start(sys.executable, args)

    def _tested(self, code: int, _status=None) -> None:
        from ..main import log_folder

        self.check.setEnabled(True)
        try:
            text = (log_folder() / "self-test.txt").read_text(encoding="utf-8")
        except OSError:
            text = bytes(self._process.readAll()).decode("utf-8", "replace")
        self.report.setPlainText(("Passed.\n\n" if code == 0 else "FAILED: send this to whoever looks after "
                                  "Clampwise.\n\n") + text)
