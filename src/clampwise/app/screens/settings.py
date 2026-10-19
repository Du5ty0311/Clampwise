"""Settings: the theme, the project's optional features, the folders (recent
projects, the demos, the log), the project's saved results, a check of
this copy of Clampwise, and its updates (``clampwise.update``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ... import __version__, update
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
        # a cheaper PC
        lay.addWidget(_section("Performance"))
        from .. import power

        self.low_power = QtWidgets.QCheckBox("Low-power mode (a PC with few cores or little memory)")
        self.low_power.setToolTip("The camera's picture drawn at most 15 times a second (every frame still taken), "
                                  "every 2nd pixel while shown smaller, its histogram once a second, the rig's "
                                  "programs looked at once a second. On by default on a PC with fewer than 4 cores "
                                  "or less than 8 GB")
        self.low_power.setChecked(power.enabled(window._settings() if getattr(window, "persist", False) else None))
        self.low_power.toggled.connect(self._keep_low_power)
        lay.addWidget(self.low_power)
        # updates, from the releases on GitHub
        lay.addWidget(_section("Updates"))
        self.auto_update = QtWidgets.QCheckBox("Check for a new version when Clampwise starts")
        self.auto_update.setToolTip("Asks GitHub for the newest release; nothing is downloaded or installed unless "
                                    "you choose Update")
        self.auto_update.setChecked(self.updates_wanted())
        self.auto_update.toggled.connect(self._keep_auto_update)
        lay.addWidget(self.auto_update)
        row = box("h", "s")
        self.check_update = widgets.button("Check for updates", lambda: self.check_updates(quiet=False),
                                           "Ask GitHub for the newest release of Clampwise")
        row.addWidget(self.check_update)
        self.do_update = widgets.primary("Update", self.install_update, "Download the new version (checked against "
                                         "the checksum GitHub lists), close Clampwise, replace it and open it again; "
                                         "projects and settings are kept")
        self.do_update.setVisible(False)
        row.addWidget(self.do_update)
        self.update_bar = QtWidgets.QProgressBar()
        self.update_bar.setMaximumHeight(8)
        self.update_bar.setTextVisible(False)
        self.update_bar.setVisible(False)
        row.addWidget(self.update_bar, 1)
        row.addStretch(1)
        lay.addLayout(row)
        self.update_note = label("", "muted", wrap=True)
        lay.addWidget(self.update_note)
        self._release: update.Release | None = None
        self._got = [0, 0]                                 # bytes downloaded, of
        self._bar_timer = QtCore.QTimer(self)
        self._bar_timer.timeout.connect(self._show_progress)
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

    # ---- updates
    def updates_wanted(self) -> bool:
        if not getattr(self.window, "persist", False):
            return False
        return str(self.window._settings().value("updates/check", "true")).lower() == "true"

    def _keep_low_power(self, on: bool) -> None:
        """Kept for this PC, and applied at once to the camera and the rig's readout."""
        from .. import power

        if getattr(self.window, "persist", False):
            self.window._settings().setValue(power.KEY, "true" if on else "false")
        live = self.window.screens.get("live")
        if live is None:
            return
        live.camera._power(on=on)
        if getattr(live, "_thread", None) is not None:
            live._thread.every_s = power.limits(on)["rig_s"]

    def _keep_auto_update(self, on: bool) -> None:
        if getattr(self.window, "persist", False):
            self.window._settings().setValue("updates/check", "true" if on else "false")

    def startup_check(self) -> None:
        """When Clampwise starts: a quiet check, only for the built program (or a test server), if wanted."""
        if self.updates_wanted() and (update.install_folder() is not None or os.environ.get("CLAMPWISE_UPDATE_URL")):
            self.check_updates(quiet=True)

    def check_updates(self, quiet: bool = False) -> None:
        from .. import worker

        self.check_update.setEnabled(False)
        if not quiet:
            self.update_note.setText("Asking GitHub for the newest release...")
        worker.run(update.latest, lambda result, q=quiet: self._checked(result, q))

    def _checked(self, result, quiet: bool) -> None:
        self.check_update.setEnabled(True)
        if isinstance(result, Exception):
            if not quiet:
                self.update_note.setText(f"GitHub could not be reached ({result}); try again later.")
            return
        if result is None or not update.is_newer(result.version):
            self.do_update.setVisible(False)
            if not quiet:
                self.update_note.setText(f"Clampwise {__version__} is the newest version.")
            return
        self._release = result
        self.do_update.setText(f"Update to {result.version}")
        self.do_update.setVisible(True)
        self.update_note.setText(f"Clampwise {result.version} is available (this copy is {__version__}): "
                                 f"{result.page}")
        self.window.say(f"Clampwise {result.version} is available: Settings, Update.")

    def install_update(self) -> None:
        from .. import worker

        r = self._release
        if r is None:
            return
        rec = getattr(self.window.screens.get("live"), "recorder", None)
        if rec is not None and (rec.busy or rec.test.running):
            self.update_note.setText("A recording or membrane test is running: stop it first, then update.")
            return
        install = update.install_folder()
        if install is None:
            self.update_note.setText(f"This copy runs from source: update it with git pull ({r.page}).")
            return
        notes = r.notes.strip()
        notes = notes[:1200] + ("..." if len(notes) > 1200 else "")
        answer = QtWidgets.QMessageBox.question(
            self, f"Update to Clampwise {r.version}",
            f"Clampwise {r.version} ({r.size / 1e6:.0f} MB) will be downloaded and checked; then Clampwise closes, "
            f"is replaced and opens again. Your projects and settings are kept.\n\n{notes}")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.do_update.setEnabled(False)
        self.check_update.setEnabled(False)
        self.update_bar.setVisible(True)
        self._got = [0, r.size]
        self._bar_timer.start(200)
        self.update_note.setText(f"Downloading Clampwise {r.version}...")
        folder = update.log_path().parent / "updates"

        def work():
            z = update.download(r, folder, progress=lambda done, total: self._got.__setitem__(slice(0, 2),
                                                                                               [done, total]))
            return update.stage(z, install)

        worker.run(work, lambda result: self._staged(result, install))

    def _show_progress(self) -> None:
        done, total = self._got
        self.update_bar.setRange(0, max(1, total // 1024))
        self.update_bar.setValue(done // 1024)

    def _staged(self, result, install: Path) -> None:
        from .export import open_path

        self._bar_timer.stop()
        self.update_bar.setVisible(False)
        self.do_update.setEnabled(True)
        self.check_update.setEnabled(True)
        r = self._release
        if isinstance(result, PermissionError):
            self.update_note.setText(f"Clampwise's folder ({install}) cannot be written from this account: "
                                     "download the new version from its page and unzip it in its place.")
            open_path(r.page)
            return
        if isinstance(result, Exception):
            self.update_note.setText(f"Not updated: {result}.")
            return
        update.start_swap(result, install)
        self.update_note.setText(f"Updating to {r.version}: Clampwise closes and opens again.")
        QtCore.QTimer.singleShot(300, self.window.close)

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
