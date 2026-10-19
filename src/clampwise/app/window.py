"""The main window: the skeleton, the seven destinations, and what lets the
user move without getting lost (B0 items 1-6).

- one path through the work, left to right in the sidebar;
- the same selected recording on every screen, named in the breadcrumb;
- back (Alt+Left) to the previous screen and recording;
- undo (Ctrl+Z) for identity edits and review decisions;
- Ctrl+K to find any action, destination or recording by typing;
- anything longer than about a second runs in the background with progress
  and Cancel; the window never waits on it.
"""

from __future__ import annotations

import time
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .. import APP_NAME
from ..analysis import registry
from ..core.recording import Identity
from ..project import Project  # noqa: F401  (imported at launch: the first folder opened must not wait on it)
from . import skeleton, theme, widgets
from .screens.analyze import AnalyzeScreen
from .screens.check import CheckScreen
from .screens.export import ExportScreen
from .screens.figures import FiguresScreen
from .screens.help import HelpScreen
from .screens.files import FilesScreen
from .screens.live import LiveScreen
from .screens.results import ResultsScreen
from .screens.settings import SettingsScreen

SETTINGS_ORG = "Clampwise"


class AnalyzeJob(QtCore.QThread):
    """A batch through the project store, off the window's thread."""

    progressed = QtCore.Signal(int, int, object)
    finished_all = QtCore.Signal(int, int)          # analyzed, failed

    def __init__(self, folder: Path, paths: list[Path], force: bool) -> None:
        super().__init__()
        self.folder, self.paths, self.force = folder, paths, force
        self._cancel = False
        self.cells_error = ""

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from ..analysis import batch

        ok = bad = 0
        for outcome in batch.run(self.folder, self.paths, force=self.force, cancel=lambda: self._cancel,
                                 progress=lambda d, t, o: self.progressed.emit(d, t, o),
                                 in_process=len(self.paths) <= 2):
            ok += outcome.ok
            bad += not outcome.ok
        if not self._cancel:
            # the analyses of a cell's files together (DSI from events), from the results just made
            try:
                project = Project.open(self.folder, import_legacy=False)
                project.analyze_cells()
                project.close()
            except Exception as exc:          # a cell analysis never fails the batch
                self.cells_error = f"{type(exc).__name__}: {exc}"
        self.finished_all.emit(ok, bad)


class IdentityEdit(QtWidgets.QUndoCommand):
    def __init__(self, window, paths: list[Path], typed: Identity) -> None:
        super().__init__(f"identity of {len(paths)} recording(s)")
        self.window, self.paths, self.typed = window, paths, typed
        self.before = {p: window.project.identity(p) for p in paths}

    def redo(self) -> None:
        for p in self.paths:
            self.window.project.set_identity(p, self.typed)
        self.window.project_changed()

    def undo(self) -> None:
        import json

        for p, ident in self.before.items():
            with self.window.project.db:
                self.window.project.db.execute(
                    "UPDATE identity SET animal=?, slice=?, cell=?, condition=?, label=?, sources=? WHERE recording_id=?",
                    (ident.animal, ident.slice, ident.cell, ident.condition, ident.label, json.dumps(ident.sources),
                     self.window.project._rid(p)))
        self.window.project_changed()


class Finder(QtWidgets.QDialog):
    """Ctrl+K: type a few letters of an action, a destination or a recording."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.setWindowTitle("Find")
        self.window = window
        lay = widgets.box("v", "s", "m")
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("Type to find an action, a screen or a recording")
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        widgets.resizable(self, lay, "finder", (520, 380))
        self.items = window.findable()
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._go)
        self.list.itemActivated.connect(lambda _: self._go())
        self._filter("")

    def _filter(self, text: str) -> None:
        words = text.lower().split()
        self.list.clear()
        for label, kind, action in self.items:
            if all(w in label.lower() for w in words):
                item = QtWidgets.QListWidgetItem(f"{label}    — {kind}")
                item.setData(QtCore.Qt.UserRole, action)
                self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _go(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self.accept()
            item.data(QtCore.Qt.UserRole)()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, palette: theme.Palette | None = None, *, persist: bool = True) -> None:
        super().__init__()
        self.palette = palette or theme.LIGHT
        self.persist = persist
        self.project = None
        self.selected: Path | None = None
        self._recordings: dict[Path, object] = {}
        self._history: list[tuple[str, Path | None]] = []
        self._job: AnalyzeJob | None = None
        registry.load()                   # every analysis declared before anything asks
        self._watcher = None
        self._watch_timer = None
        self._watch_queue: list = []
        self.undo = QtWidgets.QUndoStack(self)
        self.setWindowTitle(APP_NAME)
        self.frame = skeleton.Frame()
        self.frame.breadcrumb.set_palette(self.palette)
        self.setCentralWidget(self.frame)
        self.status = skeleton.StatusBar()
        self.status.cancel_requested.connect(self._cancel_job)
        self.setStatusBar(self.status)
        self.screens = {}
        self._add(FilesScreen(self))
        self._add(CheckScreen(self))
        self._add(AnalyzeScreen(self))
        self._add(ResultsScreen(self))
        self._add(FiguresScreen(self))
        self._add(ExportScreen(self))
        self._add(LiveScreen(self))
        self._add(HelpScreen(self))
        self._add(SettingsScreen(self))
        self.frame.sidebar.chosen.connect(self.go)
        self.frame.breadcrumb.level_chosen.connect(self._crumb)
        self._shortcuts()
        self._restore()
        self.go("files", remember=False)
        self.project_changed()

    # ---- assembly
    def _add(self, screen) -> None:
        self.screens[screen.key] = screen
        self.frame.stack.addWidget(screen)
        self.frame.details.addWidget(screen.details)

    def _shortcuts(self) -> None:
        def sc(keys, slot, text=""):
            a = QtWidgets.QAction(text or keys, self)
            a.setShortcut(QtGui.QKeySequence(keys))
            a.triggered.connect(slot)
            self.addAction(a)
            return a

        for d in skeleton.DESTINATIONS + skeleton.APART:
            sc(d.shortcut, lambda _=False, k=d.key: self.go(k), f"Go to {d.title}")
        sc("Alt+Left", self.back, "Back")
        sc("Ctrl+K", self.find, "Find")
        sc("Ctrl+O", self.open_folder, "Open folder")
        sc("Ctrl+D", lambda: self.frame.details_toggle.toggle(), "Show or hide details")
        sc("Ctrl+Shift+L", self.toggle_theme, "Switch light and dark")
        undo = self.undo.createUndoAction(self, "Undo")
        undo.setShortcut(QtGui.QKeySequence.Undo)
        self.addAction(undo)
        redo = self.undo.createRedoAction(self, "Redo")
        redo.setShortcut(QtGui.QKeySequence.Redo)
        self.addAction(redo)

    # ---- navigation
    def go(self, key: str, *, remember: bool = True) -> None:
        current = self.frame.stack.currentWidget()
        if remember and current is not None and getattr(current, "key", "") != key:
            self._history.append((current.key, self.selected))
            del self._history[:-50]
        screen = self.screens[key]
        self.frame.stack.setCurrentWidget(screen)
        self.frame.details.setCurrentWidget(screen.details)
        self.frame.sidebar.select(key)
        screen.on_shown()

    def back(self) -> None:
        if not self._history:
            self.say("Nothing to go back to.")
            return
        key, path = self._history.pop()
        if path is not None and path != self.selected:
            self.select(path, remember=False)
        self.go(key, remember=False)

    def find(self) -> None:
        Finder(self).exec_()

    def findable(self) -> list[tuple[str, str, object]]:
        out = [(d.title, d.tip, lambda k=d.key: self.go(k)) for d in skeleton.DESTINATIONS + skeleton.APART]
        out += [("Open folder", "open an experiment folder as a project", self.open_folder),
                ("Open the demos", "example recordings: a current-step family, miniature events, a spike train", self.open_demos),
                ("Analyze every recording", "run the analyses in the background", self.analyze_all),
                ("Make a figure", "the figure builder", lambda: self.screens["figures"].open_builder()),
                ("Switch light and dark theme", "the theme", self.toggle_theme),
                ("Back", "the previous screen", self.back)]
        if self.project is not None:
            for row in self.project.summaries():
                out.append((row["name"], f"recording, {row['animal']} {row['slice']} {row['cell']}".strip(),
                            lambda p=row["path"]: (self.select(p), self.go("check"))))
        return out

    def _crumb(self, level: str, value: str) -> None:
        if level == "experiment":
            self.go("files")
        else:
            self.go("files")
            self.screens["files"].filter.setText(value)

    # ---- the project
    def open_folder(self) -> None:
        start = str(self.project.folder) if self.project else str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open an experiment folder", start)
        if folder:
            self.open_project(Path(folder))

    def open_demos(self) -> None:
        from .main import demo_folder

        self.open_project(demo_folder())
        self.analyze_all()

    # ---- watching the folder while Clampex writes into it (B1)
    def set_watching(self, on: bool) -> None:
        from ..io.live import FolderWatcher

        if not on or self.project is None:
            self._watcher = None
            if self._watch_timer is not None:
                self._watch_timer.stop()
            if on:
                self.say("Open a folder first: its project is the one watched.")
            return
        self._watcher = FolderWatcher(self.project.folder)
        if self._watch_timer is None:
            self._watch_timer = QtCore.QTimer(self)
            self._watch_timer.timeout.connect(self._poll_watch)
        self._watch_timer.start(2000)
        self.say(f"Watching {self.project.folder.name}: each new recording is added and analyzed once Clampex "
                 "has finished writing it.")

    @property
    def watching(self) -> bool:
        return self._watcher is not None

    def _poll_watch(self) -> None:
        if self._watcher is None or self.project is None:
            return
        new = self._watcher.poll()
        if new:
            self.project.add_recordings(new)
            self._watch_queue += [p for p in new if p not in self._watch_queue]
            self.say(f"New recording(s): {', '.join(p.name for p in new)}.")
            self.project_changed()
        if self._watch_queue and not (self._job is not None and self._job.isRunning()):
            queued, self._watch_queue = self._watch_queue, []
            self.analyze_paths(queued)

    def open_project(self, folder: Path) -> None:
        from ..io import find_recordings, group_recordings

        if getattr(self, "_opening", False):              # a folder is being looked through already
            return
        self._opening = True
        try:
            # a large folder takes seconds to look through (5,000 recordings: 2 s; a deep tree on a slow disk more):
            # off the window's thread, the window going on and saying so
            paths, notes = self.in_background(f"Opening {folder.name}: looking for its recordings",
                                              lambda: group_recordings(find_recordings(folder)))
        finally:
            self._opening = False
        if self.project is not None:
            self.project.close()
        if self._watcher is not None:
            self.set_watching(False)
        self.project = Project.open(folder)
        added = len(paths) - len(self.project.recordings())
        # hashed when first analyzed, in the background: opening never reads every file
        self.project.add_recordings(paths, hash_files=False)
        self._recordings.clear()
        self.selected = None
        self.undo.clear()
        self._remember_project(folder)
        imported = self.project.meta("imported_from")
        msg = f"Opened {folder.name}: {len(self.project.recordings())} recording(s)"
        if added > 0:
            msg += f", {added} new"
        if imported:
            msg += "; the Synaptoscope project in this folder was imported"
        self.say(msg + ".")
        for n in notes:
            self.say(n)
        self.project_changed()
        self.go("files")
        self.organize()

    def in_background(self, what: str, fn):
        """``fn()`` run off the window's thread, the window answering meanwhile, and after half a second a line
        saying what it is doing and for how long (B0: nothing on screen waits on the data); its result, or the
        exception it raised, raised here."""
        from . import worker

        loop, out, began = QtCore.QEventLoop(), {}, time.monotonic()

        def done(result):
            out["result"] = result
            loop.quit()

        def tick():
            s = time.monotonic() - began
            if s >= 0.5:
                self.say(f"{what} ({s:.0f} s)...")

        timer = QtCore.QTimer()
        timer.timeout.connect(tick)
        timer.start(500)
        worker.run(fn, done)
        if "result" not in out:
            loop.exec_()
        timer.stop()
        result = out["result"]
        if isinstance(result, Exception):
            raise result
        return result

    def organize(self) -> None:
        """Each cell's Notes.txt, episodes and rigs, worked out in the background from the
        files' headers; the notes' labels and details are then applied (features)."""
        from ..io import load
        from ..project import organize
        from . import worker

        p = self.project
        if p is None:
            return
        f = p.features()
        if not (f["notes"] or f["session_grouping"] or f["rig_presets"]):
            p.organization = None
            self.screens["files"].on_project()
            return
        rows = []
        for r in p.recordings():
            ident = p.identity(r.path)
            key = "/".join(x for x in ident.cell_key()) if any(ident.cell_key()) else ""
            rows.append((r.path, key, ident.condition))

        def done(result, project=p, notes_on=f["notes"]):
            if project is not self.project:
                return
            if isinstance(result, Exception):
                self.say(f"Could not read the notes and episodes: {result}")
                return
            changed = organize.apply_notes(project, result) if notes_on else 0
            project.organization = result
            self.project_changed()
            said = []
            if result.notes:
                flags = sum(len(n.flags) for n in result.notes.values())
                said.append(f"read {len(result.notes)} Notes.txt" + (f" ({changed} recording(s) labelled)" if changed
                                                                     else "") + (f", {flags} difference(s) flagged"
                                                                                 if flags else ""))
            if result.episodes:
                said.append(f"{result.count()} episode(s)")
            if said:
                self.say("Organized: " + "; ".join(said) + ".")

        worker.run(lambda: organize.work_out(rows, load, f["notes"], f["session_grouping"], f["rig_presets"]), done)

    def add_files(self) -> None:
        if self.project is None:
            self.say("Open a folder first: files are added to its project.")
            return
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add recordings", str(self.project.folder),
                                                          "Recordings (*.abf *.atf *.asc *.spk *.P0 *.P1 *.FP0 *.FP1 *.BP0 *.BP1)")
        if files:
            self.project.add_recordings(files)
            self.project_changed()

    def project_changed(self) -> None:
        name = self.project.name if self.project else "No project open"
        self.frame.sidebar.project.setText(name)
        self.setWindowTitle(f"{name} – {APP_NAME}" if self.project else APP_NAME)
        for s in self.screens.values():
            s.on_project()
        live_on = self.project is None or self.project.features().get("live_experiment", True)
        self.frame.sidebar.buttons["live"].setVisible(live_on)      # Settings > Features turns it off
        self._crumbs()

    def recording(self, path: Path):
        from ..io import load

        path = Path(path)
        if path not in self._recordings:
            self._recordings[path] = load(path)
            if len(self._recordings) > 20:
                self._recordings.pop(next(iter(self._recordings)))
        return self._recordings[path]

    def select(self, path, *, remember: bool = True) -> None:
        path = Path(path) if path is not None else None
        if path == self.selected:
            return
        self.selected = path
        for s in self.screens.values():
            s.on_selection(path)
        self.screens["files"].select_path(path) if path is not None else None
        self._crumbs()

    def _crumbs(self) -> None:
        if self.project is None:
            self.frame.breadcrumb.set_path([])
            return
        parts = [("experiment", self.project.name)]
        if self.selected is not None:
            ident = self.project.identity(self.selected)
            for level in ("animal", "slice", "cell"):
                v = getattr(ident, level)
                if v:
                    parts.append((level, v))
            parts.append(("recording", self.selected.name))
        self.frame.breadcrumb.set_path(parts)

    def set_identity(self, paths: list[Path], typed: Identity) -> None:
        self.undo.push(IdentityEdit(self, paths, typed))
        self.say(f"Identity of {len(paths)} recording(s) updated (Ctrl+Z undoes).")

    # ---- analysis in the background
    def analyze_all(self, force: bool = False) -> None:
        if self.project is None:
            self.open_folder()
            return
        if self._job is not None and self._job.isRunning():
            self.say("An analysis is already running.")
            return
        self.analyze_paths([r.path for r in self.project.recordings()], force)
        self._nudge()

    def _nudge(self) -> None:
        """At analysis time: the recordings with no age, sex, drugs or internal solution recorded."""
        p = self.project
        missing: dict[str, int] = {}
        for r in p.recordings():
            meta = p.metadata(r.path)
            for k, word in (("age", "age"), ("sex", "sex"), ("drugs", "drugs"), ("internal_solution", "internal")):
                if not getattr(meta, k):
                    missing[word] = missing.get(word, 0) + 1
        if missing:
            self.say("Not recorded: " + ", ".join(f"{w} for {n}" for w, n in missing.items()) + " recording(s); "
                     "Files, Overview lists the cells (type them under Experiment details, or in the cell's "
                     "Notes.txt).")

    def analyze_paths(self, paths: list[Path], force: bool = False) -> None:
        """Analyze these recordings in the background (the cache keeps what did not change)."""
        if self.project is None or not paths:
            return
        if self._job is not None and self._job.isRunning():
            self.say("An analysis is already running.")
            return
        self._job = AnalyzeJob(self.project.folder, paths, force)
        self._job.progressed.connect(self._progress)
        self._job.finished_all.connect(self._done)
        self.status.job(0, len(paths), "Analyzing")
        self._job.start()

    def _progress(self, done: int, total: int, outcome) -> None:
        self.status.job(done, total, "Analyzing")
        if not outcome.ok:
            first = outcome.error.splitlines()[0] if outcome.error else "unknown error"
            self.say(f"Could not analyze {outcome.path.name}: {first}")

    def _done(self, ok: int, bad: int) -> None:
        self.status.job(0, 0)
        self.say(f"Analyzed {ok} recording(s)" + (f"; {bad} could not be read (see the messages above)" if bad else "")
                 + ".")
        if self._job is not None and self._job.cells_error:
            self.say(f"The analyses of a cell's files together could not run: {self._job.cells_error}")
        self.project_changed()
        if self.selected is not None:
            path, self.selected = self.selected, None
            self.select(path, remember=False)

    def _cancel_job(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self.say("Cancelling: no new files start; the ones running finish and are kept.")

    # ---- look, messages, settings
    def toggle_theme(self) -> None:
        self.set_theme(theme.DARK if self.palette.name == "light" else theme.LIGHT)

    def set_theme(self, palette: theme.Palette) -> None:
        self.palette = palette
        app = QtWidgets.QApplication.instance()
        theme.apply(app, palette)
        self.frame.breadcrumb.set_palette(palette)
        for s in self.screens.values():
            s.on_theme(palette)
        if self.persist:
            self._settings().setValue("theme", palette.name)

    def say(self, text: str) -> None:
        """A message in the status bar that stays until the next one; the full
        list goes to the log."""
        self.status.showMessage(text, 15000)

    def _settings(self) -> QtCore.QSettings:
        return QtCore.QSettings(SETTINGS_ORG, APP_NAME)

    def _remember_project(self, folder: Path) -> None:
        if not self.persist:
            return
        s = self._settings()
        recent = [f for f in (s.value("recent_projects", []) or []) if f != str(folder)]
        s.setValue("recent_projects", [str(folder)] + recent[:7])

    def _restore(self) -> None:
        if not self.persist:
            return
        s = self._settings()
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        sizes = s.value("splitter")
        if sizes is not None:
            self.frame.splitter.restoreState(sizes)
        self.frame.details_toggle.setChecked(str(s.value("details_shown", "true")).lower() == "true")

    def closeEvent(self, event) -> None:
        if self._job is not None and self._job.isRunning():
            self._job.cancel()
            self._job.wait(10000)
        self.screens["live"].shutdown()                  # the rig's readout ends with the window
        if self.persist:
            s = self._settings()
            s.setValue("geometry", self.saveGeometry())
            s.setValue("splitter", self.frame.splitter.saveState())
            s.setValue("details_shown", self.frame.details_toggle.isChecked())
        if self.project is not None:
            self.project.close()
        super().closeEvent(event)
