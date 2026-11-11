"""Pressing every control of a screen and seeing what it did.

A control is a button, a checkbox, a tool button's menu action, a right-click
menu's action or a pick list. The harness finds every one a person could
reach under some widgets, presses each from the same starting state, and
records what changed: the thing being edited (the figure, the measurement),
the picture drawn, the other widgets, a dialog asked, a file written, the
clipboard, a line said, a window opened. A control that raises is broken; so
is one that changes nothing. When the edited thing changed, Undo must put it
back.

No dialog ever opens: every static dialog call and every ``exec_`` is
answered here (a file dialog with a prepared file, a question with Yes), and a
drop-down button's menu is walked through its actions (clicking it would open
the menu modally and wait for ever offscreen).
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------- answering dialogs

@dataclass
class Log:
    errors: list = field(default_factory=list)       # tracebacks of what raised inside Qt's calls
    dialogs: list = field(default_factory=list)      # every dialog asked, in words
    menus: list = field(default_factory=list)        # right-click menus shown: their actions' texts
    opened: list = field(default_factory=list)       # links and files handed to the desktop


class Answers:
    """What the dialogs answer. ``files``: {a word of the dialog's filter or caption: the path given}; a save
    dialog gives a path in ``out``."""

    def __init__(self, out: Path, files: dict[str, Path] | None = None) -> None:
        self.out, self.files = Path(out), dict(files or {})
        self.out.mkdir(parents=True, exist_ok=True)
        self.pick_action: int | None = None          # which action of the next right-click menu is triggered
        self.on_dialog: Callable | None = None       # called with a dialog instead of showing it
        self.text: str | None = None                 # what a typed answer is (None: the dialog's own default)

    def open_path(self, caption: str, filt: str) -> str:
        words = f"{caption} {filt}".lower()
        for word, path in self.files.items():
            if word.lower() in words:
                return str(path)
        return ""

    def save_path(self, default: str, filt: str) -> str:
        name = Path(default).name or "out"
        return str(self.out / name)


class Patched:
    """Every dialog answered and every error inside Qt's calls caught, for as long as this is entered."""

    def __init__(self, answers: Answers) -> None:
        self.answers, self.log = answers, Log()
        self._undo: list[tuple[object, str, object]] = []

    def _set(self, owner, name: str, value) -> None:
        self._undo.append((owner, name, owner.__dict__.get(name, getattr(owner, name))))
        setattr(owner, name, value)

    def __enter__(self) -> "Patched":
        a, log = self.answers, self.log
        self._hook = sys.excepthook
        sys.excepthook = lambda t, v, tb: log.errors.append("".join(traceback.format_exception(t, v, tb)))

        def arg(args, kwargs, i, name, default=""):
            return kwargs.get(name, args[i] if len(args) > i else default)

        def open_file(*args, **kwargs):
            caption, filt = arg(args, kwargs, 1, "caption"), arg(args, kwargs, 3, "filter")
            path = a.open_path(str(caption), str(filt))
            log.dialogs.append(f"open a file: {caption}" + ("" if path else " (cancelled: no such file prepared)"))
            return path, ""

        def save_file(*args, **kwargs):
            caption, default, filt = (arg(args, kwargs, 1, "caption"), arg(args, kwargs, 2, "directory"),
                                      arg(args, kwargs, 3, "filter"))
            log.dialogs.append(f"save a file: {caption}")
            return a.save_path(str(default), str(filt)), ""

        def folder(*args, **kwargs):
            log.dialogs.append(f"choose a folder: {arg(args, kwargs, 1, 'caption')}")
            return str(a.out)

        def message(kind, answer):
            def show(*args, **kwargs):
                log.dialogs.append(f"{kind}: {arg(args, kwargs, 1, 'title')}: {arg(args, kwargs, 2, 'text')}")
                return answer
            return show

        def get_text(*args, **kwargs):
            log.dialogs.append(f"type: {arg(args, kwargs, 1, 'title')}: {arg(args, kwargs, 2, 'label')}")
            return (a.text if a.text is not None else (arg(args, kwargs, 4, "text") or "harness")), True

        def get_item(*args, **kwargs):
            items = list(arg(args, kwargs, 3, "items", []))
            log.dialogs.append(f"pick: {arg(args, kwargs, 1, 'title')}: {arg(args, kwargs, 2, 'label')}")
            k = int(arg(args, kwargs, 4, "current", 0))
            return (items[k] if items else ""), bool(items)

        def get_number(cast):
            def show(*args, **kwargs):
                log.dialogs.append(f"number: {arg(args, kwargs, 1, 'title')}: {arg(args, kwargs, 2, 'label')}")
                return cast(arg(args, kwargs, 3, "value", 0)), True
            return show

        def colour(*args, **kwargs):
            log.dialogs.append("choose a colour")
            return QtGui.QColor("#AA3377")

        def dialog_exec(dlg, *args, **kwargs):
            log.dialogs.append(f"dialog: {dlg.windowTitle() or type(dlg).__name__}")
            if a.on_dialog is not None:
                return a.on_dialog(dlg)
            return QtWidgets.QDialog.Rejected

        def menu_exec(menu, *args, **kwargs):
            acts = [x for x in menu.actions() if not x.isSeparator() and x.menu() is None]
            log.menus.append([(x.text(), x.isEnabled(), x.toolTip() if x.toolTip() != x.text() else "",
                               x.isCheckable() and x.isChecked()) for x in acts])
            k = a.pick_action
            if k is not None and 0 <= k < len(acts) and acts[k].isEnabled():
                acts[k].trigger()
            return None

        def open_url(url):
            log.opened.append(url.toString() if hasattr(url, "toString") else str(url))
            return True

        fd, mb, idg = QtWidgets.QFileDialog, QtWidgets.QMessageBox, QtWidgets.QInputDialog
        self._set(fd, "getOpenFileName", staticmethod(open_file))
        self._set(fd, "getOpenFileNames", staticmethod(lambda *x, **k: ([open_file(*x, **k)[0]], "")))
        self._set(fd, "getSaveFileName", staticmethod(save_file))
        self._set(fd, "getExistingDirectory", staticmethod(folder))
        for name, answer in (("information", mb.Ok), ("warning", mb.Ok), ("critical", mb.Ok), ("about", None),
                             ("question", mb.Yes)):
            self._set(mb, name, staticmethod(message(name, answer)))
        self._set(idg, "getText", staticmethod(get_text))
        self._set(idg, "getItem", staticmethod(get_item))
        self._set(idg, "getDouble", staticmethod(get_number(float)))
        self._set(idg, "getInt", staticmethod(get_number(int)))
        self._set(QtWidgets.QColorDialog, "getColor", staticmethod(colour))
        self._set(QtWidgets.QDialog, "exec_", dialog_exec)
        self._set(QtWidgets.QDialog, "exec", dialog_exec)
        self._set(QtWidgets.QMenu, "exec_", menu_exec)
        self._set(QtWidgets.QMenu, "exec", menu_exec)
        self._set(QtWidgets.QMenu, "popup", menu_exec)
        self._set(QtGui.QDesktopServices, "openUrl", staticmethod(open_url))
        return self

    def __exit__(self, *exc) -> None:
        for owner, name, old in reversed(self._undo):
            setattr(owner, name, old)
        sys.excepthook = self._hook


# ---------------------------------------------------------------- finding the controls

STACKS = (QtWidgets.QStackedWidget,)


def label_of(w) -> str:
    """What a person would call this control: its words, else its tooltip's first words, else its kind."""
    text = w.accessibleName() if isinstance(w, QtWidgets.QWidget) else ""
    if not text and isinstance(w, (QtWidgets.QAbstractButton, QtWidgets.QAction)):
        text = w.text()
    if not text:
        tip = w.toolTip() if hasattr(w, "toolTip") else ""
        text = tip.split(":")[0].split(".")[0][:60]
    if not text and hasattr(w, "objectName"):
        text = w.objectName()
    return " ".join(text.replace("&", "").split()) or f"({type(w).__name__})"


def _in_tabs(page) -> bool:
    """Whether ``page`` is a page of a tab widget (reached by clicking its tab), not of a bare stack."""
    stack = page.parentWidget()
    return isinstance(stack, STACKS) and isinstance(stack.parentWidget(), QtWidgets.QTabWidget)


def reachable(w, root) -> bool:
    """Shown, or on a tab a click away; not hidden, and not on a stack's page that only the program shows."""
    p = w
    while p is not None and p is not root:
        if p.isHidden() and not (_in_tabs(p) and p.parentWidget().indexOf(p) >= 0):
            return False
        p = p.parentWidget()
    return p is root


def bring_forward(w, root) -> None:
    """Click the tabs that lead to ``w``."""
    p = w
    while p is not None and p is not root:
        if _in_tabs(p):
            p.parentWidget().parentWidget().setCurrentWidget(p)
        p = p.parentWidget()


@dataclass
class Control:
    key: str                       # stable between presses: where | kind | label | which of those alike
    kind: str                      # button, checkbox, menu action, right-click action, pick list
    label: str
    where: str                     # the root it was found under, in words
    enabled: bool = True
    tooltip: str = ""
    choice: str = ""               # a pick list: the entry chosen
    ticked: bool = False           # a menu's entry already ticked: choosing it again changes nothing, rightly


def _nth(seen: dict, base: str) -> str:
    seen[base] = seen.get(base, 0) + 1
    return f"{base}|{seen[base]}"


def _owner_words(w, root) -> str:
    """The section a control sits in: the nearest table row's name, tab or group, so alike controls tell apart."""
    p = w.parentWidget()
    while p is not None and p is not root:
        if _in_tabs(p):
            tabs = p.parentWidget().parentWidget()
            return tabs.tabText(tabs.indexOf(p))
        p = p.parentWidget()
    return ""


def menu_actions(menu: QtWidgets.QMenu, path: str = "") -> list[tuple[str, QtWidgets.QAction]]:
    """A menu's actions as a person meets them (its ``aboutToShow`` emitted first), submenus walked."""
    menu.aboutToShow.emit()
    out = []
    for a in menu.actions():
        if a.isSeparator():
            continue
        words = " ".join(a.text().replace("&", "").split())
        if a.menu() is not None:
            out += menu_actions(a.menu(), f"{path}{words} > ")
        else:
            out.append((f"{path}{words}", a))
    return out


def find_controls(roots: dict[str, QtWidgets.QWidget], pick_lists: bool = True) -> list[Control]:
    """Every control reachable under ``roots`` ({its words: widget}), in a stable order."""
    out: list[Control] = []
    seen: dict[str, int] = {}
    for where, root in roots.items():
        for w in root.findChildren(QtWidgets.QWidget):
            if not reachable(w, root):
                continue
            tab = _owner_words(w, root)
            place = f"{where} / {tab}" if tab else where
            if isinstance(w, QtWidgets.QAbstractButton):
                if isinstance(w.parentWidget(), (QtWidgets.QAbstractSpinBox, QtWidgets.QTabBar, QtWidgets.QScrollBar)):
                    continue
                name = label_of(w)
                if isinstance(w, QtWidgets.QToolButton) and w.menu() is not None:
                    for text, a in menu_actions(w.menu()):
                        out.append(Control(_nth(seen, f"{place}|menu action|{name} > {text}"), "menu action",
                                           f"{name} > {text}", place, a.isEnabled() and w.isEnabled(),
                                           a.toolTip() if a.toolTip() != a.text() else w.toolTip(),
                                           ticked=a.isCheckable() and a.isChecked()))
                    continue
                kind = "checkbox" if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)) or w.isCheckable() \
                    else "button"
                # one of several ways already chosen (a pressed mode button): pressing it again rightly changes nothing
                chosen = bool(w.property("oneOfSeveral")) and w.isChecked()
                out.append(Control(_nth(seen, f"{place}|{kind}|{name}"), kind, name, place, w.isEnabled(), w.toolTip(),
                                   ticked=bool(chosen)))
            elif pick_lists and isinstance(w, QtWidgets.QComboBox) and not isinstance(w, QtWidgets.QFontComboBox):
                name = label_of(w)
                base = _nth(seen, f"{place}|pick list|{name}")
                model = w.model()
                picks = [i for i in range(w.count()) if i != w.currentIndex()
                         and (not hasattr(model, "item") or model.item(i) is None or model.item(i).isEnabled())]
                for i in picks[:1] + picks[-1:] if len(picks) > 1 else picks:
                    out.append(Control(f"{base}|{i}", "pick list", name, place, w.isEnabled(), w.toolTip(),
                                       w.itemText(i)))
            if w.contextMenuPolicy() == QtCore.Qt.CustomContextMenu:
                out.append(Control(_nth(seen, f"{place}|right-click|{label_of(w)}"), "right-click", label_of(w), place,
                                   w.isEnabled(), ""))
    return out


def find_widget(roots: dict[str, QtWidgets.QWidget], control: Control):
    """The live widget (or action) of ``control``, found again after the screen was put back."""
    seen: dict[str, int] = {}
    for where, root in roots.items():
        for w in root.findChildren(QtWidgets.QWidget):
            if not reachable(w, root):
                continue
            tab = _owner_words(w, root)
            place = f"{where} / {tab}" if tab else where
            if isinstance(w, QtWidgets.QAbstractButton):
                if isinstance(w.parentWidget(), (QtWidgets.QAbstractSpinBox, QtWidgets.QTabBar, QtWidgets.QScrollBar)):
                    continue
                name = label_of(w)
                if isinstance(w, QtWidgets.QToolButton) and w.menu() is not None:
                    for text, a in menu_actions(w.menu()):
                        if _nth(seen, f"{place}|menu action|{name} > {text}") == control.key:
                            return root, w, a
                    continue
                kind = "checkbox" if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)) or w.isCheckable() \
                    else "button"
                if _nth(seen, f"{place}|{kind}|{name}") == control.key:
                    return root, w, None
            elif isinstance(w, QtWidgets.QComboBox) and not isinstance(w, QtWidgets.QFontComboBox):
                if control.key.startswith(_nth(seen, f"{place}|pick list|{label_of(w)}") + "|") \
                        and control.kind == "pick list":
                    return root, w, None
            if w.contextMenuPolicy() == QtCore.Qt.CustomContextMenu:
                if _nth(seen, f"{place}|right-click|{label_of(w)}") == control.key:
                    return root, w, None
    return None, None, None


# ---------------------------------------------------------------- what the screen looks like

def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", "replace")).hexdigest()


def _plot_state(plot) -> list:
    """What a pyqtgraph plot draws: each item's kind and data (not the view's range, which follows the data)."""
    import pyqtgraph as pg

    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    out = []
    for it in item.items:
        row: list = [type(it).__name__]
        try:
            if isinstance(it, (pg.PlotDataItem, pg.ScatterPlotItem)):
                x, y = it.getData()
                if x is not None and len(x):
                    row += [len(x), hashlib.sha1(np.ascontiguousarray(np.nan_to_num(np.asarray(x, float))).tobytes()
                                                 + np.ascontiguousarray(np.nan_to_num(np.asarray(y, float))).tobytes()
                                                 ).hexdigest()[:12]]
                if isinstance(it, pg.PlotDataItem):
                    pen = pg.mkPen(it.opts.get("pen")) if it.opts.get("pen") is not None else None
                    row.append(pen.color().name() if pen is not None else "")
            elif isinstance(it, pg.LinearRegionItem):
                row += [round(float(v), 6) for v in it.getRegion()]
            elif isinstance(it, pg.InfiniteLine):
                row += [round(float(it.value()), 6), it.angle]
            elif isinstance(it, pg.ImageItem) and it.image is not None:
                row.append(hashlib.sha1(np.ascontiguousarray(np.nan_to_num(it.image)).tobytes()).hexdigest()[:12])
            elif isinstance(it, pg.TextItem):
                row.append(it.toPlainText())
            elif isinstance(it, pg.BarGraphItem):
                row.append(str(np.asarray(it.opts.get("height")).tolist())[:200])
        except Exception as exc:                     # a half-made item is still an item
            row.append(f"unreadable: {exc}")
        out.append(row)
    out.append([item.getAxis("left").labelText, item.getAxis("bottom").labelText])
    return out


def widget_state(root, skip=None) -> list:
    """Every widget under ``root`` as a person sees it: shown or not, enabled or not, and what it says."""
    import pyqtgraph as pg

    out = []
    for w in root.findChildren(QtWidgets.QWidget):
        if w is skip or isinstance(w, QtWidgets.QScrollBar):
            continue
        if isinstance(w, pg.PlotWidget):
            out.append(["plot", w.isHidden(), _plot_state(w)])
            continue
        row: list = [type(w).__name__, w.isHidden(), w.isEnabled()]
        if isinstance(w, QtWidgets.QAbstractButton):
            row += [w.text(), w.isChecked()]
        elif isinstance(w, QtWidgets.QLabel):
            row.append(w.text())
        elif isinstance(w, QtWidgets.QComboBox):
            row += [w.currentIndex(), w.count(), w.currentText()]
        elif isinstance(w, QtWidgets.QLineEdit):
            row.append(w.text())
        elif isinstance(w, QtWidgets.QAbstractSpinBox):
            row.append(w.text())
        elif isinstance(w, QtWidgets.QListWidget):
            row += [w.currentRow(), [w.item(i).text() + str(int(w.item(i).checkState())) for i in range(w.count())]]
        elif isinstance(w, QtWidgets.QTableWidget):
            row += [w.rowCount(), w.columnCount(), sorted(i.row() for i in w.selectedIndexes())[:5],
                    [[w.item(r, c).text() if w.item(r, c) else "" for c in range(w.columnCount())]
                     for r in range(min(w.rowCount(), 40))]]
        elif isinstance(w, QtWidgets.QTableView) and w.model() is not None:
            m = w.model()
            row += [m.rowCount(), m.columnCount(),
                    [str(m.headerData(c, QtCore.Qt.Horizontal)) for c in range(m.columnCount())],
                    [[str(m.data(m.index(r, c))) for c in range(m.columnCount())] for r in range(min(m.rowCount(), 40))]]
        elif isinstance(w, QtWidgets.QTabBar):
            row += [w.currentIndex(), [w.tabText(i) for i in range(w.count())]]
        elif isinstance(w, QtWidgets.QTextEdit):
            row.append(w.toPlainText())
        elif isinstance(w, QtWidgets.QPlainTextEdit):
            row.append(w.toPlainText())
        elif isinstance(w, QtWidgets.QStackedWidget):
            row.append(w.currentIndex())
        out.append(row)
    return out


def canvas_hash(fig) -> str:
    """A matplotlib figure as drawn (at a fixed small resolution, so the window's size does not matter)."""
    import io

    buf = io.BytesIO()
    dpi = fig.get_dpi()
    fig.savefig(buf, format="raw", dpi=50)
    fig.set_dpi(dpi)
    return hashlib.sha1(buf.getvalue()).hexdigest()


def windows_state() -> list:
    return sorted((type(w).__name__, w.windowTitle()) for w in QtWidgets.QApplication.topLevelWidgets()
                  if w.isVisible() and not isinstance(w, QtWidgets.QMenu))


def files_state(folder: Path) -> list:
    return sorted((p.name, p.stat().st_size) for p in Path(folder).rglob("*") if p.is_file())


def clipboard_state() -> list:
    cb = QtWidgets.QApplication.clipboard()
    return [cb.text(), cb.pixmap().cacheKey() if not cb.pixmap().isNull() else 0]


# ---------------------------------------------------------------- pressing

def flush(app) -> None:
    """Events handled, and the widgets a screen replaced deleted (Qt deletes them when control returns to the
    event loop; no loop runs here, and a widget waiting to be deleted is not one a person can press)."""
    app.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@dataclass
class Outcome:
    control: Control
    state: str                                # the starting state's name
    effects: list[str] = field(default_factory=list)
    undo: str = ""                            # "restored", "NOT restored", or "" when nothing to undo
    error: str = ""
    detail: str = ""                          # a dialog's words, a file's name: what the effect was


class Session:
    """A screen under test. Subclasses say what is being edited (``model``), how the screen is put back
    (``reset``), how it settles after a press (``settle``) and how the last change is undone (``undo``)."""

    name = ""
    has_undo = True               # False: what changes here is a view or a dialog's picks, not an edit to undo

    def __init__(self, patched: Patched) -> None:
        self.patched = patched
        self.app = QtWidgets.QApplication.instance()

    # -- for subclasses
    def roots(self) -> dict[str, QtWidgets.QWidget]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def model(self) -> str:
        return ""

    def picture(self) -> str:
        return ""

    def said(self) -> str:
        return ""

    def settle(self) -> None:
        flush(self.app)

    def undo(self) -> None:
        pass

    def extra_windows(self) -> list[QtWidgets.QWidget]:
        return []

    # -- the pressing
    def snapshot(self, skip=None) -> dict:
        log, a = self.patched.log, self.patched.answers
        return {"model": self.model(), "picture": self.picture(),
                "widgets": _digest([widget_state(r, skip) for r in self.roots().values()]),
                "dialogs": len(log.dialogs), "menus": len(log.menus), "opened": len(log.opened),
                "files": _digest(files_state(a.out)), "clipboard": _digest(clipboard_state()),
                "said": self.said(), "windows": _digest(windows_state())}

    WORDS = {"model": "what is edited changed", "picture": "the picture changed", "widgets": "the screen changed",
             "dialogs": "asked in a dialog", "menus": "showed a menu", "opened": "opened a link or a file",
             "files": "wrote a file", "clipboard": "put something on the clipboard", "said": "said something",
             "windows": "opened or closed a window"}

    def press(self, control: Control, state: str = "") -> Outcome:
        out = Outcome(control, state)
        log, answers = self.patched.log, self.patched.answers
        n_err = len(log.errors)
        try:
            self.reset()
            self.settle()
            if len(log.errors) > n_err:
                out.error = "putting the screen back raised: " + log.errors[-1]
                return out
            root, w, action = find_widget(self.roots(), control)
            if w is None:
                out.error = "the control was not there after the screen was put back"
                return out
            bring_forward(w, root)
            self.settle()
            before = self.snapshot(skip=w if control.kind in ("checkbox", "pick list") else None)
            n_dlg = len(log.dialogs)
            if control.kind == "menu action":
                action.trigger()
            elif control.kind == "right-click":
                answers.pick_action = None
                w.customContextMenuRequested.emit(QtCore.QPoint(max(1, w.width() // 2), max(1, w.height() // 2)))
            elif control.kind == "pick list":
                w.setCurrentIndex(int(control.key.rsplit("|", 1)[1]))
                if hasattr(w, "activated"):
                    w.activated.emit(w.currentIndex())
            else:
                w.click()
            self.settle()
            after = self.snapshot(skip=w if control.kind in ("checkbox", "pick list") else None)
            out.effects = [self.WORDS[k] for k in before if before[k] != after[k]]
            out.detail = "; ".join(log.dialogs[n_dlg:])[:300]
            if len(log.errors) > n_err:
                out.error = log.errors[n_err]
            elif before["model"] != after["model"] and self.has_undo:
                self.undo()
                self.settle()
                out.undo = "restored" if self.model() == before["model"] else "NOT restored"
                if len(log.errors) > n_err:
                    out.error = "Undo raised: " + log.errors[n_err]
        except Exception:                                # raised straight into the harness
            out.error = traceback.format_exc()
        finally:
            answers.pick_action = None
        return out

    def press_menu(self, control: Control, state: str = "") -> list[Outcome]:
        """A right-click menu: shown once to read its actions, then each action triggered from the starting
        state."""
        log, answers = self.patched.log, self.patched.answers
        first = self.press(control, state)
        if first.error or not log.menus:
            return [first]
        acts = log.menus[-1]
        outs = [first]
        for k, (text, enabled, tip, ticked) in enumerate(acts):
            c = Control(f"{control.key}>{k}", "right-click action", f"{control.label} > {text}", control.where,
                        enabled, tip, ticked=ticked)
            o = Outcome(c, state)
            if not enabled:
                outs.append(o)
                continue
            try:
                self.reset()
                self.settle()
                root, w, _ = find_widget(self.roots(), control)
                bring_forward(w, root)
                self.settle()
                before = self.snapshot()
                n_err, n_dlg = len(log.errors), len(log.dialogs)
                answers.pick_action = k
                w.customContextMenuRequested.emit(QtCore.QPoint(max(1, w.width() // 2), max(1, w.height() // 2)))
                answers.pick_action = None
                self.settle()
                after = self.snapshot()
                o.effects = [self.WORDS[x] for x in before if before[x] != after[x] and x != "menus"]
                o.detail = "; ".join(log.dialogs[n_dlg:])[:300]
                if len(log.errors) > n_err:
                    o.error = log.errors[-1]
                elif before["model"] != after["model"] and self.has_undo:
                    self.undo()
                    self.settle()
                    o.undo = "restored" if self.model() == before["model"] else "NOT restored"
            except Exception:
                o.error = traceback.format_exc()
            finally:
                answers.pick_action = None
            outs.append(o)
        return outs

    def run(self, state: str = "", only: Callable[[Control], bool] | None = None,
            progress: Callable[[Control], None] | None = None) -> list[Outcome]:
        """Every control of the screen pressed from the starting state."""
        self.reset()
        self.settle()
        found = find_controls(self.roots())
        outs: list[Outcome] = []
        for c in found:
            if only is not None and not only(c):
                continue
            if progress is not None:
                progress(c)
            if not c.enabled:
                outs.append(Outcome(c, state))
            elif c.kind == "right-click":
                outs += self.press_menu(c, state)
            else:
                outs.append(self.press(c, state))
        self.reset()
        return outs


def record(outs: list[Outcome], path: str | Path) -> None:
    """The outcomes added to ``path`` (one JSON object a line), for CONTROLS.md to be written from."""
    with open(path, "a", encoding="utf-8") as f:
        for o in outs:
            c = o.control
            f.write(json.dumps({"name": doc_name(c), "state": o.state, "enabled": c.enabled, "tooltip": c.tooltip,
                                "choice": c.choice, "ticked": c.ticked, "effects": o.effects, "undo": o.undo,
                                "error": o.error.strip().splitlines()[-1] if o.error else "", "detail": o.detail})
                    + "\n")


def problems(outs: list[Outcome], allowed: dict[str, str] | None = None) -> list[str]:
    """What is broken in ``outs``, one line each: a control that raised, one that did nothing, an edit Undo did
    not put back, a disabled control that does not say why. ``allowed``: {a label: why it may do nothing}."""
    allowed = allowed or {}
    bad = []
    for o in outs:
        c = o.control
        tag = f"[{o.state}] {c.where}: {c.kind} “{c.label}”" + (f" = {c.choice}" if c.choice else "")
        if not c.enabled:
            if not c.tooltip.strip():
                bad.append(f"{tag} is disabled and does not say why")
            continue
        if o.error:
            bad.append(f"{tag} raised: {o.error.strip().splitlines()[-1]}")
        elif not o.effects and c.label not in allowed and not c.ticked:
            bad.append(f"{tag} did nothing")
        if o.undo == "NOT restored":
            bad.append(f"{tag}: Undo did not put it back")
    return bad


def doc_name(c: Control) -> str:
    """A control as CONTROLS.md lists it: "screen | control | kind", the same whatever the data (a graph's or a
    saved style's own name left out)."""
    import re

    label = c.label.replace("|", "/")
    if label.startswith("Show > "):
        label = "Show > (each graph the results give)"
    elif label.startswith("Colour > "):
        label = "Colour > " + re.sub(r"^(cividis|viridis|magma|inferno|CET-\S+)$", "(each colour map)",
                                     label[len("Colour > "):])
    label = re.sub(r"“[^”]*”", "(a saved style)", label)
    label = re.sub(r" panel [A-F]\b", " this panel", label)
    label = re.sub(r"^Presets > ([^:]+):.*$", r"Presets > \1", label)
    screen = c.where.split(" / ")[0]
    return f"{screen} | {label} | {c.kind}"


def first_menu_whole(c: Control) -> bool:
    """On Figures the four graphs' Show (and Colour) menus list the same entries: every entry of the first
    graph's menu is pressed, the other three menus' entries are the same code."""
    if c.kind == "menu action" and c.label.startswith(("Show > ", "Colour > ")):
        return c.key.rsplit("|", 1)[1] == "1"
    return True


class DialogSession(Session):
    """A dialog of its own (the trace picker, the digitizer): made anew for every press by ``make``, put in a
    state with something for every button by ``ready``."""

    has_undo = False

    def __init__(self, patched: Patched, make: Callable, ready: Callable | None = None) -> None:
        super().__init__(patched)
        self.make, self.ready, self.d = make, ready, None

    def roots(self):
        return {self.title: self.d}

    def reset(self) -> None:
        if self.d is not None:
            self.d.close()
            self.d.deleteLater()
        self.d = self.make()
        self.d.show()
        if self.ready is not None:
            self.ready(self.d)
        self.title = getattr(self.d, "controls_name", None) or self.d.windowTitle()
        QtWidgets.QApplication.clipboard().clear()

    def settle(self) -> None:
        flush(self.app)
        t = getattr(self.d, "timer", None)
        if t is not None and t.isActive():
            t.stop()
            self.d._preview()
        flush(self.app)

    def model(self) -> str:
        from dataclasses import asdict, is_dataclass

        d = self.d
        return _digest([[asdict(s) if is_dataclass(s) else s for s in getattr(d, "sets", [])],
                        getattr(d, "clicks", None), getattr(d, "points", None), getattr(d, "mode", None),
                        d.result(), d.isVisible()])

    def said(self) -> str:
        say = getattr(self.d, "say", None)
        return say.text() if say is not None else ""

    def run(self, state: str = "", only=None, progress=None) -> list[Outcome]:
        outs = super().run(state, only, progress)
        self.d.close()
        return outs


def picker_ready(d) -> None:
    """The trace picker with a trace made of three sweeps (selected in the list) and three other sweeps
    selected in the table."""
    d.controls_name = "Trace picker"
    n = d.table.rowCount()
    d._select_rows(list(range(min(3, n))))
    d.name.setText("Before")
    d._add()
    d._select_rows(list(range(min(3, n)))[-1:] + list(range(3, min(6, n))))
    d._add()
    d.list.selectRow(1)
    d._select_rows(list(range(min(2, n))))


def digitizer_ready(d) -> None:
    """The digitizer calibrated (two known values on each axis) with three points clicked."""
    d.controls_name = "Digitizer"
    for col, row, v in ((50, 250, 0.0), (350, 250, 1.0), (50, 250, 0.0), (50, 30, 1.0)):
        d.clicked(col, row, v)
    d._mode("points")
    for c, r in ((100, 200), (200, 150), (300, 100)):
        d.clicked(c, r)


# ---------------------------------------------------------------- a project to press on

def _dummy_abf(path: Path) -> Path:
    """A small real file for a simulated recording to be kept under (the project wants a file to hash)."""
    import pyabf.abfWriter

    rng = np.random.default_rng(sum(map(ord, path.name)))
    pyabf.abfWriter.writeABF1(rng.normal(0, 1, (1, 200)).astype(np.float32), str(path), 10_000, units="mV")
    return path


def simulated() -> dict[str, Callable]:
    """One simulated recording of every recording type the simulator makes, by the type it is analyzed as."""
    from clampwise import simulate as sim

    return {
        "cc_evoked": lambda: sim.photostim_steps(seed=3, success=0.8, jitter_s=0.0006).recording,
        "cc_steps": lambda: sim.firing_steps(pattern="adapting").recording,
        "vc_evoked": lambda: sim.evoked().recording,
        "vc_spontaneous": lambda: sim.spontaneous(duration_s=20.0).recording,
        "dsi": lambda: sim.dsi_evoked(durations=(1.0, 2.0), trials=2).recording,
        "cc_short_square": lambda: sim.short_squares().recording,
        "cc_ramp": lambda: sim.ramp().recording,
        "vc_steps": lambda: sim.vc_family().recording,
        "membrane_test": lambda: sim.membrane_test().recording,
    }


def open_window(folder: Path, recordings: dict[str, object], analyze: bool = True):
    """A window on a project in ``folder`` holding ``recordings`` ({file name: a Recording to keep under a small
    file of that name, or a Path already in the folder}), each analyzed. Returns (window, {name: path})."""
    from clampwise.app import theme
    from clampwise.app.window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, theme.LIGHT)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths, made = {}, {}
    for name, rec in recordings.items():
        if isinstance(rec, Path):
            paths[name] = rec
        else:
            paths[name] = _dummy_abf(folder / name)
            made[name] = rec
    w = MainWindow(theme.LIGHT, persist=False)
    w.resize(1500, 950)
    w.show()
    w.open_project(folder)
    app.processEvents()
    for name, rec in made.items():
        rec.source = paths[name]
        w._recordings[paths[name]] = rec
    if analyze:
        for name, path in paths.items():
            w.project.analyze(path, w.recording(path))
    return w, paths


# ---------------------------------------------------------------- the three screens

class BuilderSession(Session):
    name = "Figure builder"

    def __init__(self, patched: Patched, builder) -> None:
        super().__init__(patched)
        self.b = builder
        self.baseline = builder.spec.to_json()

    def roots(self):
        return {"Figure builder": self.b.centralWidget()}

    def reset(self) -> None:
        b = self.b
        b.typing.stop()
        b.timer.stop()
        b.current = 0
        b._history, b._future = [], []
        b._restore(self.baseline)
        b._snapshot()
        b.sections.setCurrentIndex(0)
        b.zoom.setCurrentIndex(0)
        b.statusBar().clearMessage()
        # one saved style, unlike the figure's look, for Styles > Use and Delete to act on
        from dataclasses import asdict

        from clampwise.figure.recipe import Style

        b.window_._settings().setValue("figure_styles", json.dumps({"big text": asdict(Style(font_size=12.0))}))
        QtWidgets.QApplication.clipboard().clear()

    def settle(self) -> None:
        flush(self.app)
        b = self.b
        if b.typing.isActive():
            b.typing.stop()
            b._edited()
        if b.timer.isActive():
            b.timer.stop()
            b.redraw()
        flush(self.app)

    def model(self) -> str:
        return self.b.spec.to_json()

    def picture(self) -> str:
        return canvas_hash(self.b.fig)

    def said(self) -> str:
        return self.b.statusBar().currentMessage()

    def undo(self) -> None:
        self.b.undo()


class FiguresSession(Session):
    name = "Figures"
    has_undo = False

    def __init__(self, patched: Patched, window, path, across: bool = False) -> None:
        super().__init__(patched)
        self.w, self.path, self.across = window, path, across
        self.s = window.screens["figures"]
        self.lines: list[str] = []
        say = window.say
        window.say = lambda text: (self.lines.append(text), say(text))[1]

    def roots(self):
        return {"Figures": self.s}

    def reset(self) -> None:
        s, w = self.s, self.w
        for win in list(s.windows):
            win.close()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        if s.builder is not None:
            s.builder.close()
            s.builder = None
        if s.maximized is not None:
            s.maximize(s.maximized)
        w.go("figures")
        if w.selected != self.path:
            w.select(self.path)
        s.across_button.setChecked(self.across)
        s.across.correlations.setChecked(False)
        s.across.colour.setCurrentIndex(0)
        s._type = ""
        for p in s.panels:
            p.heat = {"cmap": "", "mode": "percentile", "fixed": None, "db": False}
        s._toggle()
        QtWidgets.QApplication.clipboard().clear()

    def model(self) -> str:
        s = self.s
        return _digest([[p._key for p in s.panels], s.stack.currentIndex(), s.across_button.isChecked(),
                        s.maximized is not None, [dict(p.heat) for p in s.panels],
                        [c.currentText() for c in (s.across.kind, s.across.x, s.across.y, s.across.colour)],
                        s.across.correlations.isChecked()])

    def said(self) -> str:
        return str(len(self.lines)) + (self.lines[-1] if self.lines else "")

    def snapshot(self, skip=None) -> dict:
        out = super().snapshot(skip)
        b = self.s.builder
        out["windows"] = _digest([windows_state(), b is not None and b.isVisible(),
                                  b.spec.to_json() if b is not None else ""])
        return out


def busy_workbench(screen) -> None:
    """The Workbench with something for every button to act on: the measure region selected, a transform, a
    fit, a sweep left out, an onset set, a kept measurement and a template."""
    from clampwise.analysis import workbench as wb

    s, rec = screen, screen._recording
    m = s._m
    measure = next((r for r in m.regions if r.kind != "baseline"), m.regions[-1])
    m.transforms = [wb.Transform("lowpass", {"hz": 2000.0})]
    m.fit_model, m.fit_region = "exp1", measure.name
    if rec.n_sweeps > 1:
        m.exclusions = [wb.Exclusion(rec.n_sweeps - 1, "noisy")]
    m.overrides = [wb.Override(0, "onset", measure.start_s + 0.004, measure.name)]
    s._apply_state(m.to_dict())
    s.region_list.setCurrentRow(m.regions.index(measure))
    s.keep()
    s.window.project.save_template(wb.Template.from_measurement("light response", "*", s._m))
    s._fill_templates()
    s.window.undo.clear()


class WorkbenchSession(Session):
    name = "Workbench"

    def __init__(self, patched: Patched, window, path, prepare: Callable | None = None) -> None:
        super().__init__(patched)
        self.w, self.path = window, path
        self.s = window.screens["workbench"]
        self.lines: list[str] = []
        say = window.say
        window.say = lambda text: (self.lines.append(text), say(text))[1]
        window.go("workbench")
        window.select(path)
        self.app.processEvents()
        if prepare is not None:
            prepare(self.s)
        self.baseline = self.s._m.to_dict()
        self.region_row = self.s.region_list.currentRow()
        self.kept = [(h["measurement"], h["summary"]) for h in window.project.hands(path)]
        self.templates = list(window.project.templates())

    def roots(self):
        return {"Workbench": self.s, "Workbench details": self.s.details}

    def reset(self) -> None:
        from clampwise.analysis import workbench as wb

        s, w, p = self.s, self.w, self.w.project
        w.go("workbench")
        for h in p.hands(self.path):
            p.remove_hand(h["id"])
        for m, rows in self.kept:
            p.add_hand(self.path, wb.HandMeasurement.from_dict(m), rows)
        for t in p.templates():
            p.delete_template(t.name)
        for t in self.templates:
            p.save_template(t)
        s._set_click_mode("")
        for section in s.details.findChildren(QtWidgets.QToolButton):      # the folded Options open, to be pressed
            if section.isCheckable() and section.arrowType() != QtCore.Qt.NoArrow:
                section.setChecked(True)
        s._apply_state(self.baseline)
        s.region_list.setCurrentRow(self.region_row)
        w.undo.clear()
        s._fill_kept()
        s._fill_templates()
        if s.kept.count():
            s.kept.setCurrentRow(0)
        if s.transform_list.count():
            s.transform_list.setCurrentRow(0)
        s.tabs.setCurrentIndex(0)
        QtWidgets.QApplication.clipboard().clear()

    def model(self) -> str:
        p = self.w.project
        return _digest([self.s._m.to_dict() | {"made_at": "", "app_version": ""},
                        [h["name"] for h in p.hands(self.path)], [t.name for t in p.templates()]])

    def said(self) -> str:
        return str(len(self.lines)) + (self.lines[-1] if self.lines else "") + self.s._click_mode

    def undo(self) -> None:
        self.w.undo.undo()
