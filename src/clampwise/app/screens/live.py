"""Live: the cell being recorded and the rig's programs, on one screen (analysis while
patching).

What it shows, each with where it comes from: the amplifier, read directly (its mode, holding, its own Rs and Cm,
its gain and filter, its meters); what the acquisition program last did (where it writes, the last file and how it
ended, the last Membrane Test); the patterned light's session (its pattern sequence and trigger); the camera's last
picture; and the rig programs' own windows, live, each brought to the front with a click to be worked in
directly. Clampwise changes nothing in any of them. The reading runs off the window's thread, twice a second.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import digidata, growing, hardware, multiclamp
from ...rig import state as rig_state
from ...rig import windows as rig_windows
from .. import theme, widgets
from ..widgets import box, label
from .base import Screen

RS_CHANGE = 0.20                       # B19: a badge when Rs moves more than 20% from where watching began
HISTORY_S = 3 * 3600
ORDER = ("acquisition", "camera", "amplifier", "light")
# (row, column, rows, columns): the acquisition's trace a wide rectangle across the top, as
# big as possible, the camera the largest below it
PLACES = {"acquisition": (0, 0, 1, 3), "camera": (1, 0, 1, 1), "amplifier": (1, 1, 1, 1), "light": (1, 2, 1, 1)}
COLUMNS = (2, 1, 1)                    # the camera's column twice the others'
WHOLE = "whole"                        # a view's part: the whole window, chosen
FILL = {"acquisition"}                 # stretched to fill by default


class RigThread(QtCore.QThread):
    """Looks at the rig every ``every_s`` and sends each snapshot to the window's thread."""

    snapshot = QtCore.Signal(object)

    done = QtCore.Signal(str)                # an amplifier control's outcome: "" done, else why not

    def __init__(self, make_watch=rig_state.Watch, every_s: float = 0.5) -> None:
        super().__init__()
        self.make_watch, self.every_s, self._stop = make_watch, every_s, False
        self.folder = ""                     # the open project's folder, read beside the acquisition's (set any time)
        self.pending: list = []              # controls asked for before the watch began
        self._now = False

    def send(self, command: tuple) -> None:
        """An amplifier control (rig.amplifier_control), run in this thread at once (before the next look)."""
        watch = getattr(self, "watch", None)
        if watch is None or not hasattr(watch, "commands"):
            self.pending.append(command)
            return
        watch.commands.append((command, self.done.emit))
        self._now = True

    def run(self) -> None:
        watch = self.make_watch()
        self.watch = watch
        for command in self.pending:
            if hasattr(watch, "commands"):
                watch.commands.append((command, self.done.emit))
        self.pending = []
        try:
            while not self._stop:
                try:
                    watch.folder = self.folder
                    self.snapshot.emit(watch.look())
                except Exception as exc:                  # one bad look never ends the watch
                    self.snapshot.emit(exc)
                    self.msleep(2000)
                waited = 0
                self._now = False
                while waited < self.every_s * 1000 and not self._now and not self._stop:     # a control: at once
                    self.msleep(20)
                    waited += 20
        finally:
            watch.close()

    def stop(self) -> None:
        self._stop = True
        self.wait(5000)


class WindowView(QtWidgets.QFrame):
    """A rig program's window shown live in this frame by the compositor; a click brings the program forward.
    It can be enlarged over the other views or hidden, and show only part of the window: the
    one the user drags out, else the program's usual part (rig.windows.PARTS: the camera's picture)."""

    thumbnail_cls = rig_windows.Thumbnail
    enlarge = QtCore.Signal(str)
    hide_me = QtCore.Signal(str)
    pop = QtCore.Signal(str)                          # out into a window of its own, or back
    fill_changed = QtCore.Signal(str, bool)
    part_chosen = QtCore.Signal(str, object)          # role, (left, top, right, bottom) px in from the edges, or WHOLE

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role
        self.part: tuple | str | None = None          # the user's: a part, WHOLE, or None (the program's usual)
        self.panel: QtWidgets.QWidget | None = None   # Clampwise's own picture in place of the program's window
        self.own = False
        self.popped = False                           # a window of its own
        self.fill = role in FILL                      # the picture stretched to the frame's shape, or kept in proportion
        self._band: QtWidgets.QRubberBand | None = None
        self._from: QtCore.QPoint | None = None
        self.setObjectName("desk")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setMinimumSize(160, 110)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        lay = box("v", "xs", "s")
        head = box("h", "s")
        self.name = label(rig_windows.ROLES[role][0], "heading")
        self.name.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # clipped, not wider
        self.name.setMinimumWidth(60)
        self.badge = widgets.Badge("muted", "not open")
        head.addWidget(self.name, 1)
        head.addWidget(self.badge)
        # small, as on a window's own title bar, so a narrow view keeps its name
        self.enlarge_button = self._tool(QtWidgets.QStyle.SP_TitleBarMaxButton, lambda: self.enlarge.emit(self.role))
        self.hide_button = self._tool(QtWidgets.QStyle.SP_TitleBarCloseButton, lambda: self.hide_me.emit(self.role))
        self.hide_button.setToolTip("Hide this view (Views, above, brings it back)")
        self.pop_button = self._tool(QtWidgets.QStyle.SP_TitleBarNormalButton, lambda: self.pop.emit(self.role))
        self.pop_button.setToolTip("Pop out: a window of its own, to resize, maximize or lay over another "
                                   "(closing it puts it back)")
        self.set_enlarged(False)
        self.head = head
        head.addWidget(self.pop_button)
        head.addWidget(self.enlarge_button)
        head.addWidget(self.hide_button)
        lay.addLayout(head)
        self.detail = label("", "muted", wrap=True)
        self.detail.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        lay.addWidget(self.detail)
        lay.addStretch(1)
        self.state = label("Not open", "muted", wrap=True)
        self.state.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.state)
        lay.addStretch(1)
        self.setLayout(lay)
        self.source: rig_windows.AppWindow | None = None
        self._thumb = None
        self._timer = QtCore.QTimer(self)                  # the frame can move with the layout: follow it
        self._timer.timeout.connect(self._place)

    def set_status(self, status: str, word: str, detail: str = "", tip: str = "") -> None:
        self.badge.set(status, word, tip or detail)
        self.detail.setText(detail)
        self.detail.setToolTip(detail)

    def show_window(self, window: rig_windows.AppWindow | None) -> None:
        same = window is not None and self.source is not None and window.hwnd == self.source.hwnd
        self.source = window
        if window is None:
            self._drop()
            self.state.setText("Not open")
            self.setToolTip("")
        elif window.minimized:
            self._drop()
            self.state.setText("Minimized: click to bring it back and show it here")
        else:
            self.state.setText("")
            if not same:
                self._drop()
            self._attach()
        if window is not None:
            self.setToolTip("Click to bring it to the front and work in it (Clampwise only shows it)")

    def hold(self, panel: QtWidgets.QWidget, take, release,
             tip: str = "Use the camera in Clampwise (its own program must be closed)",
             held: str = "the camera itself, held by Clampwise (Let go gives it back)") -> None:
        """Let Clampwise's own ``panel`` replace the program's window here: ``take`` and ``release`` it."""
        self.panel, self.held = panel, held
        self.layout().insertWidget(2, panel, 10)
        panel.hide()
        self.use_button = widgets.button("Use here", lambda: release() if self.own else take(), tip, quiet=True)
        self.head.insertWidget(2, self.use_button)

    def set_own(self, on: bool) -> None:
        """Clampwise's own picture (on) or the program's window (off)."""
        self.own = on
        if self.panel is not None:
            self.panel.setVisible(on)
            self.use_button.setText("Let go" if on else "Use here")
        self.state.setVisible(not on)
        if on:
            self._drop()
            self.set_status("ok", "in Clampwise", self.held)
        else:
            self.show_window(self.source)

    def _attach(self) -> None:
        if self.own or self._thumb is not None or self.source is None or not self.isVisible():
            return
        try:
            self._thumb = self.thumbnail_cls(int(self.window().winId()), self.source.hwnd)
        except (OSError, ValueError) as exc:
            self._thumb = None
            self.state.setText(f"Cannot be shown here ({exc})")
            return
        self._place()
        self._timer.start(500)

    def _drop(self) -> None:
        self._timer.stop()
        if self._thumb is not None:
            self._thumb.close()
            self._thumb = None

    def _place(self) -> None:
        if self._thumb is None:
            return
        top = self.window()
        origin = self.mapTo(top, QtCore.QPoint(0, 0))
        ratio = self.devicePixelRatioF()
        pad = theme.SPACE["s"]
        head = self.detail.geometry().bottom() + pad
        self._thumb.place(int((origin.x() + pad) * ratio), int((origin.y() + head) * ratio),
                          int((self.width() - 2 * pad) * ratio), int((self.height() - head - pad) * ratio),
                          visible=self.isVisible(), part=self.shown_part(), fill=self.fill)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.source is not None and not self.source.minimized:
            self._attach()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._drop()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place()

    def _tool(self, icon, slot) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setIcon(self.style().standardIcon(icon))
        b.setAutoRaise(True)
        b.setCursor(QtCore.Qt.ArrowCursor)
        b.clicked.connect(slot)
        return b

    def set_enlarged(self, on: bool) -> None:
        self.enlarged = on
        self.enlarge_button.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_TitleBarMinButton if on else QtWidgets.QStyle.SP_TitleBarMaxButton))
        self.enlarge_button.setToolTip("Show all the views again" if on else
                                       "Enlarge: this window alone fills the views")

    def shown_part(self) -> tuple | None:
        if self.part == WHOLE:
            return None
        if self.part is not None:
            return self.part
        return rig_windows.usual_part(self.source) if self.source is not None else None

    def set_fill(self, on: bool) -> None:
        self.fill = on
        self._place()
        self.fill_changed.emit(self.role, on)

    def fit_to_source(self) -> None:
        """A popped-out view in the shape of what it shows, as large as its screen allows (stretchable after)."""
        if not self.popped or self._thumb is None:
            return
        pw, ph = self._thumb.part_size(self.shown_part())
        if pw <= 0 or ph <= 0:
            return
        screen = (self.screen() or QtWidgets.QApplication.primaryScreen()).availableGeometry()
        frame = self.frameGeometry().size() - self.geometry().size()           # the window's own border and title
        pad = theme.SPACE["s"]
        head = self.detail.geometry().bottom() + pad
        ratio = self.devicePixelRatioF()
        room_w = screen.width() - frame.width() - 2 * pad
        room_h = screen.height() - frame.height() - head - pad
        scale = min(room_w / (pw / ratio), room_h / (ph / ratio))
        w, h = int(pw / ratio * scale) + 2 * pad, int(ph / ratio * scale) + head + pad
        self.showNormal()
        self.resize(w, h)
        self.move(screen.x() + (screen.width() - w - frame.width()) // 2, screen.y())

    def set_free(self, on: bool) -> None:
        """In a window inside Clampwise, whose own title bar maximizes and closes it."""
        self.enlarge_button.setVisible(not on and not self.popped)
        self.hide_button.setVisible(not on and not self.popped)

    def set_popped(self, on: bool) -> None:
        self.popped = on
        self.enlarge_button.setVisible(not on)
        self.hide_button.setVisible(not on)
        self.pop_button.setToolTip("Put back into Live" if on else "Pop out: a window of its own, to resize, "
                                   "maximize or lay over another (closing it puts it back)")

    def set_on_top(self, on: bool) -> None:
        self._drop()
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, on)
        self.show()                                    # a changed flag hides the window; the thumbnail follows it

    def closeEvent(self, event) -> None:
        if self.popped:                                # closing the window puts the view back
            event.ignore()
            self.pop.emit(self.role)
            return
        super().closeEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QtWidgets.QMenu(self)
        if self.popped:
            menu.addAction("Put back into Live", lambda: self.pop.emit(self.role))
            top = menu.addAction("Keep on top")
            top.setCheckable(True)
            top.setChecked(bool(self.windowFlags() & QtCore.Qt.WindowStaysOnTopHint))
            top.toggled.connect(self.set_on_top)
            see = menu.addMenu("See-through (to lay it over another)")
            for pct in (100, 85, 70, 55, 40):
                a = see.addAction(f"{pct}%" if pct < 100 else "Opaque")
                a.setCheckable(True)
                a.setChecked(abs(self.windowOpacity() * 100 - pct) < 1)
                a.triggered.connect(lambda _=False, v=pct: self.setWindowOpacity(v / 100))
            menu.addAction("Normal size" if self.isFullScreen() else "Full screen (Esc leaves)",
                           lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())
            menu.addAction("Fit to the program's shape (as large as the screen)", self.fit_to_source)
            menu.addSeparator()
        else:
            menu.addAction("Pop out into a window of its own", lambda: self.pop.emit(self.role))
            menu.addAction("Show all the views again" if self.enlarged else "Enlarge this view",
                           lambda: self.enlarge.emit(self.role))
        menu.addAction("Show only part of the window (drag over it)...", self.choose_part).setEnabled(
            self._thumb is not None)
        if self.source is not None and self.source.program in rig_windows.PARTS:
            menu.addAction("Show only its usual part (the trace, the picture)",
                           lambda: self.part_chosen.emit(self.role, None))
        menu.addAction("Show the whole window", lambda: self.part_chosen.emit(self.role, WHOLE))
        fill = menu.addAction("Stretch to fill the view")
        fill.setCheckable(True)
        fill.setChecked(self.fill)
        fill.toggled.connect(self.set_fill)
        if not self.popped:
            menu.addSeparator()
            menu.addAction("Hide this view", lambda: self.hide_me.emit(self.role))
        menu.exec_(event.globalPos())

    def choose_part(self) -> None:
        """The next drag over the view chooses the part shown (Esc cancels)."""
        if self._thumb is None:
            return
        self._from = None
        self.setCursor(QtCore.Qt.CrossCursor)
        self.state.setText("")
        self.setFocus()
        self.grabMouse()

    def _choosing(self) -> bool:
        return self.cursor().shape() == QtCore.Qt.CrossCursor

    def _end_choice(self) -> None:
        self.releaseMouse()
        self.setCursor(QtCore.Qt.PointingHandCursor)
        if self._band is not None:
            self._band.hide()
        self._from = None

    def keyPressEvent(self, event) -> None:
        if self._choosing() and event.key() == QtCore.Qt.Key_Escape:
            self._end_choice()
            return
        if self.popped and event.key() == QtCore.Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if self.own:                                    # Clampwise's own picture: nothing to bring forward
            super().mousePressEvent(event)
            return
        if self._choosing():
            self._from = event.globalPos()
            if self._band is None:                       # a window of its own: the compositor draws over our frame
                self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle)
            self._band.setGeometry(QtCore.QRect(self._from, QtCore.QSize()))
            self._band.show()
            return
        if self.source is not None:
            rig_windows.bring_to_front(self.source.hwnd)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._choosing() and self._from is not None and self._band is not None:
            self._band.setGeometry(QtCore.QRect(self._from, event.globalPos()).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if not (self._choosing() and self._from is not None):
            super().mouseReleaseEvent(event)
            return
        rect = QtCore.QRect(self._from, event.globalPos()).normalized()
        self._end_choice()
        part = self.part_from(rect)
        if part is not None:
            self.part_chosen.emit(self.role, part)

    def part_from(self, rect: QtCore.QRect) -> tuple | None:
        """A rectangle dragged on the screen as the source window's part: px in from its edges (None: too small)."""
        if self._thumb is None or rect.width() < 8 or rect.height() < 8:
            return None
        top = self.window()
        ratio = self.devicePixelRatioF()
        a, b = top.mapFromGlobal(rect.topLeft()), top.mapFromGlobal(rect.bottomRight())
        p1 = self._thumb.to_source(int(a.x() * ratio), int(a.y() * ratio))
        p2 = self._thumb.to_source(int(b.x() * ratio), int(b.y() * ratio))
        sw, sh = self._thumb.source_size()
        if p1 is None or p2 is None or not sw or p2[0] - p1[0] < 8 or p2[1] - p1[1] < 8:
            return None
        return (p1[0], p1[1], sw - p2[0], sh - p2[1])


class Readout:
    """A number read from the rig: its name and its value (placed by the panel), where it comes from in the
    tooltip."""

    def __init__(self, caption: str) -> None:
        self.caption = label(caption, "muted")
        self.value = label("–", wrap=True)

    def set(self, text: str, tip: str = "") -> None:
        self.value.setText(text or "–")
        self.value.setToolTip(tip)
        self.caption.setToolTip(tip)


def channel_words(device) -> str:
    """Which amplifier channel, without the product's name (the interface names the rig's roles)."""
    if device is None:
        return "its first channel"
    return f"channel {device.channel}" + (f", serial {device.serial}" if device.serial else "")


def telegraph_words(tg) -> str:
    """A telegraph in a line: the mode, the primary output's scale and gain, its filter, the command's sensitivity."""
    scale = tg.primary_v_per_unit / (tg.primary_gain or 1.0)
    per = f"{scale * 1e3:g} V/nA" if tg.primary_unit == "pA" else f"{scale * 1e3:g} mV/mV"
    lpf = f", {tg.primary_lpf_hz / 1e3:g} kHz" if tg.primary_lpf_hz and tg.primary_lpf_hz < 1e5 else ", bypass"
    cmd = f", command {tg.command_per_volt:g} {tg.command_unit}/V" if tg.command_per_volt else ", no command"
    return f"{tg.mode}, {per} x{tg.primary_gain:g}{lpf}{cmd}"


def amplifier_words(status: str) -> str:
    """The amplifier readout's status in the interface's words (the rig's roles; Help names the programs)."""
    from ...rig import multiclamp as mc

    return {mc.ERRORS[mc.NOT_OPEN]: "the amplifier's program is not open",
            mc.ERRORS[mc.TIMED_OUT]: "the amplifier's program did not answer in time",
            "Commander is open but lists no amplifier": "its program is open but lists no amplifier"}.get(
        status, status.replace("MultiClamp Commander", "the amplifier's program").replace("Commander", "its program"))


class LiveScreen(Screen):
    key = "live"
    title = "Live"
    compact = True
    purpose = "The cell you are recording and the rig's programs, live in one place: the amplifier read directly, " \
              "what the acquisition last did, the patterned light's session, the camera's last picture."

    make_watch = rig_state.Watch

    def __init__(self, window) -> None:
        super().__init__(window)
        self.watch_button = widgets.primary("Start watching", self.toggle_watching,
                                            "Add and analyze each new recording of the open project once the "
                                            "acquisition has finished writing it")
        self.add_action(self.watch_button)
        self._thread: RigThread | None = None
        self._history: list[tuple[float, float, float, float]] = []      # time, Rs, Cm, holding
        self._sweeps: list = []                                            # the cell's sweeps (rig.growing.SweepTest)
        self._cell = ""                                                    # the folder they came from
        self._first_rs: float | None = None
        self.last: rig_state.Snapshot | None = None

        # the rig's programs are the main focus, the acquisition's trace and the camera first: the programs' views fill the
        # screen (the acquisition and the camera on top, largest); the cell sits in a narrow panel beside them, laid
        # out like the amplifier's own panel
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        programs = QtWidgets.QWidget()
        side = box("v", "xs")
        bar = box("h", "s")
        bar.addWidget(label("The rig's programs", "section"))
        bar.addStretch(1)
        self.views_button = widgets.button("Views", tooltip="Show or hide each program's view", quiet=True)
        self.views_menu = QtWidgets.QMenu(self.views_button)
        self.views_menu.aboutToShow.connect(self._fill_views_menu)
        self.views_button.setMenu(self.views_menu)
        bar.addWidget(self.views_button)
        side.addLayout(bar)
        views = QtWidgets.QGridLayout()
        views.setSpacing(theme.SPACE["s"])
        self.views = {role: WindowView(role) for role in ORDER}
        for role in ORDER:
            views.addWidget(self.views[role], *PLACES[role])
            self.views[role].enlarge.connect(self.toggle_enlarged)
            self.views[role].hide_me.connect(lambda r: self.set_view_shown(r, False))
            self.views[role].part_chosen.connect(self.set_part)
            self.views[role].pop.connect(self.toggle_popped)
            self.views[role].fill_changed.connect(lambda *_: self._save_layout())
        self.grid = views
        tidy = QtWidgets.QWidget()
        tidy.setLayout(views)
        self.mdi = QtWidgets.QMdiArea()                    # the views as windows inside Clampwise
        self.mdi.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.mdi.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.mdi.subWindowActivated.connect(self._raised)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(tidy)
        self.stack.addWidget(self.mdi)
        self.free = False                                  # the views as windows inside Clampwise, or laid out
        self._subs: dict[str, QtWidgets.QMdiSubWindow] = {}
        from .camera_panel import CameraPanel

        self.camera = CameraPanel(window, folder=self._snapshot_folder)
        self.views["camera"].hold(self.camera, self.use_camera, self.camera.release)
        self.camera.released.connect(lambda: self.views["camera"].set_own(False))
        from .acquisition_panel import AcquisitionPanel

        self.recorder = AcquisitionPanel(window)           # the acquisition in Clampwise (rig.recorder)
        self._recording = ""                               # the folder being recorded into
        self.views["acquisition"].hold(
            self.recorder, self.use_acquisition, self.recorder.release,
            "Record in Clampwise: your protocols played on the digitizer, into ABF files (the acquisition "
            "program must be closed)",
            "recording in Clampwise, into ABF files the acquisition program reads (Let go gives the view back)")
        self.recorder.released.connect(lambda: self.views["acquisition"].set_own(False))
        self.recorder.recording.connect(self._recording_into)
        side.addWidget(self.stack, 1)
        programs.setLayout(side)
        programs.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        programs.customContextMenuRequested.connect(lambda pos: self.views_menu.exec_(programs.mapToGlobal(pos)))
        split.addWidget(programs)

        # the cell, compact: the channel as the amplifier's panel shows it, then the recorded trace, Rs, the records
        panel = QtWidgets.QWidget()
        lay = box("v", "s")
        # the rig's hardware, found by itself (rig.hardware): what is connected and who is using it
        found = QtWidgets.QFrame()
        found.setObjectName("desk")
        found.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        fl = box("v", "xs", "s")
        ftop = box("h", "s")
        ftop.addWidget(label("Hardware", "heading"), 1)
        ftop.addWidget(widgets.button("Detect", self.detect_hardware, "Look for the digitizer, the amplifier and the "
                                      "camera now (they are looked for every few seconds anyway)", quiet=True))
        fl.addLayout(ftop)
        self.hardware_grid = QtWidgets.QGridLayout()
        self.hardware_grid.setHorizontalSpacing(theme.SPACE["m"])
        self.hardware_grid.setVerticalSpacing(theme.SPACE["xs"])
        self.hardware_grid.setColumnStretch(1, 1)
        self.hardware_rows: dict[str, widgets.Badge] = {}
        for row, role in enumerate(hardware.ROLES):
            badge = widgets.Badge("muted", "looking...")
            badge.setVisible(role != "light")              # the patterned light's IOBox only when it is there
            caption = label(hardware.WORDS[role], "muted")
            caption.setVisible(role != "light")
            self.hardware_grid.addWidget(caption, row, 0)
            self.hardware_grid.addWidget(badge, row, 1)
            self.hardware_rows[role] = badge
        fl.addLayout(self.hardware_grid)
        found.setLayout(fl)
        lay.addWidget(found)
        self.hardware: list[hardware.Device] = []
        self._detecting = False
        self._hardware_timer = QtCore.QTimer(self)
        self._hardware_timer.timeout.connect(self.detect_hardware)
        channel = QtWidgets.QFrame()
        channel.setObjectName("desk")
        channel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        ch = box("v", "s", "s")
        top = box("h", "s")
        self.channel_name = label("Amplifier", "heading")
        top.addWidget(self.channel_name, 1)
        self.readouts = {k: Readout(c) for k, c in (
            ("mode", "Mode"), ("holding", "Holding"), ("rs", "Rs"), ("cm", "Cm"), ("rs_comp", "Rs compensation"),
            ("bridge", "Bridge balance"), ("output", "Primary output"), ("filter", "Low-pass"),
            ("seal", "Seal test"), ("telegraph", "Telegraph"))}
        top.addWidget(self.readouts["mode"].value)
        ch.addLayout(top)
        meters = QtWidgets.QGridLayout()                  # the channel's two meters, as the amplifier's panel has them
        meters.setHorizontalSpacing(theme.SPACE["s"])
        self.meter_v, self.meter_i = label("–", "meter"), label("–", "meter")
        for k, (cap, w) in enumerate((("V (mV)", self.meter_v), ("I (pA)", self.meter_i))):
            w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            meters.addWidget(label(cap, "muted"), 0, k)
            meters.addWidget(w, 1, k)
        ch.addLayout(meters)
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(theme.SPACE["m"])
        form.setVerticalSpacing(theme.SPACE["xs"])
        for row, key in enumerate(k for k in self.readouts if k != "mode"):
            form.addWidget(self.readouts[key].caption, row, 0)
            form.addWidget(self.readouts[key].value, row, 1)
        form.setColumnStretch(1, 1)
        ch.addLayout(form)
        self.meters = label("", "muted", wrap=True)
        ch.addWidget(self.meters)
        channel.setLayout(ch)
        lay.addWidget(channel)
        # the amplifier's controls, sent through its own program (rig.amplifier_control; Settings > Features)
        from .amplifier_panel import AmplifierPanel

        self.controls_frame = QtWidgets.QFrame()
        self.controls_frame.setObjectName("desk")
        self.controls_frame.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        cl = box("v", "xs", "s")
        ctop = box("h", "s")
        ctop.addWidget(label("Controls", "heading"), 1)
        self.controls_toggle = widgets.button("Show", self.toggle_controls, "The amplifier's settings and buttons, "
                                              "sent through its own program (which shows each change too)",
                                              quiet=True)
        ctop.addWidget(self.controls_toggle)
        cl.addLayout(ctop)
        self.amp_controls = AmplifierPanel(self._send_control)
        self.amp_controls.setVisible(False)
        cl.addWidget(self.amp_controls)
        self.controls_frame.setLayout(cl)
        lay.addWidget(self.controls_frame)
        # the cell's health: each reading from its best source, against the quality limits (rig.health)
        well = QtWidgets.QFrame()
        well.setObjectName("desk")
        well.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        hl = box("v", "xs", "s")
        hl.addWidget(label("Cell health", "heading"))
        self.health_grid = QtWidgets.QGridLayout()
        self.health_grid.setHorizontalSpacing(theme.SPACE["m"])
        self.health_grid.setVerticalSpacing(theme.SPACE["xs"])
        self.health_grid.setColumnStretch(1, 1)
        hl.addLayout(self.health_grid)
        well.setLayout(hl)
        lay.addWidget(well)
        self.health_rows: dict[str, tuple[QtWidgets.QLabel, widgets.Badge]] = {}
        self.rs_badge = widgets.Badge("muted", "Rs steady")
        self.rs_badge.setVisible(False)
        lay.addWidget(self.rs_badge, 0, QtCore.Qt.AlignLeft)
        self.trace = label("", wrap=True)                 # each sweep's test pulse, as the acquisition writes it
        lay.addWidget(self.trace)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(110)
        self.plot.setMaximumHeight(200)
        self.plot.getPlotItem().setLabel("bottom", "Minutes")
        self.plot.getPlotItem().setLabel("left", "Rs (MOhm)")
        theme.style_plot(self.plot, window.palette)
        self.plot.addLegend(offset=(-10, 10))
        self.rs_curve = self.plot.plot([], [], pen=pg.mkPen(theme.series(window.palette)[0], width=2),
                                       name="compensation")
        self.trace_curve = self.plot.plot([], [], pen=None, symbol="o", symbolSize=5,
                                          symbolBrush=theme.series(window.palette)[1], symbolPen=None,
                                          name="test pulse")
        self.plot.setVisible(False)
        lay.addWidget(self.plot)
        lay.addWidget(label("Recorded", "section"))
        self.acquisition = label("", "muted", wrap=True)
        self.test = label("", "muted", wrap=True)
        self.light = label("", "muted", wrap=True)
        self.picture = label("", "muted", wrap=True)
        for w in (self.acquisition, self.test, self.light, self.picture):
            lay.addWidget(w)
        lay.addStretch(1)
        notes = box("h", "s")                              # a note, timestamped, kept with the project
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText("A note (drug on, drug off...)")
        self.note.returnPressed.connect(self.add_note)
        notes.addWidget(self.note, 1)
        notes.addWidget(widgets.button("Add", self.add_note, "Keep this note with the time and the last file"))
        lay.addLayout(notes)
        panel.setLayout(lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(320)
        split.addWidget(scroll)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        self.split = split
        split.splitterMoved.connect(lambda *_: self._save_layout())
        self.body(split)
        self._hidden: set[str] = set()
        self._enlarged: str | None = None
        self._split_set = False
        self._details_folded = False
        self._load_layout()
        self._apply_views()

        d = box("v", "s", "m")
        d.addWidget(label("What this screen reads", "heading"))
        d.addWidget(label("Everything here is read, never set: the amplifier through its own program's interface, "
                          "the other programs through the files they keep and the windows they show. A click on a "
                          "window brings that program forward; the work is done in it. Help names the programs.",
                          "muted", wrap=True))
        d.addWidget(label("One exception, on request: Use here in the acquisition view records in Clampwise, while the "
                          "acquisition program is closed. Your protocol is played on the digitizer as that program "
                          "plays it (the amplifier's mode and scales from its own program) and written into an ABF "
                          "file it reads; Live measures each sweep as it comes.", "muted", wrap=True))
        d.addStretch(1)
        self.details.setLayout(d)

    # ---- the camera in Clampwise (rig.lucam): taken on request while its own program is closed
    def use_camera(self) -> None:
        view = self.views["camera"]
        if view.source is not None:
            self.window.say("The camera's program is open and holds the camera: close it, then Use here.")
            return
        view.set_own(True)
        self.camera.take()

    def toggle_controls(self) -> None:
        shown = not self.amp_controls.isVisible()
        self.amp_controls.setVisible(shown)
        self.controls_toggle.setText("Hide" if shown else "Show")

    def _send_control(self, command: tuple) -> None:
        self._start()
        self._thread.send(command)

    def use_acquisition(self) -> None:
        view = self.views["acquisition"]
        if view.source is not None:
            self.window.say("The acquisition program is open and holds the digitizer: close it, then Use here.")
            return
        view.set_own(True)
        self.recorder.take()

    def _recording_into(self, folder: str) -> None:
        """Live reads the file being recorded, as it reads the acquisition program's."""
        self._recording = folder
        if folder:
            self._start()
        if self._thread is not None:
            self._thread.folder = self._watched()

    def _snapshot_folder(self) -> str:
        """Where a snapshot goes: the folder the acquisition writes to (the cell's), else the open project's."""
        folder = self.last.data_folder if self.last is not None else ""
        if folder and Path(folder).is_dir():
            return folder
        return str(self.window.project.folder) if self.window.project is not None else ""

    # ---- the views: shown or hidden, one enlarged, each one's part (kept for the user, not the project)
    def _fill_views_menu(self) -> None:
        self.views_menu.clear()
        for role in ORDER:
            a = self.views_menu.addAction(rig_windows.ROLES[role][0])
            a.setCheckable(True)
            a.setChecked(role not in self._hidden)
            a.toggled.connect(lambda on, r=role: self.set_view_shown(r, on))
        self.views_menu.addSeparator()
        free = self.views_menu.addAction("In-app windows (move, resize, overlap them inside Clampwise)")
        free.setCheckable(True)
        free.setChecked(self.free)
        free.toggled.connect(self.set_free)
        if self.free:
            self.views_menu.addAction("Tile them", self.mdi.tileSubWindows)
        if self._enlarged:
            self.views_menu.addSeparator()
            self.views_menu.addAction("Show all the views again", lambda: self.toggle_enlarged(self._enlarged))

    def set_view_shown(self, role: str, shown: bool) -> None:
        (self._hidden.discard if shown else self._hidden.add)(role)
        if not shown and self._enlarged == role:
            self._enlarged = None
        if shown:
            self._enlarged = None
        self._apply_views()
        self._save_layout()
        if not shown:
            self.window.say(f"{rig_windows.ROLES[role][0]} hidden: Views (or a right-click beside the views) brings "
                            "it back.")

    def toggle_enlarged(self, role: str) -> None:
        self._enlarged = None if self._enlarged == role else role
        self._apply_views()

    def set_part(self, role: str, part) -> None:
        view = self.views[role]
        view.part = part
        view._place()
        self._save_layout()

    # ---- in-app windows
    def set_free(self, on: bool) -> None:
        """The views as windows inside Clampwise (moved, resized, overlapped, maximized), or laid out tidily."""
        if on == self.free:
            return
        placed = [r for r in ORDER if not self.views[r].popped]
        for role in placed:
            self._undock(role)
        self.free = on
        self.stack.setCurrentIndex(1 if on else 0)
        saved = self.window._settings().value("live/sub/acquisition") if self.window.persist else None
        for role in placed:
            self._dock(role)
        if on and saved is None:
            self.mdi.tileSubWindows()
        self._enlarged = None
        self._apply_views()
        self._save_layout()

    def _undock(self, role: str) -> None:
        view = self.views[role]
        view._drop()
        if role in self._subs:
            sub = self._subs.pop(role)
            self._save_sub(role, sub)
            view.setParent(None)
            self.mdi.removeSubWindow(sub)
            sub.deleteLater()
        else:
            self.grid.removeWidget(view)
            view.setParent(None)

    def _dock(self, role: str) -> None:
        view = self.views[role]
        view.set_free(self.free)
        if not self.free:
            self.grid.addWidget(view, *PLACES[role])
            return
        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(view)
        sub.setWindowTitle(rig_windows.ROLES[role][0])
        sub.setWindowIcon(QtWidgets.QApplication.windowIcon())       # Clampwise's, not Qt's
        sub.installEventFilter(self)
        self.mdi.addSubWindow(sub)
        saved = self.window._settings().value(f"live/sub/{role}") if self.window.persist else None
        if saved:
            try:
                sub.setGeometry(*(int(x) for x in saved))
            except (TypeError, ValueError):
                pass
        self._subs[role] = sub
        view.show()
        sub.show()

    def _save_sub(self, role: str, sub) -> None:
        if self.window.persist:
            g = sub.geometry()
            self.window._settings().setValue(f"live/sub/{role}", [g.x(), g.y(), g.width(), g.height()])

    def eventFilter(self, obj, event) -> bool:
        role = next((r for r, s in self._subs.items() if s is obj), None)
        if role is not None:
            kind = event.type()
            if kind in (QtCore.QEvent.Move, QtCore.QEvent.Resize, QtCore.QEvent.WindowStateChange):
                QtCore.QTimer.singleShot(0, self.views[role]._place)        # the live picture follows its window
            elif kind == QtCore.QEvent.Close:                                  # closing one hides it (Views again)
                event.ignore()
                self.set_view_shown(role, False)
                return True
        return super().eventFilter(obj, event)

    def _raised(self, sub) -> None:
        """The window brought forward draws its live picture over the others' (the last one registered is on top)."""
        if sub is None:
            return
        view = sub.widget()
        if isinstance(view, WindowView) and view._thumb is not None:
            view._drop()
            view._attach()

    def toggle_popped(self, role: str) -> None:
        """A view out into a window of its own (resized, maximized, laid over another), or back into Live."""
        view = self.views[role]
        s = self.window._settings() if self.window.persist else None
        view._drop()
        if not view.popped:
            size, at = view.size() * 1.6, view.mapToGlobal(QtCore.QPoint(40, 40))
            self._undock(role)
            view.setWindowFlags(QtCore.Qt.Window)
            view.setWindowTitle(f"{rig_windows.ROLES[role][0]} (Clampwise)")
            view.set_popped(True)
            geometry = s.value(f"live/float/{role}") if s is not None else None
            fitted = geometry is not None and view.restoreGeometry(geometry)
            if not fitted:
                view.resize(size)
                view.move(at)
            view.show()
            if not fitted:                               # the first time: the program's shape, as large as possible
                QtCore.QTimer.singleShot(300, view.fit_to_source)
            if self._enlarged == role:
                self._enlarged = None
        else:
            if s is not None:
                s.setValue(f"live/float/{role}", view.saveGeometry())
            view.setWindowOpacity(1.0)
            view.setWindowFlags(QtCore.Qt.Widget)
            view.set_popped(False)
            self._dock(role)
        self._apply_views()

    def _apply_views(self) -> None:
        shown = {r for r in self.views if r not in self._hidden and self._enlarged in (None, r)
                 and not self.views[r].popped}
        for k in range(2):                              # an empty row or column takes no room
            self.grid.setRowStretch(k, 1 if any(PLACES[r][0] == k for r in shown) else 0)
        for k, stretch in enumerate(COLUMNS):
            self.grid.setColumnStretch(k, stretch if any(PLACES[r][1] <= k < PLACES[r][1] + PLACES[r][3]
                                                         for r in shown) else 0)
        for role, view in self.views.items():
            if role in self._subs:
                sub = self._subs[role]
                sub.setVisible(role not in self._hidden)
                view.setVisible(role not in self._hidden)
            elif not view.popped:
                view.setVisible(role in shown)
            view.set_enlarged(self._enlarged == role)

    def _load_layout(self) -> None:
        if not self.window.persist:
            return
        s = self.window._settings()
        self._hidden = {r for r in (s.value("live/hidden", []) or []) if r in self.views}
        for role, view in self.views.items():
            fill = s.value(f"live/fill/{role}")
            if fill is not None:
                view.fill = str(fill).lower() == "true"
            saved = str(s.value(f"live/part/{role}", "") or "")
            if saved == WHOLE:
                view.part = WHOLE
            elif saved:
                try:
                    view.part = tuple(int(x) for x in saved.split(","))
                except ValueError:
                    view.part = None
        state = s.value("live/split")
        if state is not None:
            self._split_set = self.split.restoreState(state)
        if str(s.value("live/free", "false")).lower() == "true":
            QtCore.QTimer.singleShot(0, lambda: self.set_free(True))

    def _save_layout(self) -> None:
        if not self.window.persist:
            return
        s = self.window._settings()
        s.setValue("live/hidden", sorted(self._hidden))
        for role, view in self.views.items():
            s.setValue(f"live/fill/{role}", view.fill)
            s.setValue(f"live/part/{role}", WHOLE if view.part == WHOLE else
                       ",".join(str(x) for x in view.part) if view.part else "")
        s.setValue("live/split", self.split.saveState())
        s.setValue("live/free", self.free)
        for role, sub_ in self._subs.items():
            self._save_sub(role, sub_)

    # ---- the rig's hardware, found by itself
    def detect_hardware(self) -> None:
        """Look for the devices in the background (Windows' device list and Commander's telegraph: nothing opened)."""
        from .. import worker

        if self._detecting or not self.isVisible():
            return
        self._detecting = True
        held = {"acquisition": digidata.held() or self.recorder.busy or self.recorder.test.running,
                "camera": self.camera.running}
        worker.run(lambda: hardware.detect(held=held), self._hardware_found)

    def _hardware_found(self, result) -> None:
        self._detecting = False
        if isinstance(result, Exception):
            return
        self.hardware = result
        have = {d.role: d for d in result}
        for role, badge in self.hardware_rows.items():
            d = have.get(role)
            if role == "light":
                badge.setVisible(d is not None)
                self.hardware_grid.itemAtPosition(hardware.ROLES.index(role), 0).widget().setVisible(d is not None)
            if d is None:
                badge.set("muted", "not connected", "Not among the USB devices Windows lists now")
                continue
            busy = d.state.startswith("in use by") and "Clampwise" not in d.state or "closed" in d.state
            badge.set("warn" if busy else "ok", d.line(), f"Windows' device: {d.instance}")
            badge.setWordWrap(True)

    # ---- the readout runs while the screen shows or the folder is watched
    def on_shown(self) -> None:
        if not self._hardware_timer.isActive():
            self._hardware_timer.start(5000)
        QtCore.QTimer.singleShot(0, self.detect_hardware)
        if not self._split_set:                          # first time: the programs take most of the width
            self._split_set = True
            total = max(self.split.width(), 2)
            panel = max(340, int(total * 0.24))
            self.split.setSizes([total - panel, panel])
        toggle = getattr(getattr(self.window, "frame", None), "details_toggle", None)
        if toggle is not None and toggle.isChecked():   # the programs get the room; Details still opens it
            self._details_folded = True
            toggle.setChecked(False)
        self._start()

    def _unfold_details(self) -> None:
        if self._details_folded:
            self._details_folded = False
            self.window.frame.details_toggle.setChecked(True)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        if not event.spontaneous():                      # left for another screen (not the window minimized)
            self._unfold_details()
        if not self.window.watching:
            self._stop()

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._thread = RigThread(self.make_watch)
        self._thread.folder = self._watched()
        self._thread.snapshot.connect(self.show_snapshot)
        self._thread.done.connect(self.amp_controls.said)
        self._thread.start()

    def _stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread = None

    def shutdown(self) -> None:
        self._unfold_details()                           # the user's own choice is what is kept
        self.camera.release()                            # the camera is given back when Clampwise closes
        if self.recorder.busy:                           # a recording ends as a whole file, never cut off
            self.recorder.rec.stop()
            self.recorder.rec.wait(30)
        if self.recorder.test.running:                   # the membrane test lets the digitizer go
            self.recorder.test.stop()
            self.recorder.test.wait(15)
        digidata.let_go()
        for role, view in self.views.items():           # the views' own windows close with Clampwise
            if view.popped:
                self.toggle_popped(role)
        self._stop()
        for v in self.views.values():
            v.show_window(None)

    def toggle_watching(self) -> None:
        self.window.set_watching(not self.window.watching)
        self.on_project()
        if self.window.watching:
            self._start()

    def on_theme(self, palette) -> None:
        theme.style_plot(self.plot, palette)
        self.rs_curve.setPen(pg.mkPen(theme.series(palette)[0], width=2))
        self.trace_curve.setSymbolBrush(theme.series(palette)[1])
        self.recorder.on_theme(palette)

    def on_project(self) -> None:
        self.watch_button.setText("Stop watching" if self.window.watching else "Start watching")
        project = self.window.project
        record_on = project is None or project.features().get("record", True)    # Settings > Features
        self.views["acquisition"].use_button.setVisible(record_on or self.views["acquisition"].own)
        self.controls_frame.setVisible(project is None or project.features().get("amplifier_controls", True))
        if self._thread is not None:
            self._thread.folder = self._watched()

    def _watched(self) -> str:
        if self._recording:
            return self._recording
        project = self.window.project
        return str(project.folder) if project is not None and self.window.watching else ""

    # ---- showing a snapshot
    def show_snapshot(self, snap) -> None:
        if isinstance(snap, Exception):
            self.window.say(f"The rig could not be read this time ({type(snap).__name__}: {snap}); trying again.")
            return
        self.last = snap
        self._cards(snap)
        self._amplifier(snap)
        self.amp_controls.show_state(snap.amplifier)
        self._trace(snap)
        self._health(snap)
        self._records(snap)
        for role, view in self.views.items():
            found = [w for w in snap.windows.get(role, []) if not w.minimized] or snap.windows.get(role, [])
            view.show_window(found[0] if found else None)

    def _cards(self, snap) -> None:
        amp = snap.amplifier
        if amp is not None and amp.connected:
            where = channel_words(amp.device)
            self.views["amplifier"].set_status("ok", "reading", f"{amp.mode_name}; {where}",
                                               "read directly from the amplifier's program, twice a second")
        else:
            self.views["amplifier"].set_status("muted", "not read", amplifier_words(amp.status if amp else ""))
        f = snap.last_file
        if self.views["acquisition"].own:                # while Clampwise records, its view says so
            st = self.recorder.rec.status if self.recorder.rec is not None else None
            self.views["acquisition"].set_status(
                "ok", "in Clampwise",
                f"{'recording' if st.busy else 'last recorded'} {st.path.name}" if st and st.path else
                "records here while the acquisition program is closed", self.views["acquisition"].held)
        else:
            self.views["acquisition"].set_status("ok" if snap.clampex_open else "muted",
                                                 "open" if snap.clampex_open else "not open",
                                                 f"last file {f.name} ({f.outcome})" if f else "")
        light = snap.light
        self.views["light"].set_status("ok" if snap.light_open else "muted", "open" if snap.light_open else "not open",
                                       f"{light.sequence or 'no pattern loaded'}"
                                       + (f", {light.regions} regions" if light and light.regions else "")
                                       if light else "", light.describe(named=False) if light else "")
        pic = snap.picture
        if not self.views["camera"].own:                 # while Clampwise holds the camera, its view says so
            self.views["camera"].set_status("ok" if snap.camera_open else "muted",
                                            "open" if snap.camera_open else "not open",
                                            f"last picture {pic.path.name}" if pic else "")

    def _amplifier(self, snap) -> None:
        amp = snap.amplifier
        r = self.readouts
        if amp is None or not amp.connected:
            for w in r.values():
                w.set("–")
            self.meter_v.setText("–")
            self.meter_i.setText("–")
            self.channel_name.setText("Amplifier")
            self.meters.setText(amplifier_words(amp.status).capitalize() + "." if amp is not None and amp.status else "")
            return
        vc = amp.mode == 0
        src = f"read from the amplifier ({channel_words(amp.device)})"
        tg = getattr(snap, "telegraph", None)
        r["telegraph"].set(telegraph_words(tg) if tg is not None else "not received",
                           "what the amplifier's program telegraphs to an acquisition program, as Clampex receives "
                           "it; Clampwise records with these scales")
        r["mode"].set(amp.mode_name, src)
        r["holding"].set(amp.shown("holding") + ("" if amp.values.get("holding_on", True) else " (off)"), src)
        on = amp.values.get("whole_cell_on")
        off = "" if on else " (off)"
        why = "the amplifier's whole-cell compensation" + ("" if on else ": off, so these are its dials' settings, "
                                                           "not an estimate of the cell")
        r["rs"].set(amp.shown("rs") + off if vc else "–", why if vc else "voltage clamp only")
        r["cm"].set(amp.shown("cm") + off if vc else "–", why)
        r["rs_comp"].set(f"{amp.shown('rs_correction')} correction" if vc and amp.values.get("rs_comp_on") else
                         ("off" if vc else "–"), src)
        r["bridge"].set(amp.shown("bridge") if not vc and amp.values.get("bridge_on") else
                        ("off" if not vc else "–"), src)
        r["output"].set(f"{amp.shown('primary_signal')}, x{amp.values.get('primary_gain', math.nan):g}", src)
        r["filter"].set(amp.shown("primary_lpf"), src)
        r["seal"].set("on" if amp.values.get("test_signal_on") else "off", src)
        words = multiclamp.meter_words(amp)
        self.meters.setText(f"Meters: {words}" if words else "")
        number = amp.device.channel if amp.device is not None else 1
        self.channel_name.setText(f"Channel {number}")
        v, i = amp.meters.get(2 * (number - 1)), amp.meters.get(2 * (number - 1) + 1)
        self.meter_v.setText(f"{v * 1e3:.1f}" if v is not None and math.isfinite(v) else "–")
        self.meter_i.setText(f"{i * 1e12:.1f}" if i is not None and math.isfinite(i) else "–")
        rs = amp.values.get("rs") if vc and amp.values.get("whole_cell_on") else None
        if rs is not None and math.isfinite(rs) and rs > 0:
            self._history.append((snap.at, rs, amp.values.get("cm", math.nan), amp.values.get("holding", math.nan)))
            self._history = [h for h in self._history if snap.at - h[0] <= HISTORY_S]
            self._first_rs = self._first_rs or rs
            t0 = self._history[0][0]
            self.rs_curve.setData([(h[0] - t0) / 60 for h in self._history], [h[1] for h in self._history])
            self.plot.setVisible(True)
            if any(math.isfinite(s.rs_mohm) for s in self._sweeps):
                return                                   # the recorded test pulse measures Rs: it sets the badge
            change = rs / self._first_rs - 1
            self.rs_badge.setVisible(True)
            self.rs_badge.set("warn" if abs(change) > RS_CHANGE else "ok",
                              f"Rs {change:+.0%} since the readout began",
                              f"{self._first_rs:.1f} to {rs:.1f} MOhm; a change over {RS_CHANGE:.0%} is flagged (B19)")

    def _trace(self, snap) -> None:
        """Each sweep's test pulse from the file being written: its numbers, Rs on the plot, the badge."""
        if snap.sweeps:
            cell = str(Path(snap.writing).parent)            # a cell's files share its folder: a new folder, a new cell
            if cell != self._cell:
                self._cell, self._sweeps = cell, []
            self._sweeps += snap.sweeps
            self._sweeps = self._sweeps[-5000:]
        if not self._sweeps:
            self.trace.setText("")
            self.trace_curve.setData([], [])
            return
        last = self._sweeps[-1]
        unit = "pA" if last.clamp == "voltage clamp" else "mV"
        parts = [f"Rs {last.rs_mohm:.1f} MOhm", f"Rm {last.rm_mohm:.0f} MOhm", f"Cm {last.cm_pf:.0f} pF"] \
            if math.isfinite(last.rs_mohm) else []
        if last.clamp == "current clamp" and math.isfinite(last.rm_mohm):    # the current step's
            parts = [f"Rin {last.rm_mohm:.0f} MOhm"] + ([f"Cm {last.cm_pf:.0f} pF"] if math.isfinite(last.cm_pf)
                                                        else [])
        if last.no_cell:
            parts.append(f"no cell (Rt {last.rt_mohm:.1f} MOhm, below the {self._rin_floor():g} MOhm input-resistance "
                         "floor: an open pipette)")
        elif last.seal:
            rt = f"Rt {last.rt_mohm / 1e3:.1f} GOhm" if math.isfinite(last.rt_mohm) else "no current through the step"
            parts.append(f"seal ({rt}, above the {self._rin_ceiling():g} MOhm input-resistance ceiling: no cell open)")
        parts.append(f"{'holding' if unit == 'pA' else 'level'} {last.holding:.1f} {unit}")
        if math.isfinite(last.noise):
            parts.append(f"noise {last.noise:.1f} {unit} SD")
        self.trace.setText(f"From the recorded trace ({last.file}, sweep {last.sweep + 1}): " + ", ".join(parts) + ".")
        timed = [(s.start.timestamp() + s.time_s if s.start else math.nan, s.rs_mohm) for s in self._sweeps
                 if math.isfinite(s.rs_mohm)]
        timed = [x for x in timed if math.isfinite(x[0])]
        if not timed:
            return
        t0 = min([timed[0][0]] + ([self._history[0][0]] if self._history else []))
        self.trace_curve.setData([(x[0] - t0) / 60 for x in timed], [x[1] for x in timed])
        self.plot.setVisible(True)
        change = growing.rs_change(self._sweeps)
        if math.isfinite(change):
            first = timed[0][1]
            self.rs_badge.setVisible(True)
            self.rs_badge.set("warn" if abs(change) > RS_CHANGE else "ok", f"Rs {change:+.0%} since this cell's "
                              "first sweeps", f"from the recorded test pulse: {first:.1f} to {timed[-1][1]:.1f} MOhm; "
                              f"a change over {RS_CHANGE:.0%} is flagged (B19)")

    def _health(self, snap) -> None:
        from ...rig import health

        rows = health.readings(snap.amplifier, self._sweeps, snap.last_test)
        keys = {r.key for r in rows}
        for key, (cap, value) in self.health_rows.items():
            cap.setVisible(key in keys)
            value.setVisible(key in keys)
        for r in rows:
            if r.key not in self.health_rows:
                n = len(self.health_rows)
                cap, value = label(r.label, "muted"), widgets.Badge("muted", "–")
                self.health_grid.addWidget(cap, n, 0)
                self.health_grid.addWidget(value, n, 1)
                self.health_rows[r.key] = (cap, value)
            cap, value = self.health_rows[r.key]
            cap.setText(r.label)
            tip = (f"From {r.source}." if r.source else "Not read yet.") + (f" Limit: {r.limit}." if r.limit else "")
            value.set(r.status, "" if r.shown() == "–" else r.shown(), tip)
            cap.setToolTip(tip)

    @staticmethod
    def _rin_floor() -> float:
        from ...analysis import registry

        return registry.load()["quality"].defaults()["rin_low_mohm"]

    @staticmethod
    def _rin_ceiling() -> float:
        from ...analysis import registry

        return registry.load()["quality"].defaults()["rin_high_mohm"]

    def _records(self, snap) -> None:
        f, t = snap.last_file, snap.last_test
        where = snap.data_folder
        if f is not None:
            protocol = f.protocol.rsplit("\\", 1)[-1] if f.protocol else "no protocol logged"
            self.acquisition.setText(f"Acquisition: writing to {where or 'its data folder'}; last file {f.name} "
                                     f"({f.outcome}), protocol {protocol}.")
        else:
            self.acquisition.setText("Acquisition: nothing logged yet.")
        self.test.setText(f"Last Membrane Test: {t.describe()}." if t is not None and t.values else "")
        light = snap.light
        self.light.setText(f"Patterned light: {light.describe(named=False)}"
                           + ("" if snap.light_open else " (the last session; its program is closed)")
                           if light is not None else "Patterned light: no session found.")
        pic = snap.picture
        if pic is None:
            self.picture.setText("Camera: no picture saved yet.")
            return
        when = rig_state.since(pic.taken)
        self.picture.setText(f"Camera's last picture{f', {when}' if when else ''}: {pic.describe()}."
                             + ("".join(f" {n}." for n in pic.notes)))

    # ---- notes
    def add_note(self) -> None:
        text = self.note.text().strip()
        if not text:
            return
        if self.window.project is None:
            self.window.say("Open the experiment's folder first: notes are kept with its project.")
            return
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        target = None
        if self.last is not None and self.last.last_file is not None:
            rows = [row for row in self.window.project.recordings() if row.name == self.last.last_file.name]
            target = rows[-1].path if rows else None
        self.window.project.add_note(target, f"{stamp} {text}", source="typed at the rig")
        self.note.clear()
        self.window.say(f"Note kept at {stamp}" + (f" with {target.name}." if target else " with the project."))
