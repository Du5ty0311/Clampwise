"""A rig program's window in Live, shown live by the compositor (a click brings the program forward), or
Clampwise's own panel in its place (Use here): enlarged over the other views or hidden, only part of the window, out
in a window of its own, full screen with the picture alone, or the picture alone in place with an even margin."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import windows as rig_windows
from .. import theme, widgets
from ..widgets import box, label

WHOLE = "whole"                        # a view's part: the whole window, chosen
FILL = {"acquisition"}                 # stretched to fill by default


class WindowView(QtWidgets.QFrame):
    """A rig program's window shown live in this frame by the compositor; a click brings the program forward.
    It can be enlarged over the other views or hidden, and show only part of the window: the
    one the user drags out, else the program's usual part (rig.windows.PARTS: the camera's picture)."""

    thumbnail_cls = rig_windows.Thumbnail
    enlarge = QtCore.Signal(str)
    hide_me = QtCore.Signal(str)
    pop = QtCore.Signal(str)                          # out into a window of its own, or back
    fullscreen = QtCore.Signal(str)                   # full screen, the picture alone (Esc leaves), or back
    fill_changed = QtCore.Signal(str, bool)
    part_chosen = QtCore.Signal(str, object)          # role, (left, top, right, bottom) px in from the edges, or WHOLE
    own_changed = QtCore.Signal(str, bool)            # role, Clampwise's own panel shown (Use here) or not

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role
        self.part: tuple | str | None = None          # the user's: a part, WHOLE, or None (the program's usual)
        self.panel: QtWidgets.QWidget | None = None   # Clampwise's own picture in place of the program's window
        self.own = False
        self.popped = False                           # a window of its own
        self.bare = False                             # the picture alone: no title row, status line or border
        self.margin = theme.SPACE["s"]                # round the picture, even on every side (0: the picture alone)
        self.bare_chosen = False                      # ... as chosen for the view (full screen is always bare)
        self.full_popped = False                      # out of Live only to go full screen: put back when left
        self._shown_before: dict = {}
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
        self.failure = ""                             # why Use here could not take the device, said until it can
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
            self.state.setText("Not open" + (f"\n\nUse here could not take it: {self.failure}" if self.failure else ""))
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
        lay = self.layout()                            # the spacers round "Not open" give way to the picture
        for k in range(lay.count()):
            if lay.itemAt(k).spacerItem() is not None:
                lay.setStretch(k, 0 if on else 1)
        if on:
            self._drop()
            self.set_status("ok", "in Clampwise", self.held)
        else:
            self.show_window(self.source)
        self.own_changed.emit(self.role, on)

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
        self._fit_head()
        self._place()

    def set_bare(self, on: bool) -> None:
        """The picture alone (on): the title row, the status line, the border and the margins hidden; or all back,
        as they were."""
        parts = [w for w in (self.name, self.badge, self.detail, self.enlarge_button, self.hide_button,
                             self.pop_button, getattr(self, "use_button", None)) if w is not None]
        if on and not self.bare:
            self._shown_before = {w: not w.isHidden() for w in parts}
            for w in parts:
                w.setVisible(False)
        elif not on and self.bare:
            for w in parts:
                w.setVisible(self._shown_before.get(w, True))
        self.bare = on
        m = 0 if self.isFullScreen() or getattr(self, "_going_full", False) else self.margin
        self.layout().setContentsMargins(m, m, m, m)
        self.layout().setSpacing(0 if on else theme.SPACE["xs"])
        self.setObjectName("bare" if on else "desk")             # no border drawn round it
        self.style().unpolish(self)
        self.style().polish(self)
        self.setFrameShape(self.frameShape())                    # its width read again from the new style
        if not on:
            self._fit_head()

    def set_margin(self, px: int) -> None:
        """The space round the picture, the same on every side (0: the picture alone, edge to edge)."""
        self.margin = int(px)
        if not self.isFullScreen():
            self.layout().setContentsMargins(self.margin, self.margin, self.margin, self.margin)

    def _fit_head(self) -> None:
        """A narrow view keeps its name and its Use here (or Let go) readable: the status word gives way first,
        then the enlarge and hide buttons (pop out stays: a window of its own is as large as wanted)."""
        if self.bare:
            return
        tools_allowed = not self.popped and not getattr(self, "_free", False)
        use = getattr(self, "use_button", None)
        need = self.name.minimumWidth() + self.pop_button.sizeHint().width() + 40
        if use is not None:
            need += use.sizeHint().width()
        tools = self.enlarge_button.sizeHint().width() + self.hide_button.sizeHint().width()
        room = self.width()
        self.enlarge_button.setVisible(tools_allowed and room >= need + tools)
        self.hide_button.setVisible(tools_allowed and room >= need + tools)
        self.badge.setVisible(room >= need + (tools if tools_allowed else 0) + self.badge.sizeHint().width())
        if use is not None:
            use.setMinimumWidth(use.sizeHint().width())

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
        if not self.popped or self._thumb is None or self.isFullScreen():
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
        self._free = on
        self.enlarge_button.setVisible(not on and not self.popped)
        self.hide_button.setVisible(not on and not self.popped)
        self._fit_head()

    def set_popped(self, on: bool) -> None:
        self.popped = on
        self.enlarge_button.setVisible(not on)
        self.hide_button.setVisible(not on)
        self._fit_head()
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
            menu.addAction("Leave full screen (Esc)" if self.isFullScreen() else "Full screen, the picture alone "
                           "(Esc leaves)", lambda: self.fullscreen.emit(self.role))
            menu.addAction("Fit to the program's shape (as large as the screen)", self.fit_to_source)
            menu.addSeparator()
        else:
            menu.addAction("Full screen, the picture alone (Esc leaves)", lambda: self.fullscreen.emit(self.role))
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
        if event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_F11) and self.isFullScreen():
            self.fullscreen.emit(self.role)
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
