"""The figure on screen as something to work on directly: click a thing to select it, drag it to move it,
double-click its text to change it there, right-click for what can be done with it.

The picture is matplotlib's (what exports is what is seen). ``figure.render`` tags everything it draws that a
person can select; this canvas asks it what is under the mouse, draws the selection's outline (and a panel's
resize handles) over the picture, and reports what the person did. It changes nothing itself: the builder turns
a move or a new text into a change of the figure's ``Spec`` and draws again.
"""

from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import elements, render
from . import theme

HANDLE = 7                   # a resize handle's side, in screen pixels
TEXT_KINDS = ("title", "xlabel", "ylabel", "annotation", "letter", "bracket", "stimbar", "figtext", "series",
              "vmlabel", "refline", "marker")
HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


class FigureCanvas(FigureCanvasQTAgg):
    selected = QtCore.Signal(object)                  # the Tag selected (None: nothing)
    moved = QtCore.Signal(object, float, float)       # a thing dragged: its Tag, and how far (figure pixels, y up)
    resized = QtCore.Signal(object, str, float, float)    # a panel's handle dragged: Tag, which handle, how far
    text_edited = QtCore.Signal(object, str)          # a thing's words changed in place
    menu_requested = QtCore.Signal(object, object)    # right-click: the Tag under it (or None), the global position
    activated = QtCore.Signal(object)                 # double-click on a thing with no words of its own
    removed = QtCore.Signal(object)                   # Delete pressed on the selection
    nudged = QtCore.Signal(object, float, float)      # an arrow key: the selection moved by so many mm

    def __init__(self, fig) -> None:
        super().__init__(fig)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.drawn: list = []
        self.selection = None                         # a Tag
        self.accent = QtGui.QColor(theme.LIGHT.accent)        # (the builder sets the window's own)
        self.can_move = lambda tag: False             # set by the builder: whether a thing may be dragged
        self.can_resize = lambda tag: False
        self.words_of = lambda tag: None              # a thing's words (None: it has none to edit)
        self.look: dict = {}                          # the figure's own settings (render.rc), drawn under
        self._press = None                            # (tag, x, y, handle) where a drag began
        self._drag = (0.0, 0.0)
        self._editor: QtWidgets.QLineEdit | None = None

    def draw(self, *args, **kwargs):
        """Drawn under the figure's own settings, as an export is: the same fonts and ticks on screen."""
        import matplotlib.pyplot as plt

        with plt.rc_context(self.look):
            return super().draw(*args, **kwargs)

    # ---- what the builder tells it
    def set_drawn(self, drawn: list) -> None:
        """After each redraw: what is drawn now (the selection stays when it still is)."""
        self.drawn = list(drawn)
        if self.selection is not None and self.outline(self.selection) is None:
            self.select(None)
        self.update()

    def select(self, tag, tell: bool = True) -> None:
        if tag != self.selection:
            self.selection = tag
            if tell:
                self.selected.emit(tag)
        self.update()

    # ---- geometry: the figure's pixels (y up) and the widget's (y down)
    def _ratio(self) -> float:
        return float(getattr(self, "device_pixel_ratio", None) or self.devicePixelRatioF() or 1.0)

    def to_figure(self, pos) -> tuple[float, float]:
        return self.mouseEventCoords(pos)

    def to_widget(self, x: float, y: float) -> QtCore.QPointF:
        k = self._ratio()
        return QtCore.QPointF(x / k, (self.figure.bbox.height - y) / k)

    def mm_per_pixel(self) -> float:
        """Millimetres on the page per figure pixel."""
        return 25.4 / float(self.figure.dpi)

    def outline(self, tag):
        """The selection's outline in figure pixels (x0, y0, x1, y1), or None when it is not drawn."""
        if tag is None:
            return None
        if tag.kind in ("panel", "xaxis", "yaxis"):
            d = self.panel_drawn(tag.panel)
            if d is None:
                return None
            if tag.kind == "panel":
                return render.panel_outline(self.figure, d)
            ax = d.axes[0] if d.axes else None
            if ax is None:
                return None
            r = render._renderer(self.figure)
            bb, tight = ax.get_window_extent(r), ax.get_tightbbox(r)
            if tight is None:
                return None
            return (tight.x0, bb.y0, bb.x0, bb.y1) if tag.kind == "yaxis" else (bb.x0, tight.y0, bb.x1, bb.y0)
        return render.extent(self.figure, tag)

    def panel_drawn(self, k: int):
        for d in self.drawn:
            if d.axes and getattr(d.axes[0], "_cw_panel", -1) == k:
                return d
        return None

    def axes_of(self, k: int):
        d = self.panel_drawn(k)
        return d.axes[0] if d is not None and d.axes else None

    def hit(self, pos):
        x, y = self.to_figure(pos)
        return render.hit(self.figure, self.drawn, x, y)

    def _handle_at(self, pos) -> str:
        """Which resize handle of the selected panel is under the mouse ("" none)."""
        if self.selection is None or self.selection.kind != "panel" or not self.can_resize(self.selection):
            return ""
        for name, rect in self._handles().items():
            if rect.adjusted(-2, -2, 2, 2).contains(QtCore.QPointF(pos)):
                return name
        return ""

    def _handles(self) -> dict:
        o = self.outline(self.selection)
        if o is None:
            return {}
        a, b = self.to_widget(o[0], o[3]), self.to_widget(o[2], o[1])       # top left, bottom right on screen
        xs = {"w": a.x(), "": (a.x() + b.x()) / 2, "e": b.x()}
        ys = {"n": a.y(), "": (a.y() + b.y()) / 2, "s": b.y()}
        out = {}
        for name in HANDLES:
            v = "n" if "n" in name else "s" if "s" in name else ""
            h = "w" if "w" in name else "e" if "e" in name else ""
            out[name] = QtCore.QRectF(xs[h] - HANDLE / 2, ys[v] - HANDLE / 2, HANDLE, HANDLE)
        return out

    # ---- the mouse
    def mousePressEvent(self, event) -> None:
        self._commit_editor()
        self.setFocus()
        if event.button() == QtCore.Qt.RightButton:
            tag = self.hit(event.pos())
            if tag is not None:
                self.select(tag)
            self.menu_requested.emit(tag, event.globalPos())
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        handle = self._handle_at(event.pos())
        tag = self.selection if handle else self.hit(event.pos())
        self.select(tag)
        if tag is not None and (handle or self.can_move(tag)):
            x, y = self.to_figure(event.pos())
            self._press, self._drag = (tag, x, y, handle), (0.0, 0.0)

    def mouseMoveEvent(self, event) -> None:
        if self._press is None:
            handle = self._handle_at(event.pos())
            cursors = {"n": QtCore.Qt.SizeVerCursor, "s": QtCore.Qt.SizeVerCursor, "e": QtCore.Qt.SizeHorCursor,
                       "w": QtCore.Qt.SizeHorCursor, "nw": QtCore.Qt.SizeFDiagCursor, "se": QtCore.Qt.SizeFDiagCursor,
                       "ne": QtCore.Qt.SizeBDiagCursor, "sw": QtCore.Qt.SizeBDiagCursor}
            if handle:
                self.setCursor(cursors[handle])
            else:
                tag = self.hit(event.pos())
                self.setCursor(QtCore.Qt.SizeAllCursor if tag is not None and self.can_move(tag) else
                               QtCore.Qt.PointingHandCursor if tag is not None else QtCore.Qt.ArrowCursor)
            return
        tag, x0, y0, _handle = self._press
        x, y = self.to_figure(event.pos())
        dx, dy = x - x0, y - y0
        how = elements.MOVES.get(tag.kind, "")
        if not _handle and how in ("axes_y", "data_y"):
            dx = 0.0                                  # up and down only
        self._drag = (dx, dy)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._press is None or event.button() != QtCore.Qt.LeftButton:
            return
        tag, _x0, _y0, handle = self._press
        dx, dy = self._drag
        self._press, self._drag = None, (0.0, 0.0)
        self.update()
        if abs(dx) + abs(dy) < 2.0:                   # a click, not a drag
            return
        if handle:
            self.resized.emit(tag, handle, dx, dy)
        else:
            self.moved.emit(tag, dx, dy)

    def mouseDoubleClickEvent(self, event) -> None:
        tag = self.hit(event.pos())
        self._press = None
        if tag is None:
            return
        self.select(tag)
        if self.words_of(tag) is not None:
            self.edit_text(tag)
        else:
            self.activated.emit(tag)

    def keyPressEvent(self, event) -> None:
        tag = self.selection
        step = 2.0 if event.modifiers() & QtCore.Qt.ShiftModifier else 0.5          # mm
        moves = {QtCore.Qt.Key_Left: (-step, 0.0), QtCore.Qt.Key_Right: (step, 0.0), QtCore.Qt.Key_Up: (0.0, step),
                 QtCore.Qt.Key_Down: (0.0, -step)}
        if tag is not None and event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.removed.emit(tag)
        elif tag is not None and event.key() in moves and self.can_move(tag):
            self.nudged.emit(tag, *moves[event.key()])
        elif tag is not None and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_F2) \
                and self.words_of(tag) is not None:
            self.edit_text(tag)
        elif event.key() == QtCore.Qt.Key_Escape:
            self.select(None)
        else:
            super().keyPressEvent(event)

    # ---- words changed in place
    def edit_text(self, tag) -> None:
        """A line of text over the thing, holding its words; Enter keeps them, Esc leaves them."""
        self._commit_editor(keep=False)
        words = self.words_of(tag)
        o = self.outline(tag)
        if words is None:
            return
        e = QtWidgets.QLineEdit(str(words), self)
        e.setObjectName("inplace")
        e.setFrame(True)
        if o is not None:
            a, b = self.to_widget(o[0], o[3]), self.to_widget(o[2], o[1])
            w = max(160, int(b.x() - a.x()) + 40)
            x = int(min(max(a.x() - 4, 0), max(self.width() - w, 0)))
            e.setGeometry(x, int(max(a.y() - 4, 0)), w, max(24, int(b.y() - a.y()) + 8))
        else:
            e.setGeometry(8, 8, 220, 26)
        e._tag = tag
        e.returnPressed.connect(lambda: self._commit_editor())
        e.installEventFilter(self)
        self._editor = e
        e.show()
        e.setFocus()
        e.selectAll()

    def _commit_editor(self, keep: bool = True) -> None:
        e, self._editor = self._editor, None
        if e is None:
            return
        tag, text = e._tag, e.text()
        e.hide()
        e.deleteLater()
        if keep and text != str(self.words_of(tag) or ""):
            self.text_edited.emit(tag, text)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._editor:
            if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
                self._commit_editor(keep=False)
                return True
            if event.type() == QtCore.QEvent.FocusOut:
                self._commit_editor()
            return False
        return False

    # ---- the selection's outline over the picture
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        tag = self.selection
        o = self.outline(tag) if tag is not None else None
        if o is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        a, b = self.to_widget(o[0], o[3]), self.to_widget(o[2], o[1])
        rect = QtCore.QRectF(a, b)
        pen = QtGui.QPen(self.accent, 1.0, QtCore.Qt.DashLine if tag.kind != "panel" else QtCore.Qt.SolidLine)
        p.setPen(pen)
        fill = QtGui.QColor(self.accent)
        fill.setAlpha(18)
        p.setBrush(fill)
        p.drawRect(rect)
        if self._press is not None and (abs(self._drag[0]) + abs(self._drag[1])) >= 2.0:
            k = self._ratio()
            dx, dy = self._drag[0] / k, -self._drag[1] / k
            moved = QtCore.QRectF(rect)
            handle = self._press[3]
            if handle:
                if "w" in handle:
                    moved.setLeft(rect.left() + dx)
                if "e" in handle:
                    moved.setRight(rect.right() + dx)
                if "n" in handle:
                    moved.setTop(rect.top() + dy)
                if "s" in handle:
                    moved.setBottom(rect.bottom() + dy)
            else:
                moved.translate(dx, dy)
            p.setPen(QtGui.QPen(self.accent, 1.2, QtCore.Qt.SolidLine))
            fill.setAlpha(45)
            p.setBrush(fill)
            p.drawRect(moved.normalized())
        if tag.kind == "panel" and self.can_resize(tag):
            p.setPen(QtGui.QPen(self.accent, 1.0))
            p.setBrush(QtGui.QColor(QtCore.Qt.white))
            for rect_h in self._handles().values():
                p.drawRect(rect_h)
        p.end()
