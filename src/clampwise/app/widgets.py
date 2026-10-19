"""The small pieces every screen is built from (B0), so every screen looks and
behaves the same: the primary button, the flag badge, the one-line hint and
"?" help, the "Advanced (n changed)" section, the empty state that teaches,
a hairline. Colours and sizes come from the theme through object names; no
widget here sets its own.
"""

from __future__ import annotations

from typing import Callable

from pyqtgraph.Qt import QtCore, QtWidgets

from . import theme

S = theme.SPACE


def primary(text: str, slot: Callable | None = None, tooltip: str = "") -> QtWidgets.QPushButton:
    """The one accent-coloured button of a screen: its next step (B0 item 3)."""
    b = QtWidgets.QPushButton(text)
    b.setObjectName("primary")
    b.setCursor(QtCore.Qt.PointingHandCursor)
    if tooltip:
        b.setToolTip(tooltip)
    if slot is not None:
        b.clicked.connect(slot)
    return b


def button(text: str, slot: Callable | None = None, tooltip: str = "", quiet: bool = False) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    if quiet:
        b.setObjectName("quiet")
    if tooltip:
        b.setToolTip(tooltip)
    if slot is not None:
        b.clicked.connect(slot)
    return b


def label(text: str, kind: str = "", wrap: bool = False) -> QtWidgets.QLabel:
    """kind: "" (body), "title", "heading", "muted", "hint"."""
    lab = QtWidgets.QLabel(text)
    if kind:
        lab.setObjectName(kind)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse)
    return lab


def hairline() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setObjectName("hairline")
    f.setFrameShape(QtWidgets.QFrame.HLine)
    return f


def box(direction: str = "v", spacing: str = "s", margins: str | None = None) -> QtWidgets.QBoxLayout:
    """A layout on the spacing scale (B0 item 11)."""
    lay = QtWidgets.QVBoxLayout() if direction == "v" else QtWidgets.QHBoxLayout()
    lay.setSpacing(S[spacing])
    m = S[margins] if margins else 0
    lay.setContentsMargins(m, m, m, m)
    return lay


def _main_window(w: QtWidgets.QWidget | None):
    """The Clampwise window ``w`` belongs to (the one that keeps the settings), or None."""
    while w is not None and not hasattr(w, "persist"):
        w = w.parentWidget()
    return w


class _KeepSize(QtCore.QObject):
    """Saves a window's size and place under its key whenever it is moved, resized or closed."""

    def __init__(self, window, key: str, parent) -> None:
        super().__init__(parent)
        self.window, self.key = window, key

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Move, QtCore.QEvent.Hide) and obj.isVisible():
            self.window._settings().setValue(f"windows/{self.key}", obj.saveGeometry())
        return False


def resizable(window: QtWidgets.QWidget, lay: QtWidgets.QLayout, key: str = "",
              size: tuple[int, int] | None = None) -> QtWidgets.QScrollArea:
    """``lay`` as ``window``'s content, in a window sized freely: a grip in its corner, the content scrolled when
    the window is made smaller than it (never a minimum size it cannot go below), and its size and place kept
    between sessions under ``key`` (``size`` until it has one)."""
    inner = QtWidgets.QWidget()
    inner.setLayout(lay)
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(inner)
    outer = box("v", "xs")
    outer.addWidget(area)
    window.setLayout(outer)
    window.setMinimumSize(240, 160)
    if isinstance(window, QtWidgets.QDialog):
        window.setSizeGripEnabled(True)
    if size:
        window.resize(*size)
    main = _main_window(window.parentWidget())
    if key and main is not None and getattr(main, "persist", False):
        saved = main._settings().value(f"windows/{key}")
        if saved is not None:
            window.restoreGeometry(saved)
        window.installEventFilter(_KeepSize(main, key, window))
    return area


class Badge(QtWidgets.QLabel):
    """One badge style for every flag (B0 item 15): a symbol and a short word,
    never colour alone; the full reason, limit and source in the tooltip."""

    SYMBOLS = {"ok": "✓", "warn": "⚠", "bad": "✖", "muted": "–"}

    def __init__(self, status: str, word: str, detail: str = "") -> None:
        super().__init__()
        self.set(status, word, detail)

    def set(self, status: str, word: str, detail: str = "") -> None:
        status = status if status in self.SYMBOLS else "muted"
        self.setObjectName(f"badge_{status}")
        self.setText(f"{self.SYMBOLS[status]} {word}")
        self.setToolTip(detail or word)
        self.setAccessibleName(f"{word}. {detail}" if detail else word)
        self.style().unpolish(self)
        self.style().polish(self)


QUALITY = {"passed": ("ok", "passed"), "flagged": ("warn", "flagged"), "not checked": ("muted", "not checked")}


class HelpButton(QtWidgets.QToolButton):
    """The "?" that opens a short explanation with an example and a link to the
    manual entry (B0 item 19). ``open_manual`` is set by the window."""

    open_manual: Callable[[str], None] | None = None

    def __init__(self, title: str, text: str, manual_key: str = "") -> None:
        super().__init__()
        self.setText("?")
        self.setToolTip(f"What is {title.lower()}?")
        self.setAccessibleName(f"Help: {title}")
        self.title, self.body, self.manual_key = title, text, manual_key
        self.clicked.connect(self._show)

    def _show(self) -> None:
        box_ = QtWidgets.QMessageBox(self)
        box_.setWindowTitle(self.title)
        box_.setTextFormat(QtCore.Qt.RichText)
        box_.setText(self.body)
        more = box_.addButton("Open the manual entry", QtWidgets.QMessageBox.ActionRole) if self.manual_key else None
        box_.addButton(QtWidgets.QMessageBox.Close)
        box_.exec_()
        if more is not None and box_.clickedButton() is more and HelpButton.open_manual:
            HelpButton.open_manual(self.manual_key)


def field(caption: str, control: QtWidgets.QWidget, hint: str = "", help_title: str = "", help_text: str = "",
          manual_key: str = "", inline: bool = False) -> QtWidgets.QWidget:
    """A control with its label above it, an optional one-line hint under it and
    a "?" beside the label: the same on every screen (B0 items 11, 19).
    ``inline``: the label on the left and the control beside it on one line, the
    hint as their tooltip (dense panels such as the figure builder's)."""
    w = QtWidgets.QWidget()
    if inline:
        row = box("h", "s")
        left = QtWidgets.QWidget()
        left.setFixedWidth(160)
        ll = box("h", "xs")
        cap = label(caption)
        cap.setWordWrap(True)
        ll.addWidget(cap, 1)
        if help_text:
            ll.addWidget(HelpButton(help_title or caption, help_text, manual_key), 0, QtCore.Qt.AlignTop)
        left.setLayout(ll)
        row.addWidget(left)
        row.addWidget(control, 1)
        if hint:
            cap.setToolTip(hint)
            if not control.toolTip():
                control.setToolTip(hint)
        w.setLayout(row)
        return w
    lay = box("v", "xs")
    top = box("h", "xs")
    top.addWidget(label(caption))
    if help_text:
        top.addWidget(HelpButton(help_title or caption, help_text, manual_key))
    top.addStretch(1)
    lay.addLayout(top)
    lay.addWidget(control)
    if hint:
        lay.addWidget(label(hint, "hint", wrap=True))
    w.setLayout(lay)
    return w


class Advanced(QtWidgets.QWidget):
    """A collapsed section for the settings people rarely change (B0 item 8).
    Its header says how many differ from their default."""

    toggled = QtCore.Signal(bool)

    def __init__(self, title: str = "Advanced") -> None:
        super().__init__()
        self._title = title
        self._changed = 0
        lay = box("v", "s")
        self.header = QtWidgets.QToolButton()
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(QtCore.Qt.RightArrow)
        self.header.toggled.connect(self._toggle)
        self.body = QtWidgets.QWidget()
        self.body_layout = box("v", "m")
        self.body.setLayout(self.body_layout)
        self.body.setVisible(False)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.setLayout(lay)
        self._refresh()

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(on)
        self.header.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        self.toggled.emit(on)

    def set_changed(self, n: int) -> None:
        self._changed = n
        self._refresh()

    def _refresh(self) -> None:
        self.header.setText(f"{self._title} ({self._changed} changed)" if self._changed else self._title)

    def add(self, widget: QtWidgets.QWidget) -> None:
        self.body_layout.addWidget(widget)


class EmptyState(QtWidgets.QFrame):
    """What a screen says when it has nothing to show: what it is for, what to
    do, and the button to do it (B0 item 10)."""

    def __init__(self, title: str, text: str, actions: list[tuple[str, Callable, bool]] = ()) -> None:
        super().__init__()
        self.setObjectName("empty")
        outer = box("v", "m", "xl")
        outer.addStretch(1)
        inner = box("v", "s")
        t = label(title, "heading")
        t.setAlignment(QtCore.Qt.AlignCenter)
        body = label(text, "muted", wrap=True)
        body.setAlignment(QtCore.Qt.AlignCenter)
        # a wrapped label in a centring layout is squeezed to its shortest line
        # and clipped; a row with stretch either side gives it a real width
        body.setMinimumWidth(min(theme.READING_WIDTH // 2 + 120, theme.READING_WIDTH))
        body.setMaximumWidth(theme.READING_WIDTH)
        body.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        inner.addWidget(t)
        centred = box("h", "s")
        centred.addStretch(1)
        centred.addWidget(body)
        centred.addStretch(1)
        inner.addLayout(centred)
        row = box("h", "s")
        row.addStretch(1)
        for text_, slot, is_primary in actions:
            row.addWidget(primary(text_, slot) if is_primary else button(text_, slot))
        row.addStretch(1)
        inner.addSpacing(S["s"])
        inner.addLayout(row)
        outer.addLayout(inner)
        outer.addStretch(2)
        self.setLayout(outer)
