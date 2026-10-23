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


class OneLine(QtWidgets.QLabel):
    """A label kept to one line (cut at the panel's edge rather than growing it), its whole text on hover."""

    def __init__(self, text: str = "", kind: str = "muted") -> None:
        super().__init__()
        if kind:
            self.setObjectName(kind)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt's name)
        super().setText(text)
        self.setToolTip(text)


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
        if getattr(self, "_shown", None) == (status, word, detail):
            return                    # unchanged: no restyle (Live sets its badges twice a second)
        self._shown = (status, word, detail)
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


class Section(QtWidgets.QFrame):
    """A panel's section that folds to its header: its title, a badge that says how it stands (so a folded section
    still shows a warning), and buttons that stay usable folded (Off, Use here). A click on the header folds or
    unfolds it; ``Sections`` keeps the order and the folds."""

    toggled = QtCore.Signal(bool)                  # True: unfolded
    menu = QtCore.Signal(object)                   # the header right-clicked, at this global point

    def __init__(self, key: str, title: str) -> None:
        super().__init__()
        self.key = key
        self.setObjectName("desk")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        lay = box("v", "xs", "s")
        self.head = QtWidgets.QWidget()
        self.head.setCursor(QtCore.Qt.PointingHandCursor)
        self.head.setToolTip("Click to fold or unfold; right-click to fold the others or move it")
        head = box("h", "s")
        self.arrow = QtWidgets.QToolButton()
        self.arrow.setAutoRaise(True)
        self.arrow.setArrowType(QtCore.Qt.DownArrow)
        self.arrow.setAccessibleName(f"Fold or unfold {title}")
        self.arrow.clicked.connect(lambda: self.set_open(not self.open))
        self.title = label(title, "heading")
        self.title.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)     # a click on it folds, not selects
        self.title.setMinimumWidth(60)
        self.title.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.badge = Badge("muted", "")
        self.badge.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # clipped, not wider
        self.badge.setVisible(False)
        head.addWidget(self.arrow)
        head.addWidget(self.title, 2)
        head.addWidget(self.badge, 3)
        self.tools = box("h", "xs")
        head.addLayout(self.tools)
        self.head.setLayout(head)
        self.head.installEventFilter(self)
        self.title.installEventFilter(self)
        self.badge.installEventFilter(self)
        lay.addWidget(self.head)
        self.body = QtWidgets.QWidget()
        self.body_layout = box("v", "xs")
        self.body.setLayout(self.body_layout)
        lay.addWidget(self.body)
        self.setLayout(lay)
        self.open = True

    def add(self, item, stretch: int = 0) -> None:
        if isinstance(item, QtWidgets.QLayout):
            self.body_layout.addLayout(item, stretch)
        else:
            self.body_layout.addWidget(item, stretch)

    def add_tool(self, widget: QtWidgets.QWidget) -> None:
        """A button on the header, usable while the section is folded."""
        self.tools.addWidget(widget)

    def set_summary(self, status: str, word: str, detail: str = "") -> None:
        self.badge.setVisible(bool(word))
        self.badge.set(status, word, detail)

    def set_open(self, on: bool) -> None:
        if on == self.open:
            return
        self.open = on
        self.body.setVisible(on)
        self.arrow.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        self.toggled.emit(on)

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            self.set_open(not self.open)
            return True
        if kind == QtCore.QEvent.ContextMenu:
            self.menu.emit(event.globalPos())
            return True
        return super().eventFilter(obj, event)


class Sections(QtWidgets.QWidget):
    """Sections one above the other, each folded or unfolded as chosen, in the order chosen: both kept between
    sessions under ``key`` (``settings`` gives the QSettings, or None when nothing is kept)."""

    def __init__(self, key: str, settings: Callable | None = None, folded: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.key, self._settings, self._folded = key, settings, set(folded)
        self.sections: dict[str, Section] = {}
        self._lay = box("v", "s")
        self._lay.addStretch(1)
        self.setLayout(self._lay)
        self._loading = True                       # nothing saved until load(): the defaults set by add() would
                                                   # overwrite what was kept before it is read

    def add(self, section: Section) -> Section:
        self.sections[section.key] = section
        self._lay.insertWidget(self._lay.count() - 1, section)
        section.set_open(section.key not in self._folded)
        section.toggled.connect(lambda *_: self.save())
        section.menu.connect(lambda at, k=section.key: self._menu(k, at))
        return section

    def arrange(self, keys) -> None:
        """The sections in this order (the default, before what was kept is loaded); others after them."""
        for k in list(keys) + [k for k in self.order() if k not in keys]:
            self._lay.removeWidget(self.sections[k])
            self._lay.insertWidget(self._lay.count() - 1, self.sections[k])

    def order(self) -> list[str]:
        return [self._lay.itemAt(i).widget().key for i in range(self._lay.count() - 1)]

    def move(self, key: str, by: int) -> None:
        keys = self.order()
        k = keys.index(key)
        to = min(max(k + by, 0), len(keys) - 1)
        if to != k:
            self._lay.removeWidget(self.sections[key])
            self._lay.insertWidget(to, self.sections[key])
            self.save()

    def only(self, key: str) -> None:
        """This section unfolded, every other one folded."""
        self._loading = True
        for k, s in self.sections.items():
            s.set_open(k == key)
        self._loading = False
        self.save()

    def every(self, on: bool) -> None:
        self._loading = True
        for s in self.sections.values():
            s.set_open(on)
        self._loading = False
        self.save()

    def _menu(self, key: str, at) -> None:
        m = QtWidgets.QMenu(self)
        m.addAction("Only this one unfolded", lambda: self.only(key))
        m.addAction("Unfold all", lambda: self.every(True))
        m.addAction("Fold all", lambda: self.every(False))
        m.addSeparator()
        keys = [k for k in self.order() if not self.sections[k].isHidden()]
        m.addAction("Move up", lambda: self.move(key, -1)).setEnabled(keys[:1] != [key])
        m.addAction("Move down", lambda: self.move(key, 1)).setEnabled(keys[-1:] != [key])
        m.exec_(at)

    def load(self) -> None:
        """The order and folds kept (sections added since keep their place after the kept ones); called once every
        section is added."""
        s = self._settings() if self._settings is not None else None
        if s is None:
            self._loading = False
            return
        self._loading = True
        # kept as text: an empty list comes back from QSettings as nothing, which would bring the default folds back
        order = [k for k in str(s.value(f"{self.key}/order", "") or "").split(",") if k in self.sections]
        for k in order + [k for k in self.order() if k not in order]:
            self._lay.removeWidget(self.sections[k])
            self._lay.insertWidget(self._lay.count() - 1, self.sections[k])
        folded = s.value(f"{self.key}/folded")
        if folded is not None:
            folded = set(str(folded).split(","))
            for k, section in self.sections.items():
                section.set_open(k not in folded)
        self._loading = False

    def save(self) -> None:
        s = self._settings() if self._settings is not None else None
        if s is None or self._loading:
            return
        s.setValue(f"{self.key}/order", ",".join(self.order()))
        s.setValue(f"{self.key}/folded", ",".join(sorted(k for k, sec in self.sections.items() if not sec.open)))


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
