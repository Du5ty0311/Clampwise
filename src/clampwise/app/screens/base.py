"""What every screen is: a title, a one-line purpose, one primary action, a
details panel, and a way to be told which recording is selected."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

from ..widgets import box, label


class Screen(QtWidgets.QWidget):
    key = ""
    title = ""
    purpose = ""
    compact = False          # a screen that needs its room (Live): the purpose goes into the title's tooltip

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.details = QtWidgets.QWidget()
        self._outer = box("v", "m")
        head = box("h", "m")
        titles = box("v", "xs")
        heading = label(self.title, "title")
        titles.addWidget(heading)
        self.subtitle = label(self.purpose, "muted", wrap=True)
        titles.addWidget(self.subtitle)
        if self.compact:
            self.subtitle.hide()
            heading.setToolTip(self.purpose)
        head.addLayout(titles, 1)
        self.actions = box("h", "s")
        head.addLayout(self.actions)
        self._outer.addLayout(head)
        self.setLayout(self._outer)

    def body(self, widget: QtWidgets.QWidget, stretch: int = 1) -> None:
        self._outer.addWidget(widget, stretch)

    def add_action(self, w: QtWidgets.QWidget) -> None:
        self.actions.addWidget(w)

    # called by the window
    def on_project(self) -> None:
        pass

    def on_selection(self, path) -> None:
        pass

    def on_shown(self) -> None:
        pass

    def is_current(self) -> bool:
        """Whether this screen is the one on show (work for a hidden screen can wait)."""
        frame = getattr(self.window, "frame", None)
        return frame is not None and frame.stack.currentWidget() is self

    def on_theme(self, palette) -> None:
        pass

