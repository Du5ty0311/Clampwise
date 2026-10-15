"""The fixed layout every screen sits in (B0 item 9): navigation on the left,
the screen in the centre under a breadcrumb, a details panel on the right that
collapses, a status bar along the bottom. Panel sizes and collapsed states
are remembered between sessions.

The left sidebar follows the lab's workflow, top to bottom (B0 item 1; the
destinations):
Files, Check, Analyze, Results, Figures, Export; Help and Settings below a line.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtWidgets

from . import theme, widgets
from .widgets import box, label

S = theme.SPACE


@dataclass(frozen=True)
class Destination:
    key: str
    title: str
    shortcut: str
    tip: str


DESTINATIONS = (
    Destination("files", "Files", "Ctrl+1", "Open recordings and assign animal, slice, cell and condition"),
    Destination("check", "Check", "Ctrl+2", "Inspect traces, stimulus timing, recording type and quality"),
    Destination("analyze", "Analyze", "Ctrl+3", "Select analyses, adjust settings and run them"),
    Destination("results", "Results", "Ctrl+4", "All measurements, with their definitions and methods"),
    Destination("figures", "Figures", "Ctrl+5", "Result plots and publication-ready figures"),
    Destination("export", "Export", "Ctrl+6", "Export to Excel, CSV and figure files, with definitions"),
    Destination("live", "Live", "Ctrl+7", "The cell you are recording and the rig's programs, live in one place"),
)
APART = (
    Destination("help", "Help", "F1", "User guide, quick start and measurement definitions"),
    Destination("settings", "Settings", "Ctrl+,", "Appearance, rig mode, folders and saved results"),
)


class Sidebar(QtWidgets.QWidget):
    chosen = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("sidebar")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setFixedWidth(theme.SIDEBAR_WIDTH)
        lay = box("v", "xs", "s")
        name = label("Clampwise", "title")
        lay.addWidget(name)
        self.project = label("No project open", "muted", wrap=True)
        lay.addWidget(self.project)
        lay.addSpacing(S["m"])
        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for d in DESTINATIONS:
            lay.addWidget(self._nav(d, group))
        lay.addStretch(1)
        lay.addWidget(widgets.hairline())
        for d in APART:
            lay.addWidget(self._nav(d, group))
        self.setLayout(lay)

    def _nav(self, d: Destination, group) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(d.title)
        b.setObjectName("nav")
        b.setCheckable(True)
        b.setToolTip(f"{d.tip} ({d.shortcut})")
        b.setAccessibleName(d.title)
        b.clicked.connect(lambda _=False, k=d.key: self.chosen.emit(k))
        group.addButton(b)
        self.buttons[d.key] = b
        return b

    def select(self, key: str) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)


class Breadcrumb(QtWidgets.QLabel):
    """You are here (B0 item 2): Experiment > Animal > Slice > Cell > Recording,
    each a link; clicking one shows that level."""

    level_chosen = QtCore.Signal(str, str)        # level, value

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("breadcrumb")
        self.setTextFormat(QtCore.Qt.RichText)
        self.linkActivated.connect(self._clicked)
        self.set_path([])

    palette: theme.Palette = theme.LIGHT

    def set_path(self, parts: list[tuple[str, str]]) -> None:
        """``parts``: (level, value) from the experiment down. A level whose
        value repeats the one above it (an experiment opened at its animal
        folder) is shown once."""
        self._parts = parts
        if not parts:
            self.setText("No recording selected")
            return
        shown: list[tuple[str, str]] = []
        for lvl, val in parts:
            if shown and shown[-1][1] == val:
                continue
            shown.append((lvl, val))
        # Qt's rich text ignores the stylesheet for links: the colour comes from the theme here
        link = f"color:{self.palette.text_muted}; text-decoration:none"
        links = [f'<a href="{lvl}:{val}" style="{link}">{val}</a>' for lvl, val in shown[:-1]]
        last = f'<span style="color:{self.palette.text}"><b>{shown[-1][1]}</b></span>'
        sep = f'<span style="color:{self.palette.text_muted}"> &nbsp;&rsaquo;&nbsp; </span>'
        self.setText(sep.join(links + [last]))
        self.setAccessibleName("You are here: " + " > ".join(v for _, v in shown))

    def set_palette(self, palette: theme.Palette) -> None:
        self.palette = palette
        self.set_path(getattr(self, "_parts", []))

    def _clicked(self, href: str) -> None:
        level, _, value = href.partition(":")
        self.level_chosen.emit(level, value)


class StatusBar(QtWidgets.QStatusBar):
    """File, sample rate, sweep, cursor time and value, and background jobs
    with their Cancel (B0 items 4, 9)."""

    cancel_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setSizeGripEnabled(False)
        self.file = label("", "muted")
        self.cursor = label("", "muted")
        self.cursor.setFont(theme.numeric_font())
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        self.cancel = widgets.button("Cancel", self.cancel_requested.emit, "Stop starting new files; what is "
                                     "running finishes and is kept")
        self.cancel.setVisible(False)
        self.addWidget(self.file, 1)
        self.addPermanentWidget(self.cursor)
        self.addPermanentWidget(self.progress)
        self.addPermanentWidget(self.cancel)

    def set_recording(self, text: str) -> None:
        self.file.setText(text)

    def set_cursor(self, text: str) -> None:
        self.cursor.setText(text)

    def job(self, done: int, total: int, what: str = "") -> None:
        """Show progress for anything over a second (B0 item 4); hide at the end."""
        busy = total > 0 and done < total
        self.progress.setVisible(busy)
        self.cancel.setVisible(busy)
        if busy:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            self.progress.setFormat(f"{what} %v of %m" if what else "%v of %m")


class Frame(QtWidgets.QWidget):
    """Breadcrumb and screen in the centre, details on the right, all in a
    splitter whose sizes are remembered."""

    def __init__(self) -> None:
        super().__init__()
        lay = box("h", "xs")
        self.sidebar = Sidebar()
        lay.addWidget(self.sidebar)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        centre = QtWidgets.QWidget()
        c = box("v", "s", "m")
        top = box("h", "s")
        self.breadcrumb = Breadcrumb()
        top.addWidget(self.breadcrumb, 1)
        self.details_toggle = QtWidgets.QToolButton()
        self.details_toggle.setText("Details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(True)
        self.details_toggle.setToolTip("Show or hide the details panel (Ctrl+D)")
        self.details_toggle.toggled.connect(self._toggle_details)
        top.addWidget(self.details_toggle)
        c.addLayout(top)
        self.stack = QtWidgets.QStackedWidget()
        c.addWidget(self.stack, 1)
        centre.setLayout(c)
        self.details = QtWidgets.QStackedWidget()
        self.details.setObjectName("details")
        self.details.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.details.setMinimumWidth(240)
        self.splitter.addWidget(centre)
        self.splitter.addWidget(self.details)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([900, theme.DETAILS_WIDTH])
        lay.addWidget(self.splitter, 1)
        self.setLayout(lay)

    def _toggle_details(self, on: bool) -> None:
        self.details.setVisible(on)
