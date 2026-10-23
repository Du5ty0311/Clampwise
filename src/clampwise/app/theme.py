"""Every colour, font, size and spacing of the interface, in one place (B0).

No widget sets its own colour, font or margin: it takes a token from here,
and the stylesheet built from the tokens styles the rest. A test greps
``app/`` for anything set inline.

The look (Part A's model: calm, uncluttered, content in front):
- a warm off-white light theme and a warm dark grey dark theme, never pure
  white or blue-grey; panels separated by space and hairline borders;
- one accent, a deep green, used only for the primary action,
  selection and focus; status colours only for status, always with a word;
- one spacing scale (4, 8, 16, 24, 32), three type sizes and two weights, soft
  6 px corners, no gradients, no shadows (costly in Qt; a hairline does);
- data series in the Okabe-Ito colour-blind-safe set, the same colour for the
  same group on every plot, table and figure.

Every text/background pair is checked against WCAG 2.2 AA (4.5:1 for text,
3:1 for large text and plot lines) by ``contrast`` and its test.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- spacing and type (B0 items 11-12)
SPACE = {"xs": 4, "s": 8, "m": 16, "l": 24, "xl": 32}
RADIUS = 6
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"
FONT_SIZE = {"body": 10, "heading": 12, "title": 15}        # points
FONT_WEIGHT = {"regular": 400, "medium": 600}
SIDEBAR_WIDTH = 200
SIDEBAR_COLLAPSED = 48
DETAILS_WIDTH = 320
READING_WIDTH = 760                                           # px: a comfortable width for text


@dataclass(frozen=True)
class Palette:
    name: str
    window: str          # the page
    surface: str         # panels, inputs, cards
    surface_alt: str     # table stripes, hover
    border: str          # hairlines
    text: str
    text_muted: str
    accent: str          # the primary action, selection, focus
    accent_text: str     # text on the accent
    accent_soft: str     # a tint of the accent for selected rows
    ok: str
    warn: str
    bad: str
    plot_background: str
    plot_text: str
    plot_grid: str
    plot_axis: str
    sweep: str           # individual sweeps, muted
    average: str         # the average trace
    regions: tuple[str, ...] = ()

    def pairs(self) -> list[tuple[str, str, str, float]]:
        """(name, foreground, background, minimum ratio) the theme must satisfy."""
        return [
            ("text on window", self.text, self.window, 4.5),
            ("text on surface", self.text, self.surface, 4.5),
            ("muted text on window", self.text_muted, self.window, 4.5),
            ("muted text on surface", self.text_muted, self.surface, 4.5),
            ("accent text on accent", self.accent_text, self.accent, 4.5),
            ("accent on window (large, lines)", self.accent, self.window, 3.0),
            ("ok on window (large)", self.ok, self.window, 3.0),
            ("warn on window (large)", self.warn, self.window, 3.0),
            ("bad on window (large)", self.bad, self.window, 3.0),
            ("plot text on plot", self.plot_text, self.plot_background, 4.5),
            ("average on plot (lines)", self.average, self.plot_background, 3.0),
            ("sweep on plot (lines)", self.sweep, self.plot_background, 3.0),
        ]


LIGHT = Palette(
    name="light",
    window="#f6f3ee", surface="#fdfbf7", surface_alt="#efebe4", border="#d9d3c9",
    text="#2b2622", text_muted="#5f574f",
    accent="#2d6a4f", accent_text="#ffffff", accent_soft="#d8e9df",
    ok="#16697a", warn="#9a5a00", bad="#a8322a",
    plot_background="#fdfbf7", plot_text="#2b2622", plot_grid="#e6e0d6", plot_axis="#8c8378",
    sweep="#8a8178", average="#2d6a4f",
    regions=("#2d6a4f", "#16697a", "#1f5f8b", "#6b4c9a", "#9a5a00"),
)

DARK = Palette(
    name="dark",
    window="#262320", surface="#2f2b27", surface_alt="#38332e", border="#4a443d",
    text="#ece7e0", text_muted="#b8b0a6",
    accent="#6fbf93", accent_text="#13241b", accent_soft="#29402f",
    ok="#7fd1c7", warn="#f0b85a", bad="#ef8a80",
    plot_background="#221f1c", plot_text="#ece7e0", plot_grid="#3b3631", plot_axis="#8c8378",
    sweep="#9a9186", average="#6fbf93",
    regions=("#6fbf93", "#7fd1c7", "#7fb6e8", "#c3a2ea", "#f0b85a"),
)

# Data series and groups: Okabe & Ito's colour-blind-safe set (2008), the same
# hue for the same group everywhere; yellow and black left out (one vanishes
# on a light plot, the other on a dark one). Each hue keeps its place in both
# themes; only its lightness is nudged where it would fall below WCAG's 3:1
# for plot lines (orange, reddish purple and sky blue on the light plot, brown
# on the dark one), by the smallest step that reaches it.
SERIES = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#8c564b", "#7f7f7f")
SERIES_LIGHT = ("#0072b2", "#d55e00", "#009e73", "#cb77a5", "#c18600", "#1d99df", "#8c564b", "#7f7f7f")
SERIES_DARK = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#945b50", "#7f7f7f")


def series(palette: "Palette") -> tuple[str, ...]:
    return SERIES_DARK if palette.name == "dark" else SERIES_LIGHT

STATUS_WORDS = {"ok": "passed", "warn": "flagged", "bad": "error"}


# ---- contrast (WCAG 2.2)

def _channel(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(fg: str, bg: str) -> float:
    a, b = luminance(fg), luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def failing_pairs(palette: Palette) -> list[str]:
    return [f"{name}: {contrast(fg, bg):.2f} < {need}" for name, fg, bg, need in palette.pairs()
            if contrast(fg, bg) < need]


# ---- the stylesheet, built from the tokens

def stylesheet(p: Palette) -> str:
    s, r = SPACE, RADIUS
    body, heading, title = FONT_SIZE["body"], FONT_SIZE["heading"], FONT_SIZE["title"]
    return f"""
QWidget {{ background: {p.window}; color: {p.text}; font-family: "{FONT_FAMILY}"; font-size: {body}pt; }}
QMainWindow, QDialog {{ background: {p.window}; }}
QFrame#surface, QScrollArea#surface, QWidget#surface {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r}px; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: {title}pt; font-weight: {FONT_WEIGHT['medium']}; }}
QLabel#heading {{ font-size: {heading}pt; font-weight: {FONT_WEIGHT['medium']}; }}
QLabel#muted, QLabel#hint {{ color: {p.text_muted}; }}
QLabel#breadcrumb {{ color: {p.text_muted}; }}
QLabel#breadcrumb a {{ color: {p.text_muted}; text-decoration: none; }}
QLabel#breadcrumb a:hover {{ color: {p.accent}; }}

QPushButton {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border}; border-radius: {r}px;
               padding: {s['xs']}px {s['m']}px; min-height: 22px; }}
QPushButton:hover {{ background: {p.surface_alt}; }}
QPushButton:pressed {{ background: {p.accent_soft}; }}
QPushButton:disabled {{ color: {p.text_muted}; border-color: {p.border}; }}
QPushButton:focus {{ border: 2px solid {p.accent}; }}
QPushButton#primary {{ background: {p.accent}; color: {p.accent_text}; border: 1px solid {p.accent};
                       font-weight: {FONT_WEIGHT['medium']}; }}
QPushButton#primary:hover {{ background: {p.accent}; border-color: {p.text}; }}
QPushButton#primary:disabled {{ background: {p.border}; color: {p.text_muted}; border-color: {p.border}; }}
QPushButton#quiet {{ background: transparent; border: 1px solid transparent; }}
QPushButton#quiet:hover {{ background: {p.surface_alt}; }}
QPushButton:checked, QPushButton#quiet:checked {{ background: {p.accent_soft}; border: 1px solid {p.accent};
                                                  font-weight: {FONT_WEIGHT['medium']}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: {r}px; padding: {s['xs']}px; }}
QToolButton:hover {{ background: {p.surface_alt}; }}
QToolButton:checked {{ background: {p.accent_soft}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background: {p.surface}; color: {p.text}; border: 1px solid {p.border}; border-radius: {r}px;
    padding: {s['xs']}px {s['s']}px; selection-background-color: {p.accent}; selection-color: {p.accent_text}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {p.accent}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QWidget#dense QPushButton {{ padding: {s['xs'] // 2}px {s['s']}px; min-height: 18px; }}
QWidget#dense QLineEdit, QWidget#dense QSpinBox, QWidget#dense QDoubleSpinBox, QWidget#dense QComboBox {{
    padding: {s['xs'] // 2}px {s['xs']}px; }}
QWidget#dense QLabel#section {{ padding-top: {s['xs']}px; }}
QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border};
    selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: {s['s']}px; }}
QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator, QTableView::indicator, QTreeView::indicator,
QListView::indicator {{ width: 14px; height: 14px; border: 1px solid {p.text_muted}; background: {p.surface}; }}
QCheckBox::indicator, QGroupBox::indicator, QTableView::indicator, QTreeView::indicator, QListView::indicator {{
    border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTableView::indicator:checked,
QTreeView::indicator:checked, QListView::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}
/* a chosen option's dot: a radial fill with a hard edge, not a shaded gradient */
QRadioButton::indicator:checked {{ border-color: {p.accent}; background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5,
    fx: 0.5, fy: 0.5, stop: 0 {p.accent}, stop: 0.55 {p.accent}, stop: 0.6 {p.surface}, stop: 1 {p.surface}); }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {p.border}; }}

QTableView, QTreeView, QListView {{ background: {p.surface}; alternate-background-color: {p.surface_alt};
    border: 1px solid {p.border}; border-radius: {r}px; gridline-color: {p.border};
    selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
QTableView::item, QTreeView::item, QListView::item {{ padding: {s['xs']}px {s['s']}px; }}
QHeaderView::section {{ background: {p.window}; color: {p.text_muted}; border: none; border-bottom: 1px solid {p.border};
    padding: {s['xs']}px {s['s']}px; font-weight: {FONT_WEIGHT['medium']}; }}
QTableCornerButton::section {{ background: {p.window}; border: none; }}

QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: {r}px; background: {p.surface}; }}
QTabBar::tab {{ background: transparent; color: {p.text_muted}; padding: {s['s']}px {s['m']}px; border: none;
    border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
QTabBar::tab:hover {{ color: {p.text}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {p.text_muted}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QStatusBar {{ background: {p.window}; color: {p.text_muted}; border-top: 1px solid {p.border}; }}
QStatusBar::item {{ border: none; }}
QMenuBar {{ background: {p.window}; }}
QMenuBar::item:selected {{ background: {p.surface_alt}; border-radius: {r}px; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r}px; padding: {s['xs']}px; }}
QMenu::item {{ padding: {s['xs']}px {s['l']}px; border-radius: {r}px; }}
QMenu::item:selected {{ background: {p.accent_soft}; }}
QToolTip {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border}; padding: {s['xs']}px {s['s']}px; }}
QProgressBar {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r}px; text-align: center;
    color: {p.text}; }}
QProgressBar::chunk {{ background: {p.accent}; border-radius: {r}px; }}
QGroupBox {{ border: 1px solid {p.border}; border-radius: {r}px; margin-top: {s['m']}px; padding-top: {s['s']}px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: {s['s']}px; color: {p.text_muted}; }}

QWidget#sidebar {{ background: {p.surface}; border-right: 1px solid {p.border}; }}
QPushButton#nav {{ background: transparent; border: 1px solid transparent; text-align: left;
    padding: {s['s']}px {s['m']}px; color: {p.text_muted}; font-size: {heading}pt; }}
QPushButton#nav:hover {{ background: {p.surface_alt}; color: {p.text}; }}
QPushButton#nav:checked {{ background: {p.accent_soft}; color: {p.text}; font-weight: {FONT_WEIGHT['medium']}; }}
QPushButton#nav:focus {{ border: 2px solid {p.accent}; }}
QWidget#details {{ background: {p.surface}; border-left: 1px solid {p.border}; }}
QLabel#badge_ok {{ color: {p.ok}; background: transparent; }}
QLabel#badge_warn {{ color: {p.warn}; background: transparent; }}
QLabel#badge_bad {{ color: {p.bad}; background: transparent; }}
QLabel#badge_muted {{ color: {p.text_muted}; background: transparent; }}
QFrame#hairline {{ background: {p.border}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#vline {{ background: {p.border}; max-width: 1px; min-width: 1px; border: none; margin: {s['xs']}px {s['xs']}px; }}
QWidget#toolbar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
QLabel#section {{ color: {p.text_muted}; font-weight: {FONT_WEIGHT['medium']}; padding-top: {s['s']}px; }}
QScrollArea#desk, QWidget#desk {{ background: {p.surface_alt}; border: 1px solid {p.border}; border-radius: {r}px; }}
QWidget#bare {{ background: {p.surface_alt}; border: none; border-radius: 0px; }}
QMdiArea {{ background: {p.window}; }}
QLabel#meter {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r}px; padding: {s['xs']}px {s['s']}px;
    font-family: '{FONT_MONO}'; font-size: {heading}pt; }}
QFrame#empty {{ background: transparent; border: 1px dashed {p.border}; border-radius: {RADIUS * 2}px; }}
"""


def apply(app, palette: Palette) -> None:
    """Style the application: the stylesheet and a matching Qt palette (for
    the few things stylesheets do not reach), and pyqtgraph's defaults."""
    from pyqtgraph.Qt import QtGui

    app.setStyleSheet(stylesheet(palette))
    qp = QtGui.QPalette()
    for role, colour in ((QtGui.QPalette.Window, palette.window), (QtGui.QPalette.Base, palette.surface),
                         (QtGui.QPalette.AlternateBase, palette.surface_alt), (QtGui.QPalette.Text, palette.text),
                         (QtGui.QPalette.WindowText, palette.text), (QtGui.QPalette.ButtonText, palette.text),
                         (QtGui.QPalette.Button, palette.surface), (QtGui.QPalette.Highlight, palette.accent),
                         (QtGui.QPalette.HighlightedText, palette.accent_text), (QtGui.QPalette.ToolTipBase, palette.surface),
                         (QtGui.QPalette.ToolTipText, palette.text), (QtGui.QPalette.PlaceholderText, palette.text_muted)):
        qp.setColor(role, QtGui.QColor(colour))
    app.setPalette(qp)
    font = QtGui.QFont(FONT_FAMILY, FONT_SIZE["body"])
    app.setFont(font)
    try:
        import pyqtgraph as pg

        pg.setConfigOptions(background=palette.plot_background, foreground=palette.plot_text, antialias=True)
    except ImportError:
        pass


def style_plot(plot, palette: Palette) -> None:
    """Make a pyqtgraph PlotItem/PlotWidget look like part of the page: the
    page's background, thin axes in the muted colour, a light grid, the
    interface font on the labels. pyqtgraph's global config only reaches
    plots made after it is set, so this runs on each plot."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    if hasattr(plot, "setBackground"):
        plot.setBackground(palette.plot_background)
    font = QtGui.QFont(FONT_FAMILY, FONT_SIZE["body"])
    for name in ("left", "bottom"):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(palette.plot_axis, width=1))
        axis.setTextPen(pg.mkPen(palette.plot_text))
        axis.setTickFont(font)
        axis.label.setFont(font)
        # the units are in the label ("Time (s)"); pyqtgraph's own prefix would
        # add "(x0.001)" beside them when zoomed in
        axis.enableAutoSIPrefix(False)
        axis.autoSIPrefixScale, axis.labelUnitPrefix = 1.0, ""     # a scale left from before would stay
        axis.picture = None
        axis.update()
    for name in ("top", "right"):
        item.showAxis(name, False)
    item.showGrid(x=False, y=True, alpha=0.15)
    item.getViewBox().setBorder(None)


def keep_in_frame(plot, x0: float | None = None, x1: float | None = None) -> None:
    """Keep what is drawn in frame: x never goes past
    [x0, x1] (a trace's 0 to its end; by default the drawn data's span with a
    3% margin, so the end points are whole), so
    zooming out stops at the whole trace and panning stops at its ends; y is
    free, and refits to what is in view whenever it is left to fit itself."""
    import numpy as np

    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    vb = item.getViewBox()
    if x0 is None or x1 is None:
        bounds = vb.childrenBounds()
        x = bounds[0] if bounds is not None else None
        x0, x1 = (x[0], x[1]) if x is not None else (None, None)
        if x0 is not None and x1 is not None and x1 > x0:
            x0, x1 = x0 - 0.03 * (x1 - x0), x1 + 0.03 * (x1 - x0)
    if x0 is not None and x1 is not None and np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
        vb.setLimits(xMin=float(x0), xMax=float(x1), maxXRange=float(x1 - x0))
    else:
        vb.setLimits(xMin=None, xMax=None, maxXRange=None)
    vb.setAutoVisible(y=True)
    vb.enableAutoRange()


def numeric_font():
    from pyqtgraph.Qt import QtGui

    font = QtGui.QFont(FONT_MONO, FONT_SIZE["body"])
    font.setStyleHint(QtGui.QFont.Monospace)
    return font


def system_prefers_dark(app) -> bool:
    """Whether Windows is in dark mode (Qt 5: read from the application palette)."""
    try:
        from pyqtgraph.Qt import QtGui

        c = app.palette().color(QtGui.QPalette.Window)
        return c.lightness() < 128
    except Exception:
        return False
