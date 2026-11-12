"""Every colour, font, size and spacing of the interface, in one place (B0).

No widget sets its own colour, font or margin: it takes a token from here,
and the stylesheet built from the tokens styles the rest. A test greps
``app/`` for anything set inline.

The look (Daniel's choice, 2026-11-12: cleaner and modern; same layout):
- a neutral light theme (white panels on a cool light-grey page) and a
  neutral dark theme (near-black page, charcoal panels); panels separated by
  a hairline and the step between page and panel, no shadows (costly in Qt);
- one accent, a green, used only for the primary action, the selection and
  focus; status colours only for status, always with a word;
- one spacing scale (4, 8, 16, 24, 32), three type sizes and two weights;
  6 px corners on controls, 8 px on panels; no gradients;
- drop-down lists, spin boxes and check boxes drawn with their own small
  arrow and check images (made when the theme is applied), so a control
  says what it is;
- data series in the Okabe-Ito colour-blind-safe set, the same colour for the
  same group on every plot, table and figure.

Every text/background pair is checked against WCAG 2.2 AA (4.5:1 for text,
3:1 for large text and plot lines) by ``contrast`` and its test.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---- spacing and type (B0 items 11-12)
SPACE = {"xs": 4, "s": 8, "m": 16, "l": 24, "xl": 32}
RADIUS = 6
RADIUS_PANEL = 8
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"
FONT_SIZE = {"body": 10, "heading": 12, "title": 16}        # points
FONT_WEIGHT = {"regular": 400, "medium": 600}
SIDEBAR_WIDTH = 200
SIDEBAR_COLLAPSED = 48
DETAILS_WIDTH = 320
READING_WIDTH = 760                                           # px: a comfortable width for text
CONTROL_HEIGHT = 24                                           # px: buttons, fields and lists line up


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
    accent_hover: str = ""       # the primary action under the mouse (the accent itself when not given)
    border_strong: str = ""      # a control's edge under the mouse (the hairline when not given)

    def pairs(self) -> list[tuple[str, str, str, float]]:
        """(name, foreground, background, minimum ratio) the theme must satisfy."""
        return [
            ("text on window", self.text, self.window, 4.5),
            ("text on surface", self.text, self.surface, 4.5),
            ("muted text on window", self.text_muted, self.window, 4.5),
            ("muted text on surface", self.text_muted, self.surface, 4.5),
            ("muted text on stripes", self.text_muted, self.surface_alt, 4.5),
            ("text on a selected row", self.text, self.accent_soft, 4.5),
            ("accent text on accent", self.accent_text, self.accent, 4.5),
            ("accent text on accent under the mouse", self.accent_text, self.accent_hover or self.accent, 4.5),
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
    window="#f3f4f6", surface="#ffffff", surface_alt="#f1f3f5", border="#dfe3e8",
    text="#1b1f24", text_muted="#59616b",
    accent="#1d7a55", accent_text="#ffffff", accent_soft="#e2f1e9",
    ok="#0b6e7f", warn="#985600", bad="#b42318",
    plot_background="#ffffff", plot_text="#1b1f24", plot_grid="#edf0f3", plot_axis="#8b939d",
    sweep="#8a929c", average="#1d7a55",
    regions=("#1d7a55", "#0b6e7f", "#2563a8", "#6d4fa0", "#985600"),
    accent_hover="#166145", border_strong="#b8bfc8",
)

DARK = Palette(
    name="dark",
    window="#131518", surface="#1b1e22", surface_alt="#24282d", border="#30353c",
    text="#e7e9ec", text_muted="#a2aab4",
    accent="#45c08a", accent_text="#0a1a12", accent_soft="#1a3328",
    ok="#5ccfc4", warn="#f0b44c", bad="#f2877e",
    plot_background="#16181b", plot_text="#e7e9ec", plot_grid="#262a2f", plot_axis="#6d7580",
    sweep="#7b848f", average="#45c08a",
    regions=("#45c08a", "#5ccfc4", "#6aaee8", "#b99ae6", "#f0b44c"),
    accent_hover="#5fd09c", border_strong="#4a525c",
)

# Data series and groups: Okabe & Ito's colour-blind-safe set (2008), the same
# hue for the same group everywhere; yellow and black left out (one vanishes
# on a light plot, the other on a dark one). Each hue keeps its place in both
# themes; only its lightness is moved where it would fall below WCAG's 3:1
# for plot lines on the light plot (orange, reddish purple and sky blue), and
# on the dark plot lifted to 4.5:1 so no series sinks into the background (the
# blue, the brown and the grey), by the smallest step that reaches it.
SERIES = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#8c564b", "#7f7f7f")
SERIES_LIGHT = ("#0072b2", "#d55e00", "#009e73", "#cb77a5", "#c18600", "#1d99df", "#8c564b", "#7f7f7f")
SERIES_DARK = ("#0086d1", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#ad7367", "#828282")


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


# ---- the small images the stylesheet needs (arrows, a check mark)

ICON_SCALE = 4                   # drawn this many times larger than shown, so they stay sharp on any screen


def icons(p: Palette, folder: Path | None = None) -> dict[str, str]:
    """The arrow and check-mark images for ``p``, drawn with Qt (no image files are shipped) and written as PNG
    to ``folder`` (a folder of the temporary directory by default): {name: its path, with forward slashes, for a
    stylesheet's url()}. Needs a QGuiApplication."""
    from pyqtgraph.Qt import QtCore, QtGui

    folder = Path(folder) if folder is not None else Path(tempfile.gettempdir()) / "clampwise-ui"
    folder.mkdir(parents=True, exist_ok=True)
    k = ICON_SCALE
    out = {}

    def draw(name: str, w: int, h: int, colour: str, points, width: float, close: bool = False) -> None:
        img = QtGui.QImage(w * k, h * k, QtGui.QImage.Format_ARGB32)
        img.fill(QtCore.Qt.transparent)
        qp = QtGui.QPainter(img)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(colour), width * k, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                         QtCore.Qt.RoundJoin)
        qp.setPen(pen)
        path = QtGui.QPainterPath(QtCore.QPointF(points[0][0] * k, points[0][1] * k))
        for x, y in points[1:]:
            path.lineTo(x * k, y * k)
        if close:
            path.closeSubpath()
        qp.drawPath(path)
        qp.end()
        f = folder / f"{name}-{p.name}.png"
        img.save(str(f), "PNG")
        out[name] = f.as_posix()

    draw("down", 10, 6, p.text_muted, [(1.5, 1.2), (5, 4.6), (8.5, 1.2)], 1.5)
    draw("down-disabled", 10, 6, p.border_strong or p.border, [(1.5, 1.2), (5, 4.6), (8.5, 1.2)], 1.5)
    draw("up", 10, 6, p.text_muted, [(1.5, 4.8), (5, 1.4), (8.5, 4.8)], 1.5)
    draw("up-disabled", 10, 6, p.border_strong or p.border, [(1.5, 4.8), (5, 1.4), (8.5, 4.8)], 1.5)
    draw("check", 12, 12, p.accent_text, [(2.6, 6.3), (5.0, 8.7), (9.6, 3.4)], 1.8)
    return out


# ---- the stylesheet, built from the tokens

def stylesheet(p: Palette, images: dict[str, str] | None = None) -> str:
    """The application's stylesheet for ``p``. ``images`` (from ``icons``) adds the drop-down and spin-box
    arrows and the check mark; without them those controls keep Qt's own."""
    s, r, rp = SPACE, RADIUS, RADIUS_PANEL
    body, heading, title = FONT_SIZE["body"], FONT_SIZE["heading"], FONT_SIZE["title"]
    hover = p.border_strong or p.border
    accent_hover = p.accent_hover or p.accent
    h = CONTROL_HEIGHT
    css = f"""
QWidget {{ background: {p.window}; color: {p.text}; font-family: "{FONT_FAMILY}"; font-size: {body}pt; }}
QMainWindow, QDialog {{ background: {p.window}; }}
QFrame#surface, QScrollArea#surface, QWidget#surface {{ background: {p.surface}; border: 1px solid {p.border};
    border-radius: {rp}px; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: {title}pt; font-weight: {FONT_WEIGHT['medium']}; }}
QLabel#heading {{ font-size: {heading}pt; font-weight: {FONT_WEIGHT['medium']}; }}
QLabel#muted, QLabel#hint {{ color: {p.text_muted}; }}
QLabel#breadcrumb {{ color: {p.text_muted}; }}
QLabel#breadcrumb a {{ color: {p.text_muted}; text-decoration: none; }}
QLabel#breadcrumb a:hover {{ color: {p.accent}; }}

QPushButton {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border}; border-radius: {r}px;
               padding: {s['xs'] + 1}px {s['m'] - 2}px; min-height: {h - 2 * (s['xs'] + 1) - 2}px; }}
QPushButton:hover {{ background: {p.surface_alt}; border-color: {hover}; }}
QPushButton:pressed {{ background: {p.accent_soft}; }}
QPushButton:disabled {{ color: {p.text_muted}; background: {p.surface_alt}; border-color: {p.border}; }}
QPushButton:focus {{ border-color: {p.accent}; }}
QPushButton#primary {{ background: {p.accent}; color: {p.accent_text}; border: 1px solid {p.accent};
                       font-weight: {FONT_WEIGHT['medium']}; }}
QPushButton#primary:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton#primary:focus {{ border-color: {p.text}; }}
QPushButton#primary:disabled {{ background: {p.border}; color: {p.text_muted}; border-color: {p.border}; }}
QPushButton#quiet {{ background: transparent; border: 1px solid transparent; }}
QPushButton#quiet:hover {{ background: {p.surface_alt}; border-color: {p.border}; }}
QPushButton#quiet:focus {{ border-color: {p.accent}; }}
QPushButton::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: center right; width: 8px;
    height: 5px; right: {s['xs']}px; }}
QPushButton:checked, QPushButton#quiet:checked {{ background: {p.accent_soft}; border: 1px solid {p.accent};
                                                  font-weight: {FONT_WEIGHT['medium']}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: {r}px;
               padding: {s['xs']}px {s['xs'] + 2}px; }}
QToolButton:hover {{ background: {p.surface_alt}; border-color: {p.border}; }}
QToolButton:pressed, QToolButton:checked {{ background: {p.accent_soft}; border-color: {p.accent}; }}
QToolButton:disabled {{ color: {p.text_muted}; }}
QToolButton::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: center right; width: 8px;
    height: 5px; right: 2px; }}
QToolButton[popupMode="2"] {{ padding-right: {s['m']}px; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit, QTextBrowser {{
    background: {p.surface}; color: {p.text}; border: 1px solid {p.border}; border-radius: {r}px;
    padding: {s['xs']}px {s['s']}px; selection-background-color: {p.accent}; selection-color: {p.accent_text}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: {h - 2 * s['xs'] - 2}px; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {hover}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {p.accent}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {p.surface_alt}; color: {p.text_muted}; }}
QComboBox {{ padding-right: {s['l']}px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: {s['l'] - 2}px;
    border: none; background: transparent; }}
QAbstractSpinBox {{ padding-right: {s['l'] - 4}px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: {s['m'] + 2}px;
    border: none; border-left: 1px solid {p.border}; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: {r}px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: {r}px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {p.surface_alt}; }}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{ background: {p.accent_soft}; }}
QWidget#dense QPushButton {{ padding: {s['xs'] // 2}px {s['s']}px; min-height: 18px; }}
QWidget#dense QLineEdit, QWidget#dense QSpinBox, QWidget#dense QDoubleSpinBox, QWidget#dense QComboBox {{
    padding: {s['xs'] // 2}px {s['xs']}px; min-height: 18px; }}
QWidget#dense QComboBox {{ padding-right: {s['l']}px; }}
QWidget#dense QLabel#section {{ padding-top: {s['xs']}px; }}
QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border}; outline: 0;
    selection-background-color: {p.accent_soft}; selection-color: {p.text}; padding: {s['xs'] // 2}px; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: {s['s']}px; }}
QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator, QTableView::indicator, QTreeView::indicator,
QListView::indicator {{ width: 14px; height: 14px; border: 1px solid {p.text_muted}; background: {p.surface}; }}
QCheckBox::indicator, QGroupBox::indicator, QTableView::indicator, QTreeView::indicator, QListView::indicator {{
    border-radius: 4px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {p.accent}; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTableView::indicator:checked,
QTreeView::indicator:checked, QListView::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}
/* a chosen option's dot: a radial fill with a hard edge, not a shaded gradient */
QRadioButton::indicator:checked {{ border-color: {p.accent}; background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5,
    fx: 0.5, fy: 0.5, stop: 0 {p.accent}, stop: 0.5 {p.accent}, stop: 0.56 {p.surface}, stop: 1 {p.surface}); }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {p.border}; background: {p.surface_alt}; }}

QTableView, QTreeView, QListView {{ background: {p.surface}; alternate-background-color: {p.surface_alt};
    border: 1px solid {p.border}; border-radius: {r}px; gridline-color: {p.border}; outline: 0;
    selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
QTableView::item, QTreeView::item, QListView::item {{ padding: {s['xs']}px {s['s']}px; }}
QTableView::item:hover, QListView::item:hover, QTreeView::item:hover {{ background: {p.surface_alt}; }}
QTableView::item:selected, QListView::item:selected, QTreeView::item:selected {{ background: {p.accent_soft};
    color: {p.text}; }}
QHeaderView {{ background: {p.surface}; border: none; }}
QHeaderView::section {{ background: {p.surface}; color: {p.text_muted}; border: none;
    border-bottom: 1px solid {p.border}; padding: {s['s'] - 2}px {s['s']}px; font-weight: {FONT_WEIGHT['medium']}; }}
QHeaderView::section:hover {{ color: {p.text}; }}
QTableCornerButton::section {{ background: {p.surface}; border: none; border-bottom: 1px solid {p.border}; }}

QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: {rp}px; background: {p.surface}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {p.text_muted}; padding: {s['s']}px {s['m'] - 2}px; border: none;
    border-bottom: 2px solid transparent; margin-right: {s['xs']}px; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; border-bottom: 2px solid {p.border}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {hover}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 3px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {hover}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {p.accent}; }}
QStatusBar {{ background: {p.surface}; color: {p.text_muted}; border-top: 1px solid {p.border}; }}
QStatusBar::item {{ border: none; }}
QMenuBar {{ background: {p.window}; }}
QMenuBar::item:selected {{ background: {p.surface_alt}; border-radius: {r}px; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {rp}px; padding: {s['xs']}px; }}
QMenu::item {{ padding: {s['xs'] + 1}px {s['l']}px {s['xs'] + 1}px {s['m']}px; border-radius: {r}px; }}
QMenu::item:selected {{ background: {p.accent_soft}; color: {p.text}; }}
QMenu::item:disabled {{ color: {p.text_muted}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: {s['xs']}px {s['s']}px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: {s['xs']}px; }}
QToolTip {{ background: {p.surface}; color: {p.text}; border: 1px solid {hover}; border-radius: {r}px;
    padding: {s['xs']}px {s['s']}px; }}
QProgressBar {{ background: {p.surface_alt}; border: 1px solid {p.border}; border-radius: {r}px; text-align: center;
    color: {p.text}; min-height: 14px; }}
QProgressBar::chunk {{ background: {p.accent}; border-radius: {r - 1}px; }}
QGroupBox {{ border: 1px solid {p.border}; border-radius: {rp}px; margin-top: {s['m']}px; padding-top: {s['s']}px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: {s['s']}px; color: {p.text_muted}; }}

QWidget#sidebar {{ background: {p.surface}; border-right: 1px solid {p.border}; }}
QPushButton#nav {{ background: transparent; border: none; border-left: 3px solid transparent; border-radius: {r}px;
    text-align: left; padding: {s['s']}px {s['m'] - 3}px; color: {p.text_muted}; font-size: {heading}pt; }}
QPushButton#nav:hover {{ background: {p.surface_alt}; color: {p.text}; }}
QPushButton#nav:checked {{ background: {p.accent_soft}; color: {p.text}; font-weight: {FONT_WEIGHT['medium']};
    border-left: 3px solid {p.accent}; }}
QWidget#details {{ background: {p.surface}; border-left: 1px solid {p.border}; }}
QLabel#badge_ok {{ color: {p.ok}; background: transparent; }}
QLabel#badge_warn {{ color: {p.warn}; background: transparent; }}
QLabel#badge_bad {{ color: {p.bad}; background: transparent; }}
QLabel#badge_muted {{ color: {p.text_muted}; background: transparent; }}
QFrame#hairline {{ background: {p.border}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#vline {{ background: {p.border}; max-width: 1px; min-width: 1px; border: none; margin: {s['xs']}px {s['xs']}px; }}
QWidget#toolbar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
QLabel#section {{ color: {p.text_muted}; font-weight: {FONT_WEIGHT['medium']}; padding-top: {s['s']}px; }}
QScrollArea#desk, QWidget#desk {{ background: {p.surface_alt}; border: 1px solid {p.border}; border-radius: {rp}px; }}
QWidget#bare {{ background: {p.surface_alt}; border: none; border-radius: 0px; }}
QMdiArea {{ background: {p.window}; }}
QLabel#meter {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r}px; padding: {s['xs']}px {s['s']}px;
    font-family: '{FONT_MONO}'; font-size: {heading}pt; }}
QFrame#empty {{ background: transparent; border: 1px dashed {hover}; border-radius: {RADIUS_PANEL * 2}px; }}
"""
    if images:
        i = images
        css += f"""
QComboBox::down-arrow {{ image: url("{i['down']}"); width: 10px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url("{i['down-disabled']}"); }}
QComboBox::down-arrow:on {{ image: url("{i['up']}"); }}
QAbstractSpinBox::up-arrow {{ image: url("{i['up']}"); width: 8px; height: 5px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url("{i['up-disabled']}"); }}
QAbstractSpinBox::down-arrow {{ image: url("{i['down']}"); width: 8px; height: 5px; }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url("{i['down-disabled']}"); }}
QToolButton::menu-indicator, QPushButton::menu-indicator {{ image: url("{i['down']}"); }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTableView::indicator:checked,
QTreeView::indicator:checked, QListView::indicator:checked {{ image: url("{i['check']}"); }}
"""
    return css


def apply(app, palette: Palette) -> None:
    """Style the application: the stylesheet (with its arrow and check images) and a matching Qt palette (for
    the few things stylesheets do not reach), and pyqtgraph's defaults."""
    from pyqtgraph.Qt import QtGui

    try:
        images = icons(palette)
    except Exception:                        # no writable temporary folder: Qt's own arrows, the rest styled
        images = None
    app.setStyleSheet(stylesheet(palette, images))
    qp = QtGui.QPalette()
    for role, colour in ((QtGui.QPalette.Window, palette.window), (QtGui.QPalette.Base, palette.surface),
                         (QtGui.QPalette.AlternateBase, palette.surface_alt), (QtGui.QPalette.Text, palette.text),
                         (QtGui.QPalette.WindowText, palette.text), (QtGui.QPalette.ButtonText, palette.text),
                         (QtGui.QPalette.Button, palette.surface), (QtGui.QPalette.Highlight, palette.accent),
                         (QtGui.QPalette.HighlightedText, palette.accent_text), (QtGui.QPalette.ToolTipBase, palette.surface),
                         (QtGui.QPalette.ToolTipText, palette.text), (QtGui.QPalette.PlaceholderText, palette.text_muted),
                         (QtGui.QPalette.Mid, palette.border), (QtGui.QPalette.Light, palette.surface),
                         (QtGui.QPalette.Dark, palette.border_strong or palette.border)):
        qp.setColor(role, QtGui.QColor(colour))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.WindowText, QtGui.QPalette.ButtonText):
        qp.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(palette.text_muted))
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
    item.showGrid(x=False, y=True, alpha=0.12)
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
