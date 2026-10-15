"""B0: the theme's colour pairs pass WCAG 2.2 AA; data colours stay apart for
colour-blind viewers; nothing in app/ sets a colour, font or margin outside
theme.py and the stylesheet."""

from __future__ import annotations

import re
from pathlib import Path

import clampwise
from clampwise.app import theme

APP = Path(clampwise.__file__).resolve().parent / "app"


def test_every_text_pair_passes_wcag_aa_in_both_themes():
    for palette in (theme.LIGHT, theme.DARK):
        bad = theme.failing_pairs(palette)
        assert not bad, f"{palette.name}: " + "; ".join(bad)


def test_contrast_formula_matches_the_known_values():
    assert abs(theme.contrast("#000000", "#ffffff") - 21.0) < 0.01
    assert abs(theme.contrast("#777777", "#ffffff") - 4.48) < 0.02


def test_series_colours_are_the_okabe_ito_hues_legible_on_each_plot():
    for palette in (theme.LIGHT, theme.DARK):
        colours = theme.series(palette)
        assert len(set(colours)) == len(colours) == 8
        for c in colours:                         # plot lines need 3:1 (WCAG 2.2, non-text contrast)
            assert theme.contrast(c, palette.plot_background) >= 3.0, (palette.name, c)
    # the same hue in the same place in both themes
    import colorsys

    def hue(c):
        return colorsys.rgb_to_hls(*(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))[0]

    for a, b in zip(theme.SERIES_LIGHT, theme.SERIES_DARK):
        assert abs(hue(a) - hue(b)) < 0.02, (a, b)


INLINE = re.compile(r"""(setStyleSheet\(|QColor\(\s*["']#|["']#[0-9a-fA-F]{6}["']|setContentsMargins\(\s*\d|setSpacing\(\s*\d|QFont\(\s*["'])""")


def test_no_widget_styles_itself_outside_the_theme():
    found = []
    for path in APP.rglob("*.py"):
        if path.name == "theme.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if INLINE.search(line) and "theme." not in line and not line.strip().startswith("#"):
                found.append(f"{path.relative_to(APP)}:{i}: {line.strip()[:80]}")
    assert not found, "\n".join(found)


def test_stylesheet_builds_from_the_tokens():
    css = theme.stylesheet(theme.LIGHT)
    assert theme.LIGHT.accent in css and f'"{theme.FONT_FAMILY}"' in css
    assert "QPushButton#primary" in css and "QPushButton#nav" in css


def test_a_switch_that_is_off_is_drawn_in_both_themes():
    """An unchecked check box or radio button drew nothing but its text (a
    feature turned off read as a label): its outline shows, and on differs from off."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for palette in (theme.LIGHT, theme.DARK):
        theme.apply(app, palette)
        background = QtGui.QColor(palette.window).rgb()
        for cls in (QtWidgets.QCheckBox, QtWidgets.QRadioButton):
            seen = []
            for on in (False, True):
                page = QtWidgets.QWidget()                       # as a screen shows it: on the page's background
                b = cls("", page)
                b.setChecked(on)
                b.move(4, 4)
                page.resize(40, 32)
                page.show()
                app.processEvents()
                image = page.grab().toImage()
                box = range(b.x(), b.x() + 18), range(b.y(), b.y() + b.height())
                seen.append({image.pixel(x, y) for x in box[0] for y in box[1]} - {background})
                page.close()
                assert seen[-1], f"{palette.name} {cls.__name__} {'on' if on else 'off'} draws nothing"
            assert seen[0] != seen[1], f"{palette.name} {cls.__name__}: on looks like off"
