"""The inspector beside the figure: the properties of whatever is selected on the canvas, and nothing else.

Select a panel, an axis, a group's curve, the legend, the scale bars, a stimulus bar, a text, a bracket, a
letter: its few properties show here, each named for what it does, each change drawn at once and undoable. With
nothing selected it shows the figure's own: the journal and size, the arrangement, the text sizes, the colours.
Every property lives in the figure's ``Spec``; this file only reads and writes it through the builder.
"""

from __future__ import annotations

from typing import Callable

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..figure import elements, journals, page
from ..figure.recipe import LINES, MARKERS, PALETTES, PLOTS, SIZES, THEMES
from . import widgets
from .widgets import box, label

LEGEND_PLACES = {"best": "Where it fits best", "upper left": "Upper left", "upper right": "Upper right",
                 "lower left": "Lower left", "lower right": "Lower right", "outside": "Outside, at the right",
                 "none": "No legend"}
WIDTHS = {"single": "One column", "one and a half": "One and a half columns", "double": "Two columns"}


def _num(text: str):
    try:
        return float(str(text).strip().replace(",", "."))
    except ValueError:
        return None


class Inspector(QtWidgets.QWidget):
    def __init__(self, builder) -> None:
        super().__init__()
        self.b = builder
        self.tag = None
        self._filling = False
        outer = box("v", "s")
        outer.setContentsMargins(0, 0, 0, 0)
        self.heading = label("Figure", "heading")
        outer.addWidget(self.heading)
        self.hint = label("", "hint", wrap=True)
        outer.addWidget(self.hint)
        self.form_holder = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.form_holder.setLayout(self.form)
        outer.addWidget(self.form_holder)
        self.buttons = box("v", "xs")
        outer.addLayout(self.buttons)
        outer.addStretch(1)
        self.setLayout(outer)
        self.show_for(None)

    # ---- the form's rows
    def _clear(self) -> None:
        """The form emptied. Its widgets are let go, not deleted on the spot: the one whose change brought this
        about (a pick list choosing "No legend", which removes the legend and so shows another form) is still in
        the middle of telling Qt about it, and is deleted once it has finished."""
        while self.form.rowCount():
            row = self.form.takeRow(0)
            for item in (row.labelItem, row.fieldItem):
                w = item.widget() if item is not None else None
                if w is not None:
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
        while self.buttons.count():
            w = self.buttons.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _done(self, what: str) -> None:
        if not self._filling:
            self.b.touch(what)

    def _row(self, caption: str, w: QtWidgets.QWidget, tip: str = "") -> QtWidgets.QWidget:
        w.setAccessibleName(caption)
        if tip:
            w.setToolTip(tip)
        lab = QtWidgets.QLabel(caption)
        if tip:
            lab.setToolTip(tip)
        self.form.addRow(lab, w)
        return w

    def text(self, caption: str, get: Callable, put: Callable, tip: str = "", blank: str = "") -> QtWidgets.QLineEdit:
        e = QtWidgets.QLineEdit(str(get() or ""))
        e.setPlaceholderText(blank)

        def apply():
            if e.text() != str(get() or ""):
                put(e.text())
                self._done(caption)

        e.editingFinished.connect(apply)
        return self._row(caption, e, tip)

    def number(self, caption: str, get: Callable, put: Callable, tip: str = "", blank: str = "automatic",
               unit: str = "") -> QtWidgets.QLineEdit:
        """A number that may be left blank (then it is chosen automatically)."""
        v = get()
        e = QtWidgets.QLineEdit("" if v in (None, "", 0, 0.0) and blank else f"{float(v):g}" if v not in (None, "") else "")
        e.setPlaceholderText(blank + (f" ({unit})" if unit and blank else unit))

        def apply():
            new = _num(e.text())
            old = get()
            if (new is None) != (old in (None, "", 0, 0.0) and bool(blank)) or (new is not None and new != old):
                put(new)
                self._done(caption)

        e.editingFinished.connect(apply)
        return self._row(caption + (f" ({unit})" if unit else ""), e, tip)

    def spin(self, caption: str, get: Callable, put: Callable, lo: float, hi: float, step: float, decimals: int = 1,
             tip: str = "", zero_words: str = "") -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        if zero_words:
            s.setSpecialValueText(zero_words)
        s.setValue(float(get() or 0.0))
        s.valueChanged.connect(lambda v: (put(int(v) if decimals == 0 else float(v)), self._done(caption)))
        return self._row(caption, s, tip)

    def choice(self, caption: str, items: dict, get: Callable, put: Callable, tip: str = "") -> QtWidgets.QComboBox:
        c = QtWidgets.QComboBox()
        c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(14)
        for k, words in items.items():
            c.addItem(words, k)
        c.setCurrentIndex(max(0, c.findData(get())))
        c.currentIndexChanged.connect(lambda _i: (put(c.currentData()), self._done(caption)))
        return self._row(caption, c, tip)

    def check(self, caption: str, get: Callable, put: Callable, tip: str = "") -> QtWidgets.QCheckBox:
        c = QtWidgets.QCheckBox()
        c.setChecked(bool(get()))
        c.toggled.connect(lambda on: (put(bool(on)), self._done(caption)))
        return self._row(caption, c, tip)

    def colour(self, caption: str, get: Callable, put: Callable, tip: str = "") -> QtWidgets.QPushButton:
        from .builder import ColourButton

        c = ColourButton(get() or "")
        c.changed.connect(lambda: (put(c.colour), self._done(caption)))
        return self._row(caption, c, tip)

    def action(self, words: str, slot: Callable, tip: str = "", on: bool = True, why_not: str = "") -> QtWidgets.QPushButton:
        b = widgets.button(words, slot, tip if on else why_not, quiet=True)
        b.setEnabled(on)
        self.buttons.addWidget(b)
        return b

    # ---- what is shown
    def refresh(self) -> None:
        """The same thing shown again, from the figure as it now is (after an undo, a drag, a new panel)."""
        focus = QtWidgets.QApplication.focusWidget()
        if focus is not None and self.isAncestorOf(focus) and isinstance(focus, (QtWidgets.QLineEdit,
                                                                                 QtWidgets.QAbstractSpinBox)):
            return                                    # someone is typing here: their box is left alone
        self.show_for(self.tag)

    def show_for(self, tag) -> None:
        self.tag = tag
        self._filling = True
        try:
            self._clear()
            spec = self.b.spec
            if tag is None or tag.kind == "figure" or not (0 <= tag.panel < len(spec.panels) or tag.kind == "figtext"):
                self._figure()
                return
            r = spec.panels[tag.panel] if tag.panel >= 0 else None
            letter = self.b.letter_of(tag.panel) if tag.panel >= 0 else ""
            where = f" of panel {letter}" if letter else ""
            self.heading.setText(elements.KIND_WORDS.get(tag.kind, tag.kind) + (where if tag.kind != "panel" else
                                                                                f" {letter}" if letter else ""))
            self.hint.setText("")
            getattr(self, f"_{tag.kind}", self._nothing)(r, tag)
        finally:
            self._filling = False

    def _nothing(self, r, tag) -> None:
        self.hint.setText("Nothing to set here.")

    # ---- the figure itself
    def _figure(self) -> None:
        spec, s = self.b.spec, self.b.spec.style
        self.tag = None
        self.heading.setText("Figure")
        self.hint.setText("Click anything on the figure to change it; drag to move it; double-click text to "
                          "rewrite it. With nothing selected, these are the figure's own settings.")
        names = {"": "No journal (85, 114, 174 mm)", **{k: j.name for k, j in journals.JOURNALS.items()}}
        self.choice("Journal", names, lambda: spec.journal, self.b.set_journal,
                    "Sets the column widths, the font and the text sizes to the journal's own rules, and what the "
                    "export checks against")
        wide = {k: f"{w} ({self.b.width_of(k):g} mm)" for k, w in WIDTHS.items() if self.b.width_of(k)}
        wide["custom"] = "A width of my own"
        self.choice("Width", wide, lambda: spec.size if spec.size in SIZES else "custom", self.b.set_width,
                    "The figure's width on the page")
        if spec.size not in SIZES:
            self.number("Page width", lambda: spec.size_mm()[0], lambda v: self.b.set_page(v, None), unit="mm",
                        blank="")
        self.number("Height", lambda: spec.height_mm if spec.size in SIZES else spec.size_mm()[1],
                    lambda v: self.b.set_page(None, v), "The page's height; blank: as high as the rows need",
                    blank="as the rows need", unit="mm")
        self.spin("Columns", lambda: spec.columns, self.b.set_columns, 1, 8, 1, 0,
                  "How many columns the grid of panels has; a panel can span several")
        self.check("Letter the panels", lambda: spec.labels, lambda on: setattr(spec, "labels", on),
                   "A, B, C at each panel's top left corner")
        self.choice("Letters as", {"A": "A", "a": "a", "(A)": "(A)", "A.": "A."}, lambda: s.label_style,
                    lambda v: setattr(s, "label_style", v))
        self.text("Font", lambda: s.font, lambda v: setattr(s, "font", v.strip() or "Arial"),
                  "The figure's font: Arial or Helvetica for most journals")
        self.spin("Text size", lambda: s.font_size, lambda v: setattr(s, "font_size", v), 4, 28, 0.5, 1,
                  "In points at final size; 5 to 8 for a printed figure")
        for key, words in (("tick", "Tick labels"), ("label", "Axis titles"), ("title", "Panel titles"),
                           ("letter", "Panel letters"), ("annotation", "Text and scale bars"), ("legend", "Legend")):
            self.spin(words, lambda k=key: getattr(s, f"{k}_size"), lambda v, k=key: setattr(s, f"{k}_size", v), 0, 28,
                      0.5, 1, f"The size of the {words.lower()} in points; 'as the text' follows the text size",
                      zero_words="as the text")
        self.spin("Line width", lambda: s.line_width, lambda v: setattr(s, "line_width", v), 0.2, 6, 0.1, 1,
                  "The curves' width in points")
        self.spin("Axis lines", lambda: s.axis_width, lambda v: setattr(s, "axis_width", v), 0.2, 3, 0.1, 1,
                  "The axis lines' and ticks' width in points")
        self.choice("Colours", {k: k for k in PALETTES}, lambda: s.palette, lambda v: setattr(s, "palette", v),
                    "The groups' colours, in order; the first sets are colour-blind-safe")
        self.choice("Theme", {k: k for k in THEMES}, lambda: s.theme, self.b.set_theme,
                    "A whole look in one click: colours, text, axis lines, grid, background")
        self.spin("Snap to", lambda: spec.snap_mm, lambda v: setattr(spec, "snap_mm", v), 0, 10, 0.5, 1,
                  "A panel dragged or resized lands on a grid of this many mm", zero_words="no grid")
        self.action("Add text on the page", self.b.add_page_text, "Words anywhere on the page, outside any panel")
        self.action("More of the look (axes, grid, background)...", lambda: self.b.show_all_settings("Theme"),
                    "Every setting of the figure's look")

    # ---- a panel
    def _panel(self, r, tag) -> None:
        k, spec, b = tag.panel, self.b.spec, self.b
        f = r.frame()
        inset = page.is_inset(spec.panels, k)
        self.hint.setText(b.source_words(k))
        self.text("Title", lambda: r.title, lambda v: setattr(r, "title", v), "Above the panel; blank: none")
        self.text("Letter", lambda: r.letter, lambda v: setattr(r, "letter", v.strip()),
                  "This panel's letter; blank: the next one in order; a hyphen: none", blank="the next one")
        self.action(f"Plot: {PLOTS.get(r.plot, (r.plot,))[0]}...", lambda: b.choose_plot(k),
                    "Choose how the numbers are drawn, from pictures of each kind", on=f is not None,
                    why_not="This panel has no data yet: choose its data first.")
        self.action("Change its data..." if f is not None else "Choose its data...", lambda: b.choose_data(k),
                    "A recording's traces, a table of results, a kept Workbench measurement, a file")
        if inset:
            names = {i: f"panel {b.letter_of(i) or i + 1}" for i in range(len(spec.panels))
                     if i != k and not page.is_inset(spec.panels, i)}
            self.choice("Inset of", names, lambda: r.inset_of, lambda v: setattr(r, "inset_of", int(v)))
            for i, words in enumerate(("Left", "Bottom", "Width", "Height")):
                self.spin(words, lambda i=i: (list(r.inset_rect) or page.inset_default())[i],
                          lambda v, i=i: b.set_inset_part(k, i, v), 0.0, 1.0, 0.02, 2,
                          "As a fraction of the panel it sits in")
            self.action("Make it a panel of its own", lambda: b.set_inset(k, -1))
        elif len(r.rect) == 4:
            for i, words in enumerate(("Left", "Top", "Width", "Height")):
                self.spin(f"{words} (mm)", lambda i=i: r.rect[i], lambda v, i=i: b.set_rect_part(k, i, v), 0.0, 400.0,
                          0.5, 1, "On the page, in millimetres from its top left corner")
            self.action("Put the panels back in the grid", lambda: b.place_freely(k, False),
                        "Every panel takes a cell of the grid again")
        else:
            cols, rows = page.spans(spec.panels)[k]
            self.spin("Spans columns", lambda: cols, lambda v: b.set_span(k, int(v), None), 1, max(1, spec.columns), 1,
                      0, "How many of the grid's columns this panel takes")
            self.spin("Spans rows", lambda: rows, lambda v: b.set_span(k, None, int(v)), 1, 6, 1, 0,
                      "How many of the grid's rows this panel takes")
            self.action("Place the panels freely (drag and resize)", lambda: b.place_freely(k, True),
                        "The panels leave the grid, each staying where it is: then drag one anywhere, pull its "
                        "corners, or type its millimetres")
        if r.question == "example" and r.plot == "trace":
            self.check("Scale bars, no axes", lambda: r.scale_bars, lambda on: setattr(r, "scale_bars", on),
                       "Traces with a scale bar for each axis instead of the axes themselves")
            self.check("Traces one under the other", lambda: bool(getattr(r, "stack", False)),
                       lambda on: setattr(r, "stack", on), "Each trace on its own level instead of overlaid")
            if r.y2:
                self.check("Command under the traces", lambda: bool(r.command.get("show")),
                           lambda on: r.command.update(show=on), "The command each trace was recorded under, below it")
            self.text("Cut the traces at", lambda: b.pair_text(r.clip_y), lambda v: setattr(r, "clip_y", b.pair(v)),
                      "From, to in y's units: what lies beyond is not drawn (spikes truncated); blank: all of it",
                      blank="from, to")
            self.action("Add a stimulus bar", lambda: b.add_stim_bar(k),
                        "A bar over the traces where the light (or a drug) was on")
        self.action("Add text", lambda: b.add_annotation(k, "text"), "Words on the panel; drag them where they go",
                    on=f is not None, why_not="This panel has no data yet.")
        self.action("Add an arrow with text", lambda: b.add_annotation(k, "arrow"), on=f is not None,
                    why_not="This panel has no data yet.")
        n = len(spec.panels)
        if not inset:
            self.action("Add an inset to this panel", lambda: b.add_inset(k),
                        "A small panel inside this one (a zoom, a single cell's trace)", on=n < b.MOST,
                        why_not=f"A figure holds {b.MOST} panels at most.")
        self.action("Duplicate this panel", lambda: b.duplicate_panel(k), on=n < b.MOST,
                    why_not=f"A figure holds {b.MOST} panels at most.")
        self.action("Remove this panel", lambda: b.remove_panel(k), "Undo brings it back", on=n > 1,
                    why_not="A figure keeps at least one panel.")
        self.action("All of this panel's settings...", lambda: b.show_all_settings("Data"),
                    "The columns, rows, averaging, statistics, axes and marks of this panel, in full")

    def _letter(self, r, tag) -> None:
        s = self.b.spec.style
        self.hint.setText("Drag the letter to move it; its size and weight are the same for every panel.")
        self.text("Letter", lambda: r.letter or self.b.letter_of(tag.panel), lambda v: setattr(r, "letter", v.strip()),
                  "Blank: the next one in order; a hyphen: none")
        self.spin("Size", lambda: s.size_of("letter"), lambda v: setattr(s, "letter_size", v), 4, 28, 0.5, 1,
                  "In points, for every panel's letter")
        self.check("Bold", lambda: s.letter_bold, lambda on: setattr(s, "letter_bold", on))
        self.action("Put it back at the corner", lambda: (setattr(r, "letter_offset", []), self._done("letter")),
                    on=bool(r.letter_offset), why_not="It is at the panel's corner.")

    def _title(self, r, tag) -> None:
        s = self.b.spec.style
        self.text("Title", lambda: r.title, lambda v: setattr(r, "title", v))
        self.spin("Size", lambda: s.size_of("title"), lambda v: setattr(s, "title_size", v), 4, 28, 0.5, 1,
                  "In points, for every panel's title")
        self.check("Bold", lambda: s.title_bold, lambda on: setattr(s, "title_bold", on))

    def _xlabel(self, r, tag) -> None:
        self._axis(r, "x")

    def _ylabel(self, r, tag) -> None:
        self._axis(r, "y")

    def _xaxis(self, r, tag) -> None:
        self._axis(r, "x")

    def _yaxis(self, r, tag) -> None:
        self._axis(r, "y")

    def _axis(self, r, which: str) -> None:
        b, s = self.b, self.b.spec.style
        self.text("Title", lambda: getattr(r, f"{which}_label"), lambda v: setattr(r, f"{which}_label", v),
                  "With its units in brackets; blank: the column's own name", blank=b.default_label(r, which))
        self.text("Range", lambda: b.pair_text(getattr(r, f"{which}_range")),
                  lambda v: setattr(r, f"{which}_range", b.pair(v)), "From, to; blank: fit the data", blank="from, to")
        self.check("Logarithmic", lambda: getattr(r, f"{which}_log"), lambda on: setattr(r, f"{which}_log", on))
        if which == "y":
            self.check("Starts at zero", lambda: bool(r.y_from_zero if r.y_from_zero is not None else False),
                       lambda on: setattr(r, "y_from_zero", on), "A bar's height means something only from zero")
        else:
            self.spin("Labels' angle", lambda: r.x_rotate, lambda v: setattr(r, "x_rotate", v), 0, 90, 15, 0,
                      "Turn the labels, for long group names")
        self.check("Axes cross at zero", lambda: r.zero_axes, lambda on: setattr(r, "zero_axes", on),
                   "Both axes drawn through 0, 0: for a current-voltage plot")
        self.spin("Tick labels' size", lambda: s.size_of("tick"), lambda v: setattr(s, "tick_size", v), 4, 28, 0.5, 1,
                  "In points, for every panel")
        self.spin("Titles' size", lambda: s.size_of("label"), lambda v: setattr(s, "label_size", v), 4, 28, 0.5, 1,
                  "In points, for every panel's axis titles")

    def _legend(self, r, tag) -> None:
        s = self.b.spec.style
        self.hint.setText("Drag the legend to put it anywhere on the panel.")
        places = dict(LEGEND_PLACES)
        if len(r.legend_at) == 2:
            places = {"dragged": "Where I dragged it", **places}
        self.choice("Place", places, lambda: "dragged" if len(r.legend_at) == 2 else r.legend,
                    lambda v: self.b.set_legend_place(tag.panel, v))
        self.spin("Columns", lambda: s.legend_cols, lambda v: setattr(s, "legend_cols", int(v)), 1, 6, 1, 0)
        self.check("Frame", lambda: s.legend_frame, lambda on: setattr(s, "legend_frame", on))
        self.spin("Size", lambda: s.size_of("legend"), lambda v: setattr(s, "legend_size", v), 4, 28, 0.5, 1)
        self.action("Remove the legend", lambda: self.b.remove_thing(tag))

    def _scalebar(self, r, tag) -> None:
        bar = r.scale_bar if tag.key != "command" else r.command
        xu, yu = self.b.units_of(r, "x"), self.b.units_of(r, "y2" if tag.key == "command" else "y")
        self.hint.setText("Drag the bars to move them. A blank length is a round value near a fifth of what is "
                          "drawn.")
        if tag.key != "command":
            self.number(f"Time bar", lambda: bar.get("x"), lambda v: self.b.set_dict(bar, "x", v), unit=xu)
            self.text("Its label", lambda: bar.get("x_label", ""), lambda v: self.b.set_dict(bar, "x_label", v.strip()),
                      blank="the length and its unit")
            self.check("Show the time bar", lambda: bar.get("show_x", True), lambda on: bar.update(show_x=on))
        self.number("Vertical bar", lambda: bar.get("y"), lambda v: self.b.set_dict(bar, "y", v), unit=yu)
        self.text("Its label", lambda: bar.get("y_label", ""), lambda v: self.b.set_dict(bar, "y_label", v.strip()),
                  blank="the length and its unit")
        if tag.key != "command":
            self.check("Show the vertical bar", lambda: bar.get("show_y", True), lambda on: bar.update(show_y=on))
            self.choice("The bars open to the", {"right": "left (the corner at the right)", "left": "right (the "
                        "corner at the left)"}, lambda: bar.get("corner", "right"), lambda v: bar.update(corner=v))
        self.number("Line width", lambda: bar.get("width"), lambda v: self.b.set_dict(bar, "width", v), unit="pt",
                    blank="as the curves")
        if tag.key != "command":
            self.action("Put them back at the corner", lambda: (bar.pop("at", None), self._done("scale bars")),
                        on="at" in bar, why_not="They are at the corner.")
            self.action("Use axes instead of scale bars", lambda: (setattr(r, "scale_bars", False),
                                                                   self.b.select(None), self._done("axes")))

    def _stimbar(self, r, tag) -> None:
        i = int(tag.key)
        if not 0 <= i < len(r.stim_bars):
            return self._nothing(r, tag)
        bar = r.stim_bars[i]
        xu = self.b.units_of(r, "x")
        self.hint.setText("A bar over the plot while the stimulus was on. Drag it up or down.")
        self.number("From", lambda: bar.get("start"), lambda v: bar.update(start=float(v or 0.0)), unit=xu, blank="")
        self.number("To", lambda: bar.get("end"), lambda v: self.b.set_dict(bar, "end", v), unit=xu,
                    blank="a tick at its time")
        lights = {"": "A colour of my own", "electrical": "An electrical shock (a black tick)",
                  **{str(nm): words for words, nm in elements.LIGHTS.items()}}
        self.choice("Stimulus", lights, lambda: "electrical" if bar.get("kind") == "electrical" else
                    str(int(bar["nm"])) if bar.get("nm") and not bar.get("colour") else "",
                    lambda v: self.b.set_stim_light(bar, v), "The light's wavelength sets the bar's colour")
        self.colour("Colour", lambda: elements.stim_colour(bar), lambda v: bar.update(colour=v))
        self.text("Label", lambda: bar.get("label", ""), lambda v: bar.update(label=v.strip()),
                  "Written once, over the first bar that has it; e.g. 470 nm, 5 ms")
        self.spin("Height above the plot", lambda: bar.get("y", 1.02), lambda v: bar.update(y=v), -0.2, 1.6, 0.01, 2,
                  "Where the bar's lower edge is: 1 is the top of the plot area")
        self.spin("Thickness", lambda: bar.get("height", 0.035), lambda v: bar.update(height=v), 0.005, 0.3, 0.005, 3)
        self.action("Same height and colour for every bar", lambda: self.b.match_stim_bars(tag.panel, i),
                    on=len(r.stim_bars) > 1, why_not="This is the panel's only stimulus bar.")
        self.action("Remove this bar", lambda: self.b.remove_thing(tag))

    def _command(self, r, tag) -> None:
        self.check("Show the command", lambda: bool(r.command.get("show")), lambda on: r.command.update(show=on))
        self.spin("Its share of the height", lambda: r.command.get("share", 0.22), lambda v: r.command.update(share=v),
                  0.08, 0.6, 0.02, 2)

    def _annotation(self, r, tag) -> None:
        i = int(tag.key)
        if not 0 <= i < len(r.annotations):
            return self._nothing(r, tag)
        a = r.annotations[i]
        self.hint.setText("Drag to move; double-click to rewrite.")
        self.text("Text", lambda: a.get("text", ""), lambda v: a.update(text=v))
        self.number("Size", lambda: a.get("size"), lambda v: self.b.set_dict(a, "size", v), unit="pt",
                    blank="as other text")
        self.check("Bold", lambda: a.get("bold", False), lambda on: a.update(bold=on))
        self.colour("Colour", lambda: a.get("colour") or self.b.spec.style.ink, lambda v: a.update(colour=v))
        self.number("At x", lambda: a.get("x"), lambda v: a.update(x=float(v or 0.0)), blank="")
        self.number("At y", lambda: a.get("y"), lambda v: a.update(y=float(v or 0.0)), blank="")
        if a.get("kind") == "arrow":
            self.number("Text at x", lambda: a.get("x2"), lambda v: self.b.set_dict(a, "x2", v), blank="")
            self.number("Text at y", lambda: a.get("y2"), lambda v: self.b.set_dict(a, "y2", v), blank="")
        self.action("Remove it", lambda: self.b.remove_thing(tag))

    def _bracket(self, r, tag) -> None:
        own = self.b.bracket(tag.panel, tag.key)
        self.hint.setText("Drag the bracket up or down; double-click its text to rewrite it.")
        self.text("Text", lambda: own.get("text", ""), lambda v: self.b.set_bracket(tag, "text", v.strip()),
                  blank="the exact p value")
        self.choice("Every bracket says", {"p": "The exact p value", "stars": "Stars", "both": "Stars and p"},
                    lambda: r.stats_text, lambda v: setattr(r, "stats_text", v),
                    "Journals ask for exact p values; J Physiol does not accept asterisks alone")
        self.number("Height", lambda: own.get("y"), lambda v: self.b.set_bracket(tag, "y", v), blank="automatic")
        self.action("Hide this bracket", lambda: self.b.remove_thing(tag))
        self.action("Put every bracket back", lambda: (setattr(r, "brackets", []), self._done("brackets")),
                    on=bool(r.brackets), why_not="Every bracket is where it was placed automatically.")

    def _series(self, r, tag) -> None:
        g = tag.key
        look = self.b.look_of(tag.panel, g)
        d = r.series.get(g, {})

        def put(key, value, default):
            mine = dict(r.series.get(g, {}))
            if value == default:
                mine.pop(key, None)
            else:
                mine[key] = value
            r.series = {**r.series, g: mine} if mine else {k: v for k, v in r.series.items() if k != g}

        self.hint.setText(f"The look of “{g}” in this panel.")
        self.text("Name", lambda: d.get("label", ""), lambda v: put("label", v.strip(), ""), blank=g)
        self.colour("Colour", lambda: d.get("colour") or (look.colour if look else ""),
                    lambda v: self.b.set_group_colour(g, v),
                    "This group's colour in every panel of the figure (and in this project's later figures): the "
                    "same condition, the same colour")
        self.choice("Marker", dict(MARKERS), lambda: d.get("marker") or (look.marker if look else "o"),
                    lambda v: put("marker", v, look.marker if look else "o"))
        self.choice("Line", dict(LINES), lambda: d.get("line") if d.get("line") is not None else
                    (look.line if look else "-"), lambda v: put("line", v, look.line if look else "-"))
        self.spin("Width", lambda: d.get("width") or (look.width if look else 1.0),
                  lambda v: put("width", v, look.width if look else 1.0), 0.2, 6, 0.1, 1)
        self.check("Filled markers", lambda: d.get("filled", True), lambda on: put("filled", on, True))
        self.action("Hide this group", lambda: (put("hidden", True, False), self.b.select(None), self._done("hide")),
                    "Leave it out of the panel and of its numbers; Groups, in All settings, brings it back")
        self.action("Every group's look, and their order...", lambda: self.b.show_all_settings("Groups"))

    def _refline(self, r, tag) -> None:
        i = int(tag.key)
        if not 0 <= i < len(r.ref_lines):
            return self._nothing(r, tag)
        line = r.ref_lines[i]
        self.choice("Line", {"y": "Horizontal, at y", "x": "Vertical, at x"}, lambda: line.get("axis", "y"),
                    lambda v: line.update(axis=v))
        self.number("At", lambda: line.get("at"), lambda v: line.update(at=float(v or 0.0)), blank="")
        self.text("Label", lambda: line.get("label", ""), lambda v: line.update(label=v.strip()))
        self.action("Remove it", lambda: self.b.remove_thing(tag))

    def _marker(self, r, tag) -> None:
        i = int(tag.key)
        if not 0 <= i < len(r.markers):
            return self._nothing(r, tag)
        m = r.markers[i]
        self.number("From x", lambda: m.get("at"), lambda v: m.update(at=float(v or 0.0)), blank="")
        self.number("To x", lambda: m.get("end"), lambda v: m.update(end=v), blank="a line, not a span")
        self.text("Label", lambda: m.get("label", ""), lambda v: m.update(label=v.strip()))
        self.action("Remove it", lambda: self.b.remove_thing(tag))

    def _vmlabel(self, r, tag) -> None:
        self.hint.setText("What is written at the start of each trace: the potential it rested at, or was held at.")
        self.text("Text", lambda: r.vm_labels.get(tag.key, ""), lambda v: r.vm_labels.update({tag.key: v.strip()}))
        self.action("Hide these labels", lambda: (setattr(r, "show_vm", False), self.b.select(None),
                                                  self._done("labels")))

    def _figtext(self, r, tag) -> None:
        i = int(tag.key)
        texts = self.b.spec.texts
        if not 0 <= i < len(texts):
            return self._nothing(r, tag)
        t = texts[i]
        self.heading.setText("Text on the page")
        self.text("Text", lambda: t.get("text", ""), lambda v: t.update(text=v))
        self.number("Size", lambda: t.get("size"), lambda v: self.b.set_dict(t, "size", v), unit="pt",
                    blank="as other text")
        self.check("Bold", lambda: t.get("bold", False), lambda on: t.update(bold=on))
        self.number("From the left", lambda: t.get("x"), lambda v: t.update(x=float(v or 0.0)), unit="mm", blank="")
        self.number("From the top", lambda: t.get("y"), lambda v: t.update(y=float(v or 0.0)), unit="mm", blank="")
        self.action("Remove it", lambda: self.b.remove_thing(tag))
