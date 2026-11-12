"""Workbench: measure anything by hand, as in Clampfit, and keep how it was measured (the manual workbench,
section 3 of the analysis expansion brief).

The trace on top (the chosen sweeps overlaid, averaged, or the one nearest their mean; any channel, the stimulus
marked), named regions dragged on it (a baseline, a measure, a fit; a region can repeat on every stimulus
pulse), the table of every measure per sweep below, a plot of any measure against sweep, time, command level
or another measure, and the fit drawn with its residuals. Clicks set an onset, a peak or a threshold on a sweep;
a sweep is left out with a reason. "Keep" saves the measurement with the recording (Results shows it as measured
by hand, with every choice); "Save as template" keeps the regions and measures for every file of a protocol.
Nothing here computes: ``analysis.workbench`` does, so a kept measurement is made again from its description.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ... import __version__
from ...analysis import workbench as wb
from .. import theme, widgets
from ..widgets import box, label
from .base import Screen
from .results import RowsModel

MAX_DRAWN = 60              # sweeps drawn on the trace at most (the rest are still measured)


class WorkbenchEdit(QtWidgets.QUndoCommand):
    """One change to the measurement (regions, transforms, options, exclusions...), undone through the window's
    stack (Ctrl+Z)."""

    def __init__(self, screen, before: dict, after: dict, what: str) -> None:
        super().__init__(f"workbench: {what}")
        self.screen, self.before, self.after = screen, before, after
        self._done = True                      # the change is already made when the command is pushed

    def redo(self) -> None:
        if self._done:                         # pushed: nothing to do again (the screen measures once, not twice)
            self._done = False
            return
        self.screen._apply_state(self.after)

    def undo(self) -> None:
        self.screen._apply_state(self.before)


class KeptEdit(QtWidgets.QUndoCommand):
    """A change to what the project keeps from the Workbench (a measurement kept or deleted, a template saved or
    deleted, a template run on a protocol's files), undone through the window's stack like any other step.
    ``before`` and ``after``: {"hands": {path: its kept measurements}, "templates": [...]}."""

    def __init__(self, screen, before: dict, after: dict, what: str) -> None:
        super().__init__(f"workbench: {what}")
        self.screen, self.before, self.after = screen, before, after
        self._done = True                      # the change is already made when the command is pushed

    def redo(self) -> None:
        if self._done:
            self._done = False
            return
        self.screen._restore_kept(self.after)

    def undo(self) -> None:
        self.screen._restore_kept(self.before)


def able(widget, on: bool, does: str, why_not: str) -> None:
    """``widget`` enabled when it can do something, its tooltip saying what it does, or why it cannot now."""
    widget.setEnabled(bool(on))
    widget.setToolTip(does if on else why_not)


class WorkbenchScreen(Screen):
    key = "workbench"
    title = "Workbench"
    purpose = ("Measure anything by hand: sweeps, regions, Clampfit's statistics, fits and plots, kept with the "
               "recording so every number can be made again.")

    def __init__(self, window) -> None:
        super().__init__(window)
        self._recording = None
        self._path: Path | None = None
        self._m = wb.HandMeasurement("by hand")
        self._result: dict | None = None
        self._aux: dict[str, np.ndarray] = {}
        self._region_items: dict[str, pg.LinearRegionItem] = {}
        self._click_mode = ""                 # "", "onset", "peak", "threshold", "exclude"
        self._pending = None
        self._quiet = False
        self._browse: int | None = None       # the sweep stepped to (its number from 0), None: none singled out
        self._cursors: list[float] | None = None      # the two cursors' times (s), kept while the file is open
        self._cursor_items: list = []
        self.keep_button = widgets.primary("Keep this measurement", self.keep)
        self.template_button = widgets.button("Save as template...", self.save_template)
        self.send_button = QtWidgets.QToolButton()
        self.send_button.setText("Send to a figure")
        self.send_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        send = QtWidgets.QMenu(self.send_button)
        send.setToolTipsVisible(True)
        self.send_traces_action = send.addAction("The traces shown, as a panel of example traces", self.send_traces)
        self.send_table_action = send.addAction("This measurement's table, as a graph", self.send_table)
        self.send_button.setMenu(send)
        self.add_action(self.send_button)
        self.add_action(self.keep_button)
        self.add_action(self.template_button)
        # ---- the body: the trace, the table and plot under it
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        tl = box("v", "xs")
        bar = box("h", "s")
        self.sweeps_box = QtWidgets.QLineEdit()
        self.sweeps_box.setPlaceholderText("sweeps: 1-10, 15 | every 2nd | odd | last 5 | all")
        self.sweeps_box.editingFinished.connect(self._sweeps_typed)
        bar.addWidget(label("Sweeps", "muted"))
        bar.addWidget(self.sweeps_box, 2)
        self.mode = QtWidgets.QComboBox()
        for key, words in wb.MODES.items():
            self.mode.addItem(words, key)
        self.mode.currentIndexChanged.connect(lambda _i: self._changed("mode"))
        self.mode.setAccessibleName("Sweeps shown and measured as")
        self.mode.setToolTip("Measure every sweep on its own, or one trace made from them: their average, their "
                             "median, or the sweep closest to their mean")
        bar.addWidget(self.mode)
        self.channel = QtWidgets.QComboBox()
        self.channel.currentIndexChanged.connect(lambda _i: self._changed("channel"))
        self.channel.setAccessibleName("Channel")
        self.channel.setToolTip("The channel drawn and measured")
        bar.addWidget(self.channel)
        self.show_stimulus = QtWidgets.QCheckBox("Stimulus")
        self.show_stimulus.setChecked(True)
        self.show_stimulus.toggled.connect(lambda _on: self._draw())
        bar.addWidget(self.show_stimulus)
        self.show_command = QtWidgets.QCheckBox("Command")
        self.show_command.toggled.connect(lambda _on: self._draw())
        bar.addWidget(self.show_command)
        bar.addStretch(1)
        self.click_label = label("", "muted")
        bar.addWidget(self.click_label)
        tl.addLayout(bar)
        # the second line: one sweep at a time, and the cursors
        row2 = box("h", "s")
        self.prev_button = widgets.button("◀ Sweep", lambda: self.step_sweep(-1), quiet=True)
        self.next_button = widgets.button("Sweep ▶", lambda: self.step_sweep(1), quiet=True)
        self.prev_button.setAccessibleName("Previous sweep")
        self.next_button.setAccessibleName("Next sweep")
        self.alone = QtWidgets.QCheckBox("Alone")
        self.alone.setAccessibleName("Show the sweep alone")
        self.alone.toggled.connect(lambda _on: self._draw())
        self.all_button = widgets.button("All sweeps", lambda: self.browse(None), quiet=True)
        self.browse_label = label("", "muted")
        self.cursors_box = QtWidgets.QCheckBox("Cursors")
        self.cursors_box.toggled.connect(self._cursors_toggled)
        self.cursors_to_region = widgets.button("Region to the cursors", self._region_to_cursors, quiet=True)
        self.cursor_label = label("", "muted")
        self.cursor_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.cursor_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        for w in (self.prev_button, self.next_button, self.alone, self.all_button, self.browse_label,
                  self.cursors_box, self.cursors_to_region):
            row2.addWidget(w)
        row2.addStretch(1)
        tl.addLayout(row2)
        tl.addWidget(self.cursor_label)           # the cursors' readout, on a line of its own
        self.cursor_label.setVisible(False)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.plot.setFocusPolicy(QtCore.Qt.StrongFocus)          # a click on the trace, then the keys act on it
        tl.addWidget(self.plot, 1)
        top.setLayout(tl)
        split.addWidget(top)
        self.tabs = QtWidgets.QTabWidget()
        self.table = QtWidgets.QTableView()
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.rows_model = RowsModel([], self.table)       # one model, its rows replaced on every measurement
        self.table.setModel(self.rows_model)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tabs.addTab(self.table, "Table")
        plot_page = QtWidgets.QWidget()
        pl = box("v", "xs")
        pbar = box("h", "s")
        self.plot_y = QtWidgets.QComboBox()
        self.plot_x = QtWidgets.QComboBox()
        self.plot_region = QtWidgets.QComboBox()
        for w, name, tip in ((self.plot_y, "Measure", "The measure plotted, one point per sweep (and pulse)"),
                             (self.plot_x, "against", "What it is plotted against: the sweep's number, its time in "
                              "the recording, the command level, or another measure"),
                             (self.plot_region, "region", "The region whose measures are plotted")):
            pbar.addWidget(label(name, "muted"))
            pbar.addWidget(w)
            w.setAccessibleName(f"Plot: {name}")
            w.setToolTip(tip)
            w.currentIndexChanged.connect(lambda _i: self._draw_series())
        pbar.addStretch(1)
        pl.addLayout(pbar)
        self.series = pg.PlotWidget()
        pl.addWidget(self.series, 1)
        plot_page.setLayout(pl)
        self.tabs.addTab(plot_page, "Plot")
        fit_page = QtWidgets.QWidget()
        fl = box("v", "xs")
        self.fit_words = label("No fit yet: choose a model and a fit region in the details panel.", "muted", wrap=True)
        fl.addWidget(self.fit_words)
        self.residuals = pg.PlotWidget()
        fl.addWidget(self.residuals, 1)
        fit_page.setLayout(fl)
        self.tabs.addTab(fit_page, "Fit")
        self.summary = label("", "", wrap=True)
        self.tabs.addTab(self._scrolled(self.summary), "Summary")
        split.addWidget(self.tabs)
        split.setSizes([500, 300])
        self.body(split)
        self._details()
        for p in (self.plot, self.series, self.residuals):
            theme.style_plot(p, window.palette)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.keep, context=QtCore.Qt.WidgetWithChildrenShortcut)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, lambda: self._set_click_mode(""),
                            context=QtCore.Qt.WidgetWithChildrenShortcut)
        # the keys, with the trace clicked: the arrows move the selected region (0.1 ms; with Shift, 1 ms; with
        # Alt, its end alone), Page Up and Page Down step through the sweeps
        self.keys = {}
        for keys, slot in (("Left", lambda: self.nudge_region(-1e-4)), ("Right", lambda: self.nudge_region(1e-4)),
                           ("Shift+Left", lambda: self.nudge_region(-1e-3)),
                           ("Shift+Right", lambda: self.nudge_region(1e-3)),
                           ("Alt+Left", lambda: self.nudge_region(-1e-4, end_only=True)),
                           ("Alt+Right", lambda: self.nudge_region(1e-4, end_only=True)),
                           ("PgUp", lambda: self.step_sweep(-1)), ("PgDown", lambda: self.step_sweep(1))):
            self.keys[keys] = QtWidgets.QShortcut(QtGui.QKeySequence(keys), self.plot, slot,
                                                  context=QtCore.Qt.WidgetWithChildrenShortcut)
        self.table.clicked.connect(self._row_clicked)

    @staticmethod
    def _scrolled(widget) -> QtWidgets.QScrollArea:
        """``widget`` scrolling up and down only: it is as wide as the room, its rows wrapping, so no control
        ends up past the right edge."""
        s = QtWidgets.QScrollArea()
        s.setWidgetResizable(True)
        s.setFrameShape(QtWidgets.QFrame.NoFrame)
        s.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        s.setWidget(widget)
        return s

    # ---- the details panel: regions, measures, options, transforms, clicks, kept measurements
    def _details(self) -> None:
        dl = box("v", "s", "m")
        dl.addWidget(label("Regions", "heading"))
        self.region_list = QtWidgets.QListWidget()
        self.region_list.setMaximumHeight(110)
        self.region_list.currentRowChanged.connect(self._region_picked)
        dl.addWidget(self.region_list)
        self.add_region_button = widgets.button("Add region", self._add_region)
        self.remove_region_button = widgets.button("Remove region", self._remove_region, quiet=True)
        dl.addWidget(widgets.Wrap(self.add_region_button, self.remove_region_button, stretch=(1, 1), spacing="xs"))
        self.r_name = QtWidgets.QLineEdit()
        self.r_kind = QtWidgets.QComboBox()
        self.r_kind.addItems(list(wb.REGION_KINDS))
        self.r_start = QtWidgets.QDoubleSpinBox()
        self.r_end = QtWidgets.QDoubleSpinBox()
        for sp, name in ((self.r_start, "The region's start"), (self.r_end, "The region's end")):
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(2)
            sp.setSuffix(" ms")
            sp.setAccessibleName(name)
            sp.setToolTip(f"{name}, in ms from the sweep's start (from each pulse when the region repeats on every "
                          "pulse)")
        self.r_kind.setAccessibleName("Region kind")
        self.r_per_pulse = QtWidgets.QCheckBox("Repeat on each stimulus pulse")
        self.r_pulse = QtWidgets.QSpinBox()
        self.r_pulse.setRange(0, 999)
        self.r_pulse.setSpecialValueText("every pulse")
        for w, sig in ((self.r_name, self.r_name.editingFinished), (self.r_kind, self.r_kind.currentIndexChanged),
                       (self.r_start, self.r_start.editingFinished), (self.r_end, self.r_end.editingFinished),
                       (self.r_per_pulse, self.r_per_pulse.toggled), (self.r_pulse, self.r_pulse.editingFinished)):
            sig.connect(lambda *_a: self._region_edited())
        dl.addWidget(widgets.field("Name", self.r_name))
        dl.addWidget(widgets.field("Kind", self.r_kind, "baseline: taken off the measures; fit: the fit's stretch"))
        dl.addWidget(widgets.field("From, to", widgets.Wrap(self.r_start, self.r_end, stretch=(1, 1), spacing="xs")))
        dl.addWidget(self.r_per_pulse)
        dl.addWidget(widgets.field("Pulse", self.r_pulse, "0: every pulse"))
        dl.addWidget(widgets.hairline())
        dl.addWidget(label("Measures", "heading"))
        dl.addWidget(label("Clampfit's Statistics, in each measure region of each sweep", "hint", wrap=True))
        self.measure_list = QtWidgets.QListWidget()
        self.measure_list.setMaximumHeight(150)
        for key, (title, words) in wb.MEASURES.items():
            item = QtWidgets.QListWidgetItem(title)
            item.setData(QtCore.Qt.UserRole, key)
            item.setToolTip(words)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if key in self._m.measures else QtCore.Qt.Unchecked)
            self.measure_list.addItem(item)
        self.measure_list.itemChanged.connect(lambda _i: self._changed("measures"))
        dl.addWidget(self.measure_list)
        opts = widgets.Advanced("Options")
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems(["auto", "up", "down"])
        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(1, 999)
        self.rise = QtWidgets.QLineEdit("10-90")
        self.decay = QtWidgets.QLineEdit("90-10")
        self.onset_method = QtWidgets.QComboBox()
        self.onset_method.addItems(list(wb.ONSET_METHODS))
        self.onset_sd = QtWidgets.QDoubleSpinBox()
        self.onset_sd.setRange(0.5, 20)
        self.onset_sd.setValue(2.0)
        self.value_at = QtWidgets.QLineEdit()
        self.value_at.setPlaceholderText("ms (blank: the region's start)")
        self.threshold = QtWidgets.QLineEdit()
        self.threshold.setPlaceholderText("level for threshold crossings")
        for w, sig in ((self.direction, self.direction.currentIndexChanged), (self.smooth, self.smooth.editingFinished),
                       (self.rise, self.rise.editingFinished), (self.decay, self.decay.editingFinished),
                       (self.onset_method, self.onset_method.currentIndexChanged),
                       (self.onset_sd, self.onset_sd.editingFinished), (self.value_at, self.value_at.editingFinished),
                       (self.threshold, self.threshold.editingFinished)):
            sig.connect(lambda *_a: self._changed("options"))
        opts.add(widgets.field("Direction", self.direction, "of the peak"))
        opts.add(widgets.field("Smoothing points", self.smooth, "a running mean before the peak is found"))
        opts.add(widgets.field("Rise %", self.rise, "lower-upper"))
        opts.add(widgets.field("Decay %", self.decay, "upper-lower"))
        opts.add(widgets.field("Onset by", self.onset_method, "threshold: past the noise multiple, sustained; foot: "
                                                             "the 20-80% rise extrapolated to the baseline; "
                                                             "curvature: the largest before the peak"))
        opts.add(widgets.field("Onset noise SD", self.onset_sd, "the lab's: 2"))
        opts.add(widgets.field("Value at", self.value_at))
        opts.add(widgets.field("Threshold", self.threshold))
        dl.addWidget(opts)
        dl.addWidget(widgets.hairline())
        dl.addWidget(label("Fit", "heading"))
        self.fit_model = QtWidgets.QComboBox()
        self.fit_model.addItem("none", "")
        for key, words in wb.FIT_MODELS.items():
            self.fit_model.addItem(words, key)
        self.fit_region = QtWidgets.QComboBox()
        self.fit_custom = QtWidgets.QLineEdit()
        self.fit_custom.setPlaceholderText("a*exp(-t/b)+c  (t in ms from the region's start)")
        self.fit_model.currentIndexChanged.connect(lambda _i: self._changed("fit"))
        self.fit_region.currentIndexChanged.connect(lambda _i: self._changed("fit"))
        self.fit_custom.editingFinished.connect(lambda: self._changed("fit"))
        dl.addWidget(widgets.field("Model", self.fit_model))
        dl.addWidget(widgets.field("On region", self.fit_region))
        dl.addWidget(widgets.field("Expression", self.fit_custom, "for the typed model"))
        dl.addWidget(widgets.hairline())
        dl.addWidget(label("Transforms", "heading"))
        dl.addWidget(label("Applied in order before measuring; the raw data stay as they are", "hint", wrap=True))
        self.transform_list = QtWidgets.QListWidget()
        self.transform_list.setMaximumHeight(90)
        dl.addWidget(self.transform_list)
        add = QtWidgets.QToolButton()
        add.setText("Add transform")
        add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(add)
        for key, words in wb.TRANSFORMS.items():
            menu.addAction(words, lambda k=key: self._add_transform(k))
        add.setMenu(menu)
        self.add_transform_button = add
        self.remove_transform_button = widgets.button("Remove", self._remove_transform, quiet=True)
        self.remove_transform_button.setAccessibleName("Remove transform")
        self.clear_transforms_button = widgets.button("Clear all", lambda: self._set_transforms([]), quiet=True)
        self.clear_transforms_button.setAccessibleName("Clear transforms")
        dl.addWidget(widgets.Wrap(add, self.remove_transform_button, self.clear_transforms_button, spacing="xs"))
        self.transform_list.currentRowChanged.connect(lambda _r: self._enable())
        dl.addWidget(widgets.hairline())
        dl.addWidget(label("By click on the trace", "heading"))
        self.click_buttons = {}
        for what, words, tip in (
                ("onset", "Set onset", "Then click the trace: the onset of the sweep under the mouse is set there"),
                ("peak", "Set peak", "Then click the trace: the peak of the sweep under the mouse is set there"),
                ("threshold", "Set threshold", "Then click the trace: a spike threshold is set there"),
                ("exclude", "Leave out a sweep", "Then click a sweep: it is left out of the measures, with the "
                                                 "reason you give")):
            b = widgets.button(words, lambda _=False, w=what: self._set_click_mode(
                "" if self._click_mode == w else w), tip)
            b.setCheckable(True)
            self.click_buttons[what] = (b, tip)
        row = [b for b, _tip in self.click_buttons.values()]
        dl.addWidget(widgets.Wrap(*row[:2], stretch=(1, 1), spacing="xs"))
        dl.addWidget(widgets.Wrap(*row[2:], stretch=(1, 1), spacing="xs"))
        self.set_words = label("", "muted", wrap=True)
        dl.addWidget(self.set_words)
        self.clear_set_button = widgets.button("Clear clicks and exclusions", self._clear_overrides, quiet=True)
        dl.addWidget(self.clear_set_button)
        dl.addWidget(widgets.hairline())
        dl.addWidget(label("Kept with this recording", "heading"))
        self.name_box = QtWidgets.QLineEdit("by hand")
        dl.addWidget(widgets.field("Name", self.name_box))
        self.kept = QtWidgets.QListWidget()
        self.kept.setMaximumHeight(90)
        self.kept.itemDoubleClicked.connect(self._load_kept)
        self.kept.currentRowChanged.connect(lambda _r: self._enable())
        dl.addWidget(self.kept)
        self.load_kept_button = widgets.button("Load", self._load_kept)
        self.load_kept_button.setAccessibleName("Load kept")
        self.delete_kept_button = widgets.button("Delete", self._delete_kept, quiet=True)
        self.delete_kept_button.setAccessibleName("Delete kept")
        dl.addWidget(widgets.Wrap(self.load_kept_button, self.delete_kept_button, stretch=(1, 1), spacing="xs"))
        dl.addWidget(label("Templates", "heading"))
        self.template_list = QtWidgets.QComboBox()
        self.template_list.setAccessibleName("Template")
        self.template_list.setToolTip("The saved templates: a measurement's regions, measures, fit and transforms, "
                                      "for every file of a protocol")
        self.template_list.currentIndexChanged.connect(lambda _i: self._enable())
        dl.addWidget(self.template_list)
        self.apply_template_button = widgets.button("Apply here", self._apply_template)
        self.run_template_button = widgets.button("Run on its protocol's files", self._run_template)
        self.delete_template_button = widgets.button("Delete", self._delete_template, quiet=True)
        self.delete_template_button.setAccessibleName("Delete template")
        dl.addWidget(widgets.Wrap(self.apply_template_button, self.delete_template_button, stretch=(1, 1),
                                  spacing="xs"))
        dl.addWidget(self.run_template_button)
        dl.addStretch(1)
        body = QtWidgets.QWidget()
        body.setLayout(dl)
        outer = box("v")
        outer.addWidget(self._scrolled(body))
        self.details.setLayout(outer)
        for w, name in ((self.r_pulse, "Which pulse"), (self.direction, "Direction of the peak"),
                        (self.smooth, "Smoothing points"), (self.onset_method, "Onset by"),
                        (self.onset_sd, "Onset noise multiple"), (self.fit_model, "Fit model"),
                        (self.fit_region, "Fit on region")):
            w.setAccessibleName(name)
        for combo in body.findChildren(QtWidgets.QComboBox):     # a long choice is cut, never widens the panel
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0) + 24)     # its list shows it whole
        self._enable()

    def _enable(self) -> None:
        """Every control on when it can do something, off (saying why) when it cannot."""
        if not hasattr(self, "delete_template_button"):          # called while the panel is still being made
            return
        rec, m, project = self._recording, self._m, self.window.project
        no_rec = "Select a recording in Files first."
        has = rec is not None
        stim = has and bool(rec.stimulus.times_s) and any(len(x) for x in rec.stimulus.times_s)
        able(self.show_stimulus, stim, "Mark each stimulus pulse on the trace (dashed lines)",
             no_rec if not has else "No stimulus was found in this recording, so there is none to mark.")
        able(self.show_command, has and rec.command is not None, "Draw the command (the step) under the trace, "
             "scaled to fit", no_rec if not has else "This recording has no command channel to draw.")
        able(self.add_region_button, has, "A new region over the middle of the view; drag its edges, or type its "
             "times", no_rec)
        each = has and m.selection.mode == "each"
        several = each and len(m.selection.sweeps) > 1
        not_each = no_rec if not has else ("The sweeps are measured as one trace (their average, median or the "
                                           "typical one): choose 'Every sweep' to step through them." if not each
                                           else "Only one sweep is chosen.")
        able(self.prev_button, several, "Step to the sweep before (Page Up): it is drawn in bold, its row selected "
             "in the table, and the cursors read it", not_each)
        able(self.next_button, several, "Step to the next sweep (Page Down)", not_each)
        on_one = several and self._browse is not None
        able(self.alone, on_one, "Draw the sweep stepped to without the others",
             not_each if not several else "Step to a sweep first (Sweep ▶).")
        able(self.all_button, on_one, "Back to every sweep drawn alike",
             not_each if not several else "No sweep is singled out.")
        able(self.cursors_box, has, "Two lines to drag over the trace: the time and the value under each, the "
             "difference between them, the slope and the mean between them", no_rec)
        cur = has and self.cursors_box.isChecked() and self._current_region() is not None
        able(self.cursors_to_region, cur, "The selected region is set to span from cursor 1 to cursor 2 (Undo "
             "puts it back)", no_rec if not has else "Tick Cursors first." if not self.cursors_box.isChecked()
             else "Select a region in the list first.")
        shown = has and self._result is not None and bool(self._result.get("traces"))
        self.send_button.setEnabled(has)
        self.send_button.setToolTip("Open the figure builder with what is on this screen as a new panel" if has
                                    else no_rec)
        self.send_traces_action.setEnabled(shown)
        self.send_traces_action.setToolTip("The sweeps chosen, over the stretch of time now in view, drawn as "
                                           "example traces with scale bars (as recorded: filters and other "
                                           "transforms are not applied)" if shown else no_rec)
        rows = has and self._result is not None and bool(self._result.get("rows"))
        self.send_table_action.setEnabled(rows)
        self.send_table_action.setToolTip("Every measure, sweep by sweep, as a table in the builder: the first "
                                          "measure against the sweep to start with" if rows else
                                          no_rec if not has else "Nothing is measured yet: tick a measure.")
        region = self._current_region()
        able(self.remove_region_button, region is not None, "Remove the selected region (Undo brings it back)",
             "No region is selected.")
        able(self.r_per_pulse, region is not None and (stim or region.per_pulse), "The region's times count from "
             "each stimulus pulse, so it repeats on every pulse of every sweep", "No region is selected."
             if region is None else "No stimulus was found in this recording, so a region cannot repeat on its "
             "pulses.")
        able(self.r_pulse, region is not None and region.per_pulse, "0: every pulse; 1, 2...: that pulse only",
             "For a region that repeats on each stimulus pulse.")
        able(self.add_transform_button, has, "A step applied to every sweep before it is measured, after the ones "
             "already listed; the raw data stay as they are", no_rec)
        k = self.transform_list.currentRow()
        able(self.remove_transform_button, 0 <= k < len(m.transforms), "Remove the selected transform (Undo brings "
             "it back)", "No transform is selected." if m.transforms else "There is no transform to remove.")
        able(self.clear_transforms_button, bool(m.transforms), "Remove every transform (Undo brings them back)",
             "There is no transform to clear.")
        for what, (b, tip) in self.click_buttons.items():
            able(b, has, tip, no_rec)
            b.setChecked(self._click_mode == what)
        able(self.clear_set_button, bool(m.overrides or m.exclusions), "Forget every onset, peak and threshold set "
             "by a click and bring back the sweeps left out (Undo restores them)",
             "Nothing was set by a click and no sweep is left out.")
        ready = has and project is not None and self._result is not None
        able(self.keep_button, ready, "Saved with the recording; Results lists it as measured by hand, with every "
             "choice (Ctrl+S). Undo takes it back.", "Select a recording of an open project first.")
        able(self.template_button, has and project is not None, "The regions, measures, fit and transforms, to run "
             "on every file of a protocol", "Select a recording of an open project first.")
        picked = self.kept.currentItem() is not None
        none_kept = "Nothing is kept with this recording yet (Keep this measurement)." if not self.kept.count() \
            else "Select a kept measurement in the list."
        able(self.load_kept_button, picked, "Open the selected kept measurement here (double-click does too)",
             none_kept)
        able(self.delete_kept_button, picked, "Delete the selected kept measurement (Undo brings it back)",
             none_kept)
        tpl = self.template_list.count() > 0 and self.template_list.currentData() is not None
        none_tpl = "No template is saved yet (Save as template...)."
        able(self.apply_template_button, tpl and has, "The template's regions, measures, fit and transforms on "
             "this recording's sweeps", none_tpl if not tpl else no_rec)
        able(self.run_template_button, tpl, "Every file of the project whose protocol the template names is "
             "measured and kept, as a Clampfit batch (Undo takes the batch back)", none_tpl)
        able(self.delete_template_button, tpl, "Delete the selected template (Undo brings it back)", none_tpl)

    # ---- the window's calls
    def on_theme(self, palette) -> None:
        for p in (self.plot, self.series, self.residuals):
            theme.style_plot(p, palette)
        self._draw()

    def on_shown(self) -> None:
        if self._pending is not None:
            self.on_selection(self._pending)
        self._fill_templates()

    def on_project(self) -> None:
        self._fill_templates()

    def on_selection(self, path) -> None:
        if path is not None and not self.is_current():
            self._pending = path
            return
        self._pending = None
        if path is None or self.window.project is None:
            self._recording, self._path, self._result = None, None, None
            self.plot.clear()
            self.rows_model.set_rows([])
            self._enable()
            return
        try:
            self._recording = self.window.recording(path)
        except Exception as exc:
            self.window.say(f"Could not read {path.name}: {exc}")
            return
        self._path = Path(path)
        rec = self._recording
        if not rec.stimulus.times_s:
            from ...analysis.stimulus import find_stimulus

            rec.stimulus = find_stimulus(rec)
        self._quiet = True
        self.channel.clear()
        for c in rec.channels:
            self.channel.addItem(f"{c.name} ({c.units})", c.name)
        self.channel.setCurrentIndex(0)
        self._quiet = False
        # a fresh measurement on this file keeps the regions and measures from the last file (the same protocol
        # measured the same way), with every sweep chosen
        m = self._m
        m.file = str(path)
        m.selection = wb.Selection(list(range(rec.n_sweeps)), "", m.selection.mode)
        m.overrides, m.exclusions = [], []
        if not m.regions:
            m.regions = self._default_regions()
        self._browse, self._cursors = None, None
        self._apply_state(m.to_dict(), push=False)
        self._fill_kept()
        self._fill_templates()

    def _default_regions(self) -> list[wb.Region]:
        rec = self._recording
        if rec.stimulus.times_s and any(len(x) for x in rec.stimulus.times_s):
            return [wb.Region("baseline", -0.020, -0.001, "baseline", per_pulse=True),
                    wb.Region("response", 0.001, 0.050, "measure", per_pulse=True)]
        d = rec.sweep_duration_s
        return [wb.Region("baseline", 0.0, min(0.1, d / 10), "baseline"),
                wb.Region("measure", min(0.1, d / 10), min(0.6, d), "measure")]

    # ---- state: the measurement is the one truth; the controls show it, the plot draws it
    def _snapshot(self) -> dict:
        return self._m.to_dict()

    def _commit(self, before: dict, what: str) -> None:
        after = self._snapshot()
        if after == before:
            return
        self.window.undo.push(WorkbenchEdit(self, before, after, what))

    def _apply_state(self, state: dict, push: bool = False) -> None:
        self._m = wb.HandMeasurement.from_dict(state)
        self._fill_controls()
        self._measure()

    def _fill_controls(self) -> None:
        m = self._m
        was_quiet = self._quiet
        self._quiet = True
        try:
            self.sweeps_box.setText(m.selection.text())
            i = self.mode.findData(m.selection.mode)
            self.mode.setCurrentIndex(max(i, 0))
            j = self.channel.findData(m.selection.channel) if m.selection.channel else 0
            self.channel.setCurrentIndex(max(j, 0))
            row = self.region_list.currentRow()
            self.region_list.clear()
            for r in m.regions:
                self.region_list.addItem(f"{r.name} ({r.kind})")
            self.region_list.setCurrentRow(min(max(row, 0), len(m.regions) - 1))
            self._show_region()
            for k in range(self.measure_list.count()):
                item = self.measure_list.item(k)
                item.setCheckState(QtCore.Qt.Checked if item.data(QtCore.Qt.UserRole) in m.measures else QtCore.Qt.Unchecked)
            o = m.options
            self.direction.setCurrentText(o.direction)
            self.smooth.setValue(int(o.smooth_points))
            self.rise.setText(f"{o.rise_pct[0]:g}-{o.rise_pct[1]:g}")
            self.decay.setText(f"{o.decay_pct[0]:g}-{o.decay_pct[1]:g}")
            self.onset_method.setCurrentText(o.onset_method)
            self.onset_sd.setValue(o.onset_noise_sd)
            self.value_at.setText("" if not np.isfinite(o.value_at_s) else f"{o.value_at_s * 1e3:g}")
            self.threshold.setText("" if not np.isfinite(o.threshold) else f"{o.threshold:g}")
            self.fit_region.clear()
            for r in m.regions:
                self.fit_region.addItem(r.name, r.name)
            self.fit_model.setCurrentIndex(max(self.fit_model.findData(m.fit_model), 0))
            self.fit_region.setCurrentIndex(max(self.fit_region.findData(m.fit_region), 0))
            self.fit_custom.setText(m.fit_custom)
            self.transform_list.clear()
            for x in m.transforms:
                self.transform_list.addItem(wb.describe_transform(x))
            self.plot_region.clear()
            for r in m.regions:
                if r.kind != "baseline":
                    self.plot_region.addItem(r.name, r.name)
            y_now = self.plot_y.currentData()
            self.plot_y.clear()
            for key in m.measures:
                self.plot_y.addItem(wb.MEASURES.get(key, (key,))[0], key)
            if y_now:
                self.plot_y.setCurrentIndex(max(self.plot_y.findData(y_now), 0))
            x_now = self.plot_x.currentData()
            self.plot_x.clear()
            for key, words in (("sweep", "sweep"), ("time", "time (min)"), ("command", "command level")):
                self.plot_x.addItem(words, key)
            for key in m.measures:
                self.plot_x.addItem(wb.MEASURES.get(key, (key,))[0], key)
            if x_now:
                self.plot_x.setCurrentIndex(max(self.plot_x.findData(x_now), 0))
            self.name_box.setText(m.name)
            words = [f"{o.what} of sweep {o.sweep + 1} at {o.time_s * 1e3:.2f} ms" for o in m.overrides]
            words += [f"sweep {e.sweep + 1} left out: {e.reason}" for e in m.exclusions]
            self.set_words.setText("; ".join(words))
        finally:
            self._quiet = was_quiet
        self._enable()

    def _show_region(self) -> None:
        r = self._current_region()
        was_quiet = self._quiet
        self._quiet = True
        try:
            if r is None:
                for w in (self.r_name, self.r_start, self.r_end, self.r_kind, self.r_per_pulse, self.r_pulse):
                    w.setEnabled(False)
                return
            for w in (self.r_name, self.r_start, self.r_end, self.r_kind, self.r_per_pulse, self.r_pulse):
                w.setEnabled(True)
            self.r_name.setText(r.name)
            self.r_kind.setCurrentText(r.kind)
            self.r_start.setValue(r.start_s * 1e3)
            self.r_end.setValue(r.end_s * 1e3)
            self.r_per_pulse.setChecked(r.per_pulse)
            self.r_pulse.setValue(r.pulse)
        finally:
            self._quiet = was_quiet

    def _current_region(self) -> wb.Region | None:
        k = self.region_list.currentRow()
        return self._m.regions[k] if 0 <= k < len(self._m.regions) else None

    # ---- changes from the controls
    def _changed(self, what: str) -> None:
        if self._quiet or self._recording is None:
            return
        before = self._snapshot()
        m = self._m
        if what == "mode":
            m.selection.mode = self.mode.currentData() or "each"
        elif what == "channel":
            name = self.channel.currentData() or ""
            m.selection.channel = "" if name == self._recording.signal_channel.name else name
        elif what == "measures":
            m.measures = [self.measure_list.item(k).data(QtCore.Qt.UserRole) for k in range(self.measure_list.count())
                          if self.measure_list.item(k).checkState() == QtCore.Qt.Checked]
        elif what == "options":
            o = m.options
            o.direction = self.direction.currentText()
            o.smooth_points = int(self.smooth.value())
            o.rise_pct = self._pair(self.rise.text(), (10.0, 90.0))
            o.decay_pct = self._pair(self.decay.text(), (90.0, 10.0))
            o.onset_method = self.onset_method.currentText()
            o.onset_noise_sd = float(self.onset_sd.value())
            o.value_at_s = self._number(self.value_at.text()) / 1e3
            o.threshold = self._number(self.threshold.text())
        elif what == "fit":
            m.fit_model = self.fit_model.currentData() or ""
            m.fit_region = self.fit_region.currentData() or ""
            m.fit_custom = self.fit_custom.text()
        self._commit(before, what)
        self._fill_controls()
        self._measure()

    @staticmethod
    def _pair(text: str, default: tuple) -> tuple:
        try:
            a, b = (float(x) for x in text.replace(",", "-").split("-")[:2])
            return (a, b)
        except ValueError:
            return default

    @staticmethod
    def _number(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return float("nan")

    def _sweeps_typed(self) -> None:
        if self._quiet or self._recording is None:
            return
        try:
            sweeps = wb.parse_sweep_words(self.sweeps_box.text(), self._recording.n_sweeps)
        except ValueError as exc:
            self.window.say(str(exc))
            return
        before = self._snapshot()
        self._m.selection.sweeps = sweeps
        self._commit(before, "sweeps")
        self._fill_controls()
        self._measure()

    def _region_picked(self, _row: int) -> None:
        self._show_region()
        self._highlight_region()
        self._enable()

    def _region_edited(self) -> None:
        r = self._current_region()
        if self._quiet or r is None:
            return
        before = self._snapshot()
        r.name = self.r_name.text().strip() or r.name
        r.kind = self.r_kind.currentText()
        r.start_s, r.end_s = sorted((self.r_start.value() / 1e3, self.r_end.value() / 1e3))
        r.per_pulse = self.r_per_pulse.isChecked()
        r.pulse = int(self.r_pulse.value())
        self._commit(before, "region")
        self._fill_controls()
        self._measure()

    def _add_region(self) -> None:
        if self._recording is None:
            return
        before = self._snapshot()
        (x0, x1), _ = self.plot.viewRange()
        mid, span = (x0 + x1) / 2, (x1 - x0) / 6
        n = len(self._m.regions) + 1
        self._m.regions.append(wb.Region(f"region {n}", mid - span / 2, mid + span / 2, "measure"))
        self._commit(before, "add region")
        self._fill_controls()
        self.region_list.setCurrentRow(len(self._m.regions) - 1)
        self._measure()

    def _remove_region(self) -> None:
        k = self.region_list.currentRow()
        if not (0 <= k < len(self._m.regions)):
            return
        before = self._snapshot()
        del self._m.regions[k]
        self._commit(before, "remove region")
        self._fill_controls()
        self._measure()

    def _region_dragged(self, name: str) -> None:
        item = self._region_items.get(name)
        r = next((x for x in self._m.regions if x.name == name), None)
        if item is None or r is None or self._quiet:
            return
        a, b = item.getRegion()
        before = self._snapshot()
        if r.per_pulse:
            on = self._first_pulse_s()
            a, b = a - on, b - on
        r.start_s, r.end_s = float(a), float(b)
        self._commit(before, "drag region")
        self._show_region()
        self._measure(draw=False)

    def _first_pulse_s(self) -> float:
        rec = self._recording
        k = self._m.selection.sweeps[0] if self._m.selection.sweeps else 0
        times = rec.stimulus.times_s
        return float(times[k][0]) if k < len(times) and len(times[k]) else 0.0

    def _add_transform(self, kind: str) -> None:
        params: dict = {}
        regions = [r.name for r in self._m.regions]
        if kind in ("baseline_region", "baseline_line"):
            if not regions:
                self.window.say("Add a region first.")
                return
            name, ok = QtWidgets.QInputDialog.getItem(self, "Baseline region", "Region:", regions, 0, False)
            if not ok:
                return
            params["region"] = name
            if kind == "baseline_line":
                name2, ok = QtWidgets.QInputDialog.getItem(self, "Second region", "Region:", regions,
                                                           min(1, len(regions) - 1), False)
                if not ok:
                    return
                params["region2"] = name2
        elif kind in ("lowpass", "highpass", "notch"):
            default = {"lowpass": 2000.0, "highpass": 1.0, "notch": 60.0}[kind]
            hz, ok = QtWidgets.QInputDialog.getDouble(self, wb.TRANSFORMS[kind], "Hz:", default, 0.01, 1e6, 2)
            if not ok:
                return
            params["hz"] = hz
            if kind == "notch":
                params["harmonics"] = 3
        elif kind in ("gaussian", "blank"):
            ms, ok = QtWidgets.QInputDialog.getDouble(self, wb.TRANSFORMS[kind], "ms:", 0.5 if kind == "gaussian" else 1.0,
                                                      0.01, 1e4, 2)
            if not ok:
                return
            params["ms"] = ms
        elif kind == "scale":
            f, ok = QtWidgets.QInputDialog.getDouble(self, "Scale", "Factor:", -1.0, -1e6, 1e6, 4)
            if not ok:
                return
            params["factor"] = f
        elif kind == "offset":
            v, ok = QtWidgets.QInputDialog.getDouble(self, "Offset", "Added:", 0.0, -1e9, 1e9, 3)
            if not ok:
                return
            params["value"] = v
        elif kind == "subtract":
            if not self._pick_subtraction(params):
                return
        before = self._snapshot()
        self._m.transforms.append(wb.Transform(kind, params))
        self._commit(before, "transform")
        self._fill_controls()
        self._measure()

    def _pick_subtraction(self, params: dict) -> bool:
        """What to subtract: the average of the chosen sweeps, one sweep, or another file's average (trace
        arithmetic between sweeps and files)."""
        rec = self._recording
        choices = ["the average of the chosen sweeps", "one sweep of this file", "another file's average"]
        what, ok = QtWidgets.QInputDialog.getItem(self, "Subtract", "Subtract:", choices, 0, False)
        if not ok:
            return False
        scale, ok = QtWidgets.QInputDialog.getDouble(self, "Subtract", "Scaled by:", 1.0, -1e6, 1e6, 4)
        if not ok:
            return False
        ch = self._m.selection.channel
        if what == choices[0]:
            sweeps = [k for k in self._m.selection.sweeps if k not in {e.sweep for e in self._m.exclusions}]
            avg = np.mean([wb.raw(rec, k, ch) for k in sweeps], axis=0) if sweeps else None
            if avg is None:
                return False
            label_ = f"the average of sweeps {self._m.selection.text()}"
            self._aux[label_] = avg
        elif what == choices[1]:
            k, ok = QtWidgets.QInputDialog.getInt(self, "Subtract", "Sweep (from 1):", 1, 1, rec.n_sweeps)
            if not ok:
                return False
            label_ = f"sweep {k}"
            self._aux[label_] = wb.raw(rec, k - 1, ch)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "The other file", str(self._path.parent),
                                                            "Recordings (*.abf *.atf *.dat);;All files (*)")
            if not path:
                return False
            from ...io import load

            other = load(path)
            text, ok = QtWidgets.QInputDialog.getText(self, "Subtract", "Its sweeps (blank: all):")
            if not ok:
                return False
            ks = wb.parse_sweep_words(text, other.n_sweeps)
            avg = np.mean([np.asarray(other.sweep(k), float) for k in ks], axis=0)
            label_ = f"{Path(path).name} (average of {len(ks)} sweeps)"
            self._aux[label_] = avg
            params["fs"] = other.sample_rate_hz
        params.update({"label": label_, "scale": scale})
        return True

    def _remove_transform(self) -> None:
        k = self.transform_list.currentRow()
        if not (0 <= k < len(self._m.transforms)):
            return
        before = self._snapshot()
        del self._m.transforms[k]
        self._commit(before, "remove transform")
        self._fill_controls()
        self._measure()

    def _set_transforms(self, items: list) -> None:
        before = self._snapshot()
        self._m.transforms = list(items)
        self._commit(before, "transforms")
        self._fill_controls()
        self._measure()

    # ---- clicks on the trace
    def _set_click_mode(self, mode: str) -> None:
        self._click_mode = mode
        self.click_label.setText({"": "", "onset": "Click the trace: the onset", "peak": "Click the trace: the peak",
                                  "threshold": "Click the trace: the spike threshold",
                                  "exclude": "Click a sweep to leave it out"}[mode] + (" (Esc to stop)" if mode else ""))
        for what, (b, _tip) in self.click_buttons.items():
            b.setChecked(mode == what)

    def _clicked(self, event) -> None:
        if not self._click_mode or self._recording is None or self._result is None:
            return
        vb = self.plot.getPlotItem().getViewBox()
        p = vb.mapSceneToView(event.scenePos())
        x, y = float(p.x()), float(p.y())
        # the sweep under the mouse: the drawn trace nearest the click
        t = self._recording.time_s
        # the sample nearest the click (not the one after it: a hair to the right of a sample would pick the next
        # one, and between near-identical sweeps the next sample can belong to another sweep's noise)
        i = int(np.clip(np.round((x - float(t[0])) * self._recording.sample_rate_hz), 0, t.size - 1))
        nearest, best = None, np.inf
        for lab, k, yy in self._result["traces"]:
            if k is None:
                continue
            d = abs(float(yy[i]) - y) if np.isfinite(yy[i]) else np.inf
            if d < best:
                nearest, best = k, d
        if nearest is None:
            return
        before = self._snapshot()
        mode = self._click_mode
        if mode == "exclude":
            reason, ok = QtWidgets.QInputDialog.getText(self, "Leave out", f"Why sweep {nearest + 1} is left out:")
            if not ok:
                return
            self._m.exclusions = [e for e in self._m.exclusions if e.sweep != nearest]
            self._m.exclusions.append(wb.Exclusion(nearest, reason or "left out by hand"))
        else:
            region = self._current_region()
            rname = region.name if region is not None and region.kind != "baseline" else ""
            self._m.overrides = [o for o in self._m.overrides if not (o.sweep == nearest and o.what == mode)]
            self._m.overrides.append(wb.Override(nearest, mode, x, rname))
        self._commit(before, f"set {mode}")
        self._fill_controls()
        self._measure()

    def _clear_overrides(self) -> None:
        before = self._snapshot()
        self._m.overrides, self._m.exclusions = [], []
        self._commit(before, "clear")
        self._fill_controls()
        self._measure()

    # ---- measuring and drawing
    def _measure(self, draw: bool = True) -> None:
        if self._recording is None:
            return
        try:
            self._result = wb.measure(self._recording, self._m, self._aux)
        except Exception as exc:
            self.window.say(f"The measurement failed: {type(exc).__name__}: {exc}")
            self._result = None
            self.rows_model.set_rows([])
            return
        rows = self._result["rows"]
        # the table's rows: the sweep as kept (from 0; the model shows it from 1, as every table does)
        shown = [{k: v for k, v in r.items() if k not in ("start_ms", "end_ms")} for r in rows]
        self.rows_model.headings = wb.headings(self._m.measures, self._units(), self._command_units())
        self.rows_model.set_rows(shown)
        fit = self._result.get("fit")
        self.fit_words.setText(fit["words"] if fit else "No fit yet: choose a model and a fit region in the details "
                                                        "panel.")
        self._draw_residuals()
        self._draw_series()
        self._summarize()
        if draw:
            self._draw()

    def _units(self) -> str:
        """The units of the channel measured."""
        try:
            return wb.channel_named(self._recording, self._m.selection.channel).units
        except Exception:
            return ""

    def _command_units(self) -> str:
        from ...core.recording import Role

        cmd = self._recording.channel(Role.COMMAND) if self._recording is not None else None
        return cmd.units if cmd is not None else ""

    def _draw(self) -> None:
        if self._recording is None or self._result is None:
            return
        pal = self.window.palette
        rec = self._recording
        self.plot.clear()
        self._region_items = {}
        t = rec.time_s
        tr = self._result["traces"]
        singles = [x for x in tr if x[1] is not None and not x[0].startswith("typical")]
        combined = [x for x in tr if x[1] is None or x[0].startswith("typical")]
        step = max(1, len(singles) // MAX_DRAWN)
        if self._browse is not None and not any(k == self._browse for _lab, k, _y in singles):
            self._browse = None                      # the sweep stepped to is no longer among those chosen
        one = self._browse
        for lab, k, y in singles[::step]:
            if one is not None and (k == one or self.alone.isChecked()):
                continue
            colour = pal.sweep if (combined or one is not None) else theme.series(pal)[k % 8]
            self.plot.plot(t, y, pen=pg.mkPen(colour, width=1), name=lab)
        for lab, k, y in singles:
            if one is not None and k == one:         # the sweep stepped to, over the others
                self.plot.plot(t, y, pen=pg.mkPen(pal.accent, width=2), name=lab)
        for lab, k, y in combined:
            self.plot.plot(t, y, pen=pg.mkPen(pal.average, width=2), name=lab)
        ch = wb.channel_named(rec, self._m.selection.channel)
        self.plot.setLabel("left", ch.name, ch.units)
        self.plot.setLabel("bottom", "Time", "s")
        if self.show_stimulus.isChecked() and rec.stimulus.times_s:
            k0 = self._m.selection.sweeps[0] if self._m.selection.sweeps else 0
            if k0 < len(rec.stimulus.times_s):
                for on in np.asarray(rec.stimulus.times_s[k0], float):
                    self.plot.addItem(pg.InfiniteLine(on, pen=pg.mkPen(pal.warn, width=1, style=QtCore.Qt.DashLine)))
        if self.show_command.isChecked() and rec.command is not None:
            k0 = self._m.selection.sweeps[0] if self._m.selection.sweeps else 0
            c = rec.command[k0]
            ys = [y for _, _, y in tr if y is not None]
            if ys:
                lo, hi = np.nanmin(ys[0]), np.nanmax(ys[0])
                span = (hi - lo) or 1.0
                cs = lo - 0.15 * span + 0.1 * span * (c - c.min()) / ((np.ptp(c)) or 1.0)
                self.plot.plot(t, cs, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DotLine))
        on0 = self._first_pulse_s()
        colours = pal.regions or theme.series(pal)
        for i, r in enumerate(self._m.regions):
            a, b = (r.start_s + on0, r.end_s + on0) if r.per_pulse else (r.start_s, r.end_s)
            col = QtGui.QColor(colours[i % len(colours)])
            col.setAlpha(45)
            item = pg.LinearRegionItem((a, b), movable=True, brush=pg.mkBrush(col),
                                       pen=pg.mkPen(colours[i % len(colours)], width=1))
            item.setZValue(-10)
            item.sigRegionChangeFinished.connect(lambda _it, name=r.name: self._region_dragged(name))
            self.plot.addItem(item)
            # the name rides on the region's left edge, each region a line lower so two names never sit on
            # each other
            pg.InfLineLabel(item.lines[0], r.name + (", each pulse" if r.per_pulse else ""),
                            position=0.97 - 0.07 * (i % 6), anchor=(0, 0.5), color=colours[i % len(colours)],
                            fill=pg.mkBrush(QtGui.QColor(pal.plot_background)))
            self._region_items[r.name] = item
            if r.per_pulse and self._m.selection.sweeps:
                k0 = self._m.selection.sweeps[0]
                for s0, e0, pulse in r.windows(rec, k0)[1:]:
                    ghost = pg.LinearRegionItem((s0, e0), movable=False, brush=pg.mkBrush(col),
                                                pen=pg.mkPen(colours[i % len(colours)], width=1, style=QtCore.Qt.DotLine))
                    ghost.setZValue(-11)
                    self.plot.addItem(ghost)
        fit = self._result.get("fit")
        if fit and fit.get("ok") and len(fit["curve_t"]):
            self.plot.plot(fit["curve_t"], fit["curve_y"], pen=pg.mkPen(pal.accent, width=2))
        for o in self._m.overrides:
            self.plot.addItem(pg.InfiniteLine(o.time_s, pen=pg.mkPen(pal.bad, width=1)))
        self._highlight_region()
        self._draw_cursors()
        self._say_browse()

    # ---- one sweep at a time
    def _singles(self) -> list[int]:
        """The sweeps drawn each on its own, in order."""
        if self._result is None:
            return []
        return [k for lab, k, _y in self._result["traces"] if k is not None and not lab.startswith("typical")]

    def browse(self, sweep: int | None) -> None:
        """Single out ``sweep`` (None: none): drawn in bold, its row selected, read by the cursors."""
        self._browse = sweep if sweep in self._singles() else None
        if self._browse is None:
            self.alone.blockSignals(True)
            self.alone.setChecked(False)
            self.alone.blockSignals(False)
        self._draw()
        self._select_row(self._browse)
        self._enable()

    def step_sweep(self, by: int) -> None:
        sweeps = self._singles()
        if len(sweeps) < 2:
            return
        if self._browse in sweeps:
            k = sweeps[(sweeps.index(self._browse) + by) % len(sweeps)]
        else:
            k = sweeps[0] if by > 0 else sweeps[-1]
        self.browse(k)

    def _say_browse(self) -> None:
        sweeps = self._singles()
        if self._browse is None or self._browse not in sweeps:
            self.browse_label.setText("")
            return
        self.browse_label.setText(f"sweep {self._browse + 1} ({sweeps.index(self._browse) + 1} of {len(sweeps)})")

    def _select_row(self, sweep: int | None) -> None:
        rows = (self._result or {}).get("rows") or []
        row = next((i for i, r in enumerate(rows) if r.get("sweep") == sweep), None) if sweep is not None else None
        self.table.blockSignals(True)
        if row is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(row)
            self.table.scrollTo(self.rows_model.index(row, 0))
        self.table.blockSignals(False)

    def _row_clicked(self, index) -> None:
        rows = (self._result or {}).get("rows") or []
        if index.isValid() and 0 <= index.row() < len(rows) and rows[index.row()].get("sweep") in self._singles():
            self.browse(rows[index.row()]["sweep"])

    # ---- the cursors
    def _cursors_toggled(self, _on: bool) -> None:
        self._draw()
        self._enable()

    def _cursor_trace(self):
        """(its name, its samples): the trace the cursors read: the sweep stepped to, else the one trace made of
        the sweeps, else the first drawn."""
        tr = (self._result or {}).get("traces") or []
        if self._browse is not None:
            hit = next(((lab, y) for lab, k, y in tr if k == self._browse), None)
            if hit is not None:
                return hit
        combined = next(((lab, y) for lab, k, y in tr if k is None or lab.startswith("typical")), None)
        if combined is not None:
            return combined
        return (tr[0][0], tr[0][2]) if tr else ("", None)

    def _draw_cursors(self) -> None:
        self._cursor_items = []
        if not self.cursors_box.isChecked() or self._recording is None:
            self.cursor_label.setText("")
            self.cursor_label.setVisible(False)
            return
        (x0, x1), _ = self.plot.viewRange()
        t = self._recording.time_s
        if self._cursors is None:                    # a third and two thirds across what is in view
            lo, hi = max(x0, float(t[0])), min(x1, float(t[-1]))
            self._cursors = [lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3]
        pal = self.window.palette
        for i, at in enumerate(self._cursors):
            line = pg.InfiniteLine(at, angle=90, movable=True, pen=pg.mkPen(pal.text, width=1.4),
                                   hoverPen=pg.mkPen(pal.accent, width=2), label=str(i + 1),
                                   labelOpts={"position": 0.04, "color": pal.text, "movable": False,
                                              "fill": pg.mkBrush(QtGui.QColor(pal.plot_background))})
            line.setZValue(20)
            line.sigPositionChanged.connect(lambda _l, i=i: self._cursor_moved(i))
            self.plot.addItem(line)
            self._cursor_items.append(line)
        self._read_cursors()

    def _cursor_moved(self, i: int) -> None:
        if i < len(self._cursor_items) and self._cursors is not None:
            self._cursors[i] = float(self._cursor_items[i].value())
            self._read_cursors()

    def set_cursor(self, i: int, at_s: float) -> None:
        """Cursor ``i`` (0 or 1) put at ``at_s`` (as dragging it there would)."""
        if self._cursors is None:
            return
        self._cursors[i] = float(at_s)
        if i < len(self._cursor_items):
            self._cursor_items[i].setValue(float(at_s))
        self._read_cursors()

    def cursor_values(self) -> dict:
        """What the cursors read: each one's time (s) and value, the differences, the slope (per ms) and the mean
        between them; {} when they are off."""
        name, y = self._cursor_trace()
        if self._cursors is None or y is None or not self.cursors_box.isChecked():
            return {}
        t = self._recording.time_s
        fs = self._recording.sample_rate_hz
        i1, i2 = (int(np.clip(round((at - float(t[0])) * fs), 0, t.size - 1)) for at in self._cursors)
        t1, t2, v1, v2 = float(t[i1]), float(t[i2]), float(y[i1]), float(y[i2])
        a, b = sorted((i1, i2))
        between = np.asarray(y[a:b + 1], float)
        dt_ms = (t2 - t1) * 1e3
        return {"trace": name, "t1": t1, "t2": t2, "v1": v1, "v2": v2, "dt_ms": dt_ms, "dv": v2 - v1,
                "slope": (v2 - v1) / dt_ms if dt_ms else float("nan"),
                "mean": float(np.nanmean(between)) if between.size else float("nan")}

    def _read_cursors(self) -> None:
        c = self.cursor_values()
        self.cursor_label.setVisible(bool(c))
        if not c:
            self.cursor_label.setText("")
            return
        u = self._units()
        text = (f"1: {c['t1'] * 1e3:.2f} ms, {c['v1']:.2f} {u} · 2: {c['t2'] * 1e3:.2f} ms, {c['v2']:.2f} {u} · "
                f"Δt {c['dt_ms']:.2f} ms · Δ {c['dv']:.2f} {u} · slope {c['slope']:.3g} {u}/ms · mean between "
                f"{c['mean']:.2f} {u} · on {c['trace']}")
        self.cursor_label.setText(text)
        self.cursor_label.setToolTip(text)

    def _region_to_cursors(self) -> None:
        r = self._current_region()
        if r is None or self._cursors is None or self._recording is None:
            return
        before = self._snapshot()
        a, b = sorted(self._cursors)
        if r.per_pulse:                              # such a region is kept from the pulse, not the sweep's start
            on = self._first_pulse_s()
            a, b = a - on, b - on
        r.start_s, r.end_s = float(a), float(b)
        self._commit(before, "region to the cursors")
        self._fill_controls()
        self._measure()

    # ---- the selected region moved by the keys
    def nudge_region(self, by_s: float, end_only: bool = False) -> None:
        r = self._current_region()
        if r is None or self._recording is None:
            return
        before = self._snapshot()
        least = 2.0 / self._recording.sample_rate_hz
        if end_only:
            r.end_s = max(r.end_s + by_s, r.start_s + least)
        else:
            r.start_s, r.end_s = r.start_s + by_s, r.end_s + by_s
        self._commit(before, "move region")
        self._fill_controls()
        self._measure()

    # ---- what is here, sent to a figure
    def _builder(self):
        return self.window.screens["figures"].open_builder()

    def send_traces(self) -> None:
        """The sweeps chosen, over the time in view, as a panel of example traces in the figure builder."""
        from ...figure import traces

        rec, m = self._recording, self._m
        if rec is None or self._path is None or not m.selection.sweeps:
            return
        (x0, x1), _ = self.plot.viewRange()
        t = rec.time_s
        x0, x1 = max(float(x0), float(t[0])), min(float(x1), float(t[-1]))
        if x1 <= x0:
            x0, x1 = float(t[0]), float(t[-1])
        sweeps = [self._browse] if (self._browse is not None and self.alone.isChecked()) else list(m.selection.sweeps)
        mode = {"each": "each", "average": "average"}.get(m.selection.mode, "typical") if len(sweeps) > 1 else "average"
        name = (f"sweep {sweeps[0] + 1}" if len(sweeps) == 1 else
                {"each": "sweeps", "average": f"average of {len(sweeps)}"}.get(m.selection.mode, "typical sweep"))
        sets = [traces.TraceSet(name, mode, [[str(self._path), int(k), 0, x0] for k in sweeps])]
        b = self._builder()
        b.current = b._target(None)
        if b.current < 0:
            return
        b._load_controls()
        b.set_traces(sets, traces.Window(0.0, (x1 - x0) * 1e3, False, 0.0), f"{self._path.name}: from the Workbench")
        b.redraw_now()
        b.select_panel_on_figure(b.current)
        said = f"{len(sweeps)} sweep{'s' if len(sweeps) != 1 else ''} of {self._path.name}, {x0 * 1e3:.0f} to " \
               f"{x1 * 1e3:.0f} ms, sent to the figure as panel {b.letter_of(b.current) or b.current + 1}"
        if m.transforms:
            said += " (as recorded: the Workbench's transforms are not applied to a figure's traces)"
        self.window.say(said + ".")
        b.statusBar().showMessage("From the Workbench: " + said + ".", 12000)

    def send_table(self) -> None:
        """The measurement as it stands (kept or not) as a table and a graph in the figure builder."""
        if self._recording is None or self._path is None or self._result is None:
            return
        self._m.name = self.name_box.text().strip() or self._m.name or "by hand"
        hand = {"name": self._m.name, "measurement": self._m.to_dict()}
        b = self._builder()
        b.current = b._target(None)
        if b.current < 0:
            return
        b._load_controls()
        b.use_hand(self._path, hand)
        b.redraw_now()
        b.select_panel_on_figure(b.current)
        said = f"“{self._m.name}” of {self._path.name} sent to the figure as panel " \
               f"{b.letter_of(b.current) or b.current + 1}: its table is under Data and statistics"
        self.window.say(said + ".")
        b.statusBar().showMessage("From the Workbench: " + said + ".", 12000)

    def _highlight_region(self) -> None:
        r = self._current_region()
        for name, item in self._region_items.items():
            for line in item.lines:
                line.setPen(pg.mkPen(self.window.palette.accent if r is not None and name == r.name else
                                     self.window.palette.text_muted, width=2 if r is not None and name == r.name else 1))

    def _draw_series(self) -> None:
        self.series.clear()
        if self._result is None:
            return
        y_key, x_key, region = self.plot_y.currentData(), self.plot_x.currentData(), self.plot_region.currentData()
        if not y_key or not x_key:
            return
        x, y, _ = wb.plot_series(self._result["rows"], y_key, x_key, region or "")
        ok = np.isfinite(x) & np.isfinite(y)
        pal = self.window.palette
        self.series.plot(x[ok], y[ok], pen=None, symbol="o", symbolSize=7, symbolBrush=pal.accent,
                         symbolPen=pg.mkPen(pal.accent))
        self.series.setLabel("left", wb.MEASURES.get(y_key, (y_key,))[0])
        self.series.setLabel("bottom", self.plot_x.currentText())

    def _draw_residuals(self) -> None:
        self.residuals.clear()
        fit = (self._result or {}).get("fit")
        if not fit or not fit.get("ok") or not len(fit["residuals"]):
            return
        pal = self.window.palette
        self.residuals.plot(fit["curve_t"], fit["residuals"], pen=pg.mkPen(pal.text_muted, width=1))
        self.residuals.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(pal.border)))
        self.residuals.setLabel("left", "Residual")
        self.residuals.setLabel("bottom", "Time", "s")

    def _summarize(self) -> None:
        if self._result is None:
            return
        self.summary.setText("<br>".join(wb.summary_lines(self._m, self._result, self._units())) + "<br><br><i>"
                             + self._m.describe() + "</i>")

    # ---- keeping, templates
    def keep(self) -> None:
        if self._recording is None or self.window.project is None or self._result is None:
            self.window.say("Nothing to keep: select a recording in a project first.")
            return
        before = self._kept_state()
        self._m.name = self.name_box.text().strip() or "by hand"
        kept = wb.stamp(wb.HandMeasurement.from_dict(self._m.to_dict()), __version__)
        self.window.project.add_hand(self._path, kept, wb.summary_rows(kept, self._result))
        self._commit_kept(before, f"keep '{kept.name}'")
        self._fill_kept()
        self.window.say(f"Kept '{kept.name}' with {self._path.name}; Results lists it as measured by hand "
                        "(Undo takes it back).")

    # what the project keeps from here, as one thing that Undo can put back
    def _kept_state(self, paths=None) -> dict:
        p = self.window.project
        paths = [self._path] if paths is None else list(paths)
        return {"hands": {str(x): [{k: v for k, v in h.items() if k != "id"} for h in p.hands(x)]
                          for x in paths if x is not None},
                "templates": [(t.name, t.protocol, t.measurement) for t in p.templates()]}

    def _commit_kept(self, before: dict, what: str) -> None:
        after = self._kept_state([Path(x) for x in before["hands"]])
        if after != before:
            self.window.undo.push(KeptEdit(self, before, after, what))

    def _restore_kept(self, state: dict) -> None:
        p = self.window.project
        if p is None:
            return
        for path, hands in state["hands"].items():
            p.set_hands(path, hands)
        p.set_templates([wb.Template(*t) for t in state["templates"]])
        self._fill_kept()
        self._fill_templates()

    def _fill_kept(self) -> None:
        self.kept.clear()
        if self._path is not None and self.window.project is not None:
            for h in self.window.project.hands(self._path):
                item = QtWidgets.QListWidgetItem(f"{h['name']} ({h['made_at'][:16].replace('T', ' ')})")
                item.setData(QtCore.Qt.UserRole, h["id"])
                self.kept.addItem(item)
            if self.kept.count():
                self.kept.setCurrentRow(self.kept.count() - 1)        # the newest, ready to load or delete
        self._enable()

    def _load_kept(self, *_a) -> None:
        item = self.kept.currentItem()
        if item is None or self.window.project is None:
            return
        hid = item.data(QtCore.Qt.UserRole)
        h = next((x for x in self.window.project.hands(self._path) if x["id"] == hid), None)
        if h is None:
            return
        before = self._snapshot()
        self._apply_state(h["measurement"])
        self._commit(before, "load")

    def _delete_kept(self) -> None:
        item = self.kept.currentItem()
        if item is None or self.window.project is None:
            return
        before = self._kept_state()
        self.window.project.remove_hand(item.data(QtCore.Qt.UserRole))
        self._commit_kept(before, "delete a kept measurement")
        self._fill_kept()
        self.window.say("The kept measurement is deleted (Undo brings it back).")

    def save_template(self) -> None:
        if self._recording is None or self.window.project is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Save as template", "Name:", text=self.name_box.text())
        if not ok or not name.strip():
            return
        pattern, ok = QtWidgets.QInputDialog.getText(self, "Save as template", "For protocols matching:",
                                                     text=self._recording.protocol or "*")
        if not ok:
            return
        before = self._kept_state()
        self.window.project.save_template(wb.Template.from_measurement(name.strip(), pattern.strip() or "*", self._m))
        self._commit_kept(before, f"save the template '{name.strip()}'")
        self._fill_templates()
        self.template_list.setCurrentIndex(max(0, self.template_list.findData(name.strip())))
        self.window.say(f"Template '{name.strip()}' saved for protocols matching '{pattern.strip() or '*'}'.")

    def _fill_templates(self) -> None:
        now = self.template_list.currentData()
        self.template_list.blockSignals(True)
        self.template_list.clear()
        if self.window.project is not None:
            for tpl in self.window.project.templates():
                self.template_list.addItem(f"{tpl.name} (protocols matching {tpl.protocol})", tpl.name)
            self.template_list.setCurrentIndex(max(0, self.template_list.findData(now)) if
                                               self.template_list.count() else -1)
        self.template_list.blockSignals(False)
        self._enable()

    def _template(self) -> wb.Template | None:
        name = self.template_list.currentData()
        if not name or self.window.project is None:
            return None
        return next((t for t in self.window.project.templates() if t.name == name), None)

    def _apply_template(self) -> None:
        tpl = self._template()
        if tpl is None or self._recording is None:
            return
        before = self._snapshot()
        self._apply_state(tpl.measurement_for(self._recording).to_dict())
        self._commit(before, "template")

    def _run_template(self) -> None:
        tpl = self._template()
        project = self.window.project
        if tpl is None or project is None:
            return
        from ...io import load

        done, failed, other = 0, [], 0
        rows = [r for r in project.recordings()]
        before = self._kept_state([r.path for r in rows])

        def work():
            nonlocal done, other
            for r in rows:
                try:
                    rec = load(r.path)
                    if not tpl.matches(rec.protocol):
                        other += 1
                        continue
                    if not rec.stimulus.times_s:
                        from ...analysis.stimulus import find_stimulus

                        rec.stimulus = find_stimulus(rec)
                    m = tpl.measurement_for(rec)
                    wb.stamp(m, __version__)
                    res = wb.measure(rec, m)
                    project.add_hand(r.path, m, wb.summary_rows(m, res))
                    done += 1
                except Exception as exc:
                    failed.append(f"{r.name}: {exc}")
            return done

        self.window.in_background(f"Running the template '{tpl.name}'", work)
        self._commit_kept(before, f"run the template '{tpl.name}'")
        self._fill_kept()
        self.window.say(f"Template '{tpl.name}' measured and kept on {done} file(s)"
                        + (f"; {other} file(s) of other protocols left alone" if other else "")
                        + (f"; failed on {len(failed)}: {failed[0]}" if failed else "")
                        + (" (Undo takes the batch back)." if done else
                           f": no file's protocol matches '{tpl.protocol}'."))

    def _delete_template(self) -> None:
        tpl = self._template()
        if tpl is None:
            return
        before = self._kept_state()
        self.window.project.delete_template(tpl.name)
        self._commit_kept(before, f"delete the template '{tpl.name}'")
        self._fill_templates()
        self.window.say(f"Template '{tpl.name}' deleted (Undo brings it back).")
