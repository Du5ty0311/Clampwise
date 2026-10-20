"""The Polygon in Live's patterned light view (Use here, PolyScan closed): where the light goes on the slice,
while Clampex times the light itself.

A pattern is made (a full field, a spot, a numbered grid, a line of spots, an image file, a PolyScan sequence, or
shapes drawn on the camera's picture: named regions placed by a cell's centre and a diameter in um, each with a
control region beside it, kept with the cell) and its steps shown one at a time: chosen in the list, the next one
every few seconds, or all loaded into the Polygon and stepped by a TTL at its Trig In. Several steps chosen
together (in the list, or clicked on the picture with Ctrl) are lit together: one cell of a grid of 100, or a few of
them, and nothing else. A grid can tile the camera's picture rather than the whole field, so that when the field is
larger than the picture every cell is one the camera sees. Each step shown is written
to ``polygon_log.csv`` in the cell's folder (``rig.light_record``: to the millisecond, its shapes, area and the
objective), matched to the recordings' sweeps; nothing is left lit after a recording, at Let go or at closing.

The light test and calibration (with the light on and a fluorescent slide or slice in view) shows the Polygon
nothing and everything, to learn which byte lights a mirror, then a few spots, found on the camera's picture, to
learn how the Polygon's field lies under the camera; drawing on the picture and the outlines over it need it. A
check grid of spots the fit did not use then says how far it is off, in um; it is kept for its objective.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...rig import light_record, patterns
from ...rig.polygon import Polygon
from .. import theme, widgets
from ..widgets import box, label

SOURCES = ("Full field", "Spot", "Grid", "Line of spots", "Image file", "Sequence file", "Drawn on the picture")
POLYSCAN_DATA = Path.home() / "Documents" / "Mightex" / "PolyScan3" / "User Data"
SETTLE_S = 0.4                 # after a pattern is sent, before a picture of it is used (at least 3 exposures)
PULSE_WAIT_S = 10.0            # pulsed light: how long each test picture waits for a pulse of the protocol
FLASH_GREY = 4.0               # a frame this much brighter (mean grey) than the darkest one caught a pulse
CHOSEN = (255, 220, 0)         # the steps chosen, outlined on the picture
LIT = (90, 170, 255)           # the steps the patterned light shows now (lit whenever the LED is on), filled


def _pct(value: float = 50.0, low: float = 0.0, high: float = 100.0, step: float = 1.0, suffix: str = " %"):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(low, high)
    s.setDecimals(1)
    s.setSingleStep(step)
    s.setValue(value)
    s.setSuffix(suffix)
    return s


class PolygonPanel(QtWidgets.QWidget):
    """The Polygon held by Clampwise. ``panel`` is the camera's (its picture: the calibration reads it, shapes are
    drawn on it, the steps' outlines go over it)."""

    released = QtCore.Signal()
    _done = QtCore.Signal(object, object)            # what to do next, and what the Polygon's call returned

    def __init__(self, panel, make=Polygon) -> None:
        super().__init__()
        self.panel = panel
        self.polygon = make()
        # every call to the Polygon on a thread of its own, one at a time: its SDK takes 25 s to find it (it looks
        # through every USB 3 port) and 4 s to let it go, and the window must not wait
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="polygon")
        self._pending = 0
        self._wanted = 0                             # the latest step asked for: older ones not yet sent are skipped
        self._opening = False
        self.temperature_c: int | None = None
        self._done.connect(self._finished)
        self.steps: list[patterns.Step] = []
        self.calibrations = patterns.Calibrations()   # the byte that lights, and a map for each objective
        self.calibration = patterns.Calibration()     # the one for the objective in use
        self.drawn: list[pg.ROI] = []
        self.outline_items: list = []
        self._clicking = False                        # the next click on the picture places a region's centre
        self._show_check = False                      # the check grid drawn, until steps are made
        self._cal: dict | None = None                 # the calibration under way
        self._lit: set[str] = set()                   # the names of the steps the patterned light shows now
        self._mask_images: dict = {}                  # an image step's mirrors as a picture to lay over the camera's
        lay = box("v", "s", "m")
        lay.addWidget(label("Clampwise sets where the light goes and holds it; the light itself is switched by "
                            "its LED driver, from the acquisition program, as your protocol says.", "muted", wrap=True))
        self.device_status = label(self.polygon.status, "muted", wrap=True)
        lay.addWidget(self.device_status)
        # while the patterned light is found or let go (25 s, 4 s): a bar and the seconds, the controls greyed
        self.busy_box = QtWidgets.QWidget()
        busy = box("v", "xs")
        self.busy_text = label("", "", wrap=True)
        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 100)
        self.busy_bar.setTextVisible(False)
        busy.addWidget(self.busy_text)
        busy.addWidget(self.busy_bar)
        self.busy_box.setLayout(busy)
        self.busy_box.hide()
        lay.addWidget(self.busy_box)
        self._busy_what, self._busy_s, self._busy_from = "", 0.0, 0.0
        self._busy_timer = QtCore.QTimer(self)
        self._busy_timer.setInterval(200)
        self._busy_timer.timeout.connect(self._busy_tick)
        # the calibration
        lay.addWidget(label("Light test and calibration", "section"))
        cal = box("h", "s")
        self.cal_button = widgets.button("Test and calibrate", self.calibrate,
                                         "With the light on and a fluorescent slide or slice in view: which byte "
                                         "lights a mirror, then where its field lies on the picture")
        cal.addWidget(self.cal_button)
        self.pulsed = QtWidgets.QCheckBox("Light in pulses (play the protocol)")
        self.pulsed.setChecked(True)
        self.pulsed.setToolTip("The light is on only in the protocol's pulses: play the protocol during the test, and "
                               "each test picture waits for a pulse (up to 10 s) and reads the frame it lit. "
                               "Unticked: the light is on all the time")
        cal.addWidget(self.pulsed)
        cal.addStretch(1)
        lay.addLayout(cal)
        cal = box("h", "s")                           # two rows: the panel fits a narrow view
        self.outlines = QtWidgets.QCheckBox("Outlines on the picture")
        self.outlines.setChecked(True)
        self.outlines.setToolTip("The steps' shapes drawn over the camera's picture (the one shown brighter)")
        self.outlines.toggled.connect(lambda *_: self.draw_outlines())
        cal.addWidget(self.outlines)
        self.swap = QtWidgets.QCheckBox("Lit by 0 (swap lit and unlit)")
        self.swap.setToolTip("Which byte turns a mirror onto the slice: the light test finds it; set it here by hand "
                             "when the test cannot run (the light only on in the protocol's pulses). Shown again at once.")
        self.swap.toggled.connect(self._swapped)
        cal.addWidget(self.swap)
        cal.addStretch(1)
        lay.addLayout(cal)
        self.cal_status = label("", "muted", wrap=True)
        lay.addWidget(self.cal_status)
        # the pattern
        lay.addWidget(label("Pattern", "section"))
        src = box("h", "s")
        self.source = QtWidgets.QComboBox()
        self.source.addItems(SOURCES)
        src.addWidget(self.source, 1)
        src.addWidget(widgets.button("Make the steps", self.make_steps, "The pattern's steps, from the settings "
                                                                        "below"))
        lay.addLayout(src)
        self.pages = QtWidgets.QStackedWidget()
        self._build_pages()
        self.source.currentIndexChanged.connect(self.pages.setCurrentIndex)
        lay.addWidget(self.pages)
        self.pattern_note = label("", "muted", wrap=True)
        lay.addWidget(self.pattern_note)
        # the steps
        lay.addWidget(label("Steps", "section"))
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(90)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setToolTip("Ctrl+click or Shift+click to choose several steps: Project lights them together")
        self.list.itemDoubleClicked.connect(lambda *_: self.project())
        self.list.itemSelectionChanged.connect(self.draw_outlines)
        lay.addWidget(self.list, 1)
        lay.addWidget(label("On the picture: yellow, chosen (click a step there to choose it, Ctrl+click for more, "
                            "double-click to show it); blue and filled, what the patterned light shows now (lit "
                            "whenever the LED is on); dashed, the edge of its field.", "muted", wrap=True))
        go = box("h", "s")
        go.addWidget(widgets.button("Previous", lambda: self.step_by(-1), "The step before, shown", quiet=True))
        go.addWidget(widgets.primary("Project", self.project, "Show the chosen step (lit while the LED is on)"))
        go.addWidget(widgets.button("Next", lambda: self.step_by(1), "The next step, shown", quiet=True))
        go.addWidget(widgets.button("Off", self.off, "No mirror lit"))
        go.addStretch(1)
        lay.addLayout(go)
        go = box("h", "s")
        self.auto = QtWidgets.QCheckBox("Next every")
        self.auto.setToolTip("Step on by itself (time it to your protocol's sweeps); unticked: by hand")
        self.every = _pct(10.0, 0.2, 3600.0, 1.0, " s")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.step_by(1, wrap=False))
        self.auto.toggled.connect(self._auto)
        self.every.valueChanged.connect(lambda *_: self._auto(self.auto.isChecked()))
        go.addWidget(self.auto)
        go.addWidget(self.every)
        go.addStretch(1)
        lay.addLayout(go)
        self.off_after = QtWidgets.QCheckBox("Off after each recording")
        self.off_after.setChecked(True)
        self.off_after.setToolTip("When a recording made here ends, no mirror is left lit: the next run never "
                                  "lights the slice with a pattern left from this one")
        lay.addWidget(self.off_after)
        trig = box("h", "s")
        self.on_trigger = QtWidgets.QCheckBox("Step on each trigger (Trig In)")
        self.on_trigger.setToolTip("Project loads every step into the patterned light, and each TTL at its Trig In "
                                   "shows the next one: pulse 1 of a sweep on one region, pulse 2 on another, timed "
                                   "by the protocol's digital output (sent a few ms before each light pulse). Off "
                                   "stops it. Needs the digitizer's digital output wired to Trig In.")
        trig.addWidget(self.on_trigger)
        self.held_ms = _pct(50.0, 1.0, 10000.0, 5.0, " ms")
        self.held_ms.setToolTip("How long each step stays after its trigger: longer than the light pulse")
        trig.addWidget(label("each held", "muted"))
        trig.addWidget(self.held_ms)
        trig.addStretch(1)
        lay.addLayout(trig)
        self.shown_label = label("Nothing shown.", "muted", wrap=True)
        lay.addWidget(self.shown_label)
        widgets.resizable(self, lay)
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._calibration_tick)
        self.load()
        objective = getattr(panel, "objective", None)
        if objective is not None:                     # another objective: its own calibration (or none)
            objective.valueChanged.connect(lambda *_: self._use_objective())
        adapter = getattr(getattr(panel, "dialog", None), "adapter", None)
        if adapter is not None:
            adapter.valueChanged.connect(lambda *_: self._use_objective())
        scene = panel.box.scene() if hasattr(panel.box, "scene") else None
        if scene is not None:
            scene.sigMouseClicked.connect(self._clicked)

    # ---------------------------------------------------------------- the pages of settings
    def _build_pages(self) -> None:
        def page(*rows):
            w = QtWidgets.QWidget()
            g = QtWidgets.QFormLayout()
            g.setContentsMargins(*(theme.SPACE["xs"],) * 4)
            g.setSpacing(theme.SPACE["s"])
            for caption, widget in rows:
                g.addRow(caption, widget)
            w.setLayout(g)
            self.pages.addWidget(w)

        page(("", label("Every mirror: the whole field.", "muted")))
        self.spot_u, self.spot_v, self.spot_d = _pct(50), _pct(50), _pct(5, 0.1, 100, 0.5)
        page(("Across (left to right)", self.spot_u), ("Down (top to bottom)", self.spot_v),
             ("Diameter (of the field's width)", self.spot_d))
        self.rows, self.cols = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for s in (self.rows, self.cols):
            s.setRange(1, 64)
            s.setValue(10)
        self.fill = _pct(100, 5, 100, 5)
        self.round = QtWidgets.QCheckBox("Round spots")
        self.grid_on_picture = QtWidgets.QCheckBox("Over the camera's picture")
        self.grid_on_picture.setChecked(True)
        self.grid_on_picture.setToolTip("The grid tiles what the camera sees (needs the calibration), so every cell "
                                        "is on the picture when the field is larger than it; unticked, the whole "
                                        "field, as its own program lays its grids out")
        page(("Rows", self.rows), ("Columns", self.cols), ("Each spot, of its cell", self.fill), ("", self.round),
             ("", self.grid_on_picture),
             ("", label("Numbered from 1 at the top left along each row, as its own program numbers its grids. "
                        "Choose one cell or several (Ctrl+click) to light only those.", "muted", wrap=True)))
        self.l_u0, self.l_v0, self.l_u1, self.l_v1 = _pct(20), _pct(50), _pct(80), _pct(50)
        self.l_n = QtWidgets.QSpinBox()
        self.l_n.setRange(1, 200)
        self.l_n.setValue(10)
        self.l_d = _pct(3, 0.1, 100, 0.5)
        page(("Start across", self.l_u0), ("Start down", self.l_v0), ("End across", self.l_u1),
             ("End down", self.l_v1), ("Spots", self.l_n), ("Diameter", self.l_d))
        self.image_path = QtWidgets.QLineEdit()
        self.image_path.setPlaceholderText("A black-and-white image: white is lit")
        pick = widgets.button("Choose...", lambda: self._choose(self.image_path, "Images (*.png *.tif *.tiff *.bmp "
                                                                                 "*.jpg)"), quiet=True)
        self.threshold = QtWidgets.QSpinBox()
        self.threshold.setRange(1, 255)
        self.threshold.setValue(128)
        self.invert = QtWidgets.QCheckBox("Dark is lit instead")
        page(("File", self._with(self.image_path, pick)), ("Lit from brightness", self.threshold), ("", self.invert))
        self.seq_path = QtWidgets.QLineEdit()
        self.seq_path.setPlaceholderText("A sequence its own program saved (*.seq)")
        pick = widgets.button("Choose...", lambda: self._choose(self.seq_path, "Sequences (*.seq)",
                                                                str(POLYSCAN_DATA)), quiet=True)
        page(("File", self._with(self.seq_path, pick)),
             ("", label("Its steps as its own program made them (grids, cells, drawn regions).", "muted", wrap=True)))
        draw = QtWidgets.QWidget()
        dv = box("v", "xs")
        d = box("h", "s")
        self.roi_name = QtWidgets.QLineEdit()
        self.roi_name.setPlaceholderText("Name (e.g. Cell 1)")
        self.roi_name.setToolTip("The next region's name: its step is called so, in the log and with the recordings")
        d.addWidget(self.roi_name, 1)
        self.roi_um = _pct(15.0, 1.0, 2000.0, 1.0, " um")
        self.roi_um.setToolTip("The diameter of a region placed by its centre (a soma: about 10-20 um)")
        d.addWidget(self.roi_um)
        d.addWidget(widgets.button("Click the cell's centre", self.click_centre, "The next click on the picture "
                                   "places a round region there, this diameter across", quiet=True))
        dv.addLayout(d)
        d = box("h", "s")
        d.addWidget(widgets.button("Add a rectangle", lambda: self.add_shape("rect"), quiet=True))
        d.addWidget(widgets.button("Add an ellipse", lambda: self.add_shape("ellipse"), quiet=True))
        d.addWidget(widgets.button("Clear", self.clear_shapes, quiet=True))
        d.addStretch(1)
        dv.addLayout(d)
        d = box("h", "s")
        d.addWidget(widgets.button("Add a control beside it", self.add_control, "A region of the same size and "
                                   "shape, this far from the last one (drag it to neuropil with no cell): a response "
                                   "to it is light scattered onto the cell", quiet=True))
        self.control_um = _pct(40.0, 5.0, 2000.0, 5.0, " um")
        self.control_um.setToolTip("How far the control region sits from the region it controls for")
        d.addWidget(self.control_um)
        d.addStretch(1)
        dv.addLayout(d)
        d = box("h", "s")
        self.each_step = QtWidgets.QCheckBox("Each shape its own step")
        self.each_step.setChecked(True)
        d.addWidget(self.each_step)
        d.addStretch(1)
        dv.addLayout(d)
        d = box("h", "s")
        d.addWidget(widgets.button("Keep with the cell", self.save_regions, "The regions saved in the cell's "
                                   "folder (polygon_regions.json), found again with this cell", quiet=True))
        d.addWidget(widgets.button("This cell's regions", self.load_regions, "The regions kept with this cell, "
                                   "drawn again", quiet=True))
        d.addStretch(1)
        dv.addLayout(d)
        draw.setLayout(dv)
        self.pages.addWidget(draw)

    @staticmethod
    def _with(edit, button) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = box("h", "s")
        h.addWidget(edit, 1)
        h.addWidget(button)
        w.setLayout(h)
        return w

    def _choose(self, edit: QtWidgets.QLineEdit, filt: str, start: str = "") -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose", start or edit.text(), filt)
        if path:
            edit.setText(path)
            self.make_steps()

    # ---------------------------------------------------------------- the device, on its own thread
    def _async(self, work, then=None) -> None:
        self._pending += 1

        def run():
            try:
                result = work()
            except Exception as e:                   # the SDK's own failure, said rather than lost
                result = e
            self._done.emit(then, result)

        self._pool.submit(run)

    def _finished(self, then, result) -> None:
        self._pending -= 1
        if then is not None:
            then(result)
        self._said()

    def wait(self, timeout: float = 60.0) -> bool:
        """Until the Polygon's calls asked for are done (closing, tests): True when they are."""
        end = time.monotonic() + timeout
        while self._pending and time.monotonic() < end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.005)
        return not self._pending

    def take(self) -> None:
        """The Polygon found and taken (PolyScan closed), without the window waiting; when it cannot be, said, and
        ``released`` so the view shows PolyScan's place again."""
        if self.polygon.connected or self._opening:
            return
        self._opening = True
        self.device_status.setText("")
        self._busy_start("Finding the patterned light (its driver looks through every USB 3 port)", 25.0)

        def work():
            ok = self.polygon.open()
            self.temperature_c = self.polygon.temperature() if ok else None
            return ok

        def done(ok):
            self._opening = False
            if ok is not True:
                why = str(ok) if isinstance(ok, Exception) else self.polygon.status
                self._busy_end()
                self.panel.window.say(f"The patterned light was not taken: {why}.")
                self.released.emit()
            else:
                self._busy_end()
                self._log(light_record.entry("taken"))
                self.panel.window.say("The patterned light is ready: held by Clampwise.")

        self._async(work, done)

    def _busy_start(self, what: str, expected_s: float) -> None:
        self._busy_what, self._busy_s, self._busy_from = what, expected_s, time.monotonic()
        self._controls(False)
        self.busy_box.show()
        self._busy_tick()
        self._busy_timer.start()

    def _busy_tick(self) -> None:
        s = time.monotonic() - self._busy_from
        self.busy_bar.setValue(int(min(95.0, 100.0 * s / max(self._busy_s, 0.1))))   # never full until done
        self.busy_text.setText(f"{self._busy_what}: {s:.0f} s; it usually takes about {self._busy_s:.0f} s. "
                               "Clampwise goes on meanwhile.")

    def _busy_end(self) -> None:
        self._busy_timer.stop()
        self.busy_box.hide()
        self._controls(True)

    def _controls(self, on: bool) -> None:
        """The panel's controls usable (on), or greyed while the patterned light is found or let go."""
        kinds = (QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox, QtWidgets.QListWidget,
                 QtWidgets.QLineEdit)
        for w in self.findChildren(kinds):
            w.setEnabled(on)
        if on and self._cal is not None:
            self.cal_button.setEnabled(False)

    def busy(self) -> bool:
        return self.busy_box.isVisibleTo(self)

    def _held(self) -> bool:
        if self.polygon.connected:
            return True
        self.panel.window.say("The patterned light is still being found (about 25 s)..." if self._opening else
                              "Use here first (the patterned light view): it is not held.")
        return False

    def _said(self) -> None:
        if self._opening:
            return
        t = self.temperature_c if self.polygon.connected else None
        self.device_status.setText(self.polygon.status + (f"; {t} °C inside" if t is not None else ""))

    # ---------------------------------------------------------------- the pattern
    def make_steps(self) -> None:
        kind = self.source.currentText()
        note = ""
        try:
            if kind == "Full field":
                steps = patterns.full_field()
            elif kind == "Spot":
                steps = patterns.spot(self.spot_u.value() / 100, self.spot_v.value() / 100, self.spot_d.value() / 100)
            elif kind == "Grid":
                size = self._frame_size()
                if self.grid_on_picture.isChecked() and self.calibration.ready and size[0]:
                    cal = self.calibration.at_size(size)
                    steps = patterns.grid_on_picture(self.rows.value(), self.cols.value(), cal, size,
                                                     self.fill.value() / 100, self.round.isChecked())
                    note = ("Over the camera's picture. Part of the picture lies outside the field: cells there are "
                            "cut." if patterns.picture_outside_field(cal, size) else "Over the camera's picture.")
                else:
                    steps = patterns.grid(self.rows.value(), self.cols.value(), fill=self.fill.value() / 100,
                                          round_spots=self.round.isChecked())
                    note = ("Over the whole field: laying it over the camera's picture needs the camera and the "
                            "calibration." if self.grid_on_picture.isChecked() else "Over the whole field.")
            elif kind == "Line of spots":
                steps = patterns.line(self.l_u0.value() / 100, self.l_v0.value() / 100, self.l_u1.value() / 100,
                                      self.l_v1.value() / 100, self.l_n.value(), self.l_d.value() / 100)
            elif kind == "Image file":
                if not self.image_path.text().strip():
                    raise ValueError("choose an image first")
                steps = patterns.from_image(self.image_path.text().strip(), self.threshold.value(),
                                            self.invert.isChecked())
            elif kind == "Sequence file":
                if not self.seq_path.text().strip():
                    raise ValueError("choose a sequence first")
                steps, notes = patterns.read_seq(self.seq_path.text().strip())
                note = "; ".join(sorted(set(notes))[:3])
            else:
                steps = self.drawn_steps()
        except (OSError, ValueError) as e:
            self.pattern_note.setText(f"Not made: {e}.")
            return
        self.steps = steps
        self._show_check = False
        self._mask_images = {}
        self.list.clear()
        self.list.addItems([s.name for s in steps])
        if steps:
            self.list.setCurrentRow(0, QtCore.QItemSelectionModel.ClearAndSelect)
        self.pattern_note.setText(f"{len(steps)} step(s). {note}".strip())
        self.save()
        self.draw_outlines()

    def _chosen_rows(self) -> list[int]:
        """The steps chosen in the list (one, or several with Ctrl or Shift), in their order."""
        return sorted({self.list.row(item) for item in self.list.selectedItems()})

    def _choose_names(self, names: list[str]) -> None:
        """The steps called ``names`` chosen again (after the steps were made again), else the first."""
        wanted = set(names)
        rows = [k for k, s in enumerate(self.steps) if s.name in wanted] or ([0] if self.steps else [])
        self.list.blockSignals(True)
        self.list.clearSelection()
        for k in rows:
            self.list.item(k).setSelected(True)
        if rows:
            self.list.setCurrentRow(rows[0], QtCore.QItemSelectionModel.NoUpdate)
        self.list.blockSignals(False)
        self.draw_outlines()

    def project(self) -> None:
        """The steps chosen shown: one, or several lit together as one (``3 + 17``), nothing else lit."""
        rows = [r for r in self._chosen_rows() if 0 <= r < len(self.steps)]
        if not rows:
            self.panel.window.say("Make the steps first, then choose one (or several, with Ctrl).")
            return
        if not self._held():
            return
        if self.on_trigger.isChecked():
            self.project_on_triggers()
            return
        parts = [self.steps[r] for r in rows]
        step, on, n = patterns.together(parts), self.calibration.on_value, len(self.steps)
        owners = [p.name for p in parts for _ in p.shapes]             # which step each shape is of
        cal, um, key = self.calibration.at_size(self._frame_size()), self._um_per_px(), self._objective()
        pattern = self.source.currentText()
        self._wanted += 1
        ticket = self._wanted

        def work():
            if ticket != self._wanted:                 # a newer step asked for
                return None
            mask = step.mirrors()
            if not self.polygon.show(step, on, mask):
                return False
            when = time.time()                         # the moment the Polygon took it
            e = light_record.entry("shown", step, pattern, patterns.dose(step, mask, cal, um), key, on, when)
            if len(parts) > 1:
                for d, owner in zip(e.shapes, owners):
                    d["step"] = owner
            return e

        def done(e):
            if e is None:
                return
            if isinstance(e, light_record.Shown):
                area = f", {e.area_um2:,.0f} um^2" if e.area_um2 else ""
                if len(parts) == 1:
                    which = f"step {step.name} ({rows[0] + 1} of {n}"
                else:
                    names = [p.name for p in parts]
                    which = (f"{len(parts)} steps together ({', '.join(names)}" if len(names) <= 8 else
                             f"{len(parts)} steps together ({', '.join(names[:6])} ... {names[-1]}")
                self.shown_label.setText(f"Showing {which}; {e.field_fraction:.1%} of the field{area}) since "
                                         f"{time.strftime('%H:%M:%S')}." +
                                         ("" if self._calibrated_light() else
                                          " The light test has not been done: lit and unlit may be swapped."))
                self._lit = {p.name for p in parts}
                self._log(e)
                self.draw_outlines()
            else:
                why = e if isinstance(e, Exception) else self.polygon.status
                self.shown_label.setText(f"Step {step.name} was not shown: {why}.")

        self._async(work, done)

    def project_on_triggers(self) -> None:
        """The steps loaded, the patterned light stepping through them on its own trigger (step 1 on the first
        TTL after this, then 2...): logged as one sequence, so each light pulse is matched to its step. Every step,
        or only those chosen when several are."""
        self.auto.setChecked(False)
        rows = self._chosen_rows()
        steps = [self.steps[r] for r in rows] if len(rows) > 1 else list(self.steps)
        on = self.calibration.on_value
        cal, um, key = self.calibration.at_size(self._frame_size()), self._um_per_px(), self._objective()
        held_us = int(self.held_ms.value() * 1000)
        pattern = self.source.currentText()
        self._wanted += 1

        def work():
            masks = [s.mirrors() for s in steps]
            if not self.polygon.run_sequence(masks, on, held_us):
                return False
            when = time.time()
            doses = [{"step": s.name, **patterns.dose(s, m, cal, um)} for s, m in zip(steps, masks)]
            e = light_record.entry("sequence", None, pattern, {"field_fraction": max(d["field_fraction"]
                                                                                     for d in doses)}, key, on, when)
            e.step, e.shapes = " | ".join(s.name for s in steps), doses
            return e

        def done(e):
            if isinstance(e, light_record.Shown):
                self.shown_label.setText(f"Stepping through {len(steps)} step(s) on each trigger since "
                                         f"{time.strftime('%H:%M:%S')}: {e.step}. Off stops it.")
                self._lit = {s.name for s in steps}
                self._log(e)
                self.draw_outlines()
            else:
                why = e if isinstance(e, Exception) else self.polygon.status
                self.shown_label.setText(f"The steps were not loaded for the trigger: {why}.")

        self._async(work, done)

    def step_by(self, k: int, wrap: bool = True) -> None:
        n = len(self.steps)
        if not n:
            return
        row = self.list.currentRow() + k
        if not wrap and not 0 <= row < n:
            self.auto.setChecked(False)
            self.shown_label.setText(self.shown_label.text() + " The last step was shown: stepping stopped.")
            return
        self.list.setCurrentRow(row % n, QtCore.QItemSelectionModel.ClearAndSelect)     # that step alone, not added to those chosen
        self.project()

    def off(self) -> None:
        self.auto.setChecked(False)
        if not self.polygon.connected:
            return
        on = self.calibration.on_value
        self._wanted += 1

        def work():
            return light_record.entry("off", on_value=on) if self.polygon.off(on) else False

        def done(e):
            if isinstance(e, light_record.Shown):
                self.shown_label.setText(f"Off since {time.strftime('%H:%M:%S')}: no mirror lit.")
                self._lit = set()
                self._log(e)
                self.draw_outlines()

        self._async(work, done)

    def after_run(self, files) -> None:
        """A recording made here has ended: each file given the patterns its sweeps saw (``<file>.light.json``),
        and, unless unticked, nothing left lit."""
        if self.polygon.connected and self.off_after.isChecked() and self.polygon.shown is not None:
            self.off()
        folder = Path(self.panel.folder() or "")
        if not files or not (folder / light_record.LOG_NAME).is_file():
            return
        from ...io import load
        from ..worker import run

        def work():
            matched = []
            for f in files:
                try:
                    sweeps = light_record.for_recording(load(f), [folder], own=True)
                except Exception:                     # a file that cannot be read now is matched when analysed
                    continue
                if sweeps is not None:
                    matched.append((Path(f).name, sweeps))
            return matched

        def done(matched):
            if isinstance(matched, list) and matched:
                parts = []
                for name, sweeps in matched:
                    groups = light_record.by_step(sweeps)
                    parts.append(f"{name}: " + ", ".join(f"{k} ({len(v)})" for k, v in groups.items()))
                self.shown_label.setText(self.shown_label.text() + " The patterns are kept with the recording: "
                                         + "; ".join(parts) + ".")

        self._matching = run(work, done)

    def _auto(self, on: bool) -> None:
        self.timer.stop()
        if on:
            self.timer.start(int(self.every.value() * 1000))

    def _log(self, e: light_record.Shown) -> None:
        light_record.write(self.panel.folder() or "", e)

    # ---------------------------------------------------------------- drawn on the picture
    def _frame_size(self) -> tuple[int, int]:
        last = self.panel.last
        return (last.rgb.shape[1], last.rgb.shape[0]) if last is not None else (0, 0)

    def _um_per_px(self) -> float:
        """The camera's um a pixel now (its objective, its binning), 0 when not known."""
        scale = getattr(self.panel, "scale", None)
        try:
            return float(scale()[0]) if scale is not None else 0.0
        except (TypeError, ValueError, IndexError):
            return 0.0

    def _objective(self) -> str:
        mag = getattr(self.panel, "magnification", None)
        adapter = getattr(getattr(self.panel, "dialog", None), "adapter", None)
        return patterns.objective_key(mag, adapter.value() if adapter is not None else 1.0)

    def add_shape(self, kind: str, centre=None, size=None, name: str = "", angle: float = 0.0) -> pg.ROI | None:
        """A region on the picture: ``centre`` and ``size`` in its pixels (else in the middle, a sixth of it)."""
        w, h = self._frame_size()
        if not w:
            self.panel.window.say("Take the camera first: shapes are drawn on its picture.")
            return None
        size = list(size) if size is not None else [w * 0.15, w * 0.15]
        cx, cy = centre if centre is not None else (w / 2 + 20 * len(self.drawn), h / 2)
        pos = [cx - size[0] / 2, cy - size[1] / 2]
        pen = pg.mkPen("c", width=2)
        roi = pg.RectROI(pos, size, pen=pen) if kind == "rect" else pg.EllipseROI(pos, size, pen=pen)
        if angle:
            roi.setAngle(angle, center=(0.5, 0.5))
        roi.kind = kind
        roi.name = name or self.roi_name.text().strip() or f"Region {len(self.drawn) + 1}"
        if not name:
            self.roi_name.clear()
        self.panel.box.addItem(roi)
        self.drawn.append(roi)
        roi.sigRegionChangeFinished.connect(lambda *_: self._drawn_moved())
        return roi

    def click_centre(self) -> None:
        if not self._frame_size()[0]:
            self.panel.window.say("Take the camera first: the region is placed on its picture.")
            return
        if not self._um_per_px():
            self.panel.window.say("The camera's scale is not known at this objective: calibrate its length first "
                                  "(the camera's Adjust..., Length calibration), then the diameter in um can be used.")
            return
        self._clicking = True
        self.panel.window.say(f"Click the cell's centre on the picture: a round region {self.roi_um.value():g} um "
                              "across goes there.")

    def _clicked(self, event) -> None:
        """A click on the picture: a region's centre placed (after "Click the cell's centre"), else the step under
        it chosen (with Ctrl, added to those chosen or taken from them); a double-click shows it."""
        p = self.panel.box.mapSceneToView(event.scenePos())
        if self._clicking:
            self._clicking = False
            self.place_at(p.x(), p.y())
            event.accept()
            return
        if event.button() != QtCore.Qt.LeftButton or not (self._overlaid() and self.outlines.isChecked()):
            return
        row = self.step_at(p.x(), p.y())
        if row is None:
            return
        self.choose_at(row, add=bool(event.modifiers() & QtCore.Qt.ControlModifier))
        if event.double():
            self.project()

    def step_at(self, x: float, y: float) -> int | None:
        """The step lit at (x, y) of the picture (the last made, when several overlap), None when none is or the
        field's place is not known."""
        if not (self.steps and self.calibration.ready):
            return None
        u, v = self.calibration.at_size(self._frame_size()).field_of(np.array([x]), np.array([y]))
        for k in range(len(self.steps) - 1, -1, -1):
            if self.steps[k].covers(float(u[0]), float(v[0])):
                return k
        return None

    def choose_at(self, row: int, add: bool = False) -> None:
        """Step ``row`` chosen alone, or (``add``) added to those chosen, or taken from them when it was."""
        if not add:
            self.list.setCurrentRow(row, QtCore.QItemSelectionModel.ClearAndSelect)
            return
        item = self.list.item(row)
        self.list.blockSignals(True)
        item.setSelected(not item.isSelected())
        self.list.setCurrentRow(row, QtCore.QItemSelectionModel.NoUpdate)
        self.list.blockSignals(False)
        self.draw_outlines()

    def place_at(self, x: float, y: float) -> pg.ROI | None:
        """A round region centred on (x, y) of the picture, the diameter asked for across."""
        um = self._um_per_px()
        if not um:
            return None
        d = self.roi_um.value() / um
        roi = self.add_shape("ellipse", (x, y), (d, d))
        if roi is not None and self.source.currentText() == "Drawn on the picture" and self.calibration.ready:
            self.make_steps()
        return roi

    def add_control(self) -> pg.ROI | None:
        """A control region: the last region's size and shape, ``control_um`` to its right on the picture (dragged
        then to neuropil where no cell is lit): what it evokes in the cell is the light scattered onto it."""
        mains = [r for r in self.drawn if not getattr(r, "control", False)]
        if not mains:
            self.panel.window.say("Add the region on the cell first: the control is made beside it.")
            return None
        last = mains[-1]
        um = self._um_per_px()
        sw, sh = last.size()
        x, y = self._centre(last)
        gap = self.control_um.value() / um if um else 3 * max(sw, sh)
        roi = self.add_shape(last.kind, (x + gap, y), (sw, sh), f"{last.name} control", last.angle())
        if roi is not None:
            roi.control = True
            roi.setPen(pg.mkPen("m", width=2, style=QtCore.Qt.DashLine))
            if self.source.currentText() == "Drawn on the picture" and self.calibration.ready:
                self.make_steps()
        return roi

    @staticmethod
    def _centre(roi) -> tuple[float, float]:
        sw, sh = roi.size()
        p = roi.mapToParent(QtCore.QPointF(sw / 2, sh / 2))
        return float(p.x()), float(p.y())

    def _regions_path(self) -> Path | None:
        folder = self.panel.folder() or ""
        return Path(folder) / "polygon_regions.json" if folder and Path(folder).is_dir() else None

    def save_regions(self) -> Path | None:
        """The regions drawn, kept in the cell's folder: on the picture (pixels and um), on the field (when
        calibrated), with the objective and the picture's size."""
        path = self._regions_path()
        if path is None:
            self.panel.window.say("No cell's folder to keep the regions in: choose the cell first.")
            return None
        um, size = self._um_per_px(), self._frame_size()
        regions = []
        for roi in self.drawn:
            (x, y), (sw, sh) = self._centre(roi), roi.size()
            regions.append({"name": roi.name, "kind": roi.kind, "control": bool(getattr(roi, "control", False)),
                            "centre_px": [float(x), float(y)], "size_px": [float(sw), float(sh)],
                            "angle": float(roi.angle()),
                            "centre_um": [round(x * um, 2), round(y * um, 2)] if um else None,
                            "size_um": [round(sw * um, 2), round(sh * um, 2)] if um else None})
        body = {"objective": self._objective(), "picture_px": list(size), "um_per_px": um, "regions": regions,
                "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            path.write_text(json.dumps(body, indent=1), encoding="utf-8")
        except OSError as e:
            self.panel.window.say(f"The regions were not kept: {e}.")
            return None
        self.panel.window.say(f"{len(regions)} region(s) kept with the cell.")
        return path

    def load_regions(self) -> int:
        """The regions kept with this cell drawn again (scaled to the picture's binning now); their number."""
        path = self._regions_path()
        if path is None or not path.is_file():
            self.panel.window.say("No regions are kept with this cell yet.")
            return 0
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            self.panel.window.say(f"The cell's regions could not be read: {e}.")
            return 0
        w0, h0 = body.get("picture_px") or (0, 0)
        w, h = self._frame_size()
        if not w:
            self.panel.window.say("Take the camera first: the regions are drawn on its picture.")
            return 0
        k = w / w0 if w0 else 1.0
        self.clear_shapes()
        for r in body.get("regions", []):
            (x, y), (sw, sh) = r["centre_px"], r["size_px"]
            # a binned pixel's centre is half way along the pixels it sums
            roi = self.add_shape(r.get("kind", "ellipse"), ((x + 0.5) * k - 0.5, (y + 0.5) * k - 0.5),
                                 (sw * k, sh * k), r.get("name", ""), float(r.get("angle", 0.0)))
            if roi is not None and r.get("control"):
                roi.control = True
                roi.setPen(pg.mkPen("m", width=2, style=QtCore.Qt.DashLine))
        if body.get("objective") and body["objective"] != self._objective():
            now = self._objective() or "this objective"
            self.panel.window.say(f"These regions were drawn at {body['objective']}, not at {now}: check they still "
                                  "sit on the cell.")
        return len(self.drawn)

    def _drawn_moved(self) -> None:
        """A shape moved or resized on the picture: while drawn shapes are shown, shown again where they are now."""
        if self.source.currentText() != "Drawn on the picture" or not self.calibration.ready:
            return
        names = [self.steps[r].name for r in self._chosen_rows() if r < len(self.steps)]
        showing = self.polygon.connected and self.polygon.shown is not None
        self.make_steps()
        if self.steps:
            self._choose_names(names)
            if showing:
                self.project()

    def clear_shapes(self) -> None:
        for roi in self.drawn:
            self.panel.box.removeItem(roi)
        self.drawn = []

    def drawn_steps(self) -> list[patterns.Step]:
        if not self.calibration.ready:
            why = self.calibrations.refusal(self._objective())
            raise ValueError(why if self.calibrations.maps else "calibrate first: the shapes drawn are found on the "
                             "patterned light's field through it")
        if not self.drawn:
            raise ValueError("add a shape on the picture first")
        shapes = []
        for roi in self.drawn:
            x, y = roi.pos()
            sw, sh = roi.size()
            a = np.radians(roi.angle())
            if roi.kind == "rect":
                local = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], float)
            else:
                t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                local = np.column_stack([sw / 2 + sw / 2 * np.cos(t), sh / 2 + sh / 2 * np.sin(t)])
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            cam = local @ rot.T + [x, y]
            u, v = self.calibration.at_size(self._frame_size()).field_of(cam[:, 0], cam[:, 1])
            shapes.append(patterns.Shape("polygon", tuple(np.column_stack([u, v]).ravel().tolist())))
        if self.each_step.isChecked():
            return [patterns.Step(getattr(roi, "name", "") or str(k + 1), [s])
                    for k, (roi, s) in enumerate(zip(self.drawn, shapes))]
        return [patterns.Step("Drawn", shapes)]

    # ---------------------------------------------------------------- outlines over the picture
    def draw_outlines(self) -> None:
        for item in self.outline_items:
            self.panel.box.removeItem(item)
        self.outline_items = []
        if not (self.outlines.isChecked() and self.calibration.ready and self._overlaid()):
            return
        cal = self.calibration.at_size(self._frame_size())     # the picture's binning now
        add = self._overlay
        # the field's edge, dashed (outside the picture when the field is larger than it: seen zoomed out)
        x, y = cal.camera_of(np.array([0.0, 1.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0, 0.0]))
        add(pg.PlotCurveItem(x, y, pen=pg.mkPen((128, 128, 128, 220), width=1, style=QtCore.Qt.DashLine)))
        for x0, y0, x1, y1 in (cal.check if self._show_check else []):   # the check: where each spot fell (+), from
            add(pg.ScatterPlotItem([x1], [y1], symbol="o", size=10, pen=pg.mkPen((0, 255, 255), width=1),
                                   brush=None))                # where the map put it (o)
            add(pg.PlotCurveItem([x1, x0], [y1, y0], pen=pg.mkPen((0, 255, 255), width=1)))
            add(pg.ScatterPlotItem([x0], [y0], symbol="+", size=8, pen=pg.mkPen((0, 255, 255), width=1)))
        if not self.steps:
            return
        chosen = set(self._chosen_rows())
        many = len(self.steps) > 100
        for k, step in enumerate(self.steps):
            bright, lit = k in chosen, step.name in self._lit
            if many and not (bright or lit):
                continue
            colour = LIT if lit else CHOSEN
            pen = pg.mkPen(colour if bright or lit else colour + (90,), width=2 if bright or lit else 1)
            if step.mask is not None and (bright or lit):
                add(self._mask_overlay(step, cal, colour))
            for shape in step.shapes:
                pts = shape.outline()
                x, y = cal.camera_of(pts[:, 0], pts[:, 1])
                if lit:                                          # what is lit now, filled
                    item = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([QtCore.QPointF(a, b)
                                                                           for a, b in zip(x, y)]))
                    item.setPen(pen)
                    item.setBrush(pg.mkBrush(colour + (70,)))
                    add(item)
                else:
                    add(pg.PlotCurveItem(x, y, pen=pen))
            if step.shapes and (len(self.steps) <= 25 or bright or lit):
                cu, cv = step.shapes[0].centre()
                x, y = cal.camera_of(np.array([cu]), np.array([cv]))
                text = pg.TextItem(step.name, color=colour if bright or lit else colour + (140,), anchor=(0.5, 0.5))
                text.setPos(float(x[0]), float(y[0]))
                add(text)

    def _overlaid(self) -> bool:
        """Whether the steps are drawn over the picture: while this panel is shown, or while the patterned light is
        held (its view hidden so the camera's picture can be enlarged, the picture is where steps are chosen)."""
        return self.isVisible() or self.polygon.connected

    def _overlay(self, item) -> None:
        # not counted in the picture's bounds: Fit fits the picture, not a field larger than it
        self.panel.box.addItem(item, ignoreBounds=True)
        self.outline_items.append(item)

    def _mask_overlay(self, step: patterns.Step, cal: patterns.Calibration, colour) -> pg.ImageItem:
        """An image step's mirrors laid over the picture through the map, see-through (every 2nd mirror: a mirror
        is far smaller than the outline's line). Made once per step and colour."""
        key = (id(step), colour)
        rgba = self._mask_images.get(key)
        if rgba is None:
            m = step.mirrors()[::2, ::2]
            rgba = np.zeros(m.shape + (4,), np.uint8)
            rgba[m] = colour + (110,)
            self._mask_images[key] = rgba
        item = pg.ImageItem(rgba, axisOrder="row-major")
        # a mirror (column c, row r of the image, every 2nd) at about (u, v) = (2c / 912, 2r / 1140) of the field
        a = np.vstack([np.asarray(cal.to_field, float), [0.0, 0.0, 1.0]])
        inv = np.linalg.inv(a)
        su, sv = 2.0 / patterns.COLUMNS, 2.0 / patterns.ROWS
        item.setTransform(QtGui.QTransform(inv[0, 0] * su, inv[1, 0] * su, inv[0, 1] * sv, inv[1, 1] * sv,
                                           inv[0, 2], inv[1, 2]))
        return item

    # ---------------------------------------------------------------- the light test and calibration
    def _calibrated_light(self) -> bool:
        return self.calibration.light_tested

    def calibrate(self) -> None:
        if self._cal is not None:
            return
        if self.panel._thread is None or self.panel.last is None:
            self.panel.window.say("Take the camera first: the calibration reads its picture.")
            return
        if not self._held():
            return
        self.auto.setChecked(False)
        self._wanted += 1                             # any step still waiting to be sent is not
        # each test picture: its name, the image sent and, for a spot, where it is on the field
        plan = [("none", np.zeros((patterns.ROWS, patterns.COLUMNS), np.uint8), None),
                ("all", np.full((patterns.ROWS, patterns.COLUMNS), 255, np.uint8), None)]
        self._cal = {"plan": plan, "k": -1, "pictures": {}, "spots": [], "off": None, "ring": 0}
        self.cal_button.setEnabled(False)
        self.cal_status.setText("Light test: nothing shown, then everything...")
        self._next_picture()
        self._poll.start()

    def _settle(self) -> float:
        t = self.panel._thread
        frame_s = (t.exposure_ms or self.panel.exposure.value()) / 1000.0 * max(1, t.average) if t else 0.1
        return max(SETTLE_S, 3 * frame_s)

    def _next_picture(self) -> None:
        c = self._cal
        c["k"] += 1
        if c["k"] >= len(c["plan"]):
            return
        name, img, _ = c["plan"][c["k"]]
        c["sent_at"] = None                          # until the Polygon has it

        def done(ok):
            if self._cal is not c:
                return
            if ok is not True:
                self._end_calibration(f"The test picture was not taken ({self.polygon.status}).")
                return
            c["sent_at"], c["frames_at"] = time.monotonic(), self.panel._shown
            c["seen"], c["best"], c["dim"] = self.panel._shown, None, None

        self._async(lambda: self.polygon.send(img), done)

    def _calibration_tick(self) -> None:
        c = self._cal
        if c is None:
            self._poll.stop()
            return
        if self.panel.last is None or self.panel._thread is None:
            self._end_calibration("The camera stopped: calibration not finished.")
            return
        if c.get("sent_at") is None:
            return
        if time.monotonic() - c["sent_at"] < self._settle() or self.panel._shown < c["frames_at"] + 2:
            return
        name, _, uv = c["plan"][c["k"]]
        if self.pulsed.isChecked():
            picture = self._pulse_picture(c)
            if picture is None:
                return
        else:
            picture = self.panel.last.rgb[:, :, 0].astype(np.float32)
        c["pictures"][name] = picture
        if name == "all":
            self._light_tested()
        elif name.startswith("spot"):
            found = patterns.find_spot(c["pictures"][name], c["off"])
            if found is not None:
                c["spots"].append((found, uv))
        elif name.startswith("check"):
            found = patterns.find_spot(c["pictures"][name], c["off"])
            c["checks"].append((found, uv))
        if name not in ("none", "all"):
            c["pictures"].pop(name, None)             # a picture is 11 MB in floats: only the dark one is kept
        if self._cal is None:
            return
        if c["k"] + 1 >= len(c["plan"]):
            if "checks" in c:
                self._checked()
            else:
                self._fit()
            return
        self._next_picture()

    def _pulse_picture(self, c: dict) -> np.ndarray | None:
        """Pulsed light: every new frame looked at, the brightest kept; the picture once a frame clearly brighter than
        the darkest has come (a pulse caught, and a frame after it), or after PULSE_WAIT_S (no pulse lit this one:
        its brightest frame is its dark one). None while waiting."""
        if self.panel._shown > c["seen"]:
            c["seen"] = self.panel._shown
            g = self.panel.last.rgb[:, :, 0].astype(np.float32)
            m = float(g.mean())
            if c["best"] is None or m > c["best"][0]:
                c["best"], c["after"] = (m, g), 0
            else:
                c["after"] = c.get("after", 0) + 1
            if c["dim"] is None or m < c["dim"]:
                c["dim"] = m
        waited = time.monotonic() - c["sent_at"]
        caught = c["best"] is not None and c["best"][0] - c["dim"] > FLASH_GREY and c.get("after", 0) >= 1
        if caught or waited > PULSE_WAIT_S:
            return c["best"][1] if c["best"] is not None else self.panel.last.rgb[:, :, 0].astype(np.float32)
        step = c["k"] + 1
        self.cal_status.setText(f"Test picture {step} of {len(c['plan'])}: waiting for a pulse of the protocol "
                                f"({waited:.0f} s of at most {PULSE_WAIT_S:.0f} s)...")
        return None

    def _light_tested(self) -> None:
        c = self._cal
        change = patterns.lit_difference(c["pictures"]["all"], c["pictures"]["none"])
        noise = float(np.std(c["pictures"]["none"] - c["pictures"]["none"].mean())) * 0.05 + 1.0
        if abs(change) < max(2.0, noise):
            self._end_calibration("No change on the picture between nothing and everything shown: is the "
                                  "LED on (with the light in pulses: the protocol playing meanwhile), the light "
                                  "path open and a fluorescent sample in view?")
            return
        on = 255 if change > 0 else 0
        self.calibration.on_value = on
        self.swap.blockSignals(True)
        self.swap.setChecked(on == 0)
        self.swap.blockSignals(False)
        c["off"] = c["pictures"]["none" if on == 255 else "all"]
        self.cal_status.setText(f"Light test: a mirror is lit by {on} (the picture {abs(change):.0f} grey levels "
                                "brighter with everything shown). Now the spots...")
        self.calibration.light_tested = True
        self._plan_spots(patterns.CALIBRATION_SPOTS, patterns.CALIBRATION_DIAMETER)

    def _plan_spots(self, spots, diameter: float, kind: str = "spot") -> None:
        c, on = self._cal, self.calibration.on_value
        start = sum(1 for name, _, _ in c["plan"] if name.startswith(kind))
        for k, (u, v) in enumerate(spots):
            step = patterns.spot(u, v, diameter)[0]
            c["plan"].append((f"{kind}{start + k}", patterns.image(step.mirrors(), on), (u, v)))

    def _fit(self) -> None:
        c = self._cal
        cam_found = np.array([p for p, _ in c["spots"]], float).reshape(-1, 2)
        if patterns.spread_px(cam_found) < 3.0 and c["ring"] < len(patterns.CLOSER_RINGS):
            # the field larger than the picture: the outer spots fell off it; spots nearer the centre then
            spots, diameter = patterns.ring(patterns.CLOSER_RINGS[c["ring"]])
            c["ring"] += 1
            self._plan_spots(spots, diameter)
            self.cal_status.setText(f"{len(c['spots'])} spot(s) found on the picture: the field is larger than the "
                                    "picture. Now spots nearer its centre...")
            self._next_picture()
            return
        if patterns.spread_px(cam_found) < 3.0:
            planned = sum(1 for name, _, _ in c["plan"] if name.startswith("spot"))
            self._end_calibration(f"Only {len(c['spots'])} of {planned} spots were found on the picture, too few "
                                  "to place the field: the light test is kept, the field's place is not. Is the "
                                  "picture in focus on the fluorescent sample, and the light bright enough? Then "
                                  "calibrate again.", keep_light=True)
            return
        cam = np.array([p for p, _ in c["spots"]])
        uv = np.array([q for _, q in c["spots"]])
        to_field, residual = patterns.fit_affine(cam, uv)
        self.calibration.to_field = to_field
        self.calibration.residual_px = residual
        self.calibration.camera_size = self._frame_size()
        self.calibration.objective = self._objective()
        self.calibration.um_per_px = self._um_per_px()
        self.calibration.check = []
        # then the check: spots the fit did not use, laid over the picture (so the camera sees them all), each
        # found and compared with where the map puts it
        c["checks"] = []
        spots, diameter = patterns.check_spots_on_picture(self.calibration, self._frame_size())
        c["check_total"] = len(spots)
        if not spots:
            self._checked()
            return
        self._plan_spots(spots, diameter, "check")
        self.cal_status.setText(f"Fitted with {len(c['spots'])} spots (within {residual:.1f} px); now the check "
                                f"grid of {len(spots)} spots...")
        self._next_picture()

    def _checked(self) -> None:
        """The check grid's spots compared with where the map puts them: the errors in um (pixels when the camera's
        scale is not known), kept with the calibration and drawn on the picture."""
        c, cal = self._cal, self.calibration
        rows = []
        for found, (u, v) in c["checks"]:
            if found is None:
                continue
            x, y = cal.camera_of(np.array([u]), np.array([v]))
            rows.append([found[0], found[1], float(x[0]), float(y[0])])
        cal.check = rows
        self._show_check = True
        n, total = len(rows), c.get("check_total", len(patterns.CHECK_SPOTS))
        base = (f"Calibrated with {len(c['spots'])} spots at {cal.objective or 'this objective'}: a mirror is lit by "
                f"{cal.on_value}. ")
        if not n:
            self._end_calibration(base + "None of the check grid's spots was found on the picture: the fit is kept "
                                  "but not checked.", keep_light=True)
            return
        err = cal.check_errors_px()
        if cal.um_per_px:
            e = err * cal.um_per_px
            verdict = ("within the target" if e.max() <= patterns.CHECK_TARGET_UM else
                       f"over the {patterns.CHECK_TARGET_UM:g} um target: calibrate again (focus on the slide, "
                       "brighter spots) before relying on regions of one cell")
            text = (f"Check: {n} of {total} spots found, each within {e.max():.1f} um of where the map puts it (mean "
                    f"{e.mean():.1f} um), {verdict}.")
        else:
            text = (f"Check: {n} of {total} spots found, each within {err.max():.1f} px of where the map puts it "
                    f"(mean {err.mean():.1f} px); calibrate the camera's length to read it in um.")
        self._end_calibration(base + text, keep_light=True)

    def _end_calibration(self, text: str, keep_light: bool = False) -> None:
        self._poll.stop()
        on = self.calibration.on_value
        self._cal = None
        if self.polygon.connected:
            self._async(lambda: self.polygon.off(on))
            self.shown_label.setText("Off: no mirror lit.")
            self._lit = set()
        self.cal_button.setEnabled(True)
        self.cal_status.setText(text)
        if keep_light:
            self.calibrations.keep(self.calibration)
            self.save()
            if self.calibration.ready and self.source.currentText() == "Grid" and self.grid_on_picture.isChecked():
                self.make_steps()                     # the grid laid over the picture now that it can be
        self.draw_outlines()
        self._said()

    # ---------------------------------------------------------------- kept for the user
    def _cal_text(self) -> str:
        c = self.calibration
        light = (f"a mirror is lit by {c.on_value}" if c.light_tested else
                 f"lit by {c.on_value} assumed (the light test has not been done)")
        if not c.ready:
            why = self.calibrations.refusal(self._objective())
            return f"Not calibrated here: {light}; {why}." if self.calibrations.maps else \
                f"Not calibrated: {light}; the field's place is not known yet."
        text = (f"Calibrated at {c.objective or 'an objective not noted'}: {light}; the field's place fitted within "
                f"{c.residual_px:.1f} px at {c.camera_size[0]} x {c.camera_size[1]}")
        if c.check:
            err = c.check_errors_px()
            text += (f"; checked within {err.max() * c.um_per_px:.1f} um" if c.um_per_px else
                     f"; checked within {err.max():.1f} px")
        return text + "."

    def _use_objective(self) -> None:
        """The objective (or camera adapter) changed: its own calibration used, or none."""
        stopped = self._cal is not None               # its spots were found through the other objective
        if stopped:
            self._end_calibration("")
        self.calibration = self.calibrations.at(self._objective())
        self.cal_status.setText(("The objective changed during the calibration: stopped. " if stopped else "")
                                + self._cal_text())
        self.draw_outlines()

    def load(self) -> None:
        main = widgets._main_window(self.panel)
        if main is not None and getattr(main, "persist", False):
            try:
                self.calibrations = patterns.Calibrations.from_dict(json.loads(str(main._settings().value(
                    "polygon/calibration", "{}") or "{}")))
            except (ValueError, TypeError, AttributeError):
                self.calibrations = patterns.Calibrations()
        self.calibration = self.calibrations.at(self._objective())
        self.cal_status.setText(self._cal_text())
        self.swap.blockSignals(True)
        self.swap.setChecked(self.calibration.on_value == 0)
        self.swap.blockSignals(False)

    def _swapped(self, on: bool) -> None:
        """Lit by 0 or by 255, set by hand: the step shown is sent again so the change is seen at once."""
        self.calibration.on_value = self.calibrations.on_value = 0 if on else 255
        self.save()
        self.cal_status.setText(self._cal_text() + " (lit by " + str(self.calibration.on_value) + ", set by hand)")
        if self.polygon.connected and self.polygon.shown is not None:
            self.project()

    def save(self) -> None:
        main = widgets._main_window(self.panel)
        if main is not None and getattr(main, "persist", False):
            main._settings().setValue("polygon/calibration", json.dumps(self.calibrations.to_dict()))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.draw_outlines()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.draw_outlines()                          # kept while the patterned light is held, else taken away

    def release(self) -> None:
        """Lets the Polygon go (Let go, or Clampwise closing): PolyScan can take it again."""
        self.auto.setChecked(False)
        self.timer.stop()
        self._poll.stop()
        if self._cal is not None:
            self._cal = None
            self.cal_button.setEnabled(True)
        self._wanted += 1
        if self.polygon.connected or self._opening:
            self.device_status.setText("")
            self._busy_start("Letting the patterned light go", 4.0)
            on = self.calibration.on_value

            def work():
                # no mirror left lit: the Polygon keeps showing its last picture after it is let go
                off = bool(self.polygon.connected and self.polygon.off(on))
                when = time.time()
                self.polygon.close()
                return off, when

            def done(result):
                self._busy_end()
                self._lit = set()
                self.draw_outlines()
                off, when = result if isinstance(result, tuple) else (False, time.time())
                if off:
                    self._log(light_record.entry("off", on_value=on, when=when))
                self._log(light_record.entry("let go"))
                self.shown_label.setText("Let go" + (" with no mirror lit." if off else "."))

            self._async(work, done)
        self.released.emit()
