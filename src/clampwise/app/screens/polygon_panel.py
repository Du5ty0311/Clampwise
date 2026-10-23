"""The Polygon in Live's side panel, under the camera's controls (Use here, PolyScan closed): where the light goes on
the slice, while Clampex times the light itself.

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
EXPOSE_STEADY_S = 2.0          # the calibration's own exposure: unchanged this long counts as settled
EXPOSE_MAX_S = 15.0            # (or after this long, settled or not)
LOOK_S = 1.5                   # not knowing how the LED is on: how long the first test pictures are watched for a pulse
LED_MODES = {"auto": "Find out by itself (Automatic)", "steady": "On steadily (switched on at its driver)",
             "pulsed": "Only in a protocol's pulses (play it during the calibration)"}
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
    changed = QtCore.Signal()                        # what the mirrors show, the steps or the ones chosen changed
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
        self.shown_mask: np.ndarray | None = None     # the mirrors lit now (ROWS x COLUMNS), None: none
        self._mask_images: dict = {}                  # an image step's mirrors as a picture to lay over the camera's
        lay = box("v", "s")                           # in a section of Live's side panel, which has the margins
        # what the mirrors do now, always in view, and the two buttons that change it: the light itself is still
        # switched by its LED driver, from the acquisition program, as the protocol says
        self.state = widgets.Badge("muted", "Not held")
        self.state.setWordWrap(True)
        self.state.setToolTip("What the mirrors show. Mirrors on: the slice is lit there whenever the LED fires (the "
                              "LED is switched by its driver, from the acquisition, as your protocol says). Mirrors "
                              "off: nothing is lit, whatever the LED does.")
        lay.addWidget(self.state)
        mirrors = box("h", "s")
        self.on_button = widgets.primary("Mirrors on", self.project, "The chosen step(s) on the mirrors (lit whenever "
                                                                     "the LED fires); a double-click on a step does it "
                                                                     "too")
        self.off_button = widgets.button("All mirrors off", self.off, "No mirror lit: nothing reaches the slice")
        mirrors.addWidget(self.on_button)
        mirrors.addWidget(self.off_button)
        mirrors.addStretch(1)
        lay.addLayout(mirrors)
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
        # 1, what the regions need: the camera's picture, and a calibration at the objective in use
        lay.addWidget(label("1 · Set up", "section"))
        ready = QtWidgets.QGridLayout()
        ready.setHorizontalSpacing(theme.SPACE["s"])
        ready.setVerticalSpacing(theme.SPACE["xs"])
        ready.setColumnStretch(0, 1)
        self.camera_ready = widgets.Badge("muted", "Camera")
        self.camera_ready.setWordWrap(True)
        self.camera_button = widgets.button("Use the camera", self._take_camera, "Regions are drawn on its picture "
                                            "and the calibration reads it", quiet=True)
        self.cal_ready = widgets.Badge("muted", "Not calibrated")
        self.cal_ready.setWordWrap(True)
        self.cal_button = widgets.button("Calibrate...", self.open_calibration, "Where the mirrors' field lies on "
                                         "the camera's picture, at this objective: a few minutes with a fluorescent "
                                         "slide", quiet=True)
        ready.addWidget(self.camera_ready, 0, 0)
        ready.addWidget(self.camera_button, 0, 1)
        ready.addWidget(self.cal_ready, 1, 0)
        ready.addWidget(self.cal_button, 1, 1)
        lay.addLayout(ready)
        self.take_camera = None                       # set by Live: takes the camera (its Use here)
        self.cal_status = label("", "muted", wrap=True)          # the calibration's words (its window shows them)
        self.cal_status.hide()
        # 2, the pattern
        lay.addWidget(label("2 · Pattern", "section"))
        src = box("h", "s")
        self.source = QtWidgets.QComboBox()
        self.source.addItems(SOURCES)
        src.addWidget(self.source, 1)
        src.addWidget(widgets.button("Make the steps", self.make_steps, "The pattern's steps, from the settings "
                                                                        "below"))
        lay.addLayout(src)
        self.pages = QtWidgets.QStackedWidget()
        self._build_pages()
        self.source.currentIndexChanged.connect(self._page)
        lay.addWidget(self.pages)
        self.pattern_note = label("", "muted", wrap=True)
        lay.addWidget(self.pattern_note)
        # 3, the steps: chosen here or on the picture, put on the mirrors
        lay.addWidget(label("3 · Steps", "section"))
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(90)
        self.list.setMaximumHeight(180)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setToolTip("Ctrl+click or Shift+click to choose several: Mirrors on lights them together. "
                             "A double-click puts a step on the mirrors.")
        self.list.itemDoubleClicked.connect(lambda *_: self.project())
        self.list.itemSelectionChanged.connect(self.draw_outlines)
        self.list.itemSelectionChanged.connect(self.changed.emit)
        lay.addWidget(self.list)
        go = box("h", "s")
        go.addWidget(widgets.button("< Previous", lambda: self.step_by(-1), "The step before, on the mirrors",
                                    quiet=True))
        go.addWidget(widgets.button("Next >", lambda: self.step_by(1), "The next step, on the mirrors", quiet=True))
        go.addStretch(1)
        lay.addLayout(go)
        legend = label("On the picture: yellow, chosen (click one to choose it, Ctrl+click for more, double-click to "
                       "light it); blue and filled, on the mirrors now; dashed, the edge of the field.", "muted",
                       wrap=True)
        lay.addWidget(legend)
        self.shown_label = label("", "muted", wrap=True)     # the last change's details (the banner says the state)
        lay.addWidget(self.shown_label)
        # what is rarely changed, folded
        more = widgets.Advanced("More options")
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
        more.body_layout.addLayout(go)
        self.off_after = QtWidgets.QCheckBox("All mirrors off after each recording")
        self.off_after.setChecked(True)
        self.off_after.setToolTip("When a recording made here ends, no mirror is left lit: the next run never "
                                  "lights the slice with a pattern left from this one")
        more.add(self.off_after)
        self.on_trigger = QtWidgets.QCheckBox("Step on each trigger (Trig In)")
        self.on_trigger.setToolTip("Mirrors on loads every step into the patterned light, and each TTL at its Trig In "
                                   "shows the next one: pulse 1 of a sweep on one region, pulse 2 on another, timed "
                                   "by the protocol's digital output (sent a few ms before each light pulse). All "
                                   "mirrors off stops it. Needs the digitizer's digital output wired to Trig In.")
        more.add(self.on_trigger)
        trig = box("h", "s")
        self.held_ms = _pct(50.0, 1.0, 10000.0, 5.0, " ms")
        self.held_ms.setToolTip("How long each step stays after its trigger: longer than the light pulse")
        trig.addWidget(label("Each step held", "muted"))
        trig.addWidget(self.held_ms)
        trig.addStretch(1)
        more.body_layout.addLayout(trig)
        self.outlines = QtWidgets.QCheckBox("Outlines on the picture")
        self.outlines.setChecked(True)
        self.outlines.setToolTip("The steps' shapes drawn over the camera's picture (the one shown brighter)")
        self.outlines.toggled.connect(lambda *_: self.draw_outlines())
        more.add(self.outlines)
        self.swap = QtWidgets.QCheckBox("Lit by 0 (swap lit and unlit)")
        self.swap.setToolTip("Which byte turns a mirror onto the slice: the calibration finds it; set it here by hand "
                             "only when it cannot run. Shown again at once.")
        self.swap.toggled.connect(self._swapped)
        more.add(self.swap)
        self.device_status = label(self.polygon.status, "muted", wrap=True)
        more.add(self.device_status)
        lay.addWidget(more)
        self.setLayout(lay)                           # in Live's side panel, which scrolls (no scroll area inside it)
        self.calibration_window: CalibrationWindow | None = None
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._calibration_tick)
        self.led = "auto"                             # how the LED is on while calibrating: found out, or as said
        self.load()
        self._page(0)
        self.show_state()
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

        try:
            self._pool.submit(run)
        except RuntimeError:                         # the worker already gone (Python closing): done here, so the
            run()                                    # mirrors are still put off and the device let go

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
        self.show_state()

    def busy(self) -> bool:
        return self.busy_box.isVisibleTo(self)

    # ---------------------------------------------------------------- what the mirrors do, and what is ready
    def _shown(self, mask) -> None:
        """The mirrors lit now (a bool ROWS x COLUMNS array), or None when none is."""
        self.shown_mask = None if mask is None or not np.any(mask) else np.asarray(mask, bool)
        self.changed.emit()

    def _set_lit(self, names: set[str]) -> None:
        self._lit = set(names)
        self.draw_outlines()
        self.show_state()

    def show_state(self) -> None:
        """The mirrors' state in words (on, and which; off; calibrating; not held) and the set-up's two rows: the
        camera held, the objective in use calibrated (when, how well)."""
        if self._opening:
            self.state.set("muted", "Finding the patterned light (about 25 s)...")
        elif not self.polygon.connected:
            self.state.set("muted", "Not held: Use here, on this section's header")
        elif self._cal is not None:
            self.state.set("warn", "Calibrating: test patterns on the mirrors")
        elif self._lit:
            names = sorted(self._lit)
            words = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
            self.state.set("warn", f"Mirrors ON: {words}. Lit whenever the LED fires.")
        else:
            self.state.set("ok", "Mirrors off: nothing is lit")
        held = self.polygon.connected
        self.on_button.setEnabled(held and self._cal is None and not self.busy())
        self.off_button.setEnabled(held and not self.busy())
        camera = getattr(self.panel, "_thread", None) is not None
        self.camera_ready.set("ok" if camera else "warn", "Camera held" if camera else "Camera not held")
        self.camera_button.setVisible(not camera)
        status, words = self.calibration_words()
        self.cal_ready.set(status, words, self._cal_text())
        self.cal_button.setText("Calibrate again..." if self.calibration.ready else "Calibrate...")
        self.changed.emit()

    def calibration_words(self) -> tuple[str, str]:
        """The objective in use's calibration in a few words: ok (checked within the target), warn (none, off the
        target, or old enough to check again)."""
        c, key = self.calibration, self._objective() or "this objective"
        if not c.ready:
            return "warn", f"Not calibrated at {key}"
        age = patterns.age_words(c.measured)
        words = f"Calibrated at {key}" + (f", {age.split(' (')[0]}" if age else "")
        if c.check and c.um_per_px:
            worst = float(c.check_errors_px().max() * c.um_per_px)
            words += f", within {worst:.1f} um"
            if worst > patterns.CHECK_TARGET_UM:
                return "warn", words + f" (over {patterns.CHECK_TARGET_UM:g} um: calibrate again)"
        return ("warn" if "check it again" in age else "ok"), words

    def _take_camera(self) -> None:
        if self.take_camera is not None:
            self.take_camera()

    def _page(self, index: int) -> None:
        """The chosen pattern's settings, the stack as tall as they are (not as the tallest page)."""
        for k in range(self.pages.count()):
            self.pages.widget(k).setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
                                               if k == index else QtWidgets.QSizePolicy.Ignored)
        self.pages.setCurrentIndex(index)
        self.pages.adjustSize()

    def open_calibration(self) -> None:
        if self.calibration_window is None:
            self.calibration_window = CalibrationWindow(self)
        self.calibration_window.refresh()
        self.calibration_window.show()
        self.calibration_window.raise_()
        self.calibration_window.activateWindow()

    def calibration_progress(self) -> tuple[int, int]:
        """The calibration's test pictures done and to do (the plan grows as it goes: spots nearer the centre, the
        check grid), (0, 0) when none is under way."""
        c = self._cal
        if c is None:
            return 0, 0
        to_do = len(c["plan"]) + (0 if "checks" in c else len(patterns.CHECK_SPOTS))
        return max(c["k"], 0), max(to_do, len(c["plan"]), 1)

    def summary(self) -> tuple[str, str, str]:
        """How the patterned light stands in a few words (status, words, detail): what a folded section's header
        shows. Lit is said first; a map missing for the objective in use is a warning (regions would land
        elsewhere)."""
        if self._opening:
            return "muted", "finding it...", "its driver looks through every USB 3 port (about 25 s)"
        if not self.polygon.connected:
            return "muted", "not held", ""
        if not self.calibration.ready:
            return "warn", f"not calibrated at {self._objective() or 'this objective'}", self.cal_status.text()
        lit = sorted(self._lit)
        if not lit:
            return "ok", "nothing lit", self.shown_label.text()
        words = ", ".join(lit[:3]) + (f" +{len(lit) - 3}" if len(lit) > 3 else "")
        return "ok", f"{words} lit", self.shown_label.text()

    def _held(self) -> bool:
        if self.polygon.connected:
            return True
        self.panel.window.say("The patterned light is still being found (about 25 s)..." if self._opening else
                              "Use here first (on the Patterned light section): it is not held.")
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
        self.changed.emit()

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

        sent = {}

        def work():
            if ticket != self._wanted:                 # a newer step asked for
                return None
            mask = sent["mask"] = step.mirrors()
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
                self._shown(sent.get("mask"))
                self._set_lit({p.name for p in parts})
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

        sent = {}

        def work():
            masks = [s.mirrors() for s in steps]
            sent["mask"] = np.logical_or.reduce(masks)
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
                                         f"{time.strftime('%H:%M:%S')}: {e.step}. All mirrors off stops it.")
                self._shown(sent.get("mask"))
                self._set_lit({s.name for s in steps})
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
                self._shown(None)
                self._set_lit(set())
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
        following = getattr(getattr(self.panel, "auto", None), "isChecked", lambda: False)()
        if following:                                 # the exposure first, for every mirror on (a mirror sends back
            every = np.ones((patterns.ROWS, patterns.COLUMNS), bool)        # far more than a cell), then held
            plan.insert(0, ("expose", patterns.image(every, self.calibration.on_value), None))
        else:                                         # the test pictures are compared: one exposure for them all
            self._hold_exposure(True)
        self._cal = {"plan": plan, "k": -1, "pictures": {}, "spots": [], "off": None, "ring": 0,
                     "pulsed": {"steady": False, "pulsed": True}.get(self.led)}       # None: found out
        self.show_state()
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
            self._shown(img == self.calibration.on_value)

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
        if name == "expose":                          # auto exposure settling on every mirror on: then held
            ms, now = self.panel.exposure.value(), time.monotonic()
            if ms != c.get("exposure_ms"):
                c["exposure_ms"], c["exposure_at"] = ms, now
            if now - c["exposure_at"] < EXPOSE_STEADY_S and now - c["sent_at"] < EXPOSE_MAX_S:
                self.cal_status.setText(f"Setting the camera's exposure with every mirror on: {ms:.3g} ms...")
                return
            self._hold_exposure(True)
            self._next_picture()
            return
        picture = self._test_picture(c, name)
        if picture is None:
            return
        c["pictures"][name] = picture
        if name == "all":
            self._light_tested()
        elif name == "half":
            c["sharpness"] = patterns.sharpness(picture - c["off"])
        elif name.startswith("spot"):
            found = patterns.find_spot(c["pictures"][name], c["off"])
            if found is not None:
                c["spots"].append((found, uv))
        elif name.startswith("check"):
            found = patterns.find_spot(c["pictures"][name], c["off"])
            c["checks"].append((found, uv))
        if name not in ("none", "all"):                 # "half" too: only its sharpness is kept
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

    def _test_picture(self, c: dict, name: str) -> np.ndarray | None:
        """The picture of the test pattern on the mirrors, None while waiting for it. The LED on steadily: the frame
        once the pattern has settled. In the protocol's pulses: the frame a pulse lit. Not known (Automatic): the
        frames watched for LOOK_S; a flash means pulses; a steady picture means steady (for everything shown, once it
        differs from nothing shown: until then a pulse may still come, the protocol not yet playing)."""
        last = self.panel.last.rgb[:, :, 0].astype(np.float32)
        if c["pulsed"] is False:
            return last
        picture = self._pulse_picture(c)
        if c["pulsed"] is True:
            return picture
        if picture is not None and c.get("flashed"):
            c["pulsed"] = True                        # a pulse seen: the LED follows the protocol
            return picture
        looked = time.monotonic() - c["sent_at"] >= LOOK_S and self.panel._shown >= c["frames_at"] + 3
        if looked and not c.get("flashed"):
            if name == "none":
                return last
            if name == "all" and self._light_changed(last, c["pictures"]["none"]):
                c["pulsed"] = False                   # lit and steady: the LED is on all the time
                return last
        return picture                                # waited its longest: the light test then says so

    @staticmethod
    def _light_changed(lit: np.ndarray, unlit: np.ndarray) -> bool:
        """Whether everything shown changed the picture from nothing shown (beyond its noise)."""
        noise = float(np.std(unlit - unlit.mean())) * 0.05 + 1.0
        return abs(patterns.lit_difference(lit, unlit)) >= max(2.0, noise)

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
        c["flashed"] = caught
        if caught or waited > PULSE_WAIT_S:
            return c["best"][1] if c["best"] is not None else self.panel.last.rgb[:, :, 0].astype(np.float32)
        step = c["k"] + 1
        what = ("waiting for a pulse of the protocol" if c["pulsed"] else
                "watching for the LED (on steadily, or in the protocol's pulses)")
        self.cal_status.setText(f"Test picture {step} of {len(c['plan'])}: {what} ({waited:.0f} s of at most "
                                f"{PULSE_WAIT_S:.0f} s)...")
        return None

    def _light_tested(self) -> None:
        c = self._cal
        change = patterns.lit_difference(c["pictures"]["all"], c["pictures"]["none"])
        noise = float(np.std(c["pictures"]["none"] - c["pictures"]["none"].mean())) * 0.05 + 1.0
        if abs(change) < max(2.0, noise):
            self._end_calibration("No change on the picture between all mirrors off and all on. Is the LED on "
                                  "(switched on at its driver, or a protocol that pulses it playing during the "
                                  "calibration), the light path to the slice open, and a fluorescent sample in "
                                  "focus?")
            return
        if c["pulsed"] is None:
            c["pulsed"] = False                       # it changed and no pulse was seen: on steadily
        on = 255 if change > 0 else 0
        self.calibration.on_value = on
        self.swap.blockSignals(True)
        self.swap.setChecked(on == 0)
        self.swap.blockSignals(False)
        c["off"] = c["pictures"]["none" if on == 255 else "all"]
        self.cal_status.setText(f"Light test: a mirror is lit by {on} (the picture {abs(change):.0f} grey levels "
                                "brighter with everything shown). Now the spots...")
        self.calibration.light_tested = True
        # the field's left half: sharp-edged on the picture when the camera is focused where the pattern forms, an
        # even glow when not (then no spot can be seen: said so if none is found)
        half = np.zeros((patterns.ROWS, patterns.COLUMNS), bool)
        half[:, : patterns.COLUMNS // 2] = True
        c["plan"].append(("half", patterns.image(half, on), None))
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
            why = ("Half the field lit gave an even glow over the whole picture, not a lit half: the camera is not "
                   "focused where the pattern forms. Show a Grid (Mirrors on) and focus on the mirror or slide until "
                   "its squares are sharp, then calibrate again." if c.get("sharpness", 1.0) < patterns.SHARP else
                   "Is the picture in focus on the mirror or fluorescent sample, and the light bright enough? Then "
                   "calibrate again.")
            self._end_calibration(f"Only {len(c['spots'])} of {planned} spots were found on the picture, too few "
                                  "to place the field: the light test is kept, the field's place is not. " + why,
                                  keep_light=True)
            return
        cam = np.array([p for p, _ in c["spots"]])
        uv = np.array([q for _, q in c["spots"]])
        to_field, residual = patterns.fit_affine(cam, uv)
        self.calibration.to_field = to_field
        self.calibration.residual_px = residual
        self.calibration.camera_size = self._frame_size()
        self.calibration.objective = self._objective()
        self.calibration.um_per_px = self._um_per_px()
        self.calibration.measured = time.strftime("%Y-%m-%d %H:%M")
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

    def _hold_exposure(self, on: bool) -> None:
        hold = getattr(self.panel, "hold_exposure", None)
        if hold is not None:
            hold(on)

    def show_checkerboard(self, on: bool) -> None:
        """A 4 x 4 checkerboard on the mirrors to focus by (sharp squares on the camera's picture: the camera is
        focused where the pattern forms; an even glow: it is not), or all mirrors off."""
        if not on:
            self.off()
            return
        if not self._held():
            return
        r = (np.arange(patterns.ROWS) * 4 // patterns.ROWS)[:, None]
        k = (np.arange(patterns.COLUMNS) * 4 // patterns.COLUMNS)[None, :]
        lit = (r + k) % 2 == 0
        board = patterns.image(lit, self.calibration.on_value)
        self.auto.setChecked(False)
        self._wanted += 1

        def done(ok):
            if ok is True:
                self.shown_label.setText(f"Checkerboard since {time.strftime('%H:%M:%S')}: focus until its squares "
                                         "are sharp on the camera's picture.")
                self._shown(lit)
                self._set_lit({"checkerboard"})

        self._async(lambda: self.polygon.send(board), done)

    def _end_calibration(self, text: str, keep_light: bool = False) -> None:
        self._poll.stop()
        self._hold_exposure(False)
        on = self.calibration.on_value
        self._cal = None
        if self.polygon.connected:
            self._async(lambda: self.polygon.off(on))
            self.shown_label.setText("Off: no mirror lit.")
            self._shown(None)
            self._set_lit(set())
        self.cal_status.setText(text)
        if keep_light:
            self.calibrations.keep(self.calibration)
            self.save()
            if self.calibration.ready and self.source.currentText() == "Grid" and self.grid_on_picture.isChecked():
                self.make_steps()                     # the grid laid over the picture now that it can be
        self.draw_outlines()
        self._said()
        self.show_state()

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
        age = patterns.age_words(c.measured)
        if age:                                       # the camera or the Polygon moved since: the map is off
            text += f"; measured {age}"
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
        self.show_state()

    def load(self) -> None:
        main = widgets._main_window(self.panel)
        if main is not None and getattr(main, "persist", False):
            try:
                self.calibrations = patterns.Calibrations.from_dict(json.loads(str(main._settings().value(
                    "polygon/calibration", "{}") or "{}")))
            except (ValueError, TypeError, AttributeError):
                self.calibrations = patterns.Calibrations()
            led = str(main._settings().value("polygon/led", "auto") or "auto")
            self.led = led if led in LED_MODES else "auto"
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
            main._settings().setValue("polygon/led", self.led)

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
                self._shown(None)
                self._set_lit(set())
                off, when = result if isinstance(result, tuple) else (False, time.time())
                if off:
                    self._log(light_record.entry("off", on_value=on, when=when))
                self._log(light_record.entry("let go"))
                self.shown_label.setText("Let go" + (" with no mirror lit." if off else "."))

            self._async(work, done)
        self.released.emit()


class CalibrationWindow(QtWidgets.QDialog):
    """The calibration, step by step: what to put under the objective, how the LED is on, Start and Stop, how far
    it has gone, and the outcome in a word (within the target, or what to do)."""

    HELP = {"auto": "Switch the LED on at its driver, or play a protocol that pulses it: Clampwise sees which.",
            "steady": "Switch the LED on at its driver (its continuous or manual mode) before Start.",
            "pulsed": "Press Start, then play a protocol that pulses the LED; each test picture waits for a pulse "
                      "(up to 10 s)."}

    def __init__(self, light: PolygonPanel) -> None:
        super().__init__(light)
        self.light = light
        self.setWindowTitle("Calibrate the patterned light")
        self.setModal(False)
        lay = box("v", "s", "m")
        self.title = label("", "heading", wrap=True)
        lay.addWidget(self.title)
        lay.addWidget(label("Where the mirrors' field lies on the camera's picture, so regions drawn on the picture "
                            "land on the slice where they are drawn. Once for each objective; again whenever the "
                            "camera or the patterned light is moved.", "muted", wrap=True))
        self.needs = widgets.Badge("muted", "")
        self.needs.setWordWrap(True)
        lay.addWidget(self.needs)
        lay.addWidget(label("1. Put a mirror (the pattern seen reflected) or a fluorescent slide under the objective, "
                            "only the patterned light's LED on (the transmitted lamp and any other light source off). "
                            "Show the checkerboard and focus until its squares are sharp on the camera's picture; an "
                            "even glow means the pattern is not forming there, and no spot can be found.", "",
                            wrap=True))
        board = box("h", "s")
        self.board = widgets.button("Checkerboard on the mirrors", None, "A 4 x 4 checkerboard to focus by; again: "
                                    "all mirrors off", quiet=True)
        self.board.setCheckable(True)
        self.board.toggled.connect(light.show_checkerboard)
        board.addWidget(self.board)
        board.addStretch(1)
        lay.addLayout(board)
        lay.addWidget(label("2. How is the LED on?", "", wrap=True))
        self.led = QtWidgets.QComboBox()
        for key, words in LED_MODES.items():
            self.led.addItem(words, key)
        self.led.currentIndexChanged.connect(self._led_chosen)
        lay.addWidget(self.led)
        self.led_help = label("", "muted", wrap=True)
        lay.addWidget(self.led_help)
        lay.addWidget(label("3. Start. The mirrors show test patterns: all off, all on, then a few spots, then a "
                            "check grid. About a minute (longer with pulses).", "", wrap=True))
        go = box("h", "s")
        self.start = widgets.primary("Start", self._start, "Calibrate at the objective in use")
        self.stop = widgets.button("Stop", self._stop, "Stop the calibration; all mirrors off")
        go.addWidget(self.start)
        go.addWidget(self.stop)
        go.addStretch(1)
        lay.addLayout(go)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.doing = label("", "muted", wrap=True)
        lay.addWidget(self.doing)
        self.outcome = widgets.Badge("muted", "")
        self.outcome.setWordWrap(True)
        lay.addWidget(self.outcome)
        lay.addStretch(1)
        widgets.resizable(self, lay, "polygon_calibration", (460, 520))
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.refresh)
        self._ran = False                             # Start pressed since the window opened: the outcome is said

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.led.blockSignals(True)
        self.led.setCurrentIndex(max(0, self.led.findData(self.light.led)))
        self.led.blockSignals(False)
        self.led_help.setText(self.HELP[self.light.led])
        self._timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def _led_chosen(self) -> None:
        self.light.led = self.led.currentData()
        self.led_help.setText(self.HELP[self.light.led])
        self.light.save()

    def missing(self) -> str:
        """What the calibration needs and lacks, in words ("" when ready)."""
        p = self.light
        lacks = []
        if getattr(p.panel, "_thread", None) is None or p.panel.last is None:
            lacks.append("the camera taken, its picture showing")
        if not p.polygon.connected:
            lacks.append("the patterned light held")
        return " and ".join(lacks)

    def _start(self) -> None:
        if self.missing():
            return
        self.board.blockSignals(True)                 # the calibration shows its own patterns
        self.board.setChecked(False)
        self.board.blockSignals(False)
        self._ran = True
        self.light.calibrate()
        self.refresh()

    def _stop(self) -> None:
        if self.light._cal is not None:
            self.light._end_calibration("Stopped before it finished: all mirrors off. Start again to calibrate.")
        self.refresh()

    def refresh(self) -> None:
        p = self.light
        self.title.setText(f"Calibrate at {p._objective() or 'the objective in use'}")
        running = p._cal is not None
        lacks = self.missing()
        self.needs.setVisible(bool(lacks) and not running)
        self.needs.set("warn", f"Needs {lacks} first (Use here on their sections)." if lacks else "")
        self.start.setEnabled(not running and not lacks)
        self.board.setEnabled(not running and p.polygon.connected)
        self.stop.setEnabled(running)
        self.led.setEnabled(not running)
        done, total = p.calibration_progress()
        self.bar.setVisible(running)
        if running:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
        self.doing.setText(p.cal_status.text())
        if running or not self._ran:
            self.outcome.setVisible(False)
            return
        status, words = p.calibration_words()
        self.outcome.setVisible(True)
        self.outcome.set(status if p.calibration.ready else "bad", words if p.calibration.ready else
                         "Not calibrated: see above, then Start again.")


class FieldView(QtWidgets.QWidget):
    """The patterned light's own window in Live: its field as the mirrors show it, in its 16:10 shape. What is lit
    now in white (a step, the checkerboard, a calibration picture), the pattern's steps outlined and numbered (yellow
    chosen, blue lit); a click chooses a step, Ctrl+click adds one or takes it out, a double-click lights it. Above,
    what the mirrors do and the two buttons that change it. Needs no calibration: it is the field itself, not the
    camera's picture of it."""

    def __init__(self, light: PolygonPanel) -> None:
        super().__init__()
        self.light = light
        lay = box("v", "xs")
        self.state = widgets.Badge("muted", "")
        self.state.setWordWrap(True)
        lay.addWidget(self.state)
        go = box("h", "s")
        self.on_button = widgets.primary("Mirrors on", light.project, "The chosen step(s) on the mirrors (lit "
                                         "whenever the LED fires)")
        self.off_button = widgets.button("All mirrors off", light.off, "No mirror lit: nothing reaches the slice")
        go.addWidget(self.on_button)
        go.addWidget(self.off_button)
        go.addStretch(1)
        lay.addLayout(go)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setToolTip("The patterned light's field, as its mirrors show it: white is lit now. Click a step to "
                             "choose it, Ctrl+click for more, double-click to light it.")
        self.box = self.view.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.box.setMouseEnabled(False, False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.box.addItem(self.image)
        edge = pg.PlotCurveItem([0, patterns.FIELD_W, patterns.FIELD_W, 0, 0], [0, 0, patterns.FIELD_H,
                                                                              patterns.FIELD_H, 0],
                                pen=pg.mkPen((128, 128, 128), width=1, style=QtCore.Qt.DashLine))
        self.box.addItem(edge)
        self.box.setRange(QtCore.QRectF(0, 0, patterns.FIELD_W, patterns.FIELD_H), padding=0.02)
        self.view.scene().sigMouseClicked.connect(self._clicked)
        lay.addWidget(self.view, 1)
        lay.addWidget(label("Its other controls (pattern, steps, calibration) are in the Patterned light section, "
                            "at the right.", "muted", wrap=True))
        self.setLayout(lay)
        self._items: list = []
        self._drawn = None                            # what was drawn last: nothing redrawn while it is the same
        light.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """The state's words and the buttons as the panel has them; the field drawn again only when it changed."""
        p = self.light
        self.state.set(*p.state._shown)
        self.on_button.setEnabled(p.on_button.isEnabled())
        self.off_button.setEnabled(p.off_button.isEnabled())
        chosen = tuple(p._chosen_rows())
        key = (id(p.steps), len(p.steps), chosen, frozenset(p._lit), id(p.shown_mask))
        if key == self._drawn:
            return
        self._drawn = key
        m = p.shown_mask
        lit = np.zeros((patterns.ROWS // 2, patterns.COLUMNS // 2), np.uint8) if m is None else \
            (m[::2, ::2] * np.uint8(255)).astype(np.uint8)          # every 2nd mirror: the screen shows no more
        self.image.setImage(lit, autoLevels=False, levels=(0, 255))
        self.image.setRect(QtCore.QRectF(0, 0, patterns.FIELD_W, patterns.FIELD_H))
        for item in self._items:
            self.box.removeItem(item)
        self._items = []
        many = len(p.steps) > 100
        for k, step in enumerate(p.steps):
            bright, on = k in chosen, step.name in p._lit
            if many and not (bright or on):
                continue
            colour = LIT if on else CHOSEN
            pen = pg.mkPen(colour if bright or on else colour + (110,), width=2 if bright or on else 1)
            for shape in step.shapes:
                pts = shape.outline()
                item = pg.PlotCurveItem(pts[:, 0] * patterns.FIELD_W, pts[:, 1] * patterns.FIELD_H, pen=pen)
                self.box.addItem(item)
                self._items.append(item)
            if step.shapes and (len(p.steps) <= 25 or bright or on):
                cu, cv = step.shapes[0].centre()
                text = pg.TextItem(step.name, color=colour if bright or on else colour + (150,), anchor=(0.5, 0.5))
                text.setPos(cu * patterns.FIELD_W, cv * patterns.FIELD_H)
                self.box.addItem(text)
                self._items.append(text)

    def step_at(self, u: float, v: float) -> int | None:
        """The step covering (u, v) of the field (0 to 1 each way; the last made when several overlap)."""
        for k in range(len(self.light.steps) - 1, -1, -1):
            if self.light.steps[k].covers(u, v):
                return k
        return None

    def click_at(self, u: float, v: float, add: bool = False, light_it: bool = False) -> int | None:
        """A click at (u, v) of the field: the step there chosen (``add``: added or taken out), lit when asked."""
        row = self.step_at(u, v)
        if row is None:
            return None
        self.light.choose_at(row, add=add and not light_it)
        if light_it:
            self.light.project()
        return row

    def _clicked(self, event) -> None:
        if not self.box.sceneBoundingRect().contains(event.scenePos()):
            return
        at = self.box.mapSceneToView(event.scenePos())
        u, v = at.x() / patterns.FIELD_W, at.y() / patterns.FIELD_H
        if 0 <= u <= 1 and 0 <= v <= 1:
            add = bool(event.modifiers() & QtCore.Qt.ControlModifier)
            self.click_at(u, v, add=add, light_it=event.double())
