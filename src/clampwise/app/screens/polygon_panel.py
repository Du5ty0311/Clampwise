"""The Polygon in Live's patterned light view (Use here, PolyScan closed): where the light goes on the slice,
while Clampex times the light itself.

A pattern is made (a full field, a spot, a numbered grid, a line of spots, an image file, a PolyScan sequence, or
shapes drawn on the camera's picture) and its steps shown one at a time: chosen in the list, or the next one every
few seconds. Each step shown is written to ``polygon_log.csv`` in the cell's folder (when it was shown, which step,
its shapes), to be matched to the recordings made meanwhile.

The light test and calibration (with the light on and a fluorescent slide or slice in view) shows the Polygon
nothing and everything, to learn which byte lights a mirror, then a few spots, found on the camera's picture, to
learn how the Polygon's field lies under the camera; drawing on the picture and the outlines over it need it.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import patterns
from ...rig.polygon import Polygon
from .. import theme, widgets
from ..widgets import box, label

SOURCES = ("Full field", "Spot", "Grid", "Line of spots", "Image file", "Sequence file", "Drawn on the picture")
POLYSCAN_DATA = Path.home() / "Documents" / "Mightex" / "PolyScan3" / "User Data"
SETTLE_S = 0.4                 # after a pattern is sent, before a picture of it is used (at least 3 exposures)
PULSE_WAIT_S = 10.0            # pulsed light: how long each test picture waits for a pulse of the protocol
FLASH_GREY = 4.0               # a frame this much brighter (mean grey) than the darkest one caught a pulse


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
        self.calibration = patterns.Calibration()
        self.drawn: list[pg.ROI] = []
        self.outline_items: list = []
        self._cal: dict | None = None                 # the calibration under way
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
        self.list.itemDoubleClicked.connect(lambda *_: self.project())
        self.list.currentRowChanged.connect(lambda *_: self.draw_outlines())
        lay.addWidget(self.list, 1)
        go = box("h", "s")
        go.addWidget(widgets.button("Previous", lambda: self.step_by(-1), "The step before, shown", quiet=True))
        go.addWidget(widgets.primary("Project", self.project, "Show the chosen step (lit while the LED is on)"))
        go.addWidget(widgets.button("Next", lambda: self.step_by(1), "The next step, shown", quiet=True))
        go.addWidget(widgets.button("Off", self.off, "No mirror lit"))
        go.addStretch(1)
        self.auto = QtWidgets.QCheckBox("Next every")
        self.auto.setToolTip("Step on by itself (time it to your protocol's sweeps); unticked: by hand")
        self.every = _pct(10.0, 0.2, 3600.0, 1.0, " s")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.step_by(1, wrap=False))
        self.auto.toggled.connect(self._auto)
        self.every.valueChanged.connect(lambda *_: self._auto(self.auto.isChecked()))
        go.addWidget(self.auto)
        go.addWidget(self.every)
        lay.addLayout(go)
        self.shown_label = label("Nothing shown.", "muted", wrap=True)
        lay.addWidget(self.shown_label)
        widgets.resizable(self, lay)
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._calibration_tick)
        self.load()

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
        page(("Rows", self.rows), ("Columns", self.cols), ("Each spot, of its cell", self.fill), ("", self.round),
             ("", label("Numbered from 1 at the top left along each row, as its own program numbers its grids.", "muted",
                        wrap=True)))
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
        d = box("h", "s")
        d.addWidget(widgets.button("Add a rectangle", lambda: self.add_shape("rect"), quiet=True))
        d.addWidget(widgets.button("Add an ellipse", lambda: self.add_shape("ellipse"), quiet=True))
        d.addWidget(widgets.button("Clear", self.clear_shapes, quiet=True))
        self.each_step = QtWidgets.QCheckBox("Each shape its own step")
        self.each_step.setChecked(True)
        d.addWidget(self.each_step)
        d.addStretch(1)
        draw.setLayout(d)
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
        self.device_status.setText(self.polygon.status + (f"; {t} inside" if t is not None else ""))

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
                steps = patterns.grid(self.rows.value(), self.cols.value(), fill=self.fill.value() / 100,
                                      round_spots=self.round.isChecked())
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
        self.list.clear()
        self.list.addItems([s.name for s in steps])
        if steps:
            self.list.setCurrentRow(0)
        self.pattern_note.setText(f"{len(steps)} step(s). {note}".strip())
        self.save()
        self.draw_outlines()

    def project(self) -> None:
        row = self.list.currentRow()
        if not (0 <= row < len(self.steps)):
            self.panel.window.say("Make the steps first, then choose one.")
            return
        if not self._held():
            return
        step, on, n = self.steps[row], self.calibration.on_value, len(self.steps)
        self._wanted += 1
        ticket = self._wanted

        def work():
            return None if ticket != self._wanted else self.polygon.show(step, on)   # a newer step asked for

        def done(ok):
            if ok is None:
                return
            if ok is True:
                self.shown_label.setText(f"Showing step {step.name} ({row + 1} of {n}) since "
                                         f"{time.strftime('%H:%M:%S')}." + ("" if self._calibrated_light() else
                                         " The light test has not been done: lit and unlit may be swapped."))
                self._log(step)
            else:
                self.shown_label.setText(f"Step {step.name} was not shown: {self.polygon.status}.")

        self._async(work, done)
        self.draw_outlines()

    def step_by(self, k: int, wrap: bool = True) -> None:
        n = len(self.steps)
        if not n:
            return
        row = self.list.currentRow() + k
        if not wrap and not 0 <= row < n:
            self.auto.setChecked(False)
            self.shown_label.setText(self.shown_label.text() + " The last step was shown: stepping stopped.")
            return
        self.list.setCurrentRow(row % n)
        self.project()

    def off(self) -> None:
        self.auto.setChecked(False)
        if not self.polygon.connected:
            return
        on = self.calibration.on_value
        self._wanted += 1

        def done(ok):
            if ok is True:
                self.shown_label.setText(f"Off since {time.strftime('%H:%M:%S')}: no mirror lit.")
                self._log(None)

        self._async(lambda: self.polygon.off(on), done)
        self.draw_outlines()

    def _auto(self, on: bool) -> None:
        self.timer.stop()
        if on:
            self.timer.start(int(self.every.value() * 1000))

    def _log(self, step: patterns.Step | None) -> None:
        folder = self.panel.folder() or ""
        if not folder or not Path(folder).is_dir():
            return
        path = Path(folder) / "polygon_log.csv"
        new = not path.exists()
        try:
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "step", "shapes (field: 0-1 across, down)", "pattern"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), step.name if step else "off",
                            step.describe() if step else "", self.source.currentText()])
        except OSError:
            pass

    # ---------------------------------------------------------------- drawn on the picture
    def _frame_size(self) -> tuple[int, int]:
        last = self.panel.last
        return (last.rgb.shape[1], last.rgb.shape[0]) if last is not None else (0, 0)

    def add_shape(self, kind: str) -> None:
        w, h = self._frame_size()
        if not w:
            self.panel.window.say("Take the camera first: shapes are drawn on its picture.")
            return
        size = [w * 0.15, w * 0.15]
        pos = [w / 2 - size[0] / 2 + 20 * len(self.drawn), h / 2 - size[1] / 2]
        pen = pg.mkPen("c", width=2)
        roi = pg.RectROI(pos, size, pen=pen) if kind == "rect" else pg.EllipseROI(pos, size, pen=pen)
        roi.kind = kind
        self.panel.box.addItem(roi)
        self.drawn.append(roi)
        roi.sigRegionChangeFinished.connect(lambda *_: self._drawn_moved())

    def _drawn_moved(self) -> None:
        """A shape moved or resized on the picture: while drawn shapes are shown, shown again where they are now."""
        if self.source.currentText() != "Drawn on the picture" or not self.calibration.ready:
            return
        row = max(0, self.list.currentRow())
        showing = self.polygon.connected and self.polygon.shown is not None
        self.make_steps()
        if self.steps:
            self.list.setCurrentRow(min(row, len(self.steps) - 1))
            if showing:
                self.project()

    def clear_shapes(self) -> None:
        for roi in self.drawn:
            self.panel.box.removeItem(roi)
        self.drawn = []

    def drawn_steps(self) -> list[patterns.Step]:
        if not self.calibration.ready:
            raise ValueError("calibrate first: the shapes drawn are found on the patterned light's field through it")
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
            return [patterns.Step(str(k + 1), [s]) for k, s in enumerate(shapes)]
        return [patterns.Step("Drawn", shapes)]

    # ---------------------------------------------------------------- outlines over the picture
    def draw_outlines(self) -> None:
        for item in self.outline_items:
            self.panel.box.removeItem(item)
        self.outline_items = []
        if not (self.outlines.isChecked() and self.calibration.ready and self.steps and self.isVisible()):
            return
        current = self.list.currentRow()
        many = len(self.steps) > 100
        cal = self.calibration.at_size(self._frame_size())     # the picture's binning now
        for k, step in enumerate(self.steps):
            if many and k != current:
                continue
            bright = k == current
            pen = pg.mkPen((255, 220, 0) if bright else (255, 220, 0, 90), width=2 if bright else 1)
            for shape in step.shapes:
                pts = shape.outline()
                x, y = cal.camera_of(pts[:, 0], pts[:, 1])
                item = pg.PlotCurveItem(x, y, pen=pen)
                self.panel.box.addItem(item)
                self.outline_items.append(item)
            if step.shapes and len(self.steps) <= 100:
                cu, cv = step.shapes[0].centre()
                x, y = cal.camera_of(np.array([cu]), np.array([cv]))
                text = pg.TextItem(step.name, color=(255, 220, 0) if bright else (255, 220, 0, 140),
                                   anchor=(0.5, 0.5))
                text.setPos(float(x[0]), float(y[0]))
                self.panel.box.addItem(text)
                self.outline_items.append(text)

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
        plan = [("none", np.zeros((patterns.ROWS, patterns.COLUMNS), np.uint8)),
                ("all", np.full((patterns.ROWS, patterns.COLUMNS), 255, np.uint8))]
        self._cal = {"plan": plan, "k": -1, "pictures": {}, "spots": [], "off": None}
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
        name, img = c["plan"][c["k"]]
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
        name, _ = c["plan"][c["k"]]
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
                c["spots"].append((found, patterns.CALIBRATION_SPOTS[int(name[4:])]))
        if self._cal is None:
            return
        if c["k"] + 1 >= len(c["plan"]):
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
        for k, (u, v) in enumerate(patterns.CALIBRATION_SPOTS):
            step = patterns.spot(u, v, patterns.CALIBRATION_DIAMETER)[0]
            c["plan"].append((f"spot{k}", patterns.image(step.mirrors(), on)))

    def _fit(self) -> None:
        c = self._cal
        if len(c["spots"]) < 3:
            self._end_calibration(f"Only {len(c['spots'])} of {len(patterns.CALIBRATION_SPOTS)} spots were found "
                                  "on the picture: the light test is kept, the field's place is not. Bring more of "
                                  "the field into view (a lower objective) and calibrate again.", keep_light=True)
            return
        cam = np.array([p for p, _ in c["spots"]])
        uv = np.array([q for _, q in c["spots"]])
        to_field, residual = patterns.fit_affine(cam, uv)
        self.calibration.to_field = to_field
        self.calibration.residual_px = residual
        self.calibration.camera_size = self._frame_size()
        self._end_calibration(f"Calibrated with {len(c['spots'])} spots: a mirror is lit by "
                              f"{self.calibration.on_value}; the spots land within {residual:.1f} px of where "
                              "the fit puts them.", keep_light=True)

    def _end_calibration(self, text: str, keep_light: bool = False) -> None:
        self._poll.stop()
        on = self.calibration.on_value
        self._cal = None
        if self.polygon.connected:
            self._async(lambda: self.polygon.off(on))
            self.shown_label.setText("Off: no mirror lit.")
        self.cal_button.setEnabled(True)
        self.cal_status.setText(text)
        if keep_light:
            self.save()
        self.draw_outlines()
        self._said()

    # ---------------------------------------------------------------- kept for the user
    def _cal_text(self) -> str:
        c = self.calibration
        if not c.ready:
            if c.light_tested:
                return f"Light tested (a mirror is lit by {c.on_value}); the field's place is not known yet."
            return "Not calibrated: the light test and the field's place are not known yet."
        return (f"Calibrated: a mirror is lit by {c.on_value}; the field's place fitted within "
                f"{c.residual_px:.1f} px at {c.camera_size[0]} x {c.camera_size[1]}.")

    def load(self) -> None:
        main = widgets._main_window(self.panel)
        if main is None or not getattr(main, "persist", False):
            self.cal_status.setText(self._cal_text())
            return
        try:
            self.calibration = patterns.Calibration.from_dict(json.loads(str(main._settings().value(
                "polygon/calibration", "{}") or "{}")))
        except (ValueError, TypeError):
            self.calibration = patterns.Calibration()
        self.cal_status.setText(self._cal_text())
        self.swap.blockSignals(True)
        self.swap.setChecked(self.calibration.on_value == 0)
        self.swap.blockSignals(False)

    def _swapped(self, on: bool) -> None:
        """Lit by 0 or by 255, set by hand: the step shown is sent again so the change is seen at once."""
        self.calibration.on_value = 0 if on else 255
        self.save()
        self.cal_status.setText(self._cal_text() + " (lit by " + str(self.calibration.on_value) + ", set by hand)")
        if self.polygon.connected and self.polygon.shown is not None:
            self.project()

    def save(self) -> None:
        main = widgets._main_window(self.panel)
        if main is not None and getattr(main, "persist", False):
            main._settings().setValue("polygon/calibration", json.dumps(self.calibration.to_dict()))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.draw_outlines()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        for item in self.outline_items:
            self.panel.box.removeItem(item)
        self.outline_items = []

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
            self._async(self.polygon.close, lambda *_: self._busy_end())
        self.released.emit()
