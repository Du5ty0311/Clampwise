"""The camera in Clampwise: Live's camera view can take the camera itself while the camera's
own program is closed (rig.lucam), show its live picture, change the exposure, gain and frames averaged, adjust
what is shown (rig.picture: brightness, contrast, gamma, saturation, sharpness, clarity, levels, colour balance;
each with its reset, sets of them kept by name), draw a scale bar from the length calibration, and save a snapshot
into the cell's folder. Frames are taken and adjusted off the window's thread; the camera is let go with one click
(or when Clampwise closes)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import lucam, picture
from .. import theme, widgets
from ..widgets import box, label

SHOW_EVERY_S = 1 / 60                # at most 60 redraws a second, and only once the last one is drawn
STATS_EVERY_S = 0.5                  # the histogram and the clipping warning, twice a second
COLOURS = {"green": (0, 255, 0), "red": (255, 0, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255),
           "yellow": (255, 255, 0)}
FRAMES = {"dark": 16, "background": 16, "overlay": 8}


class CameraThread(QtCore.QThread):
    """Takes frames until stopped; the camera is opened and closed in this thread."""

    opened = QtCore.Signal(object)           # {"exposure": Setting, "gain": Setting, "serial": int}
    frame = QtCore.Signal(object, object, object)   # the frame as taken, the picture as shown (uint8), its statistics
    captured = QtCore.Signal(str, object)    # "dark", "background" or "overlay", and the picture kept
    exposure_set = QtCore.Signal(float)      # auto exposure's choice
    failed = QtCore.Signal(str)

    def __init__(self, make=lucam.Camera) -> None:
        super().__init__()
        self.make = make
        self.exposure_ms: float | None = None
        self.gain: float | None = None
        self.average = 1
        self.adjust = picture.Adjust()
        self.corrections = picture.Corrections()
        self.request: tuple | None = None    # ("dark" | "background" | "overlay", frames) or ("auto",)
        self.taken = 0                       # pictures taken (each of ``average`` frames): the frame rate's count
        self.drawing = False                 # a picture sent and not yet drawn: none other is sent meanwhile
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        self.wait(10000)

    def run(self) -> None:
        cam = self.make()
        if not cam.open():
            self.failed.emit(cam.status)
            return
        try:
            exposure = cam.setting(lucam.EXPOSURE)
            self.opened.emit({"exposure": exposure, "gain": cam.setting(lucam.GAIN), "serial": cam.serial})
            shown_at = stats_at = 0.0
            while not self._stop:
                want = (self.exposure_ms or cam.exposure_ms, self.gain or cam.gain)
                if want != (cam.exposure_ms, cam.gain):
                    cam.set_exposure(*want)
                request, self.request = self.request, None
                if request is not None and not self._serve(cam, request, exposure):
                    break
                f = cam.frame(self.average)
                self.taken += 1
                if f is None:
                    self.failed.emit(cam.status)
                    break
                now = time.monotonic()
                # every frame the screen keeps up with (the camera's own frames came 31-47 ms apart: a fixed
                # 33 ms gate drew every other one), never a queue of frames behind the picture
                if not self.drawing and now - shown_at >= SHOW_EVERY_S:
                    shown_at = now
                    stats = None
                    if now - stats_at >= STATS_EVERY_S:
                        stats_at, stats = now, picture.stats(f.rgb)
                    table = picture.display_table(self.corrections, self.adjust) if picture.is_grey(f.rgb) else None
                    self.drawing = True
                    self.frame.emit(f, (f.rgb, table) if table is not None else
                                    picture.shown(f.rgb, self.corrections, self.adjust), stats)
        finally:
            cam.close()

    def _serve(self, cam, request: tuple, exposure) -> bool:
        kind = request[0]
        if kind == "auto":                       # the brightest 0.5% to 80% of full scale, a few frames at most
            low, high = (exposure.low, exposure.high) if exposure is not None else (0.01, 10000.0)
            for _ in range(8):
                f = cam.frame()
                if f is None:
                    self.failed.emit(cam.status)
                    return False
                s = picture.stats(f.rgb)
                new = picture.next_exposure(cam.exposure_ms, s["p995"], low, high, s["clipped"])
                if abs(new - cam.exposure_ms) <= 0.08 * cam.exposure_ms and s["clipped"] <= 0.001:
                    break
                cam.set_exposure(new)
                self.exposure_ms = new
            self.exposure_set.emit(cam.exposure_ms)
            return True
        f = cam.frame(request[1])
        if f is None:
            self.failed.emit(cam.status)
            return False
        kept = picture.overlay_from(f.rgb, self.corrections) if kind == "overlay" else f.rgb.astype(np.float32)
        self.captured.emit(kind, kept)
        return True


def _camera_pixel_um() -> float:
    """The sensor's pixel for the camera its program last named (Infinity Analyze's settings), else 0."""
    try:
        from ...rig.camera import read_settings

        s = read_settings()
        return picture.pixel_um_of(s.camera) if s is not None else 0.0
    except Exception:
        return 0.0


class AdjustDialog(QtWidgets.QDialog):
    """Each adjustment as a slider with its value and its reset; presets kept by name; the length calibration."""

    changed = QtCore.Signal()

    def __init__(self, panel: "CameraPanel") -> None:
        super().__init__(panel)
        self.panel = panel
        self.setWindowTitle("Adjust the camera's picture")
        self.setModal(False)
        lay = box("v", "s", "m")
        lay.addWidget(label("These change what is shown, never the picture saved (a snapshot keeps the camera's "
                            "own pixels and notes the adjustments that were on).", "muted", wrap=True))
        presets = box("h", "s")
        self.presets = QtWidgets.QComboBox()
        self.presets.setToolTip("A set of adjustments kept by name")
        self.presets.activated.connect(self._load_preset)
        presets.addWidget(self.presets, 1)
        presets.addWidget(widgets.button("Save as...", self._save_preset, "Keep the adjustments now on under a name"))
        presets.addWidget(widgets.button("Delete", self._delete_preset, "Forget the chosen preset", quiet=True))
        lay.addLayout(presets)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(theme.SPACE["s"])
        self.rows: dict[str, tuple[QtWidgets.QSlider, QtWidgets.QDoubleSpinBox]] = {}
        names = {"brightness": "Brightness", "contrast": "Contrast", "gamma": "Gamma (below 1 lifts dim parts)",
                 "saturation": "Saturation", "sharpness": "Sharpness", "clarity": "Clarity (local contrast)",
                 "red": "Red", "green": "Green", "blue": "Blue"}
        for row, (key, caption) in enumerate(names.items()):
            low, high = picture.RANGES[key]
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 1000)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            slider.valueChanged.connect(lambda v, k=key, lo=low, hi=high: self._from_slider(k, lo + (hi - lo) * v / 1000))
            spin.valueChanged.connect(lambda v, k=key: self._set(k, v))
            reset = widgets.button("Reset", lambda k=key: self._set(k, getattr(picture.Adjust(), k)), quiet=True)
            grid.addWidget(label(caption), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            grid.addWidget(reset, row, 3)
            self.rows[key] = (slider, spin)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        self.auto = QtWidgets.QCheckBox("Auto levels (the darkest 0.5% black, the brightest 0.5% white)")
        self.auto.toggled.connect(lambda on: self._set("auto_levels", on))
        lay.addWidget(self.auto)
        row = box("h", "s")
        row.addWidget(widgets.button("White balance", panel.white_balance,
                                     "Make the picture's mean grey: point the camera at an empty, bright field first"))
        row.addWidget(widgets.button("Reset all", lambda: self.show_adjust(picture.Adjust(), apply=True)))
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(label("Corrections (what each frame shown has taken out)", "section"))
        corr = QtWidgets.QGridLayout()
        corr.setHorizontalSpacing(theme.SPACE["s"])
        self.dark_state = label("none", "muted")
        self.bg_state = label("none", "muted")
        corr.addWidget(widgets.button("Take dark frame", lambda: panel.capture("dark"),
                                      "The camera's own offset: block the light, then take it (16 frames averaged)"),
                       0, 0)
        corr.addWidget(self.dark_state, 0, 1)
        corr.addWidget(widgets.button("Clear", lambda: panel.forget("dark"), quiet=True), 0, 2)
        corr.addWidget(widgets.button("Take background", lambda: panel.capture("background"),
                                      "Defocus, or move to an empty part of the slice, then take it (16 frames)"),
                       1, 0)
        corr.addWidget(self.bg_state, 1, 1)
        corr.addWidget(widgets.button("Clear", lambda: panel.forget("background"), quiet=True), 1, 2)
        self.mode = QtWidgets.QComboBox()
        for text in ("Background: not used", "Subtract it (contrast for infrared, video-enhanced)",
                     "Divide by it (flat field, for fluorescence)"):
            self.mode.addItem(text)
        self.mode.currentIndexChanged.connect(lambda i: panel.set_correction(mode=("off", "subtract", "flat")[i]))
        corr.addWidget(self.mode, 2, 0, 1, 3)
        corr.setColumnStretch(1, 1)
        lay.addLayout(corr)
        lay.addWidget(label("Overlay (a kept picture, such as fluorescence, over the live one)", "section"))
        ov = box("h", "s")
        ov.addWidget(widgets.button("Keep this picture", lambda: panel.capture("overlay"),
                                    "Take the picture now (8 frames averaged) to lay over the live one: the cells "
                                    "it shows bright stay marked while you patch"))
        ov.addWidget(widgets.button("Clear", lambda: panel.forget("overlay"), quiet=True))
        self.ov_on = QtWidgets.QCheckBox("Shown")
        self.ov_on.setChecked(True)
        self.ov_on.toggled.connect(lambda on: panel.set_correction(overlay_on=on))
        self.ov_colour = QtWidgets.QComboBox()
        for name in ("green", "red", "magenta", "cyan", "yellow"):
            self.ov_colour.addItem(name)
        self.ov_colour.currentTextChanged.connect(lambda n: panel.set_correction(overlay_colour=COLOURS[n]))
        self.ov_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ov_opacity.setRange(0, 100)
        self.ov_opacity.setValue(60)
        self.ov_opacity.setToolTip("How strongly the kept picture is laid over the live one")
        self.ov_opacity.valueChanged.connect(lambda v: panel.set_correction(overlay_opacity=v / 100))
        for w in (self.ov_on, self.ov_colour, self.ov_opacity):
            ov.addWidget(w)
        lay.addLayout(ov)
        lay.addWidget(label("Display", "section"))
        disp = box("h", "s")
        self.lut = QtWidgets.QComboBox()
        for name in picture.LUTS:
            self.lut.addItem(name)
        self.lut.setToolTip("Colour as taken, or a single-colour scale (grey, green for GFP, fire for dim signal)")
        self.lut.currentTextChanged.connect(lambda n: panel.set_correction(lut=n))
        self.clipped = QtWidgets.QCheckBox("Clipped pixels in red")
        self.clipped.setChecked(True)
        self.clipped.setToolTip("Pixels at the camera's full scale carry no measure (Waters 2009): lower the exposure")
        self.clipped.toggled.connect(lambda on: panel.set_correction(show_clipped=on))
        self.cross = QtWidgets.QCheckBox("Crosshair")
        self.cross.toggled.connect(panel.set_crosshair)
        self.compass = QtWidgets.QCheckBox("Compass")
        self.compass.setToolTip("The slice's directions in its plane over the picture (its plane and orientation "
                                "below)")
        self.compass.toggled.connect(lambda *_: panel._orientation_changed())
        for w in (self.lut, self.clipped, self.cross, self.compass):
            disp.addWidget(w)
        disp.addStretch(1)
        lay.addLayout(disp)
        self.hist = pg.PlotWidget()
        self.hist.setFixedHeight(90)
        self.hist.setMouseEnabled(False, False)
        self.hist.hideAxis("left")
        self.hist.setToolTip("The brightness of every pixel: keep the right end clear of the edge (nothing clipped)")
        theme.style_plot(self.hist, panel.window.palette)
        self.hist_curve = self.hist.plot([], [], stepMode="center", fillLevel=0,
                                         brush=theme.series(panel.window.palette)[0])
        lay.addWidget(self.hist)
        lay.addWidget(label("Orientation of the slice", "section"))
        cut = box("h", "s")
        cut.addWidget(label("Plane", "muted"))
        self.plane = QtWidgets.QComboBox()
        for p in picture.PLANES:
            self.plane.addItem(p.capitalize(), p)
        self.plane.setToolTip("How the slice was cut: its two axes are in the picture, the third is cut across "
                              "(horizontal: dorsal-ventral; coronal: rostral-caudal; sagittal: medial-lateral)")
        cut.addWidget(self.plane)
        cut.addWidget(label("Names", "muted"))
        self.naming = QtWidgets.QComboBox()
        self.naming.addItems(picture.NAMINGS)
        self.naming.setToolTip("Rostral-caudal as such or as anterior-posterior, in words or by their letters")
        cut.addWidget(self.naming)
        self.out_of_plane = label("", "muted")
        cut.addWidget(self.out_of_plane)
        cut.addStretch(1)
        lay.addLayout(cut)
        orient = box("h", "s")
        self.up_label = label("", "muted")
        orient.addWidget(self.up_label)
        self.rostral = QtWidgets.QSpinBox()                  # the plane's first direction (rostral or dorsal)
        self.rostral.setRange(0, 359)
        self.rostral.setSingleStep(15)
        self.rostral.setWrapping(True)
        self.rostral.setSuffix(" deg")
        orient.addWidget(self.rostral)
        self.turn_left = widgets.button("Turn left", lambda: self.rostral.setValue((self.rostral.value() - 90) % 360),
                                        "", quiet=True)
        self.turn_right = widgets.button("Turn right", lambda: self.rostral.setValue((self.rostral.value() + 90) % 360),
                                         "", quiet=True)
        orient.addWidget(self.turn_left)
        orient.addWidget(self.turn_right)
        self.medial = QtWidgets.QComboBox()                  # the side of it the second axis's first end is on
        self.medial.addItems(["", ""])
        orient.addWidget(self.medial)
        orient.addStretch(1)
        lay.addLayout(orient)
        self.name_orientation()
        self.plane.currentIndexChanged.connect(lambda *_: (self.name_orientation(), panel._orientation_changed()))
        self.naming.currentIndexChanged.connect(lambda *_: (self.name_orientation(), panel._orientation_changed()))
        self.rostral.valueChanged.connect(lambda *_: panel._orientation_changed())
        self.medial.currentIndexChanged.connect(lambda *_: panel._orientation_changed())
        lay.addWidget(label("Length calibration", "section"))
        cal = box("h", "s")
        self.um = QtWidgets.QDoubleSpinBox()             # the calibration of the objective chosen on the panel
        self.um.setRange(0.0, 50.0)
        self.um.setDecimals(5)
        self.um.setSingleStep(0.001)
        self.um.setSpecialValueText("not measured here")
        self.um.setSuffix(" um a pixel")
        self.um.setToolTip("Measured at the objective chosen on the camera's panel (a stage micrometer: From the "
                           "Measure line); 0: taken from another objective's calibration by the ratio of magnifications")
        from_line = widgets.button("From the Measure line...", panel.calibrate_from_ruler,
                                   "Drag the Measure line along a stage micrometer, then type its true length")
        self.bar = QtWidgets.QCheckBox("Scale bar")
        self.bar.setChecked(True)                        # shown as soon as there is a calibration
        for w in (self.um, from_line, self.bar):
            cal.addWidget(w)
        lay.addLayout(cal)
        objs = box("h", "s")
        objs.addWidget(label("Objectives", "muted"))
        self.objectives = QtWidgets.QLineEdit()
        self.objectives.setToolTip("The magnifications on the panel's slider, e.g. 2.5, 5, 10, 20, 40, 63")
        objs.addWidget(self.objectives, 1)
        lay.addLayout(objs)
        nominal = box("h", "s")                         # the scale before any calibration: the sensor's own pixel
        self.pixel_um = QtWidgets.QDoubleSpinBox()
        self.pixel_um.setRange(0.0, 50.0)
        self.pixel_um.setDecimals(3)
        self.pixel_um.setSingleStep(0.05)
        self.pixel_um.setSpecialValueText("unknown")
        self.pixel_um.setSuffix(" um camera pixel")
        self.pixel_um.setToolTip("The sensor's pixel, from the camera's specifications (INFINITY8-3M: 4.5 um, filled "
                                 "in by itself)")
        self.adapter = QtWidgets.QDoubleSpinBox()
        self.adapter.setRange(0.1, 5.0)
        self.adapter.setDecimals(3)
        self.adapter.setSingleStep(0.01)
        self.adapter.setValue(1.0)
        self.adapter.setPrefix("adapter ")
        self.adapter.setSuffix("x")
        self.adapter.setToolTip("The camera adapter's magnification, between the microscope and the camera (printed on "
                                "it: often 1x, 0.63x or 0.5x)")
        for w in (self.pixel_um, self.adapter):
            nominal.addWidget(w)
        nominal.addStretch(1)
        lay.addLayout(nominal)
        self.pixel_um.valueChanged.connect(lambda *_: panel._objective_moved())
        self.adapter.valueChanged.connect(lambda *_: panel._objective_moved())
        self.scale_note = label("", "muted", wrap=True)
        lay.addWidget(self.scale_note)
        self.um.valueChanged.connect(lambda v: panel.set_calibration(v))
        self.objectives.editingFinished.connect(lambda: panel.set_objectives(self.objectives.text()))
        self.bar.toggled.connect(lambda *_: self.changed.emit())
        self.setLayout(lay)
        self._busy = False

    # values both ways, without loops
    def name_orientation(self) -> None:
        """The orientation's words for the plane chosen: which direction is turned, which is put on its side."""
        (a0, _), (b0, _), across = picture.PLANES[self.plane.currentData()]
        naming = self.naming.currentText()
        up, side = (picture.direction_name(d, picture.in_words(naming)) for d in (a0, b0))
        self.up_label.setText(f"{up} points")
        self.rostral.setToolTip(f"Where {up.lower()} is on the screen: degrees clockwise from up (0 up, 90 right, "
                                "180 down, 270 left)")
        self.turn_left.setToolTip(f"{up} a quarter turn anticlockwise")
        self.turn_right.setToolTip(f"{up} a quarter turn clockwise")
        self.medial.setItemText(0, f"{side} to the right of {up.lower()}")
        self.medial.setItemText(1, f"{side} to the left of {up.lower()}")
        self.medial.setToolTip(f"As one faces {up.lower()}: which side is {side.lower()} (the other hemisphere's "
                               "slice, or a picture mirrored, has it on the other side)")
        self.out_of_plane.setText(f"{picture.axis_name(across, naming)} is cut across")

    def _from_slider(self, key: str, value: float) -> None:
        if not self._busy:
            self._set(key, round(value, 2))

    def _set(self, key: str, value) -> None:
        a = dataclasses.replace(self.panel.adjust, **{key: value})
        self.show_adjust(a, apply=True)

    def show_adjust(self, a: picture.Adjust, apply: bool = False) -> None:
        self._busy = True
        for key, (slider, spin) in self.rows.items():
            low, high = picture.RANGES[key]
            v = float(getattr(a, key))
            spin.setValue(v)
            slider.setValue(int(round((v - low) / (high - low) * 1000)))
        self.auto.setChecked(a.auto_levels)
        self._busy = False
        if apply:
            self.panel.set_adjust(a)

    # presets
    def fill_presets(self) -> None:
        self.presets.clear()
        self.presets.addItem("(presets)")
        for name in sorted(self.panel.presets):
            self.presets.addItem(name)

    def _load_preset(self, index: int) -> None:
        name = self.presets.itemText(index)
        if name in self.panel.presets:
            self.show_adjust(picture.preset_from_text(self.panel.presets[name]), apply=True)

    def _save_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Keep these adjustments", "Name (brightfield, GFP, IR...):")
        if ok and name.strip():
            self.panel.presets[name.strip()] = picture.preset_to_text(self.panel.adjust)
            self.panel.save_settings()
            self.fill_presets()
            self.presets.setCurrentText(name.strip())

    def _delete_preset(self) -> None:
        name = self.presets.currentText()
        if name in self.panel.presets:
            del self.panel.presets[name]
            self.panel.save_settings()
            self.fill_presets()


class CameraPanel(QtWidgets.QWidget):
    """The live picture and its controls, inside Live's camera view while Clampwise holds the camera."""

    released = QtCore.Signal()

    def __init__(self, window, folder=lambda: "", make=lucam.Camera) -> None:
        super().__init__()
        self.window, self.folder, self.make = window, folder, make
        self.adjust = picture.Adjust()
        self.corrections = picture.Corrections()
        self.presets: dict[str, str] = {}
        self.last = None                                      # the last frame as taken
        self._thread: CameraThread | None = None
        lay = box("v", "xs")
        self.view = pg.GraphicsLayoutWidget()
        self.box = self.view.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.box.addItem(self.image)
        self.bar_line = pg.PlotCurveItem(pen=pg.mkPen("w", width=4))
        self.bar_text = pg.TextItem("", color="w", anchor=(1, 1))
        for item in (self.bar_line, self.bar_text):
            self.box.addItem(item)
        # the slice's orientation: a cross of the four directions in its plane, the turned one (rostral, or dorsal)
        # the stronger arm (rig.picture.compass); keyed by the names the plane gives them
        arms = [(pg.PlotCurveItem(pen=pg.mkPen("w", width=4 if i == 0 else 2)),
                 pg.TextItem("", color="w", anchor=(0.5, 0.5), fill=pg.mkBrush(0, 0, 0, 120))) for i in range(4)]
        self.compass_lines = {name: line for name, (line, _) in zip(picture.compass(0), arms)}
        self.compass_text = {name: text for name, (_, text) in zip(picture.compass(0), arms)}
        for line, text in arms:
            for item in (line, text):
                self.box.addItem(item)
                item.setVisible(False)
        self.cross = [pg.InfiniteLine(angle=a, pen=pg.mkPen("y", width=1)) for a in (0, 90)]
        for line in self.cross:
            line.setVisible(False)
            self.box.addItem(line)
        self.ruler = None                                     # a line to measure with
        self.ruler_text = pg.TextItem("", color="y", anchor=(0, 1))
        self.box.addItem(self.ruler_text)
        lay.addWidget(self.view, 1)
        controls = QtWidgets.QGridLayout()                 # captions over values, then the actions: fits a narrow view
        controls.setHorizontalSpacing(theme.SPACE["s"])
        controls.setVerticalSpacing(theme.SPACE["xs"])
        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setDecimals(2)
        self.exposure.setRange(0.01, 10000.0)
        self.exposure.setToolTip("Exposure (ms)")
        self.gain = QtWidgets.QDoubleSpinBox()
        self.gain.setDecimals(2)
        self.gain.setToolTip("Gain, as the camera counts it")
        self.average = QtWidgets.QSpinBox()
        self.average.setRange(1, 64)
        self.average.setToolTip("Frames averaged into each picture: more for dim fluorescence (less noise, slower)")
        self.objective = QtWidgets.QSlider(QtCore.Qt.Horizontal)   # the magnification in use: the scale follows it
        self.objective.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.objective.setPageStep(1)
        self.objective_caption = label("Objective", "muted")
        self.objectives = list(picture.OBJECTIVES)
        self.calibrated: dict[float, float] = {}           # um a pixel, each objective measured with
        for k, (caption, w) in enumerate((("Exposure (ms)", self.exposure), ("Gain", self.gain),
                                          ("Average", self.average), (self.objective_caption, self.objective))):
            controls.addWidget(caption if isinstance(caption, QtWidgets.QWidget) else label(caption, "muted"), 0, k)
            controls.addWidget(w, 1, k)
        self.objective.valueChanged.connect(self._objective_moved)
        self.exposure.editingFinished.connect(self._exposure)
        self.gain.editingFinished.connect(self._exposure)
        self.average.valueChanged.connect(self._average)
        actions = box("h", "s")
        actions.addWidget(widgets.button("Adjust...", self.open_adjust, "Brightness, contrast, gamma, sharpness, "
                                                                      "clarity, colour, presets, the scale bar"))
        actions.addWidget(widgets.button("Auto exposure", self.auto_exposure,
                                         "The brightest 0.5% of the picture to 80% of full scale, nothing clipped"))
        actions.addWidget(widgets.button("Snapshot", self.snapshot, "Save the picture as taken into the cell's folder "
                                                                    "(and as shown, when anything is changed)"))
        self.measure = widgets.button("Measure", self.toggle_measure, "A line to drag over the picture: its length in "
                                                                      "um (with the calibration)", quiet=True)
        self.measure.setCheckable(True)
        actions.addWidget(self.measure)
        actions.addWidget(widgets.button("Fit", lambda: self.box.autoRange(padding=0), "The whole picture again "
                                                                                       "(the wheel zooms, a drag pans)",
                                         quiet=True))
        actions.addStretch(1)
        self.fps = label("", "muted")             # the camera's frame rate
        self.fps.setToolTip("Pictures a second: as the camera takes them (the exposure and averaging set it), and as "
                            "drawn here (at most 10 a second)")
        actions.addWidget(self.fps)
        lay.addLayout(controls)
        lay.addLayout(actions)
        self.status = label("", "muted", wrap=True)
        lay.addWidget(self.status)
        self.setLayout(lay)
        self._shown = 0                                       # pictures drawn (the frame rate's other count)
        self._fps_timer = QtCore.QTimer(self)
        self._fps_timer.timeout.connect(self._rate)
        self._loading = True                                  # nothing is saved while the saved values come back
        self.dialog = AdjustDialog(self)
        self.dialog.changed.connect(self._calibration)
        self.set_objectives("")                                # the standard objectives until the saved ones load
        self.objective.setValue(self.objectives.index(20.0))
        self.load_settings()
        self._loading = False

    # ---- taking and letting go
    @property
    def running(self) -> bool:
        return self._thread is not None

    def take(self) -> None:
        if self._thread is not None:
            return
        self.status.setText("Opening the camera...")
        t = CameraThread(self.make)
        t.adjust, t.average, t.corrections = self.adjust, self.average.value(), self.corrections
        t.opened.connect(self._opened)
        t.frame.connect(self._show)
        t.captured.connect(self._captured)
        t.exposure_set.connect(lambda ms: (self.exposure.blockSignals(True), self.exposure.setValue(ms),
                                           self.exposure.blockSignals(False), self.status.setText(
                                               f"Auto exposure: {ms:.2f} ms.")))
        t.failed.connect(self._failed)
        t.finished.connect(self._finished)
        self._thread = t
        t.start()
        self._fps_at, self._fps_taken, self._fps_shown = time.monotonic(), 0, self._shown
        self._fps_timer.start(1000)

    def release(self) -> None:
        self._fps_timer.stop()
        self.fps.setText("")
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        self.status.setText("")
        self.released.emit()

    def _opened(self, info) -> None:
        for spin, s in ((self.exposure, info.get("exposure")), (self.gain, info.get("gain"))):
            if s is not None:
                spin.blockSignals(True)
                spin.setRange(s.low, s.high)
                spin.setValue(s.value)
                spin.blockSignals(False)
        self.status.setText(f"The camera (serial {info.get('serial')}) is held by Clampwise; Let go gives it back.")

    def _failed(self, why: str) -> None:
        self.status.setText(f"The camera: {why}.")
        self.window.say(f"The camera: {why}.")

    def _finished(self) -> None:
        if self._thread is not None and self._thread.isFinished():
            self._thread = None
            self._fps_timer.stop()
            self.fps.setText("")
            self.released.emit()

    def _exposure(self) -> None:
        if self._thread is not None:
            self._thread.exposure_ms, self._thread.gain = self.exposure.value(), self.gain.value()

    def _average(self, n: int) -> None:
        if self._thread is not None:
            self._thread.average = n

    # ---- the picture
    def _rate(self) -> None:
        """The frame rate, once a second: pictures taken by the camera, and drawn."""
        t = self._thread
        now = time.monotonic()
        if t is None:
            self.fps.setText("")
            return
        dt = now - self._fps_at
        if dt > 0:
            taken = (t.taken - self._fps_taken) / dt
            shown = (self._shown - self._fps_shown) / dt
            self.fps.setText(f"{taken:.1f} fps" + (f" ({shown:.0f} shown)" if shown < taken - 0.5 else ""))
        self._fps_at, self._fps_taken, self._fps_shown = now, t.taken, self._shown

    def _show(self, frame, shown, stats=None) -> None:
        self._shown += 1
        try:
            self._draw(frame, shown, stats)
        finally:
            if self._thread is not None:
                self._thread.drawing = False

    def _draw(self, frame, shown, stats=None) -> None:
        self.last = frame
        table = None
        if isinstance(shown, tuple):                  # a grey frame and the table it is drawn through
            shown, table = shown
        self.shown = shown
        # a grey picture drawn as one channel (a monochrome camera's, untinted): a third of the work to draw; and only
        # as many pixels as the screen shows (zoomed out, every 2nd to 4th), laid over the picture's full size so
        # every coordinate (the scale bar, measuring) stays the camera's own; zoomed in, every pixel
        img = shown[:, :, 0] if picture.is_grey(shown) else shown
        self.image.setLookupTable(table if table is not None and img.ndim == 2 else None)
        step = 1
        vb = self.image.getViewBox()
        if vb is not None:
            try:
                step = int(max(1.0, min(4.0, vb.viewPixelSize()[0])))
            except Exception:                    # the view not laid out yet
                step = 1
        self.image.setImage(img[::step, ::step] if step > 1 else img, autoLevels=False, levels=(0, 255))
        self.image.setRect(QtCore.QRectF(0, 0, shown.shape[1], shown.shape[0]))
        self._place_bar()
        if stats is not None:
            edges = np.linspace(0, 256, len(stats["hist"]) + 1)
            self.dialog.hist_curve.setData(edges, stats["hist"])
            if stats["clipped"] > 0.001:
                self._clip_said = True
                self.status.setText(f"{stats['clipped']:.1%} of the picture is clipped (at full scale): lower the "
                                    "exposure or the gain, or Auto exposure.")
            elif getattr(self, "_clip_said", False):
                self._clip_said = False
                self.status.setText("")

    # ---- the microscopy tools
    def capture(self, kind: str) -> None:
        if self._thread is None:
            self.window.say("Take the camera first.")
            return
        self._thread.request = (kind, FRAMES[kind])
        self.status.setText({"dark": "Taking the dark frame (the light should be blocked)...",
                             "background": "Taking the background (defocused, or an empty field)...",
                             "overlay": "Keeping this picture as the overlay..."}[kind])

    def _captured(self, kind: str, kept) -> None:
        setattr(self.corrections, kind, kept)
        if kind == "background" and self.corrections.mode == "off":
            self.dialog.mode.setCurrentIndex(1)               # taken to be used: subtract it, unless told otherwise
        self._correction_states()
        self.status.setText({"dark": "Dark frame kept.", "background": "Background kept.",
                             "overlay": "Overlay kept: it is laid over the live picture."}[kind])

    def forget(self, kind: str) -> None:
        setattr(self.corrections, kind, None)
        self._correction_states()

    def _correction_states(self) -> None:
        self.dialog.dark_state.setText("kept" if self.corrections.dark is not None else "none")
        self.dialog.bg_state.setText("kept" if self.corrections.background is not None else "none")

    def set_correction(self, **values) -> None:
        for k, v in values.items():
            setattr(self.corrections, k, v)

    def auto_exposure(self) -> None:
        if self._thread is None:
            self.window.say("Take the camera first.")
            return
        self.status.setText("Auto exposure...")
        self._thread.request = ("auto",)

    def set_crosshair(self, on: bool) -> None:
        h, w = (self.last.rgb.shape[:2] if self.last is not None else (0, 0))
        for line, pos in zip(self.cross, (h / 2, w / 2)):
            line.setPos(pos)
            line.setVisible(on)

    def toggle_measure(self) -> None:
        if self.ruler is not None:
            self.box.removeItem(self.ruler)
            self.ruler = None
            self.ruler_text.setText("")
            self.measure.setChecked(False)
            return
        h, w = (self.last.rgb.shape[:2] if self.last is not None else (100, 100))
        self.ruler = pg.LineSegmentROI([[w * 0.4, h * 0.5], [w * 0.6, h * 0.5]], pen=pg.mkPen("y", width=2))
        self.box.addItem(self.ruler)
        self.ruler.sigRegionChanged.connect(self._measured)
        self.measure.setChecked(True)
        self._measured()

    def _measured(self) -> None:
        if self.ruler is None:
            return
        (x1, y1), (x2, y2) = self._ruler_ends()
        px = float(np.hypot(x2 - x1, y2 - y1))
        um = self.scale()[0]
        self.ruler_text.setText(f"{px * um:.1f} um" if um else f"{px:.0f} px (not calibrated)")
        self.ruler_text.setPos(max(x1, x2), min(y1, y2))

    def _ruler_ends(self):
        return [(p.x(), p.y()) for p in (self.ruler.mapToParent(h["item"].pos()) for h in self.ruler.handles)]

    def _ruler_px(self) -> float:
        (x1, y1), (x2, y2) = self._ruler_ends()
        return float(np.hypot(x2 - x1, y2 - y1))

    def set_adjust(self, a: picture.Adjust) -> None:
        self.adjust = a
        if self._thread is not None:
            self._thread.adjust = a
        self.save_settings()

    def white_balance(self) -> None:
        if self.last is None:
            self.window.say("Take the camera first: white balance needs a picture (an empty, bright field).")
            return
        r, g, b = picture.white_balance(self.last.rgb)
        self.dialog.show_adjust(dataclasses.replace(self.adjust, red=round(r, 3), green=round(g, 3),
                                                    blue=round(b, 3)), apply=True)

    # ---- the scale: the objective in use and its calibration
    @property
    def magnification(self) -> float:
        return self.objectives[min(max(self.objective.value(), 0), len(self.objectives) - 1)]

    def scale(self) -> tuple[float, str]:
        """um a pixel with the objective in use, and how it is known."""
        return picture.objective_scale(self.magnification, self.calibrated, self.dialog.pixel_um.value(),
                                       self.dialog.adapter.value())

    def set_objectives(self, text: str) -> None:
        keep = self.magnification if self.objectives else None
        self.objectives = picture.parse_objectives(text)
        self.objective.blockSignals(True)
        self.objective.setRange(0, len(self.objectives) - 1)
        self.objective.setValue(min(range(len(self.objectives)),
                                    key=lambda i: abs(self.objectives[i] - (keep or self.objectives[0]))))
        self.objective.blockSignals(False)
        self.dialog.objectives.setText(", ".join(f"{m:g}" for m in self.objectives))
        self._objective_moved()

    def set_calibration(self, um: float) -> None:
        """The objective in use measured at ``um`` a pixel (0: none; then taken from another objective's)."""
        if self.dialog.um.signalsBlocked():
            return
        if um > 0:
            self.calibrated[self.magnification] = float(um)
        else:
            self.calibrated.pop(self.magnification, None)
        self._objective_moved()

    def calibrate_from_ruler(self) -> None:
        if self.ruler is None:
            self.toggle_measure()
            self.window.say("Drag the Measure line along a stage micrometer's divisions, then choose From the "
                            "Measure line again.")
            return
        px = self._ruler_px()
        if px < 5:
            self.window.say("The Measure line is too short to calibrate with.")
            return
        um, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibrate this objective", f"The Measure line is {px:.1f} px long at {self.magnification:g}x. "
            "Its true length (um):", 100.0, 0.001, 100000.0, 3)
        if ok:
            self.dialog.um.setValue(um / px)            # set_calibration then keeps it for this objective

    def _objective_moved(self, *_) -> None:
        um, how = self.scale()
        mag = self.magnification
        # the size of one pixel at this objective, on the panel itself (what Measure and the scale bar use)
        self.objective_caption.setText(f"Objective {mag:g}x" + (f"  ({um:.3g} um/px)" if um else ""))
        self.objective.setToolTip(f"{mag:g}x: " + (f"{um:.4g} um a pixel, {how}" if um else
                                                    "not calibrated (Adjust..., Length calibration)"))
        self.dialog.um.blockSignals(True)
        self.dialog.um.setValue(self.calibrated.get(mag, 0.0))
        self.dialog.um.blockSignals(False)
        self.dialog.scale_note.setText(f"{mag:g}x: " + (f"{um:.4g} um a pixel, {how}." if um else
                                       "not known: give the camera's pixel and adapter below, or measure a stage "
                                       "micrometer at any objective (From the Measure line); the others follow by "
                                       "their magnifications."))
        self._calibration()

    def _calibration(self) -> None:
        self._place_bar()
        self._measured()
        self.save_settings()

    def orientation(self) -> tuple[int, bool]:
        """(the plane's turned direction, rostral or dorsal: degrees clockwise from up on the screen; the second
        axis's first end, medial or rostral, to the right of it)."""
        return self.dialog.rostral.value(), self.dialog.medial.currentIndex() == 0

    def plane(self) -> str:
        return self.dialog.plane.currentData()

    def orientation_text(self) -> str:
        """The slice's plane and orientation in words, as a snapshot records them."""
        (a0, _), (b0, _), across = picture.PLANES[self.plane()]
        naming = self.dialog.naming.currentText()
        up, side = (picture.direction_name(d, picture.in_words(naming)).lower() for d in (a0, b0))
        deg, right = self.orientation()
        return (f"{self.plane()} slice: {up} {deg} deg clockwise from up, {side} to its {'right' if right else 'left'}"
                f"; {picture.axis_name(across, naming)} cut across")

    def _orientation_changed(self) -> None:
        self._place_compass()
        self.save_settings()

    def _place_compass(self) -> None:
        h, w = (self.last.rgb.shape[:2] if self.last is not None else (0, 0))
        show = bool(w and self.dialog.compass.isChecked())
        deg, right = self.orientation()
        directions = picture.compass(deg, right, self.plane(), self.dialog.naming.currentText())
        lines, texts = list(self.compass_lines.values()), list(self.compass_text.values())
        self.compass_lines, self.compass_text = dict(zip(directions, lines)), dict(zip(directions, texts))
        arm = 0.08 * min(w, h)                              # in the picture's top-left corner, clear of its edge
        cx, cy = 0.04 * w + 1.6 * arm, 0.05 * h + 1.6 * arm
        for name, (dx, dy) in directions.items():
            line, text = self.compass_lines[name], self.compass_text[name]
            line.setVisible(show)
            text.setVisible(show)
            text.setText(name)
            if show:
                line.setData([cx, cx + arm * dx], [cy, cy + arm * dy])
                text.setPos(cx + 1.45 * arm * dx, cy + 1.45 * arm * dy)

    def _place_bar(self) -> None:
        self._place_compass()
        um = self.scale()[0]
        h, w = (self.last.rgb.shape[:2] if self.last is not None else (0, 0))
        show = bool(um and w and self.dialog.bar.isChecked())
        self.bar_line.setVisible(show)
        self.bar_text.setVisible(show)
        if show:
            length_um, length_px = picture.scale_bar(um, w)
            x1, y = w * 0.96, h * 0.95
            self.bar_line.setData([x1 - length_px, x1], [y, y])
            self.bar_text.setText(f"{length_um:g} um")
            self.bar_text.setPos(x1, y - h * 0.01)

    def open_adjust(self) -> None:
        self.dialog.fill_presets()
        self.dialog.show_adjust(self.adjust)
        self.dialog.show()
        self.dialog.raise_()

    # ---- saving
    def snapshot(self) -> Path | None:
        if self.last is None:
            self.window.say("No picture yet: take the camera first.")
            return None
        folder = self.folder() or ""
        if not folder or not Path(folder).is_dir():
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save the snapshot in", "")
            if not folder:
                return None
        f = self.last
        um, how = self.scale()
        um = um or None
        info = {"camera_serial": f.serial, "exposure_ms": f.exposure_ms, "gain": f.gain, "averaged": f.averaged,
                "objective": f"{self.magnification:g}x", "um_per_px": um, "scale": how,
                "slice_plane": self.plane(), "orientation": self.orientation_text(),
                "shown_with": self.adjust.changed(), "corrections_shown": self.corrections.active(),
                "saved_by": "Clampwise", "taken": time.strftime("%Y-%m-%d %H:%M:%S")}
        label_ = " ".join(x for x in (f"{self.magnification:g}x", self.dialog.presets.currentText()
                                      if self.dialog.presets.currentText() in self.presets else "") if x)
        path = picture.write_tiff(picture.snapshot_name(folder, label_), np.ascontiguousarray(f.rgb), info, um)
        if not self.adjust.neutral() or self.corrections.active():
            # the original stays as taken (Rossner & Yamada 2004); what was on screen is kept beside it
            shown = np.ascontiguousarray(picture.shown(f.rgb, self.corrections, self.adjust))
            from pyqtgraph.Qt import QtGui

            QtGui.QImage(shown.data, shown.shape[1], shown.shape[0], shown.shape[1] * 3,
                         QtGui.QImage.Format_RGB888).copy().save(str(path.with_name(path.stem + "_as_shown.png")))
        self.window.say(f"Snapshot saved: {path}")
        self.status.setText(f"Saved {path.name} in {path.parent}.")
        return path

    # ---- kept for the user
    def load_settings(self) -> None:
        if not getattr(self.window, "persist", False):
            return
        s = self.window._settings()
        try:
            self.presets = json.loads(str(s.value("camera/presets", "{}") or "{}"))
            text = str(s.value("camera/adjust", "") or "")
            if text:
                self.adjust = picture.preset_from_text(text)
        except (ValueError, TypeError):
            self.presets = {}
        try:
            self.calibrated = {float(k): float(v) for k, v in
                               json.loads(str(s.value("camera/calibrated", "{}") or "{}")).items()}
        except (ValueError, TypeError):
            self.calibrated = {}
        old_um, old_obj = float(s.value("camera/um_per_px", 0.0) or 0.0), str(s.value("camera/objective", "") or "")
        if old_um > 0 and not self.calibrated:                   # a calibration from before the slider
            mags = picture.parse_objectives(old_obj)
            if old_obj.strip() and len(mags) == 1:
                self.calibrated[mags[0]] = old_um
        self.dialog.pixel_um.blockSignals(True)
        self.dialog.adapter.blockSignals(True)
        self.dialog.pixel_um.setValue(float(s.value("camera/pixel_um", 0.0) or 0.0) or _camera_pixel_um())
        self.dialog.adapter.setValue(float(s.value("camera/adapter", 1.0) or 1.0))
        orientation = (self.dialog.compass, self.dialog.plane, self.dialog.naming, self.dialog.rostral,
                       self.dialog.medial)
        for w_ in orientation:
            w_.blockSignals(True)
        self.dialog.compass.setChecked(str(s.value("camera/compass", "false")).lower() == "true")
        self.dialog.plane.setCurrentIndex(max(0, self.dialog.plane.findData(str(s.value("camera/slice_plane",
                                                                                        "horizontal")))))
        names = str(s.value("camera/compass_names", "") or "")
        self.dialog.naming.setCurrentIndex(max(0, self.dialog.naming.findText(names)))
        self.dialog.rostral.setValue(int(float(s.value("camera/rostral_deg", 0) or 0)))
        self.dialog.medial.setCurrentIndex(0 if str(s.value("camera/medial_right", "true")).lower() == "true" else 1)
        self.dialog.name_orientation()
        for w_ in orientation:
            w_.blockSignals(False)
        self.dialog.pixel_um.blockSignals(False)
        self.dialog.adapter.blockSignals(False)
        self.set_objectives(str(s.value("camera/objectives", "") or ""))
        self.objective.setValue(int(s.value("camera/objective_index", self.objectives.index(20.0)
                                            if 20.0 in self.objectives else 0) or 0))
        self.dialog.bar.setChecked(str(s.value("camera/scale_bar", "true")).lower() == "true")
        self.average.setValue(int(s.value("camera/average", 1) or 1))

    def save_settings(self) -> None:
        if self._loading or not getattr(self.window, "persist", False):
            return
        s = self.window._settings()
        s.setValue("camera/presets", json.dumps(self.presets))
        s.setValue("camera/adjust", picture.preset_to_text(self.adjust))
        s.setValue("camera/calibrated", json.dumps({f"{k:g}": v for k, v in self.calibrated.items()}))
        s.setValue("camera/pixel_um", self.dialog.pixel_um.value())
        s.setValue("camera/adapter", self.dialog.adapter.value())
        s.setValue("camera/compass", self.dialog.compass.isChecked())
        s.setValue("camera/slice_plane", self.plane())
        s.setValue("camera/compass_names", self.dialog.naming.currentText())
        s.setValue("camera/rostral_deg", self.dialog.rostral.value())     # the plane's turned direction
        s.setValue("camera/medial_right", self.dialog.medial.currentIndex() == 0)
        s.setValue("camera/objectives", ", ".join(f"{m:g}" for m in self.objectives))
        s.setValue("camera/objective_index", self.objective.value())
        s.setValue("camera/scale_bar", self.dialog.bar.isChecked())
        s.setValue("camera/average", self.average.value())
