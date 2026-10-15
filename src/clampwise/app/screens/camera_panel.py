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

SHOW_EVERY_S = 0.1                   # the picture is redrawn at most 10 times a second
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
            shown_at = 0.0
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
                if time.monotonic() - shown_at >= SHOW_EVERY_S:
                    shown_at = time.monotonic()
                    self.frame.emit(f, picture.shown(f.rgb, self.corrections, self.adjust), picture.stats(f.rgb))
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
        for w in (self.lut, self.clipped, self.cross):
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
        lay.addWidget(label("Length calibration", "section"))
        cal = box("h", "s")
        self.um = QtWidgets.QDoubleSpinBox()
        self.um.setRange(0.0, 50.0)
        self.um.setDecimals(5)
        self.um.setSingleStep(0.001)
        self.um.setSpecialValueText("not calibrated")
        self.um.setSuffix(" um a pixel")
        self.um.setToolTip("From a stage micrometer's picture (tools/rig/calibrate_camera.py); 0: not calibrated")
        self.objective = QtWidgets.QLineEdit()
        self.objective.setPlaceholderText("objective (40x)")
        self.bar = QtWidgets.QCheckBox("Scale bar")
        for w in (self.um, self.objective, self.bar):
            cal.addWidget(w)
        self.um.valueChanged.connect(lambda *_: self.changed.emit())
        self.objective.textChanged.connect(lambda *_: self.changed.emit())
        self.bar.toggled.connect(lambda *_: self.changed.emit())
        lay.addLayout(cal)
        self.setLayout(lay)
        self._busy = False

    # values both ways, without loops
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
        for k, (caption, w) in enumerate((("Exposure (ms)", self.exposure), ("Gain", self.gain),
                                          ("Average", self.average))):
            controls.addWidget(label(caption, "muted"), 0, k)
            controls.addWidget(w, 1, k)
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
        self.last = frame
        self.shown = shown
        self.image.setImage(shown, autoLevels=False, levels=(0, 255))
        self._calibration()
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
        (x1, y1), (x2, y2) = [(p.x(), p.y()) for p in (self.ruler.mapToParent(h["item"].pos())
                                                       for h in self.ruler.handles)]
        px = float(np.hypot(x2 - x1, y2 - y1))
        um = self.dialog.um.value()
        self.ruler_text.setText(f"{px * um:.1f} um" if um else f"{px:.0f} px (not calibrated)")
        self.ruler_text.setPos(max(x1, x2), min(y1, y2))

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

    def _calibration(self) -> None:
        um = self.dialog.um.value()
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
        self.save_settings()

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
        um = self.dialog.um.value() or None
        info = {"camera_serial": f.serial, "exposure_ms": f.exposure_ms, "gain": f.gain, "averaged": f.averaged,
                "objective": self.dialog.objective.text().strip(), "um_per_px": um,
                "shown_with": self.adjust.changed(), "corrections_shown": self.corrections.active(),
                "saved_by": "Clampwise", "taken": time.strftime("%Y-%m-%d %H:%M:%S")}
        label_ = " ".join(x for x in (self.dialog.objective.text().strip(), self.dialog.presets.currentText()
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
        self.dialog.um.setValue(float(s.value("camera/um_per_px", 0.0) or 0.0))
        self.dialog.objective.setText(str(s.value("camera/objective", "") or ""))
        self.dialog.bar.setChecked(str(s.value("camera/scale_bar", "true")).lower() == "true")
        self.average.setValue(int(s.value("camera/average", 1) or 1))

    def save_settings(self) -> None:
        if self._loading or not getattr(self.window, "persist", False):
            return
        s = self.window._settings()
        s.setValue("camera/presets", json.dumps(self.presets))
        s.setValue("camera/adjust", picture.preset_to_text(self.adjust))
        s.setValue("camera/um_per_px", self.dialog.um.value())
        s.setValue("camera/objective", self.dialog.objective.text())
        s.setValue("camera/scale_bar", self.dialog.bar.isChecked())
        s.setValue("camera/average", self.average.value())
