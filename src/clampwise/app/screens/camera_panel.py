"""The camera in Clampwise: Live's camera view can take the camera itself while the camera's own program is closed
(rig.cameras: a Lumenera INFINITY through rig.lucam, a Basler through rig.basler) and show its live picture, raw and
grey: exactly what the camera gives, nothing computed on it, so every frame the camera takes can be shown. Its
controls are few: the frame rate (and the one the camera gives now), exposure and gain, auto exposure (which follows
the light as it changes, so more light gives more pictures a second), the objective (the scale bar, Measure and the
patterned light's calibration follow it), Measure, the slice's compass over the picture, a snapshot into the
cell's folder. Frames are taken off the
window's thread; the camera is let go with one click (or when Clampwise closes)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import cameras, lucam, picture
from .. import theme, widgets
from ..widgets import box, label

SHOW_EVERY_S = 1 / 30                # at most 30 redraws a second (smooth to the eye; 60 doubled the drawing for
                                     # nothing seen), and only once the last one is drawn
FPS_SHARE = 0.97                     # of each frame's time, the exposure (the rest: the camera reading the sensor out)
FOLLOW_EVERY_S = 0.5                 # auto exposure, on: how often it looks at the picture and corrects the exposure
FOLLOW_STEP = 2.0                    # each correction at most twice or half the exposure (no jumping about)


def fps_limit(taken: float, shown: float, exposure_ms: float, average: int, draw_cap: float) -> str:
    """What sets the pictures a second: the exposure (a picture cannot come faster than its exposure), the camera's
    readout (it sends its frames no faster at this size), or the drawing."""
    if taken <= 0:
        return "No pictures yet."
    by_exposure = 1000.0 / max(exposure_ms * max(1, average), 1e-3)
    if taken >= 0.85 * by_exposure:
        why = (f"The exposure sets it: {exposure_ms:g} ms allows {by_exposure:.0f} a second; a shorter exposure (more "
               "light, or more gain to keep the picture bright) gives more.")
    else:
        why = (f"The camera's readout sets it ({taken:.0f} a second at this size; the exposure would allow "
               f"{by_exposure:.0f}): its frames come no faster.")
    if shown < taken - 0.5:
        why += (f" {shown:.0f} a second are drawn: " + (f"the drawing is capped at {draw_cap:.0f} a second (the eye "
                                                         "sees no more; low-power mode in Settings: 15); every frame "
                                                         "is still taken." if shown >= 0.9 * draw_cap else
                                                         "this PC draws no faster."))
    return why


def grey(f: lucam.Frame) -> lucam.Frame:
    """The frame in grey: a colour camera's brightness, a monochrome one's own channel (as it is, no copy)."""
    if picture.is_grey(f.rgb):
        return f
    g = picture.lum8(f.rgb)
    return lucam.Frame(np.broadcast_to(g[:, :, None], g.shape + (3,)), f.exposure_ms, f.gain, f.averaged, f.serial,
                       list(f.notes))


class CameraThread(QtCore.QThread):
    """Takes frames until stopped; the camera is opened and closed in this thread."""

    opened = QtCore.Signal(object)           # {"exposure": Setting, "gain": Setting, "serial": int, "model": str, ...}
    frame = QtCore.Signal(object)            # the frame, grey, as taken
    exposure_set = QtCore.Signal(float, str)  # auto exposure's choice, and a note when it could not reach it
    failed = QtCore.Signal(str)

    def __init__(self, make=cameras.make) -> None:
        super().__init__()
        self.make = make
        self.exposure_ms: float | None = None
        self.gain: float | None = None
        self.average = 1                     # a frame each picture (the patterned light's settling reads it)
        self.follow = False                  # auto exposure on: the exposure follows the light
        self.follow_paused = False           # held still meanwhile (the patterned light's calibration)
        self.max_exposure_ms: float | None = None   # the frame rate asked for allows no longer (auto exposure too)
        self.taken = 0                       # pictures taken: the frame rate's count
        self.drawing = False                 # a picture sent and not yet drawn: none other is sent meanwhile
        self.show_every_s = SHOW_EVERY_S     # low-power mode: fewer
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        self.wait(10000)

    def run(self) -> None:
        try:                                     # whatever the camera's library does, the reason is said
            cam = self.make()
            opened = cam.open()
        except Exception as exc:
            logging.getLogger(__name__).exception("The camera could not be taken")
            self.failed.emit(f"it could not be taken ({type(exc).__name__}: {exc})")
            return
        if not opened:
            self.failed.emit(cam.status or "it could not be taken (no reason given)")
            return
        try:
            exposure = cam.setting(lucam.EXPOSURE)
            self.opened.emit({"exposure": exposure, "gain": cam.setting(lucam.GAIN), "serial": cam.serial,
                              "model": getattr(cam, "model", ""), "gain_kind": getattr(cam, "gain_kind", "raw")})
            shown_at = followed_at = 0.0
            while not self._stop:
                want = (cam.exposure_ms if self.exposure_ms is None else self.exposure_ms,     # a gain of 0 dB is
                        cam.gain if self.gain is None else self.gain)                          # a gain asked for
                if want != (cam.exposure_ms, cam.gain):
                    cam.set_exposure(*want)
                f = cam.frame()
                self.taken += 1
                if f is None:
                    self.failed.emit(cam.status)
                    break
                f = grey(f)
                now = time.monotonic()
                if self.follow and not self.follow_paused and now - followed_at >= FOLLOW_EVERY_S:
                    followed_at = now
                    self._follow(cam, f, exposure)
                # every frame the screen keeps up with, never a queue of frames behind the picture (80% of the
                # interval: frames 16.7 ms apart then give 30 a second drawn, not 21 for missing by one)
                if not self.drawing and now - shown_at >= 0.8 * self.show_every_s:
                    shown_at = now
                    self.drawing = True
                    self.frame.emit(f)
        except Exception as exc:
            logging.getLogger(__name__).exception("The camera stopped")
            self.failed.emit(f"it stopped ({type(exc).__name__}: {exc})")
        finally:
            try:
                cam.close()
            except Exception:
                logging.getLogger(__name__).exception("The camera could not be let go")

    def _follow(self, cam, f, exposure) -> None:
        """Auto exposure, one correction: the brightest 0.5% of the picture (every 4th pixel looked at) brought
        toward 80% of full scale, at most twice or half the exposure at a time, never longer than the frame rate asked
        for allows; more light, a shorter exposure, more pictures a second."""
        low, high = (exposure.low, exposure.high) if exposure is not None else (0.01, 10000.0)
        high = min(high, max(picture.AUTO_MAX_MS, low))
        capped = self.max_exposure_ms is not None and self.max_exposure_ms < high
        if capped:
            high = max(self.max_exposure_ms, low)
        s = picture.stats(f.rgb[::4, ::4])
        new = picture.next_exposure(cam.exposure_ms, s["p995"], low, high, s["clipped"], FOLLOW_STEP)
        if abs(new - cam.exposure_ms) <= 0.08 * cam.exposure_ms and s["clipped"] <= 0.001:
            return
        note = ""
        if new >= high and s["p995"] < 0.5 * picture.TARGET:
            note = (f" Dark at {high:.3g} ms, the longest the frame rate allows: raise the gain or lower the frame "
                    "rate." if capped else f" Dark at {high:g} ms: is the light on and its path open?")
        if abs(new - cam.exposure_ms) > 1e-6:
            cam.set_exposure(new)
            self.exposure_ms = new
            self.exposure_set.emit(new, note)


def _camera_pixel_um() -> float:
    """The sensor's pixel for the camera its program last named (Infinity Analyze's settings), else 0."""
    try:
        from ...rig.camera import read_settings

        s = read_settings()
        return picture.pixel_um_of(s.camera) if s is not None else 0.0
    except Exception:
        return 0.0


class ScaleDialog(QtWidgets.QDialog):
    """The picture's scale: um a pixel at each objective (measured on a stage micrometer, or from the camera's own
    pixel and its adapter), the objectives on the panel's slider, the scale bar; and the slice's plane and orientation,
    which the compass over the picture shows."""

    def __init__(self, panel: "CameraPanel") -> None:
        super().__init__(panel)
        self.setWindowTitle("The camera's scale and the slice")
        self.setModal(False)
        lay = box("v", "s", "m")
        cal = box("h", "s")
        self.um = QtWidgets.QDoubleSpinBox()             # the calibration of the objective chosen on the panel
        self.um.setRange(0.0, 50.0)
        self.um.setDecimals(5)
        self.um.setSingleStep(0.001)
        self.um.setSpecialValueText("not measured here")
        self.um.setSuffix(" um a pixel")
        self.um.setToolTip("Measured at the objective chosen on the camera's panel (a stage micrometer: From the "
                           "Measure line); 0: taken from another objective's calibration by the ratio of magnifications")
        cal.addWidget(self.um)
        cal.addWidget(widgets.button("From the Measure line...", panel.calibrate_from_ruler,
                                     "Drag the Measure line along a stage micrometer, then type its true length"))
        lay.addLayout(cal)
        self.bar = QtWidgets.QCheckBox("Scale bar on the picture")
        self.bar.setChecked(True)                        # shown as soon as there is a calibration
        lay.addWidget(self.bar)
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
        self.pixel_um.setToolTip("The sensor's pixel, from the camera's specifications (INFINITY8-3M: 4.5 um, Basler "
                                 "acA1920-40: 5.86 um; filled in by itself)")
        self.adapter = QtWidgets.QDoubleSpinBox()
        self.adapter.setRange(0.1, 5.0)
        self.adapter.setDecimals(3)
        self.adapter.setSingleStep(0.01)
        self.adapter.setValue(1.0)
        self.adapter.setPrefix("adapter ")
        self.adapter.setSuffix("x")
        self.adapter.setToolTip("The camera adapter's magnification, between the microscope and the camera (printed on "
                                "it: often 1x, 0.63x or 0.5x)")
        nominal.addWidget(self.pixel_um)
        nominal.addWidget(self.adapter)
        nominal.addStretch(1)
        lay.addLayout(nominal)
        self.scale_note = label("", "muted", wrap=True)
        lay.addWidget(self.scale_note)
        lay.addWidget(label("The slice", "section"))
        self.compass = QtWidgets.QCheckBox("Compass on the picture")
        self.compass.setToolTip("The slice's directions in its plane over the picture (its plane and orientation "
                                "below)")
        self.compass.toggled.connect(lambda *_: panel._orientation_changed())
        lay.addWidget(self.compass)
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
        cut.addStretch(1)
        lay.addLayout(cut)
        self.out_of_plane = label("", "muted")
        lay.addWidget(self.out_of_plane)
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
        orient.addStretch(1)
        lay.addLayout(orient)
        self.medial = QtWidgets.QComboBox()                  # the side of it the second axis's first end is on
        self.medial.addItems(["", ""])
        lay.addWidget(self.medial)
        self.name_orientation()
        self.plane.currentIndexChanged.connect(lambda *_: (self.name_orientation(), panel._orientation_changed()))
        self.naming.currentIndexChanged.connect(lambda *_: (self.name_orientation(), panel._orientation_changed()))
        self.rostral.valueChanged.connect(lambda *_: panel._orientation_changed())
        self.medial.currentIndexChanged.connect(lambda *_: panel._orientation_changed())
        lay.addStretch(1)
        self.pixel_um.valueChanged.connect(lambda *_: panel._objective_moved())
        self.adapter.valueChanged.connect(lambda *_: panel._objective_moved())
        self.um.valueChanged.connect(lambda v: panel.set_calibration(v))
        self.objectives.editingFinished.connect(lambda: panel.set_objectives(self.objectives.text()))
        self.bar.toggled.connect(lambda *_: panel._calibration())
        widgets.resizable(self, lay, "camera_scale", (480, 460))

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


class CameraPanel(QtWidgets.QWidget):
    """The live picture, inside Live's camera view while Clampwise holds the camera; its controls (``controls``) are
    placed by Live apart from it."""

    released = QtCore.Signal()

    def __init__(self, window, folder=lambda: "", make=cameras.make) -> None:
        super().__init__()
        self.window, self.folder, self.make = window, folder, make
        self.last = None                                      # the last frame as taken
        self.failure = ""
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
        self.ruler = None                                     # a line to measure with
        self.ruler_text = pg.TextItem("", color="y", anchor=(0, 1))
        self.box.addItem(self.ruler_text)
        controls = QtWidgets.QGridLayout()                 # captions over values, two across: fits Live's side panel
        controls.setHorizontalSpacing(theme.SPACE["s"])
        controls.setVerticalSpacing(theme.SPACE["xs"])
        self.frame_rate = QtWidgets.QDoubleSpinBox()      # pictures a second: the exposure's longest follows
        self.frame_rate.setDecimals(1)
        self.frame_rate.setRange(0.0, 500.0)
        self.frame_rate.setSingleStep(5.0)
        self.frame_rate.setSuffix(" fps")
        self.frame_rate.setSpecialValueText("fastest")
        self.frame_rate.setToolTip("The pictures a second wanted: the exposure is kept short enough for them, and the "
                                   "gain raised to keep the picture as bright. 0: the exposure alone sets the pace. "
                                   "The camera's readout sets the most it can give (hover the frame rate shown).")
        self.frame_rate.editingFinished.connect(self._frame_rate)
        self.gain_kind = "raw"                              # how the camera counts gain (rig.picture.gain_for)
        self._exposure_high = 10000.0                       # the camera's longest exposure
        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setDecimals(2)
        self.exposure.setRange(0.01, 10000.0)
        self.exposure.setToolTip("Exposure (ms): at most what the frame rate allows")
        self.gain = QtWidgets.QDoubleSpinBox()
        self.gain.setDecimals(2)
        self.gain.setToolTip("Gain, as the camera counts it")
        self.auto = QtWidgets.QCheckBox("Auto exposure: follows the light")
        self.auto.setChecked(True)
        self.auto.setToolTip("The exposure kept so the brightest of the picture is at 80% of full scale, corrected "
                             "twice a second as the light changes: more light, a shorter exposure, more pictures a "
                             "second. Within the frame rate asked for. Unticked: the exposure stays as typed.")
        self.auto.toggled.connect(self._auto)
        self.objective = QtWidgets.QSlider(QtCore.Qt.Horizontal)   # the magnification in use: the scale follows it
        self.objective.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.objective.setPageStep(1)
        self.objective_caption = label("Objective", "muted")
        self.objectives = list(picture.OBJECTIVES)
        self.calibrated: dict[float, float] = {}           # um a pixel, each objective measured with
        self.fps = label("", "heading")              # the pictures a second the camera gives now, live
        self.fps.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.fps.setToolTip("Pictures a second: as the camera takes them now, and as drawn here (at most 30 a second)")
        controls.addWidget(label("Frame rate", "muted"), 0, 0)
        controls.addWidget(self.fps, 0, 1)
        controls.addWidget(self.frame_rate, 1, 0, 1, 2)
        controls.addWidget(label("Exposure (ms)", "muted"), 2, 0)
        controls.addWidget(label("Gain", "muted"), 2, 1)
        controls.addWidget(self.exposure, 3, 0)
        controls.addWidget(self.gain, 3, 1)
        controls.addWidget(self.auto, 4, 0, 1, 2)
        controls.addWidget(self.objective_caption, 5, 0, 1, 2)
        controls.addWidget(self.objective, 6, 0, 1, 2)
        self.objective.valueChanged.connect(self._objective_moved)
        self.exposure.editingFinished.connect(self._exposure)
        self.gain.editingFinished.connect(self._exposure)
        actions = box("h", "s")
        actions.addWidget(widgets.button("Snapshot", self.snapshot, "Save the picture as taken into the cell's folder"))
        actions.addWidget(widgets.button("Scale & slice...", self.open_scale, "um a pixel at each objective (the "
                                         "scale bar, Measure, the patterned light's regions in um); the slice's plane "
                                         "and orientation for the compass"))
        tools = actions
        self.measure = widgets.button("Measure", self.toggle_measure, "A line to drag over the picture: its length in "
                                                                      "um (with the calibration)", quiet=True)
        self.measure.setCheckable(True)
        tools.addWidget(self.measure)
        self.compass = widgets.button("Compass", None, "The slice's directions over the picture (its plane and "
                                      "orientation: Scale & slice...)", quiet=True)
        self.compass.setCheckable(True)
        tools.addWidget(self.compass)
        tools.addWidget(widgets.button("Fit", lambda: self.box.autoRange(padding=0), "The whole picture again "
                                                                                     "(the wheel zooms, a drag pans)",
                                       quiet=True))
        tools.addStretch(1)
        self.status = label("", "muted", wrap=True)
        # the controls apart from the picture: Live puts them in its side panel's Camera section, so they stay in
        # reach while the picture is enlarged or in a window of its own
        below = box("v", "xs")
        below.addLayout(controls)
        below.addLayout(actions)
        below.addWidget(self.status)
        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(below)
        lay.addWidget(self.view, 1)
        self.setLayout(lay)
        self._shown = 0                                       # pictures drawn (the frame rate's other count)
        self._fps_timer = QtCore.QTimer(self)
        self._fps_timer.timeout.connect(self._rate)
        self._loading = True                                  # nothing is saved while the saved values come back
        self.dialog = ScaleDialog(self)
        self.compass.toggled.connect(self.dialog.compass.setChecked)          # one switch, in two places
        self.dialog.compass.toggled.connect(self.compass.setChecked)
        self.set_objectives("")                                # the standard objectives until the saved ones load
        self.objective.setValue(self.objectives.index(20.0))
        self.load_settings()
        self.exposure.setEnabled(not self.auto.isChecked())    # typed only while auto exposure is off
        self._loading = False

    # ---- taking and letting go
    @property
    def running(self) -> bool:
        return self._thread is not None

    def take(self) -> None:
        if self._thread is not None:
            return
        self.status.setText("Opening the camera...")
        self.failure = ""
        t = CameraThread(self.make)
        t.follow = self.auto.isChecked()
        self._power(t)
        t.opened.connect(self._opened)
        t.frame.connect(self._show)
        t.exposure_set.connect(self._followed)
        t.failed.connect(self._failed)
        t.finished.connect(self._finished)
        self._thread = t
        t.start()
        self._fps_at, self._fps_taken, self._fps_shown = time.monotonic(), 0, self._shown
        self._fps_timer.start(1000)

    def summary(self) -> tuple[str, str, str]:
        """How the camera stands in a few words (status, words, detail): what a folded section's header shows."""
        if self._thread is None:
            return "muted", "not held", ""
        if self.last is None:
            return "muted", "opening...", self.status.text()
        fps = self.fps.text().split(" (")[0]
        um = self.scale()[0]
        words = ", ".join(x for x in (f"{self.exposure.value():g} ms", fps, f"{self.magnification:g}x") if x)
        return "ok", words, f"{self.magnification:g}x: " + (f"{um:.4g} um a pixel" if um else "not calibrated")

    def _power(self, t=None, on: bool | None = None) -> None:
        """Low-power mode's limits (app.power) on the camera's drawing: ``on`` as chosen, else as kept."""
        from .. import power

        if on is None:
            main = widgets._main_window(self)
            on = power.enabled(main._settings() if main is not None and getattr(main, "persist", False) else None)
        lim = power.limits(on)
        self._min_step = lim["min_step"]
        t = t or self._thread
        if t is not None:
            t.show_every_s = 1.0 / lim["draw_fps"]

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
        if info.get("exposure") is not None:
            self._exposure_high = float(info["exposure"].high)
        self.gain_kind = info.get("gain_kind", "raw")
        self._frame_rate(quiet=True)
        model = info.get("model") or ""
        if model and self.dialog.pixel_um.value() == 0 and picture.pixel_um_of(model) > 0:
            self.dialog.pixel_um.setValue(picture.pixel_um_of(model))       # the nominal scale from the camera's own
        self.status.setText(f"The camera ({model + ', ' if model else ''}serial {info.get('serial')}) is held by "
                            "Clampwise; Let go gives it back.")

    def _failed(self, why: str) -> None:
        self.failure = why                       # said in the camera's view once it is back (rig.Live)
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

    def _frame_rate(self, quiet: bool = False) -> None:
        """The pictures a second asked for: the exposure kept short enough for them, the gain raised to keep the
        picture as bright as it was; 0, the exposure alone sets the pace."""
        fps = self.frame_rate.value()
        longest = min(self._exposure_high, FPS_SHARE * 1000.0 / fps) if fps > 0 else self._exposure_high
        was, said = self.exposure.value(), ""
        if was > longest:
            brighter = was / longest
            gain, got = picture.gain_for(self.gain.value(), brighter, self.gain_kind, self.gain.minimum(),
                                         self.gain.maximum())
            self.gain.setValue(gain)
            said = f"{fps:g} fps: the exposure shortened to {longest:.3g} ms"
            if got >= brighter * 0.99:
                said += f", the gain raised to {gain:.3g} to keep the picture as bright."
            else:
                said += (f"; the picture is {brighter / got:.1f} times dimmer" +
                         (" (the gain at its highest)" if self.gain_kind != "raw" else "") +
                         ": raise the gain, or give the slice more light.")
        self.exposure.setMaximum(longest)
        if self._thread is not None:
            self._thread.max_exposure_ms = longest if fps > 0 else None
        self._exposure()
        if said or not quiet:
            self.status.setText(said or (f"{fps:g} fps wanted: exposures up to {longest:.3g} ms." if fps > 0 else
                                         "The exposure alone sets the frame rate."))
        self.save_settings()

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
            self.fps.setToolTip(fps_limit(taken, shown, t.exposure_ms or self.exposure.value(), 1,
                                          1.0 / t.show_every_s))
        self._fps_at, self._fps_taken, self._fps_shown = now, t.taken, self._shown

    def _show(self, frame) -> None:
        self._shown += 1
        try:
            self._draw(frame)
        finally:
            if self._thread is not None:
                self._thread.drawing = False

    def _draw(self, frame) -> None:
        """The frame as taken, grey, its 256 levels as they are: only as many pixels as the screen shows (zoomed out,
        every 2nd to 4th), laid over the picture's full size so every coordinate stays the camera's own."""
        self.last = frame
        img = frame.rgb[:, :, 0]
        step = 1
        vb = self.image.getViewBox()
        if vb is not None:
            try:
                px = vb.viewPixelSize()[0]
                step = int(max(1.0, min(4.0, px)))
                if px > 1.0:                     # shown smaller than taken: low-power mode draws every 2nd pixel
                    step = max(step, getattr(self, "_min_step", 1))
            except Exception:                    # the view not laid out yet
                step = 1
        self.image.setImage(img[::step, ::step] if step > 1 else img, autoLevels=False, levels=(0, 255))
        self.image.setRect(QtCore.QRectF(0, 0, img.shape[1], img.shape[0]))
        self._place_bar()

    def _auto(self, on: bool) -> None:
        """Auto exposure on (the exposure follows the light) or off (it stays as typed)."""
        if self._thread is not None:
            self._thread.follow = on
        self.exposure.setEnabled(not on)
        self.save_settings()

    def _followed(self, ms: float, note: str) -> None:
        """Auto exposure moved the exposure: shown, and why it could not go further when it could not."""
        self.exposure.blockSignals(True)
        self.exposure.setValue(ms)
        self.exposure.blockSignals(False)
        if note:
            self.status.setText(note.strip())

    def hold_exposure(self, on: bool) -> None:
        """Auto exposure held still (the patterned light's calibration compares pictures), or left to follow again."""
        if self._thread is not None:
            self._thread.follow_paused = on

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
                                                    "not calibrated (Scale...)"))
        self.dialog.um.blockSignals(True)
        self.dialog.um.setValue(self.calibrated.get(mag, 0.0))
        self.dialog.um.blockSignals(False)
        self.dialog.scale_note.setText(f"{mag:g}x: " + (f"{um:.4g} um a pixel, {how}." if um else
                                       "not known: give the camera's pixel and adapter, or measure a stage "
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

    def open_scale(self) -> None:
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
        info = {"camera_serial": f.serial, "exposure_ms": f.exposure_ms, "gain": f.gain,
                "objective": f"{self.magnification:g}x", "um_per_px": um, "scale": how,
                "slice_plane": self.plane(), "orientation": self.orientation_text(),
                "saved_by": "Clampwise", "taken": time.strftime("%Y-%m-%d %H:%M:%S")}
        path = picture.write_tiff(picture.snapshot_name(folder, f"{self.magnification:g}x"),
                                  np.ascontiguousarray(f.rgb), info, um)
        self.window.say(f"Snapshot saved: {path}")
        self.status.setText(f"Saved {path.name} in {path.parent}.")
        return path

    # ---- kept for the user
    def load_settings(self) -> None:
        if not getattr(self.window, "persist", False):
            return
        s = self.window._settings()
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
                       self.dialog.medial, self.compass)
        for w_ in orientation:
            w_.blockSignals(True)
        on = str(s.value("camera/compass", "false")).lower() == "true"
        self.dialog.compass.setChecked(on)
        self.compass.setChecked(on)
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
        self.auto.setChecked(str(s.value("camera/auto_exposure", "true")).lower() == "true")
        self.frame_rate.setValue(float(s.value("camera/fps", 0.0) or 0.0))

    def save_settings(self) -> None:
        if self._loading or not getattr(self.window, "persist", False):
            return
        s = self.window._settings()
        s.setValue("camera/calibrated", json.dumps({f"{k:g}": v for k, v in self.calibrated.items()}))
        s.setValue("camera/pixel_um", self.dialog.pixel_um.value())
        s.setValue("camera/adapter", self.dialog.adapter.value())
        s.setValue("camera/objectives", ", ".join(f"{m:g}" for m in self.objectives))
        s.setValue("camera/objective_index", self.objective.value())
        s.setValue("camera/scale_bar", self.dialog.bar.isChecked())
        s.setValue("camera/compass", self.dialog.compass.isChecked())
        s.setValue("camera/slice_plane", self.plane())
        s.setValue("camera/compass_names", self.dialog.naming.currentText())
        s.setValue("camera/rostral_deg", self.dialog.rostral.value())     # the plane's turned direction
        s.setValue("camera/medial_right", self.dialog.medial.currentIndex() == 0)
        s.setValue("camera/auto_exposure", self.auto.isChecked())
        s.setValue("camera/fps", self.frame_rate.value())
