"""The patterned light's calibration, step by step, in a window of its own: what to put under the objective, a
checkerboard to focus by, how the LED is on, Start and Stop, how far it has gone, and the outcome in a word (the
calibration itself is the panel's, rig.patterns its geometry)."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from .. import widgets
from ..widgets import box, label
from .polygon_panel import LED_MODES, PolygonPanel


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
