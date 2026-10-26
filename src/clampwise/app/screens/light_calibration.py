"""The patterned light's calibration, step by step, in a window of its own: what to put under the objective, a
checkerboard to focus by, how the LED is on, Start and Stop, how far it has gone, and the outcome in a word; or the
field placed by hand, a frame dragged on the picture onto a target lit (the calibration itself is the panel's,
rig.patterns its geometry)."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from .. import theme, widgets
from ..widgets import box, label
from .polygon_panel import LED_MODES, NUDGE, NUDGE_DEG, PolygonPanel


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
        lay.addWidget(label("Or place it by hand", "section"))
        lay.addWidget(label("When no spot is found, or a calibration has moved a little since. The mirrors show a "
                            "target (the field's edge, a ring of spots, an L at its centre: its long arm points "
                            "right on the field, its short arm up), and the same target is drawn in orange where "
                            "Clampwise puts it. Drag the orange frame on the picture until the drawn target sits on "
                            "the lit one: inside it to move, the round handle to turn, the square ones to size. Turn "
                            "or mirror it first if the L is the wrong way round. The LED on steadily.", "muted",
                            wrap=True))
        by_hand = box("h", "s")
        self.place = widgets.button("Place by hand", self._place, "The field's frame on the picture to drag, "
                                    "starting from the calibration in use", quiet=True)
        self.keep = widgets.primary("Keep", lambda: self._end_place(True), "Keep this place for the objective in "
                                    "use (replaces its calibration)")
        self.cancel = widgets.button("Cancel", lambda: self._end_place(False), "Back to the calibration in use",
                                     quiet=True)
        for b in (self.place, self.keep, self.cancel):
            by_hand.addWidget(b)
        by_hand.addStretch(1)
        lay.addLayout(by_hand)
        self.hand_buttons: list = []
        step, arrow = light.nudge_step, QtWidgets.QStyle
        rows = ((("Move", ((arrow.SP_ArrowLeft, "Move left", lambda: light.nudge(dx=-step())),
                           (arrow.SP_ArrowRight, "Move right", lambda: light.nudge(dx=step())),
                           (arrow.SP_ArrowUp, "Move up", lambda: light.nudge(dy=-step())),
                           (arrow.SP_ArrowDown, "Move down", lambda: light.nudge(dy=step())))),
                 ("Size", (("−", "A little smaller", lambda: light.nudge(grow=1 - NUDGE)),
                           ("+", "A little larger", lambda: light.nudge(grow=1 + NUDGE))))),
                (("Turn", ((f"−{NUDGE_DEG:g}°", "Turn a little anticlockwise", lambda: light.nudge(turn=-NUDGE_DEG)),
                           (f"+{NUDGE_DEG:g}°", "Turn a little clockwise", lambda: light.nudge(turn=NUDGE_DEG)))),))
        for groups in rows:
            fine = box("h", "xs")
            for group, buttons in groups:
                fine.addWidget(label(group, "muted"))
                for face, tip, do in buttons:
                    b = widgets.button(face if isinstance(face, str) else "", do,
                                       tip + " (held: again and again)", quiet=True)
                    if not isinstance(face, str):
                        b.setIcon(self.style().standardIcon(face))
                    b.setAutoRepeat(True)
                    fine.addWidget(b)
                    self.hand_buttons.append(b)
                fine.addSpacing(theme.SPACE["m"])
            fine.addStretch(1)
            lay.addLayout(fine)
        coarse = box("h", "s")
        for text, tip, do in (("Turn 90°", "The field turned a quarter, clockwise", lambda: light.nudge(turn=90.0)),
                              ("Mirror ↔", "The field mirrored left for right", lambda: light.flip(True)),
                              ("Mirror ↕", "The field mirrored top for bottom", lambda: light.flip(False)),
                              ("Start over", "The frame back where it began", light.adjust_from_start)):
            b = widgets.button(text, do, tip, quiet=True)
            coarse.addWidget(b)
            self.hand_buttons.append(b)
        coarse.addStretch(1)
        lay.addLayout(coarse)
        lay.addStretch(1)
        widgets.resizable(self, lay, "polygon_calibration", (460, 720))
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
        if self.light._adjust is not None:            # closed while placing: nothing kept without Keep
            self.light.end_adjust(keep=False)
            self.light.panel.window.say("The field placed by hand was not kept (the window closed before Keep).")

    def _place(self) -> None:
        self.board.blockSignals(True)                 # the target is shown instead
        self.board.setChecked(False)
        self.board.blockSignals(False)
        self.light.adjust()
        self.refresh()

    def _end_place(self, keep: bool) -> None:
        self.light.end_adjust(keep)
        self.refresh()

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
        placing = p._adjust is not None
        self.start.setEnabled(not running and not lacks and not placing)
        self.board.setEnabled(not running and not placing and p.polygon.connected)
        self.place.setEnabled(not running and not placing and p._frame_size()[0] > 0)
        self.keep.setEnabled(placing)
        self.cancel.setEnabled(placing)
        for b in self.hand_buttons:
            b.setEnabled(placing)
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
