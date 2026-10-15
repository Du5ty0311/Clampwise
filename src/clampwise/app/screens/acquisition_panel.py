"""The acquisition in Clampwise, without Clampex when the user chooses (as the camera is used without its own
program), every recording an ABF file Clampex and Clampfit open: Live's acquisition view can record itself while the acquisition program is closed. One of the
user's own protocols is played on the digitizer (rig.recorder), each sweep drawn as it is recorded, into an ABF file
in the folder chosen; Let go gives the view back to the program's window."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import acquire, digidata, membrane_test, recorder, telegraph
from .. import theme, widgets
from ..widgets import box, label

KEEP = 8                                     # sweeps drawn at once, the newest strongest
GAP_FREE_SHOWN_S = 2.0



def _open_digitizer() -> None:
    """The digitizer opened as Clampwise takes it, so the first Play, Record or membrane test starts at once (opening
    it took 3 s each time); a failure is said when a run starts."""
    try:
        digidata.lend()
    except Exception:
        pass

class AcquisitionPanel(QtWidgets.QWidget):
    """The protocol, the data folder, Record / Stop, and each input's trace as it is recorded."""

    released = QtCore.Signal()
    recording = QtCore.Signal(str)           # the folder being recorded into ("" when it ends)

    def __init__(self, window, make=recorder.Recorder, make_test=membrane_test.MembraneTest) -> None:
        super().__init__()
        self.window, self.make, self.make_test = window, make, make_test
        self.rec: recorder.Recorder | None = None
        self.test = make_test()              # the membrane test (rig.membrane_test)
        self._drawn = -1                     # the last sweep drawn
        lay = box("v", "xs")
        self.scope = pg.GraphicsLayoutWidget()
        self.plots: list[pg.PlotItem] = []
        from .membrane_test_panel import MembraneTestView

        self.test_view = MembraneTestView(window.palette)
        self.stack = QtWidgets.QStackedWidget()   # the protocol's scope, or the membrane test
        self.stack.addWidget(self.scope)
        self.stack.addWidget(self.test_view)
        self.note = label("", "muted", wrap=True)   # what the scope shows, or is waiting for (never blank)
        self.note.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(self.note)
        lay.addWidget(self.stack, 1)
        self._plot_inputs: list = []
        self._n_sweeps = 0
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(theme.SPACE["s"])
        form.setVerticalSpacing(theme.SPACE["xs"])
        self.protocol = QtWidgets.QComboBox()
        self.protocol.setToolTip("One of your own protocols, played as the acquisition program plays it")
        self.protocol.currentIndexChanged.connect(self._protocol_changed)
        self.seconds = QtWidgets.QDoubleSpinBox()
        self.seconds.setRange(1.0, 24 * 3600.0)
        self.seconds.setDecimals(0)
        self.seconds.setValue(60.0)
        self.seconds.setSuffix(" s")
        self.seconds.setToolTip("A gap-free protocol's length (Stop ends it sooner)")
        self.folder = label("", "muted", wrap=True)
        form.addWidget(label("Protocol", "muted"), 0, 0)
        form.addWidget(self.protocol, 0, 1)
        pick = box("h", "xs")
        pick.addWidget(widgets.button("Edit...", self.edit_protocol, "Edit a copy of this protocol (its epochs, "
                                      "holding, sweeps), saved under a new name", quiet=True))
        pick.addWidget(widgets.button("Protocols...", self.choose_protocols, "The folder your protocols are in",
                                      quiet=True))
        form.addLayout(pick, 0, 2)
        self._seconds_caption = label("Gap-free", "muted")
        form.addWidget(self._seconds_caption, 1, 0)
        form.addWidget(self.seconds, 1, 1)
        form.addWidget(label("Into", "muted"), 2, 0)
        form.addWidget(self.folder, 2, 1)
        form.addWidget(widgets.button("Folder...", self.choose_folder, "The folder the recordings are written into",
                                      quiet=True), 2, 2)
        form.setColumnStretch(1, 1)
        lay.addLayout(form)
        actions = box("h", "s")                           # the transport: Play, Pause, Record, Stop
        style = self.style()
        self.play_button = widgets.button("Play", self.play, "Run the protocol, drawn and measured live, keeping no "
                                                             "file (as the acquisition program's View)")
        self.play_button.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.pause_button = widgets.button("Pause", self.pause, "Pause after the sweep under way (the outputs held); "
                                                               "Resume goes on with the next sweep")
        self.pause_button.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPause))
        self.record_button = widgets.button("Record", self.record,
                                            "Run the protocol and record it into an ABF file in the folder below")
        self.record_button.setIcon(self._dot(self.window.palette.bad))
        self.stop_button = widgets.button("Stop", self.stop, "End the run (the sweeps complete by then are kept)")
        self.stop_button.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaStop))
        for b in (self.play_button, self.pause_button, self.record_button, self.stop_button):
            actions.addWidget(b)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        actions.addWidget(self.progress, 1)
        lay.addLayout(actions)
        trow = box("h", "s")                 # the membrane test, beside the acquisition
        self.test_button = widgets.button(
            "Membrane test", self.toggle_test, "A test pulse played over and over: Rt (a seal's too), Rs, Rm, Cm "
            "and the holding, live; nothing is saved")
        self.test_button.setCheckable(True)
        trow.addWidget(self.test_button)
        self.test_amp = QtWidgets.QDoubleSpinBox()
        self.test_amp.setRange(-1000.0, 1000.0)
        self.test_amp.setDecimals(1)
        self.test_amp.setToolTip("The pulse, peak to peak, centred on the holding: mV in voltage clamp, pA in current "
                                 "clamp (the amplifier's mode)")
        self.test_hz = QtWidgets.QDoubleSpinBox()
        self.test_hz.setRange(1.0, 500.0)                # 500 samples a period, as Clampex's: up to 250 kHz
        self.test_hz.setDecimals(1)
        self.test_hz.setSuffix(" Hz")
        self.test_avg = QtWidgets.QSpinBox()
        self.test_avg.setRange(1, 100)
        self.test_avg.setToolTip("Pulses averaged into each reading")
        for caption, w in (("Pulse", self.test_amp), ("at", self.test_hz), ("averaging", self.test_avg)):
            trow.addWidget(label(caption, "muted"))
            trow.addWidget(w)
        trow.addStretch(1)
        lay.addLayout(trow)
        self._test_mode = "VC"
        self._test_settings = {"VC": (10.0, 50.0, 5), "IC": (50.0, 5.0, 3)}     # each mode's own pulse (VC: Clampex's Cell stage)
        self._show_test_settings("VC")
        for w in (self.test_amp, self.test_hz, self.test_avg):
            w.valueChanged.connect(self._keep_test_settings)
        self.status = label("", "muted", wrap=True)
        lay.addWidget(self.status)
        self.setLayout(lay)
        self._protocols_dir = acquire.protocols_folder()
        self._gap_free = False
        self._segment = -1
        self._kept: dict = {}
        self._folder = ""
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._test_timer = QtCore.QTimer(self)
        self._test_timer.timeout.connect(self._test_tick)
        self.load_settings()
        self._buttons()

    # ---- taken and let go, as the camera
    def take(self) -> None:
        why = recorder.available()
        self.status.setText(why + "." if why else "Clampwise records here while the acquisition program is closed "
                                                  "(it owns the digitizer while it runs).")
        if not why and self.make is recorder.Recorder and self.make_test is membrane_test.MembraneTest:
            threading.Thread(target=_open_digitizer, name="Clampwise digitizer", daemon=True).start()
        self._preview()

    def release(self) -> None:
        if self.test.running:
            self.test.stop()
            self.test.wait(5)
        if self.busy:
            self.status.setText("Stop the recording first.")
            return
        digidata.let_go()                                # the acquisition program can open the digitizer
        self.released.emit()

    @property
    def busy(self) -> bool:
        return self.rec is not None and self.rec.status.busy

    # ---- choosing
    def fill_protocols(self) -> None:
        keep = self.protocol.currentText()
        self.protocol.blockSignals(True)
        self.protocol.clear()
        found = sorted(Path(self._protocols_dir).glob("*.pro")) if Path(self._protocols_dir).is_dir() else []
        for p in found:
            self.protocol.addItem(p.stem, str(p))
        i = self.protocol.findText(keep)
        self.protocol.setCurrentIndex(max(i, 0))
        self.protocol.blockSignals(False)
        self._protocol_changed()
        if not found:
            self.status.setText(f"No protocols in {self._protocols_dir}: Protocols... chooses their folder.")

    def _protocol_changed(self, *_) -> None:
        path = self.protocol.currentData()
        gap_free = False
        if path:
            try:
                gap_free = acquire.read_protocol(path).mode == acquire.GAP_FREE
            except (OSError, ValueError):
                pass
        self._gap_free = gap_free
        self.seconds.setEnabled(gap_free)
        self.seconds.setVisible(gap_free)                 # asked only for a gap-free protocol
        self._seconds_caption.setVisible(gap_free)
        self.save_settings()
        if self.isVisible():
            self._preview()

    def edit_protocol(self) -> None:
        from .protocol_editor import ProtocolEditor

        path = self.protocol.currentData()
        if not path:
            self.status.setText("Choose a protocol to edit a copy of.")
            return
        try:
            dialog = ProtocolEditor(self, path, self.window.palette)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Not editable: {exc}.")
            return
        if dialog.exec_() and dialog.saved is not None:
            self.fill_protocols()
            self.protocol.setCurrentIndex(max(self.protocol.findText(dialog.saved.stem), 0))
            self.status.setText(f"Saved {dialog.saved.name} beside the original; it is the protocol now.")

    def choose_protocols(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "The folder your protocols are in", self._protocols_dir)
        if d:
            self._protocols_dir = d
            self.fill_protocols()

    def choose_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Record into", self._folder or str(Path.home()))
        if d:
            self.set_folder(d)

    def set_folder(self, folder: str) -> None:
        self._folder = folder
        self.folder.setText(folder or "Choose a folder")
        self.folder.setToolTip(folder)
        self.save_settings()

    # ---- the transport
    @staticmethod
    def _dot(colour: str):
        from pyqtgraph.Qt import QtGui

        pix = QtGui.QPixmap(16, 16)
        pix.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QColor(colour))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(2, 2, 12, 12)
        painter.end()
        return QtGui.QIcon(pix)

    def _buttons(self) -> None:
        state = self.rec.status.state if self.rec is not None else "idle"
        running = state in ("starting", "recording", "playing", "pausing", "paused", "finishing")
        testing = self.test.running
        self.play_button.setEnabled(not running and not testing)
        self.record_button.setEnabled(not running and not testing)
        self.test_button.setEnabled(not running)
        self.pause_button.setEnabled(state in ("recording", "playing", "paused"))
        self.pause_button.setText("Resume" if state == "paused" else "Pause")
        self.pause_button.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay if state == "paused" else QtWidgets.QStyle.SP_MediaPause))
        self.stop_button.setEnabled(running)
        self.protocol.setEnabled(not running)
        self.seconds.setEnabled(not running and self._gap_free)

    def play(self) -> None:
        self._go(save=False)

    def record(self) -> None:
        self._go(save=True)

    def pause(self) -> None:
        if self.rec is None:
            return
        if self.rec.status.state == "paused":
            self.rec.resume()
        else:
            self.rec.pause()
            self.status.setText("Pausing after the sweep under way...")
        self._buttons()

    def stop(self) -> None:
        if self.busy:
            self.rec.stop()
            self.status.setText("Stopping...")

    def toggle_record(self) -> None:                     # Record, or Stop while running
        self.stop() if self.busy else self.record()

    def _go(self, save: bool) -> None:
        if self.busy:
            return
        path = self.protocol.currentData()
        if not path or (save and not self._folder):
            self.status.setText("Choose a protocol" + (" and the folder to record into." if save else "."))
            return
        self.rec = self.make()
        try:
            self.rec.start(path, self._folder or str(Path.home()), self.seconds.value() if self._gap_free else None,
                           save=save)
        except (RuntimeError, TypeError) as exc:
            self.status.setText(f"Not started: {exc}.")
            self.rec = None
            return
        self._drawn, self._segment, self._kept = -1, -1, {}
        self.stack.setCurrentWidget(self.scope)
        self._span = None
        self._setup_plots(self._inputs_of(path))         # the axes at once, before the digitizer has started
        for p in self.plots:
            p.setXRange(0, self._span or GAP_FREE_SHOWN_S, padding=0.01)
        self._say_run(self.rec.status)
        self._buttons()
        self.recording.emit(self._folder if save else str(self.rec.work / "play"))
        self._timer.start(50)

    def _tick(self) -> None:
        st = self.rec.status
        if st.total:
            self.progress.setRange(0, st.total)
            self.progress.setValue(st.frames)
        words = {"starting": "Starting", "recording": "Recording", "playing": "Playing (nothing kept)",
                 "pausing": "Pausing", "paused": "Paused", "finishing": "Finishing the file",
                 "done": "Recorded" if st.saving else "Played", "failed": "Stopped"}.get(st.state, st.state)
        tg = getattr(st, "telegraph", None)
        amp = (f" The amplifier: {tg.mode}, {tg.primary_unit} gain {tg.primary_gain:g}, command "
               f"{tg.command_per_volt:g} {tg.command_unit}/V.") if tg is not None else ""
        self.status.setText(f"{words}: {st.message}." + amp + "".join(f" {n}." for n in st.notes[-3:]))
        self._say_run(st)
        if getattr(st, "segment", 0) != self._segment:   # a resumed run draws its sweeps afresh
            self._segment, self._kept, self._drawn = getattr(st, "segment", 0), {}, -1
        if st.raw is not None and st.inputs and st.state != "paused":
            self._draw(st)
        self._buttons()
        if not st.busy:
            self._timer.stop()
            self.recording.emit("")
            files = getattr(st, "files", None) or ([st.path] if st.path is not None else [])
            if st.state == "done" and st.saving and files:
                self.window.say(f"Recorded {', '.join(p.name for p in files)} into {files[0].parent}.")

    # ---- the membrane test
    def _show_test_settings(self, mode: str) -> None:
        amp, hz, avg = self._test_settings.get(mode, self._test_settings["VC"])
        for w, v in ((self.test_amp, amp), (self.test_hz, hz), (self.test_avg, avg)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.test_amp.setSuffix(" mV" if mode == "VC" else " pA")
        self._test_mode = mode

    def _keep_test_settings(self, *_) -> None:
        self._test_settings[self._test_mode] = (self.test_amp.value(), self.test_hz.value(), self.test_avg.value())

    def toggle_test(self) -> None:
        if self.test.running:
            self.test.stop()
            self.status.setText("Stopping the membrane test...")
            return
        if self.busy:
            self.test_button.setChecked(False)
            self.status.setText("Stop the run first: one program at a time plays on the digitizer.")
            return
        tg = telegraph.read(self.test.serial, self.test.channel, timeout=1.0)
        if tg is not None and tg.mode in ("VC", "IC") and tg.mode != self._test_mode:
            self._show_test_settings(tg.mode)              # the pulse for the amplifier's mode
        try:
            self.test.start(membrane_test.Settings(self.test_amp.value(), self.test_hz.value(),
                                                   average=self.test_avg.value()))
        except RuntimeError as exc:
            self.test_button.setChecked(False)
            self.status.setText(f"Not started: {exc}.")
            return
        self.test_view.clear()
        self.stack.setCurrentWidget(self.test_view)
        self.note.setText("")
        self.test_button.setText("Stop test")
        self._buttons()
        self._test_timer.start(int(membrane_test.READ_EVERY_S * 1000))

    def _test_tick(self) -> None:
        t = self.test
        self.test_view.show_reading(t.latest, t.history)
        self.status.setText({"starting": "Membrane test: starting", "running": f"Membrane test: {t.message}",
                             "failed": f"Membrane test stopped: {t.message}"}.get(t.state, "Membrane test: stopped")
                            + ".")
        if not t.running:
            self._test_timer.stop()
            self.test_button.setChecked(False)
            self.test_button.setText("Membrane test")
            self._buttons()
            self._preview()

    # ---- never a blank scope: the protocol before Play, the axes at once, what it is waiting for
    def _protocol_now(self, path: str) -> acquire.Protocol:
        """The protocol as the amplifier would have it played now (its telegraph, when Commander answers)."""
        pr = acquire.read_protocol(path)
        tg = telegraph.read(self.test.serial, self.test.channel, timeout=0.5)
        if tg is not None and tg.mode in ("VC", "IC"):
            pr, _ = acquire.with_telegraph(pr, tg)
        return pr

    def _inputs_of(self, path: str) -> list:
        try:
            pr = self._protocol_now(path)
            self._n_sweeps = pr.sweeps if pr.mode != acquire.GAP_FREE else 0
            self._span = (pr.per_episode // max(1, len(pr.ai))) * pr.interval_us * 1e-6 if self._n_sweeps else None
            return acquire.inputs(pr)
        except Exception:
            return []

    def _preview(self) -> None:
        """The chosen protocol drawn before it is played: every sweep's command, in the amplifier's present unit."""
        if self.busy or self.test.running:
            return
        self.stack.setCurrentWidget(self.scope)
        path = self.protocol.currentData()
        if not path:
            self._setup_plots([])
            self.note.setText("Choose a protocol: it is drawn here before it plays.")
            return
        name = Path(path).stem
        try:
            pr = self._protocol_now(path)
            if pr.mode == acquire.GAP_FREE:
                self._setup_plots([])
                self.note.setText(f"<b>{name}</b>: gap-free, every input recorded without a break. Play draws "
                                  "them as they come (nothing kept); Record also keeps them.")
                return
            out, _ = acquire.waveforms(pr)
        except Exception as exc:
            self._setup_plots([])
            self.note.setText(f"<b>{name}</b>: not drawn ({exc}). Play still plays it.")
            return
        d = pr.active_dac if 0 <= pr.active_dac < out.shape[1] else 0
        unit = pr.dac_units[d] if d < len(pr.dac_units) else ""
        self._setup_plots([("Command", unit, 1.0)])
        fs = 1e6 / pr.interval_us
        t = np.arange(out.shape[2]) / fs
        colour = theme.series(self.window.palette)[1]
        for k in range(out.shape[0]):
            self.plots[0].plot(t, out[k, d], pen=pg.mkPen(colour, width=1))
        self.plots[0].setXRange(0, t[-1] if t.size else 1.0, padding=0.01)
        self.note.setText(f"<b>{name}</b>, as it will play: {pr.sweeps} sweep(s) of {t.size / fs:.2f} s, one every "
                          f"{pr.start_to_start_s:g} s. <b>Play</b> draws the cell's response as it is recorded "
                          "(nothing kept); <b>Record</b> also keeps it.")

    def _say_run(self, st) -> None:
        """Above the scope, what it is showing or waiting for."""
        verb = "Recording" if st.saving else "Playing (nothing kept)"
        if st.state == "starting":
            text = ("Starting: reading the amplifier's telegraph and opening the digitizer (about 3 s the first "
                    "time after Use here, then under a second)...")
        elif st.state in ("recording", "playing") and st.episodic and st.step_frames and st.rate_hz:
            k = st.sweeps_done + st.frames // st.step_frames  # the earlier segments' sweeps, then this one's
            into = st.frames % st.step_frames
            of = f" of {self._n_sweeps}" if self._n_sweeps else ""
            if into < st.sweep_frames:
                text = f"{verb}: sweep {k + 1}{of}."
            else:
                text = (f"{verb}: sweep {k + 1}{of} done; the next starts in "
                        f"{(st.step_frames - into) / st.rate_hz:.1f} s.")
        elif st.state in ("recording", "playing") and st.rate_hz:
            text = f"{verb}: {st.frames / st.rate_hz:.0f} of {st.total / st.rate_hz:.0f} s."
        elif st.state == "pausing":
            text = "Pausing after the sweep under way..."
        elif st.state == "paused":
            text = f"Paused after sweep {st.sweeps_done}: Resume goes on in the same file."
        elif st.state == "finishing":
            text = "Finishing the file..."
        elif st.state == "done":
            text = ("Recorded: " + ", ".join(p.name for p in st.files) + "." if st.saving and st.files
                    else "Played: nothing kept.")
        elif st.state == "failed":
            text = f'<span style="color:{self.window.palette.bad}">Stopped: {st.message}.</span>'
        else:
            text = ""
        self.note.setText(text)

    def _setup_plots(self, inputs) -> None:
        self.scope.clear()
        self.plots = []
        self._plot_inputs = [(name, unit) for name, unit, _ in inputs]
        pal = self.window.palette
        self.scope.setBackground(pal.plot_background)
        for k, (name, unit, _) in enumerate(inputs):
            p = self.scope.addPlot(row=k, col=0)
            p.getAxis("left").setLabel(f"{name} ({unit})")
            # each pixel column drawn from its samples' minimum and maximum, so no spike or transient is lost; a
            # 2 s gap-free view at 20 kHz drawn whole took 0.93 s a redraw (the rig PC), the scope a second behind
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if k:
                p.setXLink(self.plots[0])
            if k == len(inputs) - 1:
                p.setLabel("bottom", "Time (s)")
            theme.style_plot(p, pal)
            self.plots.append(p)

    def _draw(self, st) -> None:
        """Live, as it is recorded (not drawn after the sweep ends): the sweep
        under way grows across the plot every tick, the sweeps before it fainter behind; gap-free scrolls."""
        n = len(st.inputs)
        if self._plot_inputs != [(name, unit) for name, unit, _ in st.inputs]:
            self._setup_plots(st.inputs)
            self._kept = {}
        try:
            size = st.raw.stat().st_size // (2 * n)
        except OSError:
            return
        if st.episodic:
            now = size // st.step_frames                   # the sweep under way (or the last, between sweeps)
            into = size - now * st.step_frames
            if into == 0 and now:
                now, into = now - 1, st.sweep_frames
            into = min(into, st.sweep_frames)
            for k in range(max(0, now - KEEP + 1), now):   # finished sweeps are read once
                if k not in self._kept:
                    self._kept[k] = self._read(st, k * st.step_frames, st.sweep_frames)
            self._kept = {k: v for k, v in self._kept.items() if k > now - KEEP}
            sweeps = [self._kept[k] for k in sorted(self._kept)] + [self._read(st, now * st.step_frames, into)]
            self._drawn = now if into == st.sweep_frames else now - 1
            span = st.sweep_frames / st.rate_hz
        else:
            count = min(size, int(GAP_FREE_SHOWN_S * st.rate_hz))
            sweeps = [self._read(st, size - count, count)] if count else []
            span = None
        series = theme.series(self.window.palette)
        for c, p in enumerate(self.plots):
            curves = p.listDataItems()
            while len(curves) < len(sweeps):
                curves.append(p.plot([], []))
            for j, curve in enumerate(curves):
                y = sweeps[j] if j < len(sweeps) else None
                if y is None or not len(y):
                    curve.setData([], [])
                    continue
                newest = j == len(sweeps) - 1
                colour = pg.mkColor(series[c % len(series)])
                colour.setAlphaF(1.0 if newest else 0.15 + 0.5 * (j + 1) / len(sweeps))     # older: fainter
                curve.setPen(pg.mkPen(colour, width=1))           # a wider line is drawn many times slower
                curve.setZValue(j)
                curve.setData(np.arange(len(y)) / st.rate_hz, y[:, c])
            if span is not None:
                p.setXRange(0, span, padding=0.01)          # the whole sweep's width, filled as it comes

    @staticmethod
    def _read(st, at: int, count: int) -> np.ndarray | None:
        n = len(st.inputs)
        try:
            with open(st.raw, "rb") as fh:
                fh.seek(2 * n * at)
                raw = np.frombuffer(fh.read(2 * n * count), "<i2")
        except OSError:
            return None
        frames = raw[: raw.size // n * n].reshape(-1, n).astype(float) * (10.0 / 32768)
        return frames / np.array([v or 1.0 for _, _, v in st.inputs])

    # ---- kept for the user
    def load_settings(self) -> None:
        if getattr(self.window, "persist", False):
            s = self.window._settings()
            self._protocols_dir = str(s.value("record/protocols", "") or acquire.protocols_folder())
            self._folder = str(s.value("record/folder", "") or "")
            self.seconds.setValue(float(s.value("record/seconds", 60.0) or 60.0))
            keep = str(s.value("record/protocol", "") or "")
        else:
            keep = ""
        self._loading = True
        self.fill_protocols()
        if keep:
            self.protocol.setCurrentIndex(max(self.protocol.findText(keep), 0))
        self.set_folder(self._folder)
        self._loading = False

    def save_settings(self) -> None:
        if getattr(self, "_loading", True) or not getattr(self.window, "persist", False):
            return
        s = self.window._settings()
        s.setValue("record/protocols", self._protocols_dir)
        s.setValue("record/folder", self._folder)
        s.setValue("record/protocol", self.protocol.currentText())
        s.setValue("record/seconds", self.seconds.value())

    def on_theme(self, palette) -> None:
        self.test_view.on_theme(palette)
        self.scope.setBackground(palette.plot_background)
        for p in self.plots:
            theme.style_plot(p, palette)
