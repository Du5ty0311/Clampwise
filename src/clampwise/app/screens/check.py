"""Check: look at the recording before trusting a number from it (B0, B2, B10).

Overview first: the trace (every sweep muted under their average, or one
sweep at a time), the stimulus marked where it was found, and two plain
lines above it saying what kind of recording this is and why, and where the
stimulus came from. The details panel lists every channel with its role and
the reason for it, and what the reader decided. A long gap-free recording is
drawn from its min/max envelope, so scrolling an hour stays quick.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .. import theme, widgets
from ..widgets import box, label
from ..longtrace import LONG_SAMPLES, LongTrace
from .base import Screen
from .files import TYPE_WORDS

MAX_OVERLAY = 40          # more sweeps than this: the average and a sample of sweeps
MAX_SHADED = 400          # left-out stretches shaded on one sweep, at most
INVERSE = {"reject": "restore", "restore": "reject", "add": "remove", "remove": "add"}


class ReviewEdit(QtWidgets.QUndoCommand):
    """One event-review decision, kept in the project (Ctrl+Z undoes it)."""

    def __init__(self, screen, path, sweep: int, time_s: float, action: str) -> None:
        super().__init__(f"event review: {action}")
        self.screen, self.path, self.sweep, self.time_s, self.action = screen, path, sweep, time_s, action

    def redo(self) -> None:
        self.screen.window.project.review_event(self.path, self.sweep, self.time_s, self.action)
        self.screen.review_changed()

    def undo(self) -> None:
        self.screen.window.project.review_event(self.path, self.sweep, self.time_s, INVERSE[self.action])
        self.screen.review_changed()


class FileReview(QtWidgets.QUndoCommand):
    """A file marked good or bad on Check, kept in the project (Ctrl+Z undoes it)."""

    def __init__(self, screen, path, verdict: str, reason: str) -> None:
        super().__init__(f"mark {Path(path).name} {verdict}")
        self.screen, self.path, self.after = screen, path, (verdict, reason)
        self.before = screen.window.project.file_review(path)

    def redo(self) -> None:
        self.screen.window.project.set_file_review(self.path, *self.after)
        self.screen.file_reviewed()

    def undo(self) -> None:
        self.screen.window.project.set_file_review(self.path, *self.before)
        self.screen.file_reviewed()


def _natural(text: str) -> list:
    """Numbers in a name compared as numbers (file 2 before file 10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


VERDICT = {"good": ("ok", "good"), "bad": ("bad", "bad"), "": ("muted", "not reviewed")}


class MovePoint(QtWidgets.QUndoCommand):
    """A measured point moved by hand (a step's baseline, peak, steady state or
    rebound), kept in the project; the recording is analyzed again (Ctrl+Z undoes it)."""

    def __init__(self, screen, path, sweep: int, what: str, time_s: float | None, before: float | None) -> None:
        super().__init__(f"move the {what} of sweep {sweep + 1}")
        self.screen, self.path, self.sweep, self.what = screen, path, sweep, what
        self.time_s, self.before = time_s, before

    def redo(self) -> None:
        self.screen.window.project.move_point(self.path, self.sweep, self.what, self.time_s)
        self.screen.apply_review()

    def undo(self) -> None:
        self.screen.window.project.move_point(self.path, self.sweep, self.what, self.before)
        self.screen.apply_review()


class CheckScreen(Screen):
    key = "check"
    title = "Check"
    purpose = "The selected recording as acquired: its recording type, stimulus timing and quality checks."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.next_button = widgets.primary("Go to Analyze", lambda: window.go("analyze"),
                                           "Choose and run the analyses for this recording")
        self.add_action(self.next_button)
        self.stack = QtWidgets.QStackedWidget()
        self.empty = widgets.EmptyState("Select a recording", "Pick a recording in Files to look at it here.",
                                        [("Go to Files", lambda: window.go("files"), True)])
        self.stack.addWidget(self.empty)
        page = QtWidgets.QWidget()
        lay = box("v", "s")
        # the project's files without leaving Check, and the file review (a feature, Settings)
        fb = box("h", "s")
        self.prev_file = widgets.button("◀ File", lambda: self.step_file(-1), "Previous file (J)")
        self.next_file = widgets.button("File ▶", lambda: self.step_file(1), "Next file (K)")
        self.picker = QtWidgets.QComboBox()
        self.picker.setToolTip("The project's files, grouped by cell; ✓ marked good, ✖ marked bad")
        self.picker.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.picker.setMinimumContentsLength(18)
        self.picker.activated.connect(self._picked)
        self.verdict = widgets.Badge("muted", "not reviewed")
        self.mark_good = widgets.button("Good", lambda: self.mark("good"), "Mark this file good (G)")
        self.mark_bad = widgets.button("Bad", lambda: self.mark("bad"), "Mark this file bad (B)")
        self.reason = QtWidgets.QLineEdit()
        self.reason.setPlaceholderText("Reason (optional)")
        self.reason.setToolTip("Kept with the mark; Enter saves it and goes back to the trace")
        self.reason.returnPressed.connect(self._reason_typed)
        self.unreviewed = widgets.button("Next unreviewed", self.next_unreviewed,
                                         "The next file not marked good or bad (N)", quiet=True)
        self.review_widgets = (self.verdict, self.mark_good, self.mark_bad, self.reason, self.unreviewed)
        fb.addWidget(self.prev_file)
        fb.addWidget(self.picker, 2)
        fb.addWidget(self.next_file)
        for w in self.review_widgets:
            fb.addWidget(w, 1 if w is self.reason else 0)
        lay.addLayout(fb)
        self._files: list[Path] = []
        self._marks: dict[Path, tuple[str, str]] = {}
        self.kind = label("", "heading", wrap=True)
        self.why = label("", "muted", wrap=True)
        self.stim = label("", "muted", wrap=True)
        lay.addWidget(self.kind)
        lay.addWidget(self.why)
        lay.addWidget(self.stim)
        bar = box("h", "s")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["All sweeps", "One sweep at a time"])
        self.mode.setToolTip("Every sweep together (with their average when they repeat one protocol), or one at "
                             "a time (arrow keys)")
        self.mode.currentIndexChanged.connect(self._draw)
        self.prev = widgets.button("←", lambda: self._step(-1), "Previous sweep (Left arrow)")
        self.next = widgets.button("→", lambda: self._step(1), "Next sweep (Right arrow)")
        self.sweep_label = label("", "muted")
        self.sweep_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.fit = widgets.button("Zoom to fit", self._fit, "Show the whole sweep (F)", quiet=True)
        self.reset_points = widgets.button("Reset moved points", self._reset_points,
                                           "Put every point moved by hand back where the analysis reads it", quiet=True)
        self.reset_points.setVisible(False)
        for w in (self.mode, self.prev, self.next):
            bar.addWidget(w)
        bar.addWidget(self.sweep_label, 1)      # cut to the room there is (the whole of it in its tooltip)
        bar.addWidget(self.reset_points)
        bar.addWidget(self.fit)
        lay.addLayout(bar)
        # the event review (B4): shown when the recording's events were found
        self.review = QtWidgets.QWidget()
        rv = box("v", "xs")
        self.review_count = label("", "muted")
        self.review_count.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        rv.addWidget(self.review_count)
        rb = box("h", "s")
        self.ev_prev = widgets.button("◀ Event", lambda: self.step_event(-1), "Previous event ([)")
        self.ev_next = widgets.button("Event ▶", lambda: self.step_event(1), "Next event (])")
        self.ev_toggle = widgets.button("Reject", self.toggle_event, "Reject this event, or restore it (R)")
        self.ev_add = widgets.button("Add at cursor", self.add_at_cursor,
                                     "Add an event where the cursor is on the trace (A)")
        self.ev_apply = widgets.button("Update numbers", self.apply_review,
                                       "Analyze this recording again with the review applied")
        self.review_pending = label("", "muted")
        self.review_pending.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        for w in (self.ev_prev, self.ev_next, self.ev_toggle, self.ev_add):
            rb.addWidget(w)
        rb.addWidget(self.review_pending, 1)
        rb.addWidget(self.ev_apply)
        rv.addLayout(rb)
        self.review.setLayout(rv)
        self.review.setVisible(False)
        lay.addWidget(self.review)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().setClipToView(True)
        self.plot.scene().sigMouseMoved.connect(self._mouse)
        lay.addWidget(self.plot, 1)
        page.setLayout(lay)
        self.stack.addWidget(page)
        self.body(self.stack)
        self._recording = None
        self._run = None
        self._sweep = 0
        self._events: list[dict] = []          # the last run's event table, sorted
        self._excluded: list[dict] = []
        self._event_i = -1
        self._cursor_t = float("nan")
        self._path = None
        self._review_on = False
        self._long = None                 # a long recording's envelope trace
        self._pending = None
        dl = box("v", "s", "m")
        dl.addWidget(label("Channels", "heading"))
        self.channels = label("", "", wrap=True)
        dl.addWidget(self.channels)
        dl.addWidget(widgets.hairline())
        dl.addWidget(label("What the reader decided", "heading"))
        self.reader = label("", "muted", wrap=True)
        dl.addWidget(self.reader)
        dl.addWidget(widgets.hairline())
        # the user's word over the detection (B2): kept with the recording, said on every result
        self.override = widgets.Advanced("If the detection is wrong")
        self.ov_type = QtWidgets.QComboBox()
        self.ov_channel = QtWidgets.QComboBox()
        self.ov_times = QtWidgets.QLineEdit()
        self.ov_times.setPlaceholderText("e.g. 320, 425 @ 5 (onsets in ms @ width in ms)")
        self.ov_sweeps = QtWidgets.QLineEdit()
        self.ov_sweeps.setPlaceholderText("e.g. 135-200, 455 (blank: every sweep)")
        self.ov_stretch = QtWidgets.QLineEdit()
        self.ov_stretch.setPlaceholderText("e.g. 30-90 (s; blank: all of it)")
        self.override.add(widgets.field("This recording is", self.ov_type, "replaces the detected type"))
        self.override.add(widgets.field("Stimulus channel", self.ov_channel, "the channel to read the stimulus from"))
        self.override.add(widgets.field("Stimulus times", self.ov_times, "typed times replace whatever the file "
                                        "records; flagged on every result. Current steps for a file with no "
                                        "command: steps -200:50:500 (pA; @ 500-1500 ms, else read from the "
                                        "response)"))
        self.override.add(widgets.field("Sweeps to analyze", self.ov_sweeps, "numbered from 1, as in Clampex"))
        self.override.add(widgets.field("Stretch to analyze (gap-free)", self.ov_stretch, ""))
        self.override.add(widgets.button("Apply and analyze again", self._apply_overrides,
                                         "Kept with this recording in the project"))
        dl.addWidget(self.override)
        dl.addStretch(1)
        body = QtWidgets.QWidget()
        body.setLayout(dl)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(body)
        outer = box("v")
        outer.addWidget(scroll)
        self.details.setLayout(outer)
        theme.style_plot(self.plot, window.palette)
        # keys while Check has the focus (a text box keeps the letters it types)
        for keys, slot in (("Left", lambda: self._step(-1)), ("Up", lambda: self._step(-1)),
                           ("Right", lambda: self._step(1)), ("Down", lambda: self._step(1)),
                           ("[", lambda: self.step_event(-1)), ("]", lambda: self.step_event(1)),
                           ("R", lambda: self._events and self.toggle_event()),
                           ("A", lambda: self._review_on and self.add_at_cursor()),
                           ("J", lambda: self.step_file(-1)), ("K", lambda: self.step_file(1)), ("F", self._fit),
                           ("G", lambda: self.mark("good")), ("B", lambda: self.mark("bad")),
                           ("N", self.next_unreviewed)):
            QtWidgets.QShortcut(QtGui.QKeySequence(keys), self, slot, context=QtCore.Qt.WidgetWithChildrenShortcut)

    def on_shown(self) -> None:
        if self._pending is not None:
            self.on_selection(self._pending)
        self._show_files()
        self.plot.setFocus()

    def on_theme(self, palette) -> None:
        theme.style_plot(self.plot, palette)
        self._draw()

    def _long_ready(self) -> None:
        self._draw()
        self._fit()

    def on_selection(self, path) -> None:
        if not self.is_current() and path is not None:
            self._pending = path              # drawn when Check is shown
            return
        self._pending = None
        if self._long is not None:
            self._long.stop()
            self._long = None
        # the same recording again (analyzed anew after a point was moved) stays on its sweep
        keep = self._sweep if path is not None and path == getattr(self, "_path", None) else 0
        self._recording, self._run, self._sweep = None, None, 0
        if path is None or self.window.project is None:
            self._show_empty(True)
            return
        try:
            self._recording = self.window.recording(path)
        except Exception as exc:
            self._show_empty(True)
            self.window.say(f"Could not read {path.name}: {exc}")
            return
        self._run = self.window.project.last_run(path, self._recording)
        self._path = path
        self._sweep = min(keep, self._recording.n_sweeps - 1)
        self._fill_overrides()
        self._load_events()
        self._show_files()
        self._show_empty(False)
        self._describe()
        self._draw()
        self._fit()

    def _describe(self) -> None:
        r, run = self._recording, self._run
        if run is not None:
            self.kind.setText(TYPE_WORDS.get(run.recording_type, run.recording_type))
            reason = run.type_reason.split(": ", 1)[-1] if ": " in run.type_reason else run.type_reason
            self.why.setText(f"Why: {reason}.")
            self.stim.setText(f"Stimulus: {run.stimulus_summary}.")
        else:
            # not analyzed yet: what it is and where its stimulus is, found in the background
            # (a long recording's channels take a second to read)
            from ...analysis import pipeline, stimulus
            from .. import worker

            self.kind.setText("Not analyzed yet")
            self.why.setText("Finding what kind of recording this is...")
            self.stim.setText("")

            # found by the analysis's own first steps (typed steps, chosen sweeps, artifacts, the stimulus with
            # the rig's records, the type with the mappings), so Check says what the analysis will find: the
            # project's settings are asked here, the files read in the background
            project = self.window.project
            settings = project.pipeline_settings(self._path) if project is not None else pipeline.Settings()
            rig_on = project is not None and project.features().get("rig_records", True)

            def find(r=r, path=self._path, settings=settings, rig_on=rig_on):
                from ...rig import evidence as rig_evidence

                ev = rig_evidence.for_recording(r, path=path) if rig_on else None
                settings.rig = ev if ev is not None and not ev.empty else None
                return pipeline.find(r, settings)

            def show(result, r=r):
                if self._recording is not r or isinstance(result, Exception):
                    return
                st, d = result.stimulus, result.decision
                self.kind.setText(TYPE_WORDS.get(d.recording_type, d.recording_type) + " (not analyzed yet)")
                self.why.setText(f"Why: {d.reason}.")
                self.stim.setText(f"Stimulus: {stimulus.describe(st)}.")
                if len(st) and result.recording is r:          # the times are this recording's own sweeps
                    r.stimulus = st
                    self._draw()

            worker.run(find, show)
        roles = {"signal": "the recorded signal", "command": "the command", "monitor": "a monitor",
                 "trigger": "a trigger", "light": "light", "other": "another channel"}
        rows = []
        for c in r.channels:
            rows.append(f"<b>{c.name}</b> ({c.units or 'no units'}, {'recorded' if c.recorded else 'commanded'}) "
                        f"is {roles.get(c.role.value, c.role.value)}: {c.role_reason}.")
        self.channels.setText("<br><br>".join(rows))
        lines = list(r.notes[-8:])
        if r.start_time is not None:
            lines.insert(0, f"Recorded {r.start_time:%Y-%m-%d %H:%M:%S} (the file's own clock).")
        if r.metadata.notes:
            lines += [f"{n.source}: {n.text}" for n in r.metadata.notes[:5]]
        self.reader.setText("<br>".join(lines) if lines else "Nothing to report.")
        self.window.status.set_recording(
            f"{r.source.name} · {r.sample_rate_hz / 1000:g} kHz · "
            + ("gap-free, " + f"{r.duration_s / 60:.1f} min" if r.is_gap_free else f"{r.n_sweeps} sweeps"))

    def _draw(self) -> None:
        r = self._recording
        if r is None:
            return
        pal = self.window.palette
        self.plot.clear()
        item = self.plot.getPlotItem()
        item.getViewBox().enableAutoRange(axis="y")          # a new sweep is fitted in height
        units = r.signal_units
        item.setLabel("left", f"{r.signal_channel.name} ({units})")
        item.setLabel("bottom", "Time (s)")
        one = self.mode.currentIndex() == 1 or r.n_sweeps == 1
        self.prev.setEnabled(one and r.n_sweeps > 1)
        self.next.setEnabled(one and r.n_sweeps > 1)
        for w in (self.mode, self.prev, self.next):         # nothing to choose in one continuous recording
            w.setVisible(r.n_sweeps > 1)
        shown = 0 if r.is_gap_free else self._sweep
        long = r.n_samples > LONG_SAMPLES and (r.is_gap_free or one)
        t = None if long else r.time_s
        if long:
            # a long sweep: its envelope, built in the background, redrawn on every zoom
            pen = pg.mkPen(pal.average, width=1)
            if self._long is not None and self._long.sweep != shown:
                self._long.stop()
                self._long = None
            if self._long is None:
                self._long = LongTrace(self.plot, r, shown, pen)
                self._long.ready.connect(self._long_ready)
            else:
                self._long.item.setPen(pen)
                self.plot.addItem(self._long.item)
                self._long.item.setClipToView(False)
                self._long._redraw()
            words = "one continuous recording" if r.is_gap_free else f"sweep {shown + 1} of {r.n_sweeps}"
            self.sweep_label.setText(words + ("" if self._long.is_ready else ": drawing..."))
        elif r.is_gap_free:
            y = r.sweep(0)
            self.plot.plot(t, y, pen=pg.mkPen(pal.average, width=1)).setDownsampling(auto=True, method="peak")
            self.sweep_label.setText("one continuous recording")
        elif one:
            y = r.sweep(self._sweep)
            self.plot.plot(t, y, pen=pg.mkPen(pal.average, width=1.5)).setDownsampling(auto=True, method="peak")
            self.sweep_label.setText(f"sweep {self._sweep + 1} of {r.n_sweeps}")
        else:
            data = r.signal
            idx = np.arange(r.n_sweeps)
            if idx.size > MAX_OVERLAY:
                idx = np.unique(np.linspace(0, r.n_sweeps - 1, MAX_OVERLAY).astype(int))
            shown = f"{idx.size} of {r.n_sweeps} sweeps" if idx.size < r.n_sweeps else f"all {r.n_sweeps} sweeps"
            if self._family():
                # sweeps that differ in their command are not repeats: an average of a
                # step family means nothing, so each sweep is drawn and none is averaged
                for k, i in enumerate(idx):
                    colour = theme.series(pal)[k % len(theme.series(pal))]
                    self.plot.plot(t, data[i], pen=pg.mkPen(colour, width=0.9)).setDownsampling(auto=True, method="peak")
                self.sweep_label.setText(f"{shown}; no average: each sweep has its own command")
            else:
                for i in idx:
                    self.plot.plot(t, data[i], pen=pg.mkPen(pal.sweep, width=0.7)).setDownsampling(auto=True, method="peak")
                self.plot.plot(t, data.mean(axis=0), pen=pg.mkPen(pal.average, width=1.8)).setDownsampling(auto=True, method="peak")
                self.sweep_label.setText(f"{shown}, their average in colour")
        # the stimulus, where it was found
        st = r.stimulus
        if len(st):
            times = st.times_s[self._sweep if one and self._sweep < len(st.times_s) else 0]
            for k, ts in enumerate(times[:50]):
                line = pg.InfiniteLine(ts, angle=90, pen=pg.mkPen(pal.text_muted, width=1, style=QtCore.Qt.DashLine))
                if k == 0:
                    line.setToolTip(f"Stimulus ({st.source})")
                self.plot.addItem(line)
        if self._events or self._excluded:
            self._draw_events(one or r.is_gap_free)
        if self._run is not None:
            self._draw_measured(one)

    # each kind of measured point its own colour, the same on every recording
    MARKS = ("baseline", "peak", "steady", "rebound", "spike threshold", "spike peak", "tau fit")
    MOVABLE = ("baseline", "peak", "steady", "rebound")

    def _point_moved(self, item, sweep: int, what: str, before: float | None) -> None:
        if self._path is None:
            return
        self.window.undo.push(MovePoint(self, self._path, sweep, what, float(item.pos().x()), before))

    def _reset_points(self) -> None:
        if self._path is not None and self.window.project.reset_points(self._path):
            self.window.say("Every point moved by hand is back where the analysis reads it.")
            self.apply_review()

    def _draw_measured(self, one: bool) -> None:
        """Every point the analyses measured, on the sweeps shown (NeuroExpress's coloured markers)."""
        r = self._recording
        if one or r.is_gap_free:
            shown = {0 if r.is_gap_free else self._sweep}
        else:
            idx = np.arange(r.n_sweeps)
            if idx.size > MAX_OVERLAY:
                idx = np.unique(np.linspace(0, r.n_sweeps - 1, MAX_OVERLAY).astype(int))
            shown = set(idx.tolist())
        colours = theme.series(self.window.palette)
        points: dict[str, tuple[list, list]] = {}
        movable = one and self._run.recording_type == "cc_steps"
        any_moved = False
        for res in self._run.results.values():
            for d in res.drawables:
                if getattr(d, "domain", "") != "time" or (d.sweep is not None and d.sweep not in shown):
                    continue
                kind = d.label.split(",")[0].strip()
                k = self.MARKS.index(kind) if kind in self.MARKS else len(self.MARKS) + sum(map(ord, kind))
                colour = colours[(k + 1) % len(colours)]
                if d.kind == "window":          # a stretch the analyses name: a test pulse, a stimulus artifact
                    brush = pg.mkColor(colour)
                    brush.setAlpha(40)
                    region = pg.LinearRegionItem((d.start_s, max(d.end_s, d.start_s + 2e-4)), movable=False,
                                                 brush=brush, pen=pg.mkPen(colour, width=1))
                    region.setToolTip(d.label)
                    self.plot.addItem(region)
                    points.setdefault(kind, ([], []))
                    continue
                if d.x is None:
                    continue
                any_moved = any_moved or "moved by hand" in d.label
                if movable and d.kind == "marker" and kind in self.MOVABLE and d.sweep is not None:
                    # a step's measured point, dragged to where it should be read
                    target = pg.TargetItem(pos=(float(d.x[0]), float(d.y[0])), size=11, movable=True,
                                           symbol="star" if "moved by hand" in d.label else "o",
                                           pen=pg.mkPen(colour, width=2), brush=pg.mkBrush(colour))
                    target.setToolTip(f"{d.label}: drag to read it elsewhere")
                    target.sigPositionChangeFinished.connect(
                        lambda item, k=d.sweep, what=kind, moved="moved by hand" in d.label, x0=float(d.x[0]):
                        self._point_moved(item, k, what, x0 if moved else None))
                    self.plot.addItem(target)
                    continue
                if d.kind == "curve":
                    self.plot.plot(d.x, d.y, pen=pg.mkPen(colour, width=2))
                else:
                    xs, ys = points.setdefault(kind, ([], []))
                    xs.extend(np.atleast_1d(d.x))
                    ys.extend(np.atleast_1d(d.y))
        for kind, (xs, ys) in points.items():
            k = self.MARKS.index(kind) if kind in self.MARKS else len(self.MARKS) + sum(map(ord, kind))
            self.plot.addItem(pg.ScatterPlotItem(xs, ys, size=6, pen=None, brush=colours[(k + 1) % len(colours)],
                                                 name=kind))
        self.reset_points.setVisible(any_moved)
        if points or movable:
            self.sweep_label.setText(self.sweep_label.text() + "; marked: " + ", ".join(
                list(points) + ([f"{', '.join(self.MOVABLE)} (drag one to read it elsewhere)"] if movable else [])))
        self.sweep_label.setToolTip(self.sweep_label.text())

    # ---- overrides
    def _fill_overrides(self) -> None:
        from ...analysis import registry
        from .files import TYPE_WORDS

        o = self.window.project.overrides(self._path)
        r = self._recording
        detected = TYPE_WORDS.get(self._run.recording_type, self._run.recording_type) if self._run else "not analyzed yet"
        self.ov_type.blockSignals(True)
        self.ov_type.clear()
        self.ov_type.addItem(f"As detected ({detected})", "")
        for t in registry.TYPES:
            if t != "unknown":
                self.ov_type.addItem(TYPE_WORDS.get(t, t), t)
        self.ov_type.setCurrentIndex(max(0, self.ov_type.findData(o.type_override)))
        self.ov_type.blockSignals(False)
        self.ov_channel.clear()
        self.ov_channel.addItem("As detected", "")
        for c in r.channels:
            if c.role.value in ("trigger", "light", "other"):
                self.ov_channel.addItem(f"{c.name} ({c.units or 'no units'})", c.name)
        self.ov_channel.setCurrentIndex(max(0, self.ov_channel.findData(o.stimulus_channel)))
        self.ov_times.setText(o.stimulus_override)
        self.ov_sweeps.setText(o.sweeps)
        self.ov_stretch.setText(o.stretch)
        n = sum(bool(x) for x in (o.type_override, o.stimulus_channel, o.stimulus_override, o.sweeps, o.stretch))
        self.override.set_changed(n)

    def _apply_overrides(self) -> None:
        from ...analysis import stimulus
        from ...analysis.events import parse_stretch
        from ...core.recording import parse_sweeps

        times, sweeps, stretch = self.ov_times.text().strip(), self.ov_sweeps.text().strip(), self.ov_stretch.text().strip()
        try:
            stimulus.parse_override(times)
            parse_sweeps(sweeps, self._recording.n_sweeps)
            parse_stretch(stretch)
        except ValueError as exc:
            self.window.say(f"Not applied: {exc}")
            return
        self.window.project.set_overrides(self._path, type_override=self.ov_type.currentData() or "",
                                          stimulus_channel=self.ov_channel.currentData() or "",
                                          stimulus_override=times, sweeps=sweeps, stretch=stretch)
        self._fill_overrides()
        self.window.say("Kept with this recording; analyzing it again.")
        self.window.analyze_paths([self._path])

    # ---- the event review
    def _load_events(self) -> None:
        self._events, self._excluded, self._event_i = [], [], -1
        e = self._run.results.get("events") if self._run is not None else None
        if e is not None and not e.error:
            self._events = sorted(e.tables.get("events", []), key=lambda x: (x["sweep"], x["onset_s"]))
            self._excluded = list(e.tables.get("excluded", []))
        self._review_on = e is not None
        self.review.setVisible(self._review_on)
        self._update_review_bar()

    def _edits(self):
        from ...analysis.events import parse_edits

        return parse_edits(self.window.project.event_edits(self._path)) if self._path is not None else ([], [])

    def _status(self, row: dict, rejected) -> str:
        """accepted, rejected, or unmeasured, with the review as it stands now."""
        if any(s == row["sweep"] and abs(t - row["onset_s"]) <= 0.001 for s, t in rejected):
            return "rejected"
        if row["accepted"] or np.isfinite(row.get("amplitude", np.nan)):
            return "accepted"
        return "unmeasured"

    def _draw_events(self, one_sweep: bool) -> None:
        r, pal = self._recording, self.window.palette
        sweep = 0 if r.is_gap_free else self._sweep
        if not one_sweep:
            return
        fs = r.sample_rate_hz
        if self._long is not None and self._long.sweep == sweep:
            if not self._long.is_ready:
                return                    # the markers come with the trace
            values = self._long.values_at
        else:
            y = r.sweep(sweep)

            def values(ts, y=y):
                return y[np.clip(np.round(np.asarray(ts, float) * fs).astype(np.int64), 0, y.size - 1)].astype(float)
        shaded = [x for x in self._excluded if x["sweep"] == sweep][:MAX_SHADED]
        for x in shaded:
            region = pg.LinearRegionItem((x["start_s"], x["end_s"]), movable=False,
                                         brush=pg.mkBrush(pal.surface_alt), pen=pg.mkPen(None))
            region.setZValue(-10)
            region.setToolTip(f"Left out of event detection: {x['why']}")
            self.plot.addItem(region)
        rejected, added = self._edits()
        groups = {"accepted": ([], []), "rejected": ([], []), "added": ([], [])}
        for row in self._events:
            if row["sweep"] != sweep:
                continue
            status = self._status(row, rejected)
            if status == "unmeasured":
                continue
            t = row["peak_s"] if np.isfinite(row.get("peak_s", np.nan)) else row["onset_s"]
            groups[status][0].append(t)
        known = [(row["sweep"], row["onset_s"]) for row in self._events if row["manual"]]
        for s, t in added:
            if s == sweep and not any(ks == s and abs(kt - t) <= 0.001 for ks, kt in known):
                groups["added"][0].append(t)
        for xs, ys in groups.values():          # every marker's value at once
            ys.extend(values(xs) if xs else [])
        colour = theme.series(pal)[1]
        for status, symbol, size, brush, pen in (
                ("accepted", "o", 8, pg.mkBrush(colour), pg.mkPen(None)),
                ("added", "t", 10, pg.mkBrush(colour), pg.mkPen(pal.text, width=1)),
                ("rejected", "x", 13, pg.mkBrush(pal.plot_background), pg.mkPen(pal.text, width=1.5))):
            xs, ys = groups[status]
            if xs:
                self.plot.addItem(pg.ScatterPlotItem(xs, ys, symbol=symbol, size=size, brush=brush, pen=pen,
                                                     name=status))
        self._current_line = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(pal.text, width=1))
        self._current_line.setVisible(False)
        self._current_sweep = sweep
        self.plot.addItem(self._current_line)
        if 0 <= self._event_i < len(self._events) and self._events[self._event_i]["sweep"] == sweep:
            self._current_line.setValue(self._events[self._event_i]["onset_s"])
            self._current_line.setVisible(True)

    def _update_review_bar(self) -> None:
        if not self._review_on:
            return
        rejected, added = self._edits()
        n_acc = sum(1 for row in self._events if self._status(row, rejected) == "accepted")
        n_rej = sum(1 for row in self._events if self._status(row, rejected) == "rejected")
        known = [(row["sweep"], row["onset_s"]) for row in self._events if row["manual"]]
        pending_add = sum(1 for s, t in added if not any(ks == s and abs(kt - t) <= 0.001 for ks, kt in known))
        where = ""
        if 0 <= self._event_i < len(self._events):
            row = self._events[self._event_i]
            amp = row.get("amplitude", np.nan)
            where = (f"   ·   event {self._event_i + 1}: "
                     f"{'sweep ' + str(row['sweep'] + 1) + ', ' if self._recording.n_sweeps > 1 else ''}"
                     f"{row['onset_s']:.4f} s" + (f", {amp:.3g} {self._recording.signal_units}" if np.isfinite(amp) else ""))
            self.ev_toggle.setText("Restore" if self._status(row, rejected) == "rejected" else "Reject")
        self.review_count.setText(f"Events: {n_acc} accepted, {n_rej} rejected of {len(self._events)} detected"
                                  + (f", {pending_add} added" if pending_add else "") + where)
        e = self._run.results.get("events") if self._run is not None else None
        used = e.settings.get("edits", "") if e is not None else ""
        pending = self.window.project.event_edits(self._path) != (used or "") if self._path is not None else False
        self.review_pending.setText("Not applied yet" if pending else "")
        self.review_pending.setToolTip("Rejected and added events are kept in the project and applied every time "
                                       "this recording is analyzed.")
        self.ev_apply.setEnabled(pending)
        self.ev_toggle.setEnabled(0 <= self._event_i < len(self._events))

    def review_changed(self) -> None:
        self._update_review_bar()
        self._draw()

    def step_event(self, d: int) -> None:
        if not self._events:
            return
        self._event_i = (self._event_i + d) % len(self._events) if self._event_i >= 0 else (0 if d > 0 else len(self._events) - 1)
        row = self._events[self._event_i]
        r = self._recording
        if not r.is_gap_free and r.n_sweeps > 1:
            self._sweep = row["sweep"]
            if self.mode.currentIndex() != 1:
                self.mode.blockSignals(True)
                self.mode.setCurrentIndex(1)
                self.mode.blockSignals(False)
        line = getattr(self, "_current_line", None)
        if line is not None and getattr(self, "_current_sweep", None) == row["sweep"] and line.scene() is not None:
            line.setValue(row["onset_s"])     # same sweep: only the line moves
            line.setVisible(True)
        else:
            self._draw()
        vb = self.plot.getPlotItem().getViewBox()
        vb.setAutoVisible(y=True)
        vb.setXRange(row["onset_s"] - 0.03, row["onset_s"] + 0.08, padding=0)
        vb.enableAutoRange(axis="y")
        self._update_review_bar()

    def toggle_event(self) -> None:
        if not (0 <= self._event_i < len(self._events)) or self.window.project is None:
            return
        row = self._events[self._event_i]
        rejected, _ = self._edits()
        if row["manual"]:
            action = "remove"
        else:
            action = "restore" if self._status(row, rejected) == "rejected" else "reject"
        self.window.undo.push(ReviewEdit(self, self._path, row["sweep"], row["onset_s"], action))

    def add_at_cursor(self) -> None:
        if self._recording is None or not np.isfinite(self._cursor_t) or self.window.project is None:
            self.window.say("Point at the trace where the event starts, then press A.")
            return
        sweep = 0 if self._recording.is_gap_free else self._sweep
        self.window.undo.push(ReviewEdit(self, self._path, sweep, float(self._cursor_t), "add"))

    def apply_review(self) -> None:
        if self._path is not None:
            self.window.analyze_paths([self._path])

    def _show_empty(self, empty: bool) -> None:
        """One primary action at a time: the empty state's, or the screen's."""
        self.stack.setCurrentIndex(0 if empty else 1)
        self.next_button.setVisible(not empty)

    def on_project(self) -> None:
        if self._recording is None:
            self._show_empty(True)
        else:
            self._show_files()

    def _family(self) -> bool:
        """Whether the sweeps differ in their command (a step family)."""
        r = self._recording
        cmd = r.command if r is not None else None
        if cmd is None or cmd.shape[0] < 2:
            return False
        return bool(np.any(np.ptp(cmd, axis=0) > 0.5))

    def _fit(self) -> None:
        r = self._recording
        end = (r.duration_s if r.is_gap_free else r.sweep_duration_s) if r is not None else None
        theme.keep_in_frame(self.plot, 0.0, end)

    def _step(self, d: int) -> None:
        r = self._recording
        if r is None or r.n_sweeps < 2:
            return
        if self.mode.currentIndex() != 1:
            self.mode.setCurrentIndex(1)
        self._sweep = (self._sweep + d) % r.n_sweeps
        self._draw()

    # ---- the project's files, one after another, and the file review
    def _reviewing(self) -> bool:
        p = self.window.project
        return p is not None and p.features()["file_review"]

    def _show_files(self) -> None:
        """The picker (files grouped by cell, the current one chosen) and the current file's mark."""
        p = self.window.project
        rows = sorted(p.summaries(), key=lambda r: (_natural(r["animal"]), _natural(r["slice"]),
                                                     _natural(r["cell"]), _natural(r["name"]))) if p else []
        reviewing = self._reviewing()
        self._files = [Path(r["path"]) for r in rows]
        self._marks = {Path(r["path"]): (r["review"], r["review_reason"]) for r in rows}
        self.picker.blockSignals(True)
        self.picker.clear()
        model, group = self.picker.model(), None
        for r in rows:
            g = " · ".join(x for x in (r["animal"], r["slice"], r["cell"]) if x) or "No cell named"
            if g != group:
                group = g
                self.picker.addItem(g)
                model.item(self.picker.count() - 1).setEnabled(False)
            mark = {"good": "✓ ", "bad": "✖ "}.get(r["review"], "") if reviewing else ""
            self.picker.addItem(f"    {mark}{r['name']}", str(r["path"]))
        self.picker.setCurrentIndex(self.picker.findData(str(self._path)) if self._path is not None else -1)
        self.picker.blockSignals(False)
        k = self._files.index(self._path) if self._path in self._files else -1
        self.prev_file.setEnabled(k > 0)
        self.next_file.setEnabled(0 <= k < len(self._files) - 1)
        verdict, reason = self._marks.get(self._path, ("", ""))
        status, word = VERDICT[verdict]
        self.verdict.set(status, word, (f"Marked {word} on Check" + (f": {reason}" if reason else ""))
                         if verdict else "Not marked yet: G good, B bad")
        self.reason.setText(reason)
        for w in self.review_widgets:
            w.setVisible(reviewing)

    def _picked(self, index: int) -> None:
        path = self.picker.itemData(index)
        if path:
            self.window.select(Path(path))
        self.plot.setFocus()

    def step_file(self, d: int) -> None:
        if self._path not in self._files:
            return
        k = self._files.index(self._path) + d
        if not 0 <= k < len(self._files):
            self.window.say("This is the " + ("first" if d < 0 else "last") + " file of the project.")
            return
        self.window.select(self._files[k])

    def mark(self, verdict: str) -> None:
        if self._path is None or not self._reviewing():
            return
        reason = self.reason.text().strip()
        if (verdict, reason) != self._marks.get(self._path, ("", "")):
            self.window.undo.push(FileReview(self, self._path, verdict, reason))
        self.window.say(f"{self._path.name} marked {verdict}." + ("" if self._next_unreviewed() is not None
                                                                   else " Every file is reviewed."))

    def _reason_typed(self) -> None:
        verdict, reason = self._marks.get(self._path, ("", ""))
        if verdict and self.reason.text().strip() != reason:
            self.window.undo.push(FileReview(self, self._path, verdict, self.reason.text().strip()))
        self.plot.setFocus()

    def file_reviewed(self) -> None:
        self._show_files()
        self.window.screens["files"].on_project()

    def _next_unreviewed(self) -> Path | None:
        k = self._files.index(self._path) if self._path in self._files else -1
        order = self._files[k + 1:] + self._files[:k + 1]
        return next((f for f in order if not self._marks.get(f, ("", ""))[0]), None)

    def next_unreviewed(self) -> None:
        if not self._reviewing():
            return
        f = self._next_unreviewed()
        if f is None:
            self.window.say("Every file is reviewed.")
        else:
            self.window.select(f)

    def _mouse(self, pos) -> None:
        vb = self.plot.getPlotItem().getViewBox()
        if self._recording is None or not self.plot.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        self._cursor_t = float(p.x())
        self.window.status.set_cursor(f"t {p.x():.4f} s   {p.y():.2f} {self._recording.signal_units}")
