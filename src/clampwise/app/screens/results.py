"""Results: every number from the selected recording, each with how it was
measured (B3, B14, Part A: "every number must say how it was measured").

Overview first (B0 item 7): the sentence that says what the recording's number
is, and the headline values of each analysis, each with its definition and
source one "?" away. Below, opened when wanted: every pulse, every sweep,
every number with its definition, the flags and notes, and how it was
measured (the methods applied, what was chosen and why, the settings that
differ from their defaults, and the provenance).
"""

from __future__ import annotations

import math

from pyqtgraph.Qt import QtCore, QtWidgets

from ...analysis import registry
from .. import theme, widgets
from ..widgets import box, label
from .base import Screen

# Column headings for the tables analyses return, in the lab's words.
HEADINGS = {
    "pulse": "Pulse", "time_s": "Time (s)", "amplitude": "Amplitude", "amplitude_local": "Rise from trough",
    "pn_p1": "Pn / P1", "latency_median_ms": "Latency, median (ms)", "jitter_ms": "Jitter (ms)",
    "latency_average_ms": "Latency on average (ms)", "time_to_peak_ms": "Time to peak (ms)",
    "rise_10_90_ms": "Rise 10-90% (ms)", "rise_20_80_ms": "Rise 20-80% (ms)", "decay_tau_ms": "Decay tau (ms)",
    "decay_r2": "Decay r²", "decay_tau_from_peak_ms": "Decay tau from peak (ms)", "half_width_ms": "Half-width (ms)",
    "integral": "Charge or area", "snr": "Signal / noise", "detected": "Above noise", "success_rate": "Success rate",
    "potency": "Potency", "cv": "CV", "inv_cv2": "1/CV²", "spike_probability": "Spike probability",
    "spike_latency_ms": "Spike latency (ms)", "spike_jitter_ms": "Spike jitter (ms)", "n_sweeps": "Sweeps",
    "flags": "Flags", "sweep": "Sweep", "failure": "Failure", "latency_ms": "Latency (ms)", "onset_s": "Onset (s)",
    "in_psp_measures": "In PSP measures", "spiked": "Fired", "sweep_level": "Baseline level",
    "sweep_noise_sd": "Baseline noise (SD)", "baseline_s": "Baseline length (s)",
    "event": "Event", "peak_s": "Peak (s)", "charge": "Charge or area", "iei_ms": "Interval (ms)",
    "score": "Detection score", "accepted": "Accepted", "manual": "Added by hand", "decay_poor_fit": "Poor decay fit",
    "end_s": "Measured until (s)", "bin_start_s": "From (s)", "analyzed_s": "Analyzed (s)", "n_events": "Events",
    "frequency_hz": "Frequency (Hz)", "amplitude_mean": "Amplitude (mean)", "charge_rate": "Charge per second",
    "start_s": "From (s)", "why": "Why",
    "trial": "Trial", "step_file": "Step file", "step_start_s": "Step at (s, cell's clock)",
    "step_duration_s": "Step (s)", "step_level_mv": "Step to (mV)", "settle_s": "Settling (s)",
    "first_data_after_step_s": "First data after the step (s)", "pre_analyzed_s": "Before: recorded (s)",
    "pre_events": "Before: events", "post_analyzed_s": "After: recorded (s)", "post_events": "After: events",
    "suppression_pct": "Suppression (%)", "charge_suppression_pct": "Suppression, charge (%)",
    "pre_amplitude_median": "Before: amplitude (median)", "post_amplitude_median": "After: amplitude (median)",
    "recovery_tau_s": "Recovery tau (s)", "recovery_fit_r2": "Recovery fit r²", "notes": "Notes",
    "trials": "Trials", "p": "p", "significant": "Significant", "pre_rate_hz": "Rate before (Hz)",
    "min_rate_hz": "Minimum rate (Hz)", "suppression_at_min_pct": "Suppression at the minimum (%)",
    "suppression_32_20_pct": "Suppression, 32 s / 20 s (%)", "t_dsi_s": "t DSI (s)", "onset_tau_s": "Onset tau (s)",
    "onset_fit_r2": "Onset fit r²", "recovery_S": "Recovery S", "joint_t_dsi_s": "Joint fit: t DSI (s)",
    "joint_onset_tau_s": "Joint fit: onset tau (s)", "joint_S": "Joint fit: S",
    "joint_recovery_tau_s": "Joint fit: recovery tau (s)", "joint_fit_r2": "Joint fit r²",
    "from_step_start_s": "From the step's start (s)", "from_step_end_s": "From the step's end (s)",
    "bin_width_s": "Bin (s)", "recorded_s": "Recorded (s)", "rate_hz": "Rate (Hz)", "pct_of_control": "% of control",
    "pct_between_control_and_min": "% between control and minimum", "bins": "Bins", "episode": "Episode",
    "baseline": "Baseline", "baseline_cv": "Baseline CV", "time_to_recovery_s": "Time to recovery (s)",
    "fit_S": "Recovery S", "fit_r2": "Recovery fit r²", "timed": "Recovery timed", "group": "Step (s or mV)",
    "level_mv": "Level (mV)",
    "current_pa": "Current (pA)", "v_baseline": "Baseline (mV)", "v_steady": "Steady state (mV)",
    "spikes": "Spikes", "full_spikes": "Full-sized", "attenuated": "Attenuated", "isi_mean_ms": "ISI, mean (ms)",
    "isi_cv": "ISI CV", "adaptation_index": "Adaptation index", "isi_ratio": "ISI last / first", "bursts": "Bursts",
    "doublet": "Initial doublet", "spike": "Spike", "time_ms": "Peak at (ms)", "threshold_mv": "Threshold (mV)",
    "threshold_20_mv": "Threshold at 20 mV/ms (mV)", "peak_mv": "Peak (mV)", "amplitude_mv": "Amplitude (mV)",
    "half_width_threshold_ms": "Half-width from threshold (ms)", "fast_trough_mv": "Fast trough (mV)",
    "upstroke": "Max rise (mV/ms)", "downstroke": "Max fall (mV/ms)", "ahp_mv": "AHP (mV)", "full": "Full-sized",
    "clipped": "Cut by the step's end",
    "statement": "Statement", "warning": "A warning", "v_peak": "Peak (mV)", "v_steady_nex": "End of step (mV)",
    "r_peak": "R at the peak (MOhm)", "r_steady": "R at the end (MOhm)", "sag_mv": "Sag (mV)", "sag_pct": "Sag (%)",
    "rebound_mv": "Rebound (mV)", "rebound_pct": "Rebound (%)", "tau_on_ms": "Tau at onset (ms)",
    "tau_off_ms": "Tau at offset (ms)", "capacitance_pf": "Capacitance (pF)", "threshold_5_mv": "Threshold at 5 mV/ms",
    "amplitude_5_mv": "Amplitude from it (mV)", "half_width_5_ms": "Half-width from it (ms)",
    "fall_time_ms": "Fall time (ms)", "ahp_5_mv": "AHP below threshold (mV)", "trough_mv": "Fast trough (mV)",
    "upstroke_phase": "Rise phase", "downstroke_phase": "Fall phase", "kink_max": "Kink (mV/ms2)",
    "time_to_kink_ms": "To the kink (ms)", "kink_mv": "Kink at (mV)", "initial_phase_slope": "Onset rapidness (1/ms)",
    "kink_phase_slope": "Phase slope at the kink (1/ms)", "isi_lv": "ISI LV", "isi_cv2": "ISI CV2",
    "why_not": "Fails acceptance criteria", "baseline_level": "Baseline before", "local_frequency_hz":
    "Local frequency (Hz)", "area_per_amplitude_ms": "Area / amplitude (ms)", "rise_tau_ms": "Rise tau, whole fit (ms)",
    "decay_tau_whole_ms": "Decay tau, whole fit (ms)", "from": "From", "to": "To", "count": "Count",
    "cumulative": "Cumulative", "kernel": "Kernel estimate", "region": "Region", "from_s": "From (s)", "to_s": "To (s)",
    "charge_mean": "Charge (mean)", "amplitude_median": "Amplitude (median)", "iei_cv": "Interval CV",
    "burst": "Burst", "length_s": "Length (s)", "intraburst_hz": "Intraburst (Hz)", "time_s_sdf": "Time (s)",
    "frequency_hz": "Frequency (Hz)", "amplitude_hz": "Amplitude (Hz)", "isi_s": "Interval (s)",
    "next_isi_s": "Next interval (s)", "deviation_s": "From the mean (s)", "lag_s": "Lag (s)",
    "chance_hz": "At chance (Hz)", "unit": "Unit", "n_spikes": "Spikes", "isi_mean_s": "ISI mean (s)",
    "isi_median_s": "ISI median (s)", "isi_sd_s": "ISI SD (s)", "n_bursts": "Bursts",
}
# sweeps are kept from 0 and shown from 1, as Clampex numbers them (and as the sweep picker reads them)
ONE_BASED = {"sweep"}
# The analysis that stands for the recording first, then its quality, the baseline last.
ORDER = {"quality": 1, "membrane_test": 1, "rs_error": 1, "bridge": 1, "baseline": 2}
# yes / no outputs in the lab's words
BOOL_WORDS = {"passed": ("passed", "flagged")}


def ordered(results: dict) -> list:
    return sorted(results.items(), key=lambda kv: ORDER.get(kv[0], 0))


def headline_sentence(run) -> str:
    """What the recording's numbers are, in a sentence."""
    from .files import TYPE_WORDS

    ev = run.results.get("evoked")
    # an evoked headline unless nothing answered the stimulus (a trigger with nothing
    # connected, as a Baseline_DSI protocol fires): then the events lead
    if ev is not None and ev.values.get("headline_is") and not (
            "events" in run.results and any(f.code == "no response" for f in ev.flags)):
        return ev.values["headline_is"]
    kind = TYPE_WORDS.get(run.recording_type, run.recording_type.replace("_", " "))
    ip = run.results.get("intrinsic")
    if ip is not None and ip.tables.get("steps"):
        cur = [r["current_pa"] for r in ip.tables["steps"]]
        rheo = ip.values.get("rheobase")
        fires = f", rheobase {rheo:g} pA" if isinstance(rheo, float) and math.isfinite(rheo) else ", no full-sized spike"
        return f"{kind}: {len(cur)} steps from {min(cur):g} to {max(cur):g} pA{fires}"
    tr = run.results.get("spike_train")
    if tr is not None and (run.recording_type == "spike_times" or "events" not in run.results):
        v = tr.values
        rate = v.get("rate_hz")
        rate_s = f", {rate:.2f} Hz" if isinstance(rate, float) and math.isfinite(rate) else ""
        return f"{kind}: {v.get('n_spikes', 0)} spikes{rate_s}, {v.get('n_bursts', 0)} bursts"
    e = run.results.get("events")
    if e is not None and e.values.get("n_events") is not None:
        v = e.values
        f = v.get("frequency")
        rate = f", {f:.2f} per second" if isinstance(f, float) and math.isfinite(f) else ""
        return f"{kind}: {v['n_events']} events in {v.get('analyzed_s', 0):.0f} s analyzed{rate}"
    return f"{kind}: {run.stimulus_summary}"


# tables an analysis keeps for another level, not for reading
HIDDEN_TABLES = {("events", "analyzed")}
TABLE_NAMES = {"pulses": "Every pulse", "sweeps": "Every sweep", "spikes": "Action potentials", "events": "Every event",
               "timecourse": "Events over time", "excluded": "Left out", "steps": "Every step", "fi": "f-I curve",
               "evaluation": "Evaluation", "histogram_intervals": "Intervals (histogram, s)",
               "histogram_amplitudes": "Amplitudes (histogram)", "histogram_charge": "Charge (histogram)",
               "regions": "Regions", "bursts": "Bursts", "spike_density": "Spike density over time",
               "spike_density_spectrum": "Spike density spectrum", "isi_histogram": "Intervals (histogram, s)",
               "return_map": "Return map", "autocorrelogram": "Autocorrelogram", "units": "Units"}


def units_for(units: str, result, recording) -> str:
    """An output's declared units in this recording's terms."""
    if units == "signal units":
        return recording.signal_units
    if units == "pA or mV":
        return str(result.values.get("units", recording.signal_units))
    if units == "pC or mV*ms":
        return str(result.values.get("integral_units", ""))
    return "" if units in ("ratio", "count", "p", "fraction") else units


def fmt(v, units: str = "") -> str:
    """A value as the screen shows it: three significant figures, NaN as a dash."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v}{' ' + units if units else ''}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "–"
        a = abs(v)
        text = "0" if a == 0 else f"{v:.0f}" if a >= 100 else f"{v:.3g}" if a >= 0.01 else f"{v:.2e}"
        return f"{text}{' ' + units if units else ''}"
    return str(v)


class RowsModel(QtCore.QAbstractTableModel):
    """An analysis's table rows (dicts) for a view, formatted as the screen shows numbers."""

    def __init__(self, rows: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.rows = rows
        self.cols = list(dict.fromkeys(k for r in rows[:200] for k in r))

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:          # noqa: N802 (Qt's name)
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:       # noqa: N802
        return 0 if parent.isValid() else len(self.cols)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):      # noqa: N802
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            c = self.cols[section]
            return HEADINGS.get(c, c.replace("_", " "))
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        v = self.rows[index.row()].get(self.cols[index.column()])
        numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
        if numeric and self.cols[index.column()] in ONE_BASED:
            v = int(v) + 1
        if role == QtCore.Qt.DisplayRole:
            if isinstance(v, (list, tuple)):
                return "; ".join(map(str, v))
            return fmt(v)
        if role == QtCore.Qt.TextAlignmentRole and numeric:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if role == QtCore.Qt.FontRole and numeric:
            return theme.numeric_font()
        if role == QtCore.Qt.ToolTipRole and self.cols[index.column()] == "flags" and v:
            return str(v)
        return None


class ResultsScreen(Screen):
    key = "results"
    title = "Results"
    purpose = "All measurements from the selected recording, each with its definition and method."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.next_button = widgets.primary("Go to Export", lambda: window.go("export"),
                                           "Write these results to Excel or CSV with their definitions")
        self.add_action(self.next_button)
        self.stack = QtWidgets.QStackedWidget()
        self.empty_select = widgets.EmptyState("Select a recording", "Pick a recording in Files; its results show here.",
                                               [("Go to Files", lambda: window.go("files"), True)])
        self.empty_run = widgets.EmptyState("Not analyzed yet", "This recording has no results yet. Analyze it to see "
                                            "every number with how it was measured.",
                                            [("Analyze", lambda: window.analyze_all(), True)])
        self.stack.addWidget(self.empty_select)
        self.stack.addWidget(self.empty_run)
        page = QtWidgets.QWidget()
        lay = box("v", "m")
        self.headline = label("", "heading", wrap=True)
        lay.addWidget(self.headline)
        self.flag_row = box("h", "s")
        lay.addLayout(self.flag_row)
        self.cards = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QGridLayout()
        self.cards_layout.setHorizontalSpacing(theme.SPACE["l"])
        self.cards_layout.setVerticalSpacing(theme.SPACE["s"])
        self.cards.setLayout(self.cards_layout)
        lay.addWidget(self.cards)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumHeight(360)
        self.tabs.currentChanged.connect(self._build_tab)
        self._builders: dict[int, object] = {}       # tab index -> what makes its contents, on first opening
        lay.addWidget(self.tabs, 1)
        page.setLayout(lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)
        self.body(self.stack)
        dl = box("v", "s", "m")
        dl.addWidget(label("What this number is", "heading"))
        self.definition = label("Select a number to read how it was measured.", "muted", wrap=True)
        dl.addWidget(self.definition)
        dl.addStretch(1)
        self.details.setLayout(dl)
        self._run = None
        self._cell = None
        self._pending = None

    def on_shown(self) -> None:
        if self._pending is not None:
            self.on_selection(self._pending)

    def _show(self, which: int) -> None:
        self.stack.setCurrentIndex(which)
        self.next_button.setVisible(which == 2)

    def on_project(self) -> None:
        if self.window.selected is None:
            self._show(0)

    def on_selection(self, path) -> None:
        if path is not None and not self.is_current():
            self._pending = path              # read when Results is shown
            return
        self._pending = None
        if path is None or self.window.project is None:
            self._show(0)
            return
        try:
            self._run = self.window.project.last_run(path, self.window.recording(path))
            self._cell = self.window.project.cell_result(path)
        except Exception as exc:
            self.window.say(f"Could not read the results of {path.name}: {exc}")
            self._run, self._cell = None, None
        if self._run is None or not self._run.results:
            self._show(1)
            return
        self._fill()
        self._show(2)

    # ---- the overview
    def _fill(self) -> None:
        run = self._run
        rec = run.recording
        outputs = registry.definitions_by_analysis()
        self.headline.setText(headline_sentence(run))
        while self.flag_row.count():
            item = self.flag_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        flags = run.flags + (self._cell.flags if self._cell is not None else [])
        if flags:
            codes = sorted({f.code for f in flags})
            for c in codes[:6]:
                detail = "; ".join(f.describe() for f in flags if f.code == c)
                self.flag_row.addWidget(widgets.Badge("warn", c, detail))
        else:
            self.flag_row.addWidget(widgets.Badge("ok", "no flags", "No analysis raised a flag on this recording."))
        self.flag_row.addStretch(1)
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        row = 0
        for key, result in self._ordered():
            decl = outputs.get(key, {})
            heads = [o for o in decl.values() if o.headline and o.level == "recording" and o.key in result.values
                     and o.key != "headline_is"]
            if not heads:
                continue
            title = registry.load()[key].label if key in registry.load() else key
            self.cards_layout.addWidget(label(title, "muted"), row, 0, 1, 3)
            row += 1
            for i, o in enumerate(heads):
                r, c = row + i // 3, (i % 3)
                self.cards_layout.addWidget(self._card(o, result, rec), r, c)
            row += (len(heads) + 2) // 3
        cell = self._cell
        for key, result in (cell.results.items() if cell is not None else []):
            decl = outputs.get(key, {})
            heads = [o for o in decl.values() if o.headline and o.key in result.values]
            if not heads or result.error:
                continue
            title = registry.load()[key].label if key in registry.load() else key
            self.cards_layout.addWidget(label(f"{title}: this cell's {len(cell.files)} files together", "muted"),
                                        row, 0, 1, 3)
            row += 1
            for i, o in enumerate(heads):
                self.cards_layout.addWidget(self._card(o, result, rec), row + i // 3, i % 3)
            row += (len(heads) + 2) // 3
        self._tables()

    def _card(self, o, result, rec) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = box("v", "xs")
        top = box("h", "xs")
        top.addWidget(label(o.title, "muted", wrap=True), 1)
        top.addWidget(widgets.HelpButton(o.title, f"{o.definition}<br><br><i>Reference:</i> {o.source or 'Clampwise'}"
                                         f"<br><i>Checked against:</i> {o.verified_against}", result.key))
        lay.addLayout(top)
        v = result.values.get(o.key)
        if o.key in BOOL_WORDS and (isinstance(v, bool) or v is None):
            shown = "not checked" if v is None else BOOL_WORDS[o.key][0 if v else 1]
        elif o.units == "fraction" and isinstance(v, float) and math.isfinite(v):
            shown = f"{v * 100:.0f}%"
        else:
            shown = fmt(v, units_for(o.units, result, rec))
        value = label(shown, "heading")
        value.setFont(theme.numeric_font())
        lay.addWidget(value)
        w.setLayout(lay)
        return w

    # ---- the details on demand
    def _lazy(self, make, title: str) -> None:
        """A tab whose contents are made the first time it is opened."""
        holder = QtWidgets.QWidget()
        holder.setLayout(box("v"))
        i = self.tabs.addTab(holder, title)
        self._builders[i] = make

    def _build_tab(self, i: int) -> None:
        make = self._builders.pop(i, None)
        if make is not None:
            self.tabs.widget(i).layout().addWidget(make())

    def _tables(self) -> None:
        self.tabs.blockSignals(True)
        self.tabs.clear()
        self._builders = {}
        run = self._run
        rec = run.recording
        # the analysis that stands for the recording first; the baseline's sweep table last
        order = [(k, r, "") for k, r in self._ordered()]
        if self._cell is not None:
            order += [(k, r, "cell") for k, r in self._cell.results.items()]
        for key, result, level in order:
            for name, rows in result.tables.items():
                if not rows or (key, name) in HIDDEN_TABLES:
                    continue
                title = TABLE_NAMES.get(name, name.replace("_", " ").capitalize())
                if key not in ("evoked", "events"):
                    title = f"{title} ({registry.load()[key].label.lower() if key in registry.load() else key})"
                if level:
                    title += ", this cell"
                self._lazy(lambda rows=rows, result=result: self._table(rows, result, rec), title)
        self._lazy(self._every_number, "Every number")
        self._lazy(self._flags, "Flags and notes")
        self._lazy(self._how, "How it was measured")
        self.tabs.blockSignals(False)
        self.tabs.setCurrentIndex(0)
        self._build_tab(0)

    def _table(self, rows: list[dict], result, rec) -> QtWidgets.QTableView:
        """A table of any length: a model the view asks for the rows it shows."""
        view = QtWidgets.QTableView()
        model = RowsModel(rows, view)
        view.setModel(model)
        view.verticalHeader().setVisible(False)
        view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        view.setAlternatingRowColors(True)
        view.setShowGrid(False)
        view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        if len(rows) > 500:      # sizing to every row's contents would read all of them
            view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
            view.horizontalHeader().setDefaultSectionSize(110)
        return view

    def _ordered(self) -> list:
        """The recording's analyses in reading order; the events before an evoked
        analysis that found nothing answering the stimulus."""
        out = ordered(self._run.results)
        ev = self._run.results.get("evoked")
        if ev is not None and "events" in self._run.results and any(f.code == "no response" for f in ev.flags):
            out.sort(key=lambda kv: kv[0] != "events")
            out.sort(key=lambda kv: ORDER.get(kv[0], 0))
        return out

    def _all(self) -> list[tuple[str, object, bool]]:
        """(key, result, whether it is the cell's) of the recording's analyses, then its cell's."""
        out = [(k, r, False) for k, r in self._run.results.items()]
        if self._cell is not None:
            out += [(k, r, True) for k, r in self._cell.results.items()]
        return out

    def _every_number(self) -> QtWidgets.QWidget:
        run = self._run
        defs = registry.definitions_by_analysis()
        rows = []
        for key, result, cellwise in self._all():
            for k, v in result.values.items():
                o = defs.get(key, {}).get(k)
                if o is None or k in ("units", "integral_units"):
                    continue
                rows.append((o, v, units_for(o.units, result, run.recording)))
        t = QtWidgets.QTableWidget(len(rows), 3)
        t.setHorizontalHeaderLabels(["Measure", "Value", "Reference"])
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setAlternatingRowColors(True)
        t.setShowGrid(False)
        for i, (o, v, u) in enumerate(rows):
            name = QtWidgets.QTableWidgetItem(o.title)
            name.setToolTip(o.definition)
            name.setData(QtCore.Qt.UserRole, o)
            val = QtWidgets.QTableWidgetItem(fmt(v, u))
            val.setFont(theme.numeric_font())
            t.setItem(i, 0, name)
            t.setItem(i, 1, val)
            t.setItem(i, 2, QtWidgets.QTableWidgetItem(o.source or "Clampwise"))
        t.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        t.resizeColumnToContents(1)
        t.itemSelectionChanged.connect(lambda: self._explain(t))
        return t

    def _explain(self, table) -> None:
        rows = table.selectionModel().selectedRows()
        if not rows:
            return
        o = table.item(rows[0].row(), 0).data(QtCore.Qt.UserRole)
        self.definition.setText(f"<b>{o.title}</b><br><br>{o.definition}<br><br><i>Reference:</i> "
                                f"{o.source or 'Clampwise'}<br><i>Checked against:</i> {o.verified_against}")

    def _flags(self) -> QtWidgets.QWidget:
        text = QtWidgets.QTextBrowser()
        run = self._run
        parts = []
        for key, result, cellwise in self._all():
            name = (registry.load()[key].label if key in registry.load() else key) + (", this cell" if cellwise else "")
            items = [f"<li><b>{f.code}</b>{' (' + f.where + ')' if f.where else ''}: {f.text}"
                     f"{' &mdash; limit ' + f.limit if f.limit else ''}{' [' + f.source + ']' if f.source else ''}"
                     f"{'' if f.severity == 'flag' else ' <i>(a note, not a quality flag)</i>'}</li>"
                     for f in result.flags]
            items += [f"<li>{n}</li>" for n in result.notes]
            if items:
                parts.append(f"<h3>{name}</h3><ul>{''.join(items)}</ul>")
        if run.notes:
            parts.insert(0, "<ul>" + "".join(f"<li>{n}</li>" for n in run.notes) + "</ul>")
        text.setHtml("".join(parts) or "<p>No flags and no notes.</p>")
        return text

    def _how(self) -> QtWidgets.QWidget:
        text = QtWidgets.QTextBrowser()
        run = self._run
        parts = [f"<p><b>Recording type:</b> {run.type_reason}.<br><b>Stimulus:</b> {run.stimulus_summary}.</p>"]
        for key, result, cellwise in self._all():
            name = (registry.load()[key].label if key in registry.load() else key) + (", this cell" if cellwise else "")
            a = registry.load().get(key)
            body = []
            if a is not None and a.source:
                body.append(f"<p><i>Reference:</i> {a.source}</p>")
            if result.choices:
                body.append("<p><i>Chosen, and why:</i></p><ul>" +
                            "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in result.choices.items()) + "</ul>")
            if result.methods:
                body.append("<p><i>Applied:</i></p><ul>" + "".join(f"<li>{m}</li>" for m in result.methods) + "</ul>")
            if result.changed_settings:
                body.append("<p><i>Settings changed from the default:</i> " +
                            ", ".join(f"{k} = {result.settings[k]}" for k in result.changed_settings) + "</p>")
            if result.from_cache:
                body.append("<p><i>Read from the project's saved results (nothing it depends on changed).</i></p>")
            parts.append(f"<h3>{name} (version {result.version})</h3>" + "".join(body))
        p = run.provenance
        if p is not None:
            parts.append(f"<h3>Provenance</h3><p>{p.file}<br>SHA-256 {p.content_hash[:16]}…<br>Clampwise "
                         f"{p.app_version}, analyzed {p.analyzed_at.replace('T', ' ')}"
                         f"{'<br>Preset: ' + p.preset if p.preset else ''}</p>")
        text.setHtml("".join(parts))
        return text
