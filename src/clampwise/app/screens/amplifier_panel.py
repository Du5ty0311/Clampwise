"""The amplifier's controls in Live (what MultiClamp Commander does, with Commander itself still usable beside it):
every setting and button of rig.amplifier_control, built from its table, laid out as Commander's panel is: the mode
as three buttons, then each group in the order a cell is patched (the pipette, the seal test, whole cell, Rs
compensation, the bridge in current clamp, leak subtraction, the outputs), each on/off beside its value and each Auto
on the line of the value it sets, only the current mode's shown. Each control shows the amplifier's value as read
(unless it is being edited) and sends a change to Commander, whose own panel follows it."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from ...rig import amplifier_control as ac
from .. import theme, widgets
from ..widgets import box, label

ORDER = ("Clamp", "Pipette", "Seal test", "Whole cell", "Rs compensation", "Bridge and neutralization",
         "Leak subtraction", "Outputs")                          # the order a cell is patched in
ON_FOR = {"holding": "holding_on", "test_signal_amplitude": "test_signal_on", "cm": "whole_cell_on",
          "rs_correction": "rs_comp_on", "bridge": "bridge_on", "neutralization": "neutralization_on",
          "leak_sub": "leak_sub_on"}                             # a value and the switch that puts it on, one line
AUTO_FOR = {"pipette_offset": "auto_pipette_offset", "fast_cap": "auto_fast", "slow_cap": "auto_slow",
            "cm": "auto_whole_cell", "bridge": "auto_bridge", "leak_sub": "auto_leak"}   # its Auto, on its line
MODES = ((0, "VC"), (2, "I=0"), (1, "IC"))                       # as Commander's buttons, left to right
# compact (the acquisition's window on its own screen, as Commander's own small window): the meters on top, the groups
# in three tabs, two values on a line where they belong together, shorter captions
TABS = (("Clamp", ("Clamp", "Seal test", "Pipette", "Whole cell", "Rs compensation", "Bridge and neutralization",
                   "Leak subtraction")),
        ("Outputs", ("Outputs",)))
PAIRS = {"fast_cap": "fast_tau", "slow_cap": "slow_tau", "cm": "rs", "test_signal_amplitude": "test_signal_hz",
         "rs_correction": "rs_prediction", "primary_gain": "primary_lpf", "secondary_gain": "secondary_lpf"}
SHORT = {"rs_correction": "Rs comp", "rs_bandwidth": "Bandwidth", "primary_signal": "Primary",
         "primary_gain": "Gain, filter", "secondary_signal": "Secondary", "secondary_gain": "Gain, filter",
         "leak_sub": "Leak sub", "bridge": "Bridge", "neutralization": "Neutralization"}


def _choices(c: ac.Control, mode: int) -> tuple:
    if c.key in ("primary_signal", "secondary_signal"):
        return tuple(x for x in c.choices if (x[0] < 7) == (mode == 0))         # each mode has its own signals
    return c.choices


class AmplifierPanel(QtWidgets.QWidget):
    """The controls; ``send`` takes ("set", key, value, mode) or ("act", name)."""

    def __init__(self, send, compact: bool = False) -> None:
        super().__init__()
        self.send, self.compact = send, compact
        self.mode = 0
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.rows: dict[str, list[QtWidgets.QWidget]] = {}      # a line's key -> everything on it
        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        self.groups: dict[str, QtWidgets.QLabel] = {}
        lay = box("v", "xs")
        if compact:
            lay.addLayout(self._meters())
        modes = box("h", "xs")                                   # the mode, as three buttons
        self.mode_buttons: dict[int, QtWidgets.QPushButton] = {}
        for value, words in MODES:
            b = widgets.button(words, lambda _=False, v=value: self.change("mode", v), f"{words}: the amplifier's "
                               "mode (sent through its own program)", quiet=True)
            b.setCheckable(True)
            modes.addWidget(b, 1 if compact else 0)
            self.mode_buttons[value] = b
        if not compact:
            modes.addStretch(1)
        lay.addLayout(modes)
        if compact:
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.setDocumentMode(True)
            self.pages: list[QtWidgets.QScrollArea] = []
            for name, groups in TABS:
                grid = self._grid()
                for g in groups:
                    self._group(grid, g)
                grid.setRowStretch(grid.rowCount(), 1)
                page = QtWidgets.QWidget()
                page.setLayout(grid)
                scroll = QtWidgets.QScrollArea()            # a short screen: the tab scrolls, nothing is cut off
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
                scroll.setWidget(page)
                self.pages.append(scroll)
                self.tabs.addTab(scroll, name)
            lay.addWidget(self.tabs, 1)
        else:
            grid = self._grid()
            for g in ORDER:
                self._group(grid, g)
            lay.addLayout(grid)
        self.status = label("", "muted", wrap=True)
        lay.addWidget(self.status)
        self.setLayout(lay)
        self.show_mode(0)
        if compact:
            self._fit_width()

    def _grid(self) -> QtWidgets.QGridLayout:
        grid = QtWidgets.QGridLayout()        # caption | on | value (| second value, compact) | Auto, a line each
        grid.setHorizontalSpacing(theme.SPACE["xs"])
        grid.setVerticalSpacing(theme.SPACE["xs"] // 2)
        grid.setColumnStretch(2, 1)
        if self.compact:
            grid.setColumnStretch(3, 1)
        return grid

    def _group(self, grid: QtWidgets.QGridLayout, g: str) -> None:
        """Group ``g``'s lines: each value with its switch before it and its Auto after it (compact: a second value
        that belongs with it on the same line, and no group heading: the tab and the captions say it)."""
        compact = self.compact
        auto_col = 4 if compact else 3
        paired = set(ON_FOR.values()) | (set(PAIRS.values()) if compact else set())
        if not compact:
            caption = label(g, "section")
            self.groups[g] = caption
            grid.addWidget(caption, grid.rowCount(), 0, 1, 4)
        for c in ac.CONTROLS:
            if c.group != g or c.key == "mode" or c.key in paired:
                continue
            row = grid.rowCount()
            on_key = ON_FOR.get(c.key)
            words = ac.BY_KEY[on_key].label.removesuffix(" on") if on_key else c.label
            cap = label(SHORT.get(c.key, words) if compact else words, "muted")
            cap.setToolTip(c.label)
            parts = [cap]
            grid.addWidget(cap, row, 0)
            if on_key:
                on = self._make(ac.BY_KEY[on_key])
                on.setToolTip(ac.BY_KEY[on_key].label)
                self.widgets[on_key] = on
                grid.addWidget(on, row, 1)
                parts.append(on)
            w = self._make(c)
            self.widgets[c.key] = w
            second = PAIRS.get(c.key) if compact else None
            if second:
                w.setToolTip(c.label)
                w2 = self._make(ac.BY_KEY[second])
                w2.setToolTip(ac.BY_KEY[second].label)
                self.widgets[second] = w2
                self.rows[second] = [w2]
                grid.addWidget(w, row, 2)
                grid.addWidget(w2, row, 3)
            elif compact:
                grid.addWidget(w, row, 2, 1, 2)
            else:
                grid.addWidget(w, row, 2)
            parts.append(w)
            if c.key in AUTO_FOR:
                b = self._action(AUTO_FOR[c.key], "Auto")
                grid.addWidget(b, row, auto_col)
                parts.append(b)
            self.rows[c.key] = parts
        if g == "Clamp":                                         # the membrane's own buttons, after the holding
            pair = box("h", "xs")
            for name in ("zap", "buzz"):
                pair.addWidget(self._action(name, ac.ACTIONS[name][1]))
            pair.addStretch(1)
            grid.addLayout(pair, grid.rowCount(), 1, 1, auto_col)
        if g == "Outputs":
            row = grid.rowCount()
            cap = label("Output zero", "muted")
            grid.addWidget(cap, row, 0)
            grid.addWidget(self._action("auto_output_zero", "Auto"), row, auto_col)
            self.rows["auto_output_zero"] = [cap, self.buttons["auto_output_zero"]]

    def _meters(self) -> QtWidgets.QLayout:
        """Compact: the channel and its mode, then its two meters, as on top of Commander's window."""
        lay = box("v", "xs")
        top = box("h", "s")
        self.channel = label("Amplifier", "heading")
        self.mode_words = label("", "muted")
        top.addWidget(self.channel, 1)
        top.addWidget(self.mode_words)
        lay.addLayout(top)
        meters = QtWidgets.QGridLayout()
        meters.setHorizontalSpacing(theme.SPACE["s"])
        meters.setVerticalSpacing(0)
        self.meter_v, self.meter_i = label("–", "meter"), label("–", "meter")
        for k, (cap, w) in enumerate((("V (mV)", self.meter_v), ("I (pA)", self.meter_i))):
            w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            meters.addWidget(label(cap, "muted"), 0, k)
            meters.addWidget(w, 1, k)
        lay.addLayout(meters)
        self.telegraph = widgets.OneLine("")
        lay.addWidget(self.telegraph)
        return lay

    def show_meters(self, channel: str, mode: str, v: str, i: str, telegraph: str = "") -> None:
        """Compact: the channel, its mode and its meters as Live reads them (the same as its side panel's)."""
        if not self.compact:
            return
        self.channel.setText(channel)
        self.mode_words.setText(mode)
        self.meter_v.setText(v)
        self.meter_i.setText(i)
        self.telegraph.setText(f"Telegraph: {telegraph}" if telegraph else "")

    def _fit_width(self) -> None:
        """Compact: as wide as its widest tab needs (plus a scroll bar), so no Auto is ever cut off at the edge."""
        bar = self.pages[0].verticalScrollBar().sizeHint().width()
        need = max(p.widget().minimumSizeHint().width() for p in self.pages) + bar + 4
        for p in self.pages:
            p.setMinimumWidth(need)

    def _action(self, name: str, words: str) -> QtWidgets.QPushButton:
        b = widgets.button(words, lambda _=False, n=name: self.act(n), f"{ac.ACTIONS[name][1]}, as the amplifier's "
                           "own button", quiet=True)
        self.buttons[name] = b
        return b

    def _make(self, c: ac.Control) -> QtWidgets.QWidget:
        if c.kind == "b":
            w = QtWidgets.QCheckBox()
            w.clicked.connect(lambda on, k=c.key: self.change(k, bool(on)))
        elif c.kind == "u":
            w = QtWidgets.QComboBox()
            w.activated.connect(lambda i, k=c.key: self.change(k, self.widgets[k].itemData(i)))
        else:
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(c.low, c.high)
            w.setSingleStep(c.step)
            w.setDecimals(2 if c.step < 0.1 else 1 if c.step < 1 else 0)
            w.setKeyboardTracking(False)
            w.editingFinished.connect(lambda k=c.key: self.change(k, self.widgets[k].value()))
        return w

    def show_mode(self, mode: int) -> None:
        """Only the current mode's settings and buttons (as the amplifier's own panel)."""
        self.mode = mode
        for value, b in self.mode_buttons.items():
            b.setChecked(value == mode)
        for c in ac.CONTROLS:
            w = self.widgets.get(c.key)
            if w is None:
                continue
            if c.key in self.rows:                               # a line: shown with its value's modes
                for x in self.rows[c.key]:
                    x.setVisible(mode in c.modes)
            w.setVisible(mode in c.modes)
            if c.kind == "u":
                keep = w.currentData()
                w.blockSignals(True)
                w.clear()
                for value, words in _choices(c, mode):
                    w.addItem(words, value)
                i = w.findData(keep)
                w.setCurrentIndex(i if i >= 0 else -1)
                w.blockSignals(False)
            elif c.kind == "d" and isinstance(c.unit, dict):
                w.setSuffix(f" {c.unit.get(mode, '')}")
            elif c.kind == "d":
                w.setSuffix(f" {c.unit}")
        for name, (_, _, modes) in ac.ACTIONS.items():
            self.buttons[name].setVisible(mode in modes)
        for g, caption in self.groups.items():
            shown = [c for c in ac.CONTROLS if c.group == g and c.key != "mode" and mode in c.modes] + \
                    [n for n in ("zap", "buzz") if g == "Clamp" and mode in ac.ACTIONS[n][2]]
            caption.setVisible(bool(shown))

    def show_state(self, amp) -> None:
        """The amplifier as read: each value, unless it is being edited."""
        if amp is None or not amp.connected:
            self.setEnabled(False)
            self.status.setText("Commander is not open: the controls need it (they go through it).")
            return
        self.setEnabled(True)
        if amp.mode is not None and amp.mode != self.mode:
            self.show_mode(amp.mode)
        elif amp.mode is not None:                               # a mode asked for but not taken: shown as it is
            for v, b in self.mode_buttons.items():
                b.setChecked(v == amp.mode)
        for c in ac.CONTROLS:
            w = self.widgets.get(c.key)
            if w is None or w.hasFocus() or c.key not in amp.raw:
                continue
            w.blockSignals(True)
            if c.kind == "b":
                w.setChecked(bool(amp.raw[c.key]))
            elif c.kind == "u":
                raw = amp.raw[c.key]
                i = w.findData(int(raw) if c.key not in ac.DOUBLE_CHOICES else raw)
                if i < 0 and c.key in ac.DOUBLE_CHOICES:                   # a value between the usual ones
                    i = min(range(w.count()), key=lambda j: abs(w.itemData(j) - raw), default=-1)
                w.setCurrentIndex(i)
            else:
                v = amp.values.get(c.key)
                if isinstance(v, (int, float)):
                    w.setValue(float(v))
            w.blockSignals(False)

    def change(self, key: str, value) -> None:
        c = ac.BY_KEY[key]
        if key == "mode":
            for v, b in self.mode_buttons.items():                # shown as asked until the amplifier says
                b.setChecked(v == value)
        self.status.setText(f"{c.label}: sent.")
        self.send(("set", key, value, self.mode))

    def act(self, name: str) -> None:
        words = ac.ACTIONS[name][1]
        if name == "zap":
            ok = QtWidgets.QMessageBox.question(self, "Zap", "Zap the membrane under the pipette now?")
            if ok != QtWidgets.QMessageBox.Yes:
                return
        self.status.setText(f"{words}: sent.")
        self.send(("act", name))

    def said(self, why: str) -> None:
        if why:
            self.status.setText(f"Not done: {why}.")
