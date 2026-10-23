"""The amplifier's controls in Live (what MultiClamp Commander does, with Commander itself still usable beside it):
every setting and button of rig.amplifier_control, built from its table, laid out as Commander's panel is: the mode
as three buttons, then each group in the order a cell is patched (the pipette, the seal test, whole cell, Rs
compensation, the bridge in current clamp, leak subtraction, the outputs), each on/off beside its value and each Auto
on the line of the value it sets, only the current mode's shown. Each control shows the amplifier's value as read
(unless it is being edited) and sends a change to Commander, whose own panel follows it."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

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


def _choices(c: ac.Control, mode: int) -> tuple:
    if c.key in ("primary_signal", "secondary_signal"):
        return tuple(x for x in c.choices if (x[0] < 7) == (mode == 0))         # each mode has its own signals
    return c.choices


class AmplifierPanel(QtWidgets.QWidget):
    """The controls; ``send`` takes ("set", key, value, mode) or ("act", name)."""

    def __init__(self, send) -> None:
        super().__init__()
        self.send = send
        self.mode = 0
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.rows: dict[str, list[QtWidgets.QWidget]] = {}      # a line's key -> everything on it
        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        lay = box("v", "xs")
        modes = box("h", "xs")                                   # the mode, as three buttons
        self.mode_buttons: dict[int, QtWidgets.QPushButton] = {}
        for value, words in MODES:
            b = widgets.button(words, lambda _=False, v=value: self.change("mode", v), f"{words}: the amplifier's "
                               "mode (sent through its own program)", quiet=True)
            b.setCheckable(True)
            modes.addWidget(b)
            self.mode_buttons[value] = b
        modes.addStretch(1)
        lay.addLayout(modes)
        grid = QtWidgets.QGridLayout()                           # caption | on | value | Auto, one line each
        grid.setHorizontalSpacing(theme.SPACE["xs"])
        grid.setVerticalSpacing(theme.SPACE["xs"] // 2)
        grid.setColumnStretch(2, 1)
        self.groups: dict[str, QtWidgets.QLabel] = {}
        paired = set(ON_FOR.values())
        for g in ORDER:
            caption = label(g, "section")
            self.groups[g] = caption
            grid.addWidget(caption, grid.rowCount(), 0, 1, 4)
            for c in ac.CONTROLS:
                if c.group != g or c.key == "mode" or c.key in paired:
                    continue
                row = grid.rowCount()
                on_key = ON_FOR.get(c.key)
                words = ac.BY_KEY[on_key].label.removesuffix(" on") if on_key else c.label
                cap = label(words, "muted")
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
                grid.addWidget(w, row, 2)
                parts.append(w)
                if c.key in AUTO_FOR:
                    b = self._action(AUTO_FOR[c.key], "Auto")
                    grid.addWidget(b, row, 3)
                    parts.append(b)
                self.rows[c.key] = parts
            if g == "Clamp":                                     # the membrane's own buttons, after the holding
                pair = box("h", "xs")
                for name in ("zap", "buzz"):
                    pair.addWidget(self._action(name, ac.ACTIONS[name][1]))
                pair.addStretch(1)
                grid.addLayout(pair, grid.rowCount(), 1, 1, 3)
            if g == "Outputs":
                row = grid.rowCount()
                cap = label("Output zero", "muted")
                grid.addWidget(cap, row, 0)
                grid.addWidget(self._action("auto_output_zero", "Auto"), row, 3)
                self.rows["auto_output_zero"] = [cap, self.buttons["auto_output_zero"]]
        lay.addLayout(grid)
        self.status = label("", "muted", wrap=True)
        lay.addWidget(self.status)
        self.setLayout(lay)
        self.show_mode(0)

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
