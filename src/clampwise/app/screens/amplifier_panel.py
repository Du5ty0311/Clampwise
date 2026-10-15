"""The amplifier's controls in Live (what MultiClamp Commander does, with Commander itself still usable beside it):
every setting and button of rig.amplifier_control, built from its table,
grouped as Commander groups them, only the current mode's shown. Each control shows the amplifier's value as read
(unless it is being edited) and sends a change to Commander, whose own panel follows it."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

from ...rig import amplifier_control as ac
from .. import theme, widgets
from ..widgets import box, label

ACTION_GROUPS = {"auto_pipette_offset": "Pipette", "auto_fast": "Pipette", "auto_slow": "Pipette",
                 "auto_whole_cell": "Whole cell", "auto_bridge": "Bridge and neutralization",
                 "auto_leak": "Leak subtraction", "auto_output_zero": "Outputs", "zap": "Clamp", "buzz": "Clamp"}


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
        self.rows: dict[str, tuple] = {}
        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        lay = box("v", "xs")
        self.groups: dict[str, QtWidgets.QGroupBox] = {}
        for g in ac.GROUPS:
            frame = QtWidgets.QGroupBox(g)
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(theme.SPACE["s"])
            grid.setVerticalSpacing(theme.SPACE["xs"])
            grid.setColumnStretch(1, 1)
            frame.setLayout(grid)
            self.groups[g] = frame
            lay.addWidget(frame)
        for c in ac.CONTROLS:
            grid = self.groups[c.group].layout()
            row = grid.rowCount()
            w = self._make(c)
            cap = label(c.label, "muted")
            grid.addWidget(cap, row, 0)
            grid.addWidget(w, row, 1)
            self.widgets[c.key], self.rows[c.key] = w, (cap, w)
        for name, (_, words, _) in ac.ACTIONS.items():
            grid = self.groups[ACTION_GROUPS[name]].layout()
            b = widgets.button(words, lambda _=False, n=name: self.act(n), f"{words}, as the amplifier's own button",
                               quiet=True)
            grid.addWidget(b, grid.rowCount(), 0, 1, 2)
            self.buttons[name] = b
        self.status = label("", "muted", wrap=True)
        lay.addWidget(self.status)
        self.setLayout(lay)
        self.show_mode(0)

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
        for c in ac.CONTROLS:
            cap, w = self.rows[c.key]
            for x in (cap, w):
                x.setVisible(mode in c.modes)
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
        for g, frame in self.groups.items():
            shown = [c for c in ac.CONTROLS if c.group == g and mode in c.modes] + \
                    [n for n, gg in ACTION_GROUPS.items() if gg == g and mode in ac.ACTIONS[n][2]]
            frame.setVisible(bool(shown))

    def show_state(self, amp) -> None:
        """The amplifier as read: each value, unless it is being edited."""
        if amp is None or not amp.connected:
            self.setEnabled(False)
            self.status.setText("Commander is not open: the controls need it (they go through it).")
            return
        self.setEnabled(True)
        if amp.mode is not None and amp.mode != self.mode:
            self.show_mode(amp.mode)
        for c in ac.CONTROLS:
            w = self.widgets[c.key]
            if w.hasFocus() or c.key not in amp.raw:
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
