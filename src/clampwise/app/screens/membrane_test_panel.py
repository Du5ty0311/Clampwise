"""The membrane test's view in Live's acquisition panel (rig.membrane_test): the averaged pulse, what it is reading
(a cell, a seal, the pipette alone) with Rt, Rs, Rm, Cm, tau and the holding, and Rt and Rs over the test."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .. import theme, widgets
from ..widgets import box, label

ROWS = (("rt", "Total (Rt)"), ("rs", "Access (Rs)"), ("rm", "Membrane (Rm)"), ("cm", "Capacitance (Cm)"),
        ("tau", "Tau"), ("holding", "Holding"))


def _resistance(mohm: float) -> str:
    if not math.isfinite(mohm):
        return "–"
    return f"{mohm / 1e3:.2f} GOhm" if mohm >= 1000 else f"{mohm:.1f} MOhm"


class MembraneTestView(QtWidgets.QWidget):
    def __init__(self, palette) -> None:
        super().__init__()
        self.palette_ = palette
        lay = box("h", "s")
        plots = box("v", "xs")
        self.pulse = pg.PlotWidget()
        self.pulse.getPlotItem().setLabel("bottom", "Time (ms)")
        self.history = pg.PlotWidget()
        self.history.setMaximumHeight(160)
        self.history.getPlotItem().setLabel("bottom", "Seconds")
        self.history.getPlotItem().setLabel("left", "MOhm")
        self.history.addLegend(offset=(-10, 10))
        plots.addWidget(self.pulse, 3)
        plots.addWidget(self.history, 1)
        lay.addLayout(plots, 3)
        side = QtWidgets.QFrame()
        side.setObjectName("desk")
        side.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(theme.SPACE["m"])
        grid.setVerticalSpacing(theme.SPACE["xs"])
        self.what = widgets.Badge("muted", "–")
        grid.addWidget(label("Reading", "muted"), 0, 0)
        grid.addWidget(self.what, 0, 1)
        self.values: dict[str, QtWidgets.QLabel] = {}
        for k, (key, caption) in enumerate(ROWS, start=1):
            grid.addWidget(label(caption, "muted"), k, 0)
            self.values[key] = label("–", "meter")
            grid.addWidget(self.values[key], k, 1)
        self.grain = label("", "muted", wrap=True)
        grid.addWidget(self.grain, len(ROWS) + 1, 0, 1, 2)
        grid.setRowStretch(len(ROWS) + 2, 1)
        side.setLayout(grid)
        lay.addWidget(side, 1)
        self.setLayout(lay)
        self.on_theme(palette)

    def on_theme(self, palette) -> None:
        self.palette_ = palette
        series = theme.series(palette)
        for p in (self.pulse, self.history):
            theme.style_plot(p, palette)
            p.clear()
        for p in (self.pulse, self.history):                # time never before 0, however zoomed or dragged
            p.getViewBox().setLimits(xMin=0)
        # 0 and the holding measured (the middle of the two steps): an open, zeroed pipette's trace is centred on 0
        self.zero = pg.InfiniteLine(0, angle=0, pen=pg.mkPen(palette.text_muted, width=1, style=QtCore.Qt.DashLine),
                                    label="0", labelOpts={"position": 0.02, "color": palette.text_muted})
        self.holding_line = pg.InfiniteLine(0, angle=0, pen=pg.mkPen(series[1], width=1),
                                            label="Holding", labelOpts={"position": 0.98, "color": series[1]})
        self.holding_line.setVisible(False)
        for line in (self.zero, self.holding_line):
            self.pulse.addItem(line, ignoreBounds=True)
        self.trace = self.pulse.plot([], [], pen=pg.mkPen(series[0], width=1))
        self.fit_curves = [self.pulse.plot([], [], pen=pg.mkPen(palette.bad, width=2)) for _ in range(2)]
        self.pulse.getPlotItem().enableAutoRange(y=False)
        self._shown = None
        self.rt_curve = self.history.plot([], [], pen=pg.mkPen(series[0], width=1), name="Rt")
        self.rs_curve = self.history.plot([], [], pen=pg.mkPen(series[1], width=1), name="Rs")

    def clear(self) -> None:
        self.trace.setData([], [])
        for c in self.fit_curves:
            c.setData([], [])
        self._shown = None
        self.pulse.getPlotItem().enableAutoRange(y=False)
        self._yrange = None
        self.rt_curve.setData([], [])
        self.rs_curve.setData([], [])
        for v in self.values.values():
            v.setText("–")
        self.what.set("muted", "–")
        self.grain.setText("")

    def show_reading(self, r, history) -> None:
        if r is None or r is self._shown:
            return
        self._shown = r
        self.trace.setData(r.t_ms, r.trace)
        for k, c in enumerate(self.fit_curves):          # each edge's fitted exponential, red, as Clampex draws it
            if k < len(r.fits):
                x, fy = r.fits[k]
                order = np.argsort(x)
                c.setData(np.asarray(x)[order], np.asarray(fy)[order])
            else:
                c.setData([], [])
        self._steady_y(r.trace)
        self.pulse.getPlotItem().setLabel("left", f"{'Current' if r.unit == 'pA' else 'Potential'} ({r.unit})")
        words = {"cell": ("ok", "a cell"), "seal": ("ok", "a seal"), "open pipette": ("warn", "the pipette alone"),
                 "no step": ("bad", "no step seen")}.get(r.what, ("muted", r.what))
        self.what.set(*words)
        self.values["rt"].setText(_resistance(r.rt_mohm))
        self.values["rs"].setText(_resistance(r.rs_mohm))
        self.values["rm"].setText(_resistance(r.rm_mohm))
        self.values["cm"].setText(f"{r.cm_pf:.1f} pF" if math.isfinite(r.cm_pf) else "–")
        self.values["tau"].setText(f"{r.tau_ms:.3g} ms" if math.isfinite(r.tau_ms) else "–")
        self.values["holding"].setText(f"{r.holding:.1f} {r.unit}" if math.isfinite(r.holding) else "–")
        self.holding_line.setVisible(math.isfinite(r.holding))
        if math.isfinite(r.holding):
            self.holding_line.setValue(r.holding)
        if math.isfinite(r.resolution):
            self.grain.setText(f"One sample count is {r.resolution:.2g} {r.unit} at the amplifier's present gain: "
                               "a larger pulse, more pulses averaged or a higher gain resolve Rt more finely.")
        h = list(history)[-3000:]
        if h:
            t = np.array([x.at for x in h])
            self.rt_curve.setData(t, np.array([x.rt_mohm for x in h], float), connect="finite")
            self.rs_curve.setData(t, np.array([x.rs_mohm for x in h], float), connect="finite")

    def _steady_y(self, y: np.ndarray) -> None:
        """The scale kept still while the trace fits it (a scale that follows every pulse jumps with the noise):
        set afresh only when the trace leaves it or fills less than a third of it."""
        if not y.size:
            return
        lo, hi = float(np.min(y)), float(np.max(y))
        span = max(hi - lo, 1e-9)
        now = getattr(self, "_yrange", None)
        if now is None or lo < now[0] or hi > now[1] or span < (now[1] - now[0]) / 3:
            self._yrange = (lo - 0.1 * span, hi + 0.1 * span)
            self.pulse.getPlotItem().setYRange(*self._yrange, padding=0)
