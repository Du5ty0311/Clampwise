"""A long recording drawn without stalling the window (B1's done-when: a 60-min
file opens, scrolls and zooms with no stall over 0.2 s).

The sweep is read and its min/max pyramid built in a background thread; the
plot then shows the envelope at the view's resolution (for each pixel the
lowest and highest sample under it, as an oscilloscope does), redrawn from the
pyramid on every zoom or scroll, and the raw samples once zoomed in far enough
to show them.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from ..io.pyramid import Pyramid

LONG_SAMPLES = 2_000_000          # traces longer than this are drawn as an envelope


class _Builder(QtCore.QThread):
    built = QtCore.Signal(object, object)          # (samples float32, Pyramid)

    def __init__(self, recording, sweep: int) -> None:
        super().__init__()
        self.recording, self.sweep = recording, sweep

    def run(self) -> None:
        y = np.asarray(self.recording.sweep(self.sweep), dtype=np.float32)
        self.built.emit(y, Pyramid.build(y, self.recording.sample_rate_hz))


class LongTrace(QtCore.QObject):
    """One long sweep in a plot, drawn from its envelope."""

    ready = QtCore.Signal()

    def __init__(self, plot: pg.PlotWidget, recording, sweep: int, pen) -> None:
        super().__init__()
        self.plot, self.recording, self.sweep = plot, recording, sweep
        self.fs = recording.sample_rate_hz
        self.t0 = 0.0
        self.y: np.ndarray | None = None
        self.pyramid: Pyramid | None = None
        self.item = pg.PlotDataItem(pen=pen)
        plot.addItem(self.item)
        self.item.setClipToView(False)        # a few thousand points; and removing a clip-to-view item trips pyqtgraph
        self._builder = _Builder(recording, sweep)
        self._builder.built.connect(self._built)
        self._builder.start()
        plot.getPlotItem().getViewBox().sigXRangeChanged.connect(self._redraw)

    @property
    def is_ready(self) -> bool:
        return self.y is not None

    def _built(self, y, pyramid) -> None:
        self.y, self.pyramid = y, pyramid
        self._redraw()
        self.ready.emit()

    def value_at(self, t_s: float) -> float:
        if self.y is None:
            return float("nan")
        return float(self.y[int(np.clip(round(t_s * self.fs), 0, self.y.size - 1))])

    def values_at(self, t_s) -> np.ndarray:
        """The samples at many times at once."""
        if self.y is None:
            return np.full(len(t_s), np.nan)
        i = np.clip(np.round(np.asarray(t_s, float) * self.fs).astype(np.int64), 0, self.y.size - 1)
        return self.y[i].astype(float)

    def full_range(self) -> tuple[float, float]:
        return 0.0, self.recording.n_samples / self.fs

    def _redraw(self, *_) -> None:
        if self.y is None or self.pyramid is None:
            return
        try:
            x0, x1 = self.plot.getPlotItem().getViewBox().viewRange()[0]
        except RuntimeError:              # the plot is gone
            return
        x0, x1 = max(x0, 0.0), min(x1, self.y.size / self.fs)
        if x1 <= x0:
            return
        pixels = max(200, int(self.plot.width()))
        t, mins, maxs, factor = self.pyramid.envelope(x0, x1, pixels)
        if factor == 1:                   # zoomed in: the samples themselves
            a, b = int(x0 * self.fs), min(self.y.size, int(np.ceil(x1 * self.fs)) + 1)
            self.item.setData(np.arange(a, b) / self.fs, self.y[a:b])
        else:                             # the envelope: each pixel's lowest and highest sample
            self.item.setData(np.repeat(t, 2), np.column_stack([mins, maxs]).ravel())

    def stop(self) -> None:
        """Let go of the plot; a build still running finishes in the background and
        is thrown away (waiting for it held the window for seconds)."""
        try:
            self.plot.getPlotItem().getViewBox().sigXRangeChanged.disconnect(self._redraw)
        except (TypeError, RuntimeError):
            pass
        try:
            self._builder.built.disconnect(self._built)
        except (TypeError, RuntimeError):
            pass
        if self._builder.isRunning():
            builder = self._builder
            _finishing.add(builder)
            builder.finished.connect(lambda b=builder: _finishing.discard(b))


_finishing: set = set()          # builds let go of, kept alive until they end
