"""The simple stimulus editor: one of the user's protocols opened, each
output's epochs and holding edited in a table, the run's sweeps and timing, every sweep previewed as edited, saved
under a new name as a protocol the acquisition program also opens (rig.protocol_edit)."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from ...rig import acquire, protocol_edit
from ...rig.growing import scratch
from .. import theme, widgets
from ..widgets import box, label

COLUMNS = ("Type", "Level", "Level + a sweep", "Duration (ms)", "Duration + a sweep (ms)", "Period (ms)",
           "Width (ms)", "Digital outputs")


class ProtocolEditor(QtWidgets.QDialog):
    """Edits a copy of ``path``; ``saved`` is the new protocol's path once saved."""

    def __init__(self, parent, path: str, palette) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit {Path(path).stem}")
        self.resize(1000, 720)
        self.palette_ = palette
        self.source = Path(path)
        self.edit = protocol_edit.load(path)
        self.saved: Path | None = None
        self._busy = False
        lay = box("v", "s", "m")
        run = QtWidgets.QHBoxLayout()
        self.sweeps = QtWidgets.QSpinBox()
        self.sweeps.setRange(1, 10_000)
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(0.0, 3600.0)
        self.interval.setDecimals(3)
        self.interval.setSuffix(" s")
        self.length = QtWidgets.QDoubleSpinBox()
        self.length.setRange(1.0, 600_000.0)
        self.length.setDecimals(1)
        self.length.setSuffix(" ms")
        for caption, w in (("Sweeps", self.sweeps), ("Start to start", self.interval), ("Sweep length", self.length)):
            run.addWidget(label(caption, "muted"))
            run.addWidget(w)
        run.addStretch(1)
        lay.addLayout(run)
        self.tabs = QtWidgets.QTabWidget()
        self.tables: dict[int, QtWidgets.QTableWidget] = {}
        self.holding: dict[int, QtWidgets.QDoubleSpinBox] = {}
        self.enabled: dict[int, QtWidgets.QCheckBox] = {}
        units = self.edit.protocol.dac_units
        for dac in range(acquire.N_AO):
            page = QtWidgets.QWidget()
            pl = box("v", "xs")
            top = box("h", "s")
            hold = QtWidgets.QDoubleSpinBox()
            hold.setRange(-10_000.0, 10_000.0)
            hold.setDecimals(2)
            unit = units[dac] if dac < len(units) else ""
            hold.setSuffix(f" {unit}")
            on = QtWidgets.QCheckBox("Plays its epochs")
            top.addWidget(label("Holding", "muted"))
            top.addWidget(hold)
            top.addWidget(on)
            top.addStretch(1)
            top.addWidget(widgets.button("Add epoch", lambda _=False, d=dac: self.add_epoch(d), quiet=True))
            top.addWidget(widgets.button("Remove epoch", lambda _=False, d=dac: self.remove_epoch(d), quiet=True))
            pl.addLayout(top)
            table = QtWidgets.QTableWidget(0, len(COLUMNS))
            table.setHorizontalHeaderLabels(COLUMNS)
            table.horizontalHeader().setStretchLastSection(True)
            pl.addWidget(table, 1)
            page.setLayout(pl)
            name = f"Output {dac}" + (" (the command)" if dac == 0 else "")
            self.tabs.addTab(page, name)
            self.tables[dac], self.holding[dac], self.enabled[dac] = table, hold, on
        lay.addWidget(self.tabs, 2)
        lay.addWidget(label("Each sweep as edited (the command's numbers are read in the amplifier's unit when "
                            "played: mV in voltage clamp, pA in current clamp)", "muted", wrap=True))
        self.plot = pg.PlotWidget()
        theme.style_plot(self.plot, palette)
        self.plot.getPlotItem().setLabel("bottom", "Time (ms)")
        lay.addWidget(self.plot, 2)
        self.status = label("", "muted", wrap=True)
        lay.addWidget(self.status)
        bottom = box("h", "s")
        bottom.addWidget(label("Save as", "muted"))
        self.name = QtWidgets.QLineEdit(self._free_name())
        bottom.addWidget(self.name, 1)
        bottom.addWidget(widgets.button("Cancel", self.reject, quiet=True))
        bottom.addWidget(widgets.primary("Save", self.save, "Save as a new protocol beside the original (the "
                                                            "original is never changed)"))
        lay.addLayout(bottom)
        self.setLayout(lay)
        self._fill()
        for w in (self.sweeps, self.interval, self.length):
            w.valueChanged.connect(self.changed)
        self.tabs.currentChanged.connect(lambda *_: self.preview())
        self.preview()

    # ---- the edit <-> the widgets
    def _ms(self, samples: int) -> float:
        return samples * 1000.0 / self.edit.rate_hz

    def _samples(self, ms: float) -> int:
        return int(round(ms * self.edit.rate_hz / 1000.0))

    def _fill(self) -> None:
        self._busy = True
        e = self.edit
        self.sweeps.setValue(e.sweeps)
        self.interval.setValue(e.start_to_start_s)
        self.length.setValue(self._ms(e.sweep_samples))
        for dac, table in self.tables.items():
            self.holding[dac].setValue(e.holding[dac])
            self.enabled[dac].setChecked(e.enabled[dac])
            eps = e.epochs.get(dac, [])
            table.setRowCount(len(eps))
            table.setVerticalHeaderLabels([chr(65 + k) for k in range(len(eps))])
            for k, ep in enumerate(eps):
                kind = QtWidgets.QComboBox()
                for value, words in protocol_edit.TYPES.items():
                    kind.addItem(words, value)
                kind.setCurrentIndex(max(kind.findData(ep.kind), 0))
                kind.currentIndexChanged.connect(self.changed)
                table.setCellWidget(k, 0, kind)
                values = (ep.level, ep.level_inc, self._ms(ep.samples), self._ms(ep.samples_inc), self._ms(ep.period),
                          self._ms(ep.width), ep.digital)
                for c, v in enumerate(values, start=1):
                    spin = QtWidgets.QDoubleSpinBox() if c < 7 else QtWidgets.QSpinBox()
                    if c < 7:
                        spin.setRange(-100_000.0, 600_000.0)
                        spin.setDecimals(2 if c < 3 else 3)
                    else:
                        spin.setRange(0, 255)
                        drives = dac == e.protocol.active_dac and e.protocol.digital_enable
                        spin.setToolTip("The digital outputs during this epoch, as bits (1 = output 0, 2 = output 1, "
                                        "4 = output 2...)" if drives else
                                        "The digital outputs are off in this protocol" if not
                                        e.protocol.digital_enable else
                                        f"The digital outputs follow output {e.protocol.active_dac}'s epochs")
                        spin.setEnabled(drives)
                    spin.setValue(v)
                    spin.valueChanged.connect(self.changed)
                    table.setCellWidget(k, c, spin)
            table.verticalHeader().setDefaultSectionSize(QtWidgets.QDoubleSpinBox().sizeHint().height() + 6)
            table.resizeColumnsToContents()
            self.holding[dac].valueChanged.connect(self.changed)
            self.enabled[dac].toggled.connect(self.changed)
        self._busy = False

    def _read(self) -> protocol_edit.Edit:
        e = copy.copy(self.edit)
        e.sweeps, e.start_to_start_s = self.sweeps.value(), self.interval.value()
        e.sweep_samples = self._samples(self.length.value())
        e.holding = [self.holding[d].value() for d in range(acquire.N_AO)]
        e.enabled = [self.enabled[d].isChecked() for d in range(acquire.N_AO)]
        e.epochs = {}
        for dac, table in self.tables.items():
            eps = []
            for k in range(table.rowCount()):
                w = [table.cellWidget(k, c) for c in range(len(COLUMNS))]
                eps.append(protocol_edit.Epoch(w[0].currentData(), w[1].value(), w[2].value(), self._samples(w[3].value()),
                                               self._samples(w[4].value()), self._samples(w[5].value()),
                                               self._samples(w[6].value()), w[7].value()))
            if eps:
                e.epochs[dac] = eps
        return e

    def add_epoch(self, dac: int) -> None:
        self.edit = self._read()
        eps = self.edit.epochs.setdefault(dac, [])
        level = eps[-1].level if eps else self.edit.holding[dac]
        eps.append(protocol_edit.Epoch(1, level, 0.0, self._samples(10.0)))
        self._fill()
        self.changed()

    def remove_epoch(self, dac: int) -> None:
        self.edit = self._read()
        if self.edit.epochs.get(dac):
            self.edit.epochs[dac].pop()
        self._fill()
        self.changed()

    # ---- preview and save
    def changed(self, *_) -> None:
        if not self._busy:
            self.preview()

    def preview(self) -> None:
        dac = self.tabs.currentIndex()
        self.plot.clear()
        try:
            blob = protocol_edit.to_bytes(self._read())
        except ValueError as exc:
            self.status.setText(f"Not playable as it is: {exc}.")
            return
        folder = scratch()
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", suffix=".pro", dir=folder, delete=False) as fh:
            fh.write(blob)
        try:
            pr = acquire.read_protocol(fh.name)
            units, _ = acquire.waveforms(pr)
        except Exception as exc:                          # said, never a crash: the edit is not saved
            self.status.setText(f"Not readable as it is: {exc}.")
            return
        finally:
            Path(fh.name).unlink(missing_ok=True)
        series = theme.series(self.palette_)
        t = np.arange(units.shape[2]) * 1000.0 / pr.rate_hz
        for k in range(units.shape[0]):
            self.plot.plot(t, units[k, dac], pen=pg.mkPen(series[k % len(series)], width=1))
        unit = pr.dac_units[dac] if dac < len(pr.dac_units) else ""
        self.plot.getPlotItem().setLabel("left", f"Output {dac} ({unit})")
        self.status.setText(f"{pr.sweeps} sweeps of {pr.sweep_frames * 1000.0 / pr.rate_hz:g} ms, every "
                            f"{pr.start_to_start_s:g} s.")

    def _free_name(self) -> str:
        stem, k = self.source.stem + "_edited", 1
        while (self.source.parent / f"{stem}.pro").exists():
            k += 1
            stem = f"{self.source.stem}_edited{k}"
        return stem

    def save(self) -> None:
        name = self.name.text().strip()
        if not name:
            self.status.setText("Name the new protocol.")
            return
        try:
            pr = protocol_edit.save(self._read(), self.source.parent / name)
        except (ValueError, FileExistsError, OSError) as exc:
            self.status.setText(f"Not saved: {exc}.")
            return
        self.saved = pr.path
        self.accept()
