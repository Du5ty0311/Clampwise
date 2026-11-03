"""A grid mapping protocol played from Clampwise with Next step with each sweep: a grid of cells sized in um centred
on the picture, through the calibration kept; the camera catching each light pulse. Says, sweep by sweep, which
cell should have been lit and where the light landed (its distance from that cell's centre). Moves hardware: only
with someone at the rig, the LED's driver following the Digidata (off manual), Clampwise closed.

    python -I tools/rigtest/grid_sweeps_check.py --at-the-rig [--protocol ChroME_grid_map_CC] [--grid 3,3,34,24]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main() -> int:
    try:
        return _main()
    finally:
        _common.leave_safely()


def _main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--protocol", default="ChroME_grid_map_CC")
    p.add_argument("--protocols", default="", help="the folder the protocols are in (default: Clampex's)")
    p.add_argument("--objective", type=float, default=20.0)
    p.add_argument("--grid", default="3,3,34,24", help="rows, columns, each cell's width and height (um)")
    args = p.parse_args()
    _common.require_presence(args)
    import numpy as np
    from pyqtgraph.Qt import QtCore
    from scipy import ndimage

    from clampwise.app.screens.camera_panel import flash_rise
    from clampwise.rig import patterns

    rows, cols, cw, ch = (float(x) for x in args.grid.split(","))
    rows, cols = int(rows), int(cols)
    window, app = _common.open_window(args)
    window.go("live")
    live = window.screens["live"]
    cam, light, acq = live.camera, live.patterned, live.recorder
    cam.objective.setValue(cam.objectives.index(args.objective))
    live.use_light()
    print("finding the patterned light (about 25 s)...", flush=True)
    if not _common.pump(30.0, until=lambda: cam.last is not None):
        print("no camera picture")
        return 1
    _common.pump(60.0, until=lambda: not light._opening)
    if not light.polygon.connected:
        print(f"the patterned light was not taken: {light.polygon.status}")
        return 1
    real = QtCore.QSettings("Clampwise", "Clampwise")             # the calibrations kept (read only)
    light.calibrations = patterns.Calibrations.from_dict(json.loads(str(real.value("polygon/calibration", "{}")
                                                                        or "{}")))
    light.calibration = light.calibrations.at(light._objective())
    if not light.calibration.ready:
        print(f"not calibrated at {light._objective()}")
        return 1
    um = light.calibration.um_per_px
    light.source.setCurrentText("Grid")
    light.grid_on_picture.setChecked(True)
    light.rows.setValue(rows)
    light.cols.setValue(cols)
    light.cell_w.setValue(cw)
    light.cell_h.setValue(ch)
    light.make_steps()
    h, w = cam.last.rgb.shape[:2]
    centres = [(w / 2 + (j - (cols - 1) / 2) * cw / um, h / 2 + (i - (rows - 1) / 2) * ch / um)
               for i in range(rows) for j in range(cols)]
    light.with_sweeps.setChecked(True)
    light.off_after.setChecked(False)
    cam.catch.setChecked(True)
    _common.pump(2.0)
    print(f"{rows} x {cols} cells of {cw:g} x {ch:g} um centred on the picture; {light.pattern_note.text()}",
          flush=True)
    landed: list[tuple[float, float, float, int]] = []

    def flashed(f, b, rise, when) -> None:
        k = len(landed) if not landed or when - landed[-1][3] > 0.2 else len(landed) - 1
        d = ndimage.gaussian_filter(flash_rise(f.rgb[:, :, 0], b.rgb[:, :, 0]), 1.0)
        cx, cy = centres[k % len(centres)]
        yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
        near = np.hypot(xx * 8 + 4 - cx, yy * 8 + 4 - cy) < 1.2 * max(cw, ch) / um   # the cell, not its copy
        far = np.where(near, -np.inf, d)
        top = np.unravel_index(int(np.argmax(np.where(near, d, -np.inf))), d.shape)
        wgt = np.clip(d - 0.5 * d[top], 0, None) * near
        x, y = float((wgt * (xx * 8 + 4)).sum() / max(wgt.sum(), 1e-9)), float((wgt * (yy * 8 + 4)).sum() /
                                                                               max(wgt.sum(), 1e-9))
        entry = (x, y, float(d[top]) / max(float(far.max()), 1e-3), when)
        if k < len(landed):
            if rise > landed[k][2]:
                landed[k] = entry
        else:
            landed.append(entry)

    cam._thread.flash.connect(flashed)
    live.use_acquisition()
    acq._protocols_dir = args.protocols or acq._protocols_dir
    acq.fill_protocols()
    acq.protocol.setCurrentIndex(acq.protocol.findText(args.protocol))
    if acq.protocol.currentText() != args.protocol:
        print(f"no protocol {args.protocol} in {acq._protocols_dir}")
        return 1
    n = rows * cols
    print(f"playing {args.protocol}, stopped after {n} sweeps...", flush=True)
    acq.play()

    def done() -> bool:
        st = acq.rec.status if acq.rec is not None else None
        if st is not None and st.step_frames and st.frames >= (n - 1) * st.step_frames + st.sweep_frames + \
                0.2 * st.rate_hz:
            acq.stop()
        return st is not None and not st.busy

    _common.pump(600.0, until=done)
    _common.pump(1.0)
    print(f"run: {acq.rec.status.state}; {len(landed)} pulses caught for {n} sweeps")
    errs = []
    for k, (x, y, contrast, _) in enumerate(landed[:n]):
        cx, cy = centres[k % len(centres)]
        e = float(np.hypot(x - cx, y - cy) * um)
        errs.append(e)
        print(f"  sweep {k + 1}: cell {k % n + 1} at ({cx:.0f}, {cy:.0f}) px, the light at ({x:.0f}, {y:.0f}): "
              f"{e:.1f} um off; brightest there {contrast:.1f} times anywhere else")
    if errs:
        print(f"each sweep's light from its cell's centre: median {np.median(errs):.1f} um, worst {max(errs):.1f} um "
              f"(cells {cw:g} x {ch:g} um)")
    cam.catch.setChecked(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
