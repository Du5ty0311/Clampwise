"""Where a region lands: a round region drawn at a point of the camera's picture is projected, and where the light
falls is found on the picture and compared with where it was drawn (in pixels and um). Moves hardware: only with
someone at the rig.

Needs the patterned light calibrated at the objective in use (Test and calibrate, in Clampwise), the light on (or
the protocol playing, ``--pulsed``), a fluorescent slide or slice in focus. Reads the saved calibration (it opens
Clampwise with its settings); ends with nothing lit and both devices let go.

    python -I tools/rigtest/polygon_roi.py --at-the-rig [--x 972 --y 736] [--um 15] [--pulsed]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402
from polygon_light_test import watch  # noqa: E402


def main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--x", type=float, default=None, help="camera px across (default: the picture's middle)")
    p.add_argument("--y", type=float, default=None, help="camera px down")
    p.add_argument("--um", type=float, default=15.0, help="the region's diameter")
    p.add_argument("--pulsed", action="store_true")
    p.add_argument("--seconds", type=float, default=3.0)
    args = p.parse_args()
    _common.require_presence(args)
    args.keep = True                                   # the calibration is read from Clampwise's settings
    import numpy as np

    from clampwise.rig import patterns

    window, app = _common.open_window(args)
    window.go("live")
    live = window.screens["live"]
    live.use_camera()
    if not _common.pump(20.0, until=lambda: live.camera.last is not None):
        print(f"no camera picture: {live.camera.status.text()}")
        return 1
    panel = live.patterned
    panel.load()
    panel._use_objective()
    if not panel.calibration.ready:
        print(f"not calibrated here: {panel._cal_text()}")
        return 1
    live.use_light()
    print("finding the patterned light (about 25 s)...")
    if not _common.pump(60.0, until=lambda: panel.polygon.connected or not (panel._opening or panel.busy())):
        return 1
    if not panel.polygon.connected:
        print(f"the patterned light was not taken: {panel.polygon.status}")
        return 1
    w, h = panel._frame_size()
    x = args.x if args.x is not None else w / 2
    y = args.y if args.y is not None else h / 2
    um = panel._um_per_px()
    seconds = max(args.seconds, 10.0) if args.pulsed else args.seconds
    panel.off()
    panel.wait()
    _common.pump(0.5)
    dark = watch(live.camera, seconds)["frame"].rgb[:, :, 0].astype(np.float32)
    panel.source.setCurrentText("Drawn on the picture")
    panel.clear_shapes()
    panel.roi_name.setText("Test region")
    panel.roi_um.setValue(args.um)
    if panel.place_at(x, y) is None:
        print("the camera's scale is not known at this objective: calibrate its length first")
        return 1
    panel.make_steps()
    panel.list.setCurrentRow(0)
    panel.project()
    panel.wait()
    _common.pump(0.5)
    lit = watch(live.camera, seconds)
    shot = _common.screenshot(live.camera, "roi", args.out)
    found = patterns.find_spot(lit["frame"].rgb[:, :, 0].astype(np.float32), dark)
    print(f"{panel.shown_label.text()}\nscreenshot: {shot}")
    if found is None:
        print("no lit region found on the picture: the light, its path, the focus?")
        code = 1
    else:
        err = float(np.hypot(found[0] - x, found[1] - y))
        print(f"drawn at ({x:.0f}, {y:.0f}) px, lit at ({found[0]:.1f}, {found[1]:.1f}) px: {err:.1f} px off"
              + (f" = {err * um:.1f} um (target under {patterns.CHECK_TARGET_UM:g} um)" if um else ""))
        code = 0
    panel.clear_shapes()
    _common.leave_safely()
    return code


if __name__ == "__main__":
    sys.exit(main())
