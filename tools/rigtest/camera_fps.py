"""The camera's pictures a second, taken and drawn, at each exposure and binning: what limits the frame rate (the
exposure, the camera, or the drawing). Reads the camera only (moves nothing).

    python -I tools/rigtest/camera_fps.py [--exposures 5 10 20 50 100] [--binning 1 2 4] [--seconds 3]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main() -> int:
    p = _common.parser(__doc__)
    p.add_argument("--exposures", type=float, nargs="+", default=[5.0, 10.0, 20.0, 50.0, 100.0])
    p.add_argument("--binning", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seconds", type=float, default=3.0)
    args = p.parse_args()
    proc = _common.Process()
    window, app = _common.open_window(args)
    window.go("live")
    live = window.screens["live"]
    live.use_camera()
    cam = live.camera
    if not _common.pump(20.0, until=lambda: cam.last is not None):
        print(f"no camera picture: {cam.status.text()}")
        return 1
    rows = []
    for n in args.binning:
        cam.binning.setCurrentIndex(cam.binning.findData(n))
        for ms in args.exposures:
            cam.exposure.setValue(ms)
            cam._exposure()                            # as typed: editingFinished applies it
            _common.pump(max(1.0, 3 * ms / 1000))
            t = cam._thread
            taken0, shown0, cpu0, t0 = t.taken, cam._shown, proc.cpu_s(), time.perf_counter()
            _common.pump(args.seconds)
            dt = time.perf_counter() - t0
            row = {"binning": n, "exposure_ms": ms, "taken_fps": round((t.taken - taken0) / dt, 1),
                   "drawn_fps": round((cam._shown - shown0) / dt, 1), "exposure_limit_fps": round(1000 / ms, 1),
                   "cpu_pct": round((proc.cpu_s() - cpu0) / dt * 100, 1),
                   "picture": list(cam.last.rgb.shape[1::-1])}
            rows.append(row)
            print(f"{n} x {n}, {ms:5g} ms: taken {row['taken_fps']:5.1f} fps (exposure allows "
                  f"{row['exposure_limit_fps']:5.1f}), drawn {row['drawn_fps']:5.1f}, cpu {row['cpu_pct']:5.1f}%, "
                  f"{row['picture'][0]} x {row['picture'][1]}")
    cam.binning.setCurrentIndex(cam.binning.findData(1))
    _common.leave_safely()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"camera-fps-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"numbers: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
