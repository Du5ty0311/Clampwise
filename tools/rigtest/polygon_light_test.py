"""Which byte lights a mirror: the patterned light shows nothing, then every mirror at 255, then every mirror at 0,
and the camera's picture says which one lit the slide. Moves hardware: only with someone at the rig.

Needs: the light on (its LED in a continuous mode, or the protocol playing with ``--pulsed``), its path open, a
fluorescent slide or slice in focus, the patterned light's own program closed. Ends with nothing lit and both
devices let go, however it ends.

    python -I tools/rigtest/polygon_light_test.py --at-the-rig [--pulsed] [--seconds 3]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def watch(cam, seconds: float) -> dict:
    """The camera's pictures for ``seconds``: their mean grey levels' median, brightest and darkest (pulsed light:
    the brightest is the one a pulse lit)."""
    seen, means, last = None, [], None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _common.pump(0.02)
        if cam.last is not None and cam.last is not seen:
            seen = cam.last
            means.append(float(seen.rgb[:, :, 0].mean()))
            last = seen
    means.sort()
    return {"median": means[len(means) // 2] if means else float("nan"), "max": means[-1] if means else float("nan"),
            "min": means[0] if means else float("nan"), "frames": len(means), "frame": last}


def main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--pulsed", action="store_true", help="the light only in the protocol's pulses (play it now)")
    p.add_argument("--seconds", type=float, default=3.0, help="how long each picture is watched (pulsed: >= 10)")
    args = p.parse_args()
    _common.require_presence(args)
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
    live.use_light()
    print("finding the patterned light (about 25 s)...")
    if not _common.pump(60.0, until=lambda: panel.polygon.connected or not (panel._opening or panel.busy())):
        print("the patterned light was not found in 60 s")
        return 1
    if not panel.polygon.connected:
        print(f"the patterned light was not taken: {panel.polygon.status}")
        return 1
    seconds = max(args.seconds, 10.0) if args.pulsed else args.seconds
    shape = (patterns.ROWS, patterns.COLUMNS)
    results = {}
    for name, value in (("all 0", 0), ("all 255", 255)):
        img = np.full(shape, value, np.uint8)
        panel._async(lambda img=img: panel.polygon.send(img))
        panel.wait()
        _common.pump(0.5)
        results[name] = watch(live.camera, seconds)
        shot = _common.screenshot(live.camera, f"light-test-{value}", args.out)
        r = results[name]
        print(f"every mirror at {value}: mean grey {r['median']:.1f} (brightest {r['max']:.1f}, darkest "
              f"{r['min']:.1f}, {r['frames']} pictures); {shot}")
    key = "max" if args.pulsed else "median"
    a, b = results["all 0"][key], results["all 255"][key]
    if abs(a - b) < 2.0:
        print(f"no difference ({a:.1f} vs {b:.1f} grey): is the light on, its path open, a fluorescent sample in "
              "focus" + (", the protocol playing?" if args.pulsed else "?"))
        on = None
    else:
        on = 255 if b > a else 0
        print(f"a mirror is lit by {on}: the picture is {abs(b - a):.1f} grey levels brighter with every mirror at {on}")
    off_value = 0 if on == 255 else 255 if on == 0 else None
    if off_value is not None:
        panel._async(lambda: panel.polygon.send(np.full(shape, off_value, np.uint8)))
        panel.wait()
        print("left showing nothing")
    _common.leave_safely()
    return 0 if on is not None else 1


if __name__ == "__main__":
    sys.exit(main())
