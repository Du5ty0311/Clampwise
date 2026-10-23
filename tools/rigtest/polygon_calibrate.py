"""The patterned light calibrated at one objective, as Calibrate... does it in Live: which byte lights a mirror, then
where the mirrors' field lies on the camera's picture, then a check grid. Moves hardware: only with someone at the
rig. With ``--keep`` the calibration is kept for that objective (Clampwise uses it from then on).

Needs: a mirror (the pattern seen reflected) or a fluorescent slide in focus, the LED on (at its driver, or a protocol
pulsing it: found out unless ``--led`` says), the camera's and the patterned light's own programs closed, and the
objective named the one in use. Ends with nothing lit and both devices let go, however it ends.

    python -I tools/rigtest/polygon_calibrate.py --at-the-rig --objective 20 [--keep] [--led auto|steady|pulsed]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main() -> int:
    try:
        return _main()
    finally:                                          # nothing lit and both let go, before Python starts shutting down
        _common.leave_safely()


def _main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--objective", type=float, required=True, help="the objective in use (its magnification)")
    p.add_argument("--led", choices=("auto", "steady", "pulsed"), default="auto", help="how the LED is on")
    p.add_argument("--exposure", type=float, default=0.0, help="the camera's exposure, ms (0: as it is)")
    p.add_argument("--auto-exposure", action="store_true", help="auto exposure with every mirror on, first")
    p.add_argument("--lowest-gain", action="store_true", help="the camera's gain at its lowest (a mirror)")
    p.add_argument("--focus", type=float, default=0.0, help="seconds of a checkerboard to focus on, no calibration")
    p.add_argument("--map", action="store_true", help="where the light of each of 4 x 4 squares lands; no calibration")
    p.add_argument("--format", action="store_true", help="how the device reads an image (bytes or packed bits)")
    p.add_argument("--spots-only", action="store_true", help="show two calibration spots and look; no calibration")
    p.add_argument("--minutes", type=float, default=6.0, help="the longest the calibration may take")
    args = p.parse_args()
    _common.require_presence(args)
    import numpy as np
    from pyqtgraph.Qt import QtGui

    window, app = _common.open_window(args)
    window.go("live")
    live = window.screens["live"]
    cam, light = live.camera, live.patterned
    if args.objective not in cam.objectives:
        cam.set_objectives(", ".join(f"{m:g}" for m in sorted(set(cam.objectives) | {args.objective})))
    cam.objective.setValue(cam.objectives.index(args.objective))
    live.use_light()                                  # takes the camera too
    print("finding the patterned light (about 25 s)...")
    if not _common.pump(30.0, until=lambda: cam.last is not None):
        print(f"no camera picture: {cam.status.text() or live.views['camera'].failure}")
        _common.pump(60.0, until=lambda: not light._opening)      # found first, so it can be let go properly
        _common.leave_safely()
        return 1
    if not _common.pump(60.0, until=lambda: light.polygon.connected or not (light._opening or light.busy())):
        print("the patterned light was not found in 60 s")
        return 1
    if not light.polygon.connected:
        print(f"the patterned light was not taken: {light.polygon.status}")
        return 1
    print(f"objective {cam.magnification:g}x ({light._objective()})")

    def look(name: str) -> None:
        """The camera's picture now, in numbers and saved as taken (a PNG in --out)."""
        _common.pump(1.0)
        g = np.ascontiguousarray(cam.last.rgb[:, :, 0])
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{time.strftime('%H%M%S')}-{args.objective:g}x-{name}.png"
        QtGui.QImage(g.data, g.shape[1], g.shape[0], g.shape[1], QtGui.QImage.Format_Grayscale8).copy().save(str(path))
        print(f"{name}: mean grey {g.mean():.1f}, brightest 0.5% {np.percentile(g, 99.5):.0f}, darkest 0.5% "
              f"{np.percentile(g, 0.5):.0f} of 255, exposure {cam.exposure.value():g} ms, gain "
              f"{cam.gain.value():g}: {path}", flush=True)

    light.source.setCurrentText("Full field")
    light.make_steps()
    light.project()                                   # all mirrors on: the brightest the picture gets
    light.wait()
    if args.lowest_gain:
        cam.gain.setValue(cam.gain.minimum())         # a mirror sends back far more than a cell: no gain needed
        cam._exposure()
    if args.exposure > 0:
        cam.exposure.setValue(args.exposure)
        cam._exposure()
    elif args.auto_exposure:                          # the exposure follows the light: every mirror on, settled
        cam.auto.setChecked(True)
        steady, last_ms, end = 0, -1.0, time.monotonic() + 20.0
        while steady < 4 and time.monotonic() < end:  # unchanged for 2 s (it corrects twice a second)
            _common.pump(0.5)
            steady = steady + 1 if cam.exposure.value() == last_ms else 0
            last_ms = cam.exposure.value()
        print(f"auto exposure settled at {cam.exposure.value():g} ms", flush=True)
        cam.auto.setChecked(False)                    # held there: the dark pictures must not lengthen it
    look("all-on")
    light.off()
    light.wait()
    look("all-off")
    if args.format:                                   # bytes a mirror, or bits packed 8 a byte? and their order
        from clampwise.rig import patterns

        R, C = patterns.ROWS, patterns.COLUMNS

        def frame_of(buf: np.ndarray) -> np.ndarray:
            light._async(lambda: light.polygon.send(buf))
            light.wait()
            _common.pump(max(0.5, 4 * cam.exposure.value() / 1000.0))
            return np.ascontiguousarray(cam.last.rgb[:, :, 0]).astype(np.float32)

        def packed(mask: np.ndarray, order: str = "big") -> np.ndarray:
            """``mask`` (R x C mirrors) as bits, 8 mirrors a byte, at the start of an R x C byte buffer."""
            bits = np.packbits(mask.astype(np.uint8), axis=1, bitorder=order).ravel()
            buf = np.zeros(R * C, np.uint8)
            buf[:bits.size] = bits
            return buf.reshape(R, C)

        dark = frame_of(np.zeros((R, C), np.uint8))
        full = (frame_of(np.full((R, C), 255, np.uint8)) - dark).clip(0)
        total = max(float(full.sum()), 1.0)
        h, w = full.shape
        ys, xs = np.mgrid[0:h, 0:w]

        def measure(name: str, buf: np.ndarray) -> tuple[float, float, float]:
            d = (frame_of(buf) - dark).clip(0)
            n = float(d.sum())
            x, y = float((d * xs).sum() / max(n, 1)) / w, float((d * ys).sum() / max(n, 1)) / h
            print(f"  {name:44s} {n / total:6.1%} of the light, centre x {x:.3f} y {y:.3f}", flush=True)
            return n / total, x, y

        print(f"all on: {full.mean():.1f} grey; its centre x {(full * xs).sum() / total / w:.3f} "
              f"y {(full * ys).sum() / total / h:.3f}", flush=True)
        eighth = np.zeros((R, C), np.uint8)
        eighth[: R // 8] = 255
        measure("first eighth of the bytes sent", eighth)
        second = np.zeros((R, C), np.uint8)
        second[R // 8: R // 4] = 255
        measure("second eighth of the bytes sent", second)
        left = np.zeros((R, C), bool)
        left[:, : C // 2] = True
        top = np.zeros((R, C), bool)
        top[: R // 2] = True
        measure("packed bits: left half of the mirrors", packed(left))
        measure("packed bits: top half of the mirrors", packed(top))
        line = np.zeros((R, C), bool)
        line[R // 4: 3 * R // 4, 456:464] = True                   # a whole byte wide: 8 mirrors
        _, x8, _ = measure("packed bits: a line 8 mirrors wide", packed(line))
        half = np.zeros((R, C), bool)
        half[R // 4: 3 * R // 4, 456:460] = True                   # its first 4 mirrors only
        _, xb, _ = measure("  its first 4 mirrors, high bit first", packed(half, "big"))
        _, xl, _ = measure("  its first 4 mirrors, low bit first", packed(half, "little"))
        print(f"line centre x: 8 wide {x8:.4f}, first 4 high-bit-first {xb:.4f}, low-bit-first {xl:.4f}", flush=True)
        light.off()
        light.wait()
        return 0
    if args.map:                                      # each square of a 4 x 4 grid: how much light, and where
        from clampwise.rig import patterns

        def frame_after(mask) -> np.ndarray:
            light._async(lambda: light.polygon.send(patterns.image(mask, light.calibration.on_value)))
            light.wait()
            _common.pump(max(0.5, 4 * cam.exposure.value() / 1000.0))
            return np.ascontiguousarray(cam.last.rgb[:, :, 0]).astype(np.float32)

        none = np.zeros((patterns.ROWS, patterns.COLUMNS), bool)
        dark = frame_after(none)
        full = frame_after(~none) - dark
        total = max(float(full.clip(0).sum()), 1.0)
        h, w = full.shape
        ys, xs = np.mgrid[0:h, 0:w]

        def centre(d: np.ndarray) -> tuple[float, float, float]:
            d = d.clip(0)
            n = float(d.sum())
            return n / total, float((d * xs).sum() / max(n, 1)) / w, float((d * ys).sum() / max(n, 1)) / h

        print(f"all on: {full.mean():.1f} grey over the dark {dark.mean():.1f}; centre of the whole field's light at "
              f"x {centre(full)[1]:.2f}, y {centre(full)[2]:.2f} of the picture", flush=True)
        print("square (row, col) of the field: share of all the light, centre x, y on the picture (0-1)", flush=True)
        rows = np.arange(patterns.ROWS) * 4 // patterns.ROWS
        cols = np.arange(patterns.COLUMNS) * 4 // patterns.COLUMNS
        found = []
        for r in range(4):
            for c in range(4):
                mask = (rows[:, None] == r) & (cols[None, :] == c)
                share, x, y = centre(frame_after(mask) - dark)
                found.append((r, c, share, x, y))
                print(f"  ({r}, {c}): {share:5.1%}  x {x:.2f}  y {y:.2f}", flush=True)
        xs_, ys_ = np.array([f[3] for f in found]), np.array([f[4] for f in found])
        cs, rs = np.array([f[1] for f in found], float), np.array([f[0] for f in found], float)
        print(f"centre x follows the column: r = {np.corrcoef(cs, xs_)[0, 1]:+.2f} (x spread {np.ptp(xs_):.2f}); "
              f"y follows the row: r = {np.corrcoef(rs, ys_)[0, 1]:+.2f} (y spread {np.ptp(ys_):.2f})", flush=True)
        light.off()
        light.wait()
        return 0
    if args.focus:                                    # a checkerboard flipping with its inverse, and how sharp
        off = np.ascontiguousarray(cam.last.rgb[:, :, 0]).astype(np.float32)      # all mirrors off: stray light
        light.source.setCurrentText("Grid")
        light.rows.setValue(4)
        light.cols.setValue(4)
        light.grid_on_picture.setChecked(False)
        light.make_steps()

        def board(first: bool) -> np.ndarray:
            """The checkerboard (or its inverse) on the mirrors, and the camera's picture of it."""
            light.list.clearSelection()
            for k in range(16):
                light.list.item(k).setSelected(((k // 4 + k % 4) % 2 == 0) == first)
            light.project()
            light.wait()
            _common.pump(max(0.35, 3.5 * cam.exposure.value() / 1000.0))
            return np.ascontiguousarray(cam.last.rgb[:, :, 0]).astype(np.float32)

        print("checkerboard flipping about once a second: in focus, its squares swap; out of focus, an even "
              "flicker. Sharpness: 0 no pattern, near 1 crisp squares.", flush=True)
        end, k = time.monotonic() + args.focus, 0
        while time.monotonic() < end:
            a, b = board(True), board(False)
            lit = np.clip(a + b - 2 * off, 1.0, None)
            sharp = float(np.abs(a - b).mean() / lit.mean())
            k += 1
            if k % 2 == 1:
                print(f"  sharpness {sharp:.2f} (lit {lit.mean() / 2:.0f} grey over the stray {off.mean():.0f})",
                      flush=True)
        look("checkerboard")
        light.off()
        light.wait()
        _common.leave_safely()
        return 0
    if args.spots_only:                               # what one calibration spot looks like, and whether it is found
        from clampwise.rig import patterns

        off = np.ascontiguousarray(cam.last.rgb[:, :, 0]).astype(np.float32)
        light.source.setCurrentText("Spot")
        for u, v, d in ((0.5, 0.5, patterns.CALIBRATION_DIAMETER), (0.25, 0.25, patterns.CALIBRATION_DIAMETER),
                        (0.5, 0.5, 0.25)):
            light.spot_u.setValue(u * 100)
            light.spot_v.setValue(v * 100)
            light.spot_d.setValue(d * 100)
            light.make_steps()
            light.project()
            light.wait()
            look(f"spot-{u:g}-{v:g}-d{d:g}")
            g = cam.last.rgb[:, :, 0]
            print(f"  brightest pixel {g.max()}, pixels over 50: {(g > 50).sum()}; found at",
                  patterns.find_spot(g.astype(np.float32), off), flush=True)
        light.source.setCurrentText("Grid")                # the field's left half: a sharp edge when in focus
        light.rows.setValue(1)
        light.cols.setValue(2)
        light.grid_on_picture.setChecked(False)
        light.make_steps()
        light.list.setCurrentRow(0)
        light.project()
        light.wait()
        look("left-half")
        g = cam.last.rgb[:, :, 0]
        print(f"  brightest pixel {g.max()}, pixels over 50: {(g > 50).sum()} of {g.size}", flush=True)
        light.off()
        light.wait()
        _common.leave_safely()
        return 0
    light.led = args.led
    light.calibrate()
    last, end = "", time.monotonic() + args.minutes * 60
    while light._cal is not None and time.monotonic() < end:        # each step said as it comes
        _common.pump(0.5, until=lambda: light._cal is None)
        text = light.cal_status.text().split(" (")[0]
        if text != last:
            last = text
            print(f"  {text}", flush=True)
    if light._cal is not None:
        light._end_calibration("Stopped: it took longer than allowed.")
        light.wait()
    print(light.cal_status.text())
    if args.auto_exposure:
        cam.auto.setChecked(True)                     # following the light again, as it was
    status, words = light.calibration_words()
    print(f"outcome: {status}: {words}")
    print("kept for this objective" if args.keep and light.calibration.ready else "not kept (no --keep, or not "
                                                                                    "calibrated)")
    _common.leave_safely()
    return 0 if light.calibration.ready else 1


if __name__ == "__main__":
    sys.exit(main())
