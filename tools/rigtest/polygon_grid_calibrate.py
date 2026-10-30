"""The patterned light calibrated from a grid of spots, each found by its sharp edge: for a slice or slide that sends
much of the light back as an even glow, and a blurred copy of each spot (a reflection), where Calibrate...'s spots
are lost. The exposure is held long enough to see a spot (every mirror on then saturates). Each spot is found
where the edges are sharpest (the glow and the blurred copy have none), the map fitted to all of them with strays
left out, and each checked by the map fitted to the others. Kept for the objective (and saved to a file, as Save
the one in use does) only with ``--keep-if-good`` and every check within the target. Moves hardware: only with
someone at the rig, the LED on at its driver, Clampwise closed.

    python -I tools/rigtest/polygon_grid_calibrate.py --at-the-rig --objective 20 --exposure 25 [--keep-if-good]
"""

from __future__ import annotations

import json
import sys
import time
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
    p.add_argument("--objective", type=float, required=True, help="the objective in use (its magnification)")
    p.add_argument("--exposure", type=float, default=25.0, help="the camera's exposure, ms, held")
    p.add_argument("--diameter", type=float, default=0.04, help="each spot's diameter, of the field's width")
    p.add_argument("--us", type=int, default=8, help="spots across the field")
    p.add_argument("--vs", type=int, default=10, help="spots down the field")
    p.add_argument("--keep-if-good", action="store_true", help="kept and saved when every check is within target")
    args = p.parse_args()
    _common.require_presence(args)
    import numpy as np
    from pyqtgraph.Qt import QtCore, QtGui
    from scipy import ndimage

    from clampwise.app.screens.polygon_panel import saved_folder
    from clampwise.rig import patterns

    window, app = _common.open_window(args)
    window.go("live")
    live = window.screens["live"]
    cam, light = live.camera, live.patterned
    if args.objective not in cam.objectives:
        cam.set_objectives(", ".join(f"{m:g}" for m in sorted(set(cam.objectives) | {args.objective})))
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
    real = QtCore.QSettings("Clampwise", "Clampwise")
    um = {float(k): float(v) for k, v in json.loads(str(real.value("camera/calibrated", "{}") or "{}")).items()}
    key = patterns.objective_key(args.objective, float(real.value("camera/adapter", 1.0) or 1.0))
    um_px = cam.scale()[0] or um.get(args.objective, 0.0)
    if not um_px and um:                              # the others follow by their magnifications
        m, u = next(iter(um.items()))
        um_px = u * m / args.objective
    cam.auto.setChecked(False)
    cam.exposure.setValue(args.exposure)
    cam._exposure()
    settle = max(0.5, 4 * args.exposure / 1000.0)

    def picture(n: int = 2) -> np.ndarray:
        """The mean of the next ``n`` pictures, 2 x 2 binned (float)."""
        got = []
        for _ in range(n):
            seen = cam._shown
            _common.pump(2.0, until=lambda: cam._shown > seen)
            g = cam.last.rgb[:, :, 0].astype(np.float32)
            h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
            got.append(g[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))
        return np.mean(got, axis=0)

    def show(mask) -> None:
        img = patterns.image(mask, light.calibration.on_value) if mask is not None else None
        if img is None:
            light.off()
        else:
            light._async(lambda: light.polygon.send(img))
        light.wait()
        _common.pump(settle)

    show(None)
    off = picture(4)
    h, w = off.shape
    r_px = args.diameter / 2 * 3360 / 2               # a spot's radius on the binned picture at 20x (about)
    r_px = r_px * 20.0 / args.objective
    print(f"exposure {args.exposure:g} ms held; nothing lit: grey {off.mean():.1f}; {args.us} x {args.vs} spots, "
          f"{args.diameter:g} of the field across", flush=True)

    def find(d: np.ndarray):
        """The sharpest-edged bright patch's centre (x, y, full-size pixels) and how far it stands out."""
        s = ndimage.gaussian_filter(d, 1.5)
        e = ndimage.sobel(s, 0) ** 2 + ndimage.sobel(s, 1) ** 2
        ring = ndimage.gaussian_filter(e, max(r_px * 0.8, 4))       # the edge ring's energy, peaked at its centre
        y, x = np.unravel_index(int(np.argmax(ring)), ring.shape)
        score = float(ring[y, x] / max(np.median(ring), 1e-6))
        hp = d - ndimage.gaussian_filter(d, r_px * 2.5)               # the glow and the blurred copy taken away
        y0, y1 = max(0, int(y - 1.6 * r_px)), min(h, int(y + 1.6 * r_px) + 1)
        x0, x1 = max(0, int(x - 1.6 * r_px)), min(w, int(x + 1.6 * r_px) + 1)
        win = hp[y0:y1, x0:x1]
        wt = np.clip(win - 0.5 * win.max(), 0, None)
        if wt.sum() <= 0:
            return None, score
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx, cy = float((wt * xx).sum() / wt.sum()), float((wt * yy).sum() / wt.sum())
        edge = min(cx, cy, w - 1 - cx, h - 1 - cy)
        if edge < 1.2 * r_px:                         # cut by the picture's edge: its centre pulled inwards
            return None, score
        return (2 * cx + 0.5, 2 * cy + 0.5), score

    found, scores = [], []
    args.out.mkdir(parents=True, exist_ok=True)
    for v in np.linspace(0.05, 0.95, args.vs):
        for u in np.linspace(0.08, 0.92, args.us):
            show(patterns.spot(float(u), float(v), args.diameter)[0].mirrors())
            d = picture() - off
            xy, score = find(d)
            scores.append(score)
            if xy is not None:
                found.append((xy, (float(u), float(v)), score))
            if abs(u - 0.5) < 0.07 and abs(v - 0.5) < 0.06:
                s = np.clip(d / max(np.percentile(d, 99.9), 1) * 255, 0, 255).astype(np.uint8)
                s = np.ascontiguousarray(s)
                QtGui.QImage(s.data, s.shape[1], s.shape[0], s.shape[1], QtGui.QImage.Format_Grayscale8).copy() \
                    .save(str(args.out / f"grid-spot-{u:.2f}-{v:.2f}.png"))
        print(f"  row v {v:.2f}: {len(found)} found so far", flush=True)
    show(None)
    if not found:
        print("no spot found")
        return 1
    floor = float(np.median(scores))                  # most of the grid is off the picture: what nothing scores
    good = [f for f in found if f[2] > max(4.0 * floor, 8.0)]
    print(f"{len(found)} candidate spots; edge scores: median of the grid {floor:.1f}; kept as spots "
          f"{len(good)} (over {max(4.0 * floor, 8.0):.1f})", flush=True)
    if len(good) < 6:
        print("too few spots to fit and check")
        return 1
    xy = np.array([g[0] for g in good])
    uv = np.array([g[1] for g in good])
    to_field, residual, held, kept = patterns.fit_without_strays(xy, uv)
    err = np.hypot(*(held - xy[kept]).T)
    cal = patterns.Calibration(light.calibration.on_value, True, to_field, tuple(cam.last.rgb.shape[1::-1]),
                               residual, key, np.column_stack([xy[kept], held]).tolist(), um_px,
                               time.strftime("%Y-%m-%d %H:%M"))
    frame = patterns.frame_of(cal)
    print(f"fitted to {int(kept.sum())} spots ({int((~kept).sum())} left out as strays), residual {residual:.1f} px")
    pw, ph = cam.last.rgb.shape[1], cam.last.rgb.shape[0]
    xs, ys = xy[kept, 0], xy[kept, 1]
    print(f"the spots span x {xs.min():.0f}-{xs.max():.0f} of {pw}, y {ys.min():.0f}-{ys.max():.0f} of {ph} px: "
          + ", ".join(f"({x:.0f}, {y:.0f})" for x, y in xy[kept]))
    print(f"field on the picture: centre ({frame[0]:.0f}, {frame[1]:.0f}) px, turned {frame[2]:.1f} deg, "
          f"{frame[3]:.0f} x {frame[4]:.0f} px")
    e_um = err * um_px if um_px else err
    unit = "um" if um_px else "px"
    print(f"check, each spot by the map fitted to the others: worst {e_um.max():.1f} {unit}, mean {e_um.mean():.1f} "
          f"{unit}, 90% within {np.percentile(e_um, 90):.1f} {unit}")
    ok = bool(um_px) and e_um.max() <= patterns.CHECK_TARGET_UM
    if args.keep_if_good and ok:
        cals = patterns.Calibrations.from_dict(json.loads(str(real.value("polygon/calibration", "{}") or "{}")))
        cals.keep(cal)
        real.setValue("polygon/calibration", json.dumps(cals.to_dict()))
        real.sync()
        folder = saved_folder()
        folder.mkdir(parents=True, exist_ok=True)
        stem = " ".join(f"{key} {cal.measured}".replace(",", "").replace(":", "").split())
        path, k = folder / f"{stem}.json", 2
        while path.exists():
            path, k = folder / f"{stem} ({k}).json", k + 1
        path.write_text(json.dumps(cal.to_dict(), indent=1), encoding="utf-8")
        print(f"KEPT for {key} and saved: {path}")
    else:
        print("not kept" + ("" if ok else f" (the check is over the {patterns.CHECK_TARGET_UM:g} um target)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
