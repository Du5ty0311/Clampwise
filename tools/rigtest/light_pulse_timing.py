"""A photostimulation protocol played from Clampwise with a region lit and the camera catching light pulses: each
pulse of the LED's command (recorded on an input, IN 8 by default) in the PC's time, against the frames the camera
caught it in. Says how many pulses were caught, how long after the pulse each frame came, and any frame that rose
with no pulse. Moves hardware: only with someone at the rig, the LED's driver following the Digidata, Clampwise
closed.

    python -I tools/rigtest/light_pulse_timing.py --at-the-rig --protocol "...Photostim_375ms,475ms.pro" --objective 20
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
    finally:
        _common.leave_safely()


def _main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--protocol", required=True, help="the protocol file (.pro), or a part of its name")
    p.add_argument("--protocols", default="", help="the folder the protocols are in (default: Clampex's)")
    p.add_argument("--objective", type=float, default=20.0)
    p.add_argument("--led-input", default="8", help="the input that records the LED's command (its number)")
    p.add_argument("--spot", default="0.5,0.6,0.12", help="with no calibration, the spot lit: u, v and diameter "
                                                          "on the field")
    p.add_argument("--region", default="972,736,40", help="the region lit through the calibration: its centre x, "
                                                          "y on the picture (px) and diameter (um); '' for --spot")
    p.add_argument("--sweeps", type=int, default=0, help="stop after this many sweeps (0: the whole protocol)")
    args = p.parse_args()
    _common.require_presence(args)
    import numpy as np
    from pyqtgraph.Qt import QtGui

    window, app = _common.open_window(args)
    window.go("live")
    live = window.screens["live"]
    cam, light, acq = live.camera, live.patterned, live.recorder
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
    import json

    from pyqtgraph.Qt import QtCore

    from clampwise.rig import patterns

    real = QtCore.QSettings("Clampwise", "Clampwise")             # the calibrations kept (read only: not written)
    light.calibrations = patterns.Calibrations.from_dict(json.loads(str(real.value("polygon/calibration", "{}")
                                                                        or "{}")))
    light.calibration = light.calibrations.at(light._objective())
    um_px = light.calibration.um_per_px or cam.scale()[0]
    region = None
    if args.region and light.calibration.ready:      # a circle drawn on the picture, lit through the calibration
        x, y, um = (float(s) for s in args.region.split(","))
        light.source.setCurrentText("Drawn on the picture")
        region = (x, y, um / um_px / 2)
        light.add_shape("ellipse", centre=(x, y), size=(2 * region[2], 2 * region[2]), name="Region")
        what = (f"a {um:g} um circle drawn on the picture at ({x:g}, {y:g}) px, through the calibration of "
                f"{light.calibration.measured}")
    else:
        u, v, d = (float(s) for s in args.spot.split(","))
        light.source.setCurrentText("Spot")
        light.spot_u.setValue(u * 100)
        light.spot_v.setValue(v * 100)
        light.spot_d.setValue(d * 100)
        what = f"a spot at ({u:g}, {v:g}), {d:g} of the field across (no calibration at {light._objective()})"
    light.make_steps()
    light.project()
    light.wait()
    light.off_after.setChecked(False)
    cam.catch.setChecked(True)
    _common.pump(2.0)
    t = cam._thread
    print(f"lit: {what}; camera {cam.exposure.value():.3g} ms, gain {cam.gain.value():.3g}, catching", flush=True)
    live.use_acquisition()
    acq._protocols_dir = args.protocols or acq._protocols_dir
    acq.fill_protocols()
    names = [acq.protocol.itemText(i) for i in range(acq.protocol.count())]
    pick = [i for i, n in enumerate(names) if args.protocol.lower() in n.lower()]
    if not pick:
        print(f"no protocol named like {args.protocol!r} in {acq._protocols_dir}: {names}")
        return 1
    acq.protocol.setCurrentIndex(pick[0])
    print(f"playing {names[pick[0]]} (nothing kept)...", flush=True)
    t.trace.clear()
    caught_at: list[float] = []
    landed: list[tuple[float, float, float]] = []    # where each caught pulse's light was brightest (x, y px), rise

    def flashed(f, b, rise, when) -> None:
        caught_at.append(when)
        if region is None:
            return
        from scipy import ndimage

        from clampwise.app.screens.camera_panel import flash_rise

        d = ndimage.gaussian_filter(flash_rise(f.rgb[:, :, 0], b.rgb[:, :, 0]), 1.0)
        near = np.zeros_like(d, bool)                # near the region (2 of its radii): not the reflected copy
        yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
        near[np.hypot(xx * 8 + 4 - region[0], yy * 8 + 4 - region[1]) < 2 * region[2] + 80] = True
        if not near.any():
            return
        k = np.argmax(np.where(near, d, -np.inf))
        y, x = np.unravel_index(k, d.shape)
        w = np.clip(d - 0.5 * d[y, x], 0, None) * near
        landed.append((float((w * (xx * 8 + 4)).sum() / w.sum()), float((w * (yy * 8 + 4)).sum() / w.sum()), rise))

    t.flash.connect(flashed)
    acq.play()
    polls: list[tuple[float, int]] = []
    shot = [False]

    def busy() -> bool:
        st = acq.rec.status if acq.rec is not None else None
        if st is not None and st.frames:
            polls.append((time.monotonic(), st.frames))
        if st is not None and args.sweeps and st.sweeps_done >= args.sweeps:
            acq.stop()
        if caught_at and not shot[0] and time.monotonic() - caught_at[0] > 0.3:
            shot[0] = True
            cam.view.grab().save(str(args.out / "caught-pulse-screen.png"))
        return st is not None and not st.busy

    args.out.mkdir(parents=True, exist_ok=True)
    _common.pump(600.0, until=busy)
    st = acq.rec.status
    print(f"run: {st.state}, {st.message}", flush=True)
    if not polls or st.raw is None or not Path(st.raw).exists():
        print("no samples to time the pulses by")
        return 1
    rate = st.rate_hz
    t0 = min(tp - n / rate for tp, n in polls)        # the run's first sample in the PC's time (reports come late)
    names_in = [n for n, _, _ in st.inputs]
    col = next((i for i, n in enumerate(names_in) if args.led_input in n), None)
    if col is None:
        print(f"no input named with {args.led_input!r}: {names_in}")
        return 1
    raw = np.fromfile(st.raw, "<i2").reshape(-1, len(names_in))[:, col].astype(float)
    hi = raw > (raw.min() + raw.max()) / 2 if raw.max() - raw.min() > 50 else np.zeros(len(raw), bool)
    starts = np.flatnonzero(hi[1:] & ~hi[:-1]) + 1
    ends = np.flatnonzero(~hi[1:] & hi[:-1]) + 1
    pulses = [(s / rate, ((ends[ends > s][0] if (ends > s).any() else s) - s) / rate) for s in starts]
    trace = np.array(list(t.trace)) if t.trace else np.zeros((0, 2))
    dts = np.diff(trace[:, 0]) if len(trace) > 1 else np.array([0.0])
    print(f"{len(pulses)} pulses on {names_in[col]} ({', '.join(f'{w * 1000:.1f}' for _, w in pulses[:3])} ms long); "
          f"camera: {len(trace)} frames looked at, {1 / np.median(dts):.1f} a second, exposure "
          f"{cam.exposure.value():.3g} ms (each frame's time {np.median(dts) * 1000:.1f} ms)")
    from clampwise.app.screens.camera_panel import CATCH_GREY, CATCH_NOISE_X

    thr = max(CATCH_GREY, CATCH_NOISE_X * float(np.median(trace[:, 1]))) if len(trace) else CATCH_GREY
    flagged = trace[trace[:, 1] >= thr] if len(trace) else trace
    used = np.zeros(len(flagged), bool)
    lags, missed = [], 0
    for k, (ps, width) in enumerate(pulses):
        at = t0 + ps
        near = (flagged[:, 0] >= at) & (flagged[:, 0] <= at + 0.06) if len(flagged) else np.zeros(0, bool)
        if near.any():
            i = np.flatnonzero(near)[np.argmax(flagged[near, 1])]
            used |= near
            lags.append(flagged[i, 0] - at)
            print(f"  pulse {k + 1}: at {ps:.4f} s of the run, caught {lags[-1] * 1000:.0f} ms later, "
                  f"+{flagged[i, 1]:.0f} grey ({int(near.sum())} frame(s))")
        else:
            missed += 1
            print(f"  pulse {k + 1}: at {ps:.4f} s of the run, NOT caught")
    stray = int((~used).sum())
    rises = trace[:, 1] if len(trace) else np.zeros(1)
    print(f"caught {len(pulses) - missed} of {len(pulses)}; frames that rose with no pulse: {stray}; "
          f"a frame's usual rise {np.median(rises):.1f} grey, threshold {thr:.1f}")
    if lags:
        print(f"each pulse to the frame that caught it: {np.min(lags) * 1000:.0f}-{np.max(lags) * 1000:.0f} ms "
              f"(median {np.median(lags) * 1000:.0f} ms; the frame's exposure, its readout and its transfer)")
    print(f"the panel said {cam.caught} pulses caught; a screenshot of one: {args.out / 'caught-pulse-screen.png'}")
    if region is not None and landed:
        lx, ly = np.array([p[0] for p in landed]), np.array([p[1] for p in landed])
        off = np.hypot(lx - region[0], ly - region[1]) * um_px
        print(f"where the light landed, against the region drawn ({region[2] * 2 * um_px:.0f} um across): its centre "
              f"{np.median(off):.1f} um from the region's (worst {off.max():.1f} um, {len(off)} frames); "
              f"mean place ({lx.mean():.0f}, {ly.mean():.0f}) px for ({region[0]:.0f}, {region[1]:.0f})")
    np.savetxt(args.out / "camera-trace.csv", trace, delimiter=",", header="monotonic_s,rise_grey", comments="")
    QtGui.QGuiApplication.processEvents()
    cam.catch.setChecked(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
