"""How soon after a TTL at its Trig In the patterned light shows its next pattern: the steps loaded into it, the
Digidata pulsing a digital output into Trig In and the LED (on analog output 2, as the photostim protocols drive it)
flashing at a chosen delay after, the camera saying which pattern each flash lit. Moves hardware: only with someone
at the rig, no cell patched (the amplifier's command outputs are held at 0 V throughout).

Two passes:

- ``--find``: which digital output reaches Trig In. Two patterns loaded (the field's left half, then its right), the
  LED held on, each digital output (0-15) pulsed once in turn; the pattern seen on the camera (and the Polygon's own
  count of the pattern shown) before and after each pulse.
- ``--timing --trigger-bit N``: two triggers every 10 ms (at 0 and 5 ms) and a short flash at a delay after the
  second; the camera adds up each delay's flashes over a few seconds and the picture is fitted as so much of the
  pattern before the trigger and so much of the one after. Where the flash's light moves from one to the other is
  how long the Polygon takes.

Needs: Clampex, PolyScan and Clampwise closed (this opens its own window), a sample in focus, the LED driver
following analog output 2, the 20x calibration kept.

    python -I tools/rigtest/polygon_trigger_timing.py --at-the-rig --find
    python -I tools/rigtest/polygon_trigger_timing.py --at-the-rig --timing --trigger-bit 0
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

N_AO, N_OUT = 4, 5
DELAYS_US = (-300, -100, 0, 25, 50, 75, 100, 150, 200, 300, 500, 1000)


class Digidata:
    """The Digidata through Clampwise's helper: outputs played as frames, one input recorded, the camera's pictures
    handed to ``on_frame(seconds since the start, grey)`` as they come."""

    def __init__(self, led_out: int, led_volts: float, ai: int) -> None:
        from clampwise.rig import acquire, digidata

        why = digidata.available()
        if why:
            raise RuntimeError(why)
        self.acquire = acquire
        for attempt in range(3):                      # its first count after a quiet while can come back 0
            try:
                self.helper = digidata.Helper()
                break
            except KeyError:
                if attempt == 2:
                    raise RuntimeError("the Digidata was not found (is it on, and Clampex closed?)") from None
                time.sleep(1.0)
        self.cal = self.helper.calibration
        self.hold = acquire.holding_codes(None, self.cal)          # every analog output 0 V, every digital low
        self.led_out, self.ai = led_out, ai
        self.led_code = int(acquire.dac_codes(led_volts, self.cal["dac_gains"][led_out],
                                              self.cal["dac_offsets"][led_out]))
        self.folder = Path(tempfile.mkdtemp(prefix="clampwise-trigger-"))

    def frames(self, n: int) -> "np.ndarray":
        import numpy as np

        f = np.tile(np.asarray(self.hold, "<i2"), (n, 1))
        f[:, N_AO] = 0
        return f

    @staticmethod
    def digital(frames, bit: int, a: int, b: int) -> None:
        import numpy as np

        word = frames[:, N_AO].view(np.uint16)
        word[a:b] |= np.uint16(1 << bit)

    def led(self, frames, a: int, b: int) -> None:
        frames[a:b, self.led_out] = self.led_code

    def play(self, wave, period_us: float, total: int, repeat: bool, cam, on_frame=None, every=None) -> "np.ndarray":
        """``wave`` (frames x 5 codes) played (over and over when ``repeat``) for ``total`` frames; the input
        recorded, in codes. ``every(seconds)`` is called about ten times a second meanwhile."""
        import numpy as np

        path, out = self.folder / "wave.bin", self.folder / "in.bin"
        np.ascontiguousarray(wave, "<i2").tofile(path)
        req = {"ai": [self.ai], "frames": int(total), "hold": self.hold, "waveform": str(path), "repeat": repeat,
               "period_us": float(period_us), "out": str(out), "say_s": 0.5}
        result = {}
        thread = threading.Thread(target=lambda: result.update(self.helper.record(req)), daemon=True)
        t0 = time.monotonic()
        thread.start()
        seen, last_every = cam.last, 0.0
        while thread.is_alive():
            _common.pump(0.01)
            now = time.monotonic() - t0
            if cam.last is not None and cam.last is not seen:
                seen = cam.last
                if on_frame is not None:
                    on_frame(now, seen.rgb[:, :, 0])
            if every is not None and now - last_every >= 0.1:
                last_every = now
                every(now)
        thread.join()
        if "error" in result:
            raise RuntimeError(result["error"])
        return np.fromfile(out, "<i2") if out.exists() else np.zeros(0, "<i2")

    def close(self) -> None:
        self.helper.close()


def open_light(args):
    """Clampwise's window with the camera and the patterned light taken, the objective's calibration in hand."""
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
        _common.pump(60.0, until=lambda: not light._opening)
        raise RuntimeError(f"no camera picture: {cam.status.text()}")
    if not _common.pump(60.0, until=lambda: light.polygon.connected or not (light._opening or light.busy())):
        raise RuntimeError("the patterned light was not found in 60 s")
    if not light.polygon.connected:
        raise RuntimeError(f"the patterned light was not taken: {light.polygon.status}")
    if not args.keep:                                 # the calibrations kept, read (never written) from the settings
        import json

        from pyqtgraph.Qt import QtCore

        from clampwise import APP_NAME
        from clampwise.app.window import SETTINGS_ORG
        from clampwise.rig import patterns

        kept = QtCore.QSettings(SETTINGS_ORG, APP_NAME).value("polygon/calibration", "{}") or "{}"
        light.calibrations = patterns.Calibrations.from_dict(json.loads(str(kept)))
        light.calibration = light.calibrations.at(light._objective())
    cal = light.calibration.at_size(light._frame_size())
    if not cal.ready:
        raise RuntimeError(f"no calibration kept at {light._objective()}: calibrate first")
    print(f"objective {cam.magnification:g}x ({light._objective()}), calibration kept {cal.measured or '?'}",
          flush=True)
    return cam, light, cal


def halves(cal, shape):
    """The camera's pixels on the field's left half and on its right (a band at the middle left out)."""
    import numpy as np

    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    u, v = cal.field_of(x.astype(float), y.astype(float))
    inside = (u > 0.02) & (u < 0.98) & (v > 0.02) & (v < 0.98)
    return inside & (u < 0.46), inside & (u > 0.54)


def set_exposure(cam, light, dd: Digidata, args, duty: float) -> None:
    """Every mirror on and the LED held on: the exposure the camera's auto exposure settles at, then lengthened for
    light that is on only ``duty`` of the time (and the frame rate let follow it)."""
    light.source.setCurrentText("Full field")
    light.make_steps()
    light.project()
    light.wait()
    cam.frame_rate.setValue(0.0)
    cam._frame_rate(quiet=True)
    cam.auto.setChecked(True)
    period = 100.0
    n = int(8.0 * 1e6 / period)
    wave = dd.frames(n)
    dd.led(wave, 0, n - 1)
    dd.play(wave, period, n, False, cam)
    cam.auto.setChecked(False)
    steady = cam.exposure.value()
    want = steady if duty >= 1.0 else min(steady / max(duty, 1e-6) * 0.5, args.longest_ms)
    cam.exposure.setValue(want)
    cam._exposure()
    _common.pump(1.0)
    print(f"the LED held on, every mirror on: auto exposure {steady:g} ms; for flashes {duty:.1%} of the time, "
          f"exposure {cam.exposure.value():g} ms, gain {cam.gain.value():g}", flush=True)


def load_halves(light, on: int, shown_us: int) -> None:
    from clampwise.rig import patterns

    masks = [s.mirrors() for s in patterns.grid(1, 2)]
    light._async(lambda: light.polygon.run_sequence(masks, on, shown_us))
    light.wait()
    if not light.polygon.sequence:
        raise RuntimeError(f"the two patterns were not loaded: {light.polygon.status}")
    print(f"loaded: the field's left half, then its right; each shown {shown_us} us on a trigger", flush=True)


def check(args) -> int:
    """No trigger at all: the left half and the right as still pictures, then both loaded as a sequence that steps
    itself every 0.5 s; the camera's left and right means through it all, the LED held on."""
    import numpy as np

    from clampwise.rig import patterns, polygon

    cam, light, cal = open_light(args)
    dd = Digidata(args.led_out, args.led_volts, args.ai)
    try:
        set_exposure(cam, light, dd, args, duty=1.0)
        on = cal.on_value
        masks = [s.mirrors() for s in patterns.grid(1, 2)]
        plan = [(1.0, "still: left half", lambda: light.polygon.send(patterns.image(masks[0], on))),
                (4.0, "still: right half", lambda: light.polygon.send(patterns.image(masks[1], on))),
                (7.0, "sequence stepping itself every 0.5 s", lambda: light.polygon.run_sequence(
                    masks, on, 500_000, trigger=polygon.TRIGGER_AUTO)),
                (13.0, "sequence stopped, nothing lit", lambda: light.polygon.off(on))]
        done, seen, left = [], [], {}

        def every(t):
            while plan and t >= plan[0][0]:
                _, name, act = plan.pop(0)
                light._async(lambda act=act, name=name: done.append((time.monotonic() - t_start[0], name, act())))

        def on_frame(t, g):
            if not left:
                left["l"], left["r"] = halves(cal, g.shape)
            seen.append((t, float(g[left["l"]].mean()), float(g[left["r"]].mean())))

        period = 100.0
        n = int(15.0 * 1e6 / period)
        wave = dd.frames(n)
        dd.led(wave, 0, n - 1)
        t_start = [time.monotonic()]
        dd.play(wave, period, n, False, cam, on_frame, every)
        light.wait()
        for t, name, ok in done:
            print(f"{t:5.2f} s: {name} ({'taken' if ok else 'refused: ' + light.polygon.status})")
        print("time   left  right (grey)")
        for b in np.arange(0.0, 15.0, 0.25):
            rows = [(l, r) for t, l, r in seen if b <= t < b + 0.25]
            if rows:
                l, r = np.mean(rows, axis=0)
                print(f"{b:5.2f}  {l:5.1f}  {r:5.1f}  {'#' * int(l / 8):<32}|{'#' * int(r / 8)}")
        return 0
    finally:
        dd.close()
        _common.leave_safely()


def find(args) -> int:
    import numpy as np

    cam, light, cal = open_light(args)
    dd = Digidata(args.led_out, args.led_volts, args.ai)
    try:
        set_exposure(cam, light, dd, args, duty=1.0)
        load_halves(light, cal.on_value, shown_us=args.shown_us or 1_000_000)
        left = right = None
        period, seg = 100.0, 1.5
        per = int(seg * 1e6 / period)
        bits = list(range(args.bits))
        n = per * len(bits)
        wave = dd.frames(n + 10)
        dd.led(wave, 0, n)
        for k, bit in enumerate(bits):
            a = k * per + int(0.5 * 1e6 / period)
            dd.digital(wave, bit, a, a + int(5000 / period))              # 5 ms high
        seen, counts = [], []

        def on_frame(t, g):
            nonlocal left, right
            if left is None:
                left, right = halves(cal, g.shape)
            seen.append((t, float(g[left].mean()), float(g[right].mean())))

        def every(t):
            counts.append((t, light.polygon.sequence_index()))

        print(f"the LED on; each digital output 0-{len(bits) - 1} pulsed 5 ms in turn, 1.5 s apart...", flush=True)
        ai = dd.play(wave, period, n + 10, False, cam, on_frame, every)
        lit = np.flatnonzero(ai > ai.max() * 0.5) if ai.size else []
        print(f"input {args.ai} (the LED's command): high for {len(lit) * period / 1e6:.2f} s of "
              f"{n * period / 1e6:.1f} s, peak {ai.max() if ai.size else 0} codes")
        found = []
        for k, bit in enumerate(bits):
            t0 = k * seg

            def window(a, b):
                rows = [(l, r) for t, l, r in seen if t0 + a <= t <= t0 + b]
                c = [c for t, c in counts if t0 + a <= t <= t0 + b and c is not None]
                return (np.mean(rows, axis=0) if rows else (np.nan, np.nan)), (c[-1] if c else None), len(rows)

            (lb, rb), cb, nb = window(0.1, 0.45)
            (la, ra), ca, na = window(0.7, 1.45)
            moved = abs((ra - la) - (rb - lb))
            said = (f"digital {bit}: left/right {lb:.1f}/{rb:.1f} before, {la:.1f}/{ra:.1f} after "
                    f"({nb}, {na} pictures); pattern shown {cb} -> {ca}")
            if (cb is not None and ca is not None and cb != ca) or moved > args.min_grey:
                found.append(bit)
                said += "   <-- the pattern changed"
            print(said, flush=True)
        if found:
            print(f"Trig In answers to digital output {', '.join(map(str, found))}")
        else:
            print(f"No digital output 0-{len(bits) - 1} stepped the patterned light: nothing from the Digidata reaches its Trig In "
                  "(or the LED was not on: see the grey levels).")
        return 0 if found else 1
    finally:
        dd.close()
        _common.leave_safely()


def timing(args) -> int:
    import numpy as np

    cam, light, cal = open_light(args)
    dd = Digidata(args.led_out, args.led_volts, args.ai)
    period, cycle_us = 10.0, 10_000
    cycle = int(cycle_us / period)
    second = cycle // 2                                   # the second trigger, 5 ms into each cycle
    width = int(args.pulse_us / period)
    trig = int(200 / period)                              # each trigger 200 us high
    try:
        set_exposure(cam, light, dd, args, duty=args.pulse_us / cycle_us)
        load_halves(light, cal.on_value, shown_us=args.shown_us or 4000)

        def run(name: str, flash_at: int | None):
            """One condition for ``args.seconds``: the camera's pictures added up (the first and last left out)."""
            wave = dd.frames(cycle)
            dd.digital(wave, args.trigger_bit, 0, trig)
            dd.digital(wave, args.trigger_bit, second, second + trig)
            if flash_at is not None:
                dd.led(wave, flash_at, flash_at + width)
            total = int(args.seconds * 1e6 / period) // cycle * cycle      # whole cycles: an even count of triggers
            acc = {"sum": None, "n": 0}

            def on_frame(t, g):
                if 0.3 + cam.exposure.value() / 1000 <= t <= args.seconds - 0.2:
                    g = g.astype(np.float64)
                    acc["sum"] = g if acc["sum"] is None else acc["sum"] + g
                    acc["n"] += 1

            ai = dd.play(wave, period, total, True, cam, on_frame)
            if acc["n"] == 0:
                raise RuntimeError(f"{name}: no picture taken in {args.seconds} s (exposure too long?)")
            onset = ""
            if ai.size >= 2 * cycle and flash_at is not None:
                one = ai[cycle:2 * cycle].astype(float)
                high = np.flatnonzero(one > (one.max() + one.min()) / 2)
                if high.size:
                    onset = f", the LED's command high from {(high[0] - second) * period:+.0f} us"
            img = acc["sum"] / acc["n"]
            print(f"  {name}: {acc['n']} pictures, mean grey {img.mean():.2f}{onset}", flush=True)
            return img

        print(f"two triggers on digital {args.trigger_bit} every 10 ms; flashes of {args.pulse_us:g} us; "
              f"{args.seconds:g} s a condition", flush=True)
        dark = run("no flash", None)
        before = run("flash 3 ms before the second trigger", second - int(3000 / period))
        after = run("flash 2 ms after it", second + int(2000 / period))
        left, right = halves(cal, dark.shape)
        field = left | right
        b, a = (before - dark)[field], (after - dark)[field]
        contrast = np.abs(a - b).mean() / max(np.abs(a).mean(), np.abs(b).mean(), 1e-9)
        print(f"before: left {(before - dark)[left].mean():.2f}, right {(before - dark)[right].mean():.2f}; "
              f"after: left {(after - dark)[left].mean():.2f}, right {(after - dark)[right].mean():.2f} "
              f"(grey above no flash); the two differ by {contrast:.0%}", flush=True)
        if contrast < 0.3:
            print("The pattern before and after the trigger look alike: the trigger did not step it (wrong output, "
                  "the shown time's unit, or too little light).")
            return 1
        basis = np.vstack([b, a]).T
        rows = []
        for d in DELAYS_US:
            img = run(f"flash at {d:+d} us", second + int(round(d / period)))
            (x, y), *_ = np.linalg.lstsq(basis, (img - dark)[field], rcond=None)
            rows.append((d, x, y))
        again = run("flash 2 ms after it (again)", second + int(2000 / period))
        (x2, y2), *_ = np.linalg.lstsq(basis, (again - dark)[field], rcond=None)
        print(f"\nflash {args.pulse_us:g} us from the delay; light as the pattern before / after the trigger "
              f"(1 = all of a flash):")
        for d, x, y in rows:
            share = y / (x + y) if x + y > 0.05 else float("nan")
            print(f"  {d:+6d} us: before {x:5.2f}, after {y:5.2f}  -> {share:6.0%} of it the new pattern")
        print(f"  check, 2 ms after again: before {x2:.2f}, after {y2:.2f} (should be about 0 and 1)")
        new = [d for d, x, y in rows if x + y > 0.05 and y / (x + y) >= 0.9]
        if new:
            print(f"\nThe new pattern held >= 90% of a flash from {min(new):+d} us after the trigger's edge.")
        return 0
    finally:
        dd.close()
        _common.leave_safely()


def main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--check", action="store_true", help="no trigger: the patterns as stills, then stepping itself")
    p.add_argument("--find", action="store_true", help="which digital output reaches Trig In")
    p.add_argument("--timing", action="store_true", help="how soon after a trigger the next pattern shows")
    p.add_argument("--trigger-bit", type=int, default=0, help="the digital output wired to Trig In")
    p.add_argument("--objective", type=float, default=20.0)
    p.add_argument("--led-out", type=int, default=2, help="the analog output driving the LED")
    p.add_argument("--led-volts", type=float, default=0.25, help="the LED's level (the photostim protocols': 0.25 V)")
    p.add_argument("--ai", type=int, default=8, help="the input recording the LED's command")
    p.add_argument("--pulse-us", type=float, default=100.0, help="each flash (timing)")
    p.add_argument("--seconds", type=float, default=4.0, help="each condition (timing)")
    p.add_argument("--shown-us", type=int, default=0, help="each pattern's shown time (0: 1 s finding, 4 ms timing)")
    p.add_argument("--longest-ms", type=float, default=1000.0, help="the longest exposure to use")
    p.add_argument("--bits", type=int, default=16, help="the digital outputs tried, from 0 (finding)")
    p.add_argument("--min-grey", type=float, default=3.0, help="a change in grey counted as the pattern moving")
    args = p.parse_args()
    _common.require_presence(args)
    if args.check + args.find + args.timing != 1:
        print("say --check, --find or --timing")
        return 2
    try:
        return check(args) if args.check else find(args) if args.find else timing(args)
    except RuntimeError as e:
        print(f"stopped: {e}")
        _common.leave_safely()
        return 1


if __name__ == "__main__":
    sys.exit(main())
