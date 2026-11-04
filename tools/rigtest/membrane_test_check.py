"""Clampwise's membrane test beside what the amplifier says it did: the test run as the app runs it (its Rt, and the
output scale the telegraph gave), then the same pulse played again with the amplifier's secondary output (membrane
potential in voltage clamp) recorded on IN 1, so the step that reached the pipette is measured, not assumed; Rt
from that step and the current, and from the step asked for. With the pipette in the bath (or the model cell's
BATH), Clampex closed, voltage clamp, the secondary output on membrane potential into IN 1. Moves the digitizer's
command output: +-5 mV around 0.

    python -I tools/rigtest/membrane_test_check.py --at-the-rig [--seconds 5]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402


def main() -> int:
    p = _common.parser(__doc__, hardware=True)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--channel", type=int, default=1, help="the amplifier's channel (IN 0 its primary)")
    args = p.parse_args()
    _common.require_presence(args)
    import numpy as np

    from clampwise.rig import acquire, digidata, membrane_test as mt, telegraph

    tg = telegraph.read(None, args.channel)
    if tg is None:
        print("no telegraph: is Commander open?")
        return 1
    print(f"telegraph: {tg.mode}, primary {tg.primary_unit} at {mt.scale_words(tg)}, secondary {tg.secondary_unit} "
          f"at {tg.secondary_v_per_unit:g} V a {tg.secondary_unit}, command {tg.command_per_volt:g} "
          f"{tg.command_unit}/V, filter {tg.primary_lpf_hz:g} Hz")
    if tg.mode != "VC":
        print("put the amplifier in voltage clamp")
        return 1
    test = mt.MembraneTest(channel=args.channel)
    test.start()
    time.sleep(args.seconds)
    r = test.latest
    test.stop()
    test.wait(10)
    if r is None:
        print(f"the membrane test gave nothing: {test.state}, {test.message}")
        return 1
    print(f"Clampwise's membrane test: Rt {r.rt_mohm:.2f} MOhm ({r.what}), holding {r.holding:.1f} pA, "
          f"{r.pulses} pulses; {test.message}")

    s = mt.Settings()
    period, up, down = s.period, s.period // 4, 3 * s.period // 4
    helper = digidata.lend()
    try:
        cal = helper.calibration
        volts = np.zeros((period, acquire.N_AO))
        volts[:, 0] = -0.5 * s.amplitude / tg.command_per_volt
        volts[up:down, 0] = 0.5 * s.amplitude / tg.command_per_volt
        wave = np.empty((period, acquire.N_OUT), "<i2")
        for d in range(acquire.N_AO):
            wave[:, d] = acquire.dac_codes(volts[:, d], cal["dac_gains"][d], cal["dac_offsets"][d])
        wave[:, acquire.N_AO] = 0
        work = _common.OUT
        work.mkdir(parents=True, exist_ok=True)
        wfile, raw = work / "mtcheck.wave", work / "mtcheck.raw"
        wave.tofile(wfile)
        n = 100
        line = helper.record({"ai": [0, 1], "period_us": 1e6 / s.rate_hz, "frames": period * n,
                              "hold": acquire.holding_codes(None, cal), "waveform": str(wfile), "repeat": True,
                              "out": str(raw), "say_s": 0.25}, None)
        if "error" in line:
            print(f"the check's run failed: {line['error']}")
            return 1
    finally:
        helper.close()
    counts = np.fromfile(raw, "<i2").reshape(-1, 2).astype(float) * (10.0 / 32768)
    pulses = counts[period * 10: period * n].reshape(-1, period, 2).mean(axis=0)   # the first 10 left to settle
    i = pulses[:, 0] / tg.primary_v_per_unit
    v = pulses[:, 1] / tg.secondary_v_per_unit
    w = int(0.2 * (down - up))
    di = i[down - w:down].mean() - i[up - w:up].mean()
    dv = v[down - w:down].mean() - v[up - w:up].mean()
    edge = int(np.argmax(np.abs(np.diff(v)))) + 1
    print(f"the step asked for {s.amplitude:g} mV, the secondary output says {dv:.2f} mV reached the pipette; "
          f"the current changed {di:.1f} pA")
    print(f"Rt from the step asked for {s.amplitude / di * 1e3:.2f} MOhm, from the step measured {dv / di * 1e3:.2f} "
          f"MOhm; the recorded edge at sample {edge} of {period} (expected {up} or {down})")
    for f in (wfile, raw):
        f.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _common.leave_safely()
