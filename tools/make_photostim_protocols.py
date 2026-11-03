"""The photostimulation protocols for the ChroME mapping experiments, written from one of the rig's own photostim
protocols (its inputs, rate, outputs and their scales kept), so Clampex and Clampwise both play them. Current clamp
from rest; one 5 ms pulse of the LED (Cmd 2) at 310 ms a sweep (as the rig's protocols); a -20 pA, 100 ms step at
the end of each sweep on the amplifier's command (Cmd 0), its input resistance; the 450 ms between them the response.

    python tools/make_photostim_protocols.py TEMPLATE.pro OUT_FOLDER

Never replaces a file: each protocol is written under its own new name.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clampwise.rig import protocol_edit  # noqa: E402
from clampwise.rig.protocol_edit import Epoch  # noqa: E402

LIGHT, COMMAND = 2, 0                 # the LED's output and the amplifier's
LEVEL = 5.0                           # the LED's level in the rig's photostim protocols
TEST_PA = -20.0


def ms(e, t: float) -> int:
    return int(round(t * e.rate_hz / 1000.0))


def protocol(template, sweeps: int, every_s: float, level: float, level_inc: float = 0.0, pulse_ms: float = 5.0,
             pulse_inc_ms: float = 0.0):
    e = copy.deepcopy(template)
    digital = template.epochs[LIGHT][0].digital       # the template's digital outputs kept as they were
    e.sweeps, e.start_to_start_s, e.sweep_samples = sweeps, every_s, ms(e, 1000.0)
    e.epochs = {
        COMMAND: [Epoch(1, 0.0, samples=ms(e, 750.0), digital=digital),
                  Epoch(1, TEST_PA, samples=ms(e, 100.0)),
                  Epoch(1, 0.0, samples=ms(e, 100.0))],
        LIGHT: [Epoch(1, 0.0, samples=ms(e, 295.0), digital=digital),        # after the 1/64 held: at 310.6 ms
                Epoch(1, level, level_inc, ms(e, pulse_ms), ms(e, pulse_inc_ms)),
                Epoch(1, 0.0, samples=ms(e, 600.0))],
    }
    e.holding = [0.0] * len(template.holding)
    e.enabled = [i in (COMMAND, LIGHT) for i in range(len(template.enabled))]
    return e


DESIGNS = {
    "ChroME_threshold_intensity_CC": dict(sweeps=10, every_s=5.0, level=0.5, level_inc=0.5),
    "ChroME_threshold_duration_CC": dict(sweeps=10, every_s=5.0, level=LEVEL, pulse_inc_ms=5.0),
    "ChroME_grid_map_CC": dict(sweeps=49, every_s=3.0, level=LEVEL),
    "Connectivity_CC_1pulse": dict(sweeps=50, every_s=5.0, level=LEVEL),
}


def main() -> int:
    template, out = protocol_edit.load(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, d in DESIGNS.items():
        pr = protocol_edit.save(protocol(template, **d), out / f"{name}.pro")
        print(f"{name}.pro: {pr.sweeps} sweeps, one every {pr.start_to_start_s:g} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
