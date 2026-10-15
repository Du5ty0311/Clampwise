"""The cell's health at a glance: access resistance and the other general readouts of a cell's health.

Each reading comes from the best source there is, said with it: the test pulse of each sweep the acquisition is
writing (the membrane test's own measure), else the acquisition's last Membrane Test (its Lab Book), else the
amplifier (its meters; its whole-cell compensation when on). Each is checked against the quality limits the
analyses use (analysis/quality.py: Rs at most 25 MOhm and changing at most 20%, Rs/Rm at most 0.15, holding at most
200 pA, Rin 20-1000 MOhm, noise SD at most 200 pA, resting potential -85 to -55 mV), and a reading outside them is
marked, never changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NAN = float("nan")


@dataclass
class Reading:
    key: str
    label: str
    value: float
    unit: str
    source: str = ""
    status: str = "muted"         # "ok" (within its limit), "warn" (outside), "muted" (none read, or no limit)
    limit: str = ""

    def shown(self) -> str:
        if not math.isfinite(self.value):
            return "–"
        v = self.value
        if self.unit == "%":
            return f"{v:+.0f} %"
        digits = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
        return f"{v:.{digits}f} {self.unit}".strip()


def limits() -> dict:
    from ..analysis import registry

    return registry.load()["quality"].defaults()


def _first(*pairs) -> tuple[float, str]:
    """The first (value, source) whose value is a finite number."""
    for value, source in pairs:
        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
            return float(value), source
    return NAN, ""


def readings(amp=None, sweeps=(), test=None, lim: dict | None = None) -> list[Reading]:
    """The cell's readings, voltage clamp or current clamp as the amplifier (else the recorded trace) says."""
    from . import growing

    lim = lim or limits()
    sweeps = list(sweeps)
    measured = [s for s in sweeps if math.isfinite(s.rs_mohm)]
    last = measured[-1] if measured else None
    newest = sweeps[-1] if sweeps else None
    tv = test.values if test is not None and getattr(test, "values", None) else {}
    connected = amp is not None and getattr(amp, "connected", False)
    vc = amp.mode == 0 if connected else (newest.clamp == "voltage clamp" if newest else True)
    channel = amp.device.channel if connected and amp.device is not None else 1
    meters = amp.meters if connected else {}
    meter_v = meters.get(2 * (channel - 1))
    meter_i = meters.get(2 * (channel - 1) + 1)
    pulse = f"the recorded test pulse ({last.file}, sweep {last.sweep + 1})" if last else ""
    book = "the acquisition's last Membrane Test"
    whole = connected and amp.values.get("whole_cell_on")
    comp = "the amplifier's whole-cell compensation"
    out: list[Reading] = []

    def add(key, label, value_source, unit, low=None, high=None, absolute=False, limit_text=""):
        value, source = value_source
        r = Reading(key, label, value, unit, source, limit=limit_text)
        if math.isfinite(value) and (low is not None or high is not None):
            v = abs(value) if absolute else value
            ok = (low is None or v >= low) and (high is None or v <= high)
            r.status = "ok" if ok else "warn"
        out.append(r)

    if vc:
        rs = _first((last.rs_mohm if last else None, pulse), (tv.get("ra_mohm"), book),
                    (amp.values.get("rs") if whole else None, comp))
        rm = _first((last.rm_mohm if last else None, pulse), (tv.get("rm_mohm"), book))
        rt = _first((last.rt_mohm if last else None, pulse), (tv.get("rt_mohm"), book))
        cm = _first((last.cm_pf if last else None, pulse), (tv.get("cm_pf"), book),
                    (amp.values.get("cm") if whole else None, comp))
        add("rs", "Access (Rs)", rs, "MOhm", high=lim["max_rs_mohm"],
            limit_text=f"at most {lim['max_rs_mohm']:g} MOhm")
        change = growing.rs_change(sweeps) if sweeps else NAN
        add("rs_change", "Rs change", (change * 100, "the cell's first three sweeps to its last"), "%",
            high=lim["max_rs_change"] * 100, absolute=True, limit_text=f"at most {lim['max_rs_change']:.0%}")
        add("rm", "Membrane (Rm)", rm, "MOhm", low=lim["rin_low_mohm"], high=lim["rin_high_mohm"],
            limit_text=f"{lim['rin_low_mohm']:g} to {lim['rin_high_mohm']:g} MOhm")
        add("rt", "Total (Rt)", rt, "MOhm")
        add("cm", "Capacitance (Cm)", cm, "pF")
        ratio = rs[0] / rm[0] if math.isfinite(rs[0]) and math.isfinite(rm[0]) and rm[0] > 0 else NAN
        add("rs_rm", "Rs / Rm", (ratio, "Rs over Rm, as above"), "", high=lim["max_rs_rm"],
            limit_text=f"at most {lim['max_rs_rm']:g}")
        tau = rs[0] * cm[0] * 1e-3 if math.isfinite(rs[0]) and math.isfinite(cm[0]) else NAN     # MOhm x pF = us
        add("clamp_tau", "Clamp speed (Rs x Cm)", (tau, "Rs times Cm: how fast the clamp follows the cell"), "ms")
        hold = _first((meter_i * 1e12 if meter_i is not None else None, "the amplifier's current meter, now"),
                      (newest.holding if newest else None, "the recorded trace before its test step"),
                      (tv.get("hold_pa"), book))
        add("holding", "Holding current", hold, "pA", high=lim["max_holding_pa"], absolute=True,
            limit_text=f"at most {lim['max_holding_pa']:g} pA (below -50 mV)")
        add("noise", "Noise (SD)", (newest.noise if newest else NAN, "the recorded trace before its test step"),
            "pA", high=lim["max_noise_sd_pa"], limit_text=f"at most {lim['max_noise_sd_pa']:g} pA")
    else:
        vm = _first((meter_v * 1e3 if meter_v is not None else None, "the amplifier's potential meter, now"),
                    (newest.holding if newest else None, "the recorded trace"))
        add("vm", "Membrane potential", vm, "mV", low=lim["rest_low_mv"], high=lim["rest_high_mv"],
            limit_text=f"{lim['rest_low_mv']:g} to {lim['rest_high_mv']:g} mV at rest")
        bridge = amp.values.get("bridge") if connected and amp.values.get("bridge_on") else None
        add("bridge", "Bridge balance", _first((bridge, "the amplifier's bridge balance")), "MOhm")
        add("noise", "Noise (SD)", (newest.noise if newest else NAN, "the recorded trace"), "mV")
        stepped = [s for s in sweeps if math.isfinite(s.rm_mohm)]
        step = f"the recorded current step ({stepped[-1].file}, sweep {stepped[-1].sweep + 1})" if stepped else ""
        rm = _first((stepped[-1].rm_mohm if stepped else None, step), (tv.get("rm_mohm"), book))
        add("rm", "Input resistance (Rin)", rm, "MOhm", low=lim["rin_low_mohm"], high=lim["rin_high_mohm"],
            limit_text=f"{lim['rin_low_mohm']:g} to {lim['rin_high_mohm']:g} MOhm")
        add("cm", "Capacitance (Cm)", _first((stepped[-1].cm_pf if stepped else None, f"tau / Rin, {step}")), "pF")
    return out
