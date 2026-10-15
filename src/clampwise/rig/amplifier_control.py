"""The amplifier's controls in Clampwise: the mode and holding, seal test and Auto offsets, whole-cell / Rs / bridge,
outputs and filters, as MultiClamp Commander has them, with Commander itself still usable beside it.

The readout (``rig.multiclamp``) stays read only, by construction and by test. These controls are a separate,
explicit allowance: each one is a named setter or button of Commander's own message library, listed in
:data:`CONTROLS` and :data:`ACTIONS` with its unit and the modes it belongs to; nothing else is ever bound. A call goes
to Commander, which drives the amplifier and shows the change on its own panel, so Commander stays usable beside
Clampwise. Reset (every setting to the factory's) and Commander's window calls are left out. Each call is made in the
thread that opened Commander (its library wants one thread): Live's rig thread runs the queued commands before each
look.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

VC, IC, ALL = (0,), (1, 2), (0, 1, 2)
GROUPS = ("Clamp", "Seal test", "Pipette", "Whole cell", "Rs compensation", "Bridge and neutralization",
          "Leak subtraction", "Outputs")


@dataclass(frozen=True)
class Control:
    key: str                      # the readout's key (rig.multiclamp.READINGS), so the shown value is the read one
    setter: str
    kind: str                     # "u" a choice, "b" on/off, "d" a number
    label: str
    group: str
    unit: Any = ""                # the unit shown; a dict {mode: unit} where it follows the mode
    to_si: Any = 1.0              # shown -> the library's SI; a dict {mode: factor}
    modes: tuple = ALL
    low: float = 0.0
    high: float = 1.0
    step: float = 1.0
    choices: tuple = ()           # (value, words) for "u"


MV_PA = {0: "mV", 1: "pA", 2: "pA"}
MV_PA_SI = {0: 1e-3, 1: 1e-12, 2: 1e-12}
PRIMARY_VC = ((0, "membrane current"), (1, "membrane potential"), (2, "pipette potential"),
              (3, "100x AC membrane potential"), (4, "external command potential"), (5, "auxiliary 1"),
              (6, "auxiliary 2"))
PRIMARY_IC = ((7, "membrane potential"), (8, "membrane current"), (9, "command current"),
              (10, "100x AC membrane potential"), (11, "external command current"), (12, "auxiliary 1"),
              (13, "auxiliary 2"))
SECONDARY_VC = ((0, "membrane current"), (1, "membrane potential"), (2, "pipette potential"),
                (3, "100x AC membrane potential"), (4, "external command potential"), (5, "auxiliary 1"),
                (6, "auxiliary 2"))
SECONDARY_IC = ((7, "membrane potential"), (8, "membrane current"), (10, "pipette potential"),
                (11, "100x AC membrane potential"), (12, "external command current"), (13, "auxiliary 1"),
                (14, "auxiliary 2"))
GAINS = tuple((g, f"x{g:g}") for g in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000))
LPF = tuple((f, f"{f / 1000:g} kHz") for f in (1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000))

CONTROLS = (
    Control("mode", "MCCMSG_SetMode", "u", "Mode", "Clamp", choices=((0, "VC"), (2, "I = 0"), (1, "IC"))),
    Control("holding", "MCCMSG_SetHolding", "d", "Holding", "Clamp", MV_PA, MV_PA_SI, low=-1000, high=1000, step=1),
    Control("holding_on", "MCCMSG_SetHoldingEnable", "b", "Holding on", "Clamp"),
    Control("test_signal_on", "MCCMSG_SetTestSignalEnable", "b", "Seal test on", "Seal test"),
    Control("test_signal_amplitude", "MCCMSG_SetTestSignalAmplitude", "d", "Amplitude", "Seal test", MV_PA,
            MV_PA_SI, low=-1000, high=1000, step=1),
    Control("test_signal_hz", "MCCMSG_SetTestSignalFrequency", "d", "Frequency", "Seal test", "Hz", low=1,
            high=1000, step=1),
    Control("pipette_offset", "MCCMSG_SetPipetteOffset", "d", "Pipette offset", "Pipette", "mV", 1e-3, low=-100,
            high=100, step=0.1),
    Control("fast_cap", "MCCMSG_SetFastCompCap", "d", "Cp fast", "Pipette", "pF", 1e-12, VC, high=20, step=0.01),
    Control("fast_tau", "MCCMSG_SetFastCompTau", "d", "Cp fast tau", "Pipette", "us", 1e-6, VC, low=0.1, high=10,
            step=0.01),
    Control("slow_cap", "MCCMSG_SetSlowCompCap", "d", "Cp slow", "Pipette", "pF", 1e-12, VC, high=10, step=0.01),
    Control("slow_tau", "MCCMSG_SetSlowCompTau", "d", "Cp slow tau", "Pipette", "us", 1e-6, VC, low=10, high=4000,
            step=1),
    Control("whole_cell_on", "MCCMSG_SetWholeCellCompEnable", "b", "Whole cell on", "Whole cell", modes=VC),
    Control("cm", "MCCMSG_SetWholeCellCompCap", "d", "Cm", "Whole cell", "pF", 1e-12, VC, low=1, high=1000,
            step=0.1),
    Control("rs", "MCCMSG_SetWholeCellCompResist", "d", "Rs", "Whole cell", "MOhm", 1e6, VC, low=0.4, high=1000,
            step=0.1),
    Control("rs_comp_on", "MCCMSG_SetRsCompEnable", "b", "Rs compensation on", "Rs compensation", modes=VC),
    Control("rs_correction", "MCCMSG_SetRsCompCorrection", "d", "Correction", "Rs compensation", "%", 1.0, VC,
            high=100, step=1),
    Control("rs_prediction", "MCCMSG_SetRsCompPrediction", "d", "Prediction", "Rs compensation", "%", 1.0, VC,
            high=100, step=1),
    Control("rs_bandwidth", "MCCMSG_SetRsCompBandwidth", "d", "Bandwidth", "Rs compensation", "Hz", 1.0, VC,
            low=320, high=16000, step=10),
    Control("bridge_on", "MCCMSG_SetBridgeBalEnable", "b", "Bridge balance on", "Bridge and neutralization",
            modes=IC),
    Control("bridge", "MCCMSG_SetBridgeBalResist", "d", "Bridge balance", "Bridge and neutralization", "MOhm", 1e6,
            IC, high=1000, step=0.1),
    Control("neutralization_on", "MCCMSG_SetNeutralizationEnable", "b", "Neutralization on",
            "Bridge and neutralization", modes=IC),
    Control("neutralization", "MCCMSG_SetNeutralizationCap", "d", "Neutralization", "Bridge and neutralization",
            "pF", 1e-12, IC, low=-10, high=40, step=0.1),
    Control("leak_sub_on", "MCCMSG_SetLeakSubEnable", "b", "Leak subtraction on", "Leak subtraction", modes=VC),
    Control("leak_sub", "MCCMSG_SetLeakSubResist", "d", "Leak resistance", "Leak subtraction", "MOhm", 1e6, VC,
            low=1, high=10000, step=1),
    Control("primary_signal", "MCCMSG_SetPrimarySignal", "u", "Primary output", "Outputs",
            choices=PRIMARY_VC + PRIMARY_IC),
    Control("primary_gain", "MCCMSG_SetPrimarySignalGain", "u", "Primary gain", "Outputs", choices=GAINS),
    Control("primary_lpf", "MCCMSG_SetPrimarySignalLPF", "u", "Primary low-pass", "Outputs", choices=LPF),
    Control("secondary_signal", "MCCMSG_SetSecondarySignal", "u", "Secondary output", "Outputs",
            choices=SECONDARY_VC + SECONDARY_IC),
    Control("secondary_gain", "MCCMSG_SetSecondarySignalGain", "u", "Secondary gain", "Outputs", choices=GAINS),
    Control("secondary_lpf", "MCCMSG_SetSecondarySignalLPF", "u", "Secondary low-pass", "Outputs", choices=LPF),
)
BY_KEY = {c.key: c for c in CONTROLS}
DOUBLE_CHOICES = {"primary_gain", "primary_lpf", "secondary_gain", "secondary_lpf"}   # choices the library takes
# as doubles

ACTIONS = {     # Commander's buttons: (the library's call, the words on Clampwise's button, the modes)
    "auto_pipette_offset": ("MCCMSG_AutoPipetteOffset", "Auto pipette offset", ALL),
    "auto_fast": ("MCCMSG_AutoFastComp", "Auto Cp fast", VC),
    "auto_slow": ("MCCMSG_AutoSlowComp", "Auto Cp slow", VC),
    "auto_whole_cell": ("MCCMSG_AutoWholeCellComp", "Auto whole cell", VC),
    "auto_bridge": ("MCCMSG_AutoBridgeBal", "Auto bridge balance", IC),
    "auto_leak": ("MCCMSG_AutoLeakSub", "Auto leak subtraction", VC),
    "auto_output_zero": ("MCCMSG_AutoOutputZero", "Auto output zero", ALL),
    "zap": ("MCCMSG_Zap", "Zap", VC),
    "buzz": ("MCCMSG_Buzz", "Buzz", ALL),
}
SETTERS = tuple(c.setter for c in CONTROLS) + tuple(a[0] for a in ACTIONS.values())


def to_library(c: Control, mode: int, value) -> Any:
    """A value as shown -> as the library takes it (SI)."""
    if c.kind == "b":
        return bool(value)
    if c.kind == "u" and c.key not in DOUBLE_CHOICES:
        return int(value)
    factor = c.to_si.get(mode, 1.0) if isinstance(c.to_si, dict) else c.to_si
    return float(value) * factor


class Controls:
    """The allowed setters and buttons of Commander's library, bound on an open readout connection
    (``rig.multiclamp.Commander``) and called with its handle, in its thread."""

    def __init__(self, commander) -> None:
        self.commander = commander
        self._fns: dict[str, Any] = {}

    def _bind(self) -> bool:
        if self._fns:
            return True
        dll = self.commander.loaded or self.commander._dll
        if dll is None or self.commander.handle is None:
            return False
        from ctypes import POINTER, c_double, c_int, c_uint, c_void_p
        from ctypes.wintypes import BOOL

        perr = POINTER(c_int)
        arg = {"u": c_uint, "b": BOOL, "d": c_double}
        for c in CONTROLS:
            fn = getattr(dll, c.setter)
            fn.restype = BOOL
            fn.argtypes = [c_void_p, c_double if c.key in DOUBLE_CHOICES else arg[c.kind], perr]
            self._fns[c.setter] = fn
        for name, _, _ in ACTIONS.values():
            fn = getattr(dll, name)
            fn.restype, fn.argtypes = BOOL, [c_void_p, perr]
            self._fns[name] = fn
        return True

    def run(self, command: tuple) -> str:
        """``("set", key, value_as_shown, mode)`` or ``("act", name)``; "" when done, else why not."""
        if not self.commander.open() or not self._bind():
            return self.commander.status or "Commander is not open"
        err = ctypes.c_int(0)
        if command[0] == "set":
            _, key, value, mode = command
            c = BY_KEY[key]
            if mode not in c.modes:
                return f"{c.label} is not a setting of this mode"
            ok = self._fns[c.setter](self.commander.handle, to_library(c, mode, value), ctypes.byref(err))
            what = c.label
        else:
            name, what, _ = ACTIONS[command[1]]
            ok = self._fns[name](self.commander.handle, ctypes.byref(err))
        return "" if ok else f"{what}: {self.commander._error(err.value)}"
