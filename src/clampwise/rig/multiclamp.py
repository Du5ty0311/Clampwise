"""The amplifier, live, through MultiClamp 700B Commander (read only).

Commander is the amplifier's driver: it alone talks to the MultiClamp over USB, and another program asks it
through its message library, ``AxMultiClampMsg.dll`` (Commander's ``3rd Party Support\\AxMultiClampMsg`` folder;
64-bit, like this Python). Each call is a Windows message to Commander's hidden window (class
``MultiClampMessageHandlerClass``), answered within the library's 3 s timeout, so Commander must be open (MultiClamp
700B Commander Software Application Note, April 2016). This gives Clampwise its own readout of the amplifier,
before and between recordings, with no file in between.

**Read only, by construction** (this readout never changes the amplifier while a cell is held; the controls are
``amplifier_control``'s). The
library is reached only through :data:`LIFECYCLE` and the getters of :data:`READINGS`, fixed lists of its function
names; no other name is ever looked up in it, so no Set, Auto, Zap, Buzz, Pulse, Reset, button or window call can
be made from here. The tests check the lists and this module's source.

The library answers in SI units (V, A, F, Ohm, s, Hz); the readout is in the lab's (mV, pA, pF, MOhm, us).
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Commander's own hidden window, there while it runs (Commander 2.2.2.2 on the rig PC, 2026-10-14; the library's
# "MultiClampMessageHandlerClass" window is made by the library inside the reading program, never by Commander)
STARTUP_WINDOW = "Invisible MultiClamp Startup Window"
API_VERSION = "1.0.0.9"                              # the header's MCCMSG_APIVERSION_STR
DLL_NAME = "AxMultiClampMsg.dll"
FOLDERS = (r"C:\Program Files\Molecular Devices\MultiClamp 700B Commander\3rd Party Support\AxMultiClampMsg",
           r"C:\Program Files (x86)\Molecular Devices\MultiClamp 700B Commander\3rd Party Support\AxMultiClampMsg")

ERRORS = {6001: "out of memory", 6002: "MultiClamp Commander is not open", 6003: "invalid handle",
          6004: "invalid parameter", 6005: "Commander did not answer in time", 6006: "Commander could not do it"}
NOT_OPEN, TIMED_OUT = 6002, 6005
MODES = {0: "voltage clamp", 1: "current clamp", 2: "I = 0"}
MODELS = {0: "MultiClamp 700A", 1: "MultiClamp 700B"}
PRIMARY = {0: "membrane current", 1: "membrane potential", 2: "pipette potential", 3: "100x AC membrane potential",
           4: "external command potential", 5: "auxiliary 1", 6: "auxiliary 2", 7: "membrane potential",
           8: "membrane current", 9: "command current", 10: "100x AC membrane potential",
           11: "external command current", 12: "auxiliary 1", 13: "auxiliary 2"}
SECONDARY = {**PRIMARY, 9: "command current", 10: "pipette potential", 11: "100x AC membrane potential",
             12: "external command current", 13: "auxiliary 1", 14: "auxiliary 2"}

# The library's lifecycle and selection calls (none of them changes the amplifier).
LIFECYCLE = ("MCCMSG_CheckAPIVersion", "MCCMSG_CreateObject", "MCCMSG_DestroyObject", "MCCMSG_FindFirstMultiClamp",
             "MCCMSG_FindNextMultiClamp", "MCCMSG_SelectMultiClamp", "MCCMSG_BuildErrorText")


@dataclass(frozen=True)
class Reading:
    key: str
    getter: str               # the library's function (always MCCMSG_Get...)
    kind: str                 # "u" UINT, "b" BOOL, "d" double
    label: str
    unit: str = ""
    scale: Any = 1.0          # SI -> the unit shown; a dict {mode: scale} where the unit follows the mode
    modes: tuple = (0, 1, 2)  # the modes the getter supports (calling others only fails)


VC, IC = (0,), (1, 2)
READINGS = (
    Reading("mode", "MCCMSG_GetMode", "u", "Mode"),
    Reading("holding_on", "MCCMSG_GetHoldingEnable", "b", "Holding on"),
    Reading("holding", "MCCMSG_GetHolding", "d", "Holding", {0: "mV", 1: "pA", 2: "pA"}, {0: 1e3, 1: 1e12, 2: 1e12}),
    Reading("primary_signal", "MCCMSG_GetPrimarySignal", "u", "Primary output"),
    Reading("primary_gain", "MCCMSG_GetPrimarySignalGain", "d", "Primary gain", "x"),
    Reading("primary_lpf", "MCCMSG_GetPrimarySignalLPF", "d", "Primary low-pass", "Hz"),
    Reading("primary_hpf", "MCCMSG_GetPrimarySignalHPF", "d", "Primary high-pass", "Hz"),
    Reading("secondary_signal", "MCCMSG_GetSecondarySignal", "u", "Secondary output"),
    Reading("secondary_gain", "MCCMSG_GetSecondarySignalGain", "d", "Secondary gain", "x"),
    Reading("secondary_lpf", "MCCMSG_GetSecondarySignalLPF", "d", "Secondary low-pass", "Hz"),
    Reading("pipette_offset", "MCCMSG_GetPipetteOffset", "d", "Pipette offset", "mV", 1e3),
    Reading("test_signal_on", "MCCMSG_GetTestSignalEnable", "b", "Seal test on"),
    Reading("test_signal_amplitude", "MCCMSG_GetTestSignalAmplitude", "d", "Seal test amplitude",
            {0: "mV", 1: "pA", 2: "pA"}, {0: 1e3, 1: 1e12, 2: 1e12}),
    Reading("test_signal_hz", "MCCMSG_GetTestSignalFrequency", "d", "Seal test frequency", "Hz"),
    # voltage clamp: whole-cell compensation (the amplifier's estimate of Cm and Rs), Rs compensation, pipette
    Reading("whole_cell_on", "MCCMSG_GetWholeCellCompEnable", "b", "Whole-cell compensation on", modes=VC),
    Reading("cm", "MCCMSG_GetWholeCellCompCap", "d", "Cm (whole-cell compensation)", "pF", 1e12, VC),
    Reading("rs", "MCCMSG_GetWholeCellCompResist", "d", "Rs (whole-cell compensation)", "MOhm", 1e-6, VC),
    Reading("rs_comp_on", "MCCMSG_GetRsCompEnable", "b", "Rs compensation on", modes=VC),
    Reading("rs_correction", "MCCMSG_GetRsCompCorrection", "d", "Rs correction", "%", 1.0, VC),
    Reading("rs_prediction", "MCCMSG_GetRsCompPrediction", "d", "Rs prediction", "%", 1.0, VC),
    Reading("rs_bandwidth", "MCCMSG_GetRsCompBandwidth", "d", "Rs compensation bandwidth", "Hz", 1.0, VC),
    Reading("fast_cap", "MCCMSG_GetFastCompCap", "d", "Pipette capacitance, fast", "pF", 1e12, VC),
    Reading("fast_tau", "MCCMSG_GetFastCompTau", "d", "Pipette tau, fast", "us", 1e6, VC),
    Reading("slow_cap", "MCCMSG_GetSlowCompCap", "d", "Pipette capacitance, slow", "pF", 1e12, VC),
    Reading("slow_tau", "MCCMSG_GetSlowCompTau", "d", "Pipette tau, slow", "us", 1e6, VC),
    Reading("leak_sub_on", "MCCMSG_GetLeakSubEnable", "b", "Leak subtraction on", modes=VC),
    Reading("leak_sub", "MCCMSG_GetLeakSubResist", "d", "Leak subtraction", "MOhm", 1e-6, VC),
    # current clamp: bridge balance, capacitance neutralization, slow current injection
    Reading("bridge_on", "MCCMSG_GetBridgeBalEnable", "b", "Bridge balance on", modes=IC),
    Reading("bridge", "MCCMSG_GetBridgeBalResist", "d", "Bridge balance", "MOhm", 1e-6, IC),
    Reading("neutralization_on", "MCCMSG_GetNeutralizationEnable", "b", "Capacitance neutralization on", modes=IC),
    Reading("neutralization", "MCCMSG_GetNeutralizationCap", "d", "Capacitance neutralization", "pF", 1e12, IC),
    Reading("slow_injection_on", "MCCMSG_GetSlowCurrentInjEnable", "b", "Slow current injection on", modes=IC),
    # the meters, as Commander shows them (what each shows depends on the meter settings: kept as read)
    Reading("meter_resist_on", "MCCMSG_GetMeterResistEnable", "b", "Resistance meter on", modes=VC),
    Reading("meter_irms_on", "MCCMSG_GetMeterIrmsEnable", "b", "Irms meter on", modes=VC),
)
METERS = (0, 1, 2, 3)          # MCCMSG_METER1 .. METER4: channel 1 V and I, channel 2 V and I, in V and A (meter_words)
METER_GETTER = "MCCMSG_GetMeterValue"
GETTERS = tuple(dict.fromkeys(r.getter for r in READINGS)) + (METER_GETTER,)
FORBIDDEN = ("Set", "Auto", "Zap", "Buzz", "Pulse", "Reset", "QuickSelect", "Toggle", "Clear")


@dataclass(frozen=True)
class Device:
    """One amplifier channel (a headstage) that Commander controls."""

    model: int
    serial: str
    com_port: int
    device_id: int
    channel: int

    def label(self) -> str:
        serial = f", serial {self.serial}" if self.serial else ""
        return f"{MODELS.get(self.model, 'MultiClamp')}{serial}, channel {self.channel}"


@dataclass
class AmplifierState:
    """What the amplifier is doing, read at ``read_at`` (time.time())."""

    device: Device | None = None
    connected: bool = False
    status: str = ""                       # why it is not connected, or what went wrong
    mode: int | None = None
    values: dict[str, Any] = field(default_factory=dict)      # Reading.key -> value in the unit shown
    units: dict[str, str] = field(default_factory=dict)
    meters: dict[int, float] = field(default_factory=dict)     # meter -> value as read (SI)
    errors: dict[str, str] = field(default_factory=dict)       # Reading.key -> why it could not be read
    raw: dict[str, Any] = field(default_factory=dict)          # Reading.key -> as the library gave it (a signal's id)
    read_at: float = math.nan

    @property
    def mode_name(self) -> str:
        return MODES.get(self.mode, "") if self.mode is not None else ""

    def shown(self, key: str) -> str:
        """One value as the screen shows it: a number with its unit, a word, or a dash."""
        v = self.values.get(key)
        if v is None:
            return "–"
        if isinstance(v, bool):
            return "on" if v else "off"
        if isinstance(v, str):
            return v
        unit = self.units.get(key, "")
        digits = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
        return f"{v:.{digits}f} {unit}".strip()


def find_library(folders=FOLDERS) -> Path | None:
    """Where Commander's message library is (``CLAMPWISE_MCC_DLL`` names another)."""
    given = os.environ.get("CLAMPWISE_MCC_DLL")
    for p in ([Path(given)] if given else []) + [Path(f) / DLL_NAME for f in folders]:
        if p.is_file():
            return p
    return None


def commander_open() -> bool:
    """Whether Commander's own window exists (asked before any call, so a closed Commander costs nothing)."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.FindWindowW(None, STARTUP_WINDOW))


def meter_words(state: AmplifierState) -> str:
    """The four meters as Commander shows them: each channel's potential (mV) and current (pA). The selected
    channel's current meter shows resistance or Irms when that metering is on: then it is given as read."""
    other = state.values.get("meter_resist_on") or state.values.get("meter_irms_on")
    selected = state.device.channel if state.device is not None else 1
    parts = []
    for channel, (v, i) in ((1, (0, 1)), (2, (2, 3))):
        words = []
        if v in state.meters:
            words.append(f"{state.meters[v] * 1e3:.1f} mV")
        if i in state.meters:
            words.append(f"{state.meters[i]:.3g} (resistance or Irms metering on: as read)" if other and
                         channel == selected else f"{state.meters[i] * 1e12:.1f} pA")
        if words:
            parts.append(f"Channel {channel}: " + ", ".join(words))
    return "; ".join(parts)


class _Library:
    """The allowed functions of the library, bound by name and typed; nothing else is ever looked up in it."""

    def __init__(self, dll) -> None:
        from ctypes import POINTER, c_char_p, c_double, c_int, c_uint, c_void_p
        from ctypes.wintypes import BOOL

        out = {"u": c_uint, "b": BOOL, "d": c_double}
        self._fns: dict[str, Any] = {}

        def bind(name, restype, *argtypes):
            fn = getattr(dll, name)
            fn.restype, fn.argtypes = restype, list(argtypes)
            self._fns[name] = fn

        perr = POINTER(c_int)
        bind("MCCMSG_CheckAPIVersion", BOOL, c_char_p)
        bind("MCCMSG_CreateObject", c_void_p, perr)
        bind("MCCMSG_DestroyObject", None, c_void_p)
        found = (c_void_p, POINTER(c_uint), c_char_p, c_uint, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), perr)
        bind("MCCMSG_FindFirstMultiClamp", BOOL, *found)
        bind("MCCMSG_FindNextMultiClamp", BOOL, *found)
        bind("MCCMSG_SelectMultiClamp", BOOL, c_void_p, c_uint, c_char_p, c_uint, c_uint, c_uint, perr)
        bind("MCCMSG_BuildErrorText", BOOL, c_void_p, c_int, c_char_p, c_uint)
        for r in READINGS:
            if r.getter not in self._fns:
                bind(r.getter, BOOL, c_void_p, POINTER(out[r.kind]), perr)
        bind(METER_GETTER, BOOL, c_void_p, POINTER(c_double), c_uint, perr)
        self.types = out

    def __call__(self, name: str, *args):
        return self._fns[name](*args)              # a name outside the lists is a KeyError, never a call

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._fns)


class Commander:
    """A read-only connection to Commander. Open, read, close in the same thread (a background one)."""

    def __init__(self, library=None, *, window_check=commander_open) -> None:
        self._dll = library                         # a loaded library (tests pass a fake); else found and loaded
        self._window_check = window_check
        self._lib: _Library | None = None
        self._handle = None
        self.loaded = None
        self.device: Device | None = None
        self.status = ""

    @property
    def handle(self):
        return self._handle

    # ---- opening
    def open(self) -> bool:
        if self._handle is not None:
            return True
        if not self._window_check():
            self.status = ERRORS[NOT_OPEN]
            return False
        dll = self._dll
        if dll is None:
            path = find_library()
            if path is None:
                self.status = "Commander's message library (AxMultiClampMsg.dll) is not on this PC"
                return False
            try:
                dll = ctypes.WinDLL(str(path))
            except OSError as exc:
                self.status = f"Commander's message library could not be loaded ({exc})"
                return False
        self.loaded = dll                           # rig.amplifier_control binds its own calls here, when allowed
        self._lib = _Library(dll)
        if not self._lib("MCCMSG_CheckAPIVersion", API_VERSION.encode()):
            self.status = f"Commander's message library is not version {API_VERSION}; its readings may differ"
        err = ctypes.c_int(0)
        handle = self._lib("MCCMSG_CreateObject", ctypes.byref(err))
        if not handle:
            self.status = self._error(err.value)
            return False
        self._handle = handle
        return True

    def close(self) -> None:
        if self._handle is not None and self._lib is not None:
            self._lib("MCCMSG_DestroyObject", self._handle)
        self._handle = None

    def __enter__(self) -> "Commander":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- devices
    def devices(self) -> list[Device]:
        """Every amplifier channel Commander controls (a 700B with two headstages is two)."""
        if not self.open():
            return []
        out = []
        for name in ("MCCMSG_FindFirstMultiClamp",) + ("MCCMSG_FindNextMultiClamp",) * 16:
            model, port, dev, chan = (ctypes.c_uint(0) for _ in range(4))
            serial = ctypes.create_string_buffer(16)
            err = ctypes.c_int(0)
            ok = self._lib(name, self._handle, ctypes.byref(model), serial, ctypes.c_uint(16), ctypes.byref(port),
                           ctypes.byref(dev), ctypes.byref(chan), ctypes.byref(err))
            if not ok:
                break
            d = Device(model.value, serial.value.decode("ascii", "replace").strip(), port.value, dev.value, chan.value)
            if d in out:
                break
            out.append(d)
        return out

    def select(self, device: Device) -> bool:
        """Which channel the readings come from (a choice in this program; the amplifier is not touched)."""
        if not self.open():
            return False
        err = ctypes.c_int(0)
        ok = self._lib("MCCMSG_SelectMultiClamp", self._handle, ctypes.c_uint(device.model), device.serial.encode(),
                       ctypes.c_uint(device.com_port), ctypes.c_uint(device.device_id), ctypes.c_uint(device.channel),
                       ctypes.byref(err))
        if ok:
            self.device = device
        else:
            self.status = self._error(err.value)
        return bool(ok)

    # ---- reading
    def read(self, *, meters: bool = True) -> AmplifierState:
        """Every reading the mode supports, and the meters."""
        state = AmplifierState(device=self.device, read_at=time.time())
        if not self._window_check():
            self.close()
            state.status = ERRORS[NOT_OPEN]
            return state
        if not self.open():
            state.status = self.status
            return state
        if self.device is None:
            found = self.devices()
            if not found:
                state.status = "Commander is open but lists no amplifier"
                return state
            self.select(found[0])
            state.device = self.device
        mode, why = self._get(READINGS[0])
        if why:
            state.status = why
            return state
        state.connected, state.mode = True, int(mode)
        for r in READINGS:
            if state.mode not in r.modes:
                continue
            value, why = (mode, "") if r is READINGS[0] else self._get(r)
            if why:
                state.errors[r.key] = why
                if why == ERRORS[TIMED_OUT]:            # Commander is busy: stop here rather than wait 3 s a call
                    state.status = why
                    break
                continue
            state.values[r.key], state.units[r.key] = self._shown(r, value, state.mode)
            state.raw[r.key] = value
        if meters and not state.status:
            for m in METERS:
                value = ctypes.c_double(math.nan)
                err = ctypes.c_int(0)
                if self._lib(METER_GETTER, self._handle, ctypes.byref(value), ctypes.c_uint(m), ctypes.byref(err)):
                    state.meters[m] = float(value.value)
        return state

    def _get(self, r: Reading) -> tuple[Any, str]:
        value = self._lib.types[r.kind]()
        err = ctypes.c_int(0)
        if self._lib(r.getter, self._handle, ctypes.byref(value), ctypes.byref(err)):
            return value.value, ""
        return None, self._error(err.value)

    @staticmethod
    def _shown(r: Reading, value, mode: int) -> tuple[Any, str]:
        unit = r.unit.get(mode, "") if isinstance(r.unit, dict) else r.unit
        if r.kind == "b":
            return bool(value), ""
        if r.key == "mode":
            return MODES.get(int(value), f"mode {value}"), ""
        if r.key == "primary_signal":
            return PRIMARY.get(int(value), f"signal {value}"), ""
        if r.key == "secondary_signal":
            return SECONDARY.get(int(value), f"signal {value}"), ""
        scale = r.scale.get(mode, 1.0) if isinstance(r.scale, dict) else r.scale
        return float(value) * scale, unit

    def _error(self, code: int) -> str:
        if code in ERRORS:
            return ERRORS[code]
        if self._lib is not None and self._handle is not None:
            buf = ctypes.create_string_buffer(256)
            if self._lib("MCCMSG_BuildErrorText", self._handle, ctypes.c_int(code), buf, ctypes.c_uint(256)):
                return buf.value.decode("ascii", "replace") or f"error {code}"
        return f"error {code}"
