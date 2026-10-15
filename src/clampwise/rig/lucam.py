"""The camera itself, without Infinity Analyze (Teledyne Lumenera's program): the camera used in Clampwise.

The rig's camera (an INFINITY USB 3.0 camera, Lumenera's) is not a standard video device: only Lumenera's own
library reaches it, ``lucamapi.dll``, installed with Infinity Analyze (a vendor library, as the amplifier's
is). It is called through ctypes; nothing is copied or installed. Only one program can hold
the camera, so it is taken only when asked and only while Infinity Analyze is closed, and let go with one call.

The layouts and numbers are Lumenera's ``lucamapi.h`` as Christoph Gohlke's ``lucam`` (BSD) declares them (Help
credits it; no code copied), each checked against the camera's own answers on the rig PC.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import c_float, c_int, c_ulong, c_ushort
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DLL_NAME = "lucamapi.dll"
FOLDERS = (r"C:\Program Files\Teledyne Lumenera\InfinityAnalyze", r"C:\Program Files\Lumenera Corporation\LuCam")
OWNER = "infinityanalyze.exe"              # the program that holds the camera when it runs

# property numbers (LUCAM_PROP_*)
BRIGHTNESS, CONTRAST, HUE, SATURATION, SHARPNESS, GAMMA = 0, 1, 2, 3, 4, 5
EXPOSURE, GAIN = 20, 40
# the camera's own processing, applied to every frame before it leaves the camera: Infinity Analyze left contrast
# 3.4 and brightness -5 in the rig's camera (2026-10-14), so a frame at 10 ms read exactly 0 everywhere and one at
# 30 ms exactly 255. Neutral while Clampwise holds the camera (its pictures are then the sensor's own); what was
# found is put back when it lets go, so the camera's program finds the camera as it left it.
NEUTRAL = {BRIGHTNESS: 0.0, CONTRAST: 1.0, GAMMA: 1.0, SATURATION: 1.0}
PF_8, PF_16 = 0, 1                          # pixel formats: 8 or 16 bits a pixel, the sensor's raw (Bayer) pattern
DEMOSAIC_HIGH_QUALITY, CORRECTION_NONE = 2, 0


class Version(ctypes.Structure):
    _fields_ = [("firmware", c_ulong), ("fpga", c_ulong), ("api", c_ulong), ("driver", c_ulong),
                ("serialnumber", c_ulong), ("reserved", c_ulong)]


class FrameFormat(ctypes.Structure):
    _fields_ = [("xOffset", c_ulong), ("yOffset", c_ulong), ("width", c_ulong), ("height", c_ulong),
                ("pixelFormat", c_ulong), ("subSampleX", c_ushort), ("flagsX", c_ushort),
                ("subSampleY", c_ushort), ("flagsY", c_ushort)]


class Snapshot(ctypes.Structure):
    _fields_ = [("exposure", c_float), ("gain", c_float), ("gainRed", c_float), ("gainBlue", c_float),
                ("gainGrn1", c_float), ("gainGrn2", c_float), ("useStrobe", c_int), ("strobeDelay", c_float),
                ("useHwTrigger", c_int), ("timeout", c_float), ("format", FrameFormat), ("shutterType", c_ulong),
                ("exposureDelay", c_float), ("bufferlastframe", c_int), ("ulReserved2", c_ulong),
                ("flReserved1", c_float), ("flReserved2", c_float)]


class Conversion(ctypes.Structure):
    _fields_ = [("DemosaicMethod", c_ulong), ("CorrectionMatrix", c_ulong)]


def find_library(folders=FOLDERS) -> Path | None:
    given = os.environ.get("CLAMPWISE_LUCAM_DLL")
    for p in ([Path(given)] if given else []) + [Path(f) / DLL_NAME for f in folders]:
        if p.is_file():
            return p
    return None


def found_path() -> Path:
    """Where the camera's own processing is noted while Clampwise holds it (``%LOCALAPPDATA%\\Clampwise``)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Clampwise" / "camera_found.json"


def owner_running() -> bool:
    """Whether Infinity Analyze is open (then it holds the camera, and Clampwise does not try)."""
    from .windows import top_windows

    return any(w.program == OWNER for w in top_windows())


@dataclass
class Setting:
    value: float
    low: float
    high: float
    default: float


@dataclass
class Frame:
    rgb: np.ndarray                         # (height, width, 3) uint8, the camera's picture as taken
    exposure_ms: float
    gain: float
    averaged: int = 1
    serial: int = 0
    notes: list[str] = field(default_factory=list)


class Camera:
    """The camera, taken on request. ``library``: a stand-in for tests (the real one is loaded on ``open``)."""

    def __init__(self, library=None, *, owner_check=owner_running) -> None:
        self._lib = library
        self._owner_check = owner_check
        self.handle = None
        self.format: FrameFormat | None = None
        self.serial = 0
        self.exposure_ms = 20.0
        self.gain = 1.0
        self._fast = False
        self.status = ""

    # ---- taking and letting go
    def open(self, index: int = 1) -> bool:
        if self.handle:
            return True
        if self._owner_check():
            self.status = "the camera's program is open and holds the camera: close it first"
            return False
        lib = self._lib or self._load()
        if lib is None:
            return False
        if lib.LucamNumCameras() < index:
            self.status = "no camera is connected"
            return False
        handle = lib.LucamCameraOpen(c_ulong(index))
        if not handle:
            self.status = f"the camera could not be opened (error {lib.LucamGetLastError()}); is another program using it?"
            return False
        self.handle = handle
        v = Version()
        if lib.LucamQueryVersion(handle, ctypes.byref(v)):
            self.serial = int(v.serialnumber)
        fmt, rate = FrameFormat(), c_float(0.0)
        lib.LucamGetFormat(handle, ctypes.byref(fmt), ctypes.byref(rate))
        self.format = fmt
        self._neutral()
        exp = self.setting(EXPOSURE)
        if exp is not None:
            self.exposure_ms = exp.value
        g = self.setting(GAIN)
        if g is not None:
            self.gain = g.value
        self.status = ""
        return True

    def close(self) -> None:
        if self.handle and self._lib is not None:
            if self._fast:
                self._lib.LucamDisableFastFrames(self.handle)
            self._put_back()
            self._lib.LucamCameraClose(self.handle)
        self.handle, self._fast = None, False

    def _neutral(self) -> None:
        """Note the camera's own processing (in a file too, should Clampwise stop before letting go), then make it
        neutral. A note left by an earlier session that did not let go is what the camera had before it."""
        found = {}
        note = found_path()
        if note.is_file():
            try:
                found = {int(k): float(v) for k, v in json.loads(note.read_text(encoding="utf-8")).items()}
            except (ValueError, OSError):
                found = {}
        for prop in NEUTRAL:
            s = self.setting(prop)
            if s is not None:
                found.setdefault(prop, s.value)
        self.found = found
        try:
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(json.dumps(found), encoding="utf-8")
        except OSError:
            pass
        for prop, value in NEUTRAL.items():
            if prop in found:
                self._lib.LucamSetProperty(self.handle, c_ulong(prop), c_float(value), ctypes.c_long(0))

    def _put_back(self) -> None:
        for prop, value in getattr(self, "found", {}).items():
            self._lib.LucamSetProperty(self.handle, c_ulong(prop), c_float(value), ctypes.c_long(0))
        try:
            found_path().unlink()
        except OSError:
            pass

    def _load(self):
        path = find_library()
        if path is None or sys.platform != "win32":
            self.status = "the camera's library (lucamapi.dll, installed with Infinity Analyze) is not on this PC"
            return None
        lib = ctypes.WinDLL(str(path))
        p = ctypes.POINTER
        for name, res, args in (
                ("LucamNumCameras", ctypes.c_long, ()), ("LucamCameraOpen", ctypes.c_void_p, (c_ulong,)),
                ("LucamCameraClose", c_int, (ctypes.c_void_p,)),
                ("LucamQueryVersion", c_int, (ctypes.c_void_p, p(Version))),
                ("LucamGetFormat", c_int, (ctypes.c_void_p, p(FrameFormat), p(c_float))),
                ("LucamGetProperty", c_int, (ctypes.c_void_p, c_ulong, p(c_float), p(ctypes.c_long))),
                ("LucamSetProperty", c_int, (ctypes.c_void_p, c_ulong, c_float, ctypes.c_long)),
                ("LucamPropertyRange", c_int, (ctypes.c_void_p, c_ulong, p(c_float), p(c_float), p(c_float),
                                               p(ctypes.c_long))),
                ("LucamEnableFastFrames", c_int, (ctypes.c_void_p, p(Snapshot))),
                ("LucamTakeFastFrame", c_int, (ctypes.c_void_p, ctypes.c_void_p)),
                ("LucamDisableFastFrames", c_int, (ctypes.c_void_p,)),
                ("LucamConvertFrameToRgb24", c_int, (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_ulong,
                                                     c_ulong, c_ulong, p(Conversion))),
                ("LucamGetLastError", c_ulong, ())):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, list(args)
        self._lib = lib
        return lib

    # ---- settings the camera keeps (exposure, gain): read with their ranges
    def setting(self, prop: int) -> Setting | None:
        if not self.handle:
            return None
        lib = self._lib
        value, flags = c_float(), ctypes.c_long()
        low, high, default = c_float(), c_float(), c_float()
        if not lib.LucamGetProperty(self.handle, c_ulong(prop), ctypes.byref(value), ctypes.byref(flags)):
            return None
        if not lib.LucamPropertyRange(self.handle, c_ulong(prop), ctypes.byref(low), ctypes.byref(high),
                                      ctypes.byref(default), ctypes.byref(flags)):
            return Setting(value.value, value.value, value.value, value.value)
        return Setting(value.value, low.value, high.value, default.value)

    def set_exposure(self, ms: float, gain: float | None = None) -> None:
        """The next frames' exposure (ms) and gain; fast frames are begun again with them."""
        self.exposure_ms = float(ms)
        if gain is not None:
            self.gain = float(gain)
        if self._fast and self.handle:
            self._lib.LucamDisableFastFrames(self.handle)
            self._fast = False

    # ---- frames
    def _shape(self) -> tuple[int, int, int]:
        f = self.format
        w = f.width // max(1, f.subSampleX)
        h = f.height // max(1, f.subSampleY)
        return int(w), int(h), (2 if f.pixelFormat == PF_16 else 1)

    def frame(self, average: int = 1) -> Frame | None:
        """One picture (``average`` frames averaged: dim fluorescence), demosaiced to colour by the library."""
        if not self.handle:
            return None
        lib = self._lib
        w, h, depth = self._shape()
        if not self._fast:
            snap = Snapshot(exposure=self.exposure_ms, gain=self.gain, gainRed=1.0, gainBlue=1.0, gainGrn1=1.0,
                            gainGrn2=1.0, timeout=max(1000.0, self.exposure_ms * 4 + 1000.0), format=self.format)
            if not lib.LucamEnableFastFrames(self.handle, ctypes.byref(snap)):
                self.status = f"the camera did not start (error {lib.LucamGetLastError()})"
                return None
            self._fast = True
        raw = (ctypes.c_ubyte * (w * h * depth))()
        rgb = (ctypes.c_ubyte * (w * h * 3))()
        conv = Conversion(DEMOSAIC_HIGH_QUALITY, CORRECTION_NONE)
        total = None
        n = max(1, int(average))
        for _ in range(n):
            if not lib.LucamTakeFastFrame(self.handle, raw):
                self.status = f"no frame came (error {lib.LucamGetLastError()})"
                return None
            lib.LucamConvertFrameToRgb24(self.handle, rgb, raw, c_ulong(w), c_ulong(h),
                                         c_ulong(self.format.pixelFormat), ctypes.byref(conv))
            one = np.frombuffer(rgb, np.uint8).reshape(h, w, 3)[:, :, ::-1].astype(np.float32)   # BGR -> RGB
            total = one if total is None else total + one
        image = np.clip(np.round(total / n), 0, 255).astype(np.uint8)
        return Frame(image, self.exposure_ms, self.gain, n, self.serial)
