"""The camera itself when it is a Basler camera (an ace, through pylon), without pylon Viewer: the camera used in
Clampwise on a rig whose camera is Basler's.

Only Basler's own pylon library reaches the camera: its C library (``PylonC*.dll``: ``PylonC_v6_1.dll`` in pylon 6,
``PylonC_MD_VC120.dll`` in pylon 5), installed with pylon Viewer under ``C:\\Program Files\\Basler\\pylon
*\\Runtime\\x64`` (a vendor library, as the amplifier's is). It is called through ctypes; nothing is copied or
installed. Only one program can hold the camera, so it is taken only when asked and only while pylon Viewer is
closed, and let go with one call; what Clampwise changed on the camera (the pixel format, its automatic exposure and
gain, its trigger) is put back then, so pylon Viewer finds the camera as it left it.

Frames come from the camera's free-running stream at its fastest frame rate (the exposure alone sets the pace), a
few buffers queued; each picture is the newest frame, so the live picture never lags behind a queue. A monochrome
camera's frames are taken as 8 bits (Mono8) and kept as one grey channel; a colour camera's as RGB when it gives it.
The exposure is in ms here, as for the other camera (the camera counts it in us); the gain in dB, as the camera does.

The declarations are pylon C's (``pylonc/PylonC.h``), the same in pylon 5 and 6 for every call used here. The grab
result's layout differs after its status (pylon 5 had a frame number there), so only its first four fields are read
and the picture's size is the camera's Width and Height.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from ctypes import c_bool, c_char_p, c_double, c_int, c_int32, c_int64, c_size_t, c_uint32, c_void_p
from pathlib import Path

import numpy as np

from .lucam import EXPOSURE, GAIN, Frame, Setting

OWNERS = ("pylonviewer.exe", "pylonviewerapp.exe")     # pylon Viewer (6 and later; 5): it holds the camera when open
ACCESS = 0x1 | 0x2                                     # PYLONC_ACCESS_MODE_CONTROL | PYLONC_ACCESS_MODE_STREAM
GRABBED, FAILED = 2, 4                                 # EPylonGrabStatus
BUFFERS = 8
MONO8, COLOUR = "Mono8", ("RGB8", "RGB8Packed", "BGR8", "BGR8Packed")
WIDE = {"Mono10": 2, "Mono12": 4, "Mono16": 8}         # unpacked 16-bit formats: the shift to 8 bits


class GrabResult(ctypes.Structure):
    # PylonGrabResult_t's first fields (the same in pylon 5 and 6); the rest is room for what follows, not read
    _fields_ = [("Context", c_void_p), ("hBuffer", c_void_p), ("pBuffer", c_void_p), ("Status", c_int),
                ("rest", ctypes.c_ubyte * 160)]


def _version(folder: Path) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", folder.parent.parent.name)) or (0,)


def find_library() -> Path | None:
    """pylon's C library: where ``CLAMPWISE_PYLON_DLL`` says, else in the newest pylon installed, else on PATH."""
    given = os.environ.get("CLAMPWISE_PYLON_DLL")
    if given:
        return Path(given) if Path(given).is_file() else None
    folders: list[Path] = []
    for root in dict.fromkeys((os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files")):
        folders += sorted(Path(root, "Basler").glob("pylon*/Runtime/x64"), key=_version, reverse=True)
    if os.environ.get("PYLON_DEV_DIR"):
        folders.append(Path(os.environ["PYLON_DEV_DIR"]).parent / "Runtime" / "x64")
    folders += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if "pylon" in p.lower()]
    for folder in folders:
        try:
            names = sorted(folder.glob("PylonC*.dll"), reverse=True)
        except OSError:
            continue
        for p in names:
            if re.fullmatch(r"PylonC(_.+)?\.dll", p.name, re.IGNORECASE) and p.is_file():
                return p
    return None


def owner_running() -> bool:
    """Whether pylon Viewer is open (then it may hold the camera, and Clampwise does not try)."""
    from .windows import top_windows

    return any(w.program in OWNERS for w in top_windows())


_DECLARED = (
    ("PylonInitialize", ()), ("PylonTerminate", ()),
    ("PylonEnumerateDevices", (ctypes.POINTER(c_size_t),)),
    ("PylonCreateDeviceByIndex", (c_size_t, ctypes.POINTER(c_void_p))),
    ("PylonDeviceOpen", (c_void_p, c_int)), ("PylonDeviceClose", (c_void_p,)), ("PylonDestroyDevice", (c_void_p,)),
    ("PylonDeviceFeatureFromString", (c_void_p, c_char_p, c_char_p)),
    ("PylonDeviceFeatureToString", (c_void_p, c_char_p, c_char_p, ctypes.POINTER(c_size_t))),
    ("PylonDeviceSetFloatFeature", (c_void_p, c_char_p, c_double)),
    ("PylonDeviceGetFloatFeature", (c_void_p, c_char_p, ctypes.POINTER(c_double))),
    ("PylonDeviceGetFloatFeatureMin", (c_void_p, c_char_p, ctypes.POINTER(c_double))),
    ("PylonDeviceGetFloatFeatureMax", (c_void_p, c_char_p, ctypes.POINTER(c_double))),
    ("PylonDeviceSetIntegerFeature", (c_void_p, c_char_p, c_int64)),
    ("PylonDeviceGetIntegerFeature", (c_void_p, c_char_p, ctypes.POINTER(c_int64))),
    ("PylonDeviceGetIntegerFeatureMin", (c_void_p, c_char_p, ctypes.POINTER(c_int64))),
    ("PylonDeviceGetIntegerFeatureMax", (c_void_p, c_char_p, ctypes.POINTER(c_int64))),
    ("PylonDeviceSetBooleanFeature", (c_void_p, c_char_p, c_bool)),
    ("PylonDeviceExecuteCommandFeature", (c_void_p, c_char_p)),
    ("PylonDeviceGetNumStreamGrabberChannels", (c_void_p, ctypes.POINTER(c_size_t))),
    ("PylonDeviceGetStreamGrabber", (c_void_p, c_size_t, ctypes.POINTER(c_void_p))),
    ("PylonStreamGrabberOpen", (c_void_p,)), ("PylonStreamGrabberClose", (c_void_p,)),
    ("PylonStreamGrabberGetWaitObject", (c_void_p, ctypes.POINTER(c_void_p))),
    ("PylonStreamGrabberSetMaxNumBuffer", (c_void_p, c_size_t)),
    ("PylonStreamGrabberSetMaxBufferSize", (c_void_p, c_size_t)),
    ("PylonStreamGrabberPrepareGrab", (c_void_p,)), ("PylonStreamGrabberFinishGrab", (c_void_p,)),
    ("PylonStreamGrabberRegisterBuffer", (c_void_p, c_void_p, c_size_t, ctypes.POINTER(c_void_p))),
    ("PylonStreamGrabberDeregisterBuffer", (c_void_p, c_void_p)),
    ("PylonStreamGrabberQueueBuffer", (c_void_p, c_void_p, c_void_p)),
    ("PylonStreamGrabberRetrieveResult", (c_void_p, ctypes.POINTER(GrabResult), ctypes.POINTER(c_bool))),
    ("PylonStreamGrabberCancelGrab", (c_void_p,)),
    ("PylonWaitObjectWait", (c_void_p, c_uint32, ctypes.POINTER(c_bool))),
)
_BOOLEAN = ("PylonDeviceFeatureIsAvailable", "PylonDeviceFeatureIsWritable", "PylonDeviceFeatureIsReadable")
_OPTIONAL = ("PylonStreamGrabberFlushBuffersToOutput",)      # pylon 6 and later
_DLL_FOLDERS: list = []


class Camera:
    """The Basler camera, taken on request. ``library``: a stand-in for tests (the real one is loaded on ``open``)."""

    def __init__(self, library=None, *, owner_check=owner_running) -> None:
        self._lib = library
        self._owner_check = owner_check
        self.handle = None
        self.grabber = self.wait_object = None
        self.serial = 0
        self.model = ""
        self.exposure_ms = 20.0
        self.gain = 0.0
        self.gain_kind = "db"                                      # "Gain" in dB; "GainRaw" in the camera's own steps
        self.mono = True
        self.frame_rate = 0.0
        self.status = ""
        self.library_path: Path | None = None
        self._streaming = False
        self._buffers: list = []
        self._handles: list = []
        self._found: dict[str, str] = {}               # what Clampwise changed on the camera, put back on close
        self._format = MONO8
        self._shape = (0, 0)
        self._exposure_name = "ExposureTime"
        self._gain_name = "Gain"
        self._initialized = False

    # ---- pylon's calls
    def _load(self):
        path = find_library()
        if path is None or sys.platform != "win32":
            self.status = ("the camera's library (pylon, installed with pylon Viewer: Basler's PylonC*.dll) is not "
                           "on this PC")
            return None
        self.library_path = path
        try:                                            # its own libraries beside it (GenApi, the transport layers),
            _DLL_FOLDERS.append(os.add_dll_directory(str(path.parent)))      # kept found while Clampwise runs
        except (OSError, AttributeError):
            pass
        try:
            lib = ctypes.WinDLL(str(path))
        except OSError as exc:
            self.status = f"pylon's library ({path}) could not be loaded: {exc}"
            return None
        missing = [n for n in [d[0] for d in _DECLARED] + list(_BOOLEAN) if not hasattr(lib, n)]
        if missing:
            self.status = (f"pylon's library ({path}) lacks {', '.join(missing[:5])}"
                           + (f" and {len(missing) - 5} more" if len(missing) > 5 else ""))
            return None
        for name, args in _DECLARED:
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = c_int32, list(args)
        for name in _BOOLEAN:
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = c_bool, [c_void_p, c_char_p]
        for name in _OPTIONAL:
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.restype, fn.argtypes = c_int32, [c_void_p]
        message = getattr(lib, "GenApiGetLastErrorMessage", None)
        if message is not None:
            message.restype, message.argtypes = c_int32, [c_char_p, ctypes.POINTER(c_size_t)]
        self._lib = lib
        return lib

    def _error(self, code: int) -> str:
        text = ""
        message = getattr(self._lib, "GenApiGetLastErrorMessage", None)
        if message is not None:
            try:
                buf, n = ctypes.create_string_buffer(512), c_size_t(512)
                if message(buf, ctypes.byref(n)) == 0:
                    text = buf.value.decode("utf-8", "replace").strip()
            except Exception:
                text = ""
        return f"error 0x{code & 0xFFFFFFFF:08X}" + (f": {text}" if text else "")

    def _call(self, name: str, *args) -> bool:
        code = getattr(self._lib, name)(*args)
        if code != 0:
            self.status = f"{name} failed ({self._error(code)})"
            return False
        return True

    # ---- the camera's features, by their GenICam names
    def _has(self, name: str, writable: bool = False) -> bool:
        lib, key = self._lib, name.encode()
        if not lib.PylonDeviceFeatureIsAvailable(self.handle, key):
            return False
        return bool(lib.PylonDeviceFeatureIsWritable(self.handle, key)) if writable else \
            bool(lib.PylonDeviceFeatureIsReadable(self.handle, key))

    def _text(self, name: str) -> str:
        if not self._has(name):
            return ""
        buf, n = ctypes.create_string_buffer(256), c_size_t(256)
        if self._lib.PylonDeviceFeatureToString(self.handle, name.encode(), buf, ctypes.byref(n)) != 0:
            return ""
        return buf.value.decode("utf-8", "replace").strip()

    def _put_text(self, name: str, value: str) -> bool:
        """A feature set from text (an enumeration's entry), what it was noted first to be put back on close."""
        if not self._has(name, writable=True):
            return False
        old = self._text(name)
        if old == value:
            return True
        if self._lib.PylonDeviceFeatureFromString(self.handle, name.encode(), value.encode()) != 0:
            return False
        self._found.setdefault(name, old)
        return True

    def _float(self, name: str, which: str = "") -> float | None:
        out = c_double()
        fn = getattr(self._lib, f"PylonDeviceGetFloatFeature{which}")
        return out.value if fn(self.handle, name.encode(), ctypes.byref(out)) == 0 else None

    def _int(self, name: str, which: str = "") -> int | None:
        out = c_int64()
        fn = getattr(self._lib, f"PylonDeviceGetIntegerFeature{which}")
        return out.value if self._has(name) and fn(self.handle, name.encode(), ctypes.byref(out)) == 0 else None

    # ---- taking and letting go
    def open(self, index: int = 1) -> bool:
        if self.handle:
            return True
        if self._owner_check():
            self.status = "the camera's program (pylon Viewer) is open and may hold the camera: close it first"
            return False
        lib = self._lib or self._load()
        if lib is None:
            return False
        if not self._call("PylonInitialize"):
            return False
        self._initialized = True
        n = c_size_t(0)
        if not self._call("PylonEnumerateDevices", ctypes.byref(n)):
            self._terminate()
            return False
        if n.value < index:
            self.status = "no camera is connected (pylon finds none)"
            self._terminate()
            return False
        handle = c_void_p()
        if not self._call("PylonCreateDeviceByIndex", c_size_t(index - 1), ctypes.byref(handle)):
            self._terminate()
            return False
        code = lib.PylonDeviceOpen(handle, ACCESS)
        if code != 0:
            self.status = (f"the camera could not be opened ({self._error(code)}); is another program "
                           "(pylon Viewer) using it?")
            lib.PylonDestroyDevice(handle)
            self._terminate()
            return False
        self.handle = handle
        self.model = self._text("DeviceModelName")
        serial = self._text("DeviceSerialNumber") or self._text("DeviceID")
        self.serial = int(serial) if serial.isdigit() else 0
        self._prepare()
        self.status = ""
        return True

    def _prepare(self) -> None:
        """The camera free-running at its fastest, its exposure and gain as Clampwise sets them, frames 8 bits."""
        self._put_text("AcquisitionMode", "Continuous")
        if self._put_text("TriggerSelector", "FrameStart"):           # a frame each time, not on a trigger
            self._put_text("TriggerMode", "Off")
        self._put_text("ExposureAuto", "Off")
        self._put_text("GainAuto", "Off")
        if self._has("AcquisitionFrameRateEnable", writable=True):
            old = self._text("AcquisitionFrameRateEnable")
            if self._lib.PylonDeviceSetBooleanFeature(self.handle, b"AcquisitionFrameRateEnable", False) == 0 \
                    and old:
                self._found.setdefault("AcquisitionFrameRateEnable", old)
        found = self._text("PixelFormat")
        if found.startswith("Mono") or not found:
            self.mono = True
            self._format = MONO8 if self._put_text("PixelFormat", MONO8) else found
        else:
            self.mono = False
            self._format = next((f for f in COLOUR if self._put_text("PixelFormat", f)), "")
            if not self._format:                                       # no RGB from this camera: its grey
                self.mono = True
                self._format = MONO8 if self._put_text("PixelFormat", MONO8) else found
        self._exposure_name = "ExposureTime" if self._has("ExposureTime") else "ExposureTimeAbs"
        self._gain_name = "Gain" if self._has("Gain") else "GainRaw"
        self.gain_kind = "db" if self._gain_name == "Gain" else "raw"
        exp = self.setting(EXPOSURE)
        if exp is not None:
            self.exposure_ms = exp.value
        g = self.setting(GAIN)
        if g is not None:
            self.gain = g.value

    def close(self) -> None:
        lib = self._lib
        if self.handle and lib is not None:
            self._stop_stream()
            for name, value in reversed(list(self._found.items())):
                if name == "AcquisitionFrameRateEnable":
                    lib.PylonDeviceSetBooleanFeature(self.handle, name.encode(), value.lower() in ("1", "true"))
                elif value:
                    lib.PylonDeviceFeatureFromString(self.handle, name.encode(), value.encode())
            self._found.clear()
            lib.PylonDeviceClose(self.handle)
            lib.PylonDestroyDevice(self.handle)
        self.handle = None
        self._terminate()

    def _terminate(self) -> None:
        if self._initialized and self._lib is not None:
            self._lib.PylonTerminate()
        self._initialized = False

    # ---- settings the camera keeps (exposure, gain): read with their ranges
    def setting(self, prop: int) -> Setting | None:
        if not self.handle:
            return None
        if prop == EXPOSURE:
            name = self._exposure_name
            values = [self._float(name, w) for w in ("", "Min", "Max")]
            if None in values:
                return None
            value, low, high = (v / 1000.0 for v in values)             # us -> ms
            return Setting(value, low, high, value)
        if prop == GAIN:
            name = self._gain_name
            read = self._float if name == "Gain" else self._int
            values = [read(name, w) for w in ("", "Min", "Max")]
            if None in values:
                return None
            value, low, high = (float(v) for v in values)
            return Setting(value, low, high, value)
        return None

    def set_exposure(self, ms: float, gain: float | None = None) -> None:
        """The exposure (ms) and gain, set on the camera at once (the stream goes on with them), each kept inside
        the camera's range."""
        self.exposure_ms = float(ms)
        if gain is not None:
            self.gain = float(gain)
        if not self.handle:
            return
        exp = self.setting(EXPOSURE)
        if exp is not None:
            self.exposure_ms = min(max(self.exposure_ms, exp.low), exp.high)
            self._lib.PylonDeviceSetFloatFeature(self.handle, self._exposure_name.encode(), self.exposure_ms * 1000.0)
        g = self.setting(GAIN)
        if g is not None:
            self.gain = min(max(self.gain, g.low), g.high)
            if self._gain_name == "Gain":
                self._lib.PylonDeviceSetFloatFeature(self.handle, b"Gain", self.gain)
            else:
                self._lib.PylonDeviceSetIntegerFeature(self.handle, self._gain_name.encode(), int(round(self.gain)))

    # ---- the stream
    def _stream(self) -> bool:
        """The stream grabber opened, its buffers queued, the camera's acquisition started."""
        lib = self._lib
        self.set_exposure(self.exposure_ms, self.gain)
        w, h, payload = self._int("Width"), self._int("Height"), self._int("PayloadSize")
        if not w or not h:
            self.status = "the camera did not give its picture's size"
            return False
        self._shape = (int(h), int(w))
        payload = int(payload or w * h * self._depth())
        channels, grabber, wait = c_size_t(0), c_void_p(), c_void_p()
        if not self._call("PylonDeviceGetNumStreamGrabberChannels", self.handle, ctypes.byref(channels)):
            return False
        if channels.value < 1:
            self.status = "the camera has no stream to take pictures from"
            return False
        if not (self._call("PylonDeviceGetStreamGrabber", self.handle, c_size_t(0), ctypes.byref(grabber))
                and self._call("PylonStreamGrabberOpen", grabber)):
            return False
        self.grabber = grabber
        if not (self._call("PylonStreamGrabberGetWaitObject", grabber, ctypes.byref(wait))
                and self._call("PylonStreamGrabberSetMaxNumBuffer", grabber, c_size_t(BUFFERS))
                and self._call("PylonStreamGrabberSetMaxBufferSize", grabber, c_size_t(payload))
                and self._call("PylonStreamGrabberPrepareGrab", grabber)):
            self._stop_stream()
            return False
        self.wait_object = wait
        self._buffers, self._handles = [], []
        for k in range(BUFFERS):
            buf = np.empty(payload, np.uint8)
            hbuf = c_void_p()
            if not self._call("PylonStreamGrabberRegisterBuffer", grabber, c_void_p(buf.ctypes.data),
                              c_size_t(payload), ctypes.byref(hbuf)):
                self._stop_stream()
                return False
            self._buffers.append(buf)
            self._handles.append(hbuf)
        for k, hbuf in enumerate(self._handles):
            if not self._call("PylonStreamGrabberQueueBuffer", grabber, hbuf, c_void_p(k + 1)):   # 0 reads as None
                self._stop_stream()
                return False
        if not self._call("PylonDeviceExecuteCommandFeature", self.handle, b"AcquisitionStart"):
            self._stop_stream()
            return False
        self._streaming = True
        rate = self._float("ResultingFrameRate") if self._has("ResultingFrameRate") else \
            self._float("ResultingFrameRateAbs") if self._has("ResultingFrameRateAbs") else None
        self.frame_rate = float(rate or 0.0)
        return True

    def _stop_stream(self) -> None:
        lib, grabber = self._lib, self.grabber
        if self._streaming and self.handle:
            lib.PylonDeviceExecuteCommandFeature(self.handle, b"AcquisitionStop")
        self._streaming = False
        if grabber is None:
            return
        lib.PylonStreamGrabberCancelGrab(grabber)
        flush = getattr(lib, "PylonStreamGrabberFlushBuffersToOutput", None)
        if flush is not None:
            flush(grabber)
        res, ready = GrabResult(), c_bool(True)
        for _ in range(4 * BUFFERS):                   # the canceled buffers taken back before they are let go
            if lib.PylonStreamGrabberRetrieveResult(grabber, ctypes.byref(res), ctypes.byref(ready)) != 0 or \
                    not ready.value:
                break
        for hbuf in self._handles:
            lib.PylonStreamGrabberDeregisterBuffer(grabber, hbuf)
        if self.wait_object is not None:
            lib.PylonStreamGrabberFinishGrab(grabber)
        lib.PylonStreamGrabberClose(grabber)
        self.grabber = self.wait_object = None
        self._buffers, self._handles = [], []

    def _depth(self) -> int:
        return 3 if self._format in COLOUR else 2 if self._format in WIDE else 1

    def _picture(self, k: int) -> np.ndarray:
        """Buffer ``k``'s frame copied out (the buffer goes back to the camera), as (h, w) grey or (h, w, 3) RGB."""
        h, w = self._shape
        raw = self._buffers[k]
        if self._format in COLOUR:
            rgb = raw[:h * w * 3].reshape(h, w, 3)
            return (rgb[:, :, ::-1] if self._format.startswith("BGR") else rgb).copy()
        if self._format in WIDE:
            return (raw[:h * w * 2].view("<u2").reshape(h, w) >> WIDE[self._format]).astype(np.uint8)
        return raw[:h * w].reshape(h, w).copy()

    def _next(self) -> np.ndarray | None:
        """The newest frame: those waiting already are passed over (back to the camera), else the next is waited
        for (up to three exposures and a second)."""
        lib, grabber = self._lib, self.grabber
        res, ready, newest = GrabResult(), c_bool(False), None
        timeout = int(3 * self.exposure_ms + 1000)
        for _ in range(64):
            got = c_bool(False)
            if newest is None:
                if lib.PylonWaitObjectWait(self.wait_object, c_uint32(timeout), ctypes.byref(got)) != 0 or \
                        not got.value:
                    self.status = f"no frame came within {timeout / 1000:.1f} s"
                    return None
            if not self._call("PylonStreamGrabberRetrieveResult", grabber, ctypes.byref(res), ctypes.byref(ready)):
                return None
            if not ready.value:
                if newest is not None:
                    break
                continue                                # woken with nothing to take: wait again
            k = int(res.Context or 0) - 1
            if not 0 <= k < len(self._handles):
                self.status = "the camera's library gave back a frame Clampwise did not queue"
                return None
            if res.Status == GRABBED:
                newest = self._picture(k)
            lib.PylonStreamGrabberQueueBuffer(grabber, self._handles[k], c_void_p(k + 1))
        if newest is None:
            self.status = "the camera's frames failed (is its USB 3 link, cable or port, enough?)"
        return newest

    # ---- frames
    def frame(self, average: int = 1) -> Frame | None:
        """One picture (``average`` frames averaged: dim fluorescence)."""
        if not self.handle:
            return None
        if not self._streaming and not self._stream():
            return None
        n = max(1, int(average))
        total = None
        for _ in range(n):
            one = self._next()
            if one is None:
                return None
            if n == 1:
                total = one
                break
            total = one.astype(np.uint32) if total is None else total + one
        image = total if n == 1 else ((total + n // 2) // n).astype(np.uint8)
        if image.ndim == 2:
            image = np.broadcast_to(image[:, :, None], image.shape + (3,))   # grey as RGB, not copied
        return Frame(image, self.exposure_ms, self.gain, n, self.serial)
