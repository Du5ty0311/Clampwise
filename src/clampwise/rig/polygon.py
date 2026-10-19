"""The Polygon (Mightex's patterned illuminator) driven directly, through the SDK PolyScan installs.

``MT_Polygon1000_SDK.dll`` (64-bit, beside PolyScan) is called with ctypes as PolyScan's own code calls it: the
devices counted with ``MTPLG_InitDevice(-1)``, the first connected as device 1 (devices are numbered from 1), the
display set to a static image (mode 0) and each pattern sent as one byte a mirror, 912 x 1140, row by row
(``MTPLG_SetDevStaticImageFromMemory``). The Polygon only steers light: the LED behind it is switched by its own
driver (from Clampex, as the recording's protocol says), so a pattern shown is lit only while that LED is on.

Only one program can hold the Polygon: it is not taken while PolyScan runs, and it is let go when asked, so
PolyScan can have it again. The device's module number holds its serial, which is never shown or kept.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from . import patterns

SDK_NAME = "MT_Polygon1000_SDK.dll"
SDK_FOLDERS = (r"C:\Program Files\Mightex\PolyScan", r"C:\Program Files\Mightex\PolyScan2",
               r"C:\Program Files\Mightex\PolyScan3", r"C:\Program Files\Mightex\PolyScan4")
DISPLAY_STATIC = 0
PIXEL_ROW_MAJOR = 0
ERRORS = {-1: "critical error", -2: "out of range", -3: "not connected", -4: "timed out", -5: "wrong device",
          -6: "invalid", -7: "out of memory", -8: "file name", -9: "bad data", -10: "wrong state", -13: "warning"}


DISPLAY_SEQUENCE = 1
TRIGGER_COMMAND, TRIGGER_AUTO, TRIGGER_EXTERNAL = 0, 1, 2
EDGE_RISE, EDGE_FALL, EDGE_ANY = 1, 2, 3


class PatternSetting(ctypes.Structure):
    """A sequence's settings, as PolyScan's code passes them (by value): the bits a mirror, the patterns' number,
    what steps it on (2: a TTL at Trig In), the delay after it, how long each pattern is shown (us), which edge."""
    _fields_ = [("BitDepth", ctypes.c_int), ("PtnNumber", ctypes.c_int), ("Reserved1", ctypes.c_int),
                ("TrigType", ctypes.c_int), ("TrigDelay", ctypes.c_int), ("Reserved2", ctypes.c_int),
                ("ExposureTime", ctypes.c_uint), ("Reserved3", ctypes.c_int), ("TrigEdge", ctypes.c_int)]


class DeviceState(ctypes.Structure):
    _fields_ = [("Temperature", ctypes.c_int), ("State", ctypes.c_int), ("SeqIdx", ctypes.c_int)] + \
               [(f"Reserved{i}", ctypes.c_int) for i in range(1, 10)]


def _version(path: Path) -> tuple[int, ...]:
    """A DLL's file version (2.7.0.0 for PolyScan4's SDK), or () when it has none."""
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return ()
        buf = ctypes.create_string_buffer(size)
        ctypes.windll.version.GetFileVersionInfoW(str(path), 0, size, buf)
        ptr, n = ctypes.c_void_p(), ctypes.c_uint()
        if not ctypes.windll.version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(n)):
            return ()
        ms, ls = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32 * 4)).contents[2:4]
        return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    except (AttributeError, OSError):
        return ()


def find_sdk(folders=SDK_FOLDERS) -> Path | None:
    """The Polygon's SDK as PolyScan installed it: the newest copy (by its version, then its date), or None.
    PolyScan4 installs into Mightex\\PolyScan (its program is PolyScan3.exe) with SDK 2.7; PolyScan2's SDK 2.2 stays
    beside it and found no Polygon on the rig, after 16 s."""
    found = [Path(f) / SDK_NAME for f in folders if (Path(f) / SDK_NAME).is_file()]
    return max(found, key=lambda p: (_version(p), p.stat().st_mtime)) if found else None


def _why(code: int) -> str:
    return ERRORS.get(int(code), f"error {code}")


class Polygon:
    """The first Polygon found. ``lib`` stands in for the SDK in tests (its functions as attributes)."""

    def __init__(self, lib=None, polyscan_running=None) -> None:
        self._lib = lib
        self._polyscan_running = polyscan_running
        self.index = 1
        self.connected = False
        self.status = "not connected"
        self.model = ""
        self.shown: str | None = None              # the name of the step shown, None when nothing is
        self.sequence = False                      # stepping through a sequence on its trigger

    def _load(self):
        if self._lib is None:
            path = find_sdk()
            if path is None:
                raise OSError("its driver was not found (its own program installs it)")
            os.add_dll_directory(str(path.parent))
            lib = ctypes.CDLL(str(path))
            for name in ("MTPLG_InitDevice", "MTPLG_ConnectDev", "MTPLG_DisconnectDev", "MTPLG_UnInitDevice",
                         "MTPLG_SetDevDisplayMode", "MTPLG_SetDevStaticImageFromMemory", "MTPLG_GetDevModuleNo",
                         "MTPLG_GetDevRunStatus", "MTPLG_SetDevPtnSetting", "MTPLG_SetDevPtnFromMemory",
                         "MTPLG_StartPattern", "MTPLG_StopPattern"):
                getattr(lib, name).restype = ctypes.c_int
            lib.MTPLG_SetDevPtnSetting.argtypes = [ctypes.c_int, PatternSetting]
            self._lib = lib
        return self._lib

    def open(self) -> bool:
        """Takes the Polygon (and sets it to show still pictures). False, with ``status`` saying why, when it cannot."""
        if self.connected:
            return True
        running = self._polyscan_running
        if running is None:
            from .polyscan import program_running

            running = program_running
        if running():
            self.status = "its own program holds it: close that program first"
            return False
        try:
            lib = self._load()
        except OSError as e:
            self.status = str(e)
            return False
        n = lib.MTPLG_InitDevice(-1)
        if n < 1:
            lib.MTPLG_UnInitDevice()
            self.status = "none found (is it on and plugged in?)"
            return False
        r = lib.MTPLG_ConnectDev(self.index)
        if r != 0:
            lib.MTPLG_UnInitDevice()
            self.status = f"it did not connect ({_why(r)})"
            return False
        buf = ctypes.create_string_buffer(64)
        if lib.MTPLG_GetDevModuleNo(self.index, buf) == 0:
            text = buf.value.decode("latin-1", "replace")
            self.model = "Polygon1000" + (f" ({text[:4]})" if text[:4].isalnum() else "")   # never the serial
        r = lib.MTPLG_SetDevDisplayMode(self.index, DISPLAY_STATIC)
        if r != 0:
            self.close()
            self.status = f"it would not show still pictures ({_why(r)})"
            return False
        self.connected = True
        self.status = "held by Clampwise"
        return True

    def send(self, image: np.ndarray) -> bool:
        """One image of mirrors (uint8, 1140 rows x 912 columns) shown at once."""
        if not self.connected:
            self.status = "not connected"
            return False
        image = np.ascontiguousarray(image, np.uint8)
        if image.shape != (patterns.ROWS, patterns.COLUMNS):
            raise ValueError(f"the Polygon takes {patterns.ROWS} x {patterns.COLUMNS} images, not {image.shape}")
        r = self._lib.MTPLG_SetDevStaticImageFromMemory(self.index, image.ctypes.data_as(ctypes.c_void_p),
                                                        PIXEL_ROW_MAJOR)
        if r != 0:
            self.status = f"the pattern was not taken ({_why(r)})"
            return False
        return True

    def show(self, step: patterns.Step, on_value: int = 255, mask: np.ndarray | None = None) -> bool:
        """``step`` shown (``mask``: its mirrors, when already worked out)."""
        if self.send(patterns.image(step.mirrors() if mask is None else mask, on_value)):
            self.shown = step.name
            self.status = f"showing {step.name}"
            return True
        return False

    def off(self, on_value: int = 255) -> bool:
        """Every mirror away from the slice: nothing lit even with the LED on (a sequence stopped first)."""
        if self.sequence:
            return self.stop_sequence(on_value)
        if self.send(patterns.image(np.zeros((patterns.ROWS, patterns.COLUMNS), bool), on_value)):
            self.shown = None
            self.status = "off: no mirror lit"
            return True
        return False

    def run_sequence(self, masks: list[np.ndarray], on_value: int = 255, shown_us: int = 50_000,
                     edge: int = EDGE_RISE, delay_us: int = 0, bit_depth: int = 1) -> bool:
        """The steps loaded into the Polygon and stepped by its own trigger: each TTL at Trig In shows the next
        (after ``delay_us``) for ``shown_us``, so pulse 1 of a sweep can light one region and pulse 2 another,
        timed by the protocol's digital output, not by this PC. ``bit_depth`` 1: each mirror on or off."""
        if not self.connected:
            self.status = "not connected"
            return False
        lib = self._lib
        r = lib.MTPLG_SetDevDisplayMode(self.index, DISPLAY_SEQUENCE)
        if r == 0:
            r = lib.MTPLG_SetDevPtnSetting(self.index, PatternSetting(bit_depth, len(masks), 0, TRIGGER_EXTERNAL,
                                                                      int(delay_us), 0, int(shown_us), 0, int(edge)))
        images = [patterns.image(m, on_value) for m in masks]                  # kept until the SDK has them
        for k, img in enumerate(images):
            if r != 0:
                break
            r = lib.MTPLG_SetDevPtnFromMemory(self.index, k + 1, img.ctypes.data_as(ctypes.c_void_p))
        if r == 0:
            r = lib.MTPLG_StartPattern(self.index)
        if r != 0:
            self.status = f"the sequence was not taken ({_why(r)})"
            self.stop_sequence(on_value)
            return False
        self.shown, self.sequence = f"sequence of {len(masks)}", True
        self.status = f"stepping through {len(masks)} pattern(s), one on each trigger"
        return True

    def stop_sequence(self, on_value: int = 255) -> bool:
        """The sequence stopped and the Polygon back to still pictures, nothing lit."""
        if not self.connected:
            return False
        self._lib.MTPLG_StopPattern(self.index)
        self.sequence = False
        if self._lib.MTPLG_SetDevDisplayMode(self.index, DISPLAY_STATIC) != 0:
            self.status = "it would not go back to still pictures"
            return False
        return self.off(on_value)

    def sequence_index(self) -> int | None:
        """Which pattern of the sequence is shown (from 0), as the Polygon says."""
        if not self.connected:
            return None
        s = DeviceState()
        if self._lib.MTPLG_GetDevRunStatus(self.index, ctypes.byref(s), ctypes.c_longlong(0)) != 0:
            return None
        return int(s.SeqIdx)

    def temperature(self) -> int | None:
        if not self.connected:
            return None
        s = DeviceState()
        if self._lib.MTPLG_GetDevRunStatus(self.index, ctypes.byref(s), ctypes.c_longlong(0)) != 0:
            return None
        return int(s.Temperature)

    def close(self) -> None:
        """Lets the Polygon go (PolyScan can take it again); what it shows stays until something else is sent."""
        if self._lib is not None:
            try:
                self._lib.MTPLG_DisconnectDev(self.index)
            finally:
                self._lib.MTPLG_UnInitDevice()
        self.connected = False
        self.shown = None
        self.sequence = False
        self.status = "not connected"
