"""Low-power mode: Clampwise on a cheaper PC (2 cores, 4 GB, no graphics card) without falling behind the rig.

On, it draws the camera's picture at most 15 times a second (the camera still takes every frame it can: snapshots,
averages and the calibration read them all), every second pixel each way while the picture is shown smaller than
the camera's own (a quarter of the drawing), its histogram once a second, and looks at the rig's programs once a
second instead of twice. On by default when the PC has fewer than 4 cores or less than 8 GB of memory; Settings
turns it on or off, kept for the PC (not the project).
"""

from __future__ import annotations

import ctypes
import os

KEY = "low_power"
# what each mode allows: redraws a second of the camera's picture, its statistics every n s, the least pixel step
# of the drawing, the rig's programs looked at every n s
NORMAL = {"draw_fps": 30.0, "stats_s": 0.5, "min_step": 1, "rig_s": 0.5}
LOW = {"draw_fps": 15.0, "stats_s": 1.0, "min_step": 2, "rig_s": 1.0}


def memory_gb() -> float:
    """The PC's memory, GB (0 when it cannot be read)."""
    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    s = Status()
    s.dwLength = ctypes.sizeof(s)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.ullTotalPhys / 2 ** 30
    except (AttributeError, OSError):
        pass
    return 0.0


def cheap_pc(cores: int | None = None, gb: float | None = None) -> bool:
    """Fewer than 4 cores or less than 8 GB: low-power mode on unless chosen otherwise."""
    cores = (os.cpu_count() or 1) if cores is None else cores
    gb = memory_gb() if gb is None else gb
    return cores < 4 or (0 < gb < 7.5)                 # "8 GB" PCs report 7.8-7.9


def enabled(settings=None) -> bool:
    """Low-power mode as chosen in Settings, else as the PC suggests."""
    if settings is not None:
        v = settings.value(KEY, None)
        if v is not None:
            return str(v).lower() in ("true", "1")
    return cheap_pc()


def limits(on: bool) -> dict:
    return dict(LOW if on else NORMAL)
