"""The rig programs' windows: found, shown live inside Clampwise, and brought forward on request.

Each program keeps its own window and its own controls; Clampwise only finds the windows (by the program that
owns them), shows each one live in its own screen through the Windows compositor (a DWM thumbnail: Windows
draws the other window's picture into ours, as in the taskbar's previews, so nothing is copied and the program is
not touched), and brings one to the front when its view is clicked, so it is operated with its own controls.
A minimized window has no picture to show until it is restored.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

# the rig's roles and the programs that fill them (Help names the programs; the screen shows the roles)
ROLES = {
    "amplifier": ("Amplifier", ("mc700b.exe",)),
    "acquisition": ("Acquisition", ("clampex.exe", "axoscope.exe")),
    "light": ("Patterned light", ("polyscan3.exe", "polyscan.exe", "cspolyscan2.exe")),
    "camera": ("Camera", ("infinityanalyze.exe", "pylonviewer.exe", "pylonviewerapp.exe")),   # Infinity; pylon Viewer
}
# the part of a program's window shown when the user has chosen none: (left, top, right, bottom) px in from its
# edges, or the class of the child window to follow. Infinity Analyze 7 draws the camera's picture in the panel below
# its menu, left of its Control Panel and above its status bar (measured on a rig PC: only where the image
# is displayed); Clampex draws its traces in its scope display window (the trace is what matters).
PARTS = {"infinityanalyze.exe": (8, 52, 364, 36), "clampex.exe": "AXO_ScopeDisplayClass32"}


def usual_part(window: AppWindow) -> tuple[int, int, int, int] | None:
    """The program's usual part of ``window`` as px in from its edges (a child window found now), if it has one."""
    part = PARTS.get(window.program)
    if part is None or isinstance(part, tuple):
        return part
    if sys.platform != "win32":
        return None
    from ctypes import wintypes as W

    user32 = ctypes.windll.user32
    outer = W.RECT()
    if not user32.GetWindowRect(window.hwnd, ctypes.byref(outer)):
        return None
    best: list = []
    name = ctypes.create_unicode_buffer(64)

    @ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    def each(hwnd, _):
        user32.GetClassNameW(hwnd, name, 64)
        r = W.RECT()
        if name.value == part and user32.IsWindowVisible(hwnd) and user32.GetWindowRect(hwnd, ctypes.byref(r)):
            best.append(((r.right - r.left) * (r.bottom - r.top), (r.left, r.top, r.right, r.bottom)))
        return True

    user32.EnumChildWindows(window.hwnd, each, 0)
    if not best:
        return None
    left, top, right, bottom = max(best)[1]
    return (left - outer.left, top - outer.top, outer.right - right, outer.bottom - bottom)


@dataclass(frozen=True)
class AppWindow:
    hwnd: int
    title: str
    program: str               # the executable's name
    pid: int
    minimized: bool
    width: int
    height: int


class _Placement(ctypes.Structure):
    from ctypes import wintypes as _W

    _fields_ = [("length", _W.UINT), ("flags", _W.UINT), ("showCmd", _W.UINT), ("ptMinPosition", _W.POINT),
                ("ptMaxPosition", _W.POINT), ("rcNormalPosition", _W.RECT)]


def _api():
    from ctypes import wintypes as W

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

    def p(dll, name, restype, *argtypes):
        fn = getattr(dll, name)
        fn.restype, fn.argtypes = restype, list(argtypes)
        return fn

    enum_proc = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    api = {
        "enum_proc": enum_proc,
        "EnumWindows": p(user32, "EnumWindows", W.BOOL, enum_proc, W.LPARAM),
        "IsWindowVisible": p(user32, "IsWindowVisible", W.BOOL, W.HWND),
        "IsWindow": p(user32, "IsWindow", W.BOOL, W.HWND),
        "IsIconic": p(user32, "IsIconic", W.BOOL, W.HWND),
        "GetWindow": p(user32, "GetWindow", W.HWND, W.HWND, W.UINT),
        "GetWindowTextW": p(user32, "GetWindowTextW", ctypes.c_int, W.HWND, W.LPWSTR, ctypes.c_int),
        "GetWindowLongW": p(user32, "GetWindowLongW", ctypes.c_long, W.HWND, ctypes.c_int),
        "GetWindowThreadProcessId": p(user32, "GetWindowThreadProcessId", W.DWORD, W.HWND, ctypes.POINTER(W.DWORD)),
        "GetWindowRect": p(user32, "GetWindowRect", W.BOOL, W.HWND, ctypes.POINTER(W.RECT)),
        "GetWindowPlacement": p(user32, "GetWindowPlacement", W.BOOL, W.HWND, ctypes.POINTER(_Placement)),
        "ShowWindow": p(user32, "ShowWindow", W.BOOL, W.HWND, ctypes.c_int),
        "SetForegroundWindow": p(user32, "SetForegroundWindow", W.BOOL, W.HWND),
        "OpenProcess": p(kernel32, "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD),
        "CloseHandle": p(kernel32, "CloseHandle", W.BOOL, W.HANDLE),
        "QueryFullProcessImageNameW": p(kernel32, "QueryFullProcessImageNameW", W.BOOL, W.HANDLE, W.DWORD, W.LPWSTR,
                                        ctypes.POINTER(W.DWORD)),
        "DwmGetWindowAttribute": p(dwmapi, "DwmGetWindowAttribute", ctypes.c_long, W.HWND, W.DWORD, ctypes.c_void_p,
                                   W.DWORD),
        "DwmRegisterThumbnail": p(dwmapi, "DwmRegisterThumbnail", ctypes.c_long, W.HWND, W.HWND,
                                  ctypes.POINTER(ctypes.c_void_p)),
        "DwmUnregisterThumbnail": p(dwmapi, "DwmUnregisterThumbnail", ctypes.c_long, ctypes.c_void_p),
        "DwmUpdateThumbnailProperties": p(dwmapi, "DwmUpdateThumbnailProperties", ctypes.c_long, ctypes.c_void_p,
                                          ctypes.c_void_p),
        "DwmQueryThumbnailSourceSize": p(dwmapi, "DwmQueryThumbnailSourceSize", ctypes.c_long, ctypes.c_void_p,
                                         ctypes.POINTER(W.SIZE)),
    }
    return api


_API = None


def api():
    global _API
    if _API is None:
        _API = _api()
    return _API


def _program(pid: int) -> str:
    a = api()
    handle = a["OpenProcess"](0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(1024)
        ok = a["QueryFullProcessImageNameW"](handle, 0, buf, ctypes.byref(size))
        return Path(buf.value).name.lower() if ok else ""
    finally:
        a["CloseHandle"](handle)


def top_windows() -> list[AppWindow]:
    """The visible top-level application windows (not tool windows, owned dialogs or cloaked ones)."""
    if sys.platform != "win32":
        return []
    a = api()
    found: list[int] = []

    def keep(hwnd, _):
        found.append(hwnd)
        return True

    a["EnumWindows"](a["enum_proc"](keep), 0)
    out = []
    for hwnd in found:
        if not a["IsWindowVisible"](hwnd) or a["GetWindow"](hwnd, 4):            # GW_OWNER: an owned dialog
            continue
        if a["GetWindowLongW"](hwnd, -20) & 0x80:                                # WS_EX_TOOLWINDOW
            continue
        cloaked = ctypes.c_int(0)
        if a["DwmGetWindowAttribute"](hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0 and cloaked.value:
            continue
        pid = ctypes.c_ulong(0)
        a["GetWindowThreadProcessId"](hwnd, ctypes.byref(pid))
        buf = ctypes.create_unicode_buffer(512)
        a["GetWindowTextW"](hwnd, buf, 512)
        from ctypes import wintypes as W

        minimized = bool(a["IsIconic"](hwnd))
        r = W.RECT()
        if minimized:                          # a minimized window's size is its restored one (to tell its main window)
            place = _Placement()
            place.length = ctypes.sizeof(_Placement)
            if a["GetWindowPlacement"](hwnd, ctypes.byref(place)):
                r = place.rcNormalPosition
        else:
            a["GetWindowRect"](hwnd, ctypes.byref(r))
        out.append(AppWindow(int(hwnd), buf.value, _program(pid.value), pid.value, minimized,
                             r.right - r.left, r.bottom - r.top))
    return out


def rig_windows(windows: list[AppWindow] | None = None) -> dict[str, list[AppWindow]]:
    """The rig programs' windows by role (the largest first: a program's main window before its others)."""
    windows = top_windows() if windows is None else windows
    out: dict[str, list[AppWindow]] = {role: [] for role in ROLES}
    for w in windows:
        for role, (_, programs) in ROLES.items():
            if w.program in programs:
                out[role].append(w)
    for role in out:
        out[role].sort(key=lambda w: -(w.width * w.height))
    return out


def bring_to_front(hwnd: int) -> bool:
    """Show a rig program's window in front, restored if minimized (then worked in directly)."""
    if sys.platform != "win32":
        return False
    a = api()
    if not a["IsWindow"](hwnd):
        return False
    if a["IsIconic"](hwnd):
        a["ShowWindow"](hwnd, 9)                                                   # SW_RESTORE
    return bool(a["SetForegroundWindow"](hwnd))


class _Properties(ctypes.Structure):
    from ctypes import wintypes as _W

    _fields_ = [("dwFlags", _W.DWORD), ("rcDestination", _W.RECT), ("rcSource", _W.RECT), ("opacity", ctypes.c_ubyte),
                ("fVisible", _W.BOOL), ("fSourceClientAreaOnly", _W.BOOL)]


class Thumbnail:
    """A live picture of ``source`` drawn by the compositor into ``target`` (a top-level window of ours) at a
    rectangle of its client area, in physical pixels. Unregister it when it is no longer shown."""

    RECT_DESTINATION, RECT_SOURCE, OPACITY, VISIBLE, CLIENT_ONLY = 0x1, 0x2, 0x4, 0x8, 0x10

    def __init__(self, target: int, source: int) -> None:
        self.target, self.source = int(target), int(source)
        self._handle = ctypes.c_void_p()
        self.shown: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None   # (source, destination)
        hr = api()["DwmRegisterThumbnail"](self.target, self.source, ctypes.byref(self._handle))
        if hr != 0:
            self._handle = None
            raise OSError(f"the window cannot be shown here (DWM error {hr & 0xFFFFFFFF:#x})")

    def source_size(self) -> tuple[int, int]:
        from ctypes import wintypes as W

        size = W.SIZE()
        if self._handle and api()["DwmQueryThumbnailSourceSize"](self._handle, ctypes.byref(size)) == 0:
            return int(size.cx), int(size.cy)
        return 0, 0

    def place(self, left: int, top: int, width: int, height: int, visible: bool = True,
              part: tuple[int, int, int, int] | None = None, fill: bool = False) -> None:
        """Fit the source (or its ``part``: px in from its left, top, right and bottom edges) into the rectangle,
        keeping its proportions, or stretched to fill it (``fill``)."""
        if not self._handle:
            return
        sw, sh = self.source_size()
        src = (0, 0, sw, sh)
        if part is not None and sw - part[0] - part[2] > 10 and sh - part[1] - part[3] > 10:
            src = (part[0], part[1], sw - part[2], sh - part[3])
        cw, ch = src[2] - src[0], src[3] - src[1]
        if cw > 0 and ch > 0 and width > 0 and height > 0 and not fill:
            scale = min(width / cw, height / ch)
            w, h = int(cw * scale), int(ch * scale)
            left, top, width, height = left + (width - w) // 2, top + (height - h) // 2, w, h
        self.shown = (src, (left, top, left + width, top + height))
        props = _Properties()
        props.dwFlags = self.RECT_DESTINATION | self.RECT_SOURCE | self.OPACITY | self.VISIBLE
        props.rcSource.left, props.rcSource.top, props.rcSource.right, props.rcSource.bottom = src
        props.rcDestination.left, props.rcDestination.top = left, top
        props.rcDestination.right, props.rcDestination.bottom = left + width, top + height
        props.opacity, props.fVisible = 255, bool(visible and width > 0 and height > 0)
        api()["DwmUpdateThumbnailProperties"](self._handle, ctypes.byref(props))

    def part_size(self, part: tuple[int, int, int, int] | None = None) -> tuple[int, int]:
        """The size of what is shown of the source (its ``part``, else the whole window), in its px."""
        sw, sh = self.source_size()
        if part is not None and sw - part[0] - part[2] > 10 and sh - part[1] - part[3] > 10:
            return sw - part[0] - part[2], sh - part[1] - part[3]
        return sw, sh

    def to_source(self, x: int, y: int) -> tuple[int, int] | None:
        """A point of the destination (px of our window's client area) in the source window's px, if shown there."""
        if self.shown is None:
            return None
        (sl, st, sr, sb), (dl, dt, dr, db) = self.shown
        if dr <= dl or db <= dt:
            return None
        return (round(sl + (min(max(x, dl), dr) - dl) * (sr - sl) / (dr - dl)),
                round(st + (min(max(y, dt), db) - dt) * (sb - st) / (db - dt)))

    def close(self) -> None:
        if self._handle:
            api()["DwmUnregisterThumbnail"](self._handle)
        self._handle = None
