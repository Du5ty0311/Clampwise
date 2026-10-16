"""The rig's hardware found by itself: the Digidata 1440A, the MultiClamp 700B, the camera (and the patterned light's
IOBox), from the USB devices Windows lists, without opening any of them, so a device another program uses is not
disturbed. Each is said with what it is, its serial when Windows gives it, and who is using it: the acquisition
program, the camera's program, Clampwise; for the amplifier, what MultiClamp Commander telegraphs for each channel.
As everywhere on Live, the words name the rig's roles, not the programs (Help names them): Clampex is "the
acquisition program", Infinity Analyze "the camera's program", Commander the amplifier's "program".

The devices as they identify themselves (read on the rig PC, 2026-10-16):

- Digidata 1440A: USB ``VID_09C0&PID_0145`` ("DD1440", Molecular Devices' driver)
- MultiClamp 700B: USB ``VID_09C0&PID_0136``, its serial the device's own (``...\\00832251``)
- the camera: USB ``VID_1724`` (Lumenera), "INFINITY USB 3.0 Camera", its serial the device's (``...\\0025050256``)
- the Polygon's IOBox: USB ``VID_04B4&PID_0304`` ("Mightex PolyEcho IOBox")
"""

from __future__ import annotations

import ctypes
import re
import sys
from ctypes import wintypes
from dataclasses import dataclass, field

KNOWN = (("09C0", "0145", "acquisition", "Digidata 1440A"),
         ("09C0", "0136", "amplifier", "Axon 700B"),
         ("1724", None, "camera", "Lumenera INFINITY camera"),
         ("04B4", "0304", "light", "Mightex IOBox"))
ROLES = ("acquisition", "amplifier", "camera", "light")
WORDS = {"acquisition": "Digitizer", "amplifier": "Amplifier", "camera": "Camera", "light": "Patterned light"}


@dataclass
class Device:
    role: str
    name: str
    serial: str = ""
    state: str = ""                         # who uses it, or what it reports
    instance: str = ""                      # Windows' device instance ID
    channels: dict = field(default_factory=dict)      # the amplifier: channel -> its mode, from the telegraph

    def line(self) -> str:
        return " ".join(x for x in (self.name, f"(serial {self.serial})" if self.serial else "",
                                    f"- {self.state}" if self.state else "") if x)


def usb_devices() -> list[tuple[str, str]]:
    """(instance ID, description) of every USB device present (Windows' SetupAPI; nothing is opened)."""
    if sys.platform != "win32":
        return []
    api = ctypes.WinDLL("setupapi", use_last_error=True)

    class DevInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("ClassGuid", ctypes.c_byte * 16), ("DevInst", wintypes.DWORD),
                    ("Reserved", ctypes.c_size_t)]

    api.SetupDiGetClassDevsW.restype = ctypes.c_void_p
    api.SetupDiGetClassDevsW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD]
    api.SetupDiEnumDeviceInfo.restype = wintypes.BOOL
    api.SetupDiEnumDeviceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DevInfo)]
    api.SetupDiGetDeviceInstanceIdW.restype = wintypes.BOOL
    api.SetupDiGetDeviceInstanceIdW.argtypes = [ctypes.c_void_p, ctypes.POINTER(DevInfo), wintypes.LPWSTR,
                                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.SetupDiGetDeviceRegistryPropertyW.restype = wintypes.BOOL
    api.SetupDiGetDeviceRegistryPropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(DevInfo), wintypes.DWORD,
                                                      ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
                                                      wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]
    DIGCF_PRESENT, DIGCF_ALLCLASSES, SPDRP_DEVICEDESC, SPDRP_FRIENDLYNAME = 0x2, 0x4, 0x0, 0xC
    h = api.SetupDiGetClassDevsW(None, "USB", None, DIGCF_PRESENT | DIGCF_ALLCLASSES)
    if not h or h == ctypes.c_void_p(-1).value:
        return []
    out = []
    try:
        i = 0
        info = DevInfo()
        info.cbSize = ctypes.sizeof(DevInfo)
        while api.SetupDiEnumDeviceInfo(h, i, ctypes.byref(info)):
            i += 1
            ident = ctypes.create_unicode_buffer(512)
            if not api.SetupDiGetDeviceInstanceIdW(h, ctypes.byref(info), ident, 512, None):
                continue
            desc = ""
            for prop in (SPDRP_FRIENDLYNAME, SPDRP_DEVICEDESC):
                buf = ctypes.create_unicode_buffer(256)
                if api.SetupDiGetDeviceRegistryPropertyW(h, ctypes.byref(info), prop, None, buf,
                                                         ctypes.sizeof(buf), None):
                    desc = buf.value
                    break
            out.append((ident.value, desc))
    finally:
        api.SetupDiDestroyDeviceInfoList(h)
    return out


def identify(instance: str, description: str = "") -> Device | None:
    """The rig device an instance ID belongs to, or None."""
    m = re.match(r"USB\\VID_([0-9A-F]{4})&PID_([0-9A-F]{4})(?:&[^\\]*)?\\(.*)$", instance, re.IGNORECASE)
    if not m:
        return None
    vid, pid, rest = m.group(1).upper(), m.group(2).upper(), m.group(3)
    for kvid, kpid, role, name in KNOWN:
        if vid == kvid and (kpid is None or pid == kpid):
            serial = rest if re.fullmatch(r"[0-9A-Za-z]{6,}", rest) and "&" not in rest else ""
            if role == "camera" and serial.isdigit():
                serial = str(int(serial))                 # 0025050256 -> 25050256, as the camera reports it
            return Device(role, name, serial, "", instance)
    return None


def programs() -> set[str]:
    """The programs with a window open (lower case executable names)."""
    try:
        from .windows import top_windows

        return {w.program for w in top_windows()}
    except Exception:
        return set()


def _commander_open() -> bool:
    from .multiclamp import commander_open

    return commander_open()


def detect(devices=usb_devices, running=programs, read_telegraph=None, held: dict | None = None,
           commander=_commander_open) -> list[Device]:
    """The rig's devices present now, each said with who uses it. ``held``: what Clampwise itself holds
    (role -> True). The amplifier's channels are read from Commander's telegraph when Commander is open (found by
    its own hidden window, so a Commander in the tray counts)."""
    from . import telegraph

    read_telegraph = read_telegraph or telegraph.read
    held = held or {}
    progs = running()
    found: dict[str, Device] = {}
    for instance, description in devices():
        d = identify(instance, description)
        if d is not None and d.role not in found:
            found[d.role] = d
    for role, d in found.items():
        if role == "acquisition":
            d.state = ("in use by Clampwise" if held.get(role) else
                       "in use by the acquisition program" if "clampex.exe" in progs else "free")
        elif role == "camera":
            d.name = _camera_model() or d.name
            d.state = ("in use by Clampwise" if held.get(role) else
                       "in use by the camera's program" if "infinityanalyze.exe" in progs else "free")
        elif role == "amplifier":
            if not commander():
                d.state = "its program is closed"
                continue
            for ch in (1, 2):
                tg = read_telegraph(d.serial or None, ch, timeout=0.5)
                if tg is not None:
                    d.channels[ch] = tg.mode
            d.state = ("its program is open: " + ", ".join(f"channel {c} {m}" for c, m in sorted(d.channels.items()))
                       if d.channels else "its program is open, no telegraph")
        elif role == "light":
            d.state = "its program is open" if any(p.startswith("polyscan") for p in progs) else "connected"
    return [found[r] for r in ROLES if r in found]


def _camera_model() -> str:
    """The camera's model as its own program last saw it (Infinity Analyze's settings)."""
    try:
        from .camera import read_settings

        s = read_settings()
        return f"Lumenera {s.camera}" if s is not None and s.camera else ""                # e.g. INFINITY8-3M
    except Exception:
        return ""


def summary(devices: list[Device]) -> str:
    """One line a device; the missing ones said too."""
    have = {d.role: d for d in devices}
    out = []
    for role in ROLES[:3]:
        d = have.get(role)
        out.append(f"{WORDS[role]}: " + (d.line() if d else "not found"))
    if "light" in have:
        out.append(f"{WORDS['light']}: " + have["light"].line())
    return "\n".join(out)
