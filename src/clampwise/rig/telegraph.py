"""MultiClamp Commander's telegraph: what the amplifier tells an acquisition program about each channel (its mode,
each output's scale and gain, filters, the external command's sensitivity), as Clampex receives it.

Commander broadcasts registered window messages; a program opens a channel by posting
``MultiClampTelegraphOpenMsg`` with its window and the channel's packed id (a 700B: the serial number's low 28 bits,
the channel in the top four) and receives the state as ``WM_COPYDATA`` holding an ``MC_TELEGRAPH_DATA`` (256 bytes,
4-byte packing). The layout is Molecular Devices' published ``MultiClampBroadcastMsg.hpp`` (API version 13); ACQ4's
MIT-licensed ``MultiClampTelegraph.py`` (Luke Campagnola et al.) was read for the method. Nothing is sent to the
amplifier: the telegraph only reports.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

MODES = {0: "VC", 1: "IC", 2: "I=0"}
UNITS = {0: ("mV", 1e-3), 1: ("mV", 1.0), 2: ("mV", 1e3), 3: ("pA", 1e-12), 4: ("pA", 1e-9), 5: ("pA", 1e-6),
         6: ("pA", 1e-3), 7: ("pA", 1.0)}    # MCTG_UNITS_VOLTS_PER_*: the unit shown, and V a unit -> V a mV or pA
WM_COPYDATA = 0x004A
HWND_BROADCAST = 0xFFFF


class TelegraphData(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("uVersion", ctypes.c_uint), ("uStructSize", ctypes.c_uint), ("uComPortID", ctypes.c_uint),
                ("uAxoBusID", ctypes.c_uint), ("uChannelID", ctypes.c_uint), ("uOperatingMode", ctypes.c_uint),
                ("uScaledOutSignal", ctypes.c_uint), ("dAlpha", ctypes.c_double), ("dScaleFactor", ctypes.c_double),
                ("uScaleFactorUnits", ctypes.c_uint), ("dLPFCutoff", ctypes.c_double),
                ("dMembraneCap", ctypes.c_double), ("dExtCmdSens", ctypes.c_double),
                ("uRawOutSignal", ctypes.c_uint), ("dRawScaleFactor", ctypes.c_double),
                ("uRawScaleFactorUnits", ctypes.c_uint), ("uHardwareType", ctypes.c_uint),
                ("dSecondaryAlpha", ctypes.c_double), ("dSecondaryLPFCutoff", ctypes.c_double),
                ("szAppVersion", ctypes.c_char * 16), ("szFirmwareVersion", ctypes.c_char * 16),
                ("szDSPVersion", ctypes.c_char * 16), ("szSerialNumber", ctypes.c_char * 16),
                ("dSeriesResistance", ctypes.c_double), ("pcPadding", ctypes.c_char * 76)]


assert ctypes.sizeof(TelegraphData) == 256


@dataclass
class Telegraph:
    """One channel's telegraph, in the units Clampwise shows."""

    mode: str                  # "VC", "IC" or "I=0"
    primary_unit: str          # "pA" or "mV"
    primary_v_per_unit: float  # volts at the digitizer a unit, the output gain included (VC Im at Rf 50 MOhm: 5e-5)
    primary_gain: float
    primary_lpf_hz: float
    secondary_unit: str
    secondary_v_per_unit: float
    secondary_gain: float
    command_per_volt: float    # the external command: mV a volt in VC (20), pA a volt in IC (400); 0 in I=0
    serial: str
    channel: int
    membrane_cap_pf: float
    series_resistance_mohm: float

    @property
    def command_unit(self) -> str:
        return "mV" if self.mode == "VC" else "pA"


def pack_700b(serial: int, channel: int) -> int:
    return (int(serial) & 0x0FFFFFFF) | (int(channel) << 28)


def decode(d: TelegraphData) -> Telegraph:
    mode = MODES.get(d.uOperatingMode, "")
    pu, pf = UNITS.get(d.uScaleFactorUnits, ("", 1.0))
    su, sf = UNITS.get(d.uRawScaleFactorUnits, ("", 1.0))
    alpha = d.dAlpha or 1.0
    alpha2 = d.dSecondaryAlpha or 1.0
    sens = d.dExtCmdSens
    command = 0.0 if not sens else (sens * 1e3 if mode == "VC" else sens * 1e12)   # V/V -> mV/V; A/V -> pA/V
    return Telegraph(mode, pu, (d.dScaleFactor or 1.0) * pf * alpha, alpha, d.dLPFCutoff, su,
                     (d.dRawScaleFactor or 1.0) * sf * alpha2, alpha2, command,
                     d.szSerialNumber.decode("ascii", "replace").strip("\x00 "), d.uChannelID,
                     d.dMembraneCap * 1e12, d.dSeriesResistance * 1e-6)


def unpack_700b(ident: int) -> tuple[str, int]:
    """A 700B's (serial, channel) from its telegraph ID (the serial as Commander shows it, 8 digits)."""
    return f"{ident & 0x0FFFFFFF:08d}", (ident >> 28) & 0xF


def read(serial: int | str | None, channel: int, timeout: float = 2.0) -> Telegraph | None:
    """Ask Commander for one channel's telegraph; None when it does not answer (Commander closed). With no serial,
    the amplifiers Commander serves are asked for first (the telegraph's Broadcast) and the one with that channel is read:
    any MultiClamp 700B works, not one serial written in."""
    if sys.platform != "win32":
        return None
    # its own copies of the libraries, every call declared: a module handle passed undeclared was cut to 32 bits
    # and came back too large wherever Windows had placed the program high (Clampwise.exe at 0x7ff6d8230000 on
    # the rig PC, 2026-10-15: "argument 2: OverflowError: int too long to convert", no telegraph, no recording)
    u, k = ctypes.WinDLL("user32"), ctypes.WinDLL("kernel32")
    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class COPYDATASTRUCT(ctypes.Structure):
        _fields_ = [("dwData", ctypes.c_size_t), ("cbData", wintypes.DWORD), ("lpData", ctypes.c_void_p)]

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

    u.DefWindowProcW.restype = LRESULT
    u.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.CreateWindowExW.restype = wintypes.HWND
    u.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                  wintypes.HINSTANCE, wintypes.LPVOID]
    u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT,
                               wintypes.UINT]
    u.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    u.DestroyWindow.argtypes = [wintypes.HWND]
    u.RegisterWindowMessageW.restype = wintypes.UINT
    u.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
    u.RegisterClassW.restype = wintypes.ATOM
    u.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    u.UnregisterClassW.restype = wintypes.BOOL
    u.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    u.PostMessageW.restype = wintypes.BOOL
    u.DestroyWindow.restype = wintypes.BOOL
    k.GetModuleHandleW.restype = wintypes.HMODULE
    k.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    msg = {n: u.RegisterWindowMessageW(f"MultiClampTelegraph{n}Msg") for n in ("Open", "Close", "Request",
                                                                             "Broadcast", "Id")}
    ident = pack_700b(int(serial), channel) if serial not in (None, "") else None
    got: list[Telegraph] = []
    found: list[int] = []

    def proc(hwnd, m, wparam, lparam):
        if m == msg["Id"]:                                                  # an amplifier answering the Scan
            found.append(int(lparam or 0) & 0xFFFFFFFF)
            return 0
        if m == WM_COPYDATA and lparam:
            cds = ctypes.cast(lparam, ctypes.POINTER(COPYDATASTRUCT)).contents
            if cds.dwData == msg["Request"] and cds.cbData >= ctypes.sizeof(TelegraphData):
                d = TelegraphData.from_buffer_copy(ctypes.string_at(cds.lpData, ctypes.sizeof(TelegraphData)))
                if pack_700b(int(d.szSerialNumber or b"0"), d.uChannelID) == ident:
                    got.append(decode(d))
            return 1
        return u.DefWindowProcW(hwnd, m, wparam, lparam)

    callback = WNDPROC(proc)
    name = f"ClampwiseTelegraph{id(callback)}"
    wc = WNDCLASSW(0, callback, 0, 0, k.GetModuleHandleW(None), None, None, None, None, name)
    if not u.RegisterClassW(ctypes.byref(wc)):
        return None
    # a hidden top-level window, never shown: Commander posts its answer to a Broadcast (the 700Bs it serves) to a
    # top-level window only, not to a message-only one (tried on the rig, 2026-10-15)
    hwnd = u.CreateWindowExW(0, name, name, 0, 0, 0, 0, 0, None, None, wc.hInstance, None)
    m = wintypes.MSG()

    def pump(until, done) -> None:
        while not done() and time.monotonic() < until:
            while u.PeekMessageW(ctypes.byref(m), None, 0, 0, 1):           # PM_REMOVE: SendMessage is dispatched
                u.DispatchMessageW(ctypes.byref(m))
            time.sleep(0.02)

    try:
        if ident is None:                     # which 700Bs are there: the Broadcast, as the SDK's own client scans
            u.PostMessageW(HWND_BROADCAST, msg["Broadcast"], hwnd, 0)
            pump(time.monotonic() + min(timeout, 1.0), lambda: any(unpack_700b(i)[1] == channel for i in found))
            mine = [i for i in found if unpack_700b(i)[1] == channel]
            if not mine:
                return None
            ident = mine[0]
        u.PostMessageW(HWND_BROADCAST, msg["Open"], hwnd, ident)
        u.PostMessageW(HWND_BROADCAST, msg["Request"], hwnd, ident)
        pump(time.monotonic() + timeout, lambda: bool(got))
        u.PostMessageW(HWND_BROADCAST, msg["Close"], hwnd, ident)
    finally:
        u.DestroyWindow(hwnd)
        u.UnregisterClassW(name, wc.hInstance)
    return got[-1] if got else None
