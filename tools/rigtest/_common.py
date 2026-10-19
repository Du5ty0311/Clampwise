"""What the rig tests share: Clampwise's window opened from source, screenshots, the process's CPU (each thread's)
and memory, standing in for a cheaper PC (fewer cores, lower priority), and the safety rules.

Safety: a script that moves hardware (patterns on the patterned light, the digitizer's outputs) runs only with
``--at-the-rig`` (someone there to see it and stop it), and every script ends, however it ends (done, an error,
Ctrl+C, the console closed), with the patterned light showing nothing and let go, and the camera let go. Reading
the camera moves nothing. Settings are not saved (the window opens with ``persist=False``) unless ``--keep``.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import os
import signal
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(tempfile.gettempdir()) / "clampwise-rigtest"


def parser(doc: str, hardware: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT, help="where screenshots and numbers go")
    p.add_argument("--offscreen", action="store_true", help="no window on the screen (drawing is then not real)")
    p.add_argument("--keep", action="store_true", help="save Clampwise's settings as a person's session would")
    if hardware:
        p.add_argument("--at-the-rig", action="store_true",
                       help="someone is at the rig, watching: required, this script moves hardware")
    return p


def require_presence(args) -> None:
    if not getattr(args, "at_the_rig", False):
        print("This test moves hardware (it shows patterns on the patterned light). Run it only with someone at "
              "the rig, watching, and add --at-the-rig.")
        sys.exit(2)


# ------------------------------------------------------------------ the window

_window = None


def open_window(args, size=(1366, 768)):
    """Clampwise's main window from source, as its own program opens it; the window and the QApplication."""
    global _window
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    __import__("clampwise.app")             # first, as Clampwise's own program: it limits numpy's threads before numpy
    from pyqtgraph.Qt import QtCore, QtWidgets

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Clampwise")
    app.setOrganizationName("Clampwise")
    from clampwise.app import theme
    from clampwise.app.window import MainWindow

    palette = theme.LIGHT
    theme.apply(app, palette)
    _window = MainWindow(palette, persist=bool(args.keep))
    _window.resize(*size)
    _window.show()
    pump(0.5)
    atexit.register(leave_safely)
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None), signal.SIGTERM):
        if sig is not None:
            signal.signal(sig, lambda *_: (leave_safely(), os._exit(130)))
    return _window, app


def pump(seconds: float, until=None) -> bool:
    """The window's events served for ``seconds`` (or until ``until()``): True when ``until`` came true."""
    from pyqtgraph.Qt import QtWidgets

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.005)
    return until is None


_left = threading.Lock()


def leave_safely() -> None:
    """Nothing lit, the patterned light and the camera let go, the window closed. Safe to call more than once."""
    if not _left.acquire(blocking=False):
        return
    w = _window
    if w is None:
        return
    try:
        live = w.screens["live"]
        p = live.patterned
        if p.polygon.connected:
            p.off()
            p.wait(10.0)
            print("the patterned light: Off")
        if p.polygon.connected or p._opening:
            p.release()
            p.wait(15.0)
            print("the patterned light: let go")
        live.camera.release()
    except Exception as exc:                          # said, and the window still closed
        print(f"leaving: {exc}")
    try:
        w.close()
    except Exception:
        pass


def screenshot(widget, name: str, out: Path) -> Path:
    """The widget as drawn, saved as PNG (works offscreen and over remote desktop, unlike a screen grab)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%H%M%S')}-{name}.png"
    widget.grab().save(str(path))
    return path


# ------------------------------------------------------------------ the process: CPU and memory

class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


def _ft(f: _FILETIME) -> float:
    return ((f.hi << 32) | f.lo) / 1e7


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", ctypes.c_long),
                ("tpDeltaPri", ctypes.c_long), ("dwFlags", wintypes.DWORD)]


class _MEM(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                               "PrivateUsage")]


class Process:
    """This process's CPU seconds (all of it, and each thread's) and memory, read from Windows."""

    def __init__(self) -> None:
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.OpenThread.restype = wintypes.HANDLE
        self.k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.k32.GetCurrentProcess.restype = wintypes.HANDLE
        P = ctypes.c_void_p                           # every handle declared: an undeclared one overflows
        self.k32.GetProcessTimes.argtypes = [wintypes.HANDLE, P, P, P, P]
        self.k32.GetThreadTimes.argtypes = [wintypes.HANDLE, P, P, P, P]
        self.k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.k32.Thread32First.argtypes = [wintypes.HANDLE, P]
        self.k32.Thread32Next.argtypes = [wintypes.HANDLE, P]
        self.k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        self.k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.psapi = ctypes.WinDLL("psapi")
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, P, wintypes.DWORD]
        self.pid = os.getpid()

    def cpu_s(self) -> float:
        c, e, k, u = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
        self.k32.GetProcessTimes(self.k32.GetCurrentProcess(), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k),
                                 ctypes.byref(u))
        return _ft(k) + _ft(u)

    def threads_s(self) -> dict[int, float]:
        """Each thread's CPU seconds so far, by its id."""
        out: dict[int, float] = {}
        snap = self.k32.CreateToolhelp32Snapshot(0x4, 0)                 # TH32CS_SNAPTHREAD
        if not snap or snap == wintypes.HANDLE(-1).value:
            return out
        try:
            te = _THREADENTRY32()
            te.dwSize = ctypes.sizeof(te)
            ok = self.k32.Thread32First(snap, ctypes.byref(te))
            while ok:
                if te.th32OwnerProcessID == self.pid:
                    h = self.k32.OpenThread(0x0800, False, te.th32ThreadID)     # THREAD_QUERY_LIMITED_INFORMATION
                    if h:
                        c, e, k, u = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
                        if self.k32.GetThreadTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k),
                                                   ctypes.byref(u)):
                            out[te.th32ThreadID] = _ft(k) + _ft(u)
                        self.k32.CloseHandle(h)
                ok = self.k32.Thread32Next(snap, ctypes.byref(te))
        finally:
            self.k32.CloseHandle(snap)
        return out

    def memory_mb(self) -> tuple[float, float]:
        """(working set, private bytes) in MB."""
        m = _MEM()
        m.cb = ctypes.sizeof(m)
        self.psapi.GetProcessMemoryInfo(self.k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
        return m.WorkingSetSize / 2 ** 20, m.PrivateUsage / 2 ** 20

    def cheaper(self, cores: int | None, low_priority: bool) -> str:
        """Stand in for a cheaper PC: only ``cores`` of the CPU, and below-normal priority."""
        said = []
        h = self.k32.GetCurrentProcess()
        if cores:
            mask = (1 << int(cores)) - 1
            if self.k32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)):
                said.append(f"{cores} core(s)")
        if low_priority and self.k32.SetPriorityClass(h, 0x4000):         # BELOW_NORMAL_PRIORITY_CLASS
            said.append("below-normal priority")
        return ", ".join(said) or "the whole PC"


class Sampler:
    """Where each Python thread spends its time: every few ms, the innermost of Clampwise's own functions on each
    thread's stack (and the outside call it waits in) counted; the top ones printed at the end."""

    def __init__(self, every_s: float = 0.005, idents: dict | None = None) -> None:
        self.every_s, self.counts, self._stop = every_s, {}, False
        self.idents = idents if idents is not None else {}     # Qt's threads: their Python ids, named
        self.thread = threading.Thread(target=self._run, name="sampler", daemon=True)

    def start(self) -> "Sampler":
        self.thread.start()
        return self

    def _run(self) -> None:
        me = threading.get_ident()
        names = {}
        while not self._stop:
            for t in threading.enumerate():
                names[t.ident] = t.name
            names.update(self.idents)
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                own, leaf, f = None, f"{frame.f_code.co_name} ({Path(frame.f_code.co_filename).name})", frame
                while f is not None:
                    if "clampwise" in f.f_code.co_filename and "rigtest" not in f.f_code.co_filename:
                        own = f"{Path(f.f_code.co_filename).stem}.{f.f_code.co_name}:{f.f_lineno}"
                        break
                    f = f.f_back
                key = (names.get(ident, str(ident)), own or "-", leaf)
                self.counts[key] = self.counts.get(key, 0) + 1
            time.sleep(self.every_s)

    def report(self, top: int = 25) -> list[str]:
        self._stop = True
        total: dict[str, int] = {}
        for (name, _, _), n in self.counts.items():
            total[name] = total.get(name, 0) + n
        lines = []
        for (name, own, leaf), n in sorted(self.counts.items(), key=lambda kv: -kv[1])[:top]:
            lines.append(f"{n / max(total[name], 1):6.1%} of {name}: {own}  <- {leaf}")
        return lines


def thread_names() -> dict[int, str]:
    """The Python threads' names by their Windows ids (Qt's own threads have none)."""
    return {t.native_id: t.name for t in threading.enumerate() if t.native_id}
