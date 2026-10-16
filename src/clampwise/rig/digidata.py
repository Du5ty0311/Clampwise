"""The Digidata 1440A from Clampwise: the 32-bit helper (``dd1440_helper.py``) started, asked and let go. One run at
a time (one program holds the digitizer); the recorder and the membrane test both use it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

HELPER = Path(__file__).with_name("dd1440_helper.py")


def py32() -> Path:
    """The 32-bit Python the helper runs in: the one shipped beside Clampwise.exe (``py32``), else
    %LOCALAPPDATA%\\Clampwise\\py32 (running from source: python.org's 32-bit embeddable package unzipped there)."""
    if getattr(sys, "frozen", False):
        shipped = Path(sys.executable).parent / "py32" / "python.exe"
        if shipped.exists():
            return shipped
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Clampwise" / "py32" / "python.exe"


def available() -> str:
    """Why the digitizer cannot be used here, or "" when it can."""
    if os.name != "nt":
        return "recording needs the rig PC (Windows)"
    if not py32().exists():
        return ("the 32-bit Python for the digitizer's library is not there (the built program ships it; from source, "
                "unzip python.org's 32-bit embeddable Python 3.12 into %LOCALAPPDATA%\\Clampwise\\py32)")
    return ""


class Helper:
    """The helper process: opened (the digitizer found, opened and its calibration read), runs, closed."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen([str(py32()), "-I", str(HELPER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.calibration = self.ask({"cmd": "describe"})["calibration"]

    def ask(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        out = json.loads(self.proc.stdout.readline() or '{"error": "the helper ended"}')
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def record(self, req: dict, progress=None) -> dict:
        """One run (``dd1440_helper.record``); ``progress(frames)`` as it goes; its last line (done, or error)."""
        self.proc.stdin.write(json.dumps({"cmd": "record", **req}) + "\n")
        while True:
            line = json.loads(self.proc.stdout.readline() or '{"error": "the helper ended"}')
            if "frames" in line and "done" not in line:
                if progress is not None:
                    progress(line["frames"])
                continue
            return line

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(10)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            self.proc.kill()


class _Lent:
    """The kept helper, lent for one run: closing it gives it back, still open for the next run."""

    def __init__(self, helper: Helper) -> None:
        self._helper = helper
        self.calibration = helper.calibration

    def ask(self, req: dict) -> dict:
        return self._helper.ask(req)

    def record(self, req: dict, progress=None) -> dict:
        return self._helper.record(req, progress)

    def close(self) -> None:
        pass


_kept: Helper | None = None
_lock = threading.Lock()


def lend() -> _Lent:
    """The helper kept open while Clampwise holds the digitizer: opening it (finding and opening the Digidata) takes
    3 s on the rig PC, which every Play, Record and membrane test waited through before. Opened again if it ended."""
    global _kept
    with _lock:
        if _kept is None or _kept.proc.poll() is not None:
            _kept = Helper()
        return _Lent(_kept)


def held() -> bool:
    """Whether Clampwise holds the digitizer open now (its helper kept for Play, Record and the membrane test)."""
    return _kept is not None and _kept.proc.poll() is None


def let_go() -> None:
    """The digitizer let go (Live's Let go, Clampwise closing): the acquisition program can open it."""
    global _kept
    with _lock:
        if _kept is not None:
            _kept.close()
            _kept = None
