"""Reading another program's files without getting in its way.

A rig program may be writing, renaming or replacing its log while Clampwise reads it. On Windows a plain ``open``
shares the file for reading and writing but not for deleting or renaming, and a memory map pins it; so every rig
file is read here in one go, opened with all three sharing flags and closed at once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def read_shared(path: str | Path, limit: int | None = None, offset: int = 0) -> bytes:
    """The file's bytes (``limit`` of them from ``offset`` when given), opened with read, write and delete sharing."""
    path = Path(path)
    if sys.platform != "win32":
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read(limit) if limit is not None else fh.read()
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.restype = wintypes.HANDLE
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                       wintypes.DWORD, wintypes.HANDLE]
    generic_read, share_all, open_existing, normal = 0x80000000, 0x1 | 0x2 | 0x4, 3, 0x80
    handle = create(str(path), generic_read, share_all, None, open_existing, normal, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), f"cannot read {path}", str(path))
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, "rb") as fh:                  # closing the file object closes the handle
        fh.seek(offset)
        return fh.read(limit) if limit is not None else fh.read()


def modified(path: str | Path) -> float:
    """The file's modification time, or 0 when it is not there (asked before reading again)."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0
