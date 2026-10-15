"""Watching a folder while Clampex writes into it (B1; rig mode, B19, builds on it).

Clampex writes a recording as it acquires and finishes its header at the end,
so a file is taken only when its size has stopped changing between two looks
at least ``settle_s`` apart and its header can be read. Files already there
when the watch starts are not new. Nothing here knows the window: it polls
when asked and says what has arrived.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

PATTERNS = ("*.abf",)


@dataclass
class FolderWatcher:
    folder: Path
    settle_s: float = 2.0
    recursive: bool = True
    seen: set[Path] = field(default_factory=set)
    _pending: dict[Path, tuple[int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self.seen = set(self._files())          # what is there already is not new

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for pattern in PATTERNS:
            out += list(self.folder.rglob(pattern) if self.recursive else self.folder.glob(pattern))
        return [p.resolve() for p in out]

    def poll(self, now: float | None = None) -> list[Path]:
        """The files that have arrived and finished since the last poll."""
        now = time.monotonic() if now is None else now
        done = []
        for p in self._files():
            if p in self.seen:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            before = self._pending.get(p)
            if before is None or before[0] != size:
                self._pending[p] = (size, now)              # still growing (or first seen): look again later
                continue
            if now - before[1] < self.settle_s or size == 0:
                continue
            if not _readable(p):
                continue
            self.seen.add(p)
            self._pending.pop(p, None)
            done.append(p)
        return sorted(done)


def _readable(path: Path) -> bool:
    try:
        from .abf import read_header

        read_header(path)
        return True
    except Exception:
        return False
