"""The methods ledger: what each analysis actually applied (B21, "done when").

Every shared tool that filters, fits or averages records one line here when
an analysis is running: "Butterworth 4-pole band-pass 1-200 Hz, zero-phase
(event detection)", "single exponential from 80% of the peak (decay tau)".
The pipeline opens a ledger for each recording and keeps the lines on the
result, so Results, ``Recording_Info`` and the tidy manifest list them, and
the manual's methods come from what ran rather than from memory.

Outside an analysis (a tool called from a script or a test) there is no open
ledger and ``note`` does nothing, so the tools never need to know.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_current: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("clampwise_ledger", default=None)


@contextmanager
def recording():
    """Collect the methods applied inside the block; yields the list."""
    lines: list[str] = []
    token = _current.set(lines)
    try:
        yield lines
    finally:
        _current.reset(token)


def note(method: str, purpose: str = "") -> None:
    """Record ``method`` (what was applied) for ``purpose`` (what it was for).
    The same line is kept once, in first-use order."""
    lines = _current.get()
    if lines is None:
        return
    line = f"{method} ({purpose})" if purpose else method
    if line not in lines:
        lines.append(line)


def current() -> list[str] | None:
    return _current.get()
