"""Work off the window's thread (B0: nothing on screen waits on the data)."""

from __future__ import annotations

from typing import Callable

from pyqtgraph.Qt import QtCore


class Task(QtCore.QThread):
    """Run ``fn`` in the background; ``finished_with`` carries its result, or the
    exception it raised."""

    finished_with = QtCore.Signal(object)

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:          # handed to the caller, which says what went wrong
            result = exc
        self.finished_with.emit(result)


_running: set[Task] = set()


def run(fn: Callable[[], object], done: Callable[[object], None]) -> Task:
    """Start ``fn`` in the background and call ``done(result)`` on the window's
    thread when it ends; the task is kept alive until then."""
    task = Task(fn)
    _running.add(task)
    task.finished_with.connect(done)
    # the result is sent from inside run(): the task is let go only when its thread has ended
    # (a QThread freed while it runs aborts the program), without the window waiting for it
    task.finished.connect(lambda task=task: _running.discard(task))
    task.start()
    return task
