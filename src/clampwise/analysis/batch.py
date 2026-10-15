"""Batches in the background (B18, item 7): a folder or a whole project
analyzed in worker processes, with progress, cancel, and a per-file error
that never stops the batch. Nothing here imports Qt; the window drives it
from a thread and reads the store.

Each worker opens the project itself, analyzes through the store (so the
cache is used and the results are kept) and sends back only a summary; the
window then reads the run from the store. One file starts only while the
memory it is expected to need fits in what the machine has free, so several
hour-long gap-free files cannot load together and run the machine out of
memory (a file's analysis has peaked at about ten times its size on disk).
"""

from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

PER_FILE_BYTES = 10.0
PROCESS_BYTES = 3e8


@dataclass
class Outcome:
    path: Path
    ok: bool
    recording_type: str = ""
    flags: int = 0
    error: str = ""
    seconds: float = 0.0
    from_cache: bool = False


def memory() -> tuple[float, float]:
    """(total, free) physical memory in bytes; (0, 0) when it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("free", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_free", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            s = Status()
            s.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return float(s.total), float(s.free)
        page = os.sysconf("SC_PAGE_SIZE")
        return float(page * os.sysconf("SC_PHYS_PAGES")), float(page * os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, AttributeError, ValueError):
        return 0.0, 0.0


def expected_bytes(path: Path) -> float:
    try:
        return PROCESS_BYTES + PER_FILE_BYTES * Path(path).stat().st_size
    except OSError:
        return PROCESS_BYTES


def worker_count(requested: int | None = None) -> int:
    """One core left for the window, about 2 GB per worker, at most 12."""
    if requested:
        return max(1, int(requested))
    cores = os.cpu_count() or 2
    total = memory()[0] / 1e9
    by_memory = int(total // 2) if total else 4
    return max(1, min(cores - 1, by_memory, 12))


def _work(project_folder: str, path: str, force: bool) -> Outcome:
    """Runs in a worker: analyze one file through the project's store."""
    import time

    from ..project import Project

    t0 = time.perf_counter()
    try:
        project = Project.open(project_folder, import_legacy=False)
        try:
            result = project.analyze(path, force=force)
        finally:
            project.close()
        cached = all(r.from_cache for r in result.results.values()) and bool(result.results)
        return Outcome(Path(path), True, result.recording_type, len(result.flags), "", time.perf_counter() - t0, cached)
    except Exception as exc:
        return Outcome(Path(path), False, "", 0, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                       time.perf_counter() - t0)


def run(project_folder: str | Path, paths: list[str | Path], *, force: bool = False, workers: int | None = None,
        cancel: Callable[[], bool] | None = None, progress: Callable[[int, int, Outcome], None] | None = None,
        in_process: bool = False) -> Iterator[Outcome]:
    """Analyze ``paths`` of the project in ``project_folder``; yields an Outcome
    per file as it finishes (in finishing order). ``cancel()`` true stops
    starting new files; what is running finishes and is kept. ``in_process``
    runs everything here (tests, one or two files)."""
    paths = [Path(p) for p in paths]
    total = len(paths)
    done = 0
    if in_process or total <= 2:
        for p in paths:
            if cancel and cancel():
                break
            outcome = _work(str(project_folder), str(p), force)
            done += 1
            if progress:
                progress(done, total, outcome)
            yield outcome
        return
    import multiprocessing

    from concurrent.futures.process import BrokenProcessPool

    budget = 0.7 * memory()[1]
    n_workers = worker_count(workers)
    queue = list(paths)
    suspects: list[Path] = []                     # files a broken pool held: each runs again on its own
    running: dict = {}                            # future -> (expected bytes, path, running on its own)

    def new_pool():
        return ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn"))

    pool, broken = new_pool(), False
    try:
        while queue or suspects or running:
            if cancel and cancel():
                queue, suspects = [], []
                for fut in [f for f in running if f.cancel()]:     # handed to the pool but not started
                    running.pop(fut)
            if broken and not running:            # a worker died: what is left goes to a new pool
                pool.shutdown(wait=False)
                pool, broken = new_pool(), False
            # no more files at once than workers, while their memory fits; a broken pool's files one at
            # a time, so that only the file that stops its worker again is the one that fails
            while not broken and (suspects or queue) and not (suspects and running) and len(running) < n_workers:
                waiting = suspects or queue
                cost = expected_bytes(waiting[0])
                if running and budget and sum(c for c, _, _ in running.values()) + cost > budget:
                    break
                try:
                    fut = pool.submit(_work, str(project_folder), str(waiting[0]), force)
                except BrokenProcessPool:
                    broken = True
                    break
                running[fut] = (cost, waiting.pop(0), waiting is suspects)
                if waiting is suspects:
                    break
            if not running:
                continue                          # nothing left, or a new pool for what is
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                _, p, alone = running.pop(fut)
                try:
                    outcome = fut.result()
                except BrokenProcessPool as exc:  # a worker died (out of memory, a crash) while this file ran
                    broken = True
                    if not alone:                 # it may have been another file's worker: run it again alone
                        suspects.append(p)
                        continue
                    outcome = Outcome(p, False, "", 0, f"worker failed: {type(exc).__name__}: {exc} (its worker "
                                      "process stopped while it was analyzed on its own: out of memory, or a crash)")
                except Exception as exc:          # the file, not the batch
                    outcome = Outcome(p, False, "", 0, f"worker failed: {type(exc).__name__}: {exc}")
                done += 1
                if progress:
                    progress(done, total, outcome)
                yield outcome
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
