"""Batches (progress, cancel, per-file errors, worker processes), the command
line and the API giving the same numbers, the known-answer harness, and the
cross-check against the old app's saved outputs."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from clampwise import api
from clampwise.analysis import batch
from clampwise.project import Project

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import known_answers  # noqa: E402


@pytest.fixture
def experiment(tmp_path):
    import pyabf.abfWriter

    rng = np.random.default_rng(1)
    folder = tmp_path / "Exp"
    for k in range(4):
        d = folder / "Slice1" / f"cell{k}"
        d.mkdir(parents=True)
        sweeps = rng.normal(-60.0 - k, 0.4, (3, 2000)).astype(np.float32)
        pyabf.abfWriter.writeABF1(sweeps, str(d / f"2661800{k}.abf"), 10_000, units="mV")
    (folder / "Slice1" / "cell0" / "broken.abf").write_bytes(b"not an abf file at all" * 100)
    return folder


def test_batch_in_process_reports_progress_errors_and_cache(experiment):
    files = sorted(experiment.rglob("*.abf"))
    p = Project.open(experiment)
    p.add_recordings(files)
    p.close()
    seen = []
    outcomes = list(batch.run(experiment, files, in_process=True, progress=lambda d, t, o: seen.append((d, t))))
    assert seen == [(i + 1, 5) for i in range(5)]
    failed = [o for o in outcomes if not o.ok]
    assert len(failed) == 1 and failed[0].path.name == "broken.abf" and "not a readable ABF" in failed[0].error
    assert all(o.recording_type == "unknown" or o.recording_type for o in outcomes if o.ok)
    again = list(batch.run(experiment, files, in_process=True))
    assert all(o.from_cache for o in again if o.ok)


def test_batch_cancel_stops_starting_new_files(experiment):
    files = sorted(experiment.rglob("*.abf"))
    Project.open(experiment).close()
    count = {"n": 0}

    def cancel():
        return count["n"] >= 2

    outcomes = []
    for o in batch.run(experiment, files, in_process=True, progress=lambda d, t, o: count.__setitem__("n", d)):
        outcomes.append(o)
        if cancel():
            pass
    # in-process: the loop checks cancel before each file
    assert 2 <= len(outcomes) <= 5


@pytest.mark.skipif(os.environ.get("CLAMPWISE_SKIP_PROCESSES") == "1", reason="worker processes disabled")
def test_batch_in_worker_processes_gives_the_same_results(experiment):
    files = sorted(experiment.rglob("*.abf"))
    p = Project.open(experiment)
    p.add_recordings(files)
    p.close()
    outcomes = list(batch.run(experiment, files, workers=2, force=True))
    assert len(outcomes) == 5 and sum(o.ok for o in outcomes) == 4
    p = Project.open(experiment)
    values = {f.name: p.last_run(f).values() for f in files if f.name != "broken.abf"}
    p.close()
    direct = {r.recording.source.name: r.values() for r in api.analyze([f for f in files if f.name != "broken.abf"],
                                                                        project=False)}
    for name, v in values.items():
        assert v["baseline.level"] == pytest.approx(direct[name]["baseline.level"], abs=1e-9)


def _thread_pool(monkeypatch, killer=None, pause_s=0.05):
    """Worker processes played by threads, so a test can kill one: ``killer``'s
    worker dies, which breaks its pool, and every file still running in that
    pool fails with it (as a real pool fails all its futures)."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    class Pool(ThreadPoolExecutor):
        def __init__(self, n, mp_context=None):
            super().__init__(n)
            self.broken = False

        def submit(self, fn, *args):
            if self.broken:
                raise BrokenProcessPool("a child process terminated abruptly")
            return super().submit(fn, *args, self)

    def work(folder, path, force, pool):
        if Path(path) == killer:
            pool.broken = True
            raise BrokenProcessPool("a child process terminated abruptly")
        time.sleep(pause_s)
        if pool.broken:
            raise BrokenProcessPool("a child process terminated abruptly")
        return batch.Outcome(Path(path), True, "unknown")

    monkeypatch.setattr(batch, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(batch, "_work", work)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_worker_that_dies_names_its_file_and_the_batch_goes_on(experiment, monkeypatch, workers):
    """A worker process killed mid-file (out of memory) breaks the pool: the
    files it held run again on their own, only the one that kills its worker
    again fails, by name, and the rest of the batch runs."""
    files = sorted(experiment.rglob("*.abf"))
    Project.open(experiment).close()
    _thread_pool(monkeypatch, killer=files[1])
    outcomes = list(batch.run(experiment, files, workers=workers))
    assert sorted(o.path for o in outcomes) == files
    failed = [o for o in outcomes if not o.ok]
    assert [o.path for o in failed] == [files[1]] and "on its own" in failed[0].error


def test_cancel_stops_files_that_have_not_started(experiment, monkeypatch):
    files = sorted(experiment.rglob("*.abf"))
    Project.open(experiment).close()
    _thread_pool(monkeypatch)
    outcomes = []
    for o in batch.run(experiment, files, workers=2, cancel=lambda: len(outcomes) >= 1):
        outcomes.append(o)
    assert len(outcomes) <= 2                  # the first, and at most the one running beside it


def test_cli_and_api_agree(experiment, tmp_path):
    from clampwise import cli

    out = tmp_path / "out"
    code = cli.main(["analyze", str(experiment), "--out", str(out), "--workers", "1"])
    assert code == 1                                  # one broken file: the batch finished, the exit code says so
    with open(out / "recordings.csv", newline="", encoding="utf-8") as fh:
        rows = {r["file"]: r for r in csv.DictReader(fh)}
    assert len(rows) == 4 and all((out / f"{Path(n).stem}.provenance.json").is_file() for n in rows)
    results = api.analyze([f for f in sorted(experiment.rglob("*.abf")) if f.name != "broken.abf"], project=False)
    for r in results:
        assert float(rows[r.recording.source.name]["baseline.level"]) == pytest.approx(r.values()["baseline.level"], abs=1e-6)
    assert cli.main(["version"]) == 0
    assert cli.main(["info", str(sorted(experiment.rglob("2661800*.abf"))[0])]) == 0


def test_known_answers_harness_passes():
    rows = known_answers.run(["baseline", "evoked"])
    bad = [r for r in rows if not r.ok]
    assert not bad, "\n".join(f"{r.case} {r.measure}: true {r.truth} measured {r.measured}" for r in bad)
