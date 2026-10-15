"""Record, Play, Pause, Resume and Stop, with a stand-in digitizer: every file a whole ABF file, each sweep at its real time, a change on the
amplifier never mixed into one file."""

from __future__ import annotations

import time

import numpy as np
import pytest

from clampwise.io import load
from clampwise.rig import acquire, growing, protocol_edit, recorder, telegraph
from clampwise.simulate.abf_file import membrane_test_file

CAL = {"dac_gains": [1.0] * 4, "dac_offsets": [0] * 4}


def _tg(mode="VC", gain=1.0):
    return telegraph.Telegraph(mode, "pA" if mode == "VC" else "mV", 5e-5 * gain if mode == "VC" else 0.01 * gain,
                               gain, 10_000.0, "mV" if mode == "VC" else "pA", 0.01, 1.0,
                               20.0 if mode == "VC" else 400.0, "00123456", 1, 0.0, 0.0)


class FakeHelper:
    """The digitizer as a stand-in: writes the recorded frames in chunks, as the helper does, until the run's end or
    its stop file; each frame's first input is its sweep's number (to see which sweeps a file kept)."""

    chunk_s = 0.02

    def __init__(self):
        self.calibration = CAL
        self.runs = []

    def record(self, req, progress=None):
        self.runs.append(req)
        out, stop = req["out"], req["out"] + ".stop"
        n_ai, total, done = len(req["ai"]), req["frames"], 0
        chunk = int(self.chunk_s * 1e6 / req["period_us"])
        with open(out, "wb") as fh:
            while done < total:
                if __import__("os").path.exists(stop):
                    break
                n = min(chunk, total - done)
                frames = np.zeros((n, n_ai), "<i2")
                frames[:, 0] = (np.arange(done, done + n) // self.step) + 100 * len(self.runs)
                fh.write(frames.tobytes())
                fh.flush()
                done += n
                if progress:
                    progress(done)
                time.sleep(self.chunk_s)
        return {"done": done}

    def close(self):
        pass


def _pro(tmp_path, sweeps=4, gap_s=0.1):
    """A protocol of ``sweeps`` 0.2 s sweeps, ``gap_s`` apart (a .pro as Clampex keeps one)."""
    head = membrane_test_file(tmp_path / "sim.abf", n_sweeps=sweeps).read_bytes()
    src = tmp_path / "src.pro"
    src.write_bytes(head[:growing._section(head, growing.DATA)[0] * growing.BLOCK])
    e = protocol_edit.load(src)
    e.start_to_start_s = 0.2 + gap_s
    return protocol_edit.save(e, tmp_path / "sim")


def _recorder(tmp_path, tg=None):
    helper = FakeHelper()
    box = {"tg": tg or _tg()}
    rec = recorder.Recorder(work=tmp_path / "work", read_telegraph=lambda *a, **k: box["tg"],
                            make_helper=lambda: helper)
    return rec, helper, box


def _start(rec, helper, pr, folder, **kw):
    helper.step = max(pr.interval_frames, pr.sweep_frames)
    rec.start(pr.path, folder, **kw)


def _until(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _sweep_ids(path):
    rec = load(path)
    raw = np.frombuffer(path.read_bytes(), "<i2", offset=growing._section(path.read_bytes(), growing.DATA)[0] * 512,
                        count=rec.n_sweeps * rec.n_samples * 2).reshape(rec.n_sweeps, rec.n_samples, 2)
    return [int(raw[k, 0, 0]) for k in range(rec.n_sweeps)], rec


def test_record_writes_every_sweep_into_a_whole_file(tmp_path):
    pr = _pro(tmp_path)
    rec, helper, _ = _recorder(tmp_path)
    _start(rec, helper, pr, tmp_path / "data")
    st = rec.wait(30)
    assert st.state == "done" and len(st.files) == 1 and st.sweeps_done == 4
    ids, loaded = _sweep_ids(st.files[0])
    assert ids == [100, 101, 102, 103] and loaded.n_sweeps == 4
    assert np.allclose(np.diff(loaded.sweep_start_times_s), 0.3, atol=0.01)


def test_play_keeps_nothing(tmp_path):
    pr = _pro(tmp_path)
    rec, helper, _ = _recorder(tmp_path)
    _start(rec, helper, pr, tmp_path / "data", save=False)
    st = rec.wait(30)
    assert st.state == "done" and st.message == "played: nothing kept"
    assert not (tmp_path / "data").exists() and not list((tmp_path / "work").glob("play/*"))


def test_pause_ends_after_the_sweep_under_way_and_resume_goes_on_in_the_same_file(tmp_path):
    pr = _pro(tmp_path, sweeps=5)
    rec, helper, _ = _recorder(tmp_path)
    _start(rec, helper, pr, tmp_path / "data")
    assert _until(lambda: rec.status.frames > helper.step + 100)            # sweep 2 under way
    rec.pause()
    assert _until(lambda: rec.status.state == "paused")
    ids, loaded = _sweep_ids(rec.status.path)                               # a whole file while paused
    assert ids == [100, 101] and loaded.n_sweeps == 2
    del loaded                                                              # the loader maps files: let go
    __import__("gc").collect()
    time.sleep(0.3)                                                         # the pause's gap
    rec.resume()
    st = rec.wait(30)
    assert st.state == "done" and len(st.files) == 1 and st.sweeps_done == 5
    ids, loaded = _sweep_ids(st.files[0])
    assert ids == [100, 101, 200, 201, 202]                                 # sweeps 3-5 from the second run
    gaps = np.diff(loaded.sweep_start_times_s)
    assert gaps[0] == pytest.approx(0.3, abs=0.02) and gaps[1] > 0.5        # the pause kept in the synch array
    assert len(helper.runs) == 2 and helper.runs[1]["frames"] == pr.run_frames(None, 2)


def test_a_change_on_the_amplifier_pauses_and_goes_on_into_a_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "TELEGRAPH_EVERY_S", 0.0)
    pr = _pro(tmp_path, sweeps=4)
    rec, helper, box = _recorder(tmp_path)
    _start(rec, helper, pr, tmp_path / "data")
    assert _until(lambda: rec.status.frames > 100)
    box["tg"] = _tg("VC", gain=2.0)                                         # the primary gain x2 in Commander
    assert _until(lambda: rec.status.state == "paused")
    assert any("the amplifier changed" in n for n in rec.status.notes)
    rec.resume()
    st = rec.wait(30)
    assert st.state == "done" and len(st.files) == 2
    first, _ = _sweep_ids(st.files[0])
    second, loaded = _sweep_ids(st.files[1])
    assert first == [100] and second[0] == 200 and len(first) + len(second) == 4
    head = st.files[1].read_bytes()                                         # the new file has the new scale
    assert acquire.inputs(acquire.read_protocol(pr.path))[0][2] != acquire.inputs(
        acquire.Protocol(st.files[1], head, *[getattr(pr, k) for k in ("mode", "interval_us", "per_episode",
                                                                        "sweeps", "start_to_start_s",
                                                                        "synch_unit_us", "seconds_per_run")],
                         ai=pr.ai))[0][2]


def test_stopped_before_one_sweep_leaves_no_file(tmp_path):
    pr = _pro(tmp_path)
    rec, helper, _ = _recorder(tmp_path)
    _start(rec, helper, pr, tmp_path / "data")
    assert _until(lambda: rec.status.frames > 0)
    rec.stop()
    st = rec.wait(30)
    assert st.state == "done" and st.files == [] and not list((tmp_path / "data").glob("*.abf"))
