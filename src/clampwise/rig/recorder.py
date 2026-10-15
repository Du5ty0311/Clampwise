"""Recording in Clampwise, without Clampex: every output an ABF file Clampex and Clampfit open, with Play, Pause and
Record, the amplifier's scales from MultiClamp Commander's telegraph: a Clampex protocol played on the Digidata, in the background.

**Record** writes an ABF2 file Clampex, Clampfit and pyabf read; **Play** runs the same, drawn and measured, keeping
nothing. The amplifier's telegraph gives every scale (``acquire.with_telegraph``) and is watched while the run goes:
a change of mode, gain or scale pauses it after the sweep under way, since the file's scales would no longer be
true. **Pause** ends an episodic run's segment once the sweep under way is complete (the digitizer queues its
outputs seconds ahead, so a sweep is never cut) and a gap-free one at once; the file is then a whole ABF file.
**Resume** plays on from the next sweep into the same file, each sweep at its real time in the synch array, or into
a new file when the telegraph changed or the protocol is gap-free (a gap is never hidden in a continuous file).
The file is grown as Clampex grows one (the header first, each sweep appended once complete) so Live reads it while
it is written; a run stopped before one sweep is complete leaves no file.
"""

from __future__ import annotations

import datetime as dt
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import acquire, digidata, telegraph
from .digidata import available
from .growing import EPISODIC, scratch

TELEGRAPH_EVERY_S = 1.0


def next_name(folder: Path, when: dt.datetime) -> Path:
    """Clampex's date code (26o15: 2026, October, 15th), "cw", then the next free number."""
    code = f"{when:%y}{'123456789ond'[when.month - 1]}{when:%d}cw"
    n = 0
    while (folder / f"{code}{n:03d}.abf").exists():
        n += 1
    return folder / f"{code}{n:03d}.abf"


def _scales(tg) -> tuple:
    return (tg.mode, round(tg.primary_v_per_unit, 12), round(tg.secondary_v_per_unit, 12), tg.command_per_volt) \
        if tg is not None else None


def _words(tg) -> str:
    return (f"{tg.mode}, {tg.primary_unit} at {tg.primary_v_per_unit:g} V (gain {tg.primary_gain:g}), command "
            f"{tg.command_per_volt:g} {tg.command_unit}/V") if tg is not None else "no telegraph"


@dataclass
class Status:
    state: str = "idle"           # idle, starting, recording, playing, pausing, paused, finishing, done, failed
    message: str = ""
    path: Path | None = None      # the file being written (Record), or played into (Play: removed after)
    frames: int = 0               # this segment's
    total: int = 0
    started: dt.datetime | None = None
    notes: list[str] = field(default_factory=list)
    saving: bool = True
    sweeps_done: int = 0          # complete sweeps of the run so far
    files: list[Path] = field(default_factory=list)      # every file this run made (Record)
    telegraph: object = None      # the amplifier's telegraph, as last read
    segment: int = 0              # bumped at each start or resume (the scope starts its sweeps again)
    # what a scope needs to draw the inputs while they are recorded (the raw file, its layout, each input's scale)
    raw: Path | None = None
    inputs: list[tuple[str, str, float]] = field(default_factory=list)   # (name, unit, volts a unit)
    rate_hz: float = 0.0
    sweep_frames: int = 0
    step_frames: int = 0
    episodic: bool = True

    @property
    def busy(self) -> bool:
        return self.state in ("starting", "recording", "playing", "pausing", "paused", "finishing")


class _File:
    """One ABF file being written: the header (counts 0), each sweep appended, finished with its counts and synch
    array (a whole file at every pause), opened again on resume."""

    def __init__(self, pr: acquire.Protocol, path: Path, start: dt.datetime) -> None:
        self.pr, self.path, self.start, self.t0 = pr, path, start, time.time()
        self.sweeps: list[np.ndarray] = []
        self.starts: list[float] = []
        self.head = acquire.abf_bytes(pr, np.zeros((0, len(pr.ai)), "<i2"), start, sweeps=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writing()

    def _writing(self) -> None:
        """The file as Clampex has it while writing: the header, its counts 0, then the sweeps so far."""
        with open(self.path, "wb") as fh:
            fh.write(self.head)
            for s in self.sweeps:
                fh.write(s.astype("<i2").tobytes())

    def add(self, data: np.ndarray, t_s: float) -> None:
        self.sweeps.append(data)
        self.starts.append(t_s)
        with open(self.path, "ab") as fh:
            fh.write(data.astype("<i2").tobytes())

    def finish(self, raw: Path | None = None) -> None:
        """The counts and synch array written: a whole file (gap-free: the samples of ``raw``)."""
        if self.pr.mode == EPISODIC:
            self.path.write_bytes(acquire.episodic_bytes(self.pr, self.sweeps, self.starts, self.start))
        else:
            samples = np.fromfile(raw, "<i2").reshape(-1, len(self.pr.ai)) if raw and raw.exists() else \
                np.zeros((0, len(self.pr.ai)), "<i2")
            self.path.write_bytes(acquire.abf_bytes(self.pr, samples, self.start))

    def empty(self, raw: Path | None = None) -> bool:
        if self.pr.mode == EPISODIC:
            return not self.sweeps
        return raw is None or not raw.exists() or raw.stat().st_size == 0


class Recorder:
    """One run at a time, in a background thread. ``start`` returns at once; ``status`` says how it goes."""

    def __init__(self, serial: str | None = None, channel: int = 1, work: Path | None = None,
                 read_telegraph=telegraph.read, make_helper=digidata.lend) -> None:
        self.serial, self.channel = serial, channel
        self.work = work or scratch() / "acq"
        self.read_telegraph, self.make_helper = read_telegraph, make_helper
        self.status = Status()
        self._thread: threading.Thread | None = None
        self._stop_file: Path | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._resume = threading.Event()

    def start(self, protocol: str | Path, folder: str | Path, seconds: float | None = None, save: bool = True) -> None:
        if self.status.busy:
            raise RuntimeError("a run is already going")
        why = available()
        if why and self.make_helper is digidata.lend:
            raise RuntimeError(why)
        self.status = Status("starting", "reading the amplifier's telegraph", saving=save)
        self._stop.clear()
        self._pause.clear()
        self._resume.clear()
        self._thread = threading.Thread(target=self._run, args=(Path(protocol), Path(folder), seconds, save),
                                        name="Clampwise recording", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """After the sweep under way (episodic), at once (gap-free)."""
        if self.status.state in ("recording", "playing"):
            self.status.state = "pausing"
            self._pause.set()

    def resume(self) -> None:
        if self.status.state == "paused":
            self._resume.set()

    def stop(self) -> None:
        """End: the sweeps complete by then are kept (Record)."""
        self._stop.set()
        self._resume.set()
        if self._stop_file is not None:
            self._stop_file.touch()

    def wait(self, timeout: float | None = None) -> Status:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.status

    # ---- the run
    def _run(self, protocol: Path, folder: Path, seconds: float | None, save: bool) -> None:
        st = self.status
        helper = None
        f: _File | None = None
        raw = self.work / "inputs.raw"
        try:
            tg = self.read_telegraph(self.serial, self.channel)
            source = acquire.read_protocol(protocol)
            pr, note = acquire.with_telegraph(source, tg)
            st.telegraph = tg
            if note:
                st.notes.append(note)
            st.message = "opening the digitizer"
            helper = self.make_helper()
            self.work.mkdir(parents=True, exist_ok=True)
            out = folder if save else self.work / "play"
            if not save:
                shutil.rmtree(out, ignore_errors=True)
            first = 0
            while True:
                if f is None:
                    start = dt.datetime.now()
                    f = _File(pr, next_name(out, start), start)
                    st.path, st.started = f.path, start
                    if save:
                        st.files.append(f.path)
                reason, done = self._segment(helper, pr, tg, f, first, seconds, raw)
                first += done
                st.sweeps_done = first
                st.state = "finishing"
                if f.empty(raw):
                    f.path.unlink(missing_ok=True)
                    if save and f.path in st.files:
                        st.files.remove(f.path)
                    f = None
                else:
                    f.finish(raw)
                if reason in ("done", "stop") or (pr.mode == EPISODIC and first >= pr.sweeps):
                    break
                st.state = "paused"
                st.message = st.message or "paused"
                self._resume.wait()
                self._resume.clear()
                self._pause.clear()
                if self._stop.is_set():
                    break
                new = self.read_telegraph(self.serial, self.channel)
                changed = _scales(new) != _scales(tg)
                if changed:                                # new scales: a new file, never mixed in one
                    pr, note = acquire.with_telegraph(source, new)
                    st.notes.append(f"the amplifier now: {_words(new)}; a new file")
                    tg = st.telegraph = new
                if changed or pr.mode != EPISODIC or f is None:
                    f = None
                else:
                    try:
                        f._writing()                       # writing again: the counts 0, the synch array gone
                    except OSError:                        # open elsewhere (another program has it mapped): kept
                        st.notes.append(f"{f.path.name} is open elsewhere: kept as it is, a new file goes on")
                        f.finish(raw)
                        f = None
            if not save:
                shutil.rmtree(self.work / "play", ignore_errors=True)
                st.path = None
            kept = [p for p in st.files if p.exists()]
            st.state = "done"
            st.message = ("played: nothing kept" if not save else
                          "stopped before the first sweep was complete: no file written" if not kept else
                          ", ".join(p.name for p in kept) + f": {st.sweeps_done} sweep(s)" if pr.mode == EPISODIC
                          else ", ".join(p.name for p in kept))
        except Exception as exc:                          # said on the screen; what was recorded is kept
            if f is not None and not f.empty(raw):
                try:
                    f.finish(raw)
                except OSError:
                    pass
            st.state, st.message = "failed", str(exc)
        finally:
            if helper is not None:
                helper.close()

    def _segment(self, helper, pr: acquire.Protocol, tg, f: _File, first: int, seconds: float | None,
                 raw: Path) -> tuple[str, int]:
        """Play from sweep ``first`` until done, stopped or paused; the complete sweeps go into ``f``.
        Returns (why it ended: done, stop, pause; the sweeps completed)."""
        st = self.status
        cal = helper.calibration
        wave = self.work / "waveform.bin"
        self._stop_file = Path(str(raw) + ".stop")
        self._stop_file.unlink(missing_ok=True)
        wave.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        episodic = pr.mode == EPISODIC
        if episodic:                              # gap-free: every output held, no waveform (an hour is 720 MB)
            frames = acquire.output_frames(pr, cal, seconds, first)
            frames.tofile(wave)
            total = len(frames)
        else:
            total = pr.run_frames(seconds)
        n_ai = len(pr.ai)
        step = max(pr.interval_frames, pr.sweep_frames) if episodic else pr.sweep_frames
        st.raw, st.rate_hz, st.sweep_frames, st.step_frames = raw, pr.rate_hz, pr.sweep_frames, step
        st.episodic, st.inputs, st.total, st.frames = episodic, acquire.inputs(pr), total, 0
        st.segment += 1
        offset = time.time() - f.t0                # this segment's start, in the file's time
        limit = pr.sweeps - first if episodic else 10 ** 9
        done = threading.Event()
        added = [0]

        def grow() -> None:                        # each complete sweep into the file as it comes
            while True:
                last = done.is_set()
                have = raw.stat().st_size // (2 * n_ai) if raw.exists() else 0
                while added[0] < limit and have >= added[0] * step + pr.sweep_frames:
                    with open(raw, "rb") as r:
                        r.seek(2 * n_ai * added[0] * step)
                        data = np.frombuffer(r.read(2 * n_ai * pr.sweep_frames), "<i2").reshape(-1, n_ai)
                    if episodic:
                        f.add(data, offset + added[0] * step / pr.rate_hz)
                    else:
                        with open(f.path, "ab") as fh:
                            fh.write(data.tobytes())
                    added[0] += 1
                if last:
                    return
                time.sleep(0.1)

        writer = threading.Thread(target=grow, daemon=True)
        writer.start()
        st.state = "recording" if st.saving else "playing"
        st.message = f"{tg.mode}, {pr.path.name}" + (f", from sweep {first + 1}" if first else "")
        checked = [time.monotonic()]
        why = ["done"]
        target: list = [None]

        def progress(n: int) -> None:
            st.frames = n
            if self._stop.is_set():
                why[0] = "stop"
                self._stop_file.touch()
                return
            if time.monotonic() - checked[0] >= TELEGRAPH_EVERY_S:      # the amplifier, while the run goes
                checked[0] = time.monotonic()
                now = self.read_telegraph(self.serial, self.channel, timeout=0.5)
                if now is not None:
                    st.telegraph = now
                    if _scales(now) != _scales(tg) and not self._pause.is_set():
                        st.notes.append(f"the amplifier changed ({_words(tg)} -> {_words(now)}): paused after the "
                                        "sweep under way; Resume records on into a new file with the new scales")
                        st.state = "pausing"
                        self._pause.set()
            if self._pause.is_set():
                if target[0] is None:              # the sweep under way when Pause came: stop once it is complete
                    target[0] = (n // step) * step + pr.sweep_frames if episodic else 0
                if n >= target[0]:                 # (a sweep begun after it is dropped, and played again on Resume)
                    why[0] = "pause"
                    st.message = "paused" + (f" after sweep {first + target[0] // step + 1}" if episodic else "")
                    self._stop_file.touch()

        line = helper.record({"ai": pr.ai, "period_us": pr.interval_us, "frames": total,
                              "hold": acquire.holding_codes(pr, cal), "waveform": str(wave) if episodic else "",
                              "out": str(raw)}, progress)
        done.set()
        writer.join()
        if "error" in line:
            raise RuntimeError(line["error"])
        if why[0] == "done" and line.get("done", total) < total:
            why[0] = "stop"
        return why[0], added[0] if episodic else 0
