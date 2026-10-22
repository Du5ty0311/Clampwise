"""How much of the PC Clampwise's Live takes: idle, then with the camera's picture live.

Opens Clampwise's window from source, shows Live for ``--idle`` seconds, then (unless ``--no-camera``) takes the
camera (its view's Use here: reading it moves nothing) for ``--camera`` seconds, and measures every second: the
process's CPU (in % of one core) and each thread's, its memory, the longest gap of the window's event loop (a 20 ms
timer that should fire every 20 ms), and the camera's pictures taken and drawn. ``--cores 2 --low-priority`` stands
in for a cheaper PC. The patterned light and the acquisition are never taken. Numbers go to ``--out``
(profile-<time>.json) and a summary is printed.

    python -I tools/rigtest/profile_live.py [--idle 20] [--camera 40] [--cores 2] [--low-priority] [--offscreen]

Budgets (a 4-core laptop without a graphics card): idle Live under 5% of one core; the camera live at full
resolution under 35% of one core with every picture drawn the display can take; no event-loop gap over 100 ms;
memory flat.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

BUDGET = {"idle_cpu_pct": 5.0, "camera_cpu_pct": 35.0, "gap_ms": 100.0}


IDENTS: dict[int, str] = {}                    # Qt's threads by their Python ids, for the sampler


def _name_qthreads(names: dict) -> None:
    """Qt's threads that run Python (the camera's, the rig's readout, background work) named by their class."""
    from clampwise.app import worker
    from clampwise.app.screens import camera_panel, live

    for cls in (camera_panel.CameraThread, getattr(live, "RigThread", None), worker.Task):
        if cls is None or getattr(cls, "_named", False):
            continue
        run = cls.run

        def named(self, _run=run, _label=cls.__name__):
            names[threading.get_native_id()] = _label
            IDENTS[threading.get_ident()] = _label
            return _run(self)

        cls.run, cls._named = named, True


def measure(window, proc, seconds: float, phase: str, names: dict) -> list[dict]:
    """Every second for ``seconds``, in Qt's own event loop (as Clampwise runs): the process's CPU less this
    harness's own (each thread's times are read on a thread of the harness's, subtracted), the event loop's longest
    gap, the pictures taken and drawn, the memory."""
    from pyqtgraph.Qt import QtCore

    cam = window.screens["live"].camera
    beats: list[float] = []
    beat = QtCore.QTimer()
    beat.setTimerType(QtCore.Qt.PreciseTimer)
    beat.timeout.connect(lambda: beats.append(time.perf_counter()))
    rows: list[dict] = []
    latest: dict = {}
    stop = threading.Event()

    def per_thread():                                   # 20 ms a call: kept off the window's thread
        names["harness"] = "harness"
        while not stop.is_set():
            latest["threads"] = proc.threads_s()
            latest["harness_id"] = threading.get_native_id()
            stop.wait(1.0)

    reader = threading.Thread(target=per_thread, name="harness", daemon=True)
    reader.start()
    state = {"cpu": proc.cpu_s(), "t": time.perf_counter(), "th": {}, "harness": 0.0,
             "taken": cam._thread.taken if cam._thread is not None else 0, "shown": cam._shown}

    def sample():
        cpu1, t1 = proc.cpu_s(), time.perf_counter()
        th1 = dict(latest.get("threads", {}))
        hid = latest.get("harness_id")
        harness = th1.get(hid, 0.0) if hid else 0.0
        dt = t1 - state["t"]
        gaps = [b - a for a, b in zip(beats, beats[1:])]
        beats[:] = beats[-1:]
        threads = {}
        names.update(_common.thread_names())
        for tid, s in th1.items():
            if tid == hid:
                continue
            used = (s - state["th"].get(tid, s)) / dt * 100
            if used >= 0.5:
                threads[names.get(tid, "main" if tid == threading.main_thread().native_id else f"thread {tid}")] =                     round(used, 1)
        taken = cam._thread.taken if cam._thread is not None else 0
        ws, private = proc.memory_mb()
        own = max(0.0, harness - state["harness"])
        rows.append({"phase": phase, "cpu_pct": round(max(0.0, cpu1 - state["cpu"] - own) / dt * 100, 1),
                     "gap_ms": round(max(gaps) * 1000, 1) if gaps else None, "threads": threads,
                     "taken_fps": round((taken - state["taken"]) / dt, 1),
                     "drawn_fps": round((cam._shown - state["shown"]) / dt, 1),
                     "working_set_mb": round(ws, 1), "private_mb": round(private, 1)})
        state.update(cpu=cpu1, t=t1, th=th1, harness=harness, taken=taken, shown=cam._shown)
        r = rows[-1]
        print(f"{phase:7s} cpu {r['cpu_pct']:5.1f}%  gap {r['gap_ms'] or 0:6.1f} ms  taken {r['taken_fps']:5.1f}  "
              f"drawn {r['drawn_fps']:5.1f} fps  memory {ws:6.0f} MB  " +
              ", ".join(f"{k} {v}%" for k, v in sorted(threads.items(), key=lambda kv: -kv[1])[:4]))

    loop = QtCore.QEventLoop()
    every = QtCore.QTimer()
    every.timeout.connect(sample)
    QtCore.QTimer.singleShot(int(seconds * 1000), loop.quit)
    beat.start(20)
    every.start(1000)
    loop.exec_()
    every.stop()
    beat.stop()
    stop.set()
    return rows


def summary(rows: list[dict], phase: str) -> dict:
    r = [x for x in rows if x["phase"] == phase][1:] or [x for x in rows if x["phase"] == phase]   # the first warms up
    if not r:
        return {}
    cpu = sorted(x["cpu_pct"] for x in r)
    return {"cpu_pct_median": cpu[len(cpu) // 2], "cpu_pct_max": cpu[-1],
            "gap_ms_max": max(x["gap_ms"] or 0 for x in r), "taken_fps": round(sum(x["taken_fps"] for x in r) / len(r), 1),
            "drawn_fps": round(sum(x["drawn_fps"] for x in r) / len(r), 1),
            "memory_mb_first_last": [r[0]["private_mb"], r[-1]["private_mb"]]}


def main() -> int:
    p = _common.parser(__doc__)
    p.add_argument("--idle", type=float, default=20.0)
    p.add_argument("--camera", type=float, default=40.0)
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--cores", type=int, default=0, help="only this many cores (a cheaper PC)")
    p.add_argument("--low-priority", action="store_true")
    p.add_argument("--exposure", type=float, default=0.0, help="ms (0: as the camera is)")
    p.add_argument("--stacks", action="store_true", help="also where each thread spends its time")
    p.add_argument("--low-power-mode", action="store_true", help="Clampwise's low-power mode on (Settings)")
    args = p.parse_args()
    if args.low_power_mode:
        from clampwise.app import power

        power.enabled = lambda settings=None: True
    proc = _common.Process()
    pc = proc.cheaper(args.cores, args.low_priority) + (", low-power mode" if args.low_power_mode else "")
    names: dict[int, str] = {}
    window, app = _common.open_window(args)
    _name_qthreads(names)
    window.go("live")
    live = window.screens["live"]
    sampler = _common.Sampler(idents=IDENTS).start() if args.stacks else None
    print(f"Live open on {pc}; idle for {args.idle:g} s")
    rows = measure(window, proc, args.idle, "idle", names)
    if not args.no_camera:
        live.use_camera()
        if not _common.pump(20.0, until=lambda: live.camera.last is not None):
            print(f"the camera gave no picture: {live.camera.status.text()}")
        else:
            cam = live.camera
            if args.exposure:
                cam.auto.setChecked(False)             # the exposure as asked, not as the light would have it
                cam.exposure.setValue(args.exposure)
                cam._exposure()                        # as typed: editingFinished applies it
            _common.pump(2.0)
            f = cam.last
            print(f"camera live: {f.rgb.shape[1]} x {f.rgb.shape[0]}, {f.exposure_ms:g} ms; for {args.camera:g} s")
            shot = _common.screenshot(window, "live-camera", args.out)
            rows += measure(window, proc, args.camera, "camera", names)
            print(f"screenshot: {shot}")
    if sampler is not None:
        print("where the time goes (share of each thread's samples):")
        for line in sampler.report(40):
            print("  " + line)
    _common.leave_safely()
    result = {"pc": pc, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "budget": BUDGET,
              "idle": summary(rows, "idle"), "camera": summary(rows, "camera"), "rows": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"profile-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for phase in ("idle", "camera"):
        s = result[phase]
        if s:
            limit = BUDGET[f"{phase}_cpu_pct"]
            print(f"{phase}: cpu median {s['cpu_pct_median']}% (budget {limit:g}%: "
                  f"{'within' if s['cpu_pct_median'] <= limit else 'OVER'}), worst gap {s['gap_ms_max']} ms "
                  f"({'within' if s['gap_ms_max'] <= BUDGET['gap_ms'] else 'OVER'} {BUDGET['gap_ms']:g} ms), "
                  f"taken {s['taken_fps']} / drawn {s['drawn_fps']} fps, memory {s['memory_mb_first_last']} MB")
    print(f"numbers: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
