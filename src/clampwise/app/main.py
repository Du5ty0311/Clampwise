"""Starting the window: ``python -m clampwise.app [FOLDER] [--dark|--light]``."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from . import theme


def log_folder() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# NeuroExpress's own test files (a current-step family, a mEPSC recording, a
# spike train): bundled with the build, read from the data folder otherwise.
DEMO_FILES = ("Test_IV_1.abf", "Test_mEPSC_1.abf", "Test_SpT.ASC")
DEMO_SOURCE = Path(os.environ.get("CLAMPWISE_DEMOS") or Path.home() / "synaptoscope-data" / "neuroexpress_tests")


def demo_source() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen) / "demos" if frozen else DEMO_SOURCE


def demo_folder() -> Path:
    """A writable copy of the demos (the project store lives beside its recordings)."""
    import shutil

    folder = log_folder() / "Demos"
    folder.mkdir(exist_ok=True)
    for name in DEMO_FILES:
        if not (folder / name).is_file():
            shutil.copy2(demo_source() / name, folder / name)
    return folder


# (file, analysis, output, expected, tolerance): NeuroExpress's numbers where it
# reports one, else Clampwise's own from the source run (tests/test_*.py).
SELF_TEST = (
    ("Test_IV_1.abf", "intrinsic", "holding_current", 5.70, 0.05),        # NeuroExpress Irest
    ("Test_IV_1.abf", "intrinsic", "r_steady0", 117.36, 2.0),             # NeuroExpress Rend0
    ("Test_mEPSC_1.abf", "events", "n_events", 1073, 0),                  # Clampwise, source run
    ("Test_SpT.ASC", "spike_train", "n_spikes", 1618, 0),                 # NeuroExpress
    ("Test_SpT.ASC", "spike_train", "n_bursts", 136, 0),                  # NeuroExpress
)


def _new_features(project, tmp: Path, paths: list, say) -> None:
    """The self-test of what came after the analyses: Check's file review and Figures'
    graphs (Phase 1), heat maps (2), the figure builder with its exports, statistics and
    digitizer (3), notes, episodes, rigs, cell health and the audit trail (4)."""
    import numpy as np

    from ..analysis import heatmaps
    from ..figure import data, digitize, render, stats
    from ..figure.prepare import prepare
    from ..figure.recipe import PALETTES, Recipe, Spec, pack
    from ..io import load
    from ..project import health, notes, organize, rigs
    from .window import MainWindow

    w = MainWindow(theme.LIGHT, persist=False)
    w.open_project(tmp)
    say("stimulus" not in w.screens and hasattr(w.screens["check"], "step_file")
        and hasattr(w.screens["figures"], "maximize"), "screens: Check steps through files, graphs maximize, "
        "no Stimulus")
    run = project.analyze(tmp / "Test_mEPSC_1.abf")
    maps = {k: build for k, _t, build in heatmaps.catalogue(run)}
    rate, spec_map = maps["heatmap.events.rate"](), maps["heatmap.spectrogram.hann"]()
    say(np.isfinite(rate.matrix).any() and np.isfinite(spec_map.matrix).any(),
        f"heat maps: {len(maps)} for the minis ({rate.matrix.shape[0]} x {rate.matrix.shape[1]} event rates)")
    tables = data.project_tables(project)
    steps = next(t for n, t in tables.items() if n.endswith(": fi"))()
    rng = np.random.default_rng(0)
    nested = data.typed(pd_frame({"Animal": [f"a{k // 6}" for k in range(48)], "Cell": [f"c{k}" for k in range(48)],
                                  "Group": ["control"] * 24 + ["drug"] * 24,
                                  "Value": rng.normal(0, 1, 48) + np.r_[np.zeros(24), np.ones(24)]}))
    spec = Spec([Recipe(data=pack(steps), question="relationship", plot="scatter", x="Current (pA)",
                        y=next(c for c in steps.columns if c.startswith("Rate")), fit="linear", level="row"),
                 Recipe(data=pack(nested), question="compare", plot="estimation", y="Value", group="Group",
                        cell="Cell", animal="Animal", stats=True)], columns=2, size="double")
    for ext in ("pdf", "svg", "png", "tiff"):
        out = render.export(spec, tmp / f"figure.{ext}", dpi=150)
        say(render.read_spec(out).to_json() == spec.to_json(), f"figure exported as {ext} and reopened editable")
    spec.style.apply_theme("Dark slide")
    spec.panels[1].plot = "raincloud"
    spec.panels[1].series = {"drug": {"colour": PALETTES["red and blue"][0], "marker": "D"}}
    out = render.export(spec, tmp / "themed.png", dpi=100)
    say(render.read_spec(out).panels[1].series["drug"]["marker"] == "D", "a themed figure with a group's own look")
    (comp,) = stats.compare(prepare(nested, spec.panels[1]).points)
    say(comp.mixed.startswith("Mixed model"), f"statistics: {comp.sentence()[:110]}...")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot()
    line = PALETTES["tableau 10"][2]
    ax.plot([0, 10], [0, 10], color=line)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.savefig(tmp / "plot.png")
    from matplotlib.colors import to_rgb

    cols, rows = digitize.trace(digitize.load(tmp / "plot.png"), [255 * v for v in to_rgb(line)])
    say(len(cols) > 100, f"digitizer: {len(cols)} columns read from a picture")
    from .builder import Builder

    b = Builder(w, spec)
    b.redraw()
    say("n = 24 cells from 4 animals" in b.caption_text(), "figure builder: the caption says n as cells and animals")
    b.close()
    fs = w.screens["figures"]
    w.go("figures", remember=False)
    w.select(tmp / "Test_IV_1.abf")
    eb = fs.edit_graph(fs.panels[0])
    if eb is not None:
        eb.redraw()
    say(eb is not None and eb.recipe.source.get("kind") == "graph" and bool(eb._drawn)
        and eb._drawn[0].prepared is not None, "a Figures graph opened in the figure builder, editable")
    if eb is not None:
        eb.close()
    (tmp / "Notes.txt").write_text("8 wk male mouse\nRecording CA3 pyramidal neuron\nSteps: Test_IV_1\n",
                                   encoding="utf-8")
    n = notes.read(tmp, paths)
    org = organize.work_out([(p, "demo", "") for p in paths], load)
    say(n.age == "8 wk" and n.region == "CA3", f"Notes.txt read ({n.age}, {n.sex}, {n.region}, {n.cell_type}); "
        f"{org.count()} episode(s) among the demos")
    fp = rigs.fingerprint(paths[0])
    say(fp.format == "ABF", f"rig fingerprint: {fp.describe()}")
    t = health.timeline(project, paths, load)
    say(len(t.rows) > 0, f"cell health: {len(t.rows)} points")
    project.set_setting("events", "template_decay_ms", 9.0)
    entry = project.audit()[0]
    project.revert(entry["id"])
    say("template_decay_ms" not in project.settings().get("events", {}), "audit trail: a change recorded and undone")
    # the lab's LED trigger: 0.45 V pulses on an input named IN 8 (Track R, 2026-10-09)
    from dataclasses import replace

    from .. import simulate
    from ..analysis import pipeline, stimulus
    from ..core.recording import ArraySource

    rec = simulate.evoked(n_sweeps=3, sweep_s=3.0, onsets_s=(1.078,), amplitudes=(-50.0,)).recording
    trigger = np.zeros((3, int(3.0 * rec.sample_rate_hz)))
    trigger[:, 21560:21660] = 0.45
    rec.channels[2] = replace(rec.channels[2], name="IN 8")
    rec.samples = ArraySource([rec.read(0), rec.read(1), trigger])
    rec.protocol = "blue_LED_single_pulse_NA"
    st = stimulus.find_stimulus(rec)
    typed = pipeline.analyze(rec).recording_type
    say(st.kind == "light" and st.channel == "IN 8" and typed == "vc_evoked",
        f"a 0.45 V LED trigger on IN 8 is the light stimulus ({st.kind}, {st.channel}); the file is {typed}")
    w.close()


def pd_frame(d):
    import pandas as pd

    return pd.DataFrame(d)


def _live_replay(tmp: Path, say) -> None:
    """B19's check: a small simulated file whose Rs jumps 12 -> 18 MOhm is written as the acquisition writes one,
    sweep by sweep, while it is read as Live reads it (every 0.5 s); each sweep must be read within 2 s, the jump
    flagged, and nothing left open (the file is renamed at once)."""
    import threading
    import time

    from ..rig import growing, replay
    from ..simulate.abf_file import membrane_test_file

    (tmp / "recorded").mkdir()
    src = membrane_test_file(tmp / "recorded" / "sim_live.abf", n_sweeps=8, rs_mohm=[12.0] * 4 + [18.0] * 4)
    target = tmp / "watched" / src.name
    target.parent.mkdir()
    written: list[float] = []                       # when each sweep's bytes began to be written

    def pace(seconds: float) -> None:
        time.sleep(seconds)
        written.append(time.monotonic())

    writer = threading.Thread(target=replay.write_as_recorded, args=(src, target), kwargs={"sleep": pace})
    writer.start()
    reader, got, late = growing.Growing(target, tmp / "scratch"), [], []
    end = time.monotonic() + 20
    while time.monotonic() < end and len(got) < 8:
        time.sleep(0.5)
        if target.exists():
            for t in reader.look():
                got.append(t)
                late.append(time.monotonic() - written[t.sweep])
    writer.join()
    target.rename(target.with_name("moved.abf"))    # raises if anything still holds it
    change = growing.rs_change(got)
    say(len(got) == 8 and max(late, default=99) < 2.0 and change > 0.2,
        f"live replay: {len(got)} of 8 sweeps read as written, each within {max(late, default=float('nan')):.2f} s; "
        f"Rs {got[0].rs_mohm if got else float('nan'):.1f} -> {got[-1].rs_mohm if got else float('nan'):.1f} MOhm "
        f"({change:+.0%}, flagged over 20%); no file left open")


def self_test() -> int:
    """``--self-test``: what a frozen build must be able to do. Imports every
    analysis module, builds the window (unshown), analyzes the demos through
    the batch's worker processes, checks numbers, writes a workbook. The report
    goes to stdout and to self-test.txt in the log folder; exit 0 means passed."""
    import tempfile
    import time
    import traceback

    lines, failed = [], False

    def say(ok: bool, text: str) -> None:
        nonlocal failed
        failed |= not ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} {text}")

    t0 = time.perf_counter()
    try:
        from ..analysis import batch, registry

        registry.load()
        say(not registry.LOAD_ERRORS, f"{len(registry.REGISTRY)} analyses registered "
                                      f"{'; '.join(registry.LOAD_ERRORS)}")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        from .window import MainWindow

        MainWindow(theme.LIGHT, persist=False).close()
        say(True, "window built")
        # a graph shown keeps its recording memory-mapped until the process ends: the folder is then left
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            import shutil

            from ..project import Project

            paths = [Path(shutil.copy2(demo_source() / n, Path(tmp) / n)) for n in DEMO_FILES]
            project = Project.open(tmp)
            project.add_recordings(paths)
            project.close()
            for o in batch.run(tmp, paths):
                say(o.ok, f"{o.path.name}: {o.recording_type} in {o.seconds:.1f} s {o.error[:300]}")
            project = Project.open(tmp)
            try:
                for name, key, output, expected, tol in SELF_TEST:
                    r = project.analyze(Path(tmp) / name).results.get(key)
                    got = None if r is None else r.values.get(output)
                    say(got is not None and abs(got - expected) <= tol,
                        f"{name} {key}.{output} = {got} (expected {expected} +/- {tol})")
                import openpyxl

                from .. import export

                book = openpyxl.load_workbook(export.write_workbook(project))
                head = book["Current steps"][2]
                say(any(c.comment is not None for c in head) and len(export.write_csv(project)) > 3,
                    f"Excel and CSV written ({len(book.sheetnames)} sheets)")
                _new_features(project, Path(tmp), paths, say)
            finally:
                project.close()
            _live_replay(Path(tmp), say)
        del app
    except Exception:
        say(False, traceback.format_exc())
    lines.append(f"{'FAILED' if failed else 'passed'} in {time.perf_counter() - t0:.1f} s ({__version__})")
    report = "\n".join(lines)
    print(report)
    (log_folder() / "self-test.txt").write_text(report + "\n", encoding="utf-8")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    flags = {a for a in argv[1:] if a.startswith("-")}
    positional = [a for a in argv[1:] if not a.startswith("-")]
    if "--version" in flags:
        print(f"clampwise {__version__}")
        return 0
    if "--self-test" in flags:
        return self_test()
    if "--update" in flags:                              # the update without a window (Settings, Updates has one)
        from .. import update

        try:
            print(update.update_now(launch="--open" in flags))
            return 0
        except Exception as exc:                         # GitHub unreachable, a bad download: said, nothing changed
            print(f"not updated: {exc}")
            return 1
    logging.basicConfig(filename=str(log_folder() / "clampwise-error.log"), level=logging.WARNING,
                        format="\n===== %(asctime)s =====\n%(message)s", encoding="utf-8")
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QtGui.QIcon(str(Path(__file__).parent / "resources" / "icon.png")))
    saved = str(QtCore.QSettings(APP_NAME, APP_NAME).value("theme", "") or "")
    if "--dark" in flags or (saved == "dark" and "--light" not in flags):
        palette = theme.DARK
    elif "--light" in flags or saved == "light":
        palette = theme.LIGHT
    else:
        palette = theme.DARK if theme.system_prefers_dark(app) else theme.LIGHT
    theme.apply(app, palette)
    from .window import MainWindow

    window = MainWindow(palette)
    window.resize(1366, 768)
    window.show()
    QtCore.QTimer.singleShot(4000, window.screens["settings"].startup_check)   # a newer release?
    if positional and Path(positional[0]).is_dir():
        window.open_project(Path(positional[0]))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
