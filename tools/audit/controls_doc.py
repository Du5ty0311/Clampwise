"""Write docs/development/CONTROLS.md: every control of the Workbench, the Figures screen, the figure builder and
their dialogs, with the effect that was checked.

    python tools/audit/controls_doc.py            # runs tests/test_every_control.py, then writes the list
    python tools/audit/controls_doc.py runs.jsonl # writes the list from a run already recorded

The list is made from the runs themselves: ``tests/test_every_control.py`` presses each control on a simulated
recording of every type (and on copies of the lab's own files when they are named on the machine) and records
what changed;
this file turns those records into the table. ``CHECKED`` adds, for the controls the test also checks by name
(``test_..._does_what_its_labels_say`` and the regression tests), what was asserted. A control the test finds that
is not in the list fails the test, so the list cannot fall behind the screens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "development" / "CONTROLS.md"

# what the test asserts by name, beyond "it changed something": screen | control -> the assertion, in words
CHECKED = {
    "Workbench | Keep this measurement": "the measurement is in the project's kept list and in Results; Undo takes "
                                         "it out, Redo puts it back",
    "Workbench | Save as template...": "the template is in the project and in the Templates list; Undo removes it",
    "Workbench details | Add region": "one more region, named \"region n\", in the list and on the trace; Undo "
                                      "removes it",
    "Workbench details | Remove region": "the selected region is gone; the table keeps working when no measure "
                                         "region is left (it raised IndexError); Undo brings it back",
    "Workbench details | Repeat on each stimulus pulse": "the table has one row per pulse per sweep (2 x the "
                                                         "sweeps), one per sweep when a single pulse is asked",
    "Workbench | Sweeps shown and measured as": "\"average\": one trace named \"average of n\" is measured",
    "Workbench | Channel": "the table's units become the channel's (Baseline (V) on AM_Stim)",
    "Workbench | Stimulus": "a dashed line at each pulse (2), none when unticked",
    "Workbench | Command": "one more curve (the command) under the trace",
    "Workbench details | Add transform > Low-pass (Bessel, 4th order)": "\"low-pass 2000 Hz\" in the list and in "
                                                                        "the measurement",
    "Workbench details | Remove transform": "the selected transform is gone, the other stays",
    "Workbench details | Clear transforms": "no transform left; the button then says there is none to clear",
    "Workbench details | Set onset": "the mode is on (its button pressed, the line above the trace says so); a "
                                     "click on a sweep sets that sweep's onset, and its latency reads 6.0 ms",
    "Workbench details | Leave out a sweep": "the clicked sweep leaves the table with the reason typed; pressed "
                                             "again the mode ends",
    "Workbench details | Clear clicks and exclusions": "no onset set and no sweep left out; Undo restores them",
    "Workbench details | Fit model": "the fit's words name tau, its residuals are drawn",
    "Workbench / Plot | Plot: against": "\"command level\": the points sit at the sweeps' command levels",
    "Workbench details | Load kept": "the kept measurement's fit and regions are back on the screen",
    "Workbench details | Delete kept": "gone from the project; Undo brings it back",
    "Workbench details | Apply here": "the template's fit and regions on this recording, marked as from the template",
    "Workbench details | Run on its protocol's files": "every file of the protocol has the measurement kept; Undo "
                                                       "takes the whole batch back",
    "Workbench details | Delete template": "gone from the project; Apply here then says no template is saved; "
                                           "Undo brings it back",
    "Figures | Show > (each graph the results give)": "that graph is drawn, named in the title and ticked in the "
                                                     "menu; no two panels show the same graph",
    "Figures | Edit": "the builder opens on this graph (its table, its title); off, saying why, when nothing is "
                      "drawn",
    "Figures | ⤢": "the graph fills the area; pressed again, the four are back",
    "Figures | ⧉": "a window of its own, named after the graph",
    "Figures | Across recordings": "one point per analyzed recording; pressed again, the four graphs are back",
    "Figures | Make a figure": "the builder opens, empty",
    "Trace picker | Add as a trace": "the selected sweeps become a named trace; the dialog opens on a recording "
                                     "with several sweeps (it raised on every such recording)",
    "Workbench | Next sweep": "the next sweep is drawn in bold and named (\"sweep 1 (1 of n)\"), its row selected; "
                              "Page Down does the same",
    "Workbench | Previous sweep": "the sweep before; from the first, round to the last",
    "Workbench | Show the sweep alone": "the other sweeps are not drawn",
    "Workbench | All sweeps": "no sweep singled out; every sweep drawn again",
    "Workbench | Cursors": "two lines on the trace; the readout gives each one's time and value (checked against "
                           "the samples), their difference and the slope, and follows a cursor that is dragged",
    "Workbench | Region to the cursors": "the selected region spans the cursors (from the pulse when it repeats "
                                         "on each pulse); Undo puts it back",
    "Workbench | Send to a figure > The traces shown, as a panel of example traces":
        "the builder opens with a trace panel of the chosen sweeps over the time in view, the light's bars at "
        "their recorded times",
    "Workbench | Send to a figure > This measurement's table, as a graph":
        "a second panel: the table of every row measured, the first measure against the sweep",
    "Figure builder | Add panel...": "asks for the data, then for the plot; fills the empty panel, then adds a "
                                     "panel beside it (it filled the first panel again)",
    "Figure builder | Y axis title | double-click": "a box opens over it with its words; Enter keeps the new "
                                                    "words, Esc leaves the old; Undo takes them back",
    "Figure builder | Comparison bracket | drag": "the bracket stays at the height it was dragged to; its text can "
                                                  "be replaced with a typed result",
    "Figure builder | Scale bars | drag": "the corner where the two bars meet moves with the drag",
    "Figure builder | Stimulus bar | drag": "it moves up and down only: its time is the recording's",
    "Figure builder | Panel | drag": "in the grid, it changes places with the panel it is dropped on; placed "
                                     "freely, it moves on the page (on the snap grid when one is set)",
    "Figure builder | Panel's handle (se) | drag": "a freely placed panel (or an inset) is resized",
    "Figure builder | Leave the selected rows out": "the rows are greyed in the table, out of the figure and of "
                                                    "its n, and counted in the legend; Undo puts them back",
    "Figure builder | Test": "the brackets and the legend carry the chosen test's p (Mann-Whitney); a paired test "
                             "pairs the cells",
    "Figure builder | Legend: Place": "the legend moves or goes (\"No legend\" closed the program: the pick list "
                                      "was deleted while still in use)",
    "Export | Format": "a pixel format turns the resolution on and the text-as-shapes box off, each saying why",
    "Plot chooser | Plot: Box plot with every point": "the plot is named under the heading and Use it draws the "
                                                      "panel that way",
    "Data chooser | Use it": "the kind of data picked (a source that cannot be used now is greyed with its reason)",
}

HEAD = """# Every control, and what was checked

Written by `tools/audit/controls_doc.py` from the runs of `tests/test_every_control.py`; do not edit by hand.

The test presses every button, checkbox, pick list and menu action of the Workbench, the Figures screen, the
figure builder (with its choosers of data and plot and its export dialog), the trace picker and the digitizer,
each from the same starting state. In the builder the figure itself is a control too: each kind of thing drawn on
it (a panel, a letter, a title, an axis and its title, the legend, a group, the scale bars, a stimulus bar, the
command, a text, a bracket, a reference line, a mark along x, a potential label, a text on the page) is clicked,
dragged, moved with an arrow key, double-clicked (and rewritten), deleted and right-clicked (each entry of its
menu chosen). All of it on a simulated recording
of every recording type ({types}) and, when the lab's own recordings are named on the machine
(`tests/labdata.py`), on copies of three of them. It
fails when a control raises, changes nothing, makes an edit that Undo does not put back, or is disabled without
saying why. A control found on a screen and missing from this list fails the test too.

Columns: **What changed** is what the harness saw (the thing being edited, the picture, the rest of the screen, a
dialog, a file, the clipboard, a message, a window), over every state the control was pressed in. **Checked by
name** is what the test asserts for that control beyond a change. **Undo** says whether Undo put an edit back
(a dash: the control edits nothing, so there is nothing to undo). **Off when** is what a disabled control's
tooltip said.

{n} controls, pressed {presses} times in {states} states. Last written from {runs}.
"""


def run_tests(out: Path) -> int:
    env = dict(os.environ, CLAMPWISE_CONTROLS_OUT=str(out), QT_QPA_PLATFORM="offscreen")
    return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_every_control.py"), "-q",
                            "-p", "no:cacheprovider"], cwd=ROOT, env=env)


def write(records: list[dict], runs: str) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["name"]].append(r)
    screens: dict[str, list[str]] = defaultdict(list)
    for name in by:
        screens[name.split(" | ")[0]].append(name)
    states = {r["state"] for r in records}
    types = ", ".join(sorted({s.split(",")[0] for s in states if s.endswith(", as opened")
                              and not s.startswith("his ")}))
    lines = [HEAD.format(types=types, n=len(by), presses=sum(r["enabled"] for r in records), states=len(states),
                         runs=runs)]
    order = ["Workbench", "Workbench details", "Figures", "Figure builder", "Data chooser", "Plot chooser", "Export",
             "Trace picker", "Digitizer"]
    for screen in sorted(screens, key=lambda s: (order.index(s) if s in order else 99, s)):
        lines += [f"\n## {screen}\n", "| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |",
                  "|---|---|---|---|---|---|---|"]
        for name in screens[screen]:
            rs = by[name]
            _screen, label, kind = name.split(" | ")
            pressed = [r for r in rs if r["enabled"]]
            effects = sorted({e for r in pressed for e in r["effects"]})
            if not effects and any(r["ticked"] for r in pressed):
                effects = ["already chosen: choosing it again changes nothing"]
            undo = {r["undo"] for r in pressed if r["undo"]}
            undo_word = "NOT RESTORED" if "NOT restored" in undo else "restored" if undo else "–"
            off = sorted({r["tooltip"].strip() for r in rs if not r["enabled"] and r["tooltip"].strip()})
            checked = CHECKED.get(name) or next(
                (v for k, v in CHECKED.items() if k.count(" | ") == 1 and (
                    name.startswith(k + " | ") or (k.split(" | ")[0] == screen.split(" / ")[0]
                                                   and k.split(" | ")[1] == label))), "")
            cell = lambda s: s.replace("|", "/").replace("\n", " ")       # noqa: E731
            lines.append(f"| {screen} | {cell(label)} | {kind} | {cell('; '.join(effects)) or 'never enabled'} | "
                         f"{cell(checked)} | {undo_word} | {cell(' / '.join(off))} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) > 1:
        out, code = Path(sys.argv[1]), 0
    else:
        out = Path(tempfile.mkdtemp(prefix="clampwise_controls_")) / "runs.jsonl"
        code = run_tests(out)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    his = any(r["state"].startswith("his ") for r in records)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(write(records, "the simulated recordings" + (" and copies of the lab's own files" if his else "")),
                   encoding="utf-8", newline="\n")
    bad = sorted({r["name"] for r in records if r["error"]})
    print(f"{DOC}: {len({r['name'] for r in records})} controls" + (f"; RAISED: {bad}" if bad else ""))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
