"""Phase 4: a cell's Notes.txt read (labels, details, a near-miss number, a file
that disagrees with its note), episodes from the notes and from the files
alone, rig profiles learned and matched (a second rig gets its own), the audit
trail (recorded, undone, exported), and the health timeline's drift warnings."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clampwise import simulate
from clampwise.project import Project, health, notes, organize, rigs, sessions

NOTES = """6 wk adolescent female mouse
Recording CA3 pyramidal neuron
Drugs:
NBQX 10 uM
Gabazine 5 uM

File (baseline): 2415268

Depol (5s): 24125269
5s: 24125270

Depol (1s): 24125271
1s: 24125272
"""


@pytest.fixture
def session(tmp_path):
    """A cell folder: Notes.txt and five files (the baseline's number misspelled in the
    note), the step files simulated: the "5 s" step lasts 2 s."""
    recs, _ = simulate.dsi_cell(durations=(2.0, 1.0), trials=1, gap_s=1.0)
    names = ["24125268", "24125269", "24125270", "24125271", "24125272"]     # the note says 2415268
    order = [recs[0], recs[1], recs[2], recs[4], recs[5]]
    folder = tmp_path / "Mouse1" / "Slice1" / "cell1"
    folder.mkdir(parents=True)
    (folder / "Notes.txt").write_text(NOTES, encoding="utf-8")
    paths = {}
    for name, rec in zip(names, order):
        p = folder / f"{name}.abf"
        p.write_bytes(b"")
        paths[p] = rec
    return folder, paths


def test_the_notes_label_the_files_and_flag_what_disagrees(session):
    folder, paths = session
    n = notes.read(folder, list(paths))
    assert (n.age, n.sex, n.species, n.region, n.cell_type) == ("6 wk", "female", "mouse", "CA3", "pyramidal neuron")
    assert n.drugs == ["NBQX 10 uM", "Gabazine 5 uM"]
    base = n.entries[0]
    assert base.role == "baseline" and base.path.name == "24125268.abf" and "one digit away" in base.doubt
    assert [e.role for e in n.entries] == ["baseline", "step", "after", "step", "after"]
    flags = notes.check(n, lambda p: paths[Path(p)])
    assert any("the note says a 5 s step; the file's step lasts 2 s" in f for f in flags)
    assert not any("1 s step" in f for f in flags)


def test_episodes_from_the_notes_and_from_the_files_agree(session):
    folder, paths = session
    load = lambda p: paths[Path(p)]                     # noqa: E731
    from_notes = sessions.group(sorted(paths), load, notes.read(folder, list(paths)))
    alone = sessions.group(sorted(paths), load)
    assert len(from_notes) == len(alone) == 2
    for a, b in zip(from_notes, alone):
        assert [(p.name, r) for p, r in a.members] == [(p.name, r) for p, r in b.members]
    assert [r for _, r in alone[0].members] == ["before", "step", "after"]
    assert from_notes[0].label == "2 s step (the note says 5 s)"
    # a recording that starts long after the step is kept with the doubt said
    infos = [sessions.FileInfo(Path("a.abf"), 0, 60, "Baseline", "recording"),
             sessions.FileInfo(Path("b.abf"), 61, 63, "DSI_Step_2sec", "step", 2.0),
             sessions.FileInfo(Path("c.abf"), 200, 260, "Baseline", "recording")]
    (ep,) = sessions.infer(infos)
    assert ep.role_of(Path("c.abf")) == "after" and "137 s after" in ep.doubts[0]


def test_a_second_rig_gets_its_own_profile(tmp_path):
    store = rigs.Rigs(tmp_path / "rigs.json")
    rig1 = [rigs.Fingerprint("ABF", 6, 24, "Im_prime", "pA", "DSI_Step_5sec"),
            rigs.Fingerprint("ABF", 6, 24, "Im_prime", "pA", "Baseline_DSI")]
    rig2 = [rigs.Fingerprint("ABF", 8, 26, "IN 0", "pA", "opto_grid")]
    store.learn("Rig 1", rig1)
    store.learn("Opto rig", rig2)
    again = rigs.Rigs(tmp_path / "rigs.json")                       # kept for the next session
    assert set(again.rigs) == {"Rig 1", "Opto rig"}
    assert again.match(rigs.Fingerprint("ABF", 6, 24, "Im_prime", "pA", "Baseline_DSI")).rig.name == "Rig 1"
    assert again.match(rigs.Fingerprint("ABF", 8, 26, "IN 0", "pA", "other")).rig.name == "Opto rig"
    m = again.match(rigs.Fingerprint("ABF", 6, 24, "Vm_sec2", "pA", "DSI_Step_3sec"))
    assert m.rig.name == "Rig 1" and "has not been seen" in m.doubts[0]
    assert again.match(rigs.Fingerprint("ABF", 1, 0, "x", "mV")).rig is None          # new hardware: no rig
    again.learn("Rig 1b", [rigs.Fingerprint("ABF", 6, 24, "Left 1", "mV", "Blue_LED")])   # same hardware
    tie = again.match(rigs.Fingerprint("ABF", 6, 24, "Other", "nA", "x"))
    assert tie.rig is None and any("matches as well" in d for d in tie.doubts)
    with pytest.raises(ValueError):
        again.learn("Rig 1", rig2)                                   # other hardware is not one rig


def test_the_audit_trail_records_undoes_and_exports(tmp_path):
    import pyabf.abfWriter

    from clampwise.core.recording import Identity

    folder = tmp_path / "A1" / "Slice1" / "cell1"
    folder.mkdir(parents=True)
    path = folder / "r.abf"
    pyabf.abfWriter.writeABF1(np.zeros((2, 1000), np.float32), str(path), 10_000, units="pA")
    p = Project.open(tmp_path)
    p.add_recordings([path], hash_files=False)
    p.set_identity(path, Identity(condition="carbachol"))
    p.set_setting("events", "template_decay_ms", 12.0)
    p.set_file_review(path, "bad", "Rs too high")
    with p.acting("Notes.txt"):
        p.set_feature("heat_maps", False)
    entries = p.audit()
    kinds = [e["kind"] for e in entries]
    assert kinds[:3] == ["feature", "decision added", "setting"] and entries[0]["by"] == "Notes.txt"
    ident_entry = next(e for e in entries if e["kind"] == "identity")
    assert p.revert(ident_entry["id"]).startswith("undid") and p.identity(path).condition == ""
    setting_entry = next(e for e in entries if e["kind"] == "setting")
    p.revert(setting_entry["id"])
    assert "template_decay_ms" not in p.settings().get("events", {})
    review = next(e for e in entries if e["kind"] == "decision added")
    p.revert(review["id"])
    assert p.file_review(path) == ("", "")
    assert p.revert(review["id"]) == "nothing to undo"                 # once only
    table = pd.read_csv(p.write_audit(tmp_path / "audit.csv"))
    assert len(table) == len(p.audit()) and set(table.By) >= {"you", "Notes.txt"}


def test_the_notes_are_applied_where_nothing_was_typed(tmp_path, session):
    folder, paths = session
    p = Project.open(folder.parents[2])
    p.add_recordings(list(paths), hash_files=False)
    from clampwise.core.recording import ExperimentMetadata

    typed = next(iter(paths))
    p.set_metadata(typed, ExperimentMetadata(age="P40"))
    org = organize.work_out([(q, "Mouse1/Slice1/cell1", "") for q in paths], lambda q: paths[Path(q)])
    assert org.count() == 2 and org.describe(folder / "24125270.abf") == "01 after, 02 before"
    assert organize.apply_notes(p, org) == len(paths)
    assert p.identity(folder / "24125269.abf").label == "Depol (5s)"
    assert p.metadata(typed).age == "P40" and p.metadata(folder / "24125269.abf").age == "6 wk"   # typed stays
    assert [d.name for d in p.metadata(folder / "24125269.abf").drugs] == ["NBQX 10 uM", "Gabazine 5 uM"]
    by = [e["by"] for e in p.audit()]
    assert by[-1] == "you" and set(by[:-1]) == {"Notes.txt"}           # the typed age, then the notes


def test_health_drift_warnings():
    rows = [("a.abf", m, "rs", 10.0 + (5.0 if m > 20 else 0.0), "") for m in range(40)]
    rows += [("a.abf", m, "holding", -100.0 - (150.0 if m > 30 else 0.0), "") for m in range(40)]
    frame = pd.DataFrame(rows, columns=["file", "minute", "measure", "value", "source"])
    said = health.drift(frame, {"rs_change_pct": 20.0, "holding_change": 100.0, "noise_ratio": 2.0}, "pA")
    assert any(s.startswith("Rs changed by more than 20%") and "at 21.0 min" in s for s in said)
    assert any(s.startswith("the holding moved more than 100 pA") and "at 31.0 min" in s for s in said)
    assert dt.datetime  # (kept imported for the session fixture's clock)
