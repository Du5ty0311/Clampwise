"""The rig's records in the analyses (what the rig's programs applied, detected): a line of
unknown kind that fired while the patterned light waited on its external trigger is light; a line named for an
electrical stimulator keeps its kind, the light's session said beside it; a project's recording with no test step
and nothing typed takes Rs from the Membrane Tests the acquisition's Lab Book logged around it, says where, and
is analyzed again when that record changes; the feature turns it off."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from test_rig_labbook import RTF

from clampwise import simulate
from clampwise.analysis import stimulus
from clampwise.core import Channel, Role
from clampwise.project import Project
from clampwise.rig import evidence, labbook, polyscan


def _light(waiting: bool = True) -> evidence.Evidence:
    return evidence.Evidence(session=polyscan.Session(Path("20261001_151431"), "20261001_151431", sequence="Grid1.seq",
                                                      regions=101, externally_triggered=waiting,
                                                      trigger_edge="RisingEdge"))


def _plain_line():
    rec = simulate.evoked(n_sweeps=2).recording
    rec.channels = [c if c.index != 2 else Channel(2, "IN 3", "V", Role.TRIGGER, True) for c in rec.channels]
    return rec


def test_a_line_of_unknown_kind_that_fired_while_the_light_waited_is_light():
    rec = _plain_line()
    assert stimulus.find_stimulus(rec).kind == "unknown"
    s = stimulus.find_stimulus(rec, rig=_light())
    assert s.kind == "light" and s.source == "light channel IN 3"
    assert ("light: the patterned light's session 20261001_151431 was running, waiting on its external trigger "
            "(pattern sequence Grid1.seq, 101 regions), and this line fired") in s.reason
    assert stimulus.find_stimulus(rec, rig=_light(waiting=False)).kind == "unknown"     # open, not waiting on it


def test_a_line_named_electrical_keeps_its_kind_and_the_light_is_said_beside_it():
    rec = simulate.evoked(n_sweeps=2, light=False).recording                           # its trigger line: AM_Stim
    s = stimulus.find_stimulus(rec, rig=_light())
    assert s.kind == "electrical" and "if this line triggered it, the stimulus is light" in s.reason


def _cell(tmp_path):
    cell = tmp_path / "R007" / "Slice1" / "cell1"
    cell.mkdir(parents=True)
    f = cell / "26o01000.abf"
    f.write_bytes(b"recorded")                  # the store hashes a file; the samples are the simulator's
    p = Project.open(tmp_path / "R007")
    p.add_recordings([f])
    return p, f


def _rows(result):
    return {r["check"]: r for r in result.results["quality"].tables["checks"]}


def test_rs_comes_from_the_lab_book_when_the_file_has_no_test_step_and_nothing_is_typed(tmp_path):
    labbook.DEFAULT.write_text(RTF, encoding="cp1252")            # 26o01000.abf: Ra 14.9 before, 19.4 after
    p, f = _cell(tmp_path)
    first = p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording)
    meta = first.recording.metadata
    assert meta.rs_typed_mohm == pytest.approx((14.9, 19.4)) and meta.rs_source == "the acquisition's Lab Book"
    rs = _rows(first)["Series resistance"]
    assert rs["value"] == "19.4 MOhm (Clampex's Membrane Test, read from the acquisition's Lab Book)"
    assert any(n.startswith("Acquisition's Lab Book: 26o01000.abf completed, protocol Simple_IV.pro; Membrane Test "
                            "before: Cm 52.97 pF") for n in first.notes)
    assert p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording).results["quality"].from_cache
    labbook.DEFAULT.write_text(RTF.replace("19.4 MOhms", "30.0 MOhms"), encoding="cp1252")
    os.utime(labbook.DEFAULT, (time.time() + 5, time.time() + 5))
    again = p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording)
    assert not again.results["quality"].from_cache and _rows(again)["Series resistance"]["value"].startswith("30.0")
    p.set_feature("rig_records", False)
    off = p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording)
    assert _rows(off)["Series resistance"]["result"] == "not assessed"
    p.close()


def test_a_name_alone_does_not_match_a_lab_book_record(tmp_path):
    labbook.DEFAULT.write_text(RTF, encoding="cp1252")
    elsewhere = tmp_path / "other" / "26o01000.abf"                # the same name in another cell's folders
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(b"x")
    assert evidence.for_recording(simulate.evoked().recording, path=elsewhere).record is None


def test_one_folder_in_common_is_not_enough(tmp_path):
    # the review's case: another slice's cell1 shares only the cell's folder name with the Lab Book's record
    labbook.DEFAULT.write_text(RTF, encoding="cp1252")
    other = tmp_path / "R007" / "Slice2" / "cell1" / "26o01000.abf"
    assert evidence.for_recording(simulate.evoked().recording, path=other).record is None
    same = tmp_path / "R007" / "Slice1" / "cell1" / "26o01000.abf"
    assert evidence.for_recording(simulate.evoked().recording, path=same).record is not None


def test_a_typed_rs_stays_and_the_lab_book_fills_only_the_rest(tmp_path):
    from clampwise.core.recording import ExperimentMetadata

    labbook.DEFAULT.write_text(RTF, encoding="cp1252")            # 14.9 before, 19.4 after
    p, f = _cell(tmp_path)
    p.set_metadata(f, ExperimentMetadata(rs_typed_mohm=(float("nan"), 25.0)))          # only the end typed
    meta = p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording).recording.metadata
    assert meta.rs_typed_mohm == pytest.approx((14.9, 25.0))
    assert meta.rs_source == "the acquisition's Lab Book, where nothing was typed"
    p.close()


def test_the_cache_keys_on_the_rs_the_lab_book_gives_not_on_the_session():
    a, b = _light(), _light()
    b.session.name = "another session"
    assert a.digest() == b.digest() == ""                         # no Rs: nothing more to key on
    rec = labbook.FileRecord("x.abf", 1, before=labbook.MembraneTest(1, {"ra_mohm": 14.9}))
    one, two = evidence.Evidence(record=rec), evidence.Evidence(record=rec, session=b.session)
    assert one.digest() == two.digest() != ""
    other = evidence.Evidence(record=labbook.FileRecord("x.abf", 1, before=labbook.MembraneTest(1, {"ra_mohm": 20.0})))
    assert other.digest() != one.digest()


def test_an_unreadable_record_is_no_evidence_never_a_failed_analysis(monkeypatch):
    def broken(*a, **k):
        raise ValueError("a Lab Book line Clampwise cannot read")

    monkeypatch.setattr(labbook, "current", broken)
    assert evidence.for_recording(simulate.evoked().recording).empty


def test_the_rs_timeline_shows_the_lab_book_s_rs(tmp_path):
    from clampwise.io import load
    from clampwise.project import health

    labbook.DEFAULT.write_text(RTF, encoding="cp1252")
    import pyabf.abfWriter
    import numpy as np

    cell = tmp_path / "R007" / "Slice1" / "cell1"
    cell.mkdir(parents=True)
    f = cell / "26o01000.abf"
    pyabf.abfWriter.writeABF1(np.random.default_rng(0).normal(-65, 0.5, (3, 2000)).astype(np.float32), str(f), 10_000,
                              units="mV")
    p = Project.open(tmp_path / "R007")
    p.add_recordings([f])
    p.analyze(f)
    frame = health.timeline(p, [f], load).rows
    rs = frame[frame.measure == "rs"]
    assert list(rs.value.round(1)) == [14.9, 19.4]
    assert set(rs.source) == {"Clampex's Membrane Test, read from the acquisition's Lab Book"}
    p.close()
