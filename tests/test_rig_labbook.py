"""Clampex's Lab Book read: its RTF paragraphs to events, each file's record with the protocol open and the Membrane
Tests before and after it (a new cell's folder ends the last file's), units to the lab's, and a moved recording
matched by its name and deepest folders."""

from __future__ import annotations

import pytest

from clampwise.rig import labbook

RTF = r"""{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fnil\fcharset0 Courier New;}}
\viewkind4\uc1\pard\f0\fs18  Opening Clampex logging all events.\par
 Acquired data will be written to 'E:\\lab_data\\R007\\Slice1\\cell1'.\par
 Protocol 'E:\\Protocols\\Lab\\Simple_IV.pro' opened.\par
 Membrane test initiated using analog output channel 0 and analog input channel 0.\par
 Total resistance  (Rt) = 1.2 GOhms\par
 Holding current (Hold) = -12.5 pA\par
 Membrane test initiated using analog output channel 0 and analog input channel 0.\par
 Membrane capacitance (Cm) = 52.97 pF\par
 Membrane resistance  (Rm) = 110.2 MOhms\par
 Access resistance    (Ra) = 14.9 MOhms\par
 Time constant       (Tau) = 693.6 \'b5s\par
 Holding current    (Hold) = -0.12 nA\par
 Membrane test terminated.\par
 E:\\lab_data\\R007\\Slice1\\cell1\\26o01000.abf started at 02:45:32 stopwatch time.\par
 E:\\lab_data\\R007\\Slice1\\cell1\\26o01000.abf aborted at 02:45:40 stopwatch time.\par
 E:\\lab_data\\R007\\Slice1\\cell1\\26o01000.abf started at 02:45:43 stopwatch time.\par
 E:\\lab_data\\R007\\Slice1\\cell1\\26o01000.abf completed at 02:46:50 stopwatch time.\par
 Membrane test initiated using analog output channel 0 and analog input channel 0.\par
 Access resistance    (Ra) = 19.4 MOhms\par
 Membrane test terminated.\par
 Acquired data will be written to 'E:\\lab_data\\R007\\Slice1\\cell2'.\par
 Membrane test initiated using analog output channel 0 and analog input channel 0.\par
 Access resistance    (Ra) = 9.1 MOhms\par
 Protocol 'E:\\Protocols\\Lab\\BLA_intrinsic_CC_-90mV_2xPhotostim.pro' opened.\par
 E:\\lab_data\\R007\\Slice1\\cell2\\26o01001.abf started at 03:48:05 stopwatch time.\par
 E:\\lab_data\\R007\\Slice1\\cell2\\26o01001.abf completed at 03:48:21 stopwatch time.\par
 Holding on 'Cmd 0' => -70 mV\par
 Closing Clampex.\par
\par
}
"""


def _book(tmp_path=None) -> labbook.LabBook:
    return labbook.LabBook(labbook.parse(labbook.rtf_lines(RTF)))


def test_the_rtf_becomes_one_event_a_line():
    lines = labbook.rtf_lines(RTF)
    assert lines[0] == "Opening Clampex logging all events."
    assert "E:\\lab_data\\R007\\Slice1\\cell1" in lines[1]                  # escaped backslashes undone
    assert any(line.endswith("693.6 \u00b5s") for line in lines)               # \'b5 is the micro sign


def test_each_file_has_its_protocol_outcome_and_the_tests_around_it():
    book = _book()
    first = book.record(r"E:\lab_data\R007\Slice1\cell1\26o01000.abf")
    assert first.outcome == "completed" and first.started_s == 2 * 3600 + 45 * 60 + 43       # the restart, not the abort
    assert first.protocol.endswith("Simple_IV.pro") and first.folder.endswith("cell1")
    assert first.before.values == pytest.approx({"cm_pf": 52.97, "rm_mohm": 110.2, "ra_mohm": 14.9, "tau_us": 693.6,
                                                 "hold_pa": -120.0})
    assert first.after.values == {"ra_mohm": 19.4}                           # the test after it, in its folder
    second = book.record(r"E:\lab_data\R007\Slice1\cell2\26o01001.abf")
    assert second.before.values == {"ra_mohm": 9.1}                          # cell2's own test, not cell1's end
    assert second.protocol.endswith("2xPhotostim.pro") and second.after is None
    assert [e.value for e in book.entries if e.kind == "holding"] == [-70.0]


def test_units_become_the_lab_s():
    seal = [e for e in _book().entries if e.kind == "test_value" and e.key == "rt_mohm"]
    assert seal[0].value == pytest.approx(1200.0)                            # 1.2 GOhm in MOhm


def test_a_moved_recording_is_matched_by_name_and_its_deepest_folders():
    book = _book()
    moved = book.record(r"E:\Archive\lab_data\R007\Slice1\cell2\26o01001.abf")
    assert moved is not None and moved.folder.endswith("cell2")
    assert book.record(r"E:\elsewhere\26o09999.abf") is None


def test_the_lab_book_is_read_again_only_when_it_changes(tmp_path):
    path = tmp_path / "Clampex LabBook.rtf"
    path.write_text(RTF, encoding="cp1252")
    first = labbook.current(path)
    assert labbook.current(path) is first and len(first.files) == 3
    import os
    import time

    path.write_text(RTF.replace("Closing Clampex", "Opening Clampex logging all events"), encoding="cp1252")
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert labbook.current(path) is not first
    assert labbook.current(tmp_path / "missing.rtf") is None


def test_a_new_cell_s_folder_starts_afresh():
    # the review's case: cell1's last test must not become the first cell3 file's 'before'
    lines = ["Acquired data will be written to 'E:/R007/Slice1/cell1'.",
             "E:/R007/Slice1/cell1/26o01000.abf started at 01:00:00 stopwatch time.",
             "E:/R007/Slice1/cell1/26o01000.abf completed at 01:01:00 stopwatch time.",
             "Membrane test initiated using analog output channel 0 and analog input channel 0.",
             "Access resistance    (Ra) = 19.4 MOhms",
             "Acquired data will be written to 'E:/R007/Slice1/cell3'.",
             "E:/R007/Slice1/cell3/26o01001.abf started at 01:05:00 stopwatch time."]
    book = labbook.LabBook(labbook.parse(lines))
    assert book.record("E:/R007/Slice1/cell1/26o01000.abf").after.values == {"ra_mohm": 19.4}
    assert book.record("E:/R007/Slice1/cell3/26o01001.abf").before is None


def test_a_malformed_value_is_kept_as_text_not_an_error():
    entries = labbook.parse(["Access resistance    (Ra) = . MOhms", "Holding on 'Cmd 0' => 1.2.3 mV",
                             "Total resistance  (Rt) = 63.0 MOhms"])
    assert [e.kind for e in entries] == ["other", "other", "test_value"]
