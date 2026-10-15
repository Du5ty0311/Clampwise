"""One look at the rig: the programs' windows and the amplifier at once, the files (Lab Book, the light's
session, the camera's last picture) a moment later and then every few seconds; the rig programs found by role;
times in words a glance takes in."""

from __future__ import annotations

import datetime as dt

from test_rig_camera import XMP, _tiff
from test_rig_labbook import RTF
from test_rig_multiclamp import VC_STATE, FakeCommander
from test_rig_polyscan import _session

from clampwise.rig import multiclamp, state, windows


def _rig(tmp_path):
    book = tmp_path / "Clampex LabBook.rtf"
    book.write_text(RTF, encoding="cp1252")
    _session(tmp_path / "Mightex", "20261001_151431")
    cell = tmp_path / "R007" / "Slice1" / "cell2"
    cell.mkdir(parents=True)
    _tiff(cell / "pipette.tif", xmp=XMP.format(name="Cal 40x", px=1000, length=200, unit="um"))
    ini = tmp_path / "InfinityAnalyze.ini"
    ini.write_text(f"[Images]\nLastWriteFolder={cell.as_posix()}\n[Camera]\nModelName=INFINITY8-3C\n", encoding="utf-8")
    shown = [windows.AppWindow(11, "Clampex", "clampex.exe", 1, False, 1200, 800),
             windows.AppWindow(12, "Event Log", "polyscan3.exe", 2, True, 631, 421),
             windows.AppWindow(13, "PolyScan4 - EXPERIMENT", "polyscan3.exe", 2, True, 822, 675),
             windows.AppWindow(14, "Visual Studio Code", "code.exe", 3, False, 1900, 1000)]
    amp = multiclamp.Commander(FakeCommander(VC_STATE), window_check=lambda: True)
    return state.Watch(amp, labbook_path=book, polyscan_root=tmp_path / "Mightex", camera_settings=ini,
                       find_windows=lambda: windows.rig_windows(shown))


def test_the_first_look_shows_the_programs_and_the_amplifier_and_the_files_follow(tmp_path):
    from clampwise.rig import polyscan

    polyscan._folders.clear()
    watch = _rig(tmp_path)
    first = watch.look()
    assert first.clampex_open and first.light_open and not first.camera_open
    assert first.amplifier.connected and first.amplifier.values["rs"] == 12.5
    assert first.last_file is None and first.light is None                  # the files come with the next look
    watch._slow_at -= watch.slow_every_s
    second = watch.look()
    assert second.last_file.name == "26o01001.abf" and second.data_folder.endswith("cell2")
    assert second.last_test.values == {"ra_mohm": 9.1}
    assert second.light.name == "20261001_151431" and second.light.sequence == "Grid1.seq"
    assert second.picture.path.name == "pipette.tif" and second.picture.um_per_px == 0.2
    watch.close()


def test_the_rig_programs_are_found_by_role_main_window_first():
    found = windows.rig_windows([windows.AppWindow(12, "Event Log", "polyscan3.exe", 2, True, 631, 421),
                                 windows.AppWindow(13, "PolyScan4", "polyscan3.exe", 2, True, 822, 675),
                                 windows.AppWindow(14, "Editor", "code.exe", 3, False, 1900, 1000),
                                 windows.AppWindow(15, "Commander", "mc700b.exe", 4, False, 600, 500)])
    assert [w.hwnd for w in found["light"]] == [13, 12] and [w.hwnd for w in found["amplifier"]] == [15]
    assert found["acquisition"] == [] and found["camera"] == []
    assert set(found) == set(windows.ROLES)


def test_times_are_said_in_words_a_glance_takes_in():
    now = dt.datetime(2026, 10, 13, 12, 0, 0)
    assert state.since(now - dt.timedelta(seconds=12), now) == "12 s ago"
    assert state.since(now - dt.timedelta(minutes=4), now) == "4 min ago"
    assert state.since(now - dt.timedelta(hours=2, minutes=5), now) == "2 h 5 min ago"
    assert state.since(now - dt.timedelta(days=12), now) == "12 days ago"
    assert state.since(dt.datetime(2025, 6, 9, 15, 27), now) == "on 9 Jun 2025"
    assert state.since(None, now) == ""


def test_the_usual_places_are_looked_up_when_read_so_tests_never_see_this_rig():
    watch = state.Watch(multiclamp.Commander(FakeCommander(VC_STATE), window_check=lambda: True),
                        find_windows=lambda: {})
    watch._slow_at -= watch.slow_every_s
    snap = watch.look()
    assert snap.last_file is None and snap.light is None and snap.picture is None        # the empty rig of conftest
