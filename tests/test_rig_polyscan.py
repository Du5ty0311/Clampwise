"""PolyScan's sessions read from their folders: the event log (start, stop, triggers with their counts), the
experiment and version, the wiring (an external trigger into the Polygon), the pattern sequence and its regions,
the objective; no personal detail kept; a recording matched to the session running then, a crashed session
ending when the next began."""

from __future__ import annotations

import datetime as dt

from clampwise.rig import polyscan


def _blob(*words: str) -> bytes:
    """Text runs between binary bytes, as PolyScan's .dat, .config and .settings files hold them."""
    return b"\x00\x07".join(w.encode("latin-1") for w in words) + b"\x00"


def _session(root, name, *, stop=True, triggers=0, sequence="Grid1.seq", regions=3):
    folder = root / "PolyScan3" / "User Data" / "Account" / "BLA Connectivity" / "20260724_Account" / name
    folder.mkdir(parents=True)
    t0 = dt.datetime.strptime(name, "%Y%m%d_%H%M%S")
    lines = [f"{(t0 + dt.timedelta(seconds=2)):%Y-%m-%d %H:%M:%S}.342 Msg Session ({name}) started. \""]
    for k in range(triggers):
        t = t0 + dt.timedelta(seconds=10 + 11 * k)
        lines.append(f"{t:%Y-%m-%d %H:%M:%S}.934 HardwareTrigger External Trigger #1 triggers \"TimeStamp={k * 11001160},"
                     f" Source=External Trigger #1, Trigger Count={k + 1}")
        lines.append(f"{t:%Y-%m-%d %H:%M:%S}.936 DigitalOutput Digital output to device Polygon #1, port Trig In "
                     f"\"TimeStamp={k * 11001160}, Device=Polygon #1, Port=Trig In, Step=0")
    if stop:
        lines.append(f"{(t0 + dt.timedelta(hours=1)):%Y-%m-%d %H:%M:%S}.421 Msg Session ({name}) stopped. \"")
    (folder / "EventLog.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "SessionSummary.dat").write_bytes(_blob("BLA Connectivity", "20260724_Account", name, "Account",
                                                      "A University", "someone@example.edu", "RIG-PC-01", "4.1.3.0"))
    (folder / "Hardware.config").write_bytes(_blob("HCV2", "Mightex", "DSI-K3-000", "Polygon #1", "Mightex",
                                                   "BLS-0280-2", "LED Driver #1", "LED #1", "HardwareTrigger",
                                                   "External Trigger #1", "External Trigger #1", "Polygon #1", "Trig In",
                                                   "LED Driver #1", "LED 1", "LED #1", "Current Input"))
    (folder / "External Trigger #1.settings").write_bytes(_blob("Mightex Device Settings", "External Trigger #1",
                                                                "TriggerType", "RisingEdge"))
    (folder / "Polygon #1.settings").write_bytes(_blob("Mightex Device Settings", "Polygon #1",
                                                       "OutputTrigger_PulseWidth", "100", "MappingFileName",
                                                       r"C:\Users\x\Most Recent Settings\Calibration#4.map", "Zeiss",
                                                       "40X", "WorkModeSetting", rf"C:\Users\x\{sequence}", sequence,
                                                       "someone@example.edu"))
    (folder / sequence).write_bytes(_blob("4.1.3.0", "someone@example.edu",
                                          *[w for k in range(regions) for w in ("PolyScan.Utilities.CRectangle",
                                                                                f"ROI{k}")]))
    (folder / "Workflow.config").write_bytes(_blob("Workflow 1.0", "127.0.0.1"))
    return folder


def test_a_session_is_read_from_its_folder_and_keeps_nothing_personal(tmp_path):
    s = polyscan.read_session(_session(tmp_path, "20261001_151431", triggers=3))
    assert s.started == dt.datetime(2026, 10, 1, 15, 14, 33, 342000) and s.stopped.hour == 16
    assert (s.experiment, s.template, s.version) == ("BLA Connectivity", "20260724_Account", "4.1.3.0")
    assert s.externally_triggered and s.trigger_edge == "RisingEdge" and "LED #1" in s.devices
    assert (s.sequence, s.regions, s.objective, s.calibration) == ("Grid1.seq", 3, "Zeiss 40X", "Calibration#4.map")
    assert s.output_pulse_ms == 100.0 and s.workflow == []
    assert [e.count for e in s.triggers] == [1, 2, 3] and s.triggers[1].detail["Source"] == "External Trigger #1"
    kept = repr(vars(s))
    assert "@" not in kept and "RIG-PC" not in kept                     # the e-mail and the computer are not kept
    assert s.describe() == ("PolyScan session 20261001_151431 (BLA Connectivity); pattern sequence Grid1.seq, "
                            "3 region(s); Zeiss 40X; the Polygon on its external trigger (rising edge); LED #1")


def test_a_recording_is_matched_to_the_session_running_then(tmp_path):
    _session(tmp_path, "20261001_121425")
    _session(tmp_path, "20261001_151431", stop=False)                  # never stopped (PolyScan closed or crashed)
    _session(tmp_path, "20261002_093000")
    polyscan._folders.clear()
    assert polyscan.running_at(dt.datetime(2026, 10, 1, 12, 30), tmp_path).name == "20261001_121425"
    assert polyscan.running_at(dt.datetime(2026, 10, 1, 14, 0), tmp_path) is None             # between sessions
    crashed = polyscan.running_at(dt.datetime(2026, 10, 1, 23, 0), tmp_path)
    assert crashed.name == "20261001_151431" and crashed.ends == dt.datetime(2026, 10, 2, 9, 30)
    assert polyscan.running_at(dt.datetime(2026, 9, 30, 8, 0), tmp_path) is None
    assert polyscan.latest(tmp_path).name == "20261002_093000"


def test_event_lines_are_parsed_with_their_details():
    events = polyscan.parse_events(
        '2026-09-18 12:12:30.865 Error MightexPolygon1000.SetOutputTriggerSettings: SetOutputTriggerSetting(1,5000,1)'
        ' = -10 "\n'
        '2025-05-01 14:20:11.936 AnalogOutput Analog output to device LED #1, port Current Input "TimeStamp=0, '
        'Device=LED #1, Port=Current Input, Step=0\nnot an event line\n')
    assert [e.kind for e in events] == ["Error", "AnalogOutput"]
    assert events[1].detail == {"TimeStamp": "0", "Device": "LED #1", "Port": "Current Input", "Step": "0"}


def test_a_session_left_without_a_stop_ends_at_its_last_activity_unless_polyscan_runs(tmp_path, monkeypatch):
    # the review's case: a crash with no session after it must not make every later recording's line light
    import os

    folder = _session(tmp_path, "20261001_151431", stop=False)
    last = dt.datetime(2026, 10, 1, 16, 0, 0).timestamp()
    for p in folder.iterdir():
        os.utime(p, (last, last))
    polyscan._folders.clear()
    polyscan._cache.clear()
    monkeypatch.setattr(polyscan, "program_running", lambda: False)
    assert polyscan.running_at(dt.datetime(2026, 10, 1, 15, 30), tmp_path) is not None
    assert polyscan.running_at(dt.datetime(2026, 10, 13, 10, 0), tmp_path) is None            # long after it
    monkeypatch.setattr(polyscan, "program_running", lambda: True)                        # PolyScan open now
    polyscan._cache.clear()
    assert polyscan.running_at(dt.datetime(2026, 10, 13, 10, 0), tmp_path) is not None
