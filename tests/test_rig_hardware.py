"""The rig's hardware found by itself, from Windows' USB devices (the IDs as the rig PC lists them), each said with
who uses it; nothing is opened."""

from __future__ import annotations

from clampwise.rig import hardware, telegraph

RIG = [(r"USB\VID_09C0&PID_0145\5&34E9DD7A&0&2", "DD1440"),
       (r"USB\VID_09C0&PID_0136\00123456", "USB Input Device"),
       (r"USB\VID_1724&PID_0753\0025050256", "INFINITY USB 3.0 Camera"),
       (r"USB\VID_04B4&PID_0304\5&34E9DD7A&0&4", "Mightex PolyEcho IOBox"),
       (r"USB\VID_413C&PID_2105\5&34E9DD7A&0&7", "USB Input Device")]


def _tg(mode):
    return telegraph.Telegraph(mode, "pA", 5e-5, 1.0, 1e4, "mV", 0.01, 1.0, 20.0, "00123456", 1, 0.0, 0.0)


def test_the_rig_s_devices_are_known_by_their_ids():
    d = hardware.identify(RIG[0][0])
    assert (d.role, d.name, d.serial) == ("acquisition", "Digidata 1440A", "")      # its ID holds no serial
    d = hardware.identify(RIG[1][0])
    assert (d.role, d.name, d.serial) == ("amplifier", "Axon 700B", "00123456")
    d = hardware.identify(RIG[2][0])
    assert (d.role, d.serial) == ("camera", "25050256")                             # as the camera reports it
    assert hardware.identify(RIG[4][0]) is None and hardware.identify("PCI\\VEN_8086") is None


def test_each_device_is_said_with_who_uses_it():
    seen = []

    def read(serial, ch, timeout=0.5):
        seen.append((serial, ch))
        return _tg("IC" if ch == 1 else "VC")

    devs = hardware.detect(devices=lambda: RIG, running=lambda: {"clampex.exe", "infinityanalyze.exe"},
                           read_telegraph=read, commander=lambda: True)
    by = {d.role: d for d in devs}
    assert [d.role for d in devs] == ["acquisition", "amplifier", "camera", "light"]
    assert by["acquisition"].state == "in use by the acquisition program"
    assert by["camera"].state == "in use by the camera's program"
    assert by["amplifier"].channels == {1: "IC", 2: "VC"} and seen == [("00123456", 1), ("00123456", 2)]
    assert by["amplifier"].state == "its program is open: channel 1 IC, channel 2 VC"
    # Clampwise holding them; Commander closed (its telegraph is not asked)
    seen.clear()
    devs = hardware.detect(devices=lambda: RIG[:3], running=set, read_telegraph=read, commander=lambda: False,
                           held={"acquisition": True, "camera": True})
    by = {d.role: d for d in devs}
    assert by["acquisition"].state == by["camera"].state == "in use by Clampwise"
    assert by["amplifier"].state == "its program is closed" and seen == []
    text = hardware.summary(devs)
    assert "Digitizer: Digidata 1440A - in use by Clampwise" in text and "Patterned light" not in text


def test_nothing_connected_is_said():
    devs = hardware.detect(devices=lambda: [], running=set, commander=lambda: False)
    assert devs == [] and hardware.summary(devs).count("not found") == 3
