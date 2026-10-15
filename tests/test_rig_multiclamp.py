"""The amplifier readout through MultiClamp Commander's message library: read only by construction (no setter is
ever looked up or called), each mode's readings in the lab's units, and a closed or busy Commander said, not waited
on. A fake library stands in for Commander (tests never load the real one)."""

from __future__ import annotations

import ctypes
import re
from pathlib import Path

import pytest

from clampwise.rig import multiclamp as mc


class FakeCommander:
    """Commander's library as a fake: answers the getters from ``state`` and records every name looked up."""

    def __init__(self, state: dict, *, errors: dict | None = None, meters=(1.5e9, -4.0e-11, 0.0, 0.0)) -> None:
        self.state, self.errors, self.meters = state, errors or {}, meters
        self.looked_up: list[str] = []
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        if not name.startswith("MCCMSG_"):
            raise AttributeError(name)
        self.looked_up.append(name)

        def fn(*args):
            self.calls.append(name)
            return self._answer(name, args)

        return fn

    def _answer(self, name, args):
        if name == "MCCMSG_CheckAPIVersion":
            return True
        if name == "MCCMSG_CreateObject":
            return 1
        if name == "MCCMSG_DestroyObject":
            return None
        if name == "MCCMSG_FindFirstMultiClamp":
            _, model, serial, _, port, dev, chan, _ = args
            model._obj.value, port._obj.value, dev._obj.value, chan._obj.value = 1, 0, 0, 1
            serial.value = b"00835188"
            return True
        if name in ("MCCMSG_FindNextMultiClamp",):
            return False
        if name == "MCCMSG_SelectMultiClamp":
            return True
        if name == "MCCMSG_GetMeterValue":
            out, meter = args[1], args[2]
            out._obj.value = self.meters[meter.value if hasattr(meter, "value") else meter]
            return True
        if name.startswith("MCCMSG_Get"):
            if name in self.errors:
                args[-1]._obj.value = self.errors[name]
                return False
            args[1]._obj.value = self.state.get(name, 0)
            return True
        raise AssertionError(f"the readout called {name}")


VC_STATE = {"MCCMSG_GetMode": 0, "MCCMSG_GetHolding": -0.070, "MCCMSG_GetHoldingEnable": 1,
            "MCCMSG_GetPrimarySignal": 0, "MCCMSG_GetPrimarySignalGain": 2.0, "MCCMSG_GetPrimarySignalLPF": 10_000.0,
            "MCCMSG_GetWholeCellCompEnable": 1, "MCCMSG_GetWholeCellCompCap": 25e-12,
            "MCCMSG_GetWholeCellCompResist": 12.5e6, "MCCMSG_GetRsCompEnable": 1, "MCCMSG_GetRsCompCorrection": 70.0,
            "MCCMSG_GetPipetteOffset": 0.0123, "MCCMSG_GetFastCompCap": 6.5e-12, "MCCMSG_GetMeterResistEnable": 0}
IC_STATE = {"MCCMSG_GetMode": 1, "MCCMSG_GetHolding": -50e-12, "MCCMSG_GetBridgeBalEnable": 1,
            "MCCMSG_GetBridgeBalResist": 15.2e6, "MCCMSG_GetNeutralizationCap": 3.1e-12, "MCCMSG_GetPrimarySignal": 7}


def _commander(fake) -> mc.Commander:
    return mc.Commander(fake, window_check=lambda: True)


def test_the_readout_can_reach_getters_only():
    assert all(name.startswith("MCCMSG_Get") for name in mc.GETTERS)
    assert not any(word in name for name in mc.LIFECYCLE + mc.GETTERS for word in mc.FORBIDDEN)
    lib = mc._Library(FakeCommander(VC_STATE))
    assert set(lib.names) == set(mc.LIFECYCLE) | set(mc.GETTERS)
    with pytest.raises(KeyError):
        lib("MCCMSG_SetHolding", 1, ctypes.c_double(0.0), None)       # never bound, so never callable
    source = Path(mc.__file__).read_text(encoding="utf-8")
    assert not re.search(r"MCCMSG_(Set|Auto|Zap|Buzz|Pulse|Reset|QuickSelect|Toggle|Clear)", source)


def test_voltage_clamp_is_read_in_the_lab_s_units_and_nothing_else_is_touched():
    fake = FakeCommander(VC_STATE)
    with _commander(fake) as amp:
        s = amp.read()
    assert s.connected and s.mode_name == "voltage clamp" and s.device.serial == "00835188"
    assert s.values["holding"] == pytest.approx(-70.0) and s.units["holding"] == "mV"
    assert s.values["cm"] == pytest.approx(25.0) and s.values["rs"] == pytest.approx(12.5)
    assert s.values["pipette_offset"] == pytest.approx(12.3) and s.values["primary_signal"] == "membrane current"
    assert s.shown("rs") == "12.5 MOhm" and s.shown("holding_on") == "on" and s.shown("bridge") == "–"
    assert s.meters[0] == pytest.approx(1.5e9)
    assert "MCCMSG_GetBridgeBalResist" not in fake.calls                     # a current-clamp getter, not asked
    assert all(n in mc.LIFECYCLE + mc.GETTERS for n in fake.looked_up)
    assert fake.calls[-1] == "MCCMSG_DestroyObject"


def test_current_clamp_reads_its_own_readings():
    fake = FakeCommander(IC_STATE)
    s = _commander(fake).read()
    assert s.mode_name == "current clamp" and s.values["holding"] == pytest.approx(-50.0) and s.units["holding"] == "pA"
    assert s.values["bridge"] == pytest.approx(15.2) and s.values["neutralization"] == pytest.approx(3.1)
    assert s.values["primary_signal"] == "membrane potential"
    assert "MCCMSG_GetWholeCellCompResist" not in fake.calls


def test_a_closed_commander_is_said_and_nothing_is_loaded():
    fake = FakeCommander(VC_STATE)
    s = mc.Commander(fake, window_check=lambda: False).read()
    assert not s.connected and s.status == "MultiClamp Commander is not open"
    assert fake.looked_up == []


def test_commander_is_open_when_its_own_startup_window_is(monkeypatch):
    # on the rig PC (Commander 2.2.2.2, 2026-10-14) Commander's top windows are its panel and a hidden startup
    # window; the library's message window class is made by the library in the reading program, never by Commander
    asked = []
    monkeypatch.setattr(mc.sys, "platform", "win32")
    monkeypatch.setattr(ctypes.windll.user32, "FindWindowW", lambda cls, title: asked.append((cls, title)) or 7)
    assert mc.commander_open() and asked == [(None, "Invisible MultiClamp Startup Window")]


def test_the_four_meters_are_each_channel_s_potential_and_current():
    # as Commander showed them on the rig PC (2026-10-14): Channel 1 V -6.5 mV, I -0.01 nA; Channel 2 V 0.1 mV,
    # I -2.9 pA; the meters read -6.56e-3, -9.16e-12, 7.6e-5, -2.44e-12 at the same moment
    s = mc.AmplifierState(device=mc.Device(1, "00123456", 0, 0, 1), connected=True, mode=0,
                          values={"meter_resist_on": False, "meter_irms_on": False},
                          meters={0: -6.561e-3, 1: -9.155e-12, 2: 7.629e-5, 3: -2.441e-12})
    assert mc.meter_words(s) == "Channel 1: -6.6 mV, -9.2 pA; Channel 2: 0.1 mV, -2.4 pA"
    s.values["meter_resist_on"] = True                       # the selected channel's current meter shows resistance
    assert mc.meter_words(s).startswith("Channel 1: -6.6 mV, -9.15e-12 (resistance or Irms metering on: as read);")
    assert mc.meter_words(mc.AmplifierState()) == ""


def test_a_busy_commander_stops_the_reading_instead_of_waiting_on_every_call():
    fake = FakeCommander(VC_STATE, errors={"MCCMSG_GetHolding": mc.TIMED_OUT})
    s = _commander(fake).read()
    assert s.status == "Commander did not answer in time" and s.errors["holding"] == s.status
    assert "MCCMSG_GetWholeCellCompCap" not in fake.calls and not s.meters


def test_commander_s_library_is_found_where_it_installs():
    path = mc.find_library()
    if path is None:
        pytest.skip("MultiClamp Commander is not installed on this PC")
    assert path.name == "AxMultiClampMsg.dll" and path.is_file()
