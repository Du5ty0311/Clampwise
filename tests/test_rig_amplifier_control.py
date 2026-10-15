"""The amplifier's controls in Clampwise (what MultiClamp Commander does, Commander still usable beside it): each
a named call of Commander's own library, in the lab's
units, in the thread that holds Commander; the readout stays read only."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_rig_multiclamp import VC_STATE, FakeCommander

from clampwise.rig import amplifier_control as ac
from clampwise.rig import multiclamp as mc

HEADER = Path(r"C:\Program Files\Molecular Devices\MultiClamp 700B Commander\3rd Party Support\AxMultiClampMsg"
              r"\AxMultiClampMsg.h")


class FakeSetter(FakeCommander):
    """Also answers the setters and buttons, keeping (name, value) of each."""

    def __init__(self, state, ok=True):
        super().__init__(state)
        self.sets: list[tuple] = []
        self.ok = ok

    def _answer(self, name, args):
        if name.startswith(("MCCMSG_Set", "MCCMSG_Auto", "MCCMSG_Zap", "MCCMSG_Buzz")):
            self.sets.append((name,) + tuple(a for a in args[1:-1]))
            if not self.ok:
                args[-1]._obj.value = 6006
            return self.ok
        return super()._answer(name, args)


def _controls(fake):
    amp = mc.Commander(fake, window_check=lambda: True)
    amp.read()
    return ac.Controls(amp)


def test_only_listed_calls_and_never_reset():
    assert not any(re.search(r"Reset|QuickSelect|Toggle|Clear|SetTimeOut", n) for n in ac.SETTERS)
    assert len(set(ac.SETTERS)) == len(ac.SETTERS)
    source = Path(mc.__file__).read_text(encoding="utf-8")               # the readout itself is still read only
    assert not re.search(r"MCCMSG_(Set|Auto|Zap|Buzz)", source)


@pytest.mark.skipif(not HEADER.is_file(), reason="Commander's header is on the rig PC")
def test_every_call_is_one_of_commander_s_own():
    header = HEADER.read_text(encoding="latin-1")
    assert all(re.search(rf"\b{n}\s*\(", header) for n in ac.SETTERS)


def test_values_go_in_the_library_s_units():
    fake = FakeSetter(VC_STATE)
    c = _controls(fake)
    assert c.run(("set", "holding", -70.0, 0)) == ""
    assert c.run(("set", "holding", 100.0, 1)) == ""
    assert c.run(("set", "rs", 10.0, 0)) == "" and c.run(("set", "cm", 33.0, 0)) == ""
    assert c.run(("set", "mode", 1, 0)) == "" and c.run(("set", "primary_gain", 5, 0)) == ""
    assert c.run(("act", "auto_fast")) == ""
    got = {name: v for name, *v in fake.sets}
    assert fake.sets[0] == ("MCCMSG_SetHolding", pytest.approx(-0.070))
    assert fake.sets[1] == ("MCCMSG_SetHolding", pytest.approx(100e-12))
    assert got["MCCMSG_SetWholeCellCompResist"] == [pytest.approx(10e6)]
    assert got["MCCMSG_SetWholeCellCompCap"] == [pytest.approx(33e-12)]
    assert got["MCCMSG_SetMode"] == [1] and got["MCCMSG_SetPrimarySignalGain"] == [5.0]
    assert "MCCMSG_AutoFastComp" in got


def test_a_setting_of_another_mode_or_a_refusal_is_said():
    c = _controls(FakeSetter(VC_STATE))
    assert "not a setting of this mode" in c.run(("set", "bridge", 10.0, 0))           # bridge: current clamp
    c = _controls(FakeSetter(VC_STATE, ok=False))
    assert c.run(("set", "holding", -60.0, 0)) == "Holding: Commander could not do it"


def test_the_rig_s_look_carries_multiclamp_s_telegraph():
    # MultiClamp Commander's telegraph, as Clampwise receives it
    from clampwise.app.screens.live import telegraph_words
    from clampwise.rig import state, telegraph

    tg = telegraph.Telegraph("VC", "pA", 5e-5, 1.0, 10_000.0, "mV", 0.01, 1.0, 20.0, "00835188", 1, 0.0, 0.0)
    asked = []
    watch = state.Watch(mc.Commander(FakeSetter(VC_STATE), window_check=lambda: True), find_windows=lambda: {},
                        read_telegraph=lambda serial, channel, timeout: asked.append((serial, channel)) or tg)
    snap = watch.look()
    assert snap.telegraph is tg and asked == [("00835188", 1)]
    assert telegraph_words(tg) == "VC, 0.05 V/nA x1, 10 kHz, command 20 mV/V"
    ic = telegraph.Telegraph("IC", "mV", 0.02, 2.0, 1e5, "pA", 5e-4, 1.0, 400.0, "00835188", 1, 0.0, 0.0)
    assert telegraph_words(ic) == "IC, 10 mV/mV x2, bypass, command 400 pA/V"


def test_the_rig_s_look_runs_the_controls_first_in_its_own_thread():
    from clampwise.rig import state

    fake = FakeSetter(VC_STATE)
    watch = state.Watch(mc.Commander(fake, window_check=lambda: True), find_windows=lambda: {})
    said = []
    watch.commands.append((("set", "holding_on", False, 0), said.append))
    snap = watch.look()
    assert said == [""] and fake.sets[0] == ("MCCMSG_SetHoldingEnable", False)
    assert fake.calls.index("MCCMSG_SetHoldingEnable") < fake.calls.index("MCCMSG_GetMode")    # set, then read
    assert snap.amplifier.connected and snap.amplifier.raw["mode"] == 0
