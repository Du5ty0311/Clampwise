"""A rig's own inputs that carry light (on a Polygon rig, AM_Stim on analog input 8 records
analog output 2, which drives the blue LED's light source feeding the patterned illuminator). The rule is the rig's,
kept in its profile: a file that matches the rig takes it over the words a line's name has in every lab (AM_Stim is
electrical elsewhere), the reason says it is the rig's profile, the patterned light waiting on its trigger is said
beside it, and a changed rule analyzes the file again."""

from __future__ import annotations

import json

from test_rig_evidence import _light

from clampwise import simulate
from clampwise.analysis import pipeline, stimulus
from clampwise.project import Project, rigs

WIRING = "analog output 2 drives the blue LED's light source, whose light reaches the slice through the patterned " \
         "illuminator (the rig's profile, 2026-10-14)"


def _am_stim():
    return simulate.evoked(n_sweeps=2, light=False).recording                 # its trigger line: AM_Stim


def test_the_rig_keeps_the_inputs_that_carry_light(tmp_path):
    path = tmp_path / "rigs.json"
    path.write_text(json.dumps({"format": "clampwise-rigs", "rigs": [         # a profile saved before the rule
        {"name": "Polygon rig", "format": "ABF", "digitizer": 6, "amplifier": 24, "channels": ["Left 1"],
         "units": ["mV"], "protocols": [], "preset": "", "files": 3}]}), encoding="utf-8")
    store = rigs.Rigs(path)
    assert store.rigs["Polygon rig"].light_inputs == {}
    store.set_light_inputs("Polygon rig", {"AM_Stim": WIRING})
    again = rigs.Rigs(path)
    assert again.rigs["Polygon rig"].light_inputs == {"AM_Stim": WIRING}
    again.set_light_inputs("Polygon rig", {})                                  # removable
    assert rigs.Rigs(path).rigs["Polygon rig"].light_inputs == {}


def test_a_rig_s_light_input_is_light_where_its_name_says_electrical():
    rec = _am_stim()
    assert stimulus.find_stimulus(rec).kind == "electrical"                   # every other lab: unchanged
    s = stimulus.find_stimulus(rec, light_inputs={"AM_Stim": WIRING})
    assert s.kind == "light" and s.source == "light channel AM_Stim"
    assert f"light: on this rig AM_Stim carries light ({WIRING})" in s.reason
    assert stimulus.find_stimulus(rec, light_inputs={"am_stim": WIRING}).kind == "light"   # names match in any case
    assert stimulus.find_stimulus(rec, light_inputs={"IN 8": WIRING}).kind == "electrical"


def test_the_patterned_light_waiting_on_its_trigger_is_said_beside_the_rig_s_rule():
    s = stimulus.find_stimulus(_am_stim(), light_inputs={"AM_Stim": WIRING}, rig=_light())
    assert s.kind == "light"
    assert ("the patterned light's session 20261001_151431 was running, waiting on its external trigger "
            "(pattern sequence Grid1.seq, 101 regions): it agrees") in s.reason
    assert "if this line triggered it" not in s.reason


def test_a_file_of_the_rig_takes_its_light_inputs_and_a_changed_rule_runs_again(tmp_path, monkeypatch):
    f = tmp_path / "cell1" / "26o01005.abf"
    f.parent.mkdir()
    f.write_bytes(b"recorded")                  # the store hashes a file; the samples are the simulator's
    store = rigs.Rigs()
    store.rigs["Polygon rig"] = rigs.Rig("Polygon rig", "ABF", 6, 24, ["Left 1"], ["mV"],
                                         light_inputs={"AM_Stim": WIRING})
    store.save()
    monkeypatch.setattr(rigs, "match_file", lambda path, profiles=None: (profiles or rigs.Rigs()).match(
        rigs.Fingerprint("ABF", 6, 24, "Left 1", "mV")))
    p = Project.open(tmp_path)
    p.add_recordings([f])
    s = p.pipeline_settings(f)
    assert s.shared["light_inputs"] == {"AM_Stim": WIRING}
    first = p.analyze(f, recording=_am_stim())
    assert pipeline.find(_am_stim(), s).stimulus.kind == "light"
    assert "evoked" in first.results
    store.set_light_inputs("Polygon rig", {})
    assert not p.analyze(f, recording=_am_stim()).results["evoked"].from_cache      # the rule changed: run again
    p.set_feature("rig_presets", False)                                         # the feature off: no rig's rule
    assert "light_inputs" not in p.pipeline_settings(f).shared
