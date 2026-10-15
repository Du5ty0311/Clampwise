"""The project store: recordings and identity, decisions and notes, settings
and presets, the result cache keyed by file, settings and version, reopening
without re-analysis, and the import of a Synaptoscope project."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clampwise.analysis import registry
from clampwise.core import Identity
from clampwise.project import Project, Preset, legacy
from clampwise.analysis import routing


@pytest.fixture
def experiment(tmp_path):
    """An experiment folder with two small ABF files written here."""
    import pyabf.abfWriter

    rng = np.random.default_rng(0)
    folder = tmp_path / "A1_61826"
    for cell, name in (("cell1", "26618005.abf"), ("cell2", "26618009.abf")):
        d = folder / "Slice1" / cell
        d.mkdir(parents=True)
        sweeps = rng.normal(-65.0, 0.5, (3, 2000)).astype(np.float32)
        pyabf.abfWriter.writeABF1(sweeps, str(d / name), 10_000, units="mV")
    return folder


def test_open_creates_the_store_and_parses_identity(experiment):
    p = Project.open(experiment)
    assert (experiment / "Clampwise project.clampwise").is_file()
    assert Project.is_project(experiment)
    files = sorted(experiment.rglob("*.abf"))
    rows = p.add_recordings(files)
    assert [r.name for r in rows] == ["26618005.abf", "26618009.abf"] and all(len(r.sha256) == 64 for r in rows)
    who = p.identity(files[0])
    assert (who.animal, who.slice, who.cell) == ("A1_61826", "Slice1", "cell1")
    merged = p.set_identity(files[0], Identity(condition="carbachol", cell="cell 1b"))
    assert merged.cell == "cell 1b" and merged.sources["cell"] == "typed" and merged.animal == "A1_61826"
    assert p.identity(files[0]).condition == "carbachol"
    p.add_recordings(files)                                   # adding again does not duplicate
    assert len(p.recordings()) == 2
    p.close()


def test_decisions_notes_overrides_and_settings_persist(experiment):
    p = Project.open(experiment)
    f = sorted(experiment.rglob("*.abf"))[0]
    p.add_recordings([f])
    d = p.add_decision(f, "event_reject", "sweep 0 at 1.2340 s", "noise")
    p.add_decision(f, "exclude", "26618005.abf", "lost the cell")
    assert [x.kind for x in p.decisions(f)] == ["event_reject", "exclude"]
    p.remove_decision(d.id)
    assert [x.kind for x in p.decisions(f)] == ["exclude"]
    p.add_note(f, "drug on", time_s=120.0)
    p.add_note(None, "a project note")
    assert p.notes(f)[0]["text"] == "drug on" and len(p.notes()) == 2
    o = p.set_overrides(f, stimulus_override="320, 425 @ 5", sweeps="1-2")
    assert o.stimulus_override == "320, 425 @ 5" and p.overrides(f).sweeps == "1-2"
    with pytest.raises(ValueError):
        p.set_overrides(f, nonsense="x")
    p.set_setting("baseline", "max_baseline_s", 0.05)
    with pytest.raises(ValueError):
        p.set_setting("baseline", "max_baseline_s", -1)
    p.set_mappings([routing.Mapping("Blue_LED*", "cc_evoked", ("baseline",), "BlueLED", "light")])
    s = p.pipeline_settings(f)
    assert s.by_analysis["baseline"]["max_baseline_s"] == 0.05 and s.stimulus_override == "320, 425 @ 5"
    assert s.mappings[0].pattern == "Blue_LED*" and s.shared["sweeps"] == "1-2"
    p.close()
    again = Project.open(experiment)
    assert again.settings()["baseline"]["max_baseline_s"] == 0.05
    assert again.decisions(f)[0].reason == "lost the cell"


def test_results_are_cached_by_file_settings_and_version_and_reopen_instantly(experiment):
    p = Project.open(experiment)
    f = sorted(experiment.rglob("*.abf"))[0]
    p.add_recordings([f])
    first = p.analyze(f)
    assert not first.results["baseline"].from_cache
    second = p.analyze(f)
    assert second.results["baseline"].from_cache
    assert second.values() == first.values()
    # a changed setting re-runs only the analysis it affects
    p.set_setting("baseline", "max_baseline_s", 0.05)
    third = p.analyze(f)
    assert not third.results["baseline"].from_cache
    # a bumped version invalidates
    a = registry.REGISTRY["baseline"]
    registry.REGISTRY["baseline"] = registry.Analysis(a.key, a.label, a.version + 1, a.types, a.run, a.settings,
                                                      a.outputs, a.description, a.source, a.modality, a.requires)
    try:
        fourth = p.analyze(f)
        assert not fourth.results["baseline"].from_cache
    finally:
        registry.REGISTRY["baseline"] = a
    # reopening reads the last run without analyzing
    p.close()
    again = Project.open(experiment)
    run = again.last_run(f)
    assert run is not None and run.results["baseline"].from_cache and run.provenance.analyses
    assert again.summaries()[0]["recording_type"] == run.recording_type
    assert again.summaries()[0]["quality"] == "not checked"
    assert again.cache_stats()["results"] >= 3
    assert again.clear_cache() >= 3 and again.last_run(f) is None
    again.close()


def test_a_changed_setting_re_runs_the_analyses_that_read_its_result(tmp_path):
    """Events, intrinsic and quality read baseline's result: a cached one keyed
    only on its own settings would show numbers from the old baseline."""
    import pyabf.abfWriter

    d = tmp_path / "A1" / "Slice1" / "cell1"
    d.mkdir(parents=True)
    f = d / "26618005.abf"
    sweeps = np.random.default_rng(0).normal(-65.0, 0.5, (1, 120_000)).astype(np.float32)   # 12 s: spontaneous
    pyabf.abfWriter.writeABF1(sweeps, str(f), 10_000, units="mV")
    p = Project.open(tmp_path / "A1")
    p.add_recordings([f])
    first = p.analyze(f)
    readers = [k for k in first.results if "baseline" in registry.REGISTRY[k].requires]
    assert readers                                            # the file runs analyses that read baseline
    p.set_setting("baseline", "max_baseline_s", 0.05)
    again = p.analyze(f)
    assert not again.results["baseline"].from_cache
    assert [k for k in readers if again.results[k].from_cache] == []
    assert again.results["artifacts"].from_cache              # what does not read baseline stays cached
    p.close()


def test_a_stimulus_found_at_other_times_is_not_served_from_the_cache(tmp_path, monkeypatch):
    """The stimulus finder has no version of its own: a result keys on the times it was given."""
    from clampwise import simulate
    from clampwise.analysis import pipeline

    folder = tmp_path / "A1"
    folder.mkdir()
    f = folder / "evoked.abf"
    f.write_bytes(b"evoked")                                  # the store hashes a file; the samples are the simulator's
    p = Project.open(folder)
    p.add_recordings([f])
    assert not p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording).results["evoked"].from_cache
    assert p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording).results["evoked"].from_cache
    finder = pipeline.stimulus_finder.find_stimulus

    def a_ms_later(recording, **kw):
        s = finder(recording, **kw)
        s.times_s = [t + 0.001 for t in s.times_s]
        return s

    monkeypatch.setattr(pipeline.stimulus_finder, "find_stimulus", a_ms_later)
    assert not p.analyze(f, recording=simulate.evoked(n_sweeps=3).recording).results["evoked"].from_cache
    p.close()


def test_a_rerecorded_file_is_not_served_from_the_cache(experiment):
    import pyabf.abfWriter

    p = Project.open(experiment)
    f = sorted(experiment.rglob("*.abf"))[0]
    p.add_recordings([f])
    before = p.analyze(f).values()
    sweeps = np.random.default_rng(5).normal(-55.0, 0.5, (3, 2000)).astype(np.float32)
    pyabf.abfWriter.writeABF1(sweeps, str(f), 10_000, units="mV")
    p.refresh_hash(f)
    after = p.analyze(f)
    assert not after.results["baseline"].from_cache
    assert abs(after.values()["baseline.level"] - before["baseline.level"]) > 5
    p.close()


def test_presets_apply_and_are_printed_in_provenance(experiment):
    p = Project.open(experiment)
    f = sorted(experiment.rglob("*.abf"))[0]
    p.add_recordings([f])
    preset = Preset("Short baseline", "for testing", {"baseline": {"max_baseline_s": 0.02}},
                    mappings=[routing.Mapping("*", "cc_steps")])
    p.save_preset(preset)
    assert "Short baseline" in p.presets() and "Defaults" in p.presets()
    p.apply_preset("Short baseline")
    r = p.analyze(f)
    assert r.provenance.preset == "Short baseline"
    assert r.results["baseline"].settings["max_baseline_s"] == 0.02
    assert r.recording_type == "cc_steps" and "mapping" in r.type_reason
    round_trip = Preset.from_dict(json.loads(json.dumps(preset.to_dict())))
    assert round_trip.mappings[0].pattern == "*" and round_trip.settings == preset.settings
    p.delete_preset("Short baseline")
    assert "Short baseline" not in p.presets()
    p.close()


def test_a_synaptoscope_project_is_imported_once_and_left_in_place(experiment):
    files = sorted(experiment.rglob("*.abf"))
    old = {"format": "synaptoscope-session", "version": 1, "saved": "2026-09-30T10:00:00", "app_version": "0.4.0",
           "watch_folder": "D:/rig", "settings": {"detect_sd": 4.0, "blank_ms": 0.3},
           "recordings": [{"source": str(files[0]), "sweeps": "1-2",
                           "identity": {"animal": "A1", "slice": "S1", "cell": "c1", "condition": "control",
                                        "sex": "F", "drug_windows": "baseline 0-10; WIN 10-25",
                                        "rs_typed": "12, 15", "stim_override": "320 @ 5",
                                        "recording_type": "cc_evoked", "induction_min": 10.0,
                                        "event_edits": json.dumps({"rejected": [[0, 1.5]], "added": [[1, 2.25]]})}},
                          {"source": str(experiment / "missing.abf"), "identity": {"animal": "A2"}}],
           "statistics": {"exclusions": [{"key": files[0].name, "reason": "Rs too high", "level": "recording"}]}}
    (experiment / "Synaptoscope project.synaptoscope").write_text(json.dumps(old), encoding="utf-8")
    p = Project.open(experiment)
    assert (experiment / "Synaptoscope project.synaptoscope").is_file()
    notes = json.loads(p.meta("import_notes"))
    assert any("imported" in n for n in notes) and any("missing.abf" in n for n in notes)
    who = p.identity(files[0])
    assert (who.animal, who.slice, who.cell, who.condition) == ("A1", "S1", "c1", "control")
    meta = p.metadata(files[0])
    assert meta.sex == "F" and meta.rs_typed_mohm == (12.0, 15.0)
    assert [d.name for d in meta.drugs] == ["baseline", "WIN", "induction"] and meta.drugs[1].on_s == 600.0
    o = p.overrides(files[0])
    assert o.stimulus_override == "320 @ 5" and o.type_override == "cc_evoked" and o.sweeps == "1-2"
    kinds = sorted(d.kind for d in p.decisions(files[0]))
    assert kinds == ["event_add", "event_reject", "exclude"]
    assert json.loads(p.meta("legacy_settings"))["detect_sd"] == 4.0
    assert p.meta("watch_folder") == "D:/rig"
    p.close()
    # opening again does not import twice
    again = Project.open(experiment)
    assert len(again.decisions(files[0])) == 3
    again.close()


def test_old_presets_and_protocol_maps_import(tmp_path):
    preset_file = tmp_path / "CA3.json"
    preset_file.write_text(json.dumps({"format": "synaptoscope-analysis-preset", "version": 1, "name": "CA3 DSI",
                                       "description": "d", "settings": {"event_rise_ms": 0.5},
                                       "mappings": [{"pattern": "*DSI*", "recording_type": "dsi", "analyses": []}]}),
                           encoding="utf-8")
    preset = legacy.import_preset(preset_file)
    assert preset.name == "CA3 DSI" and preset.legacy == {"event_rise_ms": 0.5}
    assert preset.mappings[0].recording_type == "dsi" and preset.settings == {}
    pm = tmp_path / "protocol_map.json"
    pm.write_text(json.dumps({"format": "synaptoscope-protocol-map", "version": 1,
                              "mappings": [{"pattern": "Blue_LED*", "recording_type": "cc_evoked",
                                            "stimulus_channel": "BlueLED", "stimulus_kind": "light"}]}), encoding="utf-8")
    maps = legacy.import_protocol_map(pm)
    assert maps[0].pattern == "Blue_LED*" and maps[0].stimulus_kind == "light"
    assert legacy.parse_rs_typed("") == pytest.approx((np.nan, np.nan), nan_ok=True)
    assert set(legacy.old_app_folders()) == {"presets", "protocol_map", "figure_styles", "figure_templates",
                                             "stimulus_library"}
