"""The registry, stimulus detection, routing and the pipeline, on simulated
recordings with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import pipeline, registry, routing, stimulus
from clampwise.analysis.registry import Output, Setting
from clampwise.analysis.result import Result
from clampwise.core import ArraySource, Channel, Role


# ---- registry

def test_settings_resolve_defaults_check_values_and_count_changes():
    a = registry.load()["baseline"]
    resolved, changed = a.resolve({})
    assert resolved["max_baseline_s"] == 0.1 and changed == []
    resolved, changed = a.resolve({"max_baseline_s": 0.05})
    assert resolved["max_baseline_s"] == 0.05 and changed == ["max_baseline_s"]
    with pytest.raises(ValueError, match="unknown setting"):
        a.resolve({"nonsense": 1})
    with pytest.raises(ValueError, match="below the minimum"):
        a.resolve({"max_baseline_s": 0.0})
    assert registry.definitions()["baseline.level" if False else "level"].units == "signal units"


def test_registering_an_analysis_is_a_declaration(monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "toy", None)

    @registry.register("toy", "A toy", version=3, types=["cc_steps"], requires=("baseline",),
                       settings=[Setting("k", 2.0, "Gain", "Multiplies the baseline level.")],
                       outputs=[Output("scaled", "k times the baseline level.", "signal units")])
    def toy(recording, context, settings):
        base = context.result("baseline")
        return Result("toy", 3, {"scaled": settings["k"] * base.values["level"]})

    order = [a.key for a in registry.for_type("cc_steps")]
    assert order.index("baseline") < order.index("toy")
    with pytest.raises(ValueError, match="unknown recording type"):
        registry.register("bad", "Bad", version=1, types=["nothing"])(lambda *a: None)
    result = pipeline.analyze(simulate.steps(n_sweeps=3).recording, pipeline.Settings(by_analysis={"toy": {"k": 3.0}}))
    assert result.results["toy"].values["scaled"] == pytest.approx(3.0 * result.results["baseline"].values["level"])
    assert result.results["toy"].changed_settings == ["k"]
    assert result.provenance.analyses["toy"] == 3
    registry.REGISTRY.pop("toy", None)


# ---- stimulus

def test_stimulus_found_on_the_trigger_channel_with_times_and_kind():
    sim = simulate.evoked(onsets_s=(0.2, 0.25))
    s = stimulus.find_stimulus(sim.recording)
    assert s.kind == "light" and s.source == "light channel BlueLED"
    assert np.allclose(s.fixed_times(), [0.2, 0.25], atol=1.0 / sim.recording.sample_rate_hz)
    assert len(s) == 2 * sim.recording.n_sweeps
    assert "taken (pulses found)" in s.reason
    assert "interval 50 ms" in stimulus.describe(s)


def test_monitors_gates_and_frame_sync_lines_are_not_stimuli():
    sim = simulate.evoked(n_sweeps=2)
    rec = sim.recording
    fs = rec.sample_rate_hz
    n = rec.n_samples
    gate = np.zeros((2, n)); gate[:, int(0.1 * fs):int(0.7 * fs)] = 5.0
    sync = np.zeros((2, n)); sync[:, ::int(fs / 1900) or 1] = 5.0
    monitor = np.zeros((2, n)); monitor[:, int(0.3 * fs):int(0.4 * fs)] = 500.0
    arrays = [rec.read(0), rec.read(1), rec.read(2), gate, sync, monitor]
    rec.samples = ArraySource(arrays)
    rec.channels += [Channel(3, "IN 3", "V", Role.TRIGGER, True), Channel(4, "IN 7", "V", Role.TRIGGER, True),
                     Channel(5, "I_monitor", "pA", Role.MONITOR, True)]
    s = stimulus.find_stimulus(rec)
    assert s.channel == "BlueLED"
    assert "IN 3: not taken (held high" in s.reason
    assert "IN 7: not taken (toggles" in s.reason
    assert "I_monitor: not taken (in pA" in s.reason


def test_typed_times_override_the_file_and_say_so():
    sim = simulate.evoked(n_sweeps=2)
    s = stimulus.find_stimulus(sim.recording, override="320, 425 @ 5")
    assert np.allclose(s.fixed_times(), [0.32, 0.425]) and np.allclose(s.widths_s[0], [0.005, 0.005])
    assert s.source == "typed"
    assert stimulus.parse_override("none") is None


def test_no_stimulus_is_said_not_invented():
    s = stimulus.find_stimulus(simulate.steps(n_sweeps=2).recording)
    assert len(s) == 0 and s.reason.startswith("no stimulus found")


# ---- routing

@pytest.mark.parametrize("sim, expected", [
    (lambda: simulate.evoked(n_sweeps=2), "vc_evoked"),
    (lambda: simulate.evoked(n_sweeps=2, voltage_clamp=False), "cc_evoked"),
    (lambda: simulate.steps(n_sweeps=5), "cc_steps"),
    (lambda: simulate.gap_free_events(duration_s=5.0), "vc_spontaneous"),
    (lambda: simulate.membrane_test(n_sweeps=2), "membrane_test"),
])
def test_recording_types_are_detected_with_a_reason(sim, expected):
    rec = sim().recording
    d = routing.detect(rec, stimulus.find_stimulus(rec))
    assert d.recording_type == expected, d.describe()
    assert d.reason and d.source == "detected" and "clamp" in d.evidence


def test_mapping_and_typed_type_take_precedence_but_keep_the_evidence():
    rec = simulate.steps(n_sweeps=3).recording
    s = stimulus.find_stimulus(rec)
    m = routing.Mapping("sim_st*", "cc_evoked", analyses=("baseline",))
    d = routing.decide(rec, s, mappings=[m])
    assert d.recording_type == "cc_evoked" and d.source == "mapping sim_st*" and "detected: cc_steps" in d.reason
    d2 = routing.decide(rec, s, mappings=[m], typed="vc_steps")
    assert d2.recording_type == "vc_steps" and d2.source == "typed"


# ---- pipeline

def test_pipeline_runs_baseline_with_provenance_methods_and_choices():
    sim = simulate.steps(n_sweeps=5, v_rest=-65.0, noise=0.3)
    r = pipeline.analyze(sim.recording)
    assert r.recording_type == "cc_steps"
    base = r.results["baseline"]
    assert abs(base.values["level"] + 65.0) < 0.2
    assert abs(base.values["noise_sd"] - 0.3) < 0.1
    assert base.values["drift_range"] < 1.0
    assert len(base.tables["sweeps"]) == 5
    assert base.methods and base.methods[0].startswith("baseline: median")
    assert "drift measure" in base.choices
    p = r.provenance
    assert p.app_version and p.analyses["baseline"] == 1 and "max_baseline_s" in p.settings["baseline"]
    assert set(p.analyses) == {"artifacts", "baseline", "bridge", "quality", "intrinsic"}     # a current-step family
    assert p.libraries["numpy"] and p.analyzed_at
    assert r.quality() == "passed"


def test_gap_free_baseline_is_judged_in_pieces():
    sim = simulate.gap_free_events(duration_s=35.0, holding_pa=-150.0)
    r = pipeline.analyze(sim.recording)
    base = r.results["baseline"]
    assert len(base.tables["sweeps"]) == 3
    assert abs(base.values["level"] + 150.0) < 5.0


def test_a_failing_analysis_is_a_flag_not_an_exception(monkeypatch):
    @registry.register("broken", "Broken", version=1, types=["cc_steps"])
    def broken(recording, context, settings):
        raise RuntimeError("boom")

    try:
        r = pipeline.analyze(simulate.steps(n_sweeps=2).recording)
        assert "broken" in r.results and r.results["broken"].error.startswith("RuntimeError: boom")
        assert any(f.code == "software error" for f in r.flags)
        assert "baseline" in r.results                          # the others still ran
    finally:
        registry.REGISTRY.pop("broken", None)


def test_same_inputs_give_identical_numbers():
    a = pipeline.analyze(simulate.steps(seed=4).recording).values()
    b = pipeline.analyze(simulate.steps(seed=4).recording).values()
    assert a == b
