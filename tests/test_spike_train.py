"""B24: spike trains, from files of spike times and from spontaneous firing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from clampwise import simulate
from clampwise.analysis import pipeline
from clampwise.io import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import known_answers  # noqa: E402


def test_the_known_answers_pass():
    bad = [r for r in known_answers.run(["spike train"]) if not r.ok]
    assert not bad, "\n".join(f"{r.case}: {r.measure} true {r.truth:.4g} measured {r.measured:.4g}" for r in bad)


def test_a_file_of_two_units_reads_each(tmp_path):
    p = tmp_path / "two.asc"
    p.write_text("0.1\t0.15\n0.5\t0.9\n1.2\t1.3\n2.0\t2.2\n")
    rec = load(p)
    assert rec.header["recording_kind"] == "spike_times" and len(rec.header["spike_times_s"]) == 2
    r = pipeline.analyze(rec)
    assert r.recording_type == "spike_times" and list(r.results) == ["spike_train"]
    assert [u["n_spikes"] for u in r.results["spike_train"].tables["units"]] == [4, 4]


def test_a_file_without_times_says_so(tmp_path):
    p = tmp_path / "empty.asc"
    p.write_text("no numbers here\n")
    with pytest.raises(ValueError, match="no spike times"):
        load(p)


def test_spontaneous_firing_in_current_clamp_is_a_spike_train():
    sim = simulate.spontaneous(duration_s=30.0, voltage_clamp=False, rate_hz=1.0, amp=1.0, noise=0.2,
                               baseline=-65.0, spikes_at=tuple(np.arange(1.0, 29.0, 0.7)), seed=3)
    r = pipeline.analyze(sim.recording)
    assert r.recording_type == "cc_spontaneous"
    st = r.results["spike_train"]
    assert st.values["n_spikes"] == len(np.arange(1.0, 29.0, 0.7))
    assert abs(st.values["isi_median_s"] - 0.7) < 1e-3
    assert "detected in the trace" in st.choices["spikes"]


def test_neuroexpress_test_train_matches_its_counts():
    from labdata import DEMOS

    p = DEMOS / "Test_SpT.ASC"
    if not p.is_file():
        pytest.skip("NeuroExpress's test files are not on this machine")
    st = pipeline.analyze(load(p)).results["spike_train"]
    assert st.values["n_spikes"] == 1618                     # NeuroExpress: 1618
    assert st.values["n_bursts"] == 136                      # NeuroExpress: 136 (intraburst limit)
    assert abs(st.values["rate_hz"] - 1.93) < 0.02           # NeuroExpress: 1.93 Hz
    assert abs(st.values["isi_median_s"] - 0.086) < 0.002    # NeuroExpress: 0.086 s
