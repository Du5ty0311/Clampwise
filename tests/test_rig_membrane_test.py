"""The membrane test in Clampwise: the model cell's three positions read as they are (CELL a
cell with its Rs, Rm, Cm; BATH the pipette alone; PATCH a seal), in voltage and current clamp, live, nothing kept."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from clampwise.rig import membrane_test as mt
from clampwise.rig import telegraph

FS = 25_000.0                                   # 500 samples a period at 50 Hz, as Clampex's Cell stage


def _pulse(rs=10.0, rm=500.0, cm=33.0, amp=10.0, period=500, noise=0.5, seed=0, fs=FS):
    """One period of the current (pA) for a square wave ``amp`` mV peak to peak centred on the holding, high for the
    middle half (Rs-(Rm||Cm), exact, settled into its period): each sample the current averaged over its interval,
    as a filtered, sampled current carries the transient's whole charge."""
    up, down = period // 4, 3 * period // 4
    rs_, rm_, cm_ = rs * 1e6, rm * 1e6, cm * 1e-12
    tau = rs_ * rm_ * cm_ / (rs_ + rm_)
    dt = 1.0 / fs
    vc = np.where((np.arange(period) >= up) & (np.arange(period) < down), amp / 2, -amp / 2) * 1e-3
    vm = 0.0
    y = np.zeros(period)
    for _ in range(3):                                     # three periods: the last one settled
        for k in range(period):
            inf = vc[k] * rm_ / (rs_ + rm_)
            # the mean over the interval of (vc - vm) / rs, vm relaxing toward inf with tau
            mean_vm = inf + (vm - inf) * tau / dt * (1 - np.exp(-dt / tau))
            y[k] = (vc[k] - mean_vm) / rs_ * 1e12
            vm = inf + (vm - inf) * np.exp(-dt / tau)
    return np.roll(y, 1) - 20.0 + np.random.default_rng(seed).normal(0, noise, period)   # one sample's delay


def test_the_model_cell_s_positions_read_as_they_are():
    y = _pulse()                                                            # CELL
    r = mt.measure(y, FS, 10.0, "VC")
    assert r.what == "cell" and r.rt_mohm == pytest.approx(510, rel=0.01)
    assert r.rs_mohm == pytest.approx(10, rel=0.03) and r.rm_mohm == pytest.approx(500, rel=0.01)
    assert r.cm_pf == pytest.approx(33, rel=0.03) and r.holding == pytest.approx(-20, abs=0.5)
    assert r.tau_ms == pytest.approx(0.3235, rel=0.03) and len(r.fits) == 2      # both edges
    y = _pulse(rs=10.0, rm=0.001, cm=0.001)                                 # BATH: 10 MOhm, nothing else
    r = mt.measure(y, FS, 10.0, "VC")
    assert r.what == "open pipette" and r.rt_mohm == pytest.approx(10, rel=0.02) and np.isnan(r.rs_mohm)
    y = _pulse(rs=10_000.0, rm=0.001, cm=0.001, noise=3.0)                 # PATCH: 10 GOhm, 1 pA under the noise
    r = mt.measure(y, FS, 10.0, "VC")
    assert r.what == "seal" and np.isnan(r.rs_mohm)


def test_the_amplifier_s_compensation_spike_at_the_edge_is_not_the_cell_s_charge():
    """The rig's 700B puts three samples of the other sign at each edge (its fast capacitance compensation: -319,
    -1164, -116 pA on a 10 mV step); Clampex's Ra and Cm on the model cell are those with it left out."""
    y = np.roll(_pulse(), 3)                                # the transient three samples after the edge, as on the rig
    clean = mt.measure(y, FS, 10.0, "VC")
    spiked = y.copy()
    for at, s in ((125, 1), (375, -1)):                     # the spike in those three samples, of the other sign
        spiked[at + 1:at + 4] -= s * np.array([319.0, 1164.0, 116.0])
    r = mt.measure(spiked, FS, 10.0, "VC")
    assert r.rs_mohm == pytest.approx(clean.rs_mohm, rel=0.02) and r.cm_pf == pytest.approx(clean.cm_pf, rel=0.02)


def test_rt_is_pclamps_the_means_of_the_last_fifth_of_each_half():
    """Rt = dV / (I1 - I2), I1 and I2 the mean current over the last 20% of each half (pCLAMP 10 User Guide p. 192):
    a mean, so the noise dithers below one count of the digitizer, where a median snaps to whole counts."""
    count = 6.1035
    rng = np.random.default_rng(1)
    y = np.mean([np.round((_pulse(noise=0.0) + rng.normal(0, 12.0, 500)) / count) * count for _ in range(50)], axis=0)
    r = mt.measure(y, FS, 10.0, "VC")
    assert r.rt_mohm == pytest.approx(510, rel=0.03)


def test_current_clamp_gives_rin_and_cm():
    period = 500
    fs = 2_500.0                                                            # 5 Hz: 2.5 kHz, as Clampex's rule
    up, down = period // 4, 3 * period // 4
    t = np.arange(period) / fs
    y = np.full(period, -30.0)
    y[up:down] += 50.0 * (1 - np.exp(-(t[up:down] - t[up]) / 0.0165)) + 1.0    # 100 pA: 500 MOhm charging
    r = mt.measure(y, fs, 100.0, "IC")
    assert r.what == "cell" and r.rt_mohm == pytest.approx(510, rel=0.03) and r.cm_pf == pytest.approx(33, rel=0.1)


class FakeHelper:
    """The digitizer as a stand-in: the model cell's current for the test pulse, into a ring file, until stopped."""

    def __init__(self):
        self.calibration = {"dac_gains": [1.0] * 4, "dac_offsets": [0] * 4}
        self.req = None

    def record(self, req, progress=None):
        self.req = req
        period = os.path.getsize(req["waveform"]) // 10
        ring, stop, done = req["ring_frames"], req["out"] + ".stop", 0
        with open(req["out"], "wb") as fh:
            while not os.path.exists(stop):
                y = _pulse(period=period, noise=12.0, seed=done)             # the rig's noise: it dithers the counts
                counts = np.round(y * 5e-5 * 32768 / 10.0).astype("<i2")     # 0.05 V/nA at the digitizer
                fh.seek(2 * (done % ring))
                fh.write(counts.tobytes())
                fh.flush()
                done += period
                if progress:
                    progress(done)
                time.sleep(period / FS)
        return {"done": done}

    def close(self):
        pass


def test_the_test_runs_live_until_stopped_and_keeps_nothing(tmp_path):
    tg = telegraph.Telegraph("VC", "pA", 5e-5, 1.0, 10_000.0, "mV", 0.01, 1.0, 20.0, "00123456", 1, 0.0, 0.0)
    helper = FakeHelper()
    test = mt.MembraneTest(work=tmp_path, read_telegraph=lambda *a, **k: tg, make_helper=lambda: helper)
    test.start()
    end = time.time() + 15
    while time.time() < end and (test.latest is None or len(test.history) < 3):
        time.sleep(0.05)
    r = test.latest
    # 10 mV over 510 MOhm is 19.6 pA, 3.2 counts of 6.1 pA at 0.05 V/nA: the rig's noise dithers them and the
    # pulses averaged recover the fraction (without noise it read 546 MOhm, as the model cell's 20 mV steps did)
    assert r is not None and r.what == "cell" and r.rt_mohm == pytest.approx(510, rel=0.1) and len(r.trace) == 500
    assert helper.req["repeat"] and helper.req["ring_frames"] == 500 * mt.RING_PULSES
    assert helper.req["period_us"] == pytest.approx(40.0) and helper.req["say_s"] == mt.READ_EVERY_S
    wave = np.fromfile(helper.req["waveform"], "<i2").reshape(-1, 5)[:, 0]    # centred on the holding: +-5 mV
    assert wave[0] == -wave[250] and wave[0] < 0 and len(set(wave[125:375])) == 1
    test.stop()
    test.wait(10)
    assert test.state == "idle" and not list(tmp_path.glob("test.*"))
