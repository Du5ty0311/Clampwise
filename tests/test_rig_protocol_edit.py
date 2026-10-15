"""The simple stimulus editor: a protocol edited in its own header,
saved under a new name, read back with its epochs as edited; never over the original, never past a sweep."""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

import numpy as np
import pytest

from clampwise.rig import acquire, growing, protocol_edit
from clampwise.simulate.abf_file import membrane_test_file


def _pro(tmp_path, name="sim.pro"):
    head = membrane_test_file(tmp_path / "sim.abf", n_sweeps=3, holding_mv=-70.0, step_mv=-5.0).read_bytes()
    pro = tmp_path / name
    pro.write_bytes(head[:growing._section(head, growing.DATA)[0] * growing.BLOCK])
    return pro


def test_an_edit_is_saved_under_a_new_name_and_plays_as_edited(tmp_path):
    e = protocol_edit.load(_pro(tmp_path))
    assert [round(x.level) for x in e.epochs[0]] == [-70, -75, -70] and e.sweeps == 3
    e.epochs[0][1].level, e.epochs[0][1].level_inc = -80.0, 10.0       # an IV: -80, -70, -60 mV
    e.epochs[0][2].samples -= 2000 - e.epochs[0][1].samples             # the step longer, the rest shorter
    e.epochs[0][1].samples = 2000
    e.holding[0] = -65.0
    e.sweeps = 4
    with pytest.raises(FileExistsError):
        protocol_edit.save(e, tmp_path / "sim.pro")                       # never over the original
    pr = protocol_edit.save(e, tmp_path / "sim_iv")
    assert pr.path.name == "sim_iv.pro" and pr.sweeps == 4 and pr.holding[0] == -65.0
    units, _ = acquire.waveforms(pr)
    pre = pr.sweep_frames // 64
    a = pre + e.epochs[0][0].samples
    assert [round(float(units[k, 0, a + 10])) for k in range(4)] == [-80, -70, -60, -50]
    assert round(float(units[0, 0, a + 2000 + 10])) == -70 and round(float(units[0, 0, 0])) == -65


def test_an_edit_past_the_sweep_or_with_no_room_is_refused(tmp_path):
    e = protocol_edit.load(_pro(tmp_path))
    e.epochs[0][1].samples_inc = 4000                                    # longer each sweep: past the end by sweep 3
    with pytest.raises(ValueError, match="past the sweep"):
        protocol_edit.to_bytes(e)
    e = protocol_edit.load(_pro(tmp_path))
    e.epochs[0] += [protocol_edit.Epoch(1, -70.0, samples=1) for _ in range(40)]
    with pytest.raises(ValueError, match="room for"):
        protocol_edit.to_bytes(e)
    e = protocol_edit.load(_pro(tmp_path))
    e.epochs[0][1].kind, e.epochs[0][1].period, e.epochs[0][1].width = 3, 100, 0
    with pytest.raises(ValueError, match="pulse train"):
        protocol_edit.to_bytes(e)


RIG = sorted(glob.glob(str(Path(acquire.protocols_folder()) / "*.pro")))      # the folder Clampex itself uses here


@pytest.mark.skipif(not RIG, reason="no Clampex protocols on this PC")
def test_the_rig_s_protocols_come_back_unchanged_when_nothing_is_edited(tmp_path):
    for p in RIG:
        src = tmp_path / Path(p).name
        shutil.copyfile(p, src)
        pr = acquire.read_protocol(src)
        if pr.mode != growing.EPISODIC:
            continue
        e = protocol_edit.load(src)
        blob = protocol_edit.to_bytes(e)
        before, after = acquire.waveforms(pr), acquire.waveforms(protocol_edit.save(e, tmp_path / ("same_" + src.name)))
        assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]), p
        assert len(blob) == len(pr.head)
