"""Clampwise's reading of ABF files against Axon's own (ABFFIO.dll through ``axon_read.py`` and the 32-bit Python):
every sweep's values and start time, file by file.

    python tools/rig/axon_compare.py <file.abf> [...]

Each file is copied to Clampwise's scratch folder first (nothing is opened where the data are). Prints, per file,
the sweeps Axon and Clampwise read, the largest difference between their values (in the channel's unit, and in the
digitizer's counts), and the largest difference between their sweep start times.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clampwise.io import load                     # noqa: E402
from clampwise.rig import acquire                 # noqa: E402
from clampwise.rig.digidata import py32           # noqa: E402
from clampwise.rig.growing import scratch         # noqa: E402

HERE = Path(__file__).resolve().parent


def axon(path: Path, channels: list[int]) -> dict:
    out = subprocess.run([str(py32()), "-I", str(HERE / "axon_read.py"), str(path)] + [str(c) for c in channels],
                         capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not out.stdout.strip():
        return {"error": f"Axon's reader said nothing (exit {out.returncode}): {out.stderr.strip()[-300:]}"}
    return json.loads(out.stdout)


def compare(path: Path) -> dict:
    work = scratch() / "validate"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / path.name
    shutil.copyfile(path, copy)
    pr = acquire.read_protocol(copy)
    scales = [v for _, _, v in acquire.inputs(pr)]
    a = axon(copy, pr.ai)
    if "error" in a:
        return {"file": path.name, "error": a["error"]}
    rec = load(copy)
    worst_value = worst_counts = worst_t = 0.0
    refused = 0
    for i, ch in enumerate(pr.ai):
        rows = a["channels"][str(ch)]
        refused += sum("error" in r for r in rows)             # a piece Axon's reader refused: counted, said
        rows = [r for r in rows if "error" not in r]
        data = rec.read(rec.channels[i]) if i < len(rec.channels) else None
        if pr.mode == acquire.GAP_FREE and data is not None:  # Axon reads gap-free in pieces: the whole trace
            n = sum(r["n"] for r in rows)
            mean = sum(r["mean"] * r["n"] for r in rows) / n
            ours = np.asarray(data[0], float)
            diff = max(float(np.max(np.abs(ours[:5] - np.asarray(rows[0]["first"], float)))),
                       abs(float(np.mean(ours)) - mean), abs(ours.size - n))
            worst_value = max(worst_value, diff)
            worst_counts = max(worst_counts, diff / ((10.0 / 32768) / scales[i]))
            continue
        for k, row in enumerate(rows):
            if data is None or k >= len(data):
                continue
            ours = np.asarray(data[k][:5], float)
            diff = float(np.max(np.abs(ours - np.asarray(row["first"], float))))
            diff = max(diff, abs(float(np.mean(data[k])) - row["mean"]))
            worst_value = max(worst_value, diff)
            counts = (10.0 / 32768) / scales[i]
            worst_counts = max(worst_counts, diff / counts)
            if k < len(rec.sweep_start_times_s):
                worst_t = max(worst_t, abs(rec.sweep_start_times_s[k] * 1e3 - (row["start_ms"] - rows[0]["start_ms"])))
    del rec
    copy.unlink(missing_ok=True)
    return {"file": path.name, "sweeps_axon": a["episodes"], "samples_axon": a["samples"],
            "worst_value_diff": worst_value, "worst_diff_counts": worst_counts, "worst_start_ms": worst_t,
            "refused_by_axon": refused}


def main() -> None:
    for f in sys.argv[1:]:
        r = compare(Path(f))
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
