"""Agreement between Clampwise's and Clampex's recordings of the same protocol on the same circuit (a model cell): each
file measured the same way, then each program's mean and spread over its runs and the
difference between them with a bootstrap 95% interval.

    python tools/rig/agreement.py --clampwise <files...> --clampex <files...>

Each file (copied to the scratch folder first) gives: the resistance from its whole IV (each step's steady response
over the step actually delivered, read on the secondary channel, means not medians; a fit over every sweep, which resolves well below
one sample count), and per test step the analysis's own measures: in voltage clamp Rs, Rm, Cm (passive.measure_step,
steps of 20 mV or less), in current clamp Rin, tau and Cm (the charging fitted, as the membrane test does).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clampwise.analysis import passive                        # noqa: E402
from clampwise.io import load                                  # noqa: E402
from clampwise.rig import membrane_test as mt                  # noqa: E402
from clampwise.rig.growing import scratch                      # noqa: E402


def steps(cmd: np.ndarray) -> tuple[int, int] | None:
    edges = np.flatnonzero(np.abs(np.diff(cmd)) > 0.25)
    return (int(edges[0]) + 1, int(edges[1]) + 1) if len(edges) >= 2 else None


def measure(path: Path) -> dict:
    work = scratch() / "validate"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / path.name
    shutil.copyfile(path, copy)
    rec = load(copy)
    fs = rec.sample_rate_hz
    vc = rec.is_voltage_clamp
    main = rec.read(rec.channels[0])
    other = rec.read(rec.channels[1]) if len(rec.channels) > 1 else None
    cmd = rec.command
    dx, dy, per = [], [], {"rs": [], "rm": [], "cm": [], "tau": [], "rin": []}
    for k in range(rec.n_sweeps):
        s = steps(cmd[k])
        if s is None:
            continue
        i0, i1 = s
        pre = slice(max(0, i0 - int(0.01 * fs)), i0)
        tail = slice(i1 - (i1 - i0) // 5, i1)
        y, x = main[k], (other[k] if other is not None else cmd[k])
        dy.append(float(np.mean(y[tail]) - np.mean(y[pre])))           # means: the noise dithers below a count,
        dx.append(float(np.mean(x[tail]) - np.mean(x[pre])))           # a median snaps to whole counts
        step = float(np.median(cmd[k][tail]) - np.median(cmd[k][pre]))
        if vc and 1 <= abs(step) <= 20:
            m = passive.measure_step(rec.time_s, y, i0, i1, step, fs)
            per["rs"].append(m["rs_mohm"])
            per["rm"].append(m["rm_mohm"])
            per["cm"].append(m["cm_tau_pf"])
        if not vc and abs(step) >= 5:
            r = mt.measure(y[: i1 + 10], fs, i0, i1, step, "IC")
            per["rin"].append(r.rm_mohm)
            per["tau"].append(r.tau_ms)
            per["cm"].append(r.cm_pf)
    del rec, main, other, cmd
    copy.unlink(missing_ok=True)
    dx, dy = np.array(dx), np.array(dy)
    slope = float(np.polyfit(dx, dy, 1)[0]) if dx.size >= 2 else float("nan")
    out = {"file": path.name, "mode": "VC" if vc else "IC",
           "r_iv_mohm": (1e3 / slope) if vc else slope * 1e3}         # VC slope pA/mV (nS), IC mV/pA (GOhm)
    for k, v in per.items():
        v = np.array([x for x in v if np.isfinite(x)])
        if v.size:
            out[k] = float(np.median(v))
    return out


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 20_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clampwise", nargs="+", type=Path, required=True)
    ap.add_argument("--clampex", nargs="+", type=Path, required=True)
    a = ap.parse_args()
    cw = [measure(p) for p in a.clampwise]
    cx = [measure(p) for p in a.clampex]
    for r in cw + cx:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    keys = [k for k in ("r_iv_mohm", "rs", "rm", "cm", "rin", "tau") if all(k in r for r in cw + cx)]
    print(f"{'measure':10s} {'Clampwise mean (SD), n':28s} {'Clampex mean (SD), n':28s} difference [95% CI]   %")
    for k in keys:
        x, y = np.array([r[k] for r in cw]), np.array([r[k] for r in cx])
        d, lo, hi = boot_diff(x, y)
        print(f"{k:10s} {x.mean():9.3f} ({x.std(ddof=1):.3f}), {x.size:<10d} {y.mean():9.3f} ({y.std(ddof=1):.3f}), "
              f"{y.size:<10d} {d:+8.3f} [{lo:+.3f}, {hi:+.3f}]  {100 * d / y.mean():+.2f}%")


if __name__ == "__main__":
    main()
