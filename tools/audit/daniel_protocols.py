"""Daniel's protocols, checked by hand: the measures an expert would take in Clampfit, made independently of the
analyses (plain numpy on the raw sweeps, the windows written here), against what Clampwise gives.

    python tools/audit/daniel_protocols.py E:\\Daniel [--all]

For each protocol, two files (every file with --all): the hand measure, Clampwise's, and the difference. The
windows are the lab's: steady state the last 100 ms of a step, baseline the 100 ms before it, Rin a line over the
hyperpolarizing steps (Ahmed 2021), the first spike at rheobase for its shape, the light response at the 0 pA step
against the 20 ms before the light. Read in place; nothing is written.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np


def _spike_shape(t, v, fs, i0, i1):
    """The first spike in [i0, i1): threshold at 20 mV/ms, peak, amplitude, half-width at half the amplitude from
    the threshold (Ahmed 2021), AHP 1 ms before the peak minus the trough within 10 ms (Ahmed 2021)."""
    d = np.diff(v) * fs / 1e3
    seg = d[i0:i1]
    above = seg >= 20.0
    runs = np.convolve(above.astype(int), np.ones(2, int), "valid") == 2
    hits = np.flatnonzero(runs)
    if not hits.size:
        return None
    thr_i = i0 + int(hits[0])
    peak_i = thr_i + int(np.argmax(v[thr_i:thr_i + int(0.005 * fs)]))
    if v[peak_i] - v[thr_i] < 20:
        return None
    thr, peak = float(v[thr_i]), float(v[peak_i])
    half = thr + (peak - thr) / 2
    up = np.flatnonzero(v[thr_i:peak_i + 1] < half)
    dn = np.flatnonzero(v[peak_i:peak_i + int(0.05 * fs)] < half)
    hw = ((peak_i + dn[0]) - (thr_i + up[-1])) / fs * 1e3 if up.size and dn.size else np.nan
    ahp = float(v[peak_i - int(0.001 * fs)] - v[peak_i:peak_i + int(0.01 * fs)].min())
    return {"threshold_20": thr, "peak": peak, "amplitude": peak - thr, "half_width": hw, "ahp": ahp,
            "time_ms": peak_i / fs * 1e3}


def steps_by_hand(rec) -> dict:
    """A current-step family: Rin (the lab's line), tau (10-95% of the way to the peak, the smallest
    hyperpolarizing step deeper than 2 mV), rheobase, the first spike's shape, sag (Ahmed 2021), max rate (Ahmed)."""
    from clampwise.analysis.tools import fits

    fs, t = rec.sample_rate_hz, rec.time_s
    cmd, sig = rec.command, rec.signal
    out = {}
    # the step from the command
    c0 = cmd[int(np.argmax(np.abs(cmd).max(axis=1)))]
    hold = np.median(c0[:c0.size // 64])
    on = np.flatnonzero(np.abs(c0 - hold) > 0.5)
    a, b = int(on[0]), int(on[-1]) + 1
    cur = np.array([float(np.median(cmd[k, a:b]) - np.median(cmd[k, :a])) for k in range(rec.n_sweeps)])
    vb = np.array([float(np.mean(sig[k, a - int(0.1 * fs):a])) for k in range(rec.n_sweeps)])
    vss = np.array([float(np.mean(sig[k, b - int(0.1 * fs):b])) for k in range(rec.n_sweeps)])
    d = np.diff(sig, axis=1) * fs / 1e3
    fired = np.array([bool((d[k, a:b] >= 20).any() and (sig[k, a:b].max() - vb[k] >= 20)) for k in range(rec.n_sweeps)])
    out["v_rest"] = float(np.median(vb))
    sub = (~fired) & (cur <= 3) & (cur >= -203)
    if np.unique(np.round(cur[sub])).size >= 2:
        out["rin_lab"] = float(np.polyfit(cur[sub], vss[sub], 1)[0]) * 1e3
    hyper = [k for k in range(rec.n_sweeps) if cur[k] < -3 and not fired[k]]
    if hyper and b - a > int(0.15 * fs):                  # a step long enough for a peak before its last 100 ms
        k = min(hyper, key=lambda k: abs(cur[k]))                     # the smallest hyperpolarizing step
        v = sig[k]
        pk = a + int(np.argmin(v[a:b - int(0.1 * fs)]))
        defl = v[pk] - vb[k]
        if abs(defl) > 2.0:
            lo = a + int(np.flatnonzero(v[a:pk + 1] <= vb[k] + 0.1 * defl)[0])
            hi = a + int(np.flatnonzero(v[a:pk + 1] <= vb[k] + 0.95 * defl)[0])
            if hi - lo > 10:
                fit = fits.exp_decay(t[lo:hi], v[lo:hi])
                out["tau_ms"] = fit["tau"] * 1e3 if fit.ok else np.nan
        deepest = min(hyper, key=lambda k: cur[k])
        v = sig[deepest]
        out["sag_ahmed"] = float(v[b - int(0.1 * fs):b].min() / v[a:a + int(0.1 * fs)].min())
        out["sag_step"] = float(cur[deepest])
    dep = [k for k in range(rec.n_sweeps) if cur[k] > 3 and fired[k]]
    if dep:
        rk = min(dep, key=lambda k: cur[k])
        out["rheobase"] = float(cur[rk])
        shape = _spike_shape(t, sig[rk], fs, a, b)
        if shape:
            out.update({"ap_amplitude": shape["amplitude"], "ap_half_width": shape["half_width"],
                        "ap_ahp": shape["ahp"], "ap_threshold_20": shape["threshold_20"]})
    near = [k for k in range(rec.n_sweeps) if abs(cur[k] - 200) <= 25]
    if near:
        k = min(near, key=lambda k: abs(cur[k] - 200))
        seg = d[k, a:b]
        runs = np.convolve((seg >= 20).astype(int), np.ones(2, int), "valid") == 2
        starts = np.flatnonzero(np.diff(np.r_[0, runs.astype(int)]) == 1)
        starts = [s for i, s in enumerate(starts) if i == 0 or s - starts[i - 1] > 0.002 * fs]
        isi = np.diff(starts) / fs
        out["max_rate_ahmed"] = float(1 / isi.mean()) if isi.size else 0.0
    return out


def light_by_hand(rec) -> dict:
    """A photostim file: at the 0 pA sweep (or every sweep when the command never moves), pulse 1's response
    against the 20 ms before the light: the peak within 50 ms, the onset past twice the noise sustained 0.5 ms,
    and whether the response begins under 1.5 ms and is at its height when the light ends."""
    from clampwise.analysis import stimulus as st_mod

    fs, t = rec.sample_rate_hz, rec.time_s
    st = st_mod.find_stimulus(rec)
    if not len(st):
        return {"note": "no stimulus found by hand either"}
    cmd, sig = rec.command, rec.signal
    rows = []
    for k in range(rec.n_sweeps):
        if not st.times_s[k].size:
            continue
        on = float(st.times_s[k][0])
        w = float(st.widths_s[k][0]) if st.widths_s[k].size else 0.005
        i0 = int(round(on * fs))
        level = float(np.median(cmd[k, i0 - 100:i0])) if cmd is not None else 0.0
        base = sig[k, i0 - int(0.02 * fs):i0 - 10]
        b = float(np.median(base))
        noise = 1.4826 * float(np.median(np.abs(base - b)))
        seg = sig[k, i0:i0 + int(0.05 * fs)] - b
        pk = int(np.argmax(np.abs(seg)))
        amp = float(seg[pk])
        sign = np.sign(amp) or 1.0
        above = sign * seg > 2 * noise
        runs = np.convolve(above.astype(int), np.ones(int(0.0005 * fs), int), "valid") == int(0.0005 * fs)
        hit = np.flatnonzero(runs[: pk + 1])
        onset = hit[0] / fs * 1e3 if hit.size else np.nan
        at_end = float(sign * seg[int(round(w * fs)) - 1]) if int(round(w * fs)) - 1 < seg.size else np.nan
        rows.append({"sweep": k, "level": level, "amp": amp, "onset_ms": onset, "peak_ms": pk / fs * 1e3,
                     "at_light_end_frac": at_end / abs(amp) if amp else np.nan, "noise": noise})
    zero = [r for r in rows if abs(r["level"]) < 1.0]
    head = zero[0] if zero else (rows[len(rows) // 2] if rows else None)
    out = {"n_sweeps_with_light": len(rows)}
    if head:
        out.update({"p1_amplitude_0pA": head["amp"], "p1_onset_ms": head["onset_ms"], "p1_peak_ms": head["peak_ms"],
                    "direct_by_hand": bool(np.isfinite(head["onset_ms"]) and head["onset_ms"] < 1.5
                                           and head["at_light_end_frac"] >= 0.5 and head["peak_ms"] <= 8.0)})
    levels = np.array([r["level"] for r in rows])
    if np.unique(np.round(levels)).size >= 3:
        vm = np.array([float(np.median(sig[r["sweep"], int(round(st.times_s[r["sweep"]][0] * fs)) - 400:
                                           int(round(st.times_s[r["sweep"]][0] * fs)) - 10])) for r in rows])
        amps = np.array([r["amp"] for r in rows])
        slope, icpt = np.polyfit(vm, amps, 1)
        out["reversal_mv"] = float(-icpt / slope) if slope else np.nan
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(r"E:\Daniel")
    every = "--all" in argv
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from clampwise.analysis import pipeline
    from clampwise.io import load

    by_proto: dict[str, list[Path]] = {}
    for p in sorted(glob.glob(str(root / "**" / "*.abf"), recursive=True)):
        try:
            rec = load(p)
        except Exception as exc:
            print(p, "could not be read:", exc)
            continue
        by_proto.setdefault(rec.protocol, []).append(Path(p))
    for proto, files in sorted(by_proto.items()):
        print("=" * 100)
        print(f"{proto} ({len(files)} files)")
        picked = files if every else [f for f in files][:2]
        for f in picked:
            rec = load(f)
            res = pipeline.analyze(rec)
            print(f"  {f.name}  [{res.recording_type}]")
            it = res.results.get("intrinsic")
            ev = res.results.get("evoked")
            try:
                if rec.command is not None and not rec.is_voltage_clamp and rec.n_sweeps >= 3 \
                        and "steps" in res.recording_type or (it is not None and it.values.get("r_input") is not None):
                    hand = steps_by_hand(rec)
                    for key, cw in (("v_rest", "v_rest"), ("rin_lab", "r_input_lab"), ("tau_ms", "tau_m"),
                                    ("rheobase", "rheobase"), ("ap_amplitude", "ap_amplitude_20"),
                                    ("ap_half_width", "ap_half_width_lab"), ("ap_ahp", "ap_ahp_lab"),
                                    ("sag_ahmed", "sag_ratio_lab"), ("max_rate_ahmed", "max_rate_lab")):
                        if key in hand and it is not None:
                            h, c = hand[key], it.values.get(cw, np.nan)
                            print(f"      {key:16s} hand {h:10.3f}   Clampwise {c:10.3f}   diff {c - h:+.3f}")
                if ev is not None:
                    hand = light_by_hand(rec)
                    v = ev.values
                    for key, cw in (("p1_amplitude_0pA", "p1_amplitude"), ("p1_onset_ms", "p1_latency_median"),
                                    ("reversal_mv", "reversal")):
                        if key in hand:
                            c = v.get(cw, np.nan)
                            c = np.nan if c is None else c
                            print(f"      {key:16s} hand {hand[key]:10.3f}   Clampwise {c:10.3f}")
                    if "direct_by_hand" in hand:
                        print(f"      direct           hand {hand['direct_by_hand']}   Clampwise {v.get('direct_photocurrent')}")
                    cn = res.results.get("connectivity")
                    if cn is not None:
                        print(f"      connected: {cn.values.get('connected')} (p {cn.values.get('p', np.nan):.3f})")
            except Exception as exc:
                print("      hand check failed:", type(exc).__name__, exc)
            for fl in res.flags[:4]:
                print("      flag:", fl.code, "|", fl.text[:110])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
