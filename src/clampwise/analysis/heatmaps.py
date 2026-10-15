"""Heat maps as data (Phase 2; methods read from Synaptoscope's heatmap_data and
Cohen 2014, Analyzing Neural Time Series Data, chapters 12-13 and 18). Pure
functions, no Qt: the panel draws only what a ``Heatmap`` holds, so what a map
shows is tested here.

The rules, for every map (stated in the manual, "Heat maps"):
- A missing value stays missing (NaN, drawn blank), never zero.
- **Sequential** data (power, rate, counts, a signal) use a perceptually
  uniform map (cividis); limits are the 1st-99th percentile by default, or the
  full range, or fixed. When most of the matrix sits at its minimum (a PSTH is
  mostly empty bins) the percentiles are taken over the values above it, so
  the few busy bins do not all saturate.
- **Diverging** data (% of baseline, ratios to pulse 1, dB from a median,
  correlations) are symmetric about their meaningful centre (100, 1, 0).
- Limits can be shared by the maps compared side by side (``shared_levels``).

Each builder takes a run (its results and recording) and returns a Heatmap,
or None when the run has nothing for it; ``catalogue`` says which a run can
make without building them, so a map costs nothing until it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable

import numpy as np

NAN = float("nan")
MAX_COLUMNS = 1500          # wider rows are reduced to this many columns for drawing
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 80),
         ("high gamma", 80, 150), ("ripple", 150, 250))


@dataclass
class Marker:
    """An event on the map: a line at ``at`` (``end`` None) or a span, along x or y."""
    axis: str                   # "x" or "y"
    at: float
    end: float | None = None
    label: str = ""


@dataclass
class Heatmap:
    matrix: np.ndarray          # rows (y) x columns (x); NaN where there is no value
    x: np.ndarray               # column centres, in x units
    y: np.ndarray               # row centres, in y units
    title: str
    x_label: str
    x_unit: str
    y_label: str
    y_unit: str
    value_label: str
    value_unit: str
    scale: str = "sequential"   # or "diverging"
    center: float = 0.0         # a diverging map's neutral value
    fixed: tuple[float, float] | None = None       # limits that belong to the measure (a correlation: -1 to 1)
    power: bool | None = None   # the values can be shown in dB: True for a power (10 log10), False for an amplitude
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None
    markers: list[Marker] = field(default_factory=list)
    mask: np.ndarray | None = None                 # True where a cell is marked (significant)
    notes: list[str] = field(default_factory=list)

    @property
    def colour_label(self) -> str:
        return f"{self.value_label} ({self.value_unit})" if self.value_unit else self.value_label

    def extent(self) -> tuple[float, float, float, float]:
        """x0, x1, y0, y1 of the image: the centres plus half a step each side."""
        def edges(c):
            c = np.asarray(c, float)
            if c.size < 2:
                v = float(c[0]) if c.size else 0.0
                return v - 0.5, v + 0.5
            return float(c[0] - (c[1] - c[0]) / 2), float(c[-1] + (c[-1] - c[-2]) / 2)
        return (*edges(self.x), *edges(self.y))


# --------------------------------------------------------------------------- the colour scale


def levels(hm: Heatmap, mode: str = "percentile", fixed: tuple[float, float] | None = None,
           lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """Colour limits: ``mode`` "percentile" (1st-99th), "full" or "fixed"."""
    if mode == "fixed" and fixed is not None:
        return float(fixed[0]), float(fixed[1])
    if hm.fixed is not None:
        return hm.fixed
    finite = np.asarray(hm.matrix, float)[np.isfinite(hm.matrix)]
    c = hm.center
    if not finite.size:
        return (c - 1.0, c + 1.0) if hm.scale == "diverging" else (0.0, 1.0)
    if hm.scale == "diverging":
        dev = np.abs(finite - c)
        v = float(dev.max() if mode == "full" else np.percentile(dev, hi)) or float(dev.max()) or 1.0
        return c - v, c + v
    floor = float(finite.min())
    if mode == "full":
        a, b = floor, float(finite.max())
    elif np.mean(finite == floor) > 0.5 and (finite > floor).any():
        a, b = floor, float(np.percentile(finite[finite > floor], hi))      # mostly empty: scale on what is there
    else:
        a, b = float(np.percentile(finite, lo)), float(np.percentile(finite, hi))
    return (a, b) if b > a else (a, a + 1.0)


def shared_levels(maps: list[Heatmap], mode: str = "percentile") -> tuple[float, float]:
    """One scale for maps compared side by side: the limits of all their values together."""
    if not maps:
        return 0.0, 1.0
    pooled = replace(maps[0], matrix=np.concatenate([np.ravel(m.matrix) for m in maps])[None, :])
    return levels(pooled, mode)


def in_db(hm: Heatmap) -> Heatmap:
    """The same map in decibels: 10 log10 of a power, 20 log10 of an amplitude (re 1 of its unit)."""
    if hm.power is None:
        return hm
    m = np.asarray(hm.matrix, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = (10.0 if hm.power else 20.0) * np.log10(np.where(m > 0, m, np.nan))
    unit = hm.value_unit.replace("²", "") if hm.power else hm.value_unit
    return replace(hm, matrix=db, value_unit=f"dB re 1 {unit}".strip(), power=None, fixed=None)


def _reduce(t: np.ndarray, m: np.ndarray, how: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """At most MAX_COLUMNS columns: blocks of columns averaged ("mean", for power and
    rates) or represented by their most extreme value ("extreme", so a spike or a
    response survives the reduction of a trace)."""
    n = m.shape[1]
    if n <= MAX_COLUMNS:
        return t, m
    k = int(np.ceil(n / MAX_COLUMNS))
    cut = (n // k) * k
    blocks = m[:, :cut].reshape(m.shape[0], -1, k)
    tt = t[:cut].reshape(-1, k).mean(axis=1)
    if how == "mean":
        with np.errstate(invalid="ignore"):
            return tt, np.nanmean(blocks, axis=2)
    dev = np.abs(blocks - np.nanmedian(m, axis=1)[:, None, None])
    pick = np.nanargmax(np.where(np.isfinite(dev), dev, -np.inf), axis=2)
    return tt, np.take_along_axis(blocks, pick[..., None], axis=2)[..., 0]


# --------------------------------------------------------------------------- the maps


def sweeps(run) -> Heatmap | None:
    """Every sweep of a recording against time: the recorded current or voltage."""
    rec = getattr(run, "recording", None)
    if rec is None or rec.n_sweeps < 2 or rec.is_gap_free:
        return None
    fs = rec.sample_rate_hz
    m = np.array([np.asarray(rec.sweep(k), float) for k in range(rec.n_sweeps)])
    t, m = _reduce(np.arange(m.shape[1]) / fs, m, "extreme")
    markers = [Marker("x", float(s), label="stimulus") for s in rec.stimulus.fixed_times()]
    cmd = rec.command
    if cmd is not None and cmd.shape[0] and np.ptp(cmd[0]) > 0:
        edges = np.flatnonzero(np.abs(np.diff(cmd[0])) > 0.05 * np.ptp(cmd[0]))
        markers += [Marker("x", float(i + 1) / fs, label="step") for i in edges[:2]]
    field_ = run.results.get("field") if getattr(run, "results", None) else None
    rows = field_.tables.get("sweeps", []) if field_ is not None else []
    induction = field_.values.get("induction_min", NAN) if field_ is not None else NAN
    if rows and np.isfinite(induction):
        after = [r["sweep"] for r in rows if r["time_min"] >= induction]
        if after:
            markers.append(Marker("y", after[0] + 0.5, label="induction"))
    notes = []
    if m.shape[1] < rec.n_samples:
        notes.append("each column stands for a stretch of samples, shown by its value farthest from the sweep's "
                     "median, so spikes and responses are not averaged away")
    return Heatmap(m, t, np.arange(rec.n_sweeps) + 1.0, "Every sweep vs. time", "Time", "s", "Sweep", "",
                   rec.signal_channel.name, rec.signal_units, markers=markers, notes=notes)


def dsi_course(run, akey: str = "dsi", col: str = "pct") -> Heatmap | None:
    """DSI: each episode's responses as % of its own baseline (diverging about 100)."""
    res = run.results.get(akey)
    rows = res.tables.get("course", []) if res is not None else []
    if not rows:
        return None
    eps = sorted({r["episode"] for r in rows})
    lo, hi = min(r["response"] for r in rows), max(r["response"] for r in rows)
    x = np.arange(lo, hi + 1, dtype=float)
    m = np.full((len(eps), x.size), NAN)
    for r in rows:
        m[eps.index(r["episode"]), int(r["response"] - lo)] = r[col]
    dur = {r["episode"]: r["duration_s"] for r in rows}
    what = "Peak" if col == "pct" else "Fourier power"
    return Heatmap(m, x, np.array(eps, float), f"DSI: {what.lower()}, % of baseline, every episode",
                   "Response from the step (before: negative)", "", "Episode", "", f"{what}, % of baseline", "%",
                   "diverging", 100.0, row_labels=[f"{e} ({dur[e]:g} s)" for e in eps],
                   markers=[Marker("x", 0.0, label="step")],
                   notes=["the blank column at 0 is the step: no response is measured during it"])


def dsi_fourier(run, akey: str = "dsi") -> Heatmap | None:
    """Every response's Fourier amplitude against frequency (sequential; dB optional)."""
    res = run.results.get(akey)
    rows = res.tables.get("fourier", []) if res is not None else []
    if not rows:
        return None
    order: dict[tuple, int] = {}
    for r in rows:
        order.setdefault((r.get("file", ""), r["sweep"], r["pulse"]), len(order))
    freqs = np.array(sorted({r["frequency_hz"] for r in rows}), float)
    where = {f: i for i, f in enumerate(freqs)}
    m = np.full((len(order), freqs.size), NAN)
    for r in rows:
        m[order[(r.get("file", ""), r["sweep"], r["pulse"])], where[r["frequency_hz"]]] = r["fft_amplitude"]
    units = res.values.get("units", "")
    labels = [f"{f + ' ' if f else ''}sweep {s + 1}, pulse {p}" for f, s, p in order]
    return Heatmap(m, freqs, np.arange(len(order)) + 1.0, "DSI: Fourier amplitude of every response",
                   "Frequency", "Hz", "Response, in recorded order", "", "Fourier amplitude", units,
                   power=False, row_labels=labels)


def event_rate(run, bin_ms: float | None = None) -> Heatmap | None:
    """Events per second: sweeps x time from the stimulus (a PSTH per sweep), or,
    gap-free, minutes x seconds within the minute. A bin less than half analyzed is blank."""
    res = run.results.get("events")
    rec = getattr(run, "recording", None)
    rows = [r for r in res.tables.get("events", []) if r.get("accepted")] if res is not None else []
    if rec is None or not rows:
        return None
    segs = res.tables.get("analyzed", []) or [{"sweep": k, "start_s": 0.0, "end_s": rec.sweep_duration_s}
                                              for k in range(rec.n_sweeps)]
    if rec.is_gap_free:
        width = 1.0 if bin_ms is None else bin_ms / 1e3
        n_rows = int(np.ceil(rec.duration_s / 60.0))
        x = np.arange(0.0, 60.0, width)

        def place(sweep, t):                        # (row, column start) on the minute clock
            return int(t // 60.0), t - 60.0 * (t // 60.0)
        y, ylab, yu, xlab, zero = np.arange(n_rows) + 1.0, "Minute", "", "Time within the minute", 0.0
    else:
        stim = rec.stimulus.fixed_times()
        zero = float(stim[0]) if stim.size else 0.0
        width = (0.01 if stim.size else rec.sweep_duration_s / 100) if bin_ms is None else bin_ms / 1e3
        x = np.arange(-zero, rec.sweep_duration_s - zero, width)
        n_rows = rec.n_sweeps

        def place(sweep, t):
            return int(sweep), t - zero
        y, ylab, yu = np.arange(n_rows) + 1.0, "Sweep", ""
        xlab = "Time from the stimulus" if stim.size else "Time in the sweep"
    counts = np.zeros((n_rows, x.size))
    covered = np.zeros((n_rows, x.size))
    for sg in segs:                                  # the seconds analyzed in each bin
        a, b = float(sg["start_s"]), float(sg["end_s"])
        t = a
        while t < b - 1e-12:
            r, c0 = place(sg.get("sweep", 0), t)
            j = int(np.floor((c0 - x[0]) / width + 1e-9))
            stop = min(b, t + max(x[0] + (j + 1) * width - c0, 1e-9))
            if 0 <= r < n_rows and 0 <= j < x.size:
                covered[r, j] += stop - t
            t = stop
    for e in rows:
        r, c0 = place(e.get("sweep", 0), e["onset_s"])
        j = int(np.floor((c0 - x[0]) / width + 1e-9))
        if 0 <= r < n_rows and 0 <= j < x.size:
            counts[r, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(covered >= 0.5 * width, counts / covered, np.nan)
    return Heatmap(rate, x + width / 2, y, "Event rate over time", xlab, "s", ylab, yu, "Events per second", "Hz",
                   markers=[Marker("x", 0.0, label="stimulus")] if not rec.is_gap_free and zero else [],
                   notes=[f"bins of {width * 1e3:g} ms" if width < 1 else f"bins of {width:g} s",
                          "blank: less than half of the bin was analyzed"])


def _signal(run, fmax: float):
    """The first sweep (a gap-free recording's whole trace), resampled to about
    2.5 x ``fmax`` (scipy's polyphase filter removes what lies above)."""
    from scipy.signal import resample_poly

    rec = run.recording
    fs = rec.sample_rate_hz
    y = np.asarray(rec.sweep(0), float)
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    ratio = Fraction(min(1.0, 2.5 * fmax / fs)).limit_denominator(200)
    if ratio < 1:
        y = resample_poly(y - y.mean(), ratio.numerator, ratio.denominator)
        fs = fs * ratio.numerator / ratio.denominator
    return y - y.mean(), fs


def spectrogram(run, method: str = "hann", fmax: float = 250.0) -> Heatmap | None:
    """Power over time and frequency. "hann": a short-time Fourier transform, 1 s Hann
    windows overlapping by half; "morlet": complex Morlet wavelets, 3 to 10 cycles
    from the lowest frequency to the highest (Cohen 2014, ch. 12-13)."""
    rec = getattr(run, "recording", None)
    if rec is None or (rec.sweep_duration_s if not rec.is_gap_free else rec.duration_s) < 4.0:
        return None
    fmax = min(fmax, rec.sample_rate_hz / 2.5)
    y, fs = _signal(run, fmax)
    units = f"{rec.signal_units}²/Hz"
    if method == "hann":
        from scipy.signal import spectrogram as stft

        n = int(round(fs))
        f, t, p = stft(y, fs, window="hann", nperseg=n, noverlap=n // 2, scaling="density", mode="psd")
        keep = (f > 0) & (f <= fmax)
        f, p = f[keep], p[keep]
        how = "1 s Hann windows, 50% overlap (1 Hz resolution)"
    else:
        step = max(1.0, round(fmax / 60.0))
        f = np.arange(max(step, 2.0), fmax + 1e-9, step)            # about 60 rows on round frequencies
        cycles = np.geomspace(3.0, 10.0, f.size)
        half = int(3 * cycles[0] / (2 * np.pi * f[0]) * fs)        # the longest wavelet's half length
        nfft = int(2 ** np.ceil(np.log2(y.size + 2 * half + 1)))
        spec = np.fft.fft(y, nfft)
        t_all = np.arange(y.size) / fs
        rows = []
        for fr, nc in zip(f, cycles):
            sd = nc / (2 * np.pi * fr)                       # the Gaussian's width in s
            h = int(3 * sd * fs)
            tw = np.arange(-h, h + 1) / fs
            g = np.exp(-tw ** 2 / (2 * sd ** 2))
            w = np.exp(2j * np.pi * fr * tw) * g / (g.sum() / 2)    # a sine of amplitude A gives |coefficient| A
            c = np.fft.ifft(spec * np.fft.fft(w, nfft))[h:h + y.size]
            t, row = _reduce(t_all, (np.abs(c) ** 2)[None, :], "mean")       # reduced as it goes: an hour fits
            rows.append(row[0])
        p = np.array(rows)
        units = f"{rec.signal_units}²"
        how = f"Morlet wavelets, {f[0]:g} to {f[-1]:g} Hz, 3 to 10 cycles"
    t, p = _reduce(np.asarray(t, float), p, "mean")
    stim = rec.stimulus.fixed_times() if not rec.is_gap_free else np.array([])
    which = "" if rec.is_gap_free else " (sweep 1)"
    return Heatmap(p, t, np.asarray(f, float), f"Time-frequency power{which}", "Time", "s", "Frequency", "Hz",
                   "Power", units, power=True, markers=[Marker("x", float(s), label="stimulus") for s in stim],
                   notes=[how])


def band_power(run, fmax: float = 250.0) -> Heatmap | None:
    """Each band's power over time in dB from its own median, so bands of very
    different power share one scale (diverging about 0 dB)."""
    sp = spectrogram(run, "hann", fmax)
    if sp is None:
        return None
    rows, names = [], []
    for name, lo, hi in BANDS:
        sel = (sp.y >= lo) & (sp.y < hi)
        if sel.any():
            bp = np.nanmean(sp.matrix[sel], axis=0)
            med = np.nanmedian(bp)
            with np.errstate(divide="ignore", invalid="ignore"):
                rows.append(10 * np.log10(bp / med) if med > 0 else np.full(bp.size, NAN))
            names.append(f"{name} {lo:g}-{hi:g} Hz")
    if not rows:
        return None
    return Heatmap(np.array(rows), sp.x, np.arange(len(rows)) + 1.0, "Band power over time", "Time", "s", "Band",
                   "", "Band power", "dB re its median", "diverging", 0.0, row_labels=names,
                   markers=sp.markers, notes=sp.notes)


def pulses(run) -> Heatmap | None:
    """Paired pulses and trains: each pulse over pulse 1 of its own sweep, pulse x
    sweep (diverging about 1); a sweep's interval in its label when they differ."""
    for akey in ("evoked",):
        res = run.results.get(akey)
        rows = res.tables.get("sweeps", []) if res is not None else []
        if not rows or not {"sweep", "pulse", "amplitude"} <= set(rows[0]):
            continue
        sw = sorted({r["sweep"] for r in rows})
        pn = sorted({r["pulse"] for r in rows})
        if len(sw) < 2 or len(pn) < 2:
            continue
        amp = np.full((len(sw), len(pn)), NAN)
        onset = np.full((len(sw), len(pn)), NAN)
        for r in rows:
            amp[sw.index(r["sweep"]), pn.index(r["pulse"])] = r["amplitude"]
            onset[sw.index(r["sweep"]), pn.index(r["pulse"])] = r.get("onset_s", NAN)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = amp / amp[:, :1]
        isi = (onset[:, 1] - onset[:, 0]) * 1e3
        varied = np.isfinite(isi).any() and np.nanmax(isi) - np.nanmin(isi) > 0.5
        labels = [f"{s + 1} ({i:.0f} ms)" if varied and np.isfinite(i) else f"{s + 1}" for s, i in zip(sw, isi)]
        return Heatmap(ratio, np.array(pn, float), np.array(sw, float) + 1.0, "Each pulse over pulse 1, every sweep",
                       "Pulse", "", "Sweep" + (" (interval)" if varied else ""), "", "Amplitude / pulse 1", "ratio",
                       "diverging", 1.0, row_labels=labels)
    return None


def correlation(numbers: dict[str, np.ndarray], q: float = 0.05) -> Heatmap | None:
    """Spearman correlations between measures across recordings (diverging about 0,
    -1 to 1), pairs significant after Benjamini-Hochberg at ``q`` marked; ordered by
    average-linkage clustering on 1 - r."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    names = [k for k, v in numbers.items() if np.isfinite(v).sum() >= 4 and np.nanstd(v) > 0]
    k = len(names)
    if k < 2:
        return None
    r, p, n = np.eye(k), np.full((k, k), NAN), np.zeros((k, k), int)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = numbers[names[i]], numbers[names[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            n[i, j] = n[j, i] = ok.sum()
            if ok.sum() >= 4 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
                rr, pp = spearmanr(a[ok], b[ok])
                r[i, j] = r[j, i] = rr
                p[i, j] = p[j, i] = pp
            else:
                r[i, j] = r[j, i] = NAN
    iu = np.triu_indices(k, 1)
    pv = p[iu]
    sig = np.zeros((k, k), bool)
    ok = np.isfinite(pv)
    if ok.any():                                         # Benjamini-Hochberg step-up
        ranked = np.sort(pv[ok])
        passed = ranked <= q * np.arange(1, ranked.size + 1) / ranked.size
        cut = ranked[np.flatnonzero(passed)[-1]] if passed.any() else -1.0
        hit = ok & (pv <= cut)
        sig[iu[0][hit], iu[1][hit]] = sig[iu[1][hit], iu[0][hit]] = True
    order = np.arange(k)
    if k > 2:
        dist = np.clip(1 - np.nan_to_num(r), 0, None)
        np.fill_diagonal(dist, 0.0)
        order = leaves_list(linkage(squareform((dist + dist.T) / 2, checks=False), method="average"))
    named = [names[i] for i in order]
    return Heatmap(r[np.ix_(order, order)], np.arange(k) + 1.0, np.arange(k) + 1.0, "Correlations between measures",
                   "Measure", "", "Measure", "", "Spearman r", "", "diverging", 0.0, fixed=(-1.0, 1.0),
                   row_labels=named, col_labels=[str(i + 1) for i in range(k)], mask=sig[np.ix_(order, order)],
                   notes=[f"dots: significant after Benjamini-Hochberg (q = {q:g}); "
                          f"{int(n[iu].min())}-{int(n[iu].max())} recordings per pair"])


# --------------------------------------------------------------------------- what a run can show


def catalogue(run) -> list[tuple[str, str, Callable[[], Heatmap | None]]]:
    """(key, title, build) for every map ``run`` qualifies for; nothing is built here."""
    out = []
    res = getattr(run, "results", {}) or {}
    rec = getattr(run, "recording", None)
    kind = getattr(run, "recording_type", "")
    if rec is not None and rec.n_sweeps >= 2 and not rec.is_gap_free:
        out.append(("heatmap.sweeps", "Every sweep vs. time", lambda: sweeps(run)))
    for akey in ("dsi", "dsi_cell"):
        r = res.get(akey)
        if r is not None and r.tables.get("course"):
            out.append((f"heatmap.{akey}.pct", "DSI: % of baseline, episode x response",
                        lambda a=akey: dsi_course(run, a, "pct")))
            out.append((f"heatmap.{akey}.power_pct", "DSI: Fourier power, % of baseline, episode x response",
                        lambda a=akey: dsi_course(run, a, "power_pct")))
        if r is not None and r.tables.get("fourier"):
            out.append((f"heatmap.{akey}.fourier", "DSI: Fourier amplitude, response x frequency",
                        lambda a=akey: dsi_fourier(run, a)))
    ev = res.get("events")
    if ev is not None and any(e.get("accepted") for e in ev.tables.get("events", [])):
        out.append(("heatmap.events.rate", "Event rate over time", lambda: event_rate(run)))
    if rec is not None and kind.endswith("spontaneous"):
        out.append(("heatmap.spectrogram.hann", "Time-frequency power (Hann windows)",
                    lambda: spectrogram(run, "hann")))
        out.append(("heatmap.spectrogram.morlet", "Time-frequency power (Morlet wavelets)",
                    lambda: spectrogram(run, "morlet")))
        out.append(("heatmap.band_power", "Band power over time", lambda: band_power(run)))
    evk = res.get("evoked")
    rows = evk.tables.get("sweeps", []) if evk is not None else []
    if len({r.get("pulse") for r in rows}) > 1 and len({r.get("sweep") for r in rows}) > 1:
        out.append(("heatmap.pulses", "Each pulse over pulse 1, pulse x sweep", lambda: pulses(run)))
    return out
