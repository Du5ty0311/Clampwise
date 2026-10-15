"""Curve fits, in one place (B21, item 3).

Every fit in the app goes through ``nonlinear`` (or one of the named models
below, which call it), so each returns the same things: the parameters with
their 95% confidence intervals, the residuals, r2, AIC and AICc, the window
fitted, and flags when the fit failed or a parameter sits on a bound. A fit
never raises on bad data: it returns NaNs with ``ok=False`` and the calling
analysis decides whether to report, flag or drop it. Every fit can be drawn
over its data with ``curve``.

Confidence intervals are the asymptotic ones from the covariance matrix of
the least-squares solution (parameter +/- t(0.975, n - k) x standard error),
as Motulsky & Christopoulos (2004, *Fitting Models to Biological Data*,
ch. "Confidence intervals of fit parameters") describe them, with their
caution: they are approximate, symmetric by construction, and too narrow
when the parameters are strongly correlated or the model is wrong. Model
comparison (one against two exponentials) uses the extra-sum-of-squares F
test or AICc from the same book.

The models are the standard ones of the field: single and double
exponentials (decays, membrane charging), the exponential rise, the
four-parameter logistic (Hill) for input-output curves, Boltzmann for
conductance-voltage curves (Hille 2001), a Gaussian for histograms, the
synaptic waveform (Clements & Bekkers 1997), straight lines, and the
recovery forms used by paired-pulse and DSI analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import curve_fit

from . import ledger

NAN = float("nan")

# What each model is called in the ledger and in Results.
MODEL_NAMES = {
    "exp1": "single-exponential fit", "exp2": "two-exponential fit", "rise1": "exponential-rise fit",
    "logistic": "four-parameter logistic (Hill) fit", "recovery": "recovery fit, 1 - exp(-t/tau)",
    "suppression": "suppression-recovery fit, 1 - S exp(-t/tau)", "boltzmann": "Boltzmann fit",
    "gaussian": "Gaussian fit", "synaptic": "synaptic-waveform fit", "linear": "straight-line fit",
    "delayed exponential": "delayed-exponential onset fit (100% until t0, then 100 exp(-(t - t0)/tau))",
    "onset and recovery": "onset-and-recovery fit (100% until t0, then 100 (1 - S (1 - exp(-(t - t0)/tau_on)) "
                          "exp(-(t - end)/tau_rec)))",
}


@dataclass
class Fit:
    params: dict[str, float] = field(default_factory=dict)
    r2: float = NAN
    aic: float = NAN
    n: int = 0
    ok: bool = False
    model: str = ""
    # Added with the shared tools; a fit built the old positional way has them empty.
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)      # 95% confidence intervals
    se: dict[str, float] = field(default_factory=dict)                    # standard errors
    residuals: np.ndarray | None = None
    window: tuple[float, float] = (NAN, NAN)                              # x range fitted
    flags: list[str] = field(default_factory=list)
    aicc: float = NAN
    predict: Callable | None = None                                       # x -> fitted y

    def __getitem__(self, key: str) -> float:
        return self.params.get(key, NAN)

    def curve(self, x) -> np.ndarray:
        """The fitted model at ``x``, for drawing over the data."""
        if self.predict is None or not self.ok:
            return np.full(np.asarray(x, dtype=float).shape, NAN)
        return np.asarray(self.predict(np.asarray(x, dtype=float)), dtype=float)

    def describe(self) -> str:
        """One line: the model, its parameters with intervals, r2 and flags."""
        if not self.ok:
            return f"{self.model}: fit failed" + (f" ({'; '.join(self.flags)})" if self.flags else "")
        parts = []
        for k, v in self.params.items():
            if isinstance(v, bool):
                continue
            lo, hi = self.ci.get(k, (NAN, NAN))
            parts.append(f"{k} {v:.4g} [{lo:.4g}, {hi:.4g}]" if np.isfinite(lo) else f"{k} {v:.4g}")
        text = f"{self.model}: " + ", ".join(parts) + (f", r2 {self.r2:.3f}" if np.isfinite(self.r2) else "")
        return text + (f" ({'; '.join(self.flags)})" if self.flags else "")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return NAN
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot


def _aic(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = y.size
    rss = float(np.sum((y - pred) ** 2))
    if n <= k or rss <= 0:
        return NAN
    return n * np.log(rss / n) + 2 * k


def _aicc(aic: float, n: int, k: int) -> float:
    """AIC corrected for small samples (Motulsky & Christopoulos ch. 22)."""
    if not np.isfinite(aic) or n - k - 1 <= 0:
        return NAN
    return aic + 2 * k * (k + 1) / (n - k - 1)


def _clean(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _intervals(p: np.ndarray, cov, n: int, names: Sequence[str]) -> tuple[dict, dict]:
    """95% confidence intervals from the covariance matrix: p +/- t x SE."""
    from scipy.stats import t as student_t

    k = len(p)
    se, ci = {}, {}
    if cov is None or n - k <= 0:
        return se, ci
    diag = np.diag(np.asarray(cov, dtype=float))
    tcrit = float(student_t.ppf(0.975, n - k))
    for name, value, var in zip(names, p, diag):
        s = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else NAN
        se[name] = s
        ci[name] = (float(value - tcrit * s), float(value + tcrit * s)) if np.isfinite(s) else (NAN, NAN)
    return se, ci


def _at_bound(p: np.ndarray, bounds, names: Sequence[str], rel: float = 1e-3) -> list[str]:
    """Which parameters sit on a bound of the fit (a value there is not a measurement)."""
    if bounds is None:
        return []
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    out = []
    for name, value, a, b in zip(names, p, np.broadcast_to(lo, len(p)), np.broadcast_to(hi, len(p))):
        span = abs(b - a) if np.isfinite(a) and np.isfinite(b) else max(abs(value), 1.0)
        tol = rel * span
        if np.isfinite(a) and value - a <= tol or np.isfinite(b) and b - value <= tol:
            out.append(f"{name} at the fit's limit")
    return out


def nonlinear(model: Callable, x, y, p0, names: Sequence[str], *, label: str, bounds=None,
              jac=None, maxfev: int | None = None, extra: dict | None = None,
              clean: bool = True, purpose: str = "") -> Fit:
    """The one least-squares fit: ``model(x, *params)`` to ``y`` from ``p0``.

    Wraps ``scipy.optimize.curve_fit`` with exactly the arguments given (so a
    caller that passes its old starting values and bounds gets its old
    answer), and returns everything a result needs. ``extra`` adds derived
    parameters to ``params``. A failed fit returns ``ok=False`` with the
    reason in ``flags``.
    """
    if clean:
        x, y = _clean(x, y)
    else:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    names = list(names)
    k = len(names)
    window = (float(x[0]), float(x[-1])) if x.size else (NAN, NAN)
    kwargs = {}
    if bounds is not None:
        kwargs["bounds"] = bounds
    if jac is not None:
        kwargs["jac"] = jac
    if maxfev is not None:
        kwargs["maxfev"] = maxfev
    try:
        p, cov = curve_fit(model, x, y, p0=p0, **kwargs)
    except (RuntimeError, ValueError, TypeError) as exc:
        return Fit(model=label, n=int(x.size), window=window,
                   flags=[f"fit failed: {str(exc).splitlines()[0][:80]}"])
    p = np.asarray(p, dtype=float)
    pred = np.asarray(model(x, *p), dtype=float)
    se, ci = _intervals(p, cov, int(x.size), names)
    aic = _aic(y, pred, k)
    params = {name: float(v) for name, v in zip(names, p)}
    if extra:
        params.update(extra(params) if callable(extra) else extra)
    fit = Fit(params, _r2(y, pred), aic, int(x.size), True, label, ci, se, y - pred, window,
              _at_bound(p, bounds, names), _aicc(aic, int(x.size), k),
              predict=lambda xx, _p=tuple(p): model(xx, *_p))
    ledger.note(MODEL_NAMES.get(label, label), purpose)
    return fit


# ------------------------------------------------------------------ the models


def linear(x, y, purpose: str = "") -> Fit:
    """y = slope * x + intercept."""
    x, y = _clean(x, y)
    if x.size < 2 or np.ptp(x) == 0:
        return Fit(model="linear", n=int(x.size))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    fit = Fit({"slope": float(slope), "intercept": float(intercept)}, _r2(y, pred),
              _aic(y, pred, 2), int(x.size), True, "linear",
              residuals=y - pred, window=(float(x.min()), float(x.max())),
              predict=lambda xx, s=float(slope), i=float(intercept): s * xx + i)
    # the line's own uncertainty, from the residual scatter (n - 2 degrees of freedom)
    if x.size > 2:
        resid = y - pred
        s2 = float(np.sum(resid ** 2)) / (x.size - 2)
        sxx = float(np.sum((x - x.mean()) ** 2))
        cov = np.array([[s2 / sxx, -s2 * x.mean() / sxx],
                        [-s2 * x.mean() / sxx, s2 * (1 / x.size + x.mean() ** 2 / sxx)]])
        fit.se, fit.ci = _intervals(np.array([slope, intercept]), cov, int(x.size), ("slope", "intercept"))
    fit.aicc = _aicc(fit.aic, int(x.size), 2)
    ledger.note(MODEL_NAMES["linear"], purpose)
    return fit


def exp_decay(t, y, offset: float | None = None, purpose: str = "") -> Fit:
    """y = amp * exp(-t / tau) + offset, t from the first point.

    With ``offset`` given it is held fixed, for a decay already measured
    against its own asymptote over a window too short to find that asymptote.
    """
    t, y = _clean(t, y)
    if offset is not None:
        if t.size < 3:
            return Fit(model="exp1", n=int(t.size))
        t = t - t[0]

        def fixed(tt, amp, tau):
            return amp * np.exp(-tt / tau) + offset

        def fixed_jac(tt, amp, tau):
            e = np.exp(-tt / tau)
            return np.column_stack([e, amp * e * tt / tau ** 2])
        amp0 = y[0] - offset
        fit = nonlinear(fixed, t, y, (amp0, max(t[-1] / 1.5, 1e-6)), ("amp", "tau"), label="exp1",
                        jac=fixed_jac, bounds=([-np.inf, 1e-7], [np.inf, 100 * max(t[-1], 1e-3)]),
                        maxfev=5000, clean=False, purpose=purpose)
        if fit.ok:
            fit.params["offset"] = float(offset)
        return fit
    if t.size < 5:
        return Fit(model="exp1", n=int(t.size))
    t = t - t[0]
    amp0 = y[0] - y[-1]
    target = y[-1] + amp0 / np.e
    below = np.flatnonzero(np.sign(amp0) * (y - target) <= 0)
    tau0 = t[below[0]] if below.size and t[below[0]] > 0 else (t[-1] / 3 or 1e-3)

    def model(tt, amp, tau, offset):
        return amp * np.exp(-tt / tau) + offset

    def jac(tt, amp, tau, offset):
        e = np.exp(-tt / tau)
        return np.column_stack([e, amp * e * tt / tau ** 2, np.ones_like(tt)])

    return nonlinear(model, t, y, (amp0, tau0, y[-1]), ("amp", "tau", "offset"), label="exp1", jac=jac,
                     bounds=([-np.inf, 1e-6, -np.inf], [np.inf, 10 * max(t[-1], 1e-3), np.inf]),
                     maxfev=5000, clean=False, purpose=purpose)


PROFILE_GRID = 48     # log-spaced taus tried before refining (1 us to 10x the window)


def exp_decay_profiled(t, y, purpose: str = "") -> Fit:
    """The same fit as ``exp_decay`` (free offset, tau between 1 us and ten times
    the window), solved by variable projection: for any tau the amplitude and
    offset are a straight-line least-squares answer, so only tau is searched,
    over the whole allowed range: a log-spaced grid finds the deepest minimum and
    a bounded Brent search refines it.

    For spontaneous events, which fit thousands of decays per recording: the
    grid finds the deepest minimum where a gradient fit can stop in a local one
    (a noise spike at the start makes a fast false decay), and it is several
    times faster.
    """
    t, y = _clean(t, y)
    if t.size < 5:
        return Fit(model="exp1", n=int(t.size))
    t = t - t[0]
    ym = y.mean()
    yc = y - ym

    def solve(log_tau):
        e = np.exp(-t / np.exp(log_tau))
        ec = e - e.mean()
        d = float(ec @ ec)
        amp = float(ec @ yc) / d if d > 0 else 0.0
        offset = ym - amp * float(e.mean())
        r = y - amp * e - offset
        return float(r @ r), amp, offset

    from scipy.optimize import minimize_scalar

    # The residual can have more than one minimum in tau (a noise spike at the
    # start makes a fast one), and a one-dimensional search stops in whichever
    # it meets first. A grid over the whole range finds the deepest; the search
    # then only refines it.
    lo, hi = np.log(1e-6), np.log(10 * max(t[-1], 1e-3))
    grid = np.linspace(lo, hi, PROFILE_GRID)
    e = np.exp(-t[None, :] / np.exp(grid)[:, None])          # every grid tau at once
    ec = e - e.mean(axis=1, keepdims=True)
    d = np.einsum("ij,ij->i", ec, ec)
    amp = np.divide(ec @ yc, d, out=np.zeros_like(d), where=d > 0)
    resid = yc[None, :] - amp[:, None] * ec
    best = int(np.argmin(np.einsum("ij,ij->i", resid, resid)))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    res = minimize_scalar(lambda v: solve(v)[0], bounds=(a, b), method="bounded",
                          options={"xatol": 1e-10})
    if not np.isfinite(res.x):
        return Fit(model="exp1", n=int(t.size))
    tau = float(np.exp(res.x))
    _, amp, offset = solve(res.x)
    pred = amp * np.exp(-t / tau) + offset
    aic = _aic(y, pred, 3)
    ledger.note(MODEL_NAMES["exp1"] + " by variable projection", purpose)
    return Fit({"amp": amp, "tau": tau, "offset": offset}, _r2(y, pred), aic,
               int(t.size), True, "exp1", residuals=y - pred, window=(float(t[0]), float(t[-1])),
               flags=["tau at the fit's limit"] if tau <= 1.01e-6 or tau >= 0.99 * np.exp(hi) else [],
               aicc=_aicc(aic, int(t.size), 3),
               predict=lambda tt, a=amp, k=tau, o=offset: a * np.exp(-tt / k) + o)


def biexp_decay(t, y, tau_mono: float | None = None, purpose: str = "") -> Fit:
    """y = a1 exp(-t/tau1) + a2 exp(-t/tau2) + offset, tau1 < tau2.

    Seeded from the single-exponential tau (a third and three times it). The
    weighted tau, (a1 tau1 + a2 tau2) / (a1 + a2), is in ``params['tau_w']``
    and the fast component's share of the amplitude in ``params['fast_fraction']``.
    ``params['slow_beyond_window']`` is true when the slow tau reaches ten
    times the fitted stretch: a component that slow cannot be told from the
    offset over that stretch, so it is not a decay.
    """
    t, y = _clean(t, y)
    if t.size < 8:
        return Fit(model="exp2", n=int(t.size))
    t = t - t[0]
    mono = exp_decay(t, y, purpose=purpose)
    tau = tau_mono if tau_mono and np.isfinite(tau_mono) else mono["tau"]
    if not np.isfinite(tau) or tau <= 0:
        return Fit(model="exp2", n=int(t.size))
    amp = y[0] - y[-1]

    def model(tt, a1, tau1, a2, tau2, offset):
        return a1 * np.exp(-tt / tau1) + a2 * np.exp(-tt / tau2) + offset

    lo = [-np.inf, 1e-6, -np.inf, 1e-6, -np.inf]
    hi = [np.inf, 10 * max(t[-1], 1e-3), np.inf, 20 * max(t[-1], 1e-3), np.inf]
    raw = nonlinear(model, t, y, (amp * 0.6, tau / 3, amp * 0.4, tau * 3, y[-1]),
                    ("a1", "tau1", "a2", "tau2", "offset"), label="exp2", bounds=(lo, hi),
                    maxfev=20000, clean=False, purpose=purpose)
    if not raw.ok:
        return Fit(model="exp2", n=int(t.size), flags=raw.flags)
    a1, t1, a2, t2, off = (raw.params[k] for k in ("a1", "tau1", "a2", "tau2", "offset"))
    ci, se = dict(raw.ci), dict(raw.se)
    if t1 > t2:
        a1, t1, a2, t2 = a2, t2, a1, t1
        ci = {"a1": ci.get("a2"), "tau1": ci.get("tau2"), "a2": ci.get("a1"), "tau2": ci.get("tau1"),
              "offset": ci.get("offset")}
        se = {"a1": se.get("a2"), "tau1": se.get("tau2"), "a2": se.get("a1"), "tau2": se.get("tau1"),
              "offset": se.get("offset")}
    total = a1 + a2
    pred = model(t, a1, t1, a2, t2, off)
    params = {"a1": a1, "tau1": t1, "a2": a2, "tau2": t2, "offset": off,
              "tau_w": (a1 * t1 + a2 * t2) / total if total else NAN,
              "fast_fraction": a1 / total if total else NAN,
              "slow_beyond_window": bool(t2 >= 0.99 * 10 * max(t[-1], 1e-3))}
    same_sign = np.sign(a1) == np.sign(a2) and a1 != 0
    aic = _aic(y, pred, 5)
    return Fit(params, _r2(y, pred), aic, int(t.size), bool(same_sign), "exp2",
               {k: v for k, v in ci.items() if v is not None}, {k: v for k, v in se.items() if v is not None},
               y - pred, raw.window, raw.flags + ([] if same_sign else ["components of opposite sign"]),
               _aicc(aic, int(t.size), 5),
               predict=lambda tt, p=(a1, t1, a2, t2, off): model(tt, *p))


def choose_exponentials(t, y, criterion: str = "F test", alpha: float = 0.05) -> tuple[Fit, Fit, str, str]:
    """One or two exponentials? Fits both and chooses by the extra-sum-of-squares
    F test (p below ``alpha`` prefers two) or by AICc (the lower wins), after
    Motulsky & Christopoulos (chs. 21-22). Returns (mono, bi, chosen, reason);
    two exponentials are never preferred when their slow component runs past
    the window or their amplitudes have opposite signs."""
    mono = exp_decay(t, y)
    bi = biexp_decay(t, y, mono["tau"] if mono.ok else None)
    if not mono.ok:
        return mono, bi, "exp2" if bi.ok else "none", "the single exponential could not be fitted"
    if not bi.ok or bi.params.get("slow_beyond_window"):
        why = "two exponentials could not be fitted" if not bi.ok else \
            "the slow component runs past the window: it is an offset, not a decay"
        return mono, bi, "exp1", why
    rss1 = float(np.sum(mono.residuals ** 2))
    rss2 = float(np.sum(bi.residuals ** 2))
    n = bi.n
    if criterion.lower().startswith("aic"):
        better = bi.aicc < mono.aicc
        reason = f"AICc {bi.aicc:.1f} against {mono.aicc:.1f}"
        return mono, bi, "exp2" if better else "exp1", reason
    from scipy.stats import f as f_dist

    df1, df2 = 5 - 3, n - 5
    if df2 <= 0 or rss2 <= 0 or rss1 <= rss2 * (1 + 1e-12):
        return mono, bi, "exp1", "two exponentials fit no better"
    F = ((rss1 - rss2) / df1) / (rss2 / df2)
    p = float(f_dist.sf(F, df1, df2))
    return mono, bi, ("exp2" if p < alpha else "exp1"), f"F test p = {p:.3g}"


def exp_rise(t, y, purpose: str = "") -> Fit:
    """y = amp * (1 - exp(-t / tau)) + offset, t from the first point."""
    t, y = _clean(t, y)
    if t.size < 5:
        return Fit(model="rise1", n=int(t.size))
    t = t - t[0]
    amp0 = y[-1] - y[0]

    def model(tt, amp, tau, offset):
        return amp * (1 - np.exp(-tt / tau)) + offset

    return nonlinear(model, t, y, (amp0, max(t[-1] / 3, 1e-5), y[0]), ("amp", "tau", "offset"), label="rise1",
                     bounds=([-np.inf, 1e-7, -np.inf], [np.inf, 10 * max(t[-1], 1e-3), np.inf]),
                     maxfev=5000, clean=False, purpose=purpose)


UNSATURATED = ("extrapolated: the strongest stimulus is under 1.5 x this half-maximum, so the curve "
               "has not saturated; test stronger stimuli")


def unsaturated_flag(fit: "Fit") -> str:
    """The flag for a logistic fit whose half-maximum is extrapolated, else ''."""
    return UNSATURATED if fit.ok and fit.params.get("unsaturated") else ""


def logistic(x, y, purpose: str = "") -> Fit:
    """Four-parameter logistic (Hill): bottom + (top - bottom) / (1 + (ec50 / x)^n).

    For input-output curves against a positive stimulus strength. ``ec50`` is
    the half-maximal strength in the units of x.
    """
    x, y = _clean(x, y)
    keep = x > 0
    x, y = x[keep], y[keep]
    if x.size < 4 or np.unique(x).size < 3:
        return Fit(model="logistic", n=int(x.size))
    sign = 1.0 if (y[np.argmax(x)] - y[np.argmin(x)]) >= 0 else -1.0

    def model(xx, bottom, top, ec50, n):
        return bottom + (top - bottom) / (1.0 + (ec50 / xx) ** n)

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    top0, bottom0 = (ys.max(), ys.min()) if sign > 0 else (ys.min(), ys.max())
    half = bottom0 + (top0 - bottom0) / 2
    crossing = np.flatnonzero(sign * (ys - half) >= 0)
    ec0 = xs[crossing[0]] if crossing.size else float(np.median(xs))
    span = abs(top0 - bottom0) or 1.0
    bounds = ([bottom0 - 2 * span if sign > 0 else -np.inf, -np.inf, xs[0] / 20, 0.2],
              [np.inf, np.inf, xs[-1] * 20, 20.0])
    raw = nonlinear(model, x, y, (bottom0, top0, ec0, 2.0), ("bottom", "top", "ec50", "n"), label="logistic",
                    bounds=bounds, maxfev=20000, clean=False, purpose=purpose)
    if not raw.ok:
        return Fit(model="logistic", n=int(x.size), flags=raw.flags)
    p = [raw.params[k] for k in ("bottom", "top", "ec50", "n")]
    # An EC50 pinned at a bound means the curve never saturated (or never
    # rose) inside the tested range: there is no half-maximum to report.
    lo, hi = xs[0] / 20, xs[-1] * 20
    saturated = lo * 1.05 < p[2] < hi / 1.05 and xs[0] / 2 <= p[2] <= xs[-1] * 2
    # When the strongest stimulus is under 1.5 x the EC50 (the curve at 60-69% of
    # its maximum for n 1-2) the half-maximum is extrapolated: a curve tested to
    # 3 V with a true EC50 of 4 V read 2.2-3.1 V (audit phase 2, 2026-09-29).
    # Kept and flagged - flags are evidence, never an edit.
    reaches = p[2] * 1.5 <= xs[-1]
    # A curve that does not rise (r2 low) has no half-maximum either.
    r2 = raw.r2
    good = bool(saturated and np.isfinite(r2) and r2 >= 0.8)
    raw.params["unsaturated"] = bool(not reaches)
    raw.ok = good
    if not good:
        raw.flags = raw.flags + ["no half-maximum inside the tested range"]
    return raw


def hill(x, y, purpose: str = "") -> Fit:
    """The Hill equation is the four-parameter logistic above; one name for both."""
    return logistic(x, y, purpose=purpose)


def recovery(t, r, purpose: str = "") -> Fit:
    """r = 1 - exp(-t / tau): recovery from depression, 0 at t = 0, 1 when recovered."""
    t, r = _clean(t, r)
    if t.size < 2:
        return Fit(model="recovery", n=int(t.size))

    def model(tt, tau):
        return 1 - np.exp(-tt / tau)

    return nonlinear(model, t, r, (max(np.median(t), 1e-3),), ("tau",), label="recovery",
                     bounds=([1e-6], [1e4]), clean=False, purpose=purpose)


def suppression_recovery(t, n, purpose: str = "") -> Fit:
    """n = 1 - S exp(-t / tau): a suppressed, normalized response recovering to 1."""
    t, n = _clean(t, n)
    if t.size < 3:
        return Fit(model="suppression", n=int(t.size))

    def model(tt, s, tau):
        return 1 - s * np.exp(-tt / tau)

    s0 = float(np.clip(1 - n[0], 0.05, 1.5))
    return nonlinear(model, t, n, (s0, max(float(np.median(t)), 1.0)), ("s", "tau"), label="suppression",
                     bounds=([0.0, 1e-3], [2.0, 1e4]), maxfev=5000, clean=False, purpose=purpose)


def _delayed_exp(t, t0, tau):
    t = np.asarray(t, float)
    return np.where(t < t0, 100.0, 100.0 * np.exp(-(t - t0) / tau))


def delayed_exponential(t, pct, purpose: str = "") -> Fit:
    """A DSI onset on the scale normalized between control (100) and the
    minimum (0): 100 until t0, then 100 exp(-(t - t0) / tau) (Heinbockel et
    al. 2005's exponential extrapolated to the control level gives t0 = t_DSI).
    The model has a kink at t0, so the fit starts from the best of a grid of
    starts; t0 cannot precede the stimulus (bounded at 0). Not fitted when
    the data never fall below 70%."""
    t, pct = _clean(t, pct)
    if t.size < 4 or float(np.min(pct)) > 70.0:
        return Fit(model="delayed exponential", n=int(t.size),
                   flags=["less than 30% suppression: no onset to fit"] if t.size >= 4 else [])
    best = min(((float(np.sum((pct - _delayed_exp(t, t0, tau)) ** 2)), t0, tau)
                for t0 in np.linspace(0.0, max(0.05, float(t.max()) - 0.4), 40) for tau in (0.1, 0.2, 0.4, 0.8, 1.6)))
    fit = nonlinear(_delayed_exp, t, pct, best[1:], ("t0", "tau"), label="delayed exponential",
                    bounds=([0.0, 0.01], [float(t.max()), 20.0]), maxfev=5000, clean=False, purpose=purpose)
    if not fit.ok:                       # the grid's best stands, said as such
        pred = _delayed_exp(t, *best[1:])
        fit = Fit({"t0": best[1], "tau": best[2]}, _r2(pct, pred), NAN, int(t.size), True, "delayed exponential",
                  flags=["the least-squares refinement failed; the best of the starting grid is given"],
                  predict=lambda tt, a=best[1], b=best[2]: _delayed_exp(tt, a, b))
    return fit


def onset_recovery(t, pct, stimulus_s: float, purpose: str = "") -> Fit:
    """A DSI time course as % of control from the stimulus's onset (it lasts
    ``stimulus_s``): 100 until t0, then 100 (1 - S (1 - exp(-(t - t0)/tau_on))
    exp(-(t - stimulus_s)/tau_rec)), the recovery counted from the stimulus's
    end. Onset and recovery together, so the onset needs no minimum read 2-3 s
    after the stimulus (which recovery has already begun to lift). From the
    best of a grid of starts (a kink at t0)."""
    t, pct = _clean(t, pct)
    d = float(stimulus_s)

    def model(tt, t0, tau_on, sup, tau_rec):
        tt = np.asarray(tt, float)
        grow = np.where(tt > t0, 1.0 - np.exp(-(tt - t0) / tau_on), 0.0)
        decay = np.where(tt > d, np.exp(-(tt - d) / tau_rec), 1.0)
        return 100.0 * (1.0 - sup * grow * decay)

    if t.size < 6 or float(np.min(pct)) > 80.0:
        return Fit(model="onset and recovery", n=int(t.size))
    sup0 = float(np.clip(1 - np.min(pct) / 100, 0.05, 1.2))
    best = min(((float(np.sum((pct - model(t, t0, ton, sup0, trec)) ** 2)), t0, ton, sup0, trec)
                for t0 in np.linspace(0.0, min(float(t.max()), d + 3.0), 25) for ton in (0.1, 0.3, 0.8)
                for trec in (2.0, 6.0, 20.0)))
    return nonlinear(model, t, pct, best[1:], ("t0", "tau_on", "s", "tau_rec"), label="onset and recovery",
                     bounds=([0.0, 0.02, 0.0, 0.3], [float(t.max()), 5.0, 1.5, 300.0]), maxfev=20000, clean=False,
                     purpose=purpose)


def boltzmann(v, y, falling: bool = False, purpose: str = "") -> Fit:
    """y = top / (1 + exp((V1/2 - V) / k)) (activation), or
    top / (1 + exp((V - V1/2) / k)) (inactivation, ``falling``): the
    conductance-voltage relation of a voltage-gated channel (Hille 2001,
    *Ion Channels of Excitable Membranes*, ch. 1 and 2, the Boltzmann
    distribution of gating charge)."""
    v, y = _clean(v, y)
    if v.size < 4 or np.ptp(y) <= 0:
        return Fit(model="boltzmann", n=int(v.size))
    sign = -1.0 if falling else 1.0

    def model(x, vh, k, top):
        return top / (1.0 + np.exp(sign * (vh - x) / k))

    top0 = float(np.max(y))
    vh0 = float(v[np.argmin(np.abs(y - top0 / 2))])
    return nonlinear(model, v, y, (vh0, 5.0, top0), ("v_half", "slope", "top"), label="boltzmann",
                     bounds=([v.min() - 50, 0.5, 0], [v.max() + 50, 50, 10 * top0 + 1e-12]),
                     maxfev=20000, clean=False, purpose=purpose)


def gaussian(x, c, p0: tuple[float, float, float], maxfev: int = 10000, purpose: str = "") -> Fit:
    """c = amp exp(-(x - mu)^2 / 2 sd^2): one Gaussian through a histogram
    (the all-points histogram's event-free side, for the tonic current)."""

    def model(xx, mu, sd, amp):
        return amp * np.exp(-0.5 * ((xx - mu) / sd) ** 2)

    return nonlinear(model, x, c, p0, ("mu", "sd", "amp"), label="gaussian", maxfev=maxfev, purpose=purpose)


def synaptic_waveform(t, y, purpose: str = "", rise0: float | None = None, decay0: float | None = None,
                      rise_max: float | None = None) -> Fit:
    """y = amp (1 - exp(-t/tau_rise)) exp(-t/tau_decay) + offset, t from the
    onset: the difference-of-exponentials shape of a synaptic current, the
    template shape event detection uses (Clements & Bekkers 1997). Starting
    taus and a ceiling on the rise (s) keep a short, noisy event from trading
    its rise for its decay."""
    t, y = _clean(t, y)
    if t.size < 8:
        return Fit(model="synaptic", n=int(t.size))
    t = t - t[0]
    dev = y - y[0]
    peak = int(np.argmax(np.abs(dev)))
    amp0 = float(dev[peak]) if peak else float(dev[-1])
    if rise0 is None or not np.isfinite(rise0) or rise0 <= 0:
        rise0 = max(float(t[peak]) / 3, 1e-5) if peak else max(t[-1] / 20, 1e-5)
    if decay0 is None or not np.isfinite(decay0) or decay0 <= rise0:
        decay0 = max((t[-1] - t[peak]) / 3, rise0 * 2)
    top_rise = rise_max if rise_max is not None and rise_max > 0 else 10 * max(t[-1], 1e-3)
    rise0 = min(rise0, 0.9 * top_rise)

    def model(tt, amp, tau_rise, tau_decay, offset):
        return amp * (1 - np.exp(-tt / tau_rise)) * np.exp(-tt / tau_decay) + offset

    return nonlinear(model, t, y, (amp0 * 2, rise0, decay0, y[0]), ("amp", "tau_rise", "tau_decay", "offset"),
                     label="synaptic", bounds=([-np.inf, 1e-6, 1e-6, -np.inf],
                                               [np.inf, top_rise, 10 * max(t[-1], 1e-3), np.inf]),
                     maxfev=20000, clean=False, purpose=purpose)


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n <= 0:
        return NAN, NAN
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
