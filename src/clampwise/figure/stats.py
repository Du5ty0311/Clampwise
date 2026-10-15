"""Statistics at the right level (B23). Comparisons are made on one value per
cell (the prepared points), never on pooled events or sweeps:

- the **effect**: the difference in means with its 95% confidence interval by
  bootstrap over cells, or, when animals are named, the hierarchical bootstrap
  (animals resampled, then cells within them; Saravanan, Berman & Sober 2020),
  and Hedges' g (Ho et al. 2019; Cumming 2014);
- the **tests reviewers ask for**, each with its assumptions: Welch's t and
  Mann-Whitney for independent groups, paired t and Wilcoxon for the same cells
  in two conditions; with more than two groups each is compared with the first
  and the p values are Holm-corrected;
- a **mixed model** (statsmodels, optional) with animal as a random intercept,
  when animals are named and statsmodels is installed.

Each comparison is said in one plain sentence, and ``methods`` writes the
paragraph for a paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

RESAMPLES = 5000


@dataclass
class Comparison:
    reference: str
    other: str
    difference: float
    ci: tuple[float, float]
    hedges_g: float
    test: str
    p: float                    # Holm-corrected when there are more than two groups
    p_raw: float
    other_test: str
    other_p: float
    paired: bool
    n: str
    resampled: str              # what the bootstrap resampled
    mixed: str = ""             # the mixed model's line, when there is one

    def sentence(self, units: str = "") -> str:
        u = f" {units}" if units else ""
        p = f"{self.test} {fmt_p(self.p)}" + (" (Holm-corrected)" if self.p != self.p_raw else "")
        return (f"{self.other} vs. {self.reference}: {'mean paired difference' if self.paired else 'difference'} "
                f"{self.difference:+.3g}{u} (95% CI {self.ci[0]:.3g} to {self.ci[1]:.3g}; Hedges' g "
                f"{self.hedges_g:.2f}); {p}; {self.other_test} {fmt_p(self.other_p)}; {self.n}."
                + (f" {self.mixed}" if self.mixed else ""))


def fmt_p(p: float) -> str:
    """"p < 0.001", "p = 0.012", "p = 0.34"."""
    if not np.isfinite(p):
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def stars(p: float) -> str:
    return "n/a" if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * (b.mean() - a.mean()) / sp) if sp > 0 else np.nan


def _resample_mean(values: np.ndarray, animals: np.ndarray | None, rng, n: int) -> np.ndarray:
    """``n`` bootstrap means of ``values``: cells resampled, or animals then cells within them."""
    if animals is None or len(set(animals)) < 2:
        idx = rng.integers(0, values.size, (n, values.size))
        return values[idx].mean(axis=1)
    groups = [values[animals == a] for a in np.unique(animals)]
    out = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        out[k] = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in (groups[i] for i in pick)]).mean()
    return out


def bootstrap_difference(a, b, animals_a=None, animals_b=None, paired: bool = False, seed: int = 0,
                         n: int = RESAMPLES) -> tuple[float, float]:
    """The 95% percentile interval of mean(b) - mean(a) (paired: of the mean of b - a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        d = b - a
        boots = _resample_mean(d, None if animals_a is None else np.asarray(animals_a), rng, n)
    else:
        boots = _resample_mean(b, None if animals_b is None else np.asarray(animals_b), rng, n) - \
            _resample_mean(a, None if animals_a is None else np.asarray(animals_a), rng, n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (p.size - rank) * p[i]))
        out[i] = running
    return out.tolist()


def _mixed(points: pd.DataFrame, ref: str, other: str) -> str:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return ""
    d = points[points.group.isin([ref, other]) & (points.animal != "")].copy()
    if d.animal.nunique() < 4:
        return ""
    d["is_other"] = (d.group == other).astype(float)
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm("y ~ is_other", d, groups=d["animal"]).fit(reml=True)
        est, (lo, hi), p = fit.params["is_other"], fit.conf_int().loc["is_other"], fit.pvalues["is_other"]
        return (f"Mixed model (animal as a random intercept, statsmodels MixedLM): {est:+.3g} (95% CI {lo:.3g} to "
                f"{hi:.3g}), {fmt_p(p)}.")
    except Exception:
        return "The mixed model could not be fitted (too few animals or no variance between them)."


def compare(points: pd.DataFrame, paired: bool = False, panel: str | None = None) -> list[Comparison]:
    """Each group against the first, on one value per cell (``points``: group, y,
    cell, animal)."""
    if panel is not None:
        points = points[points.panel == panel]
    groups = list(dict.fromkeys(points.group))
    if len(groups) < 2:
        return []
    ref = groups[0]
    out = []
    for other in groups[1:]:
        a_rows = points[points.group == ref]
        b_rows = points[points.group == other]
        named = (points.animal != "").any()
        is_paired = paired and points.cell.ne("").any()
        if is_paired:
            m = a_rows.merge(b_rows, on="cell", suffixes=("_a", "_b"))
            a, b = m["y_a"].to_numpy(float), m["y_b"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            animals = m["animal_a"].to_numpy()[ok] if named else None
            diff = float(np.mean(b - a)) if a.size else np.nan
            ci = bootstrap_difference(a, b, animals, paired=True) if a.size > 1 else (np.nan, np.nan)
            p = sps.ttest_rel(b, a).pvalue if a.size > 1 else np.nan
            p2 = sps.wilcoxon(b, a).pvalue if a.size > 1 and np.any(b != a) else np.nan
            names = ("paired t test (differences roughly normal)", "Wilcoxon signed-rank (no normality assumed)")
            n = f"n = {a.size} cells measured in both" + (f" from {len(set(animals))} animals" if named else "")
        else:
            a, b = a_rows["y"].to_numpy(float), b_rows["y"].to_numpy(float)
            fa, fb = np.isfinite(a), np.isfinite(b)
            aa = a_rows["animal"].to_numpy()[fa] if named else None
            ab = b_rows["animal"].to_numpy()[fb] if named else None
            a, b = a[fa], b[fb]
            diff = float(b.mean() - a.mean()) if a.size and b.size else np.nan
            ci = bootstrap_difference(a, b, aa, ab) if a.size > 1 and b.size > 1 else (np.nan, np.nan)
            p = sps.ttest_ind(b, a, equal_var=False).pvalue if a.size > 1 and b.size > 1 else np.nan
            p2 = sps.mannwhitneyu(b, a).pvalue if a.size and b.size else np.nan
            names = ("Welch's t test (cell means roughly normal, variances may differ)",
                     "Mann-Whitney U (no normality assumed)")
            n = f"n = {a.size} and {b.size} cells" + (
                f" from {len(set(aa))} and {len(set(ab))} animals" if named else "")
        g = hedges_g(a, b)
        resampled = "animals, then cells within them" if named else ("cells" if not is_paired else "cells")
        out.append(Comparison(ref, other, diff, ci, g, names[0].split(" (")[0], float(p), float(p),
                              names[1].split(" (")[0], float(p2), is_paired, n, resampled,
                              _mixed(points, ref, other) if named and not is_paired else ""))
    if len(out) > 1:
        for c, adj in zip(out, holm([c.p_raw for c in out])):
            c.p = adj
    return out


def methods(comparisons: list[Comparison]) -> str:
    """The statistics paragraph for a paper."""
    if not comparisons:
        return ""
    c = comparisons[0]
    level = "animal and cell" if "animals" in c.resampled else "cell"
    text = (f"Comparisons were made on one value per cell (the mean of its measurements). Effects are given as the "
            f"{'mean paired difference' if c.paired else 'difference in means'} with a 95% confidence interval from "
            f"{RESAMPLES} bootstrap resamples of {c.resampled} (percentile interval"
            + ("; hierarchical bootstrap, Saravanan et al. 2020" if level != "cell" else "") +
            ") and as Hedges' g. ")
    text += (f"p values are from {'paired t tests and Wilcoxon signed-rank tests' if c.paired else 'Welch' + chr(39) + 's t tests and Mann-Whitney U tests'} "
             "on the cell values" + (", Holm-corrected for the comparisons with the first group" if len(comparisons) > 1
                                     else "") + ".")
    if any(x.mixed.startswith("Mixed model") for x in comparisons):
        text += " A linear mixed model with animal as a random intercept (statsmodels MixedLM, REML) is reported beside."
    return text


def nested_false_positives(n_sims: int = 400, animals: int = 4, cells: int = 4, events: int = 30,
                           animal_sd: float = 1.0, cell_sd: float = 1.0, event_sd: float = 2.0,
                           seed: int = 1) -> dict[str, float]:
    """B23's teaching example: two groups with no true difference, nested events
    in cells in animals. How often each level of analysis calls the difference
    significant at 0.05: animals (one mean each), cells, and pooled events."""
    rng = np.random.default_rng(seed)
    hits = {"animals": 0, "cells": 0, "pooled events": 0}
    for _ in range(n_sims):
        data = []
        for g in range(2):
            for a in range(animals):
                am = rng.normal(0, animal_sd)
                for c in range(cells):
                    cm = am + rng.normal(0, cell_sd)
                    data.append((g, a, c, cm + rng.normal(0, event_sd, events)))
        for level in hits:
            vals = [[], []]
            for g in range(2):
                mine = [d for d in data if d[0] == g]
                if level == "pooled events":
                    vals[g] = np.concatenate([d[3] for d in mine])
                elif level == "cells":
                    vals[g] = np.array([d[3].mean() for d in mine])
                else:
                    vals[g] = np.array([np.mean([d[3].mean() for d in mine if d[1] == a]) for a in range(animals)])
            hits[level] += sps.ttest_ind(vals[0], vals[1]).pvalue < 0.05
    return {k: v / n_sims for k, v in hits.items()}
