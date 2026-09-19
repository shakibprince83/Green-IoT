from __future__ import annotations

import numpy as np
from scipy import stats


def corrections(pvalues: list[float]) -> dict[str, np.ndarray]:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    bonf = np.minimum(1.0, p * m)
    order = np.argsort(p)
    holm_sorted = np.maximum.accumulate(np.minimum(1.0, p[order] * (m - np.arange(m))))
    bh_sorted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    holm = np.empty(m)
    holm[order] = holm_sorted
    bh = np.empty(m)
    bh[order] = np.minimum(1.0, bh_sorted)
    return {"bonferroni": bonf, "holm": holm, "benjamini_hochberg": bh}


def paired_analysis(
    a: np.ndarray, b: np.ndarray, bootstrap_samples: int = 10000, seed: int = 2026
) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("Paired samples must be equal-length one-dimensional arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(diff, size=(bootstrap_samples, len(diff)), replace=True), axis=1)
    shapiro_p = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else None
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(diff)
        wilcoxon_p = float(w.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    sd = float(np.std(diff, ddof=1))
    return {
        "n": len(diff),
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "sd_difference": sd,
        "bootstrap_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "shapiro_p": shapiro_p,
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "cohens_dz": float(np.mean(diff) / sd) if sd else 0.0,
    }


def require_complete_pairs(
    rows: list[dict], keys: tuple[str, ...] = ("dataset", "architecture", "seed")
) -> None:
    groups: dict[tuple, set[str]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, set()).add(row["framework"])
    missing = [key for key, frameworks in groups.items() if frameworks != {"pytorch", "tensorflow"}]
    if missing:
        raise ValueError(f"Missing framework partner(s): {missing}")
