from __future__ import annotations

import numpy as np


def symmetric_difference(a: float, b: float) -> float:
    denominator = (abs(a) + abs(b)) / 2
    return 0.0 if denominator == 0 else 100.0 * (a - b) / denominator


def green_efficiency(rows: list[dict[str, float]], weights: dict[str, float]) -> np.ndarray:
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("GES weights must sum to one")
    benefit = {"macro_f1", "throughput"}
    result = np.zeros(len(rows))
    for metric, weight in weights.items():
        values = np.array([row[metric] for row in rows], dtype=float)
        lo, hi = values.min(), values.max()
        norm = np.ones_like(values) if hi == lo else (values - lo) / (hi - lo)
        if metric not in benefit:
            norm = 1 - norm
        result += weight * norm
    return result


def pareto_front(points: np.ndarray, maximize: tuple[bool, ...]) -> np.ndarray:
    points = np.asarray(points, float)
    adjusted = points * np.array([1 if x else -1 for x in maximize])
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(adjusted):
        dominated = np.any(np.all(adjusted >= point, axis=1) & np.any(adjusted > point, axis=1))
        keep[i] = not dominated
    return keep
