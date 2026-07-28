"""Threshold robustness and final-plateau stability analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .constants import ANALYSIS_METRICS, METRIC_DIRECTIONS
from .io import validate_scores


def convergence_onset(
    thresholds: np.ndarray,
    values: np.ndarray,
    *,
    tolerance_fraction: float = 0.05,
    smooth_sigma: float | None = 1.0,
) -> dict[str, float | bool]:
    """Find the earliest threshold in the contiguous plateau ending at the final value."""
    order = np.argsort(thresholds)
    x = np.asarray(thresholds, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 or x[-1] == x[0]:
        return {"stability_onset": float("nan"), "stable_fraction": 0.0, "has_stable_region": False}
    smoothed = gaussian_filter1d(y, smooth_sigma) if smooth_sigma and len(y) >= 3 else y
    dynamic_range = float(np.ptp(smoothed))
    tolerance = tolerance_fraction * dynamic_range
    stable = np.abs(smoothed - smoothed[-1]) <= tolerance + np.finfo(float).eps
    onset_index = len(x) - 1
    while onset_index > 0 and stable[onset_index - 1]:
        onset_index -= 1
    has_region = onset_index < len(x) - 1
    stable_fraction = float((x[-1] - x[onset_index]) / (x[-1] - x[0])) if has_region else 0.0
    return {
        "stability_onset": float(x[onset_index]),
        "stable_fraction": stable_fraction,
        "has_stable_region": has_region,
    }


def summarize_threshold_sensitivity(
    scores: pd.DataFrame,
    *,
    tolerance_fraction: float = 0.05,
    smooth_sigma: float | None = 1.0,
    analysis_metrics: tuple[str, ...] = ANALYSIS_METRICS,
) -> pd.DataFrame:
    """Summarize hard-score IQR robustness and final-plateau stability for each curve."""
    data = validate_scores(scores)
    data = data[
        data["formulation"].eq("hard") & data["metric"].isin(analysis_metrics)
    ].copy()
    data["aligned_value"] = data.apply(
        lambda row: row["value"] * METRIC_DIRECTIONS[row["metric"]], axis=1
    )
    curve_keys = ["dataset", "method", "gt_type", "metric"]
    curves = data.groupby(curve_keys + ["threshold"], as_index=False, observed=True).agg(
        mean_aligned_value=("aligned_value", "mean")
    )

    rows: list[dict[str, object]] = []
    for key, curve in curves.groupby(curve_keys, observed=True):
        values = curve["mean_aligned_value"].to_numpy()
        stability = convergence_onset(
            curve["threshold"].to_numpy(),
            values,
            tolerance_fraction=tolerance_fraction,
            smooth_sigma=smooth_sigma,
        )
        rows.append({
            **dict(zip(curve_keys, key, strict=False)),
            "sensitivity_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            **stability,
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    lower, upper = summary["sensitivity_iqr"].quantile([1 / 3, 2 / 3])
    summary["threshold_robustness"] = np.select(
        [summary["sensitivity_iqr"] <= lower, summary["sensitivity_iqr"] >= upper],
        ["high", "low"],
        default="moderate",
    )
    return summary.sort_values(curve_keys).reset_index(drop=True)
