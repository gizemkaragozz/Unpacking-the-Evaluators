"""Input validation and reproducible tabular output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FORMULATIONS, GT_TYPES, METHODS, METRIC_DIRECTIONS

REQUIRED_SCORE_COLUMNS = {
    "dataset",
    "sample_id",
    "method",
    "gt_type",
    "formulation",
    "threshold",
    "metric",
    "value",
}


def validate_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the long-form score table."""
    missing = REQUIRED_SCORE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"score table is missing columns: {sorted(missing)}")
    scores = frame.copy()
    scores["sample_id"] = scores["sample_id"].astype(str)
    scores["method"] = scores["method"].astype(str).str.upper()
    scores["gt_type"] = scores["gt_type"].astype(str).str.lower()
    scores["formulation"] = scores["formulation"].astype(str).str.lower()
    scores["metric"] = scores["metric"].astype(str).str.lower()
    scores["threshold"] = pd.to_numeric(scores["threshold"], errors="coerce")
    scores["value"] = pd.to_numeric(scores["value"], errors="raise")

    checks = {
        "method": (set(scores["method"]) - set(METHODS)),
        "gt_type": (set(scores["gt_type"]) - set(GT_TYPES)),
        "formulation": (set(scores["formulation"]) - set(FORMULATIONS)),
        "metric": (set(scores["metric"]) - set(METRIC_DIRECTIONS)),
    }
    errors = {name: sorted(values) for name, values in checks.items() if values}
    if errors:
        raise ValueError(f"score table contains unsupported labels: {errors}")
    if scores["dataset"].isna().any() or scores["sample_id"].isna().any():
        raise ValueError("dataset and sample_id must be non-empty")
    if scores["value"].isna().any():
        raise ValueError("value contains NaN; remove failed metric rows before analysis")
    infinite = ~np.isfinite(scores["value"])
    if (infinite & ~scores["metric"].eq("distance")).any():
        raise ValueError("infinite values are supported only for undefined distance scores")
    hard_without_threshold = scores["formulation"].eq("hard") & scores["threshold"].isna()
    if hard_without_threshold.any():
        raise ValueError("every hard-formulation row requires a threshold")
    return scores


def read_scores(path: str | Path) -> pd.DataFrame:
    return validate_scores(pd.read_csv(path))


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, float_format="%.8g")
