"""Pair/triple preference analysis and soft--hard comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .constants import ANALYSIS_METRICS, METRIC_DIRECTIONS
from .io import validate_scores


def _sample_zscore(values: pd.Series) -> pd.Series:
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - values.mean()) / standard_deviation


def filter_analysis_scores(
    scores: pd.DataFrame,
    *,
    formulation: str,
    hard_threshold: float = 0.10,
) -> pd.DataFrame:
    """Select soft rows or one hard threshold for preference analysis."""
    selected = scores[scores["formulation"].eq(formulation)].copy()
    if formulation == "hard":
        selected = selected[np.isclose(selected["threshold"], hard_threshold)].copy()
    if selected.empty:
        raise ValueError(f"no rows found for formulation={formulation!r}")
    return selected


def rank_preferences(
    scores: pd.DataFrame,
    *,
    formulation: str,
    hard_threshold: float = 0.10,
    top_gt: int = 2,
    top_metrics: int = 2,
    analysis_metrics: tuple[str, ...] = ANALYSIS_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank XAI--GT pairs and the metrics supporting each selected pair.

    Raw scores are direction-aligned and z-normalized within each dataset, metric, and GT type.
    """
    data = filter_analysis_scores(validate_scores(scores), formulation=formulation,
                                  hard_threshold=hard_threshold)
    data = data[data["metric"].isin(analysis_metrics)].copy()
    if data.empty:
        raise ValueError("no rows remain after metric selection")
    group_columns = ["dataset", "formulation", "method", "gt_type", "metric"]
    metric_means = data.groupby(group_columns, as_index=False, observed=True)["value"].mean()
    metric_means["aligned_value"] = metric_means.apply(
        lambda row: row["value"] * METRIC_DIRECTIONS[row["metric"]], axis=1
    )
    z_groups = ["dataset", "formulation", "metric", "gt_type"]
    metric_means["z_score"] = metric_means.groupby(
        z_groups, observed=True
    )["aligned_value"].transform(_sample_zscore)

    pair_groups = ["dataset", "formulation", "method", "gt_type"]
    pairs = metric_means.groupby(pair_groups, as_index=False, observed=True).agg(
        mean_z_score=("z_score", "mean"),
        std_z_score=("z_score", lambda values: values.std(ddof=1)),
        n_metrics=("metric", "nunique"),
    )
    pairs["pair_rank"] = pairs.groupby(
        ["dataset", "formulation", "method"], observed=True
    )["mean_z_score"].rank(method="dense", ascending=False).astype(int)
    pairs = pairs.sort_values(["dataset", "method", "pair_rank"]).reset_index(drop=True)

    selected_pairs = pairs[pairs["pair_rank"] <= top_gt]
    supporting = metric_means.merge(
        selected_pairs[pair_groups + ["pair_rank", "mean_z_score"]],
        on=pair_groups,
        how="inner",
    )
    supporting["metric_rank"] = supporting.groupby(
        pair_groups, observed=True
    )["z_score"].rank(method="dense", ascending=False).astype(int)
    supporting = supporting.sort_values(
        ["dataset", "method", "pair_rank", "metric_rank", "metric"]
    ).reset_index(drop=True)
    triples = supporting[supporting["metric_rank"] <= top_metrics].copy()
    return pairs, supporting, triples


def build_submitted_flow_paths(
    triples: pd.DataFrame,
    *,
    top_gt: int = 2,
    top_metrics: int = 2,
) -> pd.DataFrame:
    """Apply the submitted ordinal path weighting (4, 3, 2, 1)."""
    required = {
        "dataset", "formulation", "method", "gt_type", "metric",
        "pair_rank", "metric_rank",
    }
    if missing := required - set(triples.columns):
        raise ValueError(f"triple table is missing columns: {sorted(missing)}")
    if top_gt != 2 or top_metrics != 2:
        raise ValueError("the submitted ordinal weighting uses two GT and metric ranks")

    selected = triples[
        triples["pair_rank"].le(top_gt) & triples["metric_rank"].le(top_metrics)
    ].copy()
    selected = selected.sort_values(
        ["dataset", "formulation", "method", "pair_rank", "metric_rank"]
    )
    selected = selected.drop_duplicates(
        ["dataset", "formulation", "method", "gt_type", "metric_rank"],
        keep="first",
    )
    selected["path_order"] = (
        selected.groupby(["dataset", "formulation", "method"], observed=True)
        .cumcount()
        .add(1)
    )
    selected["weight"] = top_gt * top_metrics - selected["path_order"] + 1
    return selected


def build_flow_tables(triples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selection-count and ordinal-rank flow edges."""
    required = {"dataset", "formulation", "method", "gt_type", "metric", "pair_rank", "metric_rank"}
    if missing := required - set(triples.columns):
        raise ValueError(f"triple table is missing columns: {sorted(missing)}")

    method_gt = triples[
        ["dataset", "formulation", "method", "gt_type", "pair_rank"]
    ].drop_duplicates()
    gt_metric = triples[
        ["dataset", "formulation", "method", "gt_type", "metric", "pair_rank", "metric_rank"]
    ]

    count_left = method_gt.groupby(["formulation", "method", "gt_type"], as_index=False).agg(
        weight=("dataset", "nunique")
    ).assign(stage="method_to_gt")
    count_right = gt_metric.groupby(["formulation", "gt_type", "metric"], as_index=False).agg(
        weight=("dataset", "count")
    ).assign(stage="gt_to_metric")
    counts = pd.concat([count_left, count_right], ignore_index=True, sort=False)

    ranked = build_submitted_flow_paths(triples)
    ranked_left = ranked.groupby(
        ["dataset", "formulation", "method", "gt_type"], as_index=False
    ).agg(weight=("weight", "sum"))
    ranked_left = ranked_left.groupby(
        ["formulation", "method", "gt_type"], as_index=False
    ).agg(weight=("weight", "sum")).assign(stage="method_to_gt")
    ranked_right = ranked.groupby(
        ["formulation", "gt_type", "metric"], as_index=False
    ).agg(
        weight=("weight", "sum")
    ).assign(stage="gt_to_metric")
    rank_weighted = pd.concat([ranked_left, ranked_right], ignore_index=True, sort=False)
    return counts, rank_weighted


def compare_formulations(pair_rankings: pd.DataFrame) -> pd.DataFrame:
    """Compute paired soft--hard rank agreement at dataset and method levels."""
    rows: list[dict[str, object]] = []
    for (dataset, method), group in pair_rankings.groupby(["dataset", "method"], observed=True):
        pivot = group.pivot_table(index="gt_type", columns="formulation", values="pair_rank")
        if not {"soft", "hard"}.issubset(pivot.columns) or len(pivot) < 2:
            continue
        tau, p_value = kendalltau(pivot["soft"], pivot["hard"])
        rows.append({
            "dataset": dataset,
            "scope": "method",
            "method": method,
            "kendall_tau": float(tau),
            "p_value": float(p_value),
            "n_gt_types": int(len(pivot)),
            "mean_absolute_rank_shift": float(
                np.abs(pivot["soft"] - pivot["hard"]).mean()
            ),
            "top1_changed": bool(
                pivot["soft"].idxmin() != pivot["hard"].idxmin()
            ),
        })
    for dataset, group in pair_rankings.groupby("dataset", observed=True):
        pivot = group.pivot_table(
            index=["method", "gt_type"], columns="formulation", values="pair_rank"
        )
        if not {"soft", "hard"}.issubset(pivot.columns) or len(pivot) < 2:
            continue
        tau, p_value = kendalltau(pivot["soft"], pivot["hard"])
        rows.append({
            "dataset": dataset,
            "scope": "dataset",
            "method": "ALL",
            "kendall_tau": float(tau),
            "p_value": float(p_value),
            "n_gt_types": int(len(pivot)),
            "mean_absolute_rank_shift": float(
                np.abs(pivot["soft"] - pivot["hard"]).mean()
            ),
            "top1_changed": bool(
                any(
                    sub["soft"].idxmin() != sub["hard"].idxmin()
                    for _, sub in pivot.groupby(level="method")
                )
            ),
        })
    return pd.DataFrame(rows)

