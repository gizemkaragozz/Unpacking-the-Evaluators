"""Sample-level bootstrap analysis for pair and triple selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import rank_preferences
from .constants import ANALYSIS_METRICS
from .io import validate_scores


def bootstrap_preferences(
    scores: pd.DataFrame,
    *,
    formulation: str,
    hard_threshold: float = 0.10,
    n_bootstrap: int = 1000,
    seed: int = 42,
    top_gt: int = 2,
    top_metrics: int = 2,
    analysis_metrics: tuple[str, ...] = ANALYSIS_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate selection frequencies and percentile intervals by resampling samples."""
    data = validate_scores(scores)
    rng = np.random.default_rng(seed)
    pair_runs: list[pd.DataFrame] = []
    triple_runs: list[pd.DataFrame] = []

    for _dataset, dataset_rows in data.groupby("dataset", observed=True):
        sample_ids = dataset_rows["sample_id"].unique()
        if not len(sample_ids):
            continue
        indexed = {sample_id: dataset_rows[dataset_rows["sample_id"].eq(sample_id)]
                   for sample_id in sample_ids}
        for iteration in range(n_bootstrap):
            sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
            sampled = pd.concat(
                [indexed[sample_id] for sample_id in sampled_ids], ignore_index=True
            )
            pairs, _, triples = rank_preferences(
                sampled,
                formulation=formulation,
                hard_threshold=hard_threshold,
                top_gt=top_gt,
                top_metrics=top_metrics,
                analysis_metrics=analysis_metrics,
            )
            pairs = pairs.assign(iteration=iteration, selected=pairs["pair_rank"] <= top_gt)
            triples = triples.assign(iteration=iteration, selected=True)
            pair_runs.append(pairs)
            triple_runs.append(triples)

    if not pair_runs:
        return pd.DataFrame(), pd.DataFrame()

    pair_all = pd.concat(pair_runs, ignore_index=True)
    pair_keys = ["dataset", "formulation", "method", "gt_type"]
    pair_summary = pair_all.groupby(pair_keys, as_index=False, observed=True).agg(
        selection_frequency=("selected", "mean"),
        mean_z_score=("mean_z_score", "mean"),
        mean_z_ci_low=("mean_z_score", lambda values: np.percentile(values, 2.5)),
        mean_z_ci_high=("mean_z_score", lambda values: np.percentile(values, 97.5)),
        mean_rank=("pair_rank", "mean"),
        n_bootstrap=("iteration", "nunique"),
    )

    triple_all = pd.concat(triple_runs, ignore_index=True) if triple_runs else pd.DataFrame()
    if triple_all.empty:
        return pair_summary, triple_all
    triple_keys = pair_keys + ["metric"]
    selected_counts = triple_all.groupby(triple_keys, as_index=False, observed=True).agg(
        selected_iterations=("iteration", "nunique"),
        mean_metric_z=("z_score", "mean"),
        metric_z_ci_low=("z_score", lambda values: np.percentile(values, 2.5)),
        metric_z_ci_high=("z_score", lambda values: np.percentile(values, 97.5)),
    )
    total_runs = pair_all.groupby(pair_keys, as_index=False, observed=True).agg(
        n_bootstrap=("iteration", "nunique")
    )
    triple_summary = selected_counts.merge(total_runs, on=pair_keys, how="left")
    triple_summary["selection_frequency"] = (
        triple_summary["selected_iterations"] / triple_summary["n_bootstrap"]
    )
    triple_summary = triple_summary.merge(
        pair_summary[pair_keys + ["selection_frequency"]].rename(
            columns={"selection_frequency": "pair_selection_frequency"}
        ),
        on=pair_keys,
        how="left",
    )
    triple_summary["confidence"] = np.minimum(
        triple_summary["pair_selection_frequency"], triple_summary["selection_frequency"]
    )
    triple_summary["confidence_level"] = pd.cut(
        triple_summary["confidence"],
        bins=[-np.inf, 0.50, 0.75, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    ).astype(str)
    return pair_summary, triple_summary
