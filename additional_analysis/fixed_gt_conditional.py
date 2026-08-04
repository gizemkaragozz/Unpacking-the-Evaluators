#!/usr/bin/env python
"""Condition existing alignment results on each available GT representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DIRECTIONS = {
    "iou": 1,
    "pg": 1,
    "epg": 1,
    "ber": -1,
    "distance": -1,
    "hd": -1,
}
METHOD_ORDER = ("GC", "GGC", "IG", "SL")
GT_ORDER = ("region", "boundary", "bbox", "point")
METRIC_ORDER = ("iou", "pg", "epg", "distance")


def zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / standard_deviation


def conditional_tables(
    scores: pd.DataFrame,
    metrics: tuple[str, ...],
    hard_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "dataset", "sample_id", "method", "gt_type", "formulation",
        "threshold", "metric", "value",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"score table is missing columns: {sorted(missing)}")
    data = scores.copy()
    for column in ("method", "gt_type", "formulation", "metric"):
        data[column] = data[column].astype(str)
    data["gt_type"] = data["gt_type"].str.lower()
    data["formulation"] = data["formulation"].str.lower()
    data["metric"] = data["metric"].str.lower()
    data["threshold"] = pd.to_numeric(data["threshold"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[data["metric"].isin(metrics)].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["value"])
    hard = data["formulation"].eq("hard")
    data = data[~hard | np.isclose(data["threshold"], hard_threshold)].copy()
    if data.empty:
        raise ValueError("no rows remain after metric and formulation selection")

    group = ["dataset", "formulation", "gt_type", "metric", "method"]
    metric_summary = data.groupby(group, as_index=False, observed=True).agg(
        n=("sample_id", "nunique"), mean_value=("value", "mean")
    )
    metric_summary["direction"] = metric_summary["metric"].map(DEFAULT_DIRECTIONS)
    if metric_summary["direction"].isna().any():
        missing_directions = sorted(
            metric_summary.loc[metric_summary["direction"].isna(), "metric"].unique()
        )
        raise ValueError(f"missing metric directions: {missing_directions}")
    metric_summary["aligned_value"] = (
        metric_summary["mean_value"] * metric_summary["direction"]
    )
    comparison = ["dataset", "formulation", "gt_type", "metric"]
    metric_summary["z_score"] = metric_summary.groupby(
        comparison, observed=True
    )["aligned_value"].transform(zscore)
    metric_summary["method_rank"] = metric_summary.groupby(
        comparison, observed=True
    )["aligned_value"].rank(method="dense", ascending=False).astype(int)

    overall_group = ["dataset", "formulation", "gt_type", "method"]
    overall = metric_summary.groupby(overall_group, as_index=False, observed=True).agg(
        mean_z_score=("z_score", "mean"),
        n_metrics=("metric", "nunique"),
    )
    overall["conditional_rank"] = overall.groupby(
        ["dataset", "formulation", "gt_type"], observed=True
    )["mean_z_score"].rank(method="dense", ascending=False).astype(int)
    return (
        metric_summary.sort_values(comparison + ["method_rank", "method"]),
        overall.sort_values(["dataset", "formulation", "gt_type", "conditional_rank"]),
    )


def plot_results(metric_summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    for (dataset, formulation), frame in metric_summary.groupby(
        ["dataset", "formulation"], sort=True
    ):
        available_gt = set(frame["gt_type"].unique())
        gt_types = [gt for gt in GT_ORDER if gt in available_gt]
        gt_types.extend(sorted(available_gt - set(gt_types)))
        figure, axes = plt.subplots(
            1,
            len(gt_types),
            figsize=(4 * len(gt_types), 4),
            squeeze=False,
            layout="constrained",
        )
        for axis, gt_type in zip(axes[0], gt_types, strict=True):
            current = frame[frame["gt_type"].eq(gt_type)]
            pivot = current.pivot(index="method", columns="metric", values="z_score")
            methods = [method for method in METHOD_ORDER if method in pivot.index]
            methods.extend(sorted(set(pivot.index) - set(methods)))
            metrics = [metric for metric in METRIC_ORDER if metric in pivot.columns]
            metrics.extend(sorted(set(pivot.columns) - set(metrics)))
            pivot = pivot.reindex(index=methods, columns=metrics)
            image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
            axis.set_title(gt_type.title())
            axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            axis.set_yticks(range(len(pivot.index)), pivot.index)
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Within-metric standardized score",
            shrink=0.82,
            pad=0.02,
        )
        figure.suptitle(f"{dataset}: {formulation.title()} evaluation conditioned on GT")
        safe_dataset = dataset.lower().replace(" ", "_")
        figure.savefig(
            output / f"fixed_gt_{safe_dataset}_{formulation}.png",
            dpi=300,
            bbox_inches="tight",
        )
        figure.savefig(
            output / f"fixed_gt_{safe_dataset}_{formulation}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=["iou", "pg", "epg", "distance"])
    parser.add_argument("--hard-threshold", type=float, default=0.10)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_summary, overall = conditional_tables(
        pd.read_csv(args.scores),
        tuple(metric.lower() for metric in args.metrics),
        args.hard_threshold,
    )
    metric_summary.to_csv(output / "fixed_gt_metric_summary.csv", index=False)
    overall.to_csv(output / "fixed_gt_overall_summary.csv", index=False)
    overall[overall["conditional_rank"].eq(1)].to_csv(
        output / "fixed_gt_top_methods.csv", index=False
    )
    if args.figures:
        plot_results(metric_summary, output)
    print(overall[overall["conditional_rank"].eq(1)].to_string(index=False))


if __name__ == "__main__":
    main()
