"""Figures generated from descriptor, score, and ranked-path tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import METRIC_DIRECTIONS, METHODS
from .io import validate_scores


METHOD_LABELS = {"GC": "GC", "GGC": "GGC", "IG": "IG", "SL": "SL"}
GT_LABELS = {
    "region": "Region",
    "boundary": "Boundary",
    "bbox": "BBox",
    "point": "Point",
}
METRIC_LABELS = {
    "iou": "IoU",
    "pg": "PG",
    "epg": "EPG",
    "ber": "BER",
    "distance": "HD",
}
METHOD_COLORS = {
    "GC": "#1f77b4",
    "GGC": "#ff7f0e",
    "IG": "#2ca02c",
    "SL": "#d62728",
}
HEATMAP_GT_ORDER = ("region", "boundary", "bbox", "point")


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise ImportError("figure generation requires `pip install -e .[plots]`") from error
    return plt


def plot_alignment_heatmaps(
    scores: pd.DataFrame,
    output_dir: str | Path,
    *,
    hard_threshold: float = 0.10,
) -> list[Path]:
    """Save raw mean-score heatmaps for each dataset and formulation."""
    plt = _import_matplotlib()
    data = validate_scores(scores)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for (dataset, formulation), group in data.groupby(
        ["dataset", "formulation"], observed=True
    ):
        if formulation == "hard":
            group = group[np.isclose(group["threshold"], hard_threshold)]
        mean_scores = group.groupby(
            ["method", "gt_type", "metric"], as_index=False, observed=True
        )["value"].mean()
        metrics = [metric for metric in METRIC_LABELS if metric in set(mean_scores["metric"])]
        if not metrics:
            continue
        columns = min(3, len(metrics))
        rows = int(np.ceil(len(metrics) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.0 * columns, 3.4 * rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for axis, metric in zip(flat_axes, metrics, strict=False):
            subset = mean_scores[mean_scores["metric"].eq(metric)].copy()
            matrix = subset.pivot(index="method", columns="gt_type", values="value")
            matrix = matrix.reindex(index=METHODS, columns=HEATMAP_GT_ORDER)
            lower_is_better = METRIC_DIRECTIONS[metric] < 0
            image = axis.imshow(
                matrix.to_numpy(),
                aspect="auto",
                cmap="viridis_r" if lower_is_better else "viridis",
            )
            axis.set_title(METRIC_LABELS[metric])
            axis.set_xticks(
                range(len(HEATMAP_GT_ORDER)),
                [GT_LABELS[name] for name in HEATMAP_GT_ORDER],
                rotation=45,
                ha="right",
            )
            axis.set_yticks(range(len(METHODS)), METHODS)
            for row in range(len(METHODS)):
                for column in range(len(HEATMAP_GT_ORDER)):
                    value = matrix.iloc[row, column]
                    if pd.notna(value):
                        label = f"{value:.1e}" if 0 < abs(value) < 1e-3 else f"{value:.2f}"
                        axis.text(
                            column,
                            row,
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            direction = "lower better" if lower_is_better else "higher better"
            colorbar.set_label(f"{METRIC_LABELS[metric]} ({direction})")
        for axis in flat_axes[len(metrics):]:
            axis.set_visible(False)
        fig.suptitle(f"{dataset}: {formulation.capitalize()} alignment")
        fig.subplots_adjust(
            left=0.08,
            right=0.96,
            bottom=0.12,
            top=0.88,
            wspace=0.45,
            hspace=0.45,
        )
        path = output / f"{dataset}_{formulation}_alignment_heatmaps.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_rank_shift(pair_rankings: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save the submitted soft--hard paired-rank comparison for each dataset."""
    plt = _import_matplotlib()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for dataset, group in pair_rankings.groupby("dataset", observed=True):
        pivot = group.pivot_table(
            index=["method", "gt_type"], columns="formulation", values="pair_rank"
        ).dropna()
        if pivot.empty:
            continue
        fig, axis = plt.subplots(figsize=(8, 5))
        for (method, gt_type), row in pivot.iterrows():
            axis.plot(
                [0, 1], [row["soft"], row["hard"]],
                marker="o", color=METHOD_COLORS.get(method, "gray"), alpha=0.75,
            )
            axis.text(-0.02, row["soft"], GT_LABELS.get(gt_type, gt_type),
                      ha="right", va="center", fontsize=8)
        axis.set_xticks([0, 1], ["Soft", "Hard"])
        axis.set_ylabel("XAI–GT pair rank")
        axis.invert_yaxis()
        axis.set_title(f"{dataset}: formulation rank shift")
        path = output / f"{dataset}_soft_hard_rank_shift.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_taxonomy(descriptors: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save dataset-level BMR--LE method means."""
    plt = _import_matplotlib()
    required = {"dataset", "method", "bmr", "le"}
    if missing := required - set(descriptors.columns):
        raise ValueError(f"descriptor table is missing columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for dataset, group in descriptors.groupby("dataset", observed=True):
        means = group.groupby("method", as_index=False, observed=True)[["bmr", "le"]].mean()
        fig, axis = plt.subplots(figsize=(7, 5.5))
        for row in means.itertuples(index=False):
            axis.scatter(row.bmr, row.le, s=90, color=METHOD_COLORS.get(row.method, "gray"))
            axis.annotate(row.method, (row.bmr, row.le), xytext=(4, 4),
                          textcoords="offset points")
        if len(means):
            axis.axvline((means["bmr"].min() + means["bmr"].max()) / 2,
                         color="steelblue", alpha=0.3)
            axis.axhline((means["le"].min() + means["le"].max()) / 2,
                         color="steelblue", alpha=0.3)
        axis.set_xlabel("Boundary Mass Ratio (BMR)")
        axis.set_ylabel("Local Entropy (LE)")
        axis.set_title(f"{dataset}: attribution-map behavior")
        path = output / f"{dataset}_bmr_le.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_sankey(
    paths: pd.DataFrame,
    output_path: str | Path,
    *,
    formulation: str,
):
    """Write an interactive method–GT–metric Sankey from weighted paths."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover
        raise ImportError("Sankey generation requires `pip install -e .[plots]`") from error

    data = paths[paths["formulation"].eq(formulation)].copy()
    if data.empty:
        raise ValueError(f"no weighted paths for formulation={formulation!r}")
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    data["gt_label"] = data["gt_type"].map(GT_LABELS).fillna(data["gt_type"])
    data["metric_label"] = data["metric"].map(METRIC_LABELS).fillna(data["metric"])
    method_nodes = [f"M:{name}" for name in METHODS if name in set(data["method_label"])]
    gt_nodes = [f"G:{GT_LABELS[name]}" for name in HEATMAP_GT_ORDER
                if GT_LABELS[name] in set(data["gt_label"])]
    metric_nodes = [f"T:{METRIC_LABELS[name]}" for name in METRIC_LABELS
                    if METRIC_LABELS[name] in set(data["metric_label"])]
    nodes = method_nodes + gt_nodes + metric_nodes
    node_index = {node: index for index, node in enumerate(nodes)}

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    colors: list[str] = []
    rgba = {
        "GC": "rgba(31,119,180,0.45)",
        "GGC": "rgba(255,127,14,0.45)",
        "IG": "rgba(44,160,44,0.45)",
        "SL": "rgba(214,39,40,0.45)",
    }
    left = data.groupby(["method_label", "gt_label"], as_index=False)["weight"].sum()
    for row in left.itertuples(index=False):
        sources.append(node_index[f"M:{row.method_label}"])
        targets.append(node_index[f"G:{row.gt_label}"])
        values.append(float(row.weight))
        colors.append(rgba.get(row.method_label, "rgba(120,120,120,0.35)"))
    right = data.groupby(
        ["method_label", "gt_label", "metric_label"], as_index=False
    )["weight"].sum()
    for row in right.itertuples(index=False):
        sources.append(node_index[f"G:{row.gt_label}"])
        targets.append(node_index[f"T:{row.metric_label}"])
        values.append(float(row.weight))
        colors.append(rgba.get(row.method_label, "rgba(120,120,120,0.35)"))

    labels = [node.split(":", 1)[1] for node in nodes]
    figure = go.Figure(go.Sankey(
        arrangement="snap",
        node={"pad": 22, "thickness": 20, "label": labels},
        link={"source": sources, "target": targets, "value": values, "color": colors},
    ))
    figure.update_layout(
        title=f"{formulation.capitalize()} Metric Selection Flow: Method → GT → Metric",
        width=850,
        height=480,
        font={"size": 14},
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(destination)
    return figure
