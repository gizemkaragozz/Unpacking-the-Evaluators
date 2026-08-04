#!/usr/bin/env python
"""Evaluate BMR--LE sensitivity to the retained attribution-support fraction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from xai_alignment.taxonomy import boundary_mass_ratio, local_spatial_entropy


METHOD_ALIASES = {
    "GradCAM": "GC",
    "Guided GradCAM": "GGC",
    "Integrated Gradients": "IG",
    "Saliency": "SL",
    "gradcam": "GC",
    "guided_gradcam": "GGC",
    "integrated_gradients": "IG",
    "saliency": "SL",
    "deeplift": "DL",
    "eigencam": "EC",
    "feature_ablation": "FA",
    "gradcampp": "GC++",
    "gradient_shap": "GS",
    "guidedbp": "GBP",
    "input_x_gradient": "IxG",
    "layercam": "LC",
    "lrp": "LRP",
    "occlusion": "Occ",
    "smoothgrad": "SG",
}


def behavior_label(bmr: float, le: float, bmr_split: float, le_split: float) -> str:
    """Assign the dataset-relative descriptive region used in the paper."""
    if bmr < bmr_split and le < le_split:
        return "Region"
    if bmr >= bmr_split and le < le_split:
        return "Boundary"
    if bmr < bmr_split and le >= le_split:
        return "Spread"
    return "Scatter"


def quadrant_signature(bmr: float, le: float, bmr_split: float, le_split: float) -> str:
    """Return the low/high descriptor signature for compatibility with analyses."""
    return ("low" if bmr < bmr_split else "high") + "-BMR/" + (
        "low-LE" if le < le_split else "high-LE"
    )


def iter_archives(root: Path):
    for path in sorted(root.glob("*/*.npz")):
        with np.load(path) as archive:
            for key in archive.files:
                if not key.startswith("_"):
                    yield f"{path.parent.name}:{path.stem}", METHOD_ALIASES.get(key, key), archive[key]


def iter_npy_maps(root: Path):
    for path in sorted(root.glob("*/*.npy")):
        method = path.stem.rsplit("_target", 1)[0]
        yield path.parent.name, METHOD_ALIASES.get(method, method), path


def _compute_npy_map(task):
    sample_id, method, path, support_fractions, patch_size = task
    attribution = np.load(path)
    return [
        {
            "sample_id": sample_id,
            "method": method,
            "support_fraction": support_fraction,
            "bmr": boundary_mass_ratio(attribution, threshold=support_fraction),
            "le": local_spatial_entropy(
                attribution,
                patch_size=patch_size,
                threshold=support_fraction,
            ),
        }
        for support_fraction in support_fractions
    ]


def compute_descriptors(
    input_root: Path,
    support_fractions: tuple[float, ...],
    patch_size: tuple[int, ...],
    input_format: str = "npz",
    workers: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if input_format == "npz":
        for sample_id, method, attribution in iter_archives(input_root):
            for support_fraction in support_fractions:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "support_fraction": support_fraction,
                        "bmr": boundary_mass_ratio(attribution, threshold=support_fraction),
                        "le": local_spatial_entropy(
                            attribution,
                            patch_size=patch_size,
                            threshold=support_fraction,
                        ),
                    }
                )
    else:
        tasks = (
            (sample_id, method, path, support_fractions, patch_size)
            for sample_id, method, path in iter_npy_maps(input_root)
        )
        if workers == 1:
            for result in map(_compute_npy_map, tasks):
                rows.extend(result)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(_compute_npy_map, tasks, chunksize=8):
                    rows.extend(result)
    if not rows:
        raise ValueError(f"no attribution maps found under {input_root}")
    return pd.DataFrame(rows)


def summarize(descriptors: pd.DataFrame, reference: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_summary = (
        descriptors.groupby(["support_fraction", "method"], as_index=False)
        .agg(n=("sample_id", "nunique"), mean_bmr=("bmr", "mean"), mean_le=("le", "mean"))
        .sort_values(["support_fraction", "method"])
    )
    for descriptor in ("bmr", "le"):
        column = f"mean_{descriptor}"
        minimum = method_summary.groupby("support_fraction")[column].transform("min")
        maximum = method_summary.groupby("support_fraction")[column].transform("max")
        span = maximum - minimum
        method_summary[f"relative_{descriptor}"] = np.divide(
            method_summary[column] - minimum,
            span,
            out=np.zeros(len(method_summary), dtype=float),
            where=span.to_numpy() > 0,
        )
    method_summary["relative_label"] = [
        behavior_label(bmr, le, 0.5, 0.5)
        for bmr, le in zip(method_summary["relative_bmr"], method_summary["relative_le"], strict=True)
    ]
    method_summary["bmr_rank_low_to_high"] = method_summary.groupby(
        "support_fraction", observed=True
    )["mean_bmr"].rank(method="average", ascending=True)
    method_summary["le_rank_low_to_high"] = method_summary.groupby(
        "support_fraction", observed=True
    )["mean_le"].rank(method="average", ascending=True)

    reference_rows = method_summary[np.isclose(method_summary["support_fraction"], reference)]
    if reference_rows.empty:
        raise ValueError("reference support fraction is absent from the evaluated grid")
    reference_rows = reference_rows.set_index("method")
    bmr_split = 0.5 * (reference_rows["mean_bmr"].min() + reference_rows["mean_bmr"].max())
    le_split = 0.5 * (reference_rows["mean_le"].min() + reference_rows["mean_le"].max())
    method_summary["reference_boundary_label"] = [
        behavior_label(bmr, le, bmr_split, le_split)
        for bmr, le in zip(method_summary["mean_bmr"], method_summary["mean_le"], strict=True)
    ]

    rows: list[dict[str, object]] = []
    reference_rows = method_summary[
        np.isclose(method_summary["support_fraction"], reference)
    ].set_index("method")
    method_summary["bmr_rank_shift_vs_0.10"] = [
        rank - reference_rows.loc[method, "bmr_rank_low_to_high"]
        for method, rank in zip(
            method_summary["method"], method_summary["bmr_rank_low_to_high"], strict=True
        )
    ]
    method_summary["le_rank_shift_vs_0.10"] = [
        rank - reference_rows.loc[method, "le_rank_low_to_high"]
        for method, rank in zip(
            method_summary["method"], method_summary["le_rank_low_to_high"], strict=True
        )
    ]
    reference_relative = reference_rows["relative_label"]
    reference_fixed = reference_rows["reference_boundary_label"]
    for support_fraction, current in method_summary.groupby("support_fraction", sort=True):
        current = current.set_index("method").loc[reference_rows.index]
        rows.append(
            {
                "support_fraction": support_fraction,
                "kendall_bmr_vs_0.10": kendalltau(
                    current["mean_bmr"], reference_rows["mean_bmr"]
                ).statistic,
                "kendall_le_vs_0.10": kendalltau(
                    current["mean_le"], reference_rows["mean_le"]
                ).statistic,
                "relative_label_agreement_vs_0.10": (
                    current["relative_label"] == reference_relative
                ).mean(),
                "fixed_boundary_label_agreement_vs_0.10": (
                    current["reference_boundary_label"] == reference_fixed
                ).mean(),
            }
        )
    return method_summary, pd.DataFrame(rows)


def plot_results(method_summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for method, frame in method_summary.groupby("method", sort=True):
        axes[0].plot(frame["support_fraction"], frame["mean_bmr"], marker="o", label=method)
        axes[1].plot(frame["support_fraction"], frame["mean_le"], marker="o", label=method)
    axes[0].set_ylabel("Mean BMR")
    axes[1].set_ylabel("Mean LE")
    for axis in axes:
        axis.set_xlabel("Retained support fraction")
        axis.axvline(0.10, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
    axes[1].legend(title="XAI method", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output / "taxonomy_support_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    fractions = sorted(method_summary["support_fraction"].unique())
    figure, axes = plt.subplots(1, len(fractions), figsize=(3.4 * len(fractions), 3.6), sharex=True, sharey=True)
    for axis, support_fraction in zip(np.atleast_1d(axes), fractions, strict=True):
        current = method_summary[np.isclose(method_summary["support_fraction"], support_fraction)]
        axis.axvline(0.5, color="gray", linewidth=1)
        axis.axhline(0.5, color="gray", linewidth=1)
        for row in current.itertuples(index=False):
            axis.scatter(row.relative_bmr, row.relative_le, s=45)
            axis.annotate(row.method, (row.relative_bmr, row.relative_le), xytext=(4, 4),
                          textcoords="offset points", fontsize=8)
        axis.set_title(f"Support = {support_fraction:.0%}")
        axis.set_xlabel("Relative BMR")
        axis.set_xlim(-0.08, 1.08)
        axis.set_ylim(-0.08, 1.08)
        axis.grid(alpha=0.15)
    np.atleast_1d(axes)[0].set_ylabel("Relative LE")
    figure.tight_layout()
    figure.savefig(output / "taxonomy_support_quadrants.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root")
    parser.add_argument("--npy-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", nargs="+", type=int, default=[16, 16])
    parser.add_argument(
        "--support-fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
    )
    parser.add_argument("--reference", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    if bool(args.archive_root) == bool(args.npy_root):
        raise ValueError("provide exactly one of --archive-root or --npy-root")
    if args.workers < 1:
        raise ValueError("workers must be at least one")

    support_fractions = tuple(dict.fromkeys(float(value) for value in args.support_fractions))
    if any(not 0 < value <= 1 for value in support_fractions):
        raise ValueError("support fractions must lie in (0, 1]")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.archive_root or args.npy_root)
    descriptors = compute_descriptors(
        input_root,
        support_fractions,
        tuple(args.patch_size),
        input_format="npz" if args.archive_root else "npy",
        workers=args.workers,
    )
    method_summary, robustness = summarize(descriptors, args.reference)
    descriptors.to_csv(output / "taxonomy_support_per_explanation.csv", index=False)
    method_summary.to_csv(output / "taxonomy_support_method_summary.csv", index=False)
    robustness.to_csv(output / "taxonomy_support_robustness.csv", index=False)
    if args.figures:
        plot_results(method_summary, output)
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
