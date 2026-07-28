#!/usr/bin/env python
"""Compute descriptors and alignment scores from attribution maps and region masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xai_alignment.ground_truth import build_gt_representations
from xai_alignment.metrics import compute_alignment_scores
from xai_alignment.taxonomy import boundary_mass_ratio, local_spatial_entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--descriptors-output", required=True)
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = pd.read_csv(args.manifest)
    required = {"dataset", "sample_id", "method", "attribution_path", "region_mask_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    metrics = tuple(config["alignment"]["metrics"])
    thresholds = tuple(float(value) for value in config["alignment"]["thresholds"])
    soft_pg_alpha = float(config["alignment"]["soft_pg_alpha"])
    support_threshold = float(config["descriptors"]["support_threshold"])
    patch_size = tuple(int(value) for value in config["descriptors"]["patch_size"])
    point_config = config["ground_truth"]["point"]
    point_method = str(point_config["method"])
    point_radius = int(point_config["radius"])
    point_connectivity = int(point_config.get("connectivity", 6))
    score_rows, descriptor_rows = [], []

    for row in manifest.itertuples(index=False):
        attribution = np.load(row.attribution_path)
        region = np.load(row.region_mask_path)
        if attribution.shape != region.shape:
            raise ValueError(f"shape mismatch for sample {row.sample_id}")

        descriptor_rows.append({
            "dataset": row.dataset,
            "sample_id": str(row.sample_id),
            "method": row.method,
            "bmr": boundary_mass_ratio(attribution, threshold=support_threshold),
            "le": local_spatial_entropy(
                attribution, patch_size=patch_size, threshold=support_threshold
            ),
        })

        for gt_type, gt in build_gt_representations(
            region,
            point_method=point_method,
            point_radius=point_radius,
            point_connectivity=point_connectivity,
        ).items():
            for metric, value in compute_alignment_scores(
                attribution,
                gt,
                formulation="soft",
                metrics=metrics,
                soft_pg_alpha=soft_pg_alpha,
            ).items():
                score_rows.append({
                    "dataset": row.dataset, "sample_id": str(row.sample_id),
                    "method": row.method, "gt_type": gt_type, "formulation": "soft",
                    "threshold": np.nan, "metric": metric, "value": value,
                })
            for threshold in thresholds:
                for metric, value in compute_alignment_scores(
                    attribution,
                    gt,
                    formulation="hard",
                    metrics=metrics,
                    threshold=threshold,
                ).items():
                    score_rows.append({
                        "dataset": row.dataset, "sample_id": str(row.sample_id),
                        "method": row.method, "gt_type": gt_type, "formulation": "hard",
                        "threshold": threshold, "metric": metric, "value": value,
                    })

    scores_output = Path(args.scores_output)
    descriptors_output = Path(args.descriptors_output)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    descriptors_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(scores_output, index=False)
    pd.DataFrame(descriptor_rows).to_csv(descriptors_output, index=False)


if __name__ == "__main__":
    main()
