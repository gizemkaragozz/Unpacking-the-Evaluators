#!/usr/bin/env python
"""Generate rankings, flow tables, bootstrap summaries, and threshold sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from xai_alignment.analysis import (
    build_flow_tables,
    build_submitted_flow_paths,
    compare_formulations,
    rank_preferences,
)
from xai_alignment.bootstrap import bootstrap_preferences
from xai_alignment.sensitivity import summarize_threshold_sensitivity


def write(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    frame.to_csv(output_dir / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--descriptors")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    scores = pd.read_csv(args.scores)
    selected_metrics = tuple(config["analysis"]["metrics"])
    hard_threshold = float(config["analysis"]["hard_threshold"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_tables, triple_tables = [], []
    for formulation in ("soft", "hard"):
        pairs, supporting, triples = rank_preferences(
            scores,
            formulation=formulation,
            hard_threshold=hard_threshold,
            top_gt=int(config["analysis"]["top_gt"]),
            top_metrics=int(config["analysis"]["top_metrics"]),
            analysis_metrics=selected_metrics,
        )
        write(pairs, output_dir, f"{formulation}_pair_rankings.csv")
        write(supporting, output_dir, f"{formulation}_metric_support.csv")
        write(triples, output_dir, f"{formulation}_triples.csv")
        pair_tables.append(pairs)
        triple_tables.append(triples)

        pair_bootstrap, metric_bootstrap = bootstrap_preferences(
            scores,
            formulation=formulation,
            hard_threshold=hard_threshold,
            n_bootstrap=int(config["analysis"]["bootstrap_iterations"]),
            seed=int(config["analysis"]["seed"]),
            top_gt=int(config["analysis"]["top_gt"]),
            top_metrics=int(config["analysis"]["top_metrics"]),
            analysis_metrics=selected_metrics,
        )
        write(pair_bootstrap, output_dir, f"{formulation}_pair_bootstrap.csv")
        write(metric_bootstrap, output_dir, f"{formulation}_metric_bootstrap.csv")

    all_pairs = pd.concat(pair_tables, ignore_index=True)
    all_triples = pd.concat(triple_tables, ignore_index=True)
    flow_counts, flow_weighted = build_flow_tables(all_triples)
    flow_paths = build_submitted_flow_paths(all_triples)
    write(all_pairs, output_dir, "pair_rankings_all.csv")
    write(all_triples, output_dir, "triples_all.csv")
    write(flow_paths, output_dir, "flow_paths.csv")
    write(flow_counts, output_dir, "flow_counts.csv")
    write(flow_weighted, output_dir, "flow_rank_weighted.csv")
    write(compare_formulations(all_pairs), output_dir, "soft_hard_rank_agreement.csv")
    write(
        summarize_threshold_sensitivity(scores, analysis_metrics=selected_metrics),
        output_dir,
        "threshold_sensitivity.csv",
    )
    descriptors = pd.read_csv(args.descriptors) if args.descriptors else None

    if args.figures:
        from xai_alignment.figures import (
            make_sankey,
            plot_alignment_heatmaps,
            plot_rank_shift,
            plot_taxonomy,
        )

        figure_dir = output_dir / "figures"
        plot_alignment_heatmaps(scores, figure_dir, hard_threshold=hard_threshold)
        plot_rank_shift(all_pairs, figure_dir)
        for formulation in ("soft", "hard"):
            make_sankey(
                flow_paths,
                figure_dir / f"{formulation}_selection_flow.html",
                formulation=formulation,
            )
        if descriptors is not None:
            plot_taxonomy(descriptors, figure_dir)


if __name__ == "__main__":
    main()
