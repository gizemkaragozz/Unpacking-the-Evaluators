import numpy as np
import pandas as pd

from xai_alignment.analysis import (
    build_submitted_flow_paths,
    compare_formulations,
    rank_preferences,
)
from xai_alignment.constants import GT_TYPES, METHODS, METRIC_DIRECTIONS


def make_scores():
    rows = []
    for formulation in ("soft", "hard"):
        for sample in range(4):
            for method_index, method in enumerate(METHODS):
                for gt_index, gt_type in enumerate(GT_TYPES):
                    for metric, direction in METRIC_DIRECTIONS.items():
                        preference = 10 - abs(method_index - gt_index)
                        value = preference if direction == 1 else -preference
                        rows.append({
                            "dataset": "d", "sample_id": str(sample), "method": method,
                            "gt_type": gt_type, "formulation": formulation,
                            "threshold": np.nan if formulation == "soft" else 0.10,
                            "metric": metric, "value": value + sample * 0.01,
                        })
    return pd.DataFrame(rows)


def test_rank_preferences_uses_selected_metrics():
    selected = ("iou", "pg", "epg", "distance")
    pairs, supporting, triples = rank_preferences(
        make_scores(), formulation="soft", analysis_metrics=selected
    )
    assert len(pairs) == len(METHODS) * len(GT_TYPES)
    assert (pairs.groupby("method")["pair_rank"].nunique() == 4).all()
    assert (pairs["n_metrics"] == 4).all()
    assert "ber" not in set(supporting["metric"])
    assert "ber" not in set(triples["metric"])


def test_submitted_flow_paths_use_ordinal_431_weights_without_ties():
    rows = []
    for pair_rank, gt_type in ((1, "region"), (2, "boundary")):
        for metric_rank, metric in ((1, "iou"), (2, "pg")):
            rows.append({
                "dataset": "d", "formulation": "soft", "method": "GC",
                "gt_type": gt_type, "metric": metric, "pair_rank": pair_rank,
                "metric_rank": metric_rank, "mean_z_score": 0.0, "z_score": 0.0,
            })
    paths = build_submitted_flow_paths(pd.DataFrame(rows))
    assert paths["weight"].tolist() == [4, 3, 2, 1]


def test_identical_formulations_have_perfect_rank_agreement():
    scores = make_scores()
    soft, _, _ = rank_preferences(scores, formulation="soft")
    hard, _, _ = rank_preferences(scores, formulation="hard")
    agreement = compare_formulations(pd.concat([soft, hard]))
    assert np.allclose(agreement["kendall_tau"], 1.0)


def test_infinite_distance_is_handled_without_nonfinite_rankings():
    scores = make_scores()
    mask = (
        scores["formulation"].eq("hard")
        & scores["metric"].eq("distance")
        & scores["method"].eq("GC")
        & scores["gt_type"].eq("region")
    )
    scores.loc[mask, "value"] = np.inf

    _, supporting, _ = rank_preferences(scores, formulation="hard")
    distance_region = supporting[
        supporting["metric"].eq("distance")
        & supporting["gt_type"].eq("region")
    ]
    assert np.allclose(distance_region["z_score"], 0.0)

