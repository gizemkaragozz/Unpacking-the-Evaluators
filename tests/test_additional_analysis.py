import numpy as np
import pandas as pd

from additional_analysis.fixed_gt_conditional import conditional_tables
from additional_analysis.taxonomy_support_sensitivity import behavior_label
from additional_analysis.taxonomy_support_sensitivity import quadrant_signature


def test_behavior_labels_follow_bmr_le_axes():
    assert behavior_label(0.2, 0.2, 0.5, 0.5) == "Region"
    assert behavior_label(0.8, 0.2, 0.5, 0.5) == "Boundary"
    assert behavior_label(0.2, 0.8, 0.5, 0.5) == "Spread"
    assert behavior_label(0.8, 0.8, 0.5, 0.5) == "Scatter"


def test_quadrant_signature_is_independent_of_behavior_names():
    assert quadrant_signature(0.2, 0.8, 0.5, 0.5) == "low-BMR/high-LE"
    assert quadrant_signature(0.8, 0.2, 0.5, 0.5) == "high-BMR/low-LE"


def test_fixed_gt_ranking_respects_metric_direction():
    scores = pd.DataFrame(
        {
            "dataset": ["example", "example"],
            "sample_id": ["1", "1"],
            "method": ["A", "B"],
            "gt_type": ["region", "region"],
            "formulation": ["soft", "soft"],
            "threshold": [np.nan, np.nan],
            "metric": ["distance", "distance"],
            "value": [1.0, 2.0],
        }
    )
    metric_summary, overall = conditional_tables(scores, ("distance",), 0.10)
    assert metric_summary.loc[metric_summary["method_rank"].eq(1), "method"].item() == "A"
    assert overall.loc[overall["conditional_rank"].eq(1), "method"].item() == "A"
