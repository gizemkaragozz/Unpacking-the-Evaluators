import numpy as np

from xai_alignment.metrics import (
    compute_alignment_scores,
    hard_distance,
    soft_distance,
    soft_epg,
    support_boundary,
    topk_support,
)


def test_topk_support_selects_exact_count_with_ties():
    values = np.ones((4, 5))
    support = topk_support(values, 0.10)
    assert support.sum() == 2
    assert support.ravel()[:2].all()


def test_topk_support_is_empty_for_zero_attribution():
    assert not topk_support(np.zeros((4, 5)), 0.10).any()


def test_perfect_soft_energy_score():
    attribution = np.zeros((6, 6))
    attribution[2:4, 2:4] = 1
    gt = attribution.copy()
    assert np.isclose(soft_epg(attribution, gt), 1.0)
    scores = compute_alignment_scores(attribution, gt, formulation="soft")
    assert np.isclose(scores["iou"], 1.0)
    assert np.isclose(scores["ber"], 0.0)


def test_hard_distance_is_zero_for_identical_exact_support():
    attribution = np.arange(16, dtype=float).reshape(4, 4)
    gt = topk_support(attribution, 0.25)
    assert hard_distance(attribution, gt, 0.25) == 0.0


def test_soft_distance_is_directed_from_gt_boundary_to_weighted_attribution():
    attribution = np.zeros((5, 5), dtype=float)
    attribution[2, 2] = 1.0
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[2, 4] = 1
    boundary_points = np.argwhere(support_boundary(gt))
    expected = np.linalg.norm(boundary_points - np.array([2, 2]), axis=1).mean()
    assert np.isclose(soft_distance(attribution, gt), expected)


def test_3d_metrics_ignore_slices_without_gt():
    attribution = np.zeros((3, 3, 3), dtype=float)
    attribution[1, 1, 1] = 1.0
    attribution[0, 0, 0] = 100.0
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    gt[1, 1, 1] = 1
    scores = compute_alignment_scores(
        attribution, gt, formulation="soft", metrics=("pg", "epg")
    )
    assert np.isclose(scores["pg"], 1.0)
    assert np.isclose(scores["epg"], 1.0)


def test_3d_hard_support_is_selected_before_gt_slice_restriction():
    attribution = np.zeros((3, 3, 3), dtype=float)
    attribution[0, 0, 0] = 2.0
    attribution[1, 1, 1] = 1.0
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    gt[1, 1, 1] = 1
    scores = compute_alignment_scores(
        attribution,
        gt,
        formulation="hard",
        threshold=1 / 27,
        metrics=("epg",),
    )
    assert np.isclose(scores["epg"], 0.0)


def test_metric_selection_is_configurable():
    attribution = np.eye(4)
    gt = np.eye(4)
    scores = compute_alignment_scores(
        attribution, gt, formulation="soft", metrics=("iou", "distance")
    )
    assert set(scores) == {"iou", "distance"}
