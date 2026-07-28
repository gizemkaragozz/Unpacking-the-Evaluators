"""Soft and hard spatial-alignment metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

EPSILON = 1e-12
AVAILABLE_METRICS = ("iou", "pg", "epg", "ber", "distance")


def normalize_attribution(attribution: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float)))
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0 else np.zeros_like(values)


def restrict_to_gt_slices(
    attribution: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For 3D inputs, retain slices containing GT, as used in the study."""
    values = np.asarray(attribution)
    target = as_target(gt)
    if values.shape != target.shape:
        raise ValueError("attribution and GT must have identical shapes")
    if values.ndim != 3:
        return values, target
    keep = target.sum(axis=(0, 1)) > 0
    if not keep.any():
        return values, target
    return values[:, :, keep], target[:, :, keep]


def support_boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Return the study boundary used by the distance formulations."""
    target = as_target(mask)
    if iterations < 1:
        raise ValueError("boundary iterations must be positive")
    structure = ndimage.generate_binary_structure(target.ndim, 1)
    eroded = ndimage.binary_erosion(
        target,
        structure=structure,
        iterations=iterations,
        border_value=0,
    )
    if target.ndim == 2:
        dilated = ndimage.binary_dilation(
            target,
            structure=structure,
            iterations=iterations,
            border_value=0,
        )
        return np.logical_xor(dilated, eroded)
    return target & ~eroded


def topk_support(attribution: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    values = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float))).ravel()
    if values.size == 0 or float(values.max(initial=0.0)) <= 0:
        return np.zeros(np.asarray(attribution).shape, dtype=bool)
    count = max(1, int(np.ceil(values.size * fraction)))
    order = np.argsort(-values, kind="stable")
    support = np.zeros(values.size, dtype=bool)
    support[order[:count]] = True
    return support.reshape(np.asarray(attribution).shape)


def soft_iou(attribution: np.ndarray, gt: np.ndarray) -> float:
    scores = normalize_attribution(attribution)
    target = as_target(gt).astype(float)
    union = np.maximum(scores, target).sum()
    return float(np.minimum(scores, target).sum() / (union + EPSILON))


def hard_iou(attribution: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    support = topk_support(attribution, threshold)
    support, target = restrict_to_gt_slices(support, gt)
    union = np.logical_or(support, target).sum()
    return float(np.logical_and(support, target).sum() / (union + EPSILON))


def soft_pointing_game(attribution: np.ndarray, gt: np.ndarray, alpha: float = 4.0) -> float:
    scores = normalize_attribution(attribution) ** alpha
    return float((scores * as_target(gt)).sum() / (scores.sum() + EPSILON))


def hard_pointing_game(attribution: np.ndarray, gt: np.ndarray) -> float:
    scores = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float)))
    return float(as_target(gt)[np.unravel_index(np.argmax(scores), scores.shape)])


def soft_epg(attribution: np.ndarray, gt: np.ndarray) -> float:
    scores = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float)))
    return float(scores[as_target(gt)].sum() / (scores.sum() + EPSILON))


def hard_epg(attribution: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    support = topk_support(attribution, threshold)
    support, target = restrict_to_gt_slices(support, gt)
    return float(np.logical_and(support, target).sum() / (support.sum() + EPSILON))


def soft_distance(attribution: np.ndarray, gt: np.ndarray) -> float:
    """Directed attribution-weighted distance from the GT boundary to the explanation."""
    scores = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float)))
    scores, target = restrict_to_gt_slices(scores, gt)
    target = support_boundary(target)
    total = scores.sum()
    if total > 0:
        scores = scores / total
    score_points, target_points = np.argwhere(scores > 0), np.argwhere(target)
    if not len(score_points) or not len(target_points):
        return float("inf")
    weights = scores[tuple(score_points.T)]
    weighted_sum = 0.0
    for start in range(0, len(target_points), 1024):
        distances = cdist(target_points[start:start + 1024], score_points)
        weighted_sum += float((distances * weights[None, :]).sum())
    return float(weighted_sum / (len(target_points) + EPSILON))


def hard_distance(attribution: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Symmetric boundary HD95 for a thresholded explanation support."""
    support = topk_support(attribution, threshold)
    support, target = restrict_to_gt_slices(support, gt)
    support = support_boundary(support)
    target = support_boundary(target)
    if not support.any() or not target.any():
        return float("inf")
    if support.ndim == 2:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - installation guard
            raise ImportError("2D hard distance requires opencv-python-headless") from error
        distance_to_target = cv2.distanceTransform(
            (~target).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_to_support = cv2.distanceTransform(
            (~support).astype(np.uint8), cv2.DIST_L2, 3
        )
    else:
        distance_to_target = ndimage.distance_transform_edt(~target)
        distance_to_support = ndimage.distance_transform_edt(~support)
    directed_support = float(np.percentile(distance_to_target[support], 95))
    directed_target = float(np.percentile(distance_to_support[target], 95))
    return max(directed_support, directed_target)


def as_target(gt: np.ndarray) -> np.ndarray:
    return np.asarray(gt) > 0


def compute_alignment_scores(
    attribution: np.ndarray,
    gt: np.ndarray,
    *,
    formulation: str,
    metrics: Iterable[str] = AVAILABLE_METRICS,
    threshold: float | None = None,
    soft_pg_alpha: float = 4.0,
) -> dict[str, float]:
    """Compute only the requested metric families."""
    requested = tuple(dict.fromkeys(name.lower() for name in metrics))
    unsupported = set(requested) - set(AVAILABLE_METRICS)
    if unsupported:
        raise ValueError(f"unsupported metrics: {sorted(unsupported)}")
    if formulation == "hard" and threshold is None:
        raise ValueError("hard metrics require a threshold")
    if formulation not in {"soft", "hard"}:
        raise ValueError("formulation must be 'soft' or 'hard'")
    if formulation == "soft":
        values, target = restrict_to_gt_slices(attribution, gt)
        epg = soft_epg(values, target)
        available = {
            "iou": lambda: soft_iou(values, target),
            "pg": lambda: soft_pointing_game(values, target, soft_pg_alpha),
            "epg": lambda: epg,
            "ber": lambda: 1.0 - epg,
            "distance": lambda: soft_distance(values, target),
        }
    else:
        assert threshold is not None
        values = np.asarray(attribution)
        target = np.asarray(gt)
        point_values, point_target = restrict_to_gt_slices(values, target)
        epg = hard_epg(values, target, threshold)
        available = {
            "iou": lambda: hard_iou(values, target, threshold),
            "pg": lambda: hard_pointing_game(point_values, point_target),
            "epg": lambda: epg,
            "ber": lambda: 1.0 - epg,
            "distance": lambda: hard_distance(values, target, threshold),
        }
    return {name: float(available[name]()) for name in requested}
