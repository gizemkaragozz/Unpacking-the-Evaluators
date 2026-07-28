"""Output descriptors used for attribution-map characterization."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy import ndimage

from .metrics import topk_support


def local_spatial_entropy(
    attribution: np.ndarray,
    *,
    patch_size: int | tuple[int, ...],
    threshold: float = 0.10,
) -> float:
    """Average normalized Shannon entropy over every non-overlapping spatial patch."""
    values = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float)))
    support = topk_support(values, threshold)
    retained = values * support
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * retained.ndim
    if len(patch_size) != retained.ndim:
        raise ValueError("patch_size dimensionality must match attribution")
    if any(step <= 0 for step in patch_size):
        raise ValueError("patch dimensions must be positive")
    if any(size % step for size, step in zip(retained.shape, patch_size, strict=False)):
        raise ValueError("each attribution dimension must be divisible by its patch size")

    starts = [range(0, size, step) for size, step in zip(retained.shape, patch_size, strict=False)]
    entropies: list[float] = []
    for start in product(*starts):
        slices = tuple(
            slice(offset, min(offset + step, size))
            for offset, step, size in zip(start, patch_size, retained.shape, strict=False)
        )
        active = retained[slices]
        active = active[active > 0]
        if active.size <= 1:
            entropies.append(0.0)
            continue
        probabilities = active / active.sum()
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
        entropies.append(float(entropy / np.log(active.size)))
    return float(np.mean(entropies)) if entropies else 0.0


def boundary_mass_ratio(attribution: np.ndarray, *, threshold: float = 0.10) -> float:
    """Fraction of retained attribution mass lying on its support boundary."""
    values = np.nan_to_num(np.abs(np.asarray(attribution, dtype=float)))
    support = topk_support(values, threshold)
    structure = ndimage.generate_binary_structure(support.ndim, 1)
    eroded = ndimage.binary_erosion(
        support, structure=structure, iterations=1, border_value=0
    )
    boundary = support & ~eroded
    denominator = values[support].sum()
    return float(values[boundary].sum() / denominator) if denominator > 0 else 0.0
