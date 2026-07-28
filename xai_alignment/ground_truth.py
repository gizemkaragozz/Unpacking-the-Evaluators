"""Ground-truth representations derived from a binary region mask."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def as_binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    return np.isfinite(array) & (array > 0)


def region_to_boundary(mask: np.ndarray) -> np.ndarray:
    """Create the study boundary representation for a 2D or 3D region."""
    region = as_binary(mask)
    if not region.any():
        return np.zeros_like(region, dtype=np.uint8)
    structure = ndimage.generate_binary_structure(region.ndim, 1)
    eroded = ndimage.binary_erosion(
        region,
        structure=structure,
        border_value=0,
    )
    if region.ndim == 2:
        dilated = ndimage.binary_dilation(region, structure=structure, border_value=0)
        return np.logical_xor(dilated, eroded).astype(np.uint8)
    return (region & ~eroded).astype(np.uint8)


def _component_structure(ndim: int) -> np.ndarray:
    """Match the study's 8-connected 2D and 6-connected 3D components."""
    if ndim == 2:
        return np.ones((3, 3), dtype=np.uint8)
    return ndimage.generate_binary_structure(3, 1)


def region_to_bounding_boxes(mask: np.ndarray) -> np.ndarray:
    region = as_binary(mask)
    result = np.zeros_like(region, dtype=np.uint8)
    labels, count = ndimage.label(region, structure=_component_structure(region.ndim))
    for index in range(1, count + 1):
        coordinates = np.where(labels == index)
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
        result[slices] = 1
    return result


def _distance_transform_center(component: np.ndarray) -> tuple[int, ...]:
    if component.ndim != 2:
        raise ValueError("distance-transform point generation is defined for 2D components")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - installation guard
        raise ImportError("2D point GT generation requires opencv-python-headless") from error
    distance = cv2.distanceTransform(
        component.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=3,
    )
    return tuple(int(value) for value in np.unravel_index(np.argmax(distance), component.shape))


def _centroid(component: np.ndarray) -> tuple[int, ...]:
    center = np.asarray(ndimage.center_of_mass(component))
    rounded = np.rint(center).astype(int)
    bounded = np.clip(rounded, 0, np.asarray(component.shape) - 1)
    return tuple(int(value) for value in bounded)


def _draw_2d_disc(result: np.ndarray, center: tuple[int, int], radius: int) -> None:
    row, column = center
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            if row_offset**2 + column_offset**2 > radius**2:
                continue
            rr, cc = row + row_offset, column + column_offset
            if 0 <= rr < result.shape[0] and 0 <= cc < result.shape[1]:
                result[rr, cc] = 1


def _draw_3d_axial_marker(
    result: np.ndarray,
    center: tuple[int, int, int],
    radius: int,
) -> None:
    if radius < 0:
        raise ValueError("point radius must be nonnegative")
    offsets = [(0, 0, 0)]
    for axis in range(3):
        for distance in range(1, radius + 1):
            for direction in (-1, 1):
                offset = [0, 0, 0]
                offset[axis] = direction * distance
                offsets.append(tuple(offset))
    for offset in offsets:
        position = tuple(center[axis] + offset[axis] for axis in range(3))
        if all(0 <= position[axis] < result.shape[axis] for axis in range(3)):
            result[position] = 1


def point_marker(
    shape: tuple[int, ...],
    center: tuple[int, ...],
    *,
    radius: int | None = None,
    connectivity: int = 6,
) -> np.ndarray:
    """Rasterize the canonical 2D disk or 3D seven-voxel point marker."""
    result = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 2:
        _draw_2d_disc(result, center, radius=2 if radius is None else radius)
    elif len(shape) == 3:
        if connectivity != 6:
            raise ValueError("3D point markers use 6-connectivity")
        _draw_3d_axial_marker(result, center, radius=1 if radius is None else radius)
    else:
        raise ValueError("point markers support 2D or 3D shapes")
    return result


def region_to_points(
    mask: np.ndarray,
    *,
    method: str | None = None,
    radius: int | None = None,
    connectivity: int = 6,
) -> np.ndarray:
    """Create one study-specific point marker per connected component."""
    region = as_binary(mask)
    if region.ndim not in (2, 3):
        raise ValueError("point GT generation supports 2D or 3D masks")
    selected_method = method or (
        "distance_transform_maximum" if region.ndim == 2 else "rounded_centroid"
    )
    expected_method = (
        "distance_transform_maximum" if region.ndim == 2 else "rounded_centroid"
    )
    if selected_method != expected_method:
        raise ValueError(
            f"{region.ndim}D point GT uses method={expected_method!r}"
        )
    selected_radius = (2 if region.ndim == 2 else 1) if radius is None else int(radius)
    if selected_radius < 0:
        raise ValueError("point radius must be nonnegative")
    if region.ndim == 3 and connectivity != 6:
        raise ValueError("3D point GT uses 6-connectivity")

    result = np.zeros_like(region, dtype=np.uint8)
    labels, count = ndimage.label(region, structure=_component_structure(region.ndim))
    for index in range(1, count + 1):
        component = labels == index
        if region.ndim == 2:
            _draw_2d_disc(
                result,
                _distance_transform_center(component),
                radius=selected_radius,
            )
        else:
            _draw_3d_axial_marker(
                result,
                _centroid(component),
                radius=selected_radius,
            )
    return result


def build_gt_representations(
    region_mask: np.ndarray,
    *,
    point_method: str | None = None,
    point_radius: int | None = None,
    point_connectivity: int = 6,
) -> dict[str, np.ndarray]:
    region = as_binary(region_mask).astype(np.uint8)
    return {
        "region": region,
        "boundary": region_to_boundary(region),
        "bbox": region_to_bounding_boxes(region),
        "point": region_to_points(
            region,
            method=point_method,
            radius=point_radius,
            connectivity=point_connectivity,
        ),
    }
