"""
BALANCED 3D Synthetic Shapes Dataset Generator (multi-label, class-wise, slice-wise GT)

Goal
-----
Create a true 3D synthetic dataset for multi-label classification and alignment studies
with three classes:
- sphere
- cube
- pyramid

Key properties
--------------
- Balanced over 7 label-set combinations:
    {sphere}, {cube}, {pyramid},
    {sphere,cube}, {sphere,pyramid}, {cube,pyramid},
    {sphere,cube,pyramid}
- One instance per present class
- True 3D volumes saved as .npy
- Class-wise 3D GT
- GT for each present class:
    - mask
    - boundary
    - bbox_2d
    - centroid_2d
    - dt_point_2d
- Union GT also saved



Output structure
----------------
out_dir/
    volumes/
        000000.npy

    union_masks/
        000000.npy
    union_boundaries/
        000000.npy                         [optional]
    union_diffuse_gt/
        000000.npy                         [optional]

    class_masks/
        000000_sphere.npy
        000000_cube.npy
        000000_pyramid.npy
    class_boundaries/
        000000_sphere.npy                  [optional]
        ...

    slice_class_masks/
        000000_sphere_z012.npy
        000000_cube_z018.npy
        ...
    slice_class_boundaries/
        000000_sphere_z012.npy             [optional]
        ...

    instance_masks/
        000000_obj0.npy
        ...
    instance_boundaries/
        000000_obj0.npy                    [optional]

    meta.jsonl

Dependencies
------------
- numpy
- scipy
"""

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)


# ============================================================
# Constants
# ============================================================
CLASSES = ["sphere", "cube", "pyramid"]

LABEL_SETS = [
    ("sphere",),
    ("cube",),
    ("pyramid",),
    ("sphere", "cube"),
    ("sphere", "pyramid"),
    ("cube", "pyramid"),
    ("sphere", "cube", "pyramid"),
]


# ============================================================
# Basic utilities
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npy(path: str, arr: np.ndarray) -> None:
    np.save(path, arr)


def relpath_or_none(path: Optional[str], root: str) -> Optional[str]:
    return None if path is None else os.path.relpath(path, root)


def voxel_to_physical(
    point_xyz: Optional[Tuple[float, float, float]],
    spacing_x: float,
    spacing_y: float,
    spacing_z: float,
) -> Optional[Tuple[float, float, float]]:
    if point_xyz is None:
        return None
    x, y, z = point_xyz
    return (
        float(x * spacing_x),
        float(y * spacing_y),
        float(z * spacing_z),
    )


def bbox_3d_voxel_to_physical(
    bbox_xyzxyz: Optional[Tuple[int, int, int, int, int, int]],
    spacing_x: float,
    spacing_y: float,
    spacing_z: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    if bbox_xyzxyz is None:
        return None
    x0, y0, z0, x1, y1, z1 = bbox_xyzxyz
    return (
        float(x0 * spacing_x),
        float(y0 * spacing_y),
        float(z0 * spacing_z),
        float(x1 * spacing_x),
        float(y1 * spacing_y),
        float(z1 * spacing_z),
    )


def bbox_2d_voxel_to_physical(
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    spacing_x: float,
    spacing_y: float,
) -> Optional[Tuple[float, float, float, float]]:
    if bbox_xyxy is None:
        return None
    x0, y0, x1, y1 = bbox_xyxy
    return (
        float(x0 * spacing_x),
        float(y0 * spacing_y),
        float(x1 * spacing_x),
        float(y1 * spacing_y),
    )


# ============================================================
# Dataclass
# ============================================================
@dataclass
class Shape3DSpec:
    shape: str                   # sphere | cube | pyramid
    cx: int
    cy: int
    cz: int
    size: int                    # sphere radius / cube half-side / pyramid half-height
    angles_deg: Tuple[float, float, float]
    intensity: int
    invert_pyramid: bool = False


# ============================================================
# Rotation helpers
# ============================================================
def rotation_matrix_xyz(angles_deg: Tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = [math.radians(a) for a in angles_deg]

    rx = np.array([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax),  math.cos(ax)],
    ], dtype=np.float32)

    ry = np.array([
        [ math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)],
    ], dtype=np.float32)

    rz = np.array([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az),  math.cos(az), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return rz @ ry @ rx


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([
        [math.cos(a), -math.sin(a), 0],
        [math.sin(a),  math.cos(a), 0],
        [0, 0, 1],
    ], dtype=np.float32)


def build_local_grid(
    z0: int, z1: int,
    y0: int, y1: int,
    x0: int, x1: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1),
        np.arange(y0, y1),
        np.arange(x0, x1),
        indexing="ij",
    )
    return zz, yy, xx


# ============================================================
# Shape rasterization
# ============================================================
def draw_sphere_mask(volume_shape: Tuple[int, int, int], spec: Shape3DSpec) -> np.ndarray:
    d, h, w = volume_shape
    r = spec.size

    x0 = max(0, spec.cx - r - 1)
    x1 = min(w, spec.cx + r + 2)
    y0 = max(0, spec.cy - r - 1)
    y1 = min(h, spec.cy + r + 2)
    z0 = max(0, spec.cz - r - 1)
    z1 = min(d, spec.cz + r + 2)

    zz, yy, xx = build_local_grid(z0, z1, y0, y1, x0, x1)
    dist2 = (xx - spec.cx) ** 2 + (yy - spec.cy) ** 2 + (zz - spec.cz) ** 2

    out = np.zeros((d, h, w), dtype=np.uint8)
    out[z0:z1, y0:y1, x0:x1] = (dist2 <= r ** 2).astype(np.uint8) * 255
    return out


def draw_cube_mask(volume_shape: Tuple[int, int, int], spec: Shape3DSpec) -> np.ndarray:
    d, h, w = volume_shape
    half = spec.size
    radius = int(math.ceil(math.sqrt(3) * half)) + 2

    x0 = max(0, spec.cx - radius)
    x1 = min(w, spec.cx + radius + 1)
    y0 = max(0, spec.cy - radius)
    y1 = min(h, spec.cy + radius + 1)
    z0 = max(0, spec.cz - radius)
    z1 = min(d, spec.cz + radius + 1)

    zz, yy, xx = build_local_grid(z0, z1, y0, y1, x0, x1)
    pts = np.stack([xx - spec.cx, yy - spec.cy, zz - spec.cz], axis=-1).reshape(-1, 3).astype(np.float32)

    rmat = rotation_matrix_xyz(spec.angles_deg)
    pts_local = pts @ rmat.T

    inside = (
        (np.abs(pts_local[:, 0]) <= half) &
        (np.abs(pts_local[:, 1]) <= half) &
        (np.abs(pts_local[:, 2]) <= half)
    )

    out = np.zeros((d, h, w), dtype=np.uint8)
    out_local = inside.reshape(z1 - z0, y1 - y0, x1 - x0).astype(np.uint8) * 255
    out[z0:z1, y0:y1, x0:x1] = out_local
    return out


def draw_pyramid_mask(volume_shape: Tuple[int, int, int], spec: Shape3DSpec) -> np.ndarray:
    d, h, w = volume_shape
    half_h = spec.size
    base_half = spec.size

    # Use a spherical radius for the bounding box to account for all possible rotations
    radius = int(math.ceil(math.sqrt(base_half**2 + base_half**2 + half_h**2))) + 2

    x0, x1 = max(0, spec.cx - radius), min(w, spec.cx + radius + 1)
    y0, y1 = max(0, spec.cy - radius), min(h, spec.cy + radius + 1)
    z0, z1 = max(0, spec.cz - radius), min(d, spec.cz + radius + 1)

    zz, yy, xx = build_local_grid(z0, z1, y0, y1, x0, x1)
    pts = np.stack([xx - spec.cx, yy - spec.cy, zz - spec.cz], axis=-1).reshape(-1, 3).astype(np.float32)

    # Apply the full 3D rotation matrix.
    rmat = rotation_matrix_xyz(spec.angles_deg)
    pts_local = pts @ rmat.T

    x, y, z = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]

    if spec.invert_pyramid:
        z = -z

    # Geometric logic remains the same, but now operates in fully rotated space
    valid_z = (z >= -half_h) & (z <= half_h)
    t = (z + half_h) / (2.0 * half_h)
    t = np.clip(t, 0.0, 1.0)
    allowed_half_side = base_half * (1.0 - t)

    inside = valid_z & (np.abs(x) <= allowed_half_side) & (np.abs(y) <= allowed_half_side)

    out = np.zeros((d, h, w), dtype=np.uint8)
    out[z0:z1, y0:y1, x0:x1] = inside.reshape(z1 - z0, y1 - y0, x1 - x0).astype(np.uint8) * 255
    return out


def draw_shape_mask_3d(volume_shape: Tuple[int, int, int], spec: Shape3DSpec) -> np.ndarray:
    if spec.shape == "sphere":
        return draw_sphere_mask(volume_shape, spec)
    if spec.shape == "cube":
        return draw_cube_mask(volume_shape, spec)
    if spec.shape == "pyramid":
        return draw_pyramid_mask(volume_shape, spec)
    raise ValueError(f"Unknown shape: {spec.shape}")


def overlaps(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.any((mask_a > 0) & (mask_b > 0)))


# ============================================================
# 3D GT derivations
# ============================================================
def mask_to_bbox_3d(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return (int(xmin), int(ymin), int(zmin), int(xmax), int(ymax), int(zmax))


def mask_to_centroid_3d(mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    zc, yc, xc = coords.mean(axis=0)
    return (float(xc), float(yc), float(zc))


def mask_to_dt_point_3d(mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    if mask.sum() == 0:
        return None
    dist = distance_transform_edt(mask > 0)
    z, y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (int(x), int(y), int(z))


def mask_to_boundary_3d(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = mask > 0
    dil = binary_dilation(m, iterations=iterations)
    ero = binary_erosion(m, iterations=iterations)
    boundary = np.logical_and(dil, np.logical_not(ero))
    return boundary.astype(np.uint8) * 255


def make_diffuse_mask_3d(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    m = (mask > 0).astype(np.float32)
    out = gaussian_filter(m, sigma=sigma)
    out = out - out.min()
    if out.max() > 0:
        out = out / out.max()
    return out.astype(np.float32)


# ============================================================
# 2D slice-wise GT derivations
# ============================================================
def mask_to_bbox_2d(mask_slice: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_slice > 0)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def mask_to_centroid_2d(mask_slice: np.ndarray) -> Optional[Tuple[float, float]]:
    coords = np.argwhere(mask_slice > 0)
    if coords.size == 0:
        return None
    yc, xc = coords.mean(axis=0)
    return (float(xc), float(yc))


def mask_to_dt_point_2d(mask_slice: np.ndarray) -> Optional[Tuple[int, int]]:
    if mask_slice.sum() == 0:
        return None
    dist = distance_transform_edt(mask_slice > 0)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (int(x), int(y))


def mask_to_boundary_2d(mask_slice: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = mask_slice > 0
    dil = binary_dilation(m, iterations=iterations)
    ero = binary_erosion(m, iterations=iterations)
    boundary = np.logical_and(dil, np.logical_not(ero))
    return boundary.astype(np.uint8) * 255


# ============================================================
# Balanced schedule
# ============================================================
def make_balanced_label_schedule(n: int, seed: int = 0, shuffle: bool = True) -> List[Tuple[str, ...]]:
    rng = random.Random(seed)
    reps = n // len(LABEL_SETS)
    rem = n % len(LABEL_SETS)

    schedule: List[Tuple[str, ...]] = []
    for _ in range(reps):
        schedule.extend(LABEL_SETS)

    if rem > 0:
        extra = LABEL_SETS.copy()
        rng.shuffle(extra)
        schedule.extend(extra[:rem])

    if shuffle:
        rng.shuffle(schedule)

    return schedule


# ============================================================
# Sampling
# ============================================================
def sample_shape_spec_3d(
    rng: random.Random,
    shape: str,
    volume_size: int,
    size_range: Tuple[int, int],
    margin: int,
    fixed_intensity: bool = True,
) -> Shape3DSpec:
    size = rng.randint(size_range[0], size_range[1])

    cx = rng.randint(margin, volume_size - margin - 1)
    cy = rng.randint(margin, volume_size - margin - 1)
    cz = rng.randint(margin, volume_size - margin - 1)

    if shape == "sphere":
        angles_deg = (0.0, 0.0, 0.0)
    elif shape == "cube":
        angles_deg = (
            rng.uniform(0, 360),
            rng.uniform(0, 360),
            rng.uniform(0, 360),
        )
    elif shape == "pyramid":
        angles_deg = (
            0.0,
            0.0,
            rng.uniform(0, 360),
        )
    else:
        raise ValueError(shape)

    intensity = 255 if fixed_intensity else rng.randint(220, 255)
    invert_pyramid = bool(rng.randint(0, 1)) if shape == "pyramid" else False

    return Shape3DSpec(
        shape=shape,
        cx=cx,
        cy=cy,
        cz=cz,
        size=size,
        angles_deg=angles_deg,
        intensity=intensity,
        invert_pyramid=invert_pyramid,
    )


# ============================================================
# Main generator
# ============================================================
def generate_shapes_dataset_3d_balanced(
    out_dir: str,
    n: int = 1400,
    volume_size: int = 64,
    size_range: Tuple[int, int] = (7, 14),
    allow_overlap: bool = False,
    seed: int = 42,

    export_union_boundary: bool = True,
    export_class_boundaries: bool = True,
    export_instance_boundaries: bool = True,
    export_slice_boundaries: bool = True,

    export_diffuse_gt: bool = False,
    diffuse_sigma: float = 2.0,

    fixed_intensity: bool = True,

    spacing_x: float = 1.0,
    spacing_y: float = 1.0,
    spacing_z: float = 1.0,
    slice_thickness: float = 1.0,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    # dirs
    vol_dir = os.path.join(out_dir, "volumes")

    union_mask_dir = os.path.join(out_dir, "union_masks")
    union_bnd_dir = os.path.join(out_dir, "union_boundaries")
    union_dif_dir = os.path.join(out_dir, "union_diffuse_gt")

    class_mask_dir = os.path.join(out_dir, "class_masks")
    class_bnd_dir = os.path.join(out_dir, "class_boundaries")
    class_dif_dir = os.path.join(out_dir, "class_diffuse_gt")

    slice_class_mask_dir = os.path.join(out_dir, "slice_class_masks")
    slice_class_bnd_dir = os.path.join(out_dir, "slice_class_boundaries")

    inst_mask_dir = os.path.join(out_dir, "instance_masks")
    inst_bnd_dir = os.path.join(out_dir, "instance_boundaries")

    ensure_dir(vol_dir)
    ensure_dir(union_mask_dir)
    ensure_dir(class_mask_dir)
    ensure_dir(slice_class_mask_dir)
    ensure_dir(inst_mask_dir)

    if export_union_boundary:
        ensure_dir(union_bnd_dir)
    if export_class_boundaries:
        ensure_dir(class_bnd_dir)
    if export_slice_boundaries:
        ensure_dir(slice_class_bnd_dir)
    if export_instance_boundaries:
        ensure_dir(inst_bnd_dir)
    if export_diffuse_gt:
        ensure_dir(union_dif_dir)
        ensure_dir(class_dif_dir)

    meta_path = os.path.join(out_dir, "meta.jsonl")

    d = h = w = volume_size
    volume_shape = (d, h, w)

    margin = max(size_range[1] + 4, 8)

    schedule = make_balanced_label_schedule(n=n, seed=seed, shuffle=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        for i in range(n):
            sid = f"{i:06d}"
            label_set = schedule[i]

            volume = np.zeros(volume_shape, dtype=np.uint8)
            union_mask = np.zeros(volume_shape, dtype=np.uint8)

            specs: List[Shape3DSpec] = []
            obj_masks: List[np.ndarray] = []

            # one instance per present class
            for cls in label_set:
                placed = False
                for _ in range(400):
                    spec = sample_shape_spec_3d(
                        rng=rng,
                        shape=cls,
                        volume_size=volume_size,
                        size_range=size_range,
                        margin=margin,
                        fixed_intensity=fixed_intensity,
                    )
                    m = draw_shape_mask_3d(volume_shape, spec)

                    if (not allow_overlap) and overlaps(union_mask, m):
                        continue

                    specs.append(spec)
                    obj_masks.append(m)
                    union_mask = np.maximum(union_mask, m)
                    placed = True
                    break

                if not placed:
                    spec = sample_shape_spec_3d(
                        rng=rng,
                        shape=cls,
                        volume_size=volume_size,
                        size_range=size_range,
                        margin=margin,
                        fixed_intensity=fixed_intensity,
                    )
                    m = draw_shape_mask_3d(volume_shape, spec)
                    specs.append(spec)
                    obj_masks.append(m)
                    union_mask = np.maximum(union_mask, m)

            # render volume
            for spec, m in zip(specs, obj_masks):
                volume[m > 0] = spec.intensity

            # save volume + union
            vol_path = os.path.join(vol_dir, f"{sid}.npy")
            union_mask_path = os.path.join(union_mask_dir, f"{sid}.npy")
            save_npy(vol_path, volume)
            save_npy(union_mask_path, union_mask)

            union_boundary_path = None
            if export_union_boundary:
                union_boundary = mask_to_boundary_3d(union_mask, iterations=1)
                union_boundary_path = os.path.join(union_bnd_dir, f"{sid}.npy")
                save_npy(union_boundary_path, union_boundary)

            union_diffuse_path = None
            if export_diffuse_gt:
                union_diffuse = make_diffuse_mask_3d(union_mask, sigma=diffuse_sigma)
                union_diffuse_path = os.path.join(union_dif_dir, f"{sid}.npy")
                save_npy(union_diffuse_path, union_diffuse)

            # instance entries
            object_entries: List[Dict] = []
            for j, (spec, om) in enumerate(zip(specs, obj_masks)):
                inst_mask_path = os.path.join(inst_mask_dir, f"{sid}_obj{j}.npy")
                save_npy(inst_mask_path, om)

                inst_boundary_path = None
                if export_instance_boundaries:
                    inst_boundary = mask_to_boundary_3d(om, iterations=1)
                    inst_boundary_path = os.path.join(inst_bnd_dir, f"{sid}_obj{j}.npy")
                    save_npy(inst_boundary_path, inst_boundary)

                bbox_voxel = mask_to_bbox_3d(om)
                centroid_voxel = mask_to_centroid_3d(om)
                dt_point_voxel = mask_to_dt_point_3d(om)

                object_entries.append({
                    "id": j,
                    "shape": spec.shape,
                    "center_voxel": {"x": spec.cx, "y": spec.cy, "z": spec.cz},
                    "center_physical_mm": {
                        "x": float(spec.cx * spacing_x),
                        "y": float(spec.cy * spacing_y),
                        "z": float(spec.cz * spacing_z),
                    },
                    "size": spec.size,
                    "angles_deg": {
                        "x": float(spec.angles_deg[0]),
                        "y": float(spec.angles_deg[1]),
                        "z": float(spec.angles_deg[2]),
                    },
                    "invert_pyramid": bool(spec.invert_pyramid) if spec.shape == "pyramid" else None,
                    "instance_mask_path": relpath_or_none(inst_mask_path, out_dir),
                    "instance_boundary_path": relpath_or_none(inst_boundary_path, out_dir),
                    "bbox_3d_voxel": bbox_voxel,
                    "bbox_3d_physical_mm": bbox_3d_voxel_to_physical(
                        bbox_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "centroid_3d_voxel": centroid_voxel,
                    "centroid_3d_physical_mm": voxel_to_physical(
                        centroid_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "dt_point_3d_voxel": dt_point_voxel,
                    "dt_point_3d_physical_mm": voxel_to_physical(
                        dt_point_voxel, spacing_x, spacing_y, spacing_z
                    ),
                })

            # class GT + slice GT
            class_gt: Dict[str, Dict] = {}
            for cls in CLASSES:
                cls_present = cls in label_set

                if cls_present:
                    idx = label_set.index(cls)
                    cls_mask = obj_masks[idx]
                else:
                    cls_mask = np.zeros(volume_shape, dtype=np.uint8)

                cls_mask_path = os.path.join(class_mask_dir, f"{sid}_{cls}.npy")
                save_npy(cls_mask_path, cls_mask)

                cls_boundary_path = None
                if export_class_boundaries:
                    cls_boundary = mask_to_boundary_3d(cls_mask, iterations=1)
                    cls_boundary_path = os.path.join(class_bnd_dir, f"{sid}_{cls}.npy")
                    save_npy(cls_boundary_path, cls_boundary)

                cls_diffuse_path = None
                if export_diffuse_gt:
                    cls_diffuse = make_diffuse_mask_3d(cls_mask, sigma=diffuse_sigma)
                    cls_diffuse_path = os.path.join(class_dif_dir, f"{sid}_{cls}.npy")
                    save_npy(cls_diffuse_path, cls_diffuse)

                bbox_3d_voxel = mask_to_bbox_3d(cls_mask)
                centroid_3d_voxel = mask_to_centroid_3d(cls_mask)
                dt_point_3d_voxel = mask_to_dt_point_3d(cls_mask)

                # slice-wise GT only on present slices
                slice_gt: Dict[str, Dict] = {}
                present_slices = np.where(np.any(cls_mask > 0, axis=(1, 2)))[0].tolist()

                for z in present_slices:
                    mask_slice = cls_mask[z]
                    slice_mask_path = os.path.join(slice_class_mask_dir, f"{sid}_{cls}_z{z:03d}.npy")
                    save_npy(slice_mask_path, mask_slice)

                    slice_boundary_path = None
                    if export_slice_boundaries:
                        boundary_slice = mask_to_boundary_2d(mask_slice, iterations=1)
                        slice_boundary_path = os.path.join(slice_class_bnd_dir, f"{sid}_{cls}_z{z:03d}.npy")
                        save_npy(slice_boundary_path, boundary_slice)

                    bbox_2d_voxel = mask_to_bbox_2d(mask_slice)
                    centroid_2d_voxel = mask_to_centroid_2d(mask_slice)
                    dt_point_2d_voxel = mask_to_dt_point_2d(mask_slice)

                    slice_gt[f"{z:03d}"] = {
                        "z_index": int(z),
                        "mask_path": relpath_or_none(slice_mask_path, out_dir),
                        "boundary_path": relpath_or_none(slice_boundary_path, out_dir),
                        "bbox_2d_voxel": bbox_2d_voxel,
                        "bbox_2d_physical_mm": bbox_2d_voxel_to_physical(
                            bbox_2d_voxel, spacing_x, spacing_y
                        ),
                        "centroid_2d_voxel": centroid_2d_voxel,
                        "centroid_2d_physical_mm": (
                            None if centroid_2d_voxel is None else
                            (float(centroid_2d_voxel[0] * spacing_x), float(centroid_2d_voxel[1] * spacing_y))
                        ),
                        "dt_point_2d_voxel": dt_point_2d_voxel,
                        "dt_point_2d_physical_mm": (
                            None if dt_point_2d_voxel is None else
                            (float(dt_point_2d_voxel[0] * spacing_x), float(dt_point_2d_voxel[1] * spacing_y))
                        ),
                    }

                class_gt[cls] = {
                    "present": bool(cls_present),
                    "mask_path": relpath_or_none(cls_mask_path, out_dir),
                    "boundary_path": relpath_or_none(cls_boundary_path, out_dir),
                    "diffuse_gt_path": relpath_or_none(cls_diffuse_path, out_dir),
                    "bbox_3d_voxel": bbox_3d_voxel,
                    "bbox_3d_physical_mm": bbox_3d_voxel_to_physical(
                        bbox_3d_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "centroid_3d_voxel": centroid_3d_voxel,
                    "centroid_3d_physical_mm": voxel_to_physical(
                        centroid_3d_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "dt_point_3d_voxel": dt_point_3d_voxel,
                    "dt_point_3d_physical_mm": voxel_to_physical(
                        dt_point_3d_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "slice_gt": slice_gt,
                    "n_present_slices": int(len(present_slices)),
                }

            # union summaries
            union_bbox_3d_voxel = mask_to_bbox_3d(union_mask)
            union_centroid_3d_voxel = mask_to_centroid_3d(union_mask)
            union_dt_point_3d_voxel = mask_to_dt_point_3d(union_mask)

            rec = {
                "id": sid,
                "task_type": "multi_label_classification",
                "classes": CLASSES,
                "label_set": list(label_set),
                "multi_hot_label": {
                    "sphere": int("sphere" in label_set),
                    "cube": int("cube" in label_set),
                    "pyramid": int("pyramid" in label_set),
                },

                "volume_path": relpath_or_none(vol_path, out_dir),

                "volume_shape": {
                    "z": int(d),
                    "y": int(h),
                    "x": int(w),
                },
                "voxel_spacing_mm": {
                    "x": float(spacing_x),
                    "y": float(spacing_y),
                    "z": float(spacing_z),
                },
                "slice_thickness_mm": float(slice_thickness),
                "spacing_between_slices_mm": float(spacing_z),
                "physical_extent_mm": {
                    "x": float(w * spacing_x),
                    "y": float(h * spacing_y),
                    "z": float(d * spacing_z),
                },

                "n_objects": int(len(object_entries)),
                "objects": object_entries,

                "class_gt": class_gt,

                "union_gt": {
                    "mask_path": relpath_or_none(union_mask_path, out_dir),
                    "boundary_path": relpath_or_none(union_boundary_path, out_dir),
                    "diffuse_gt_path": relpath_or_none(union_diffuse_path, out_dir),
                    "bbox_3d_voxel": union_bbox_3d_voxel,
                    "bbox_3d_physical_mm": bbox_3d_voxel_to_physical(
                        union_bbox_3d_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "centroid_3d_voxel": union_centroid_3d_voxel,
                    "centroid_3d_physical_mm": voxel_to_physical(
                        union_centroid_3d_voxel, spacing_x, spacing_y, spacing_z
                    ),
                    "dt_point_3d_voxel": union_dt_point_3d_voxel,
                    "dt_point_3d_physical_mm": voxel_to_physical(
                        union_dt_point_3d_voxel, spacing_x, spacing_y, spacing_z
                    ),
                },

                "generation_factors": {
                    "balanced_label_sets": True,
                    "allow_overlap": bool(allow_overlap),
                    "fixed_intensity": bool(fixed_intensity),
                    "volume_size": int(volume_size),
                    "size_range": [int(size_range[0]), int(size_range[1])],
                    "diffuse_sigma": float(diffuse_sigma) if export_diffuse_gt else None,
                    "seed": int(seed),
                },
            }

            f.write(json.dumps(rec) + "\n")

    print(f"Done. Wrote {n} balanced 3D samples to: {out_dir}")
    print(f"  meta: {meta_path}")
    print(f"  volumes: {vol_dir}")
    print(f"  union masks: {union_mask_dir}")
    print(f"  class masks: {class_mask_dir}")
    print(f"  slice class masks: {slice_class_mask_dir}")
    print(f"  instance masks: {inst_mask_dir}")


# ============================================================
# Example run
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate balanced 3D synthetic multi-label shapes dataset."
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=1400)
    parser.add_argument("--volume_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_overlap", action="store_true")
    parser.add_argument("--export_diffuse_gt", action="store_true")

    args = parser.parse_args()

    generate_shapes_dataset_3d_balanced(
        out_dir=args.out_dir,
        n=args.n,
        volume_size=args.volume_size,
        size_range=(7, 14),
        allow_overlap=args.allow_overlap,
        seed=args.seed,

        export_union_boundary=True,
        export_class_boundaries=True,
        export_instance_boundaries=True,
        export_slice_boundaries=True,

        export_diffuse_gt=args.export_diffuse_gt,
        diffuse_sigma=2.0,

        fixed_intensity=True,

        spacing_x=1.0,
        spacing_y=1.0,
        spacing_z=1.0,
        slice_thickness=1.0,
    )
