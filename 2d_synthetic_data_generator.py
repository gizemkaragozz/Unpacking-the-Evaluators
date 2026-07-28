"""
BALANCED Synthetic Shapes Dataset Generator (instance-aware, multi-label)

Goal: generate a balanced multi-label classification dataset containing circles,
squares, and triangles.
=> balanced label-set combinations + balanced per-class prevalence.

Key design:
- We sample from 7 label sets uniformly:
    {C}, {S}, {T}, {C,S}, {C,T}, {S,T}, {C,S,T}
- If n is a multiple of 7, we get EXACT balance across combinations and nearly exact per-class balance
  (exact per-class balance also holds because each class appears in 4/7 of combos, evenly).
- Each object has its own instance mask/boundary/bbox/points + class label.
- Union GT is still saved for "any-object" evaluation.

Outputs per sample:
- images/<id>.png
- masks/<id>.png                      (union mask)
- boundaries/<id>.png                 (union boundary) [optional]
- diffuse_gt/<id>.png                 (union diffuse) [optional]
- instance_masks/<id>_obj{k}.png      (instance mask)
- instance_boundaries/<id>_obj{k}.png (instance boundary) [optional]
- bboxes/<id>.txt                     (union bbox + per-object bboxes) [optional]
- points/<id>.txt                     (union points + per-object points) [optional]
- meta.jsonl                          (paths + per-object GT + labels + sampled label_set)

Requirements:
- Python + numpy + opencv-python
"""

import os, json, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


# -----------------------------
# Geometry helpers
# -----------------------------
@dataclass
class ShapeSpec:
    shape: str                  # "circle" | "square" | "triangle"
    cx: int
    cy: int
    size: int                   # radius for circle, half-side for square, approx radius for triangle
    angle_deg: float            # rotation (used for square/triangle)
    color: Tuple[int, int, int] # BGR


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _rotated_square_points(cx: int, cy: int, half: int, angle_deg: float) -> np.ndarray:
    pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
    theta = np.deg2rad(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]], dtype=np.float32)
    pts = (pts @ R.T) + np.array([cx, cy], dtype=np.float32)
    return pts.astype(np.int32)


def _rotated_triangle_points(cx: int, cy: int, r: int, angle_deg: float) -> np.ndarray:
    pts = []
    base = np.deg2rad(angle_deg)
    for k in range(3):
        a = base + k * 2 * math.pi / 3
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.array(pts, dtype=np.int32)


def draw_shape_mask(h: int, w: int, spec: ShapeSpec) -> np.ndarray:
    """Binary mask (uint8 0/255) for one object."""
    m = np.zeros((h, w), dtype=np.uint8)
    if spec.shape == "circle":
        cv2.circle(m, (spec.cx, spec.cy), spec.size, 255, thickness=-1, lineType=cv2.LINE_AA)
    elif spec.shape == "square":
        pts = _rotated_square_points(spec.cx, spec.cy, spec.size, spec.angle_deg)
        cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    elif spec.shape == "triangle":
        pts = _rotated_triangle_points(spec.cx, spec.cy, spec.size, spec.angle_deg)
        cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    else:
        raise ValueError(f"Unknown shape: {spec.shape}")
    return m


def overlaps(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.any((mask_a > 0) & (mask_b > 0)))


# -----------------------------
# GT derivations
# -----------------------------
def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(x_min, y_min, x_max, y_max) or None."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)


def mask_to_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Center of mass centroid (x, y) or None."""
    m = (mask > 0).astype(np.uint8)
    M = cv2.moments(m)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (float(cx), float(cy))


def mask_to_dt_point(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Deepest interior point via distance transform (x, y) int or None."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (int(x), int(y))


def mask_to_boundary(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological gradient boundary mask (uint8)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(dil, ero)


def make_diffuse_mask(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian-blurred soft GT (float32 in [0,1])."""
    m = mask.astype(np.float32) / 255.0
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def _write_bbox_txt(path: str, combined_bbox, object_bboxes: List[Optional[Tuple[int,int,int,int]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if combined_bbox is None:
            f.write("combined -1 -1 -1 -1\n")
        else:
            x0,y0,x1,y1 = combined_bbox
            f.write(f"combined {x0} {y0} {x1} {y1}\n")
        for j, bb in enumerate(object_bboxes):
            if bb is None:
                f.write(f"obj{j} -1 -1 -1 -1\n")
            else:
                x0,y0,x1,y1 = bb
                f.write(f"obj{j} {x0} {y0} {x1} {y1}\n")


def _write_points_txt(path: str,
                      combined_centroid, combined_dt_point,
                      obj_centroids: List[Optional[Tuple[float,float]]],
                      obj_dt_points: List[Optional[Tuple[int,int]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if combined_centroid is None:
            f.write("combined_centroid -1 -1\n")
        else:
            cx, cy = combined_centroid
            f.write(f"combined_centroid {cx:.4f} {cy:.4f}\n")

        if combined_dt_point is None:
            f.write("combined_dt -1 -1\n")
        else:
            x, y = combined_dt_point
            f.write(f"combined_dt {x} {y}\n")

        for j in range(len(obj_centroids)):
            c = obj_centroids[j]
            dt = obj_dt_points[j]
            if c is None:
                f.write(f"obj{j}_centroid -1 -1\n")
            else:
                cx, cy = c
                f.write(f"obj{j}_centroid {cx:.4f} {cy:.4f}\n")

            if dt is None:
                f.write(f"obj{j}_dt -1 -1\n")
            else:
                x, y = dt
                f.write(f"obj{j}_dt {x} {y}\n")


# -----------------------------
# Balanced label-set sampling
# -----------------------------
LABEL_SETS = [
    ("circle",),
    ("square",),
    ("triangle",),
    ("circle", "square"),
    ("circle", "triangle"),
    ("square", "triangle"),
    ("circle", "square", "triangle"),
]


def make_balanced_label_schedule(n: int, seed: int = 0, shuffle: bool = True) -> List[Tuple[str, ...]]:
    """
    If n is divisible by 7, each label set appears exactly n/7 times.
    If not divisible, it will be as balanced as possible (round-robin).
    """
    rng = random.Random(seed)
    reps = n // len(LABEL_SETS)
    rem = n % len(LABEL_SETS)

    schedule = []
    for _ in range(reps):
        schedule.extend(LABEL_SETS)

    # add remainder (balanced by taking a random subset)
    if rem > 0:
        extra = LABEL_SETS.copy()
        rng.shuffle(extra)
        schedule.extend(extra[:rem])

    if shuffle:
        rng.shuffle(schedule)

    return schedule


# -----------------------------
# Main generator
# -----------------------------
def generate_shapes_dataset_balanced(
    out_dir: str,
    n: int = 1400,                   # recommend multiple of 7 for exact combo balance
    image_size: int = 224,
    size_range: Tuple[int, int] = (24, 64),
    allow_overlap: bool = False,
    seed: int = 0,

    export_union_boundary: bool = True,
    export_instance_boundary: bool = True,
    export_diffuse_gt: bool = False, # keep off unless you need it now
    diffuse_sigma: float = 4.0,

    export_bbox_gt: bool = True,
    export_point_gt: bool = True,

    # Optionally make learning even easier:
    fixed_color: bool = True,        # white shapes on black
):
    """
    Balanced, instance-aware multi-label dataset for presence classification.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Output directories
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")  # union mask
    bnd_dir  = os.path.join(out_dir, "boundaries")
    dif_dir  = os.path.join(out_dir, "diffuse_gt")
    inst_m_dir = os.path.join(out_dir, "instance_masks")
    inst_b_dir = os.path.join(out_dir, "instance_boundaries")
    bbox_dir = os.path.join(out_dir, "bboxes")
    pts_dir  = os.path.join(out_dir, "points")

    _ensure_dir(img_dir); _ensure_dir(mask_dir); _ensure_dir(inst_m_dir)
    if export_union_boundary: _ensure_dir(bnd_dir)
    if export_instance_boundary: _ensure_dir(inst_b_dir)
    if export_diffuse_gt: _ensure_dir(dif_dir)
    if export_bbox_gt: _ensure_dir(bbox_dir)
    if export_point_gt: _ensure_dir(pts_dir)

    meta_path = os.path.join(out_dir, "meta.jsonl")

    H = W = image_size
    margin = max(size_range[1] + 6, 12)

    # exact-ish balanced schedule of label sets
    schedule = make_balanced_label_schedule(n, seed=seed, shuffle=True)

    def sample_spec(shape: str) -> ShapeSpec:
        size = rng.randint(size_range[0], size_range[1])
        cx = rng.randint(margin, W - margin)
        cy = rng.randint(margin, H - margin)
        angle = rng.uniform(0, 360)
        color = (255, 255, 255) if fixed_color else (
            rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255)
        )
        return ShapeSpec(shape=shape, cx=cx, cy=cy, size=size, angle_deg=angle, color=color)

    with open(meta_path, "w", encoding="utf-8") as f:
        for i in range(n):
            sid = f"{i:06d}"
            label_set = schedule[i]  # tuple of labels
            n_obj_target = len(label_set)

            img = np.zeros((H, W, 3), dtype=np.uint8)
            combined_mask = np.zeros((H, W), dtype=np.uint8)

            specs: List[ShapeSpec] = []
            obj_masks: List[np.ndarray] = []

            # One instance per label in the set (ensures separability)
            for shape in label_set:
                attempts = 0
                while attempts < 400:
                    attempts += 1
                    spec = sample_spec(shape)
                    m = draw_shape_mask(H, W, spec)
                    if (not allow_overlap) and overlaps(combined_mask, m):
                        continue
                    specs.append(spec)
                    obj_masks.append(m)
                    combined_mask = cv2.bitwise_or(combined_mask, m)
                    break
                else:
                    # If placement fails (rare), relax overlap for this object
                    spec = sample_spec(shape)
                    m = draw_shape_mask(H, W, spec)
                    specs.append(spec)
                    obj_masks.append(m)
                    combined_mask = cv2.bitwise_or(combined_mask, m)

            # draw image
            for spec, m in zip(specs, obj_masks):
                img[m > 0] = spec.color

            # save union image/mask
            img_path = os.path.join(img_dir, f"{sid}.png")
            msk_path = os.path.join(mask_dir, f"{sid}.png")
            cv2.imwrite(img_path, img)
            cv2.imwrite(msk_path, combined_mask)

            # union boundary
            union_bnd_path = None
            if export_union_boundary:
                boundary = mask_to_boundary(combined_mask, ksize=3)
                union_bnd_path = os.path.join(bnd_dir, f"{sid}.png")
                cv2.imwrite(union_bnd_path, boundary)

            # union diffuse
            dif_path = None
            if export_diffuse_gt:
                diffuse = make_diffuse_mask(combined_mask, sigma=diffuse_sigma)
                dif16 = (diffuse * 65535).astype(np.uint16)
                dif_path = os.path.join(dif_dir, f"{sid}.png")
                cv2.imwrite(dif_path, dif16)

            # per-object instance files + per-object GTs
            object_entries = []
            obj_bboxes = []
            obj_centroids = []
            obj_dt_points = []

            for j, (spec, om) in enumerate(zip(specs, obj_masks)):
                inst_mask_path = os.path.join(inst_m_dir, f"{sid}_obj{j}.png")
                cv2.imwrite(inst_mask_path, om)

                inst_bnd_path = None
                if export_instance_boundary:
                    ob = mask_to_boundary(om, ksize=3)
                    inst_bnd_path = os.path.join(inst_b_dir, f"{sid}_obj{j}.png")
                    cv2.imwrite(inst_bnd_path, ob)

                bb = mask_to_bbox(om)
                cc = mask_to_centroid(om)
                dt = mask_to_dt_point(om)

                obj_bboxes.append(bb)
                obj_centroids.append(cc)
                obj_dt_points.append(dt)

                object_entries.append({
                    "id": j,
                    "shape": spec.shape,
                    "cx": spec.cx, "cy": spec.cy,
                    "size": spec.size,
                    "angle_deg": spec.angle_deg,
                    "instance_mask_path": os.path.relpath(inst_mask_path, out_dir),
                    "instance_boundary_path": os.path.relpath(inst_bnd_path, out_dir) if inst_bnd_path else None,
                    "bbox": bb,
                    "centroid": cc,
                    "dt_point": dt,
                })

            # union GT summaries
            combined_bbox = mask_to_bbox(combined_mask)
            combined_centroid = mask_to_centroid(combined_mask)
            combined_dt_point = mask_to_dt_point(combined_mask)

            # export bbox + points files
            bbox_path = None
            if export_bbox_gt:
                bbox_path = os.path.join(bbox_dir, f"{sid}.txt")
                _write_bbox_txt(bbox_path, combined_bbox, obj_bboxes)

            pts_path = None
            if export_point_gt:
                pts_path = os.path.join(pts_dir, f"{sid}.txt")
                _write_points_txt(pts_path, combined_centroid, combined_dt_point, obj_centroids, obj_dt_points)

            rec = {
                "id": sid,
                "image_path": os.path.relpath(img_path, out_dir),
                "mask_path": os.path.relpath(msk_path, out_dir),

                "boundary_path": os.path.relpath(union_bnd_path, out_dir) if union_bnd_path else None,
                "diffuse_gt_path": os.path.relpath(dif_path, out_dir) if dif_path else None,

                "bbox_gt_path": os.path.relpath(bbox_path, out_dir) if bbox_path else None,
                "point_gt_path": os.path.relpath(pts_path, out_dir) if pts_path else None,

                "label_set": list(label_set),
                "n_objects": len(object_entries),
                "objects": object_entries,

                "combined_bbox": combined_bbox,
                "combined_centroid": combined_centroid,
                "combined_dt_point": combined_dt_point,

                "factors": {
                    "balanced_label_sets": True,
                    "allow_overlap": allow_overlap,
                    "image_size": image_size,
                    "size_range": list(size_range),
                    "diffuse_sigma": diffuse_sigma if export_diffuse_gt else None,
                    "seed": seed,
                }
            }

            f.write(json.dumps(rec) + "\n")

    print(f"Done. Wrote {n} BALANCED samples to: {out_dir}")
    print(f"   - meta: {meta_path}")
    print(f"   - images: {img_dir}")
    print(f"   - union masks: {mask_dir}")
    print(f"   - instance masks: {inst_m_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate balanced 2D synthetic multi-label shapes dataset."
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=1400)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_overlap", action="store_true")
    parser.add_argument("--export_diffuse_gt", action="store_true")

    args = parser.parse_args()

    generate_shapes_dataset_balanced(
        out_dir=args.out_dir,
        n=args.n,
        image_size=args.image_size,
        size_range=(24, 64),
        allow_overlap=args.allow_overlap,
        seed=args.seed,
        export_union_boundary=True,
        export_instance_boundary=True,
        export_diffuse_gt=args.export_diffuse_gt,
        export_bbox_gt=True,
        export_point_gt=True,
        fixed_color=True,
    )
