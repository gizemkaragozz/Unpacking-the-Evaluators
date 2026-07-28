import numpy as np

from xai_alignment.ground_truth import build_gt_representations


def test_gt_representations_are_binary_and_nonempty():
    region = np.zeros((12, 12), dtype=np.uint8)
    region[2:6, 2:6] = 1
    region[8:10, 8:11] = 1
    maps = build_gt_representations(region)

    assert set(maps) == {"region", "boundary", "bbox", "point"}
    assert all(set(np.unique(mask)) <= {0, 1} for mask in maps.values())
    assert maps["point"].sum() == 26
    assert maps["bbox"].sum() == region.sum()
    assert maps["boundary"].sum() > 0


def test_distance_transform_points_fall_inside_region():
    region = np.zeros((9, 9), dtype=np.uint8)
    region[1:8, 2:7] = 1
    point = build_gt_representations(region)["point"].astype(bool)
    assert point.sum() == 13


def test_3d_centroid_uses_seven_voxel_axial_marker():
    region = np.zeros((9, 9, 9), dtype=np.uint8)
    region[2:7, 2:7, 2:7] = 1
    point = build_gt_representations(region)["point"]
    assert point.sum() == 7
    assert point[4, 4, 4] == 1
