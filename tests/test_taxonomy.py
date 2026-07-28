import numpy as np

from xai_alignment.taxonomy import (
    boundary_mass_ratio,
    local_spatial_entropy,
)


def test_local_entropy_distinguishes_uniform_from_concentrated_mass():
    uniform = np.ones((8, 8))
    concentrated = np.zeros((8, 8))
    concentrated[0, 0] = 1
    assert local_spatial_entropy(uniform, patch_size=8, threshold=1.0) > 0.99
    assert local_spatial_entropy(concentrated, patch_size=8, threshold=1.0) == 0.0


def test_local_entropy_includes_inactive_patches_in_final_mean():
    attribution = np.zeros((2, 4), dtype=float)
    attribution[:, :2] = 1.0
    result = local_spatial_entropy(
        attribution,
        patch_size=(2, 2),
        threshold=0.50,
    )
    assert np.isclose(result, 0.5)


def test_boundary_mass_ratio_is_bounded():
    attribution = np.ones((10, 10))
    result = boundary_mass_ratio(attribution, threshold=0.50)
    assert 0 <= result <= 1


def test_boundary_mass_ratio_uses_connectivity_one_inner_boundary():
    attribution = np.zeros((5, 5), dtype=float)
    attribution[1:4, 1:4] = 1.0
    result = boundary_mass_ratio(attribution, threshold=9 / 25)
    assert np.isclose(result, 8 / 9)
