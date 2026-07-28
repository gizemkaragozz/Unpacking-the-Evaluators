import numpy as np

from xai_alignment.sensitivity import convergence_onset


def test_convergence_onset_uses_contiguous_final_plateau():
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    values = np.array([0.0, 0.4, 0.92, 0.95, 0.94])
    result = convergence_onset(thresholds, values, smooth_sigma=None, tolerance_fraction=0.05)
    assert result["has_stable_region"] is True
    assert result["stability_onset"] == 0.3
    assert np.isclose(result["stable_fraction"], 0.5)


def test_single_final_point_is_not_a_stable_region():
    result = convergence_onset(
        np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.5, 1.0]), smooth_sigma=None
    )
    assert result["has_stable_region"] is False
    assert result["stable_fraction"] == 0.0

