from pathlib import Path

import yaml


EXPECTED_THRESHOLDS = [
    0.01,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    0.90,
    1.00,
]


def test_submitted_hard_threshold_set_is_used_in_both_dimensions():
    root = Path(__file__).resolve().parents[1]
    for filename in ("2d.yaml", "3d.yaml"):
        with (root / "configs" / filename).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["alignment"]["thresholds"] == EXPECTED_THRESHOLDS
        assert config["analysis"]["hard_threshold"] in EXPECTED_THRESHOLDS
