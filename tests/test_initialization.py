import inspect
from pathlib import Path
import tempfile

from xai_alignment.init_brisc_2d import (
    KAGGLE_DATASET as BRISC_DATASET,
    _plane_from_filename,
)
from xai_alignment.init_mosmed_3d import (
    KAGGLE_DATASET as MOSMED_DATASET,
    load_split,
    save_split,
)
from xai_alignment.init_synthetic_2d import split_records
from xai_alignment.init_synthetic_3d import split_records_stratified
from xai_alignment.kaggle_data import download_kaggle_dataset


def test_public_kaggle_identifiers_and_anonymous_interface():
    assert BRISC_DATASET == "briscdataset/brisc2025"
    assert MOSMED_DATASET == "mathurinache/mosmeddata-chest-ct-scans-with-covid19"
    parameters = inspect.signature(download_kaggle_dataset).parameters
    assert "username" not in parameters
    assert "key" not in parameters


def test_synthetic_2d_split_is_deterministic_80_10_10():
    records = [{"id": str(index)} for index in range(100)]
    first = split_records(records, seed=42)
    second = split_records(records, seed=42)
    assert [len(part) for part in first] == [80, 10, 10]
    assert first == second


def test_synthetic_3d_split_preserves_labelset_strata():
    records = [
        {"id": f"a{index}", "label_set": ["sphere"]}
        for index in range(20)
    ] + [
        {"id": f"b{index}", "label_set": ["cube", "pyramid"]}
        for index in range(20)
    ]
    train, validation, test = split_records_stratified(records, seed=42)
    assert [len(train), len(validation), len(test)] == [28, 6, 6]
    assert {tuple(row["label_set"]) for row in validation} == {
        ("sphere",),
        ("cube", "pyramid"),
    }


def test_brisc_plane_parsing_and_mosmed_split_roundtrip():
    assert _plane_from_filename("brisc2025_test_00001_gl_ax_t1.jpg") == "axial"
    assert _plane_from_filename("brisc2025_test_00001_gl_co_t1.jpg") == "coronal"
    assert _plane_from_filename("brisc2025_test_00001_gl_sa_t1.jpg") == "sagittal"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "split.csv"
        save_split(path, ["train.nii"], ["validation.nii"], [0], [1])
        assert load_split(path) == (
            ["train.nii"],
            ["validation.nii"],
            [0],
            [1],
        )
