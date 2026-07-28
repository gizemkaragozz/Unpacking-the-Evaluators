"""MosMedData acquisition, loading, model construction, training, and restoration."""

from __future__ import annotations

import math
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .kaggle_data import download_kaggle_dataset


KAGGLE_DATASET = "mathurinache/mosmeddata-chest-ct-scans-with-covid19"
VOLUME_SIZE = (128, 128, 64)


def download_dataset(target_dir: str | Path, *, force: bool = False) -> Path:
    """Download the public MosMedData Kaggle mirror."""
    return download_kaggle_dataset(KAGGLE_DATASET, target_dir, force=force)


def _training_imports():
    try:
        import torch
        import torch.nn as nn
        from monai.data import DataLoader, ImageDataset
        from monai.networks.nets import DenseNet
        from monai.transforms import (
            Compose,
            EnsureChannelFirst,
            Rand3DElastic,
            RandGaussianNoise,
            RandRotate90,
            Resize,
            ScaleIntensity,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise ImportError(
            "MosMedData preparation requires `pip install -e .[study]`."
        ) from error
    return (
        torch,
        nn,
        DataLoader,
        ImageDataset,
        DenseNet,
        Compose,
        EnsureChannelFirst,
        Rand3DElastic,
        RandGaussianNoise,
        RandRotate90,
        Resize,
        ScaleIntensity,
        train_test_split,
    )


def _find_studies_root(root: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("studies")
        if path.is_dir() and any(child.name.startswith("CT-") for child in path.iterdir())
    ]
    if not candidates:
        raise FileNotFoundError(f"could not find the MosMedData studies directory under {root}")
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def _find_masks_root(root: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("masks")
        if path.is_dir() and any(path.glob("study_*_mask.nii*"))
    ]
    if not candidates:
        raise FileNotFoundError(f"could not find the MosMedData masks directory under {root}")
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def index_scans(data_root: str | Path) -> tuple[list[str], list[int], Path]:
    """Index scans and reproduce the CT-0 versus CT-1..4 binary labels."""
    root = Path(data_root)
    studies_root = _find_studies_root(root)
    paths, labels = [], []
    for class_directory in sorted(studies_root.glob("CT-*")):
        label = 0 if class_directory.name == "CT-0" else 1
        for scan in sorted(class_directory.glob("study_*.nii*")):
            paths.append(str(scan))
            labels.append(label)
    if not paths:
        raise ValueError(f"no MosMedData scans found under {studies_root}")
    return paths, labels, _find_masks_root(root)


def split_with_masks(
    scan_paths: list[str],
    scan_labels: list[int],
    mask_root: str | Path,
    *,
    validation_fraction: float = 0.30,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Place every mask-annotated scan in validation, then fill to 30%."""
    *_imports, train_test_split = _training_imports()
    mask_directory = Path(mask_root)
    annotated_ids = {
        path.name.replace("_mask.nii.gz", "").replace("_mask.nii", "")
        for path in mask_directory.glob("study_*_mask.nii*")
    }
    annotated_paths, annotated_labels = [], []
    remaining_paths, remaining_labels = [], []
    for path, label in zip(scan_paths, scan_labels):
        study_id = Path(path).name.replace(".nii.gz", "").replace(".nii", "")
        if study_id in annotated_ids:
            annotated_paths.append(path)
            annotated_labels.append(label)
        else:
            remaining_paths.append(path)
            remaining_labels.append(label)

    desired_validation_size = int(math.ceil(len(scan_paths) * validation_fraction))
    additional = max(0, desired_validation_size - len(annotated_paths))
    if additional:
        (
            train_paths,
            extra_validation_paths,
            train_labels,
            extra_validation_labels,
        ) = train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=additional,
            random_state=seed,
            stratify=remaining_labels if len(set(remaining_labels)) > 1 else None,
        )
    else:
        train_paths, train_labels = remaining_paths, remaining_labels
        extra_validation_paths, extra_validation_labels = [], []
    return (
        list(train_paths),
        annotated_paths + list(extra_validation_paths),
        list(train_labels),
        annotated_labels + list(extra_validation_labels),
    )


def save_split(
    path: str | Path,
    train_paths: list[str],
    validation_paths: list[str],
    train_labels: list[int],
    validation_labels: list[int],
) -> Path:
    """Save a reusable split manifest without embedding a repository path."""
    table = pd.concat(
        [
            pd.DataFrame(
                {"path": train_paths, "label": train_labels, "split": "train"}
            ),
            pd.DataFrame(
                {
                    "path": validation_paths,
                    "label": validation_labels,
                    "split": "validation",
                }
            ),
        ],
        ignore_index=True,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def load_split(path: str | Path):
    """Restore the train/validation path and label lists from a split manifest."""
    table = pd.read_csv(path)
    train = table[table["split"].eq("train")]
    validation = table[table["split"].isin(["validation", "val"])]
    return (
        train["path"].tolist(),
        validation["path"].tolist(),
        train["label"].astype(int).tolist(),
        validation["label"].astype(int).tolist(),
    )


def mask_path_for_scan(scan_path: str | Path, mask_root: str | Path) -> Path:
    """Resolve the Kaggle annotation corresponding to one MosMed scan."""
    scan_name = Path(scan_path).name.replace(".nii.gz", "").replace(".nii", "")
    mask_directory = Path(mask_root)
    candidates = [
        mask_directory / f"{scan_name}_mask.nii",
        mask_directory / f"{scan_name}_mask.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no annotation mask found for {scan_name}")


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Binarize and nearest-neighbor resize a MosMed annotation to 128×128×64."""
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError(
            "MosMedData mask loading requires `pip install -e .[study]`."
        ) from error
    (
        _torch,
        _nn,
        _DataLoader,
        _ImageDataset,
        _DenseNet,
        _Compose,
        _EnsureChannelFirst,
        _Rand3DElastic,
        _RandGaussianNoise,
        _RandRotate90,
        Resize,
        _ScaleIntensity,
        _split,
    ) = _training_imports()
    mask = (nib.load(str(mask_path)).get_fdata() > 0).astype(np.float32)
    resized = Resize(VOLUME_SIZE, mode="nearest")(mask[None])
    return (np.squeeze(np.asarray(resized)) > 0.5).astype(np.uint8)


def _transforms():
    (
        _torch,
        _nn,
        _DataLoader,
        _ImageDataset,
        _DenseNet,
        Compose,
        EnsureChannelFirst,
        Rand3DElastic,
        RandGaussianNoise,
        RandRotate90,
        Resize,
        ScaleIntensity,
        _split,
    ) = _training_imports()
    train_transform = Compose(
        [
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(VOLUME_SIZE),
            RandRotate90(),
            Rand3DElastic(
                prob=0.2,
                sigma_range=(2, 4),
                magnitude_range=(20, 50),
            ),
            RandGaussianNoise(prob=0.2),
        ]
    )
    validation_transform = Compose(
        [
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(VOLUME_SIZE),
        ]
    )
    return train_transform, validation_transform


def create_dataloaders(
    data_root: str | Path,
    *,
    split_manifest: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    validation_fraction: float = 0.30,
    seed: int = 42,
):
    """Create the study loaders while retaining each source scan path."""
    (
        _torch,
        _nn,
        DataLoader,
        ImageDataset,
        _DenseNet,
        _Compose,
        _EnsureChannelFirst,
        _Rand3DElastic,
        _RandGaussianNoise,
        _RandRotate90,
        _Resize,
        _ScaleIntensity,
        _split,
    ) = _training_imports()
    scan_paths, scan_labels, mask_root = index_scans(data_root)
    if split_manifest is None:
        split = split_with_masks(
            scan_paths,
            scan_labels,
            mask_root,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    else:
        split = load_split(split_manifest)
    train_paths, validation_paths, train_labels, validation_labels = split
    train_transform, validation_transform = _transforms()

    class ImagePathDataset(ImageDataset):
        def __init__(self, image_files, labels, transform):
            super().__init__(image_files=image_files, labels=labels, transform=None)
            self.study_transform = transform

        def __getitem__(self, index):
            image = self.loader(self.image_files[index])
            path = self.image_files[index]
            if (
                image is None
                or np.prod(image.shape) == 0
                or any(size == 0 for size in image.shape)
            ):
                return self.__getitem__((index + 1) % len(self.image_files))
            return self.study_transform(image), self.labels[index], path

    train_dataset = ImagePathDataset(
        train_paths, train_labels, transform=train_transform
    )
    validation_dataset = ImagePathDataset(
        validation_paths,
        validation_labels,
        transform=validation_transform,
    )
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )
    return train_dataset, validation_dataset, train_loader, validation_loader, mask_root


def build_model(*, in_features: int = 1018, num_classes: int = 2):
    """Construct the MONAI DenseNet and custom classification head in the notebook."""
    (
        _torch,
        nn,
        _DataLoader,
        _ImageDataset,
        DenseNet,
        *_transforms_and_split,
    ) = _training_imports()
    model = DenseNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        init_features=16,
    )
    model.class_layers = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_model(
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    split_manifest: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
):
    """Train the MosMed architecture and save a raw state dictionary.

    The executed alignment path restores the study checkpoint with ``load_model``.
    This training route uses the same preprocessing, annotation-aware split,
    architecture, and checkpoint format.
    """
    (
        torch,
        nn,
        _DataLoader,
        _ImageDataset,
        _DenseNet,
        *_transforms_and_split,
    ) = _training_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (
        _train_dataset,
        _validation_dataset,
        train_loader,
        validation_loader,
        _mask_root,
    ) = create_dataloaders(
        data_root,
        split_manifest=split_manifest,
        batch_size=batch_size,
        seed=seed,
    )
    model = build_model().to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    output = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets, _paths in train_loader:
            inputs = inputs.to(selected_device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=selected_device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for inputs, targets, _paths in validation_loader:
                inputs = inputs.to(selected_device)
                targets = torch.as_tensor(
                    targets, dtype=torch.long, device=selected_device
                )
                losses.append(float(criterion(model(inputs), targets)))
        validation_loss = float(np.mean(losses))
        print(f"epoch={epoch:03d} validation_loss={validation_loss:.6f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), output)
    return load_model(output, device=selected_device)


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    in_features: int = 1018,
    num_classes: int = 2,
):
    """Restore the raw MONAI DenseNet state dictionary used by the study."""
    torch, _nn, _DataLoader, _ImageDataset, _DenseNet, *_rest = _training_imports()
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_features=in_features, num_classes=num_classes)
    state = torch.load(Path(checkpoint_path), map_location=selected_device)
    model.load_state_dict(state)
    model.to(selected_device).eval()
    return model
