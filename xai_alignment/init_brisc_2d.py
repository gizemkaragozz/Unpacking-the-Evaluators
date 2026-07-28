"""BRISC 2025 acquisition, data loading, model training, and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
import copy
import re

import cv2
import numpy as np
import pandas as pd

from .kaggle_data import download_kaggle_dataset


KAGGLE_DATASET = "briscdataset/brisc2025"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3,
    "no_tumor": 3,
    "non_tumor": 3,
}


def download_dataset(target_dir: str | Path, *, force: bool = False) -> Path:
    """Download the public BRISC 2025 Kaggle dataset."""
    return download_kaggle_dataset(KAGGLE_DATASET, target_dir, force=force)


def _training_imports():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise ImportError(
            "BRISC preparation requires `pip install -e .[study]`."
        ) from error
    return (
        torch,
        nn,
        DataLoader,
        Dataset,
        models,
        transforms,
        accuracy_score,
        f1_score,
        train_test_split,
    )


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_task_root(root: Path, task_name: str) -> Path:
    normalized = task_name.lower().replace("_", "")
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and path.name.lower().replace("_", "").replace(" ", "") == normalized
    ]
    if not candidates:
        raise FileNotFoundError(f"could not find {task_name!r} below {root}")
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def _infer_class(path: Path) -> str | None:
    for part in (item.lower() for item in path.parts):
        if part in LABEL_TO_INDEX:
            return part
    return None


def _split_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return None


def _plane_from_filename(path: str | Path) -> str:
    parts = Path(path).stem.lower().split("_")
    if "ax" in parts:
        return "axial"
    if "co" in parts:
        return "coronal"
    if "sa" in parts:
        return "sagittal"
    return "unknown"


def build_classification_table(data_root: str | Path) -> pd.DataFrame:
    """Index the classification task using the directory labels and supplied split."""
    root = _find_task_root(Path(data_root), "classification_task")
    rows = [
        {
            "image_path": str(path),
            "image_name": path.name,
            "image_stem": path.stem,
            "label": _infer_class(path),
            "split": _split_from_path(path),
            "plane": _plane_from_filename(path),
        }
        for path in _image_files(root)
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"no classification images found under {root}")
    if table["label"].isna().any() or table["split"].isna().any():
        raise ValueError("unrecognized BRISC class or split directory")
    return table


def build_segmentation_table(data_root: str | Path) -> pd.DataFrame:
    """Index the image/mask pairs from the public segmentation task."""
    root = _find_task_root(Path(data_root), "segmentation_task")
    files = _image_files(root)
    masks = {}
    images = []
    for path in files:
        parents = {part.lower() for part in path.parts}
        if "masks" in parents or re.search(r"(?:_mask|_seg|_label|_gt)$", path.stem):
            masks[path.stem.lower()] = path
        else:
            images.append(path)
    rows = [
        {
            "image_path": str(path),
            "mask_path": str(masks[path.stem.lower()])
            if path.stem.lower() in masks
            else None,
            "image_name": path.name,
            "image_stem": path.stem,
            "split": _split_from_path(path),
            "plane": _plane_from_filename(path),
        }
        for path in images
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"no segmentation images found under {root}")
    table["has_mask"] = table["mask_path"].notna()
    return table


def load_region_mask(mask_path: str | Path, *, threshold: int = 127) -> np.ndarray:
    """Load a BRISC binary segmentation mask using the study threshold."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return (mask > threshold).astype(np.uint8)


def create_validation_split(
    classification_table: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create the stratified validation subset used by the study."""
    *_unused, train_test_split = _training_imports()
    table = classification_table.copy()
    train_rows = table[table["split"].eq("train")]
    _training_indices, validation_indices = train_test_split(
        train_rows.index,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_rows["label"],
    )
    table.loc[validation_indices, "split"] = "val"
    return table


def _transforms(image_size: int = 224):
    (
        _torch,
        _nn,
        _DataLoader,
        _Dataset,
        _models,
        transforms,
        _accuracy,
        _f1,
        _split,
    ) = _training_imports()
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def create_dataloaders(
    classification_table: pd.DataFrame,
    *,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
):
    """Create the submitted BRISC train, validation, and test loaders."""
    (
        torch,
        _nn,
        DataLoader,
        Dataset,
        _models,
        _transforms_module,
        _accuracy,
        _f1,
        _split,
    ) = _training_imports()
    train_transform, evaluation_transform = _transforms(image_size)

    class BRISCClassificationDataset(Dataset):
        def __init__(self, rows: pd.DataFrame, transform):
            self.rows = rows.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows.iloc[index]
            image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(row["image_path"])
            metadata = {
                "image_path": row["image_path"],
                "plane": row["plane"],
                "label_name": row["label"],
            }
            return (
                self.transform(image),
                torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long),
                metadata,
            )

    def collate(batch):
        images, labels, metadata = zip(*batch)
        return torch.stack(images), torch.stack(labels), list(metadata)

    datasets = (
        BRISCClassificationDataset(
            classification_table[classification_table["split"].eq("train")],
            train_transform,
        ),
        BRISCClassificationDataset(
            classification_table[classification_table["split"].eq("val")],
            evaluation_transform,
        ),
        BRISCClassificationDataset(
            classification_table[classification_table["split"].eq("test")],
            evaluation_transform,
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    loaders = (
        DataLoader(
            datasets[0],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
        ),
        DataLoader(
            datasets[1],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
        ),
        DataLoader(
            datasets[2],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
        ),
    )
    return (*datasets, *loaders)


def build_model(*, num_classes: int = 4, pretrained: bool = True):
    """Construct the ImageNet-initialized DenseNet-121 used for BRISC."""
    (
        _torch,
        nn,
        _DataLoader,
        _Dataset,
        models,
        _transforms,
        _accuracy,
        _f1,
        _split,
    ) = _training_imports()
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _run_epoch(model, loader, criterion, optimizer, device, accuracy_score, f1_score):
    import torch

    training = optimizer is not None
    model.train() if training else model.eval()
    running_loss = 0.0
    labels, predictions = [], []
    for images, target, _metadata in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * images.size(0)
        labels.extend(target.detach().cpu().numpy())
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def train_model(
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.15,
    image_size: int = 224,
    seed: int = 42,
    device: str | None = None,
):
    """Train DenseNet-121 and save the state with best validation macro-F1."""
    (
        torch,
        nn,
        _DataLoader,
        _Dataset,
        _models,
        _transforms,
        accuracy_score,
        f1_score,
        _split,
    ) = _training_imports()
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    table = create_validation_split(
        build_classification_table(data_root),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    (
        _train_dataset,
        _validation_dataset,
        _test_dataset,
        train_loader,
        validation_loader,
        _test_loader,
    ) = create_dataloaders(
        table,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    model = build_model(num_classes=4, pretrained=True).to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_metric = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    output = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            selected_device,
            accuracy_score,
            f1_score,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            criterion,
            None,
            selected_device,
            accuracy_score,
            f1_score,
        )
        print(
            f"epoch={epoch:02d} train_macro_f1={train_metrics['macro_f1']:.6f} "
            f"validation_macro_f1={validation_metrics['macro_f1']:.6f}"
        )
        if validation_metrics["macro_f1"] > best_metric:
            best_metric = validation_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output)
    model.load_state_dict(best_state)
    return model.eval()


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    pretrained: bool = False,
):
    """Restore the raw DenseNet-121 state dictionary saved by the study."""
    (
        torch,
        _nn,
        _DataLoader,
        _Dataset,
        _models,
        _transforms,
        _accuracy,
        _f1,
        _split,
    ) = _training_imports()
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=4, pretrained=pretrained)
    state = torch.load(Path(checkpoint_path), map_location=selected_device)
    model.load_state_dict(state)
    model.to(selected_device).eval()
    return model
