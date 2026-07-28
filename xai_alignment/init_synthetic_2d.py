"""Synthetic 2D data loading, predictive-model training, and checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np


SHAPES = ("circle", "square", "triangle")
SHAPE_TO_INDEX = {name: index for index, name in enumerate(SHAPES)}


def _training_imports():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
        import torchvision.transforms as transforms
        from PIL import Image
        from sklearn.metrics import f1_score
    except ImportError as error:
        raise ImportError(
            "Synthetic 2D model preparation requires `pip install -e .[study]`."
        ) from error
    return torch, nn, functional, DataLoader, Dataset, transforms, Image, f1_score


def load_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Read the ``meta.jsonl`` produced by ``2d_synthetic_data_generator.py``."""
    metadata_path = Path(dataset_dir) / "meta.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"meta.jsonl not found under {Path(dataset_dir)}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the submitted random 80/10/10 record split."""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("invalid train or validation ratio")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below one")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_validation = int(len(shuffled) * validation_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_validation],
        shuffled[n_train + n_validation :],
    )


def _target(record: dict[str, Any]) -> np.ndarray:
    target = np.zeros(len(SHAPES), dtype=np.float32)
    labels = record.get("label_set")
    if labels is None:
        labels = [item.get("shape") for item in record.get("objects", [])]
    for label in labels:
        if label in SHAPE_TO_INDEX:
            target[SHAPE_TO_INDEX[label]] = 1.0
    return target


def build_model(num_classes: int = 3):
    """Construct the CAM-compatible ``SimpleCNN_GAP_BN`` used in the study."""
    _torch, nn, _functional, _DataLoader, _Dataset, _transforms, _Image, _f1 = (
        _training_imports()
    )

    def block(input_channels: int, output_channels: int):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class SimpleCNN_GAP_BN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                block(3, 32),
                block(32, 64),
                block(64, 128),
                block(128, 256),
            )
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(256, num_classes)

        def forward(self, inputs):
            return self.head(self.gap(self.features(inputs)).flatten(1))

    return SimpleCNN_GAP_BN()


def create_dataloaders(
    dataset_dir: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 64,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    num_workers: int = 2,
):
    """Create the study's deterministic-resize train, validation, and test loaders."""
    torch, _nn, _functional, DataLoader, Dataset, transforms, Image, _f1 = (
        _training_imports()
    )
    root = Path(dataset_dir)
    train_records, validation_records, test_records = split_records(
        load_records(root),
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    class ShapesMultiLabelDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            record = self.rows[index]
            image = Image.open(root / record["image_path"]).convert("RGB")
            return (
                transform(image),
                torch.tensor(_target(record), dtype=torch.float32),
                str(record["id"]),
            )

    datasets = tuple(
        ShapesMultiLabelDataset(rows)
        for rows in (train_records, validation_records, test_records)
    )
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)
    loaders = (
        DataLoader(
            datasets[0],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets[1],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets[2],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )
    return (*datasets, *loaders)


def _evaluate(model, loader, device, functional):
    import torch

    model.eval()
    total_loss = 0.0
    targets, probabilities = [], []
    with torch.no_grad():
        for inputs, target, _sample_id in loader:
            inputs, target = inputs.to(device), target.to(device)
            logits = model(inputs)
            loss = functional.binary_cross_entropy_with_logits(logits, target)
            total_loss += loss.item() * inputs.size(0)
            targets.append(target.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def _tune_thresholds(targets: np.ndarray, probabilities: np.ndarray, f1_score):
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for class_index in range(targets.shape[1]):
        best_score = -1.0
        for threshold in np.linspace(0.05, 0.95, 19):
            predictions = (probabilities[:, class_index] >= threshold).astype(int)
            score = f1_score(
                targets[:, class_index], predictions, zero_division=0
            )
            if score > best_score:
                best_score = score
                thresholds[class_index] = threshold
    return thresholds


def train_model(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    seed: int = 42,
    device: str | None = None,
):
    """Train and save the best validation per-label-accuracy checkpoint."""
    torch, _nn, functional, _DataLoader, _Dataset, _transforms, _Image, f1_score = (
        _training_imports()
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (
        _train_dataset,
        _validation_dataset,
        _test_dataset,
        train_loader,
        validation_loader,
        _test_loader,
    ) = create_dataloaders(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    model = build_model(len(SHAPES)).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    output = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets, _sample_id in train_loader:
            inputs, targets = inputs.to(selected_device), targets.to(selected_device)
            optimizer.zero_grad()
            loss = functional.binary_cross_entropy_with_logits(
                model(inputs), targets
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

        validation_loss, targets, probabilities = _evaluate(
            model, validation_loader, selected_device, functional
        )
        thresholds = _tune_thresholds(targets, probabilities, f1_score)
        predictions = probabilities >= thresholds.reshape(1, -1)
        per_label_accuracy = float((predictions == targets).mean())
        print(
            f"epoch={epoch:02d} validation_loss={validation_loss:.6f} "
            f"per_label_accuracy={per_label_accuracy:.6f}"
        )
        if per_label_accuracy > best_accuracy:
            best_accuracy = per_label_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "shapes": list(SHAPES),
                    "thresholds": thresholds,
                },
                output,
            )
    return load_model(output, device=selected_device)


def load_model(checkpoint_path: str | Path, *, device: str | None = None):
    """Restore the submitted Synthetic 2D checkpoint structure."""
    torch, _nn, _functional, _DataLoader, _Dataset, _transforms, _Image, _f1 = (
        _training_imports()
    )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=selected_device, weights_only=False
    )
    model = build_model(len(checkpoint.get("shapes", SHAPES)))
    model.load_state_dict(checkpoint["model"])
    model.to(selected_device).eval()
    return model, checkpoint.get("thresholds")
