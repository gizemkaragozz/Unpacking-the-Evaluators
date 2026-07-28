"""Synthetic 3D data loading, predictive-model training, and checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


CLASSES = ("sphere", "cube", "pyramid")


def _training_imports():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise ImportError(
            "Synthetic 3D model preparation requires `pip install -e .[study]`."
        ) from error
    return torch, nn, DataLoader, Dataset


@dataclass
class TrainConfig:
    dataset_dir: str | Path
    checkpoint_path: str | Path
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    threshold: float = 0.5
    device: str | None = None
    classes: tuple[str, ...] = CLASSES


def set_seed(seed: int) -> None:
    torch, _nn, _DataLoader, _Dataset = _training_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Read metadata produced by ``3d_synthetic_data_generator.py``."""
    metadata_path = Path(dataset_dir) / "meta.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"meta.jsonl not found under {Path(dataset_dir)}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_records_stratified(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify by the complete multi-label combination, as in the notebook."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = "|".join(sorted(record["label_set"]))
        groups.setdefault(key, []).append(record)

    generator = random.Random(seed)
    train_records, validation_records, test_records = [], [], []
    for group in groups.values():
        shuffled = list(group)
        generator.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_fraction))
        n_validation = int(round(len(shuffled) * validation_fraction))
        test_records.extend(shuffled[:n_test])
        validation_records.extend(shuffled[n_test : n_test + n_validation])
        train_records.extend(shuffled[n_test + n_validation :])
    generator.shuffle(train_records)
    generator.shuffle(validation_records)
    generator.shuffle(test_records)
    return train_records, validation_records, test_records


def create_dataloaders(config: TrainConfig):
    """Build loaders over the generated NumPy volumes and multi-hot labels."""
    torch, _nn, DataLoader, Dataset = _training_imports()
    root = Path(config.dataset_dir)
    rows = split_records_stratified(
        load_records(root),
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )

    class SyntheticShapes3DDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            volume = np.load(root / record["volume_path"]).astype(np.float32)
            if volume.max() > 0:
                volume = volume / volume.max()
            target = np.array(
                [record["multi_hot_label"][name] for name in config.classes],
                dtype=np.float32,
            )
            return {
                "id": str(record["id"]),
                "image": torch.from_numpy(volume).unsqueeze(0).float(),
                "label": torch.from_numpy(target).float(),
            }

    datasets = tuple(SyntheticShapes3DDataset(records) for records in rows)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = (
        DataLoader(
            datasets[0],
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        DataLoader(
            datasets[1],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        DataLoader(
            datasets[2],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
    )
    return (*datasets, *loaders, *rows)


def build_model(*, in_channels: int = 1, num_classes: int = 3):
    """Construct the submitted ``Basic3DCNN`` architecture."""
    _torch, nn, _DataLoader, _Dataset = _training_imports()

    class Basic3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(output_size=1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

    return Basic3DCNN()


def _run_epoch(model, loader, criterion, device, threshold, optimizer=None):
    torch, _nn, _DataLoader, _Dataset = _training_imports()
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_exact = total_label_accuracy = total_samples = 0.0
    for batch in loader:
        inputs = batch["image"].to(device)
        targets = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        predictions = (torch.sigmoid(logits) >= threshold).float()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_exact += (
            (predictions == targets).all(dim=1).float().mean().item() * batch_size
        )
        total_label_accuracy += (
            (predictions == targets).float().mean().item() * batch_size
        )
        total_samples += batch_size
    return {
        "loss": total_loss / total_samples,
        "exact_match_accuracy": total_exact / total_samples,
        "per_label_accuracy": total_label_accuracy / total_samples,
    }


def train_model(config: TrainConfig):
    """Train with Adam and ReduceLROnPlateau and save the best validation loss."""
    torch, nn, _DataLoader, _Dataset = _training_imports()
    set_seed(config.seed)
    selected_device = config.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    (
        _train_dataset,
        _validation_dataset,
        _test_dataset,
        train_loader,
        validation_loader,
        _test_loader,
        *_records,
    ) = create_dataloaders(config)
    model = build_model(num_classes=len(config.classes)).to(selected_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    output = Path(config.checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            selected_device,
            config.threshold,
            optimizer,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            criterion,
            selected_device,
            config.threshold,
        )
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": config.__dict__,
                    "epoch": epoch,
                    "val_loss": best_validation_loss,
                },
                output,
            )
        scheduler.step(validation_metrics["loss"])
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    num_classes: int = 3,
):
    """Restore the submitted Synthetic 3D checkpoint dictionary."""
    torch, _nn, _DataLoader, _Dataset = _training_imports()
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=selected_device,
        weights_only=False,
    )
    model = build_model(in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device).eval()
    return model
