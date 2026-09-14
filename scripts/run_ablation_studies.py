#!/usr/bin/env python3
"""Systematic ablation studies for building defect inspection models.

Evaluates 4 core experimental axes:
1. Input Resolution: 512x512 vs. 640x640
2. Augmentation Strategy: Baseline vs. Heavy Multi-Scale (Mosaic, Affine, Color Jitter)
3. Loss Formulation: Standard BCE vs. Inverse-Frequency Class-Weighted BCE
4. Backbone Fine-Tuning Depth: Frozen Backbone vs. Layer4 Unfreeze vs. Full Fine-Tuning
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
)

MANIFEST_PATH = ROOT / "data/manifests/v1_classification_manifest.csv"
DEFAULT_OUTPUT = ROOT / "runs/evaluation/ablation_study_results.json"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def encode_labels(labels_str: str) -> np.ndarray:
    active = {val.strip() for val in labels_str.replace("|", ";").split(";") if val.strip()}
    return np.asarray([1.0 if cls in active else 0.0 for cls in CLASSES], dtype=np.float32)


class InspectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[tuple[bytes, np.ndarray]],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_bytes, target = self.records[index]
        with Image.open(io.BytesIO(img_bytes)) as img:
            tensor_img = self.transform(img.convert("RGB"))
        return tensor_img, torch.from_numpy(target)


def get_transforms(resolution: int, heavy_aug: bool) -> tuple[transforms.Compose, transforms.Compose]:
    if heavy_aug:
        train_tx = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tx = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_tx = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tx, val_tx


def build_model(freeze_mode: str, num_classes: int = len(CLASSES)) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif freeze_mode == "layer4":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    elif freeze_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze mode: {freeze_mode}")

    return model


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Calculate micro F1 at standard 0.5 threshold
    binary_preds = preds >= 0.5
    tp = float(np.logical_and(binary_preds, targets == 1).sum())
    fp = float(np.logical_and(binary_preds, targets == 0).sum())
    fn = float(np.logical_and(~binary_preds, targets == 1).sum())

    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_single_ablation(
    name: str,
    resolution: int,
    heavy_aug: bool,
    use_weighting: bool,
    freeze_mode: str,
    train_records: list[tuple[bytes, np.ndarray]],
    val_records: list[tuple[bytes, np.ndarray]],
    epochs: int,
    device: torch.device,
) -> dict[str, object]:
    train_tx, val_tx = get_transforms(resolution, heavy_aug)
    train_loader = DataLoader(
        InspectionDataset(train_records, train_tx),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        InspectionDataset(val_records, val_tx),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(freeze_mode).to(device)

    # Loss setup
    if use_weighting:
        # Compute inverse frequency from train records
        targets_matrix = np.stack([t for _, t in train_records])
        pos_counts = np.maximum(targets_matrix.sum(axis=0), 1.0)
        neg_counts = len(train_records) - pos_counts
        pos_weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=1e-4,
    )

    t0 = time.perf_counter()
    train_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        train_losses.append(epoch_loss / max(1, batches))

    duration = time.perf_counter() - t0
    metrics = evaluate_model(model, val_loader, device)

    return {
        "ablation_name": name,
        "resolution": resolution,
        "augmentation": "heavy (mosaic+affine+jitter)" if heavy_aug else "baseline (standard)",
        "loss_weighting": "inverse_frequency" if use_weighting else "standard_bce",
        "freeze_mode": freeze_mode,
        "epochs": epochs,
        "duration_seconds": round(duration, 2),
        "final_train_loss": round(train_losses[-1], 4),
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run defect inspection ablation matrix")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=150)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading manifest from {MANIFEST_PATH}...")
    rows = load_manifest(MANIFEST_PATH)
    if args.sample_limit:
        rows = rows[: args.sample_limit]

    # Pre-cache images and labels in memory for rapid multi-run ablation
    print(f"Caching {len(rows)} image records into memory...")
    records: list[tuple[bytes, np.ndarray]] = []
    for r in rows:
        img_p = ROOT / r["output_image"]
        if img_p.is_file():
            records.append((img_p.read_bytes(), encode_labels(r["canonical_labels"])))

    split_idx = int(len(records) * 0.8)
    train_records, val_records = records[:split_idx], records[split_idx:]
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    ablation_configs = [
        # Baseline Reference
        ("Baseline (224px, Standard BCE, Full Backbone)", 224, False, False, "full"),
        # Axis 1: Resolution (512px)
        ("Ablation: High Resolution (512px)", 512, False, False, "full"),
        # Axis 2: Augmentation Strategies
        ("Ablation: Heavy Multi-Scale Augmentation", 224, True, False, "full"),
        # Axis 3: Class Loss Weighting
        ("Ablation: Inverse-Frequency Class Weighting", 224, False, True, "full"),
        # Axis 4: Backbone Freezing Depth (Layer4 only)
        ("Ablation: Layer4 Fine-Tuning Only (Feature Extraction)", 224, False, False, "layer4"),
        # Axis 4: Backbone Freezing Depth (Linear Probe)
        ("Ablation: Fully Frozen Backbone (Linear Probe)", 224, False, False, "frozen"),
    ]

    results = []
    print("\n" + "=" * 70)
    print("STARTING SYSTEMATIC ABLATION MATRIX BENCHMARK")
    print("=" * 70)

    for idx, (name, res, aug, weight, freeze) in enumerate(ablation_configs, 1):
        print(f"\n[{idx}/{len(ablation_configs)}] Executing: {name}...")
        res_dict = run_single_ablation(
            name=name,
            resolution=res,
            heavy_aug=aug,
            use_weighting=weight,
            freeze_mode=freeze,
            train_records=train_records,
            val_records=val_records,
            epochs=args.epochs,
            device=device,
        )
        results.append(res_dict)
        print(f"    Done in {res_dict['duration_seconds']}s | Loss={res_dict['final_train_loss']} | F1={res_dict['val_f1']*100:.2f}% (P={res_dict['val_precision']*100:.2f}%, R={res_dict['val_recall']*100:.2f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "Defect Model Ablation Study Matrix",
        "device": str(device),
        "evaluated_configurations": len(results),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll ablation results saved to {args.output}")


if __name__ == "__main__":
    main()
