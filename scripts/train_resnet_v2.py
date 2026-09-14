"""V2 improved ResNet-50 multi-label classification — augmentation, cosine LR, class weighting, best-model selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("crack", "spalling", "honeycombing_rock_pocket", "exposed_rebar", "rust_staining", "efflorescence_leaching")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def split_rows(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted({row["group_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    validation_groups: set[str] = set()
    count = 0
    for group in groups:
        validation_groups.add(group)
        count += sum(row["group_id"] == group for row in rows)
        if count >= max(1, round(len(rows) * fraction)):
            break
    train = [row for row in rows if row["group_id"] not in validation_groups]
    validation = [row for row in rows if row["group_id"] in validation_groups]
    if not train or not validation:
        raise ValueError("Cannot produce non-empty group-safe splits")
    return train, validation


def labels(row: dict[str, str]) -> torch.Tensor:
    values = {label.strip() for label in row["canonical_labels"].replace("|", ";").split(";") if label.strip()}
    return torch.tensor([float(label in values) for label in CLASSES], dtype=torch.float32)


def compute_class_weights(rows: list[dict[str, str]]) -> torch.Tensor:
    """Compute inverse-frequency class weights for the training set."""
    counts = torch.zeros(len(CLASSES))
    for row in rows:
        target = labels(row)
        counts += target
    # Avoid division by zero — set minimum count to 1
    counts = torch.clamp(counts, min=1.0)
    # Inverse frequency weighting, normalized so mean weight = 1
    weights = len(rows) / (len(CLASSES) * counts)
    # Clamp extreme weights to prevent instability
    weights = torch.clamp(weights, min=0.1, max=10.0)
    return weights


class ManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], transform: object) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(ROOT / row["output_image"]).convert("RGB")
        return self.transform(image), labels(row)


def build_train_transform() -> transforms.Compose:
    """Training augmentation: random crop, flip, color jitter, rotation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform() -> transforms.Compose:
    """Validation transform: deterministic resize + center crop."""
    return ResNet50_Weights.DEFAULT.transforms()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/v1_classification_manifest.csv")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/resnet/resnet50-11ad3fa6.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v2_improved/resnet50_v2")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")

    manifest = (ROOT / args.manifest) if not args.manifest.is_absolute() else args.manifest
    rows = [row for row in read_rows(manifest) if row.get("eligibility") and row["geometry_qc"] == "passed"]
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)

    # Compute class weights from training set
    class_weights = compute_class_weights(train_rows).to("cuda")
    print(f"Class weights: {dict(zip(CLASSES, class_weights.tolist()))}")

    # V2: use training augmentation
    train_transform = build_train_transform()
    val_transform = build_val_transform()

    train_loader = DataLoader(
        ManifestDataset(train_rows, train_transform),
        batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        ManifestDataset(val_rows, val_transform),
        batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    model = resnet50(weights=None)
    state = torch.load(
        (ROOT / args.weights) if not args.weights.is_absolute() else args.weights,
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to("cuda", dtype=torch.float32)

    # V2: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # V2: Cosine annealing LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # V2: Weighted BCE loss for class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    args.output.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "v2_improved",
        "classes": list(CLASSES),
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "train_groups": len({row["group_id"] for row in train_rows}),
        "validation_groups": len({row["group_id"] for row in val_rows}),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "class_weights": dict(zip(CLASSES, class_weights.tolist())),
        "improvements": [
            "Training augmentation (RandomResizedCrop, flip, rotation, color jitter)",
            "Class-weighted BCEWithLogitsLoss",
            "Cosine annealing LR schedule",
            "AdamW with weight decay",
            "Best-model selection by validation loss",
            "Early stopping with patience",
            "30 epochs (up from 10)",
        ],
    }
    (args.output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # === Training ===
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = images.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                predictions = model(images)
                loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to("cuda", non_blocking=True)
                targets = targets.to("cuda", non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite val loss at epoch {epoch + 1}")
                val_loss += loss.item() * len(images)

        scheduler.step()

        avg_train_loss = train_loss / len(train_rows)
        avg_val_loss = val_loss / len(val_rows)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": float(epoch + 1),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr,
            "epoch_seconds": epoch_time,
            "best_epoch": best_epoch + 1,
        }
        history.append(record)
        print(json.dumps(record))

        # === Best model selection ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve_count = 0
            torch.save(
                {"model": model.float().state_dict(), "classes": CLASSES, "epoch": epoch + 1,
                 "val_loss": avg_val_loss, "history": history, "seed": args.seed},
                args.output / "resnet50_comparison.pt",
            )
            print(f"  -> New best model saved (val_loss={avg_val_loss:.6f})")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"  -> Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time / 3600:.1f}h)")
    print(f"Best model: epoch {best_epoch + 1}, val_loss={best_val_loss:.6f}")

    # Also save the last checkpoint
    torch.save(
        {"model": model.float().state_dict(), "classes": CLASSES, "epoch": len(history),
         "val_loss": avg_val_loss, "history": history, "seed": args.seed},
        args.output / "resnet50_last.pt",
    )
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
