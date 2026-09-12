"""Fine-tune a ResNet-50 multi-label classification baseline from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
        raise ValueError("The manifest cannot produce non-empty group-safe train and validation splits")
    return train, validation


def labels(row: dict[str, str]) -> torch.Tensor:
    values = {label.strip() for label in row["canonical_labels"].replace("|", ";").split(";") if label.strip()}
    return torch.tensor([float(label in values) for label in CLASSES], dtype=torch.float32)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/v1_classification_manifest.csv")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/resnet/resnet50-11ad3fa6.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/comparison/resnet50")
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA autocast for higher throughput.")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    manifest = (ROOT / args.manifest) if not args.manifest.is_absolute() else args.manifest
    rows = [row for row in read_rows(manifest) if row.get("eligibility") and row["geometry_qc"] == "passed"]
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    transform = ResNet50_Weights.DEFAULT.transforms()
    train_loader = DataLoader(ManifestDataset(train_rows, transform), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(ManifestDataset(val_rows, transform), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = resnet50(weights=None)
    state = torch.load((ROOT / args.weights) if not args.weights.is_absolute() else args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    # Full-FP16 training is unstable on this 4 GB Pascal GPU; keep optimization in FP32.
    model = model.to("cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and torch.cuda.is_available())
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"classes": CLASSES, "train_samples": len(train_rows), "validation_samples": len(val_rows), "train_groups": len({row["group_id"] for row in train_rows}), "validation_groups": len({row["group_id"] for row in val_rows})})
    (args.output / "training_config.json").write_text(json.dumps({key: str(value) for key, value in config.items()}, indent=2), encoding="utf-8")
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
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
                raise FloatingPointError(f"Non-finite ResNet training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)
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
                    raise FloatingPointError(f"Non-finite ResNet validation loss at epoch {epoch + 1}")
                val_loss += loss.item() * len(images)
        record = {"epoch": float(epoch + 1), "train_loss": train_loss / len(train_rows), "val_loss": val_loss / len(val_rows)}
        history.append(record)
        print(json.dumps(record))
    checkpoint = args.output / "resnet50_comparison.pt"
    torch.save({"model": model.float().state_dict(), "classes": CLASSES, "history": history, "seed": args.seed}, checkpoint)
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()