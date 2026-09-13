#!/usr/bin/env python3
"""Fit validation-only multilabel thresholds and evaluate ResNet on v1/CODEBRIM."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("crack", "spalling", "honeycombing_rock_pocket", "exposed_rebar", "rust_staining", "efflorescence_leaching")


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def split_rows(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted({row["group_id"] for row in rows})
    generator = np.random.default_rng(seed)
    generator.shuffle(groups)
    validation_groups: set[str] = set()
    count = 0
    for group in groups:
        validation_groups.add(group)
        count += sum(row["group_id"] == group for row in rows)
        if count >= max(1, round(len(rows) * fraction)):
            break
    return [r for r in rows if r["group_id"] not in validation_groups], [r for r in rows if r["group_id"] in validation_groups]


def target_vector(labels: str) -> np.ndarray:
    values = {value.strip() for value in labels.replace("|", ";").split(";") if value.strip()}
    return np.asarray([float(label in values) for label in CLASSES], dtype=np.float32)


class ImageRows(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[tuple[bytes, np.ndarray]], transform: object) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, target = self.records[index]
        with Image.open(io.BytesIO(data)) as image:
            return self.transform(image.convert("RGB")), torch.from_numpy(target)


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def predict(records: list[tuple[bytes, np.ndarray]], model: nn.Module, transform: object, batch: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ImageRows(records, transform), batch_size=batch, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            probabilities.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def binary_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = score >= threshold
    true_positive = float(np.logical_and(predicted, target == 1).sum())
    false_positive = float(np.logical_and(predicted, target == 0).sum())
    false_negative = float(np.logical_and(~predicted, target == 1).sum())
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": float((target == 1).sum())}


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    sorted_target = target[order]
    positives = float(sorted_target.sum())
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(sorted_target)
    precision = cumulative / np.arange(1, len(target) + 1)
    return float((precision * sorted_target).sum() / positives)


def choose_thresholds(target: np.ndarray, score: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 19)
    chosen = []
    for class_index in range(target.shape[1]):
        metrics = [(binary_metrics(target[:, class_index], score[:, class_index], float(value))["f1"], value) for value in grid]
        chosen.append(max(metrics, key=lambda item: item[0])[1])
    return np.asarray(chosen, dtype=np.float32)


def metric_report(target: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> dict[str, object]:
    per_class = {}
    for index, label in enumerate(CLASSES):
        result = binary_metrics(target[:, index], score[:, index], float(thresholds[index]))
        result["pr_auc"] = average_precision(target[:, index], score[:, index])
        per_class[label] = result
    all_target = target.reshape(-1)
    all_score = score.reshape(-1)
    aggregate = binary_metrics(all_target, all_score, float(np.mean(thresholds)))
    aggregate["pr_auc"] = average_precision(all_target, all_score)
    return {"samples": int(len(target)), "thresholds": dict(zip(CLASSES, thresholds.astype(float))), "per_class": per_class, "micro": aggregate}


def v1_records(manifest: Path, seed: int, fraction: float) -> tuple[list[tuple[bytes, np.ndarray]], list[tuple[bytes, np.ndarray]]]:
    rows = [row for row in rows_from_csv(manifest) if row.get("eligibility") and row["geometry_qc"] == "passed"]
    _, validation = split_rows(rows, fraction, seed)
    records = [( (ROOT / row["output_image"]).read_bytes(), target_vector(row["canonical_labels"]) ) for row in validation]
    return records, records


def codebrim_records(manifest: Path, split: str) -> list[tuple[bytes, np.ndarray]]:
    rows = [row for row in rows_from_csv(manifest) if row["split"] == split]
    archive_path = ROOT / rows[0]["archive_path"]
    records = []
    extractor = ROOT / "vendor/7zip-portable/x64/7za.exe"
    with tempfile.TemporaryDirectory(prefix="codebrim_eval_") as temporary:
        destination = Path(temporary)
        subprocess.run(
            [str(extractor), "x", str(archive_path), f"classification_dataset/{split}/*", f"-o{destination}", "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for row in rows:
            image_path = destination / row["image_member"]
            records.append((image_path.read_bytes(), target_vector(row["canonical_labels"])))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/comparison/resnet50_v1_queue/resnet50_comparison.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/v1_classification_manifest.csv")
    parser.add_argument("--codebrim-manifest", type=Path, default=ROOT / "data/manifests/codebrim_classification_source_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation/resnet_metrics.json")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint, device)
    transform = ResNet50_Weights.DEFAULT.transforms()
    validation, _ = v1_records(args.manifest if args.manifest.is_absolute() else ROOT / args.manifest, args.seed, args.val_fraction)
    validation_score, validation_target = predict(validation, model, transform, args.batch, device)
    thresholds = choose_thresholds(validation_target, validation_score)
    codebrim = codebrim_records(args.codebrim_manifest if args.codebrim_manifest.is_absolute() else ROOT / args.codebrim_manifest, "test")
    codebrim_score, codebrim_target = predict(codebrim, model, transform, args.batch, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "classes": CLASSES,
        "threshold_selection": {"source": "v1 group-safe validation only", "seed": args.seed, "val_fraction": args.val_fraction},
        "validation": metric_report(validation_target, validation_score, thresholds),
        "codebrim_test": metric_report(codebrim_target, codebrim_score, thresholds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "resnet_thresholds_v1.json").write_text(json.dumps(result["validation"]["thresholds"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
