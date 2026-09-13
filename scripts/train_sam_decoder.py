#!/usr/bin/env python3
"""Train only the SAM 2 mask decoder from frozen v1 polygon annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights/sam2.1-hiera-tiny"
MANIFEST = ROOT / "data/manifests/v1_segmentation_manifest.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def split_groups(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted({row["group_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    validation: set[str] = set()
    count = 0
    for group in groups:
        validation.add(group)
        count += sum(row["group_id"] == group for row in rows)
        if count >= max(1, round(len(rows) * fraction)):
            break
    train = [row for row in rows if row["group_id"] not in validation]
    val = [row for row in rows if row["group_id"] in validation]
    if not train or not val:
        raise ValueError("Could not create non-empty group-safe SAM train/validation splits")
    return train, val


def pair_records(rows: list[dict[str, str]], limit: int | None = None) -> list[dict[str, Any]]:
    pairs = []
    for row in rows:
        payload = json.loads((ROOT / row["normalized_annotation"]).read_text(encoding="utf-8"))
        for index, annotation in enumerate(payload.get("annotations", [])):
            if annotation.get("annotation_status") != "positive":
                continue
            polygon = next((polygon for polygon in annotation.get("polygons_xy", []) if len(polygon) >= 3), None)
            box = annotation.get("box_xywh")
            if polygon is None or not box:
                continue
            x, y, width, height = [float(value) for value in box]
            pairs.append({
                "sample_id": f"{row['sample_id']}_annotation_{index}",
                "group_id": row["group_id"],
                "image": row["output_image"],
                "width": int(payload["image"]["width"]),
                "height": int(payload["image"]["height"]),
                "box_xyxy": [x, y, x + width, y + height],
                "polygon_xy": [[float(point[0]), float(point[1])] for point in polygon],
            })
            if limit is not None and len(pairs) >= limit:
                return pairs
    if not pairs:
        raise ValueError("No positive polygon pairs found in segmentation manifest")
    return pairs


def rasterize(pair: dict[str, Any]) -> np.ndarray:
    mask = Image.new("L", (pair["width"], pair["height"]), 0)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in pair["polygon_xy"]], fill=255)
    return np.asarray(mask, dtype=np.float32) / 255.0


class SamPairs(Dataset[dict[str, Any]]):
    def __init__(self, pairs: list[dict[str, Any]], processor: Sam2Processor) -> None:
        self.pairs = pairs
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        image = Image.open(ROOT / pair["image"]).convert("RGB")
        encoded = self.processor(images=image, input_boxes=[[pair["box_xyxy"]]], return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"][0], "original_sizes": encoded["original_sizes"][0], "input_boxes": encoded["input_boxes"][0], "target": rasterize(pair), "sample_id": pair["sample_id"]}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "original_sizes": torch.stack([item["original_sizes"] for item in items]),
        "input_boxes": torch.stack([item["input_boxes"] for item in items]),
        "target": torch.from_numpy(np.stack([item["target"] for item in items])),
        "sample_ids": [item["sample_id"] for item in items],
    }


def mask_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    logits = logits.float()
    target = torch.nn.functional.interpolate(target[:, None], size=logits.shape[-2:], mode="nearest")
    probability = logits.sigmoid()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = ((2 * intersection + 1.0) / (probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)).mean()
    loss = bce + (1.0 - dice)
    predicted = probability >= 0.5
    target_binary = target >= 0.5
    union = torch.logical_or(predicted, target_binary).sum().float()
    iou = (torch.logical_and(predicted, target_binary).sum().float() / union.clamp_min(1.0)).item()
    return loss, {"bce": float(bce.detach()), "dice": float(dice.detach()), "iou": iou}


def evaluate(model: Sam2Model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "iou": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            output = model(pixel_values=batch["pixel_values"].to(device), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
            loss, metrics = mask_loss(output.pred_masks[:, 0], batch["target"].to(device))
            totals["loss"] += float(loss)
            for key in ("bce", "dice", "iou"):
                totals[key] += metrics[key]
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/sam/decoder_v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    rows = [row for row in read_rows(args.manifest if args.manifest.is_absolute() else ROOT / args.manifest) if row.get("eligibility") and row["geometry_qc"] == "passed"]
    train_rows, val_rows = split_groups(rows, args.val_fraction, args.seed)
    train_pairs = pair_records(train_rows, args.max_train_pairs)
    val_pairs = pair_records(val_rows, args.max_val_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_manifest = {"train_pairs": train_pairs, "validation_pairs": val_pairs, "split_policy": "group-safe", "seed": args.seed}
    (args.output_dir / "pair_manifest.json").write_text(json.dumps(pair_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_pairs": len(train_pairs), "validation_pairs": len(val_pairs), "output": str(args.output_dir / 'pair_manifest.json')}))
    if args.inspect_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM decoder training; use --inspect-only on CPU")
    device = torch.device("cuda")
    processor = Sam2Processor.from_pretrained(args.checkpoint.resolve(), local_files_only=True)
    model = Sam2Model.from_pretrained(args.checkpoint.resolve(), local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for parameter in model.mask_decoder.parameters():
        parameter.requires_grad = True
        trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    # The checkpoint decoder is FP16 on this path; GradScaler cannot unscale
    # FP16 leaf gradients. Larger-GPU runs should use an FP32/PEFT variant.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    train_loader = DataLoader(SamPairs(train_pairs, processor), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(SamPairs(val_pairs, processor), batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(pixel_values=batch["pixel_values"].to(device, dtype=torch.float16), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
            loss, _ = mask_loss(output.pred_masks[:, 0], batch["target"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SAM decoder training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            train_total += float(loss.detach())
        if not all(torch.isfinite(parameter).all() for parameter in trainable):
            raise FloatingPointError("SAM decoder parameters became non-finite after optimizer step")
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": train_total / max(1, len(train_loader)), "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        torch.save({"model": model.state_dict(), "trainable": "mask_decoder", "history": history, "seed": args.seed}, args.output_dir / "sam_decoder_last.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
