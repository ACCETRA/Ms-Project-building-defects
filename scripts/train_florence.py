#!/usr/bin/env python3
"""Fine-tune Florence-2 phrase-grounding targets from the reviewed manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights" / "florence-community-2-base-ft"
TARGETS = ROOT / "data" / "v1" / "training_inputs" / "florence_targets_v1_0_0.jsonl"

PHRASES = {
    "crack": "crack",
    "spalling": "spalling",
    "honeycombing_rock_pocket": "honeycombing rock pocket",
    "exposed_rebar": "exposed rebar",
    "rust_staining": "rust staining",
    "efflorescence_leaching": "efflorescence leaching",
}


def location(value: float, extent: float) -> str:
    quantized = max(0, min(1000, round(value / extent * 1000)))
    return f"<loc_{quantized}>"


def target_text(record: dict[str, Any]) -> str:
    with Image.open(ROOT / record["image"]) as image:
        width, height = image.size
    parts: list[str] = []
    for target in record["targets"]:
        phrase = PHRASES[target["label"]]
        x1, y1, x2, y2 = target["box_xyxy"]
        parts.append(
            f"{phrase}{location(x1, width)}{location(y1, height)}"
            f"{location(x2, width)}{location(y2, height)}"
        )
    if not parts:
        raise ValueError(f"No targets for {record['sample_id']}")
    return " ".join(parts)


class FlorenceTargets(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(ROOT / record["image"]).convert("RGB")
        return {"record": record, "image": image, "target_text": target_text(record)}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in items],
        "records": [item["record"] for item in items],
        "target_texts": [item["target_text"] for item in items],
    }


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    if not train or not validation or {record["group_id"] for record in train} & {record["group_id"] for record in validation}:
        raise ValueError("Florence targets must have non-empty disjoint train and validation groups")
    return train, validation


def encode_batch(processor: Any, batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    inputs = processor(images=batch["images"], text=["<CAPTION_TO_PHRASE_GROUNDING>"] * len(batch["images"]), return_tensors="pt", padding=True)
    labels = processor.tokenizer(batch["target_texts"], return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(device, dtype=dtype),
        "labels": labels.to(device),
    }


def evaluate(model: Any, processor: Any, loader: DataLoader[dict[str, Any]], device: torch.device, dtype: torch.dtype) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            encoded = encode_batch(processor, batch, device, dtype)
            loss = model(**encoded).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Florence validation loss")
            total += float(loss.detach().cpu()) * len(batch["images"])
            count += len(batch["images"])
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "florence" / "finetune-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overfit-one", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    train_records, validation_records = load_records(args.targets.resolve())
    if args.overfit_one:
        train_records = train_records[:1]
        validation_records = validation_records[:1]
    processor = AutoProcessor.from_pretrained(args.checkpoint.resolve(), local_files_only=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = Florence2ForConditionalGeneration.from_pretrained(
        args.checkpoint.resolve(), local_files_only=True, dtype=dtype, low_cpu_mem_usage=True
    ).to("cuda", dtype=dtype)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    train_loader = DataLoader(FlorenceTargets(train_records), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    validation_loader = DataLoader(FlorenceTargets(validation_records), batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    sample = next(iter(train_loader))
    encoded = encode_batch(processor, sample, torch.device("cuda"), dtype)
    if args.dry_run:
        print(json.dumps({"status": "ready", "train_records": len(train_records), "validation_records": len(validation_records), "target_text": sample["target_texts"][0], "input_shape": list(encoded["input_ids"].shape), "label_shape": list(encoded["labels"].shape)}))
        return
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            encoded = encode_batch(processor, batch, torch.device("cuda"), dtype)
            loss = model(**encoded).loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Florence loss at epoch {epoch + 1}, batch {batch_index + 1}")
            loss.backward()
            running += float(loss.detach().cpu())
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(train_loader):
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
        validation_loss = evaluate(model, processor, validation_loader, torch.device("cuda"), dtype)
        record = {"epoch": float(epoch + 1), "train_loss_scaled": running / max(1, len(train_loader)), "validation_loss": validation_loss}
        history.append(record)
        print(json.dumps(record), flush=True)
        model.save_pretrained(args.output_dir / "checkpoint-last")
        processor.save_pretrained(args.output_dir / "checkpoint-last")
        if args.max_steps and step >= args.max_steps:
            break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
