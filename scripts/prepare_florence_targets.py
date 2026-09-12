#!/usr/bin/env python3
"""Build validated, annotation-level Florence fine-tuning targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "feasibility_master_v0_1.csv"
DEFAULT_OUTPUT = ROOT / "data" / "manifests" / "florence_targets_v0_1.jsonl"
SUPPORTED_LABELS = {
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
}
TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"
GROUNDING_TEXT = (
    "A concrete or masonry surface with a crack, spalling, a honeycombing rock "
    "pocket, exposed rebar, rust staining, or efflorescence leaching."
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def split_groups(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted({row["group_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(rows) * fraction))
    validation: set[str] = set()
    count = 0
    for group in groups:
        validation.add(group)
        count += sum(row["group_id"] == group for row in rows)
        if count >= target:
            break
    train = [row for row in rows if row["group_id"] not in validation]
    valid = [row for row in rows if row["group_id"] in validation]
    if not train or not valid:
        raise ValueError("Cannot create non-empty group-safe train and validation splits")
    return train, valid


def build_target(row: dict[str, str]) -> dict[str, Any]:
    normalized_path = ROOT / row["normalized_annotation"]
    payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    targets: list[dict[str, Any]] = []
    for annotation in payload.get("annotations", []):
        label = annotation.get("canonical_label")
        if annotation.get("annotation_status") != "positive":
            continue
        if label not in SUPPORTED_LABELS:
            raise ValueError(
                f"Unsupported positive target {label!r} in {row['sample_id']}/{annotation.get('annotation_id')}"
            )
        box = annotation.get("box_xywh")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Missing valid box in {row['sample_id']}/{annotation.get('annotation_id')}")
        x, y, width, height = (float(value) for value in box)
        targets.append(
            {
                "annotation_id": annotation["annotation_id"],
                "label": label,
                "box_xyxy": [x, y, x + width, y + height],
                "polygons_xy": annotation.get("polygons_xy", []),
                "native_label": annotation.get("native_label"),
                "mapping_strength": annotation.get("mapping_strength"),
            }
        )
    image_path = ROOT / row["output_image"]
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "group_id": row["group_id"],
        "image": row["output_image"],
        "image_sha256": digest(image_path),
        "normalized_annotation": row["normalized_annotation"],
        "normalized_annotation_sha256": digest(normalized_path),
        "task_prompt": TASK_PROMPT,
        "grounding_text": GROUNDING_TEXT,
        "target_encoding": "structured_pre_tokenization",
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    with args.manifest.resolve().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "group_id", "output_image", "normalized_annotation"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest is missing required Florence target fields")
    train, validation = split_groups(rows, args.val_fraction, args.seed)
    records: list[dict[str, Any]] = []
    for split, subset in (("train", train), ("validation", validation)):
        for row in subset:
            target = build_target(row)
            target["split"] = split
            if not target["targets"]:
                raise ValueError(f"No positive Florence targets for {row['sample_id']}")
            records.append(target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"records": len(records), "train_records": len(train), "validation_records": len(validation), "output": str(args.output.resolve()), "target_encoding": "structured_pre_tokenization"}, indent=2))


if __name__ == "__main__":
    main()
