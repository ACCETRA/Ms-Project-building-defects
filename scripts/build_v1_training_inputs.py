#!/usr/bin/env python3
"""Materialize and validate training inputs from frozen v1 task views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(rows: list[dict[str, str]], task: str) -> None:
    if not rows:
        raise ValueError(f"{task} manifest is empty")
    locked = [row["sample_id"] for row in rows if row.get("test_locked", "").casefold() == "true"]
    if locked:
        raise ValueError(f"{task} input contains locked test records: {locked[:5]}")
    invalid_splits = sorted({row.get("split", "") for row in rows} - {"train", "val"})
    if invalid_splits:
        raise ValueError(f"{task} input contains invalid tuning splits: {invalid_splits}")
    for row in rows:
        if row.get("geometry_qc") != "passed":
            raise ValueError(f"{task} input failed geometry QA: {row['sample_id']}")
        image = ROOT / row["output_image"]
        annotation = ROOT / row["normalized_annotation"]
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Missing v1 asset for {task}: {row['sample_id']}")


def payload(row: dict[str, str]) -> dict[str, Any]:
    return json.loads((ROOT / row["normalized_annotation"]).read_text(encoding="utf-8"))


def yolo_lines(row: dict[str, str], task: str) -> list[str]:
    data = payload(row)
    width = data["image"]["width"]
    height = data["image"]["height"]
    class_ids = {label: index for index, label in enumerate(CLASSES)}
    lines: list[str] = []
    for annotation in data["annotations"]:
        if annotation.get("annotation_status") != "positive":
            continue
        label = annotation.get("canonical_label")
        if label not in class_ids:
            continue
        box = annotation.get("box_xywh")
        if not box:
            continue
        x, y, box_width, box_height = box
        values = [
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        ]
        if task == "segment":
            polygons = annotation.get("polygons_xy") or []
            if not polygons and annotation.get("semantic_mask"):
                mask_info = annotation["semantic_mask"]
                mask_path = ROOT / mask_info["uri"]
                mask_image = cv2.imdecode(np.frombuffer(mask_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if mask_image is None:
                    raise ValueError(f"Unable to decode semantic mask: {mask_path}")
                if mask_image.ndim == 2:
                    mask = mask_image == mask_info["rgba"][0]
                else:
                    rgba = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2RGBA) if mask_image.shape[2] == 4 else cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
                    mask = np.all(rgba == np.asarray(mask_info["rgba"], dtype=np.uint8), axis=2)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons = [contour[:, 0, :].tolist() for contour in contours if len(contour) >= 3]
            polygon = next((item for item in polygons if len(item) >= 3), None)
            if polygon is None:
                continue
            values = [value for point in polygon for value in (point[0] / width, point[1] / height)]
        lines.append(f"{class_ids[label]} " + " ".join(f"{value:.8f}" for value in values))
    return lines


def materialize_yolo(rows: list[dict[str, str]], output: Path, task: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        split_rows = [row for row in rows if row["split"] == split]
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for row in split_rows:
            sample_id = row["sample_id"]
            destination = image_dir / f"{sample_id}.jpg"
            shutil.copy2(ROOT / row["output_image"], destination)
            labels = yolo_lines(row, task)
            if not labels:
                raise ValueError(f"No usable YOLO labels for {task}: {sample_id}")
            (label_dir / f"{sample_id}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            paths.append(destination.as_posix())
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        counts[split] = len(split_rows)
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: train.txt\nval: val.txt\nnames: " + json.dumps(list(CLASSES)) + "\n",
        encoding="utf-8",
    )
    return counts


def build_florence(rows: list[dict[str, str]], output: Path) -> int:
    records: list[dict[str, Any]] = []
    for row in rows:
        data = payload(row)
        targets = []
        for annotation in data["annotations"]:
            if annotation.get("annotation_status") != "positive":
                continue
            if annotation.get("canonical_label") not in CLASSES:
                continue
            x, y, width, height = annotation["box_xywh"]
            targets.append({
                "label": annotation["canonical_label"],
                "box_xyxy": [x, y, x + width, y + height],
            })
        if not targets:
            raise ValueError(f"No Florence targets for {row['sample_id']}")
        records.append({
            "sample_id": row["sample_id"],
            "image": row["output_image"],
            "targets": targets,
            "group_id": row["group_id"],
            "split": "validation" if row["split"] == "val" else "train",
            "source_dataset": row["source_dataset"],
            "manifest_version": "v1_0_0",
            "test_locked": False,
        })
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = root / "data" / "manifests"
    output_root = root / "data" / "v1" / "training_inputs"
    detection = read_rows(manifests / "v1_detection_manifest.csv")
    segmentation = read_rows(manifests / "v1_segmentation_manifest.csv")
    classification = read_rows(manifests / "v1_classification_manifest.csv")
    florence = read_rows(manifests / "v1_florence_product_manifest.csv")
    for name, rows in (("detection", detection), ("segmentation", segmentation), ("classification", classification), ("florence", florence)):
        validate_rows(rows, name)
    if {row["sample_id"] for row in detection} != {row["sample_id"] for row in segmentation}:
        raise ValueError("Detection and segmentation v1 views do not contain the same sample IDs")
    detection_counts = materialize_yolo(detection, output_root / "yolo_detection", "detect")
    segmentation_counts = materialize_yolo(segmentation, output_root / "yolo_segmentation", "segment")
    florence_path = output_root / "florence_targets_v1_0_0.jsonl"
    florence_count = build_florence(florence, florence_path)
    classification_path = output_root / "classification_manifest_v1_0_0.csv"
    shutil.copy2(manifests / "v1_classification_manifest.csv", classification_path)
    report = {
        "manifest_version": "v1_0_0",
        "locked_test_records": 0,
        "detection": detection_counts,
        "segmentation": segmentation_counts,
        "classification_records": len(classification),
        "florence_records": florence_count,
        "outputs": {
            "yolo_detection": str((output_root / "yolo_detection").relative_to(root)),
            "yolo_segmentation": str((output_root / "yolo_segmentation").relative_to(root)),
            "classification": str(classification_path.relative_to(root)),
            "florence": str(florence_path.relative_to(root)),
        },
        "sha256": {name: digest(path) for name, path in (
            ("detection_manifest", manifests / "v1_detection_manifest.csv"),
            ("segmentation_manifest", manifests / "v1_segmentation_manifest.csv"),
            ("classification_manifest", manifests / "v1_classification_manifest.csv"),
            ("florence_manifest", manifests / "v1_florence_product_manifest.csv"),
            ("classification_training_input", classification_path),
            ("florence_training_input", florence_path),
        )},
    }
    report_path = root / "artifacts" / "data-audit" / "v1_training_inputs_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())