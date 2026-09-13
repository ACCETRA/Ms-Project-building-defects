#!/usr/bin/env python3
"""Benchmark Florence phrase-grounding on the held-out v1 validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union else 0.0


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ground_truth(row: dict[str, str]) -> list[tuple[str, list[float]]]:
    payload = json.loads((ROOT / row["normalized_annotation"]).read_text(encoding="utf-8"))
    result = []
    for annotation in payload.get("annotations", []):
        if annotation.get("annotation_status") != "positive":
            continue
        box = annotation.get("box_xywh")
        label = annotation.get("canonical_label")
        if not box or not label:
            continue
        x, y, width, height = [float(value) for value in box]
        result.append((label, [x, y, x + width, y + height]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/v1_florence_product_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation/florence_validation_metrics.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    selected = [row for row in rows(args.manifest if args.manifest.is_absolute() else ROOT / args.manifest) if row["split"] in {"val", "validation"}]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No validation rows found")
    import torch
    from bdi.florence import FlorenceRunner

    checkpoint = ROOT / "weights/florence-community-2-base-ft"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Florence benchmarking")
    runner = FlorenceRunner(checkpoint, "florence-2-base-ft", max_new_tokens=128)
    true_positive = false_positive = false_negative = 0
    detections = 0
    total_targets = 0
    elapsed = 0.0
    per_image = []
    try:
        for index, row in enumerate(selected, start=1):
            image_path = ROOT / row["output_image"]
            targets = ground_truth(row)
            total_targets += len(targets)
            with Image.open(image_path) as image:
                started = time.perf_counter()
                result = runner.infer(image.convert("RGB"))
                elapsed += time.perf_counter() - started
            predictions = [(item["defect_family"], item["box_xyxy"]) for item in result.get("detections", [])]
            detections += len(predictions)
            matched = set()
            for label, box in predictions:
                candidate = max(((iou(box, target_box), target_index) for target_index, (target_label, target_box) in enumerate(targets) if target_label == label and target_index not in matched), default=(0.0, -1))
                if candidate[0] >= args.iou:
                    true_positive += 1
                    matched.add(candidate[1])
                else:
                    false_positive += 1
            false_negative += len(targets) - len(matched)
            per_image.append({"sample_id": row["sample_id"], "targets": len(targets), "predictions": len(predictions), "matches": len(matched), "elapsed_seconds": round(time.perf_counter() - started, 4)})
            print(f"{index}/{len(selected)} {row['sample_id']} targets={len(targets)} predictions={len(predictions)} matches={len(matched)}", flush=True)
    finally:
        runner.close()
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    result = {"split": "v1 validation", "samples": len(selected), "iou_threshold": args.iou, "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "precision": precision, "recall": recall, "f1": f1, "targets": total_targets, "predictions": detections, "elapsed_seconds": elapsed, "images_per_second": len(selected) / max(1e-9, elapsed), "per_image": per_image, "limitations": ["Validation-only benchmark; no locked test data used.", "Florence sequence proxy is not a calibrated probability.", "Matching is class-aware box IoU at the selected threshold."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
