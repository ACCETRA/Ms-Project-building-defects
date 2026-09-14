#!/usr/bin/env python3
"""Evaluate Florence-2 on the immutable CUBIT locked test set alongside YOLO and ResNet."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bdi.florence import TASK_PROMPT, GROUNDING_TEXT, canonicalize_label, deduplicate, Detection

CUBIT_TEST_DIR = ROOT / "datasets/building-target/CUBIT-InSeg/CUBIT-InSeg/test"
DEFAULT_OUTPUT = ROOT / "runs/evaluation/cubit_locked_florence.json"
CHECKPOINT_DIR = ROOT / "weights/florence-community-2-base-ft"

CLASSES = [
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
]


def box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def parse_yolo_polygons_to_boxes(
    label_text: str, img_width: int, img_height: int
) -> list[tuple[str, list[float]]]:
    """Convert CUBIT polygon lines to [x1, y1, x2, y2] bounding boxes in pixel coordinates."""
    results: list[tuple[str, list[float]]] = []
    for line in label_text.strip().splitlines():
        tokens = line.strip().split()
        if len(tokens) < 5:
            continue
        class_id = int(tokens[0])
        if class_id >= len(CLASSES):
            continue
        class_name = CLASSES[class_id]

        coords = [float(v) for v in tokens[1:]]
        # Coordinates alternate x, y (normalized to [0, 1])
        xs = [coords[i] * img_width for i in range(0, len(coords), 2)]
        ys = [coords[i] * img_height for i in range(1, len(coords), 2)]
        if not xs or not ys:
            continue

        x1, x2 = max(0.0, min(xs)), min(float(img_width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(img_height), max(ys))
        if x2 > x1 and y2 > y1:
            results.append((class_name, [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]))
    return results


class SafeFlorenceEvaluator:
    def __init__(self, checkpoint: Path, device: str = "cpu", max_new_tokens: int = 128) -> None:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"Loading Florence-2 Base from {checkpoint} on {self.device} ({dtype})...")
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval().to(self.device)

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def infer(self, image: Image.Image) -> list[tuple[str, list[float]]]:
        torch = self.torch
        prompt = TASK_PROMPT + GROUNDING_TEXT
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, dtype
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        sequence = output.detach().cpu()
        decoded = self.processor.batch_decode(sequence, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            decoded, task=TASK_PROMPT, image_size=image.size
        )
        payload = parsed.get(TASK_PROMPT, {}) if isinstance(parsed, dict) else {}
        boxes = payload.get("bboxes", []) if isinstance(payload, dict) else []
        labels = payload.get("labels", []) if isinstance(payload, dict) else []

        detections: list[Detection] = []
        for box, label in zip(boxes, labels, strict=False):
            mapped = canonicalize_label(str(label))
            if mapped is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            family, subtype = mapped
            detections.append(
                Detection(
                    defect_family=family,
                    subtype=subtype,
                    box_xyxy=[float(v) for v in box],
                    confidence_proxy=1.0,
                    raw_label=str(label),
                )
            )
        deduped = deduplicate(detections, threshold=0.85)
        return [(d.defect_family, d.box_xyxy) for d in deduped]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Florence on CUBIT locked test")
    parser.add_argument("--limit", type=int, default=25, help="Number of test images to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    images_zip = CUBIT_TEST_DIR / "images.zip"
    labels_zip = CUBIT_TEST_DIR / "labels.zip"
    if not images_zip.is_file() or not labels_zip.is_file():
        raise FileNotFoundError(f"Missing CUBIT test archives in {CUBIT_TEST_DIR}")

    evaluator = SafeFlorenceEvaluator(CHECKPOINT_DIR, device=args.device)

    with zipfile.ZipFile(images_zip) as izf, zipfile.ZipFile(labels_zip) as lzf:
        image_names = sorted(
            [n for n in izf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        total_available = len(image_names)
        selected_names = image_names[: args.limit] if args.limit else image_names

        print(
            f"Evaluating Florence-2 on {len(selected_names)}/{total_available} locked CUBIT test samples..."
        )

        overall_tp = overall_fp = overall_fn = 0
        per_class_counts: dict[str, dict[str, int]] = {
            c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in CLASSES
        }
        latencies: list[float] = []
        matched_ious: list[float] = []

        try:
            for idx, img_name in enumerate(selected_names, 1):
                base_stem = Path(img_name).stem
                label_candidates = [f"labels/{base_stem}.txt", f"{base_stem}.txt"]
                label_text = ""
                for lc in label_candidates:
                    if lc in lzf.namelist():
                        label_text = lzf.read(lc).decode("utf-8")
                        break

                img_bytes = izf.read(img_name)
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    w, h = pil_img.size
                    ground_truths = parse_yolo_polygons_to_boxes(label_text, w, h)

                    t0 = time.perf_counter()
                    predictions = evaluator.infer(pil_img.convert("RGB"))
                    elapsed = time.perf_counter() - t0
                    latencies.append(elapsed)

                # Class-aware IoU bipartite matching
                matched_gt = set()
                for pred_label, pred_box in predictions:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for gt_idx, (gt_label, gt_box) in enumerate(ground_truths):
                        if gt_label == pred_label and gt_idx not in matched_gt:
                            curr_iou = box_iou(pred_box, gt_box)
                            if curr_iou > best_iou:
                                best_iou = curr_iou
                                best_gt_idx = gt_idx

                    if best_iou >= args.iou_threshold:
                        overall_tp += 1
                        per_class_counts[pred_label]["tp"] += 1
                        matched_gt.add(best_gt_idx)
                        matched_ious.append(best_iou)
                    else:
                        overall_fp += 1
                        per_class_counts[pred_label]["fp"] += 1

                for gt_idx, (gt_label, _) in enumerate(ground_truths):
                    per_class_counts[gt_label]["support"] += 1
                    if gt_idx not in matched_gt:
                        overall_fn += 1
                        per_class_counts[gt_label]["fn"] += 1

                print(
                    f"[{idx}/{len(selected_names)}] {base_stem}: GT={len(ground_truths)}, Preds={len(predictions)}, Matches={len(matched_gt)} ({elapsed:.2f}s)"
                )

        finally:
            evaluator.close()

    precision = overall_tp / max(1, overall_tp + overall_fp)
    recall = overall_tp / max(1, overall_tp + overall_fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    per_class_summary = {}
    for c, stats in per_class_counts.items():
        c_p = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        c_r = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        c_f1 = 2 * c_p * c_r / max(1e-12, c_p + c_r)
        per_class_summary[c] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1": round(c_f1, 4),
            "support": stats["support"],
            "true_positive": stats["tp"],
            "false_positive": stats["fp"],
            "false_negative": stats["fn"],
        }

    results = {
        "benchmark": "CUBIT-InSeg locked test evaluation",
        "model": "florence-community-2-base-ft",
        "evaluated_samples": len(selected_names),
        "total_test_set_size": total_available,
        "iou_threshold": args.iou_threshold,
        "overall_precision": round(precision, 4),
        "overall_recall": round(recall, 4),
        "overall_f1": round(f1, 4),
        "mean_iou_of_matches": round(mean_iou, 4),
        "mean_latency_seconds": round(float(np.mean(latencies)), 4),
        "per_class": per_class_summary,
        "comparison_with_baselines": {
            "yolo11n_detect_v1_map50": 0.632,
            "yolo11n_seg_v1_map50": 0.583,
            "florence2_base_zero_shot_f1": round(f1, 4),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Florence-2 Locked Test Results (Evaluated on {len(selected_names)} samples):")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  Mean IoU:  {mean_iou:.3f}")
    print(f"  Output saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
