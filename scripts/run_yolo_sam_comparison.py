#!/usr/bin/env python3
"""Run the YOLO-box to SAM 2.1 Tiny comparison route."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from bdi.findings import (  # noqa: E402
    build_finding,
    directory_checkpoint_sha256,
    file_sha256,
    validate_findings,
)
from bdi.florence import image_quality  # noqa: E402

CLASS_IDS = {
    "crack": 0,
    "spalling": 1,
    "honeycombing_rock_pocket": 2,
    "exposed_rebar": 3,
    "rust_staining": 4,
    "efflorescence_leaching": 5,
}
MODEL_LABELS = {
    "crack": ("crack", "unspecified"),
    "spalling": ("spalling", "not_applicable"),
    "honeycombing_rock_pocket": ("honeycombing_rock_pocket", "not_applicable"),
    "exposed_rebar": ("exposed_rebar", "not_applicable"),
    "rust_staining": ("rust_staining", "not_applicable"),
    "efflorescence_leaching": ("efflorescence_leaching", "not_applicable"),
}
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"
DEFAULT_YOLO = ROOT / "runs" / "comparison" / "yolo" / "yolo11n_detection_fyp" / "weights" / "best.pt"
DEFAULT_SAM = ROOT / "weights" / "sam2.1-hiera-tiny"
PIPELINE_VERSION = "0.1.0-yolo-sam-feasibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "comparison" / "yolo_sam")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--run-id", default="yolo-sam-feasibility")
    return parser.parse_args()


def load_samples(path: Path, sample_ids: list[str], limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0].get("split") != "train":
        raise ValueError("Comparison feasibility input must be training-only")
    requested = set(sample_ids)
    if requested:
        rows = [row for row in rows if row["sample_id"] in requested]
        missing = requested - {row["sample_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        image_path = ROOT / row["output_image"]
        if not image_path.is_file() or file_sha256(image_path) != row["sha256"]:
            raise ValueError(f"Image hash check failed for {row['sample_id']}")
    return rows


def canonical_model_label(name: str) -> tuple[str, str] | None:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return MODEL_LABELS.get(normalized)


def mask_polygon(mask: Any) -> list[list[float]] | None:
    import cv2
    import numpy as np

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2).tolist()
    return [[float(x), float(y)] for x, y in points] if len(points) >= 3 else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import Sam2Model, Sam2Processor
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    samples = load_samples(args.manifest.resolve(), args.sample_id, args.limit)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    yolo_path = args.yolo_weights.resolve()
    previous_directory = Path.cwd()
    try:
        os.chdir(yolo_path.parent)
        yolo = YOLO(yolo_path.name)
    finally:
        os.chdir(previous_directory)
    processor = Sam2Processor.from_pretrained(args.sam_checkpoint.resolve(), local_files_only=True)
    sam = Sam2Model.from_pretrained(
        args.sam_checkpoint.resolve(), local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True
    ).eval().to("cuda")
    if next(sam.parameters()).device.type != "cuda":
        raise RuntimeError("SAM silently failed to load on CUDA")

    findings: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for sample in samples:
            image_path = ROOT / sample["output_image"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            quality = image_quality(image)
            result = yolo.predict(source=image, imgsz=args.imgsz, conf=args.confidence, device=0, half=True, verbose=False)[0]
            boxes = result.boxes
            selected: list[tuple[str, str, list[float], float]] = []
            if boxes is not None:
                for box, class_index, confidence in zip(
                    boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=False
                ):
                    name = str(result.names[int(class_index)])
                    mapped = canonical_model_label(name)
                    if mapped is not None:
                        selected.append((mapped[0], mapped[1], [float(value) for value in box], float(confidence)))
            masks: list[Any] = []
            if selected:
                inputs = processor(
                    images=image,
                    input_boxes=[[item[2] for item in selected]],
                    return_tensors="pt",
                ).to("cuda", torch.float16)
                with torch.inference_mode():
                    output = sam(**inputs, multimask_output=False)
                processed = processor.post_process_masks(
                    output.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
                )[0]
                masks = [processed[index][0] for index in range(len(selected))]
            sample_findings: list[dict[str, Any]] = []
            for index, (family, subtype, box, confidence) in enumerate(selected):
                finding = build_finding(
                    run_id=args.run_id,
                    sample=sample,
                    image_path=image_path,
                    image_size=image.size,
                    defect_family=family,
                    subtype=subtype,
                    box_xyxy=box,
                    confidence_proxy=confidence,
                    pipeline_route=["yolo11n", "sam2.1-hiera-tiny"],
                    escalation_state="completed",
                    checkpoint_sha256=directory_checkpoint_sha256(args.sam_checkpoint.resolve()),
                    model_id="yolo11n-to-sam2.1-hiera-tiny",
                    quality=quality,
                    calibration=None,
                    pipeline_version=PIPELINE_VERSION,
                )
                polygon = mask_polygon(masks[index]) if index < len(masks) else None
                if polygon is not None:
                    finding["prediction"]["geometry"]["mask"] = {"encoding": "polygon_xy", "points": polygon}
                sample_findings.append(finding)
            findings.extend(sample_findings)
            image_results.append({"sample_id": sample["sample_id"], "detections": len(selected), "findings": len(sample_findings), "quality": quality})
            image.close()
    finally:
        sam.to("cpu")
        del sam, yolo
        gc.collect()
        torch.cuda.empty_cache()

    validate_findings(findings)
    (output_dir / "findings.jsonl").write_text("\n".join(json.dumps(item, separators=(",", ":")) for item in findings) + ("\n" if findings else ""), encoding="utf-8")
    (output_dir / "image_results.jsonl").write_text("\n".join(json.dumps(item, separators=(",", ":")) for item in image_results) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed",
        "scope": "training_only_feasibility_not_accuracy_evaluation",
        "samples_processed": len(samples),
        "findings_emitted": len(findings),
        "manifest": str(args.manifest),
        "yolo_weights": str(args.yolo_weights),
        "sam_checkpoint": str(args.sam_checkpoint),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "limitations": ["training-only feasibility input", "YOLO confidence is not calibrated", "SAM mask is box-prompted and requires manual review"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
