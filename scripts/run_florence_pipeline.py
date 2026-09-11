#!/usr/bin/env python3
"""Run the Base-first Florence feasibility pipeline and emit validated records."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import uuid

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bdi.findings import (  # noqa: E402
    build_finding,
    directory_checkpoint_sha256,
    file_sha256,
    utc_now,
    validate_findings,
)
from bdi.florence import FlorenceRunner, image_quality  # noqa: E402
from bdi.measurement import Calibration  # noqa: E402


BASE_PATH = ROOT / "weights" / "florence-community-2-base-ft"
LARGE_PATH = ROOT / "weights" / "florence-community-2-large-ft"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA-only Florence Base/Large feasibility inference"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calibrations", type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--escalation",
        choices=("positive_or_uncertain", "positive", "all", "none"),
        default="positive_or_uncertain",
    )
    parser.add_argument(
        "--uncertain-below",
        type=float,
        default=0.35,
        help="Sequence-likelihood proxy threshold; not an accuracy probability.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse incrementally cached model results in the selected output directory.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.uncertain_below <= 1:
        parser.error("--uncertain-below must be between 0 and 1")
    if args.max_new_tokens < 8:
        parser.error("--max-new-tokens must be at least 8")
    return args


def load_samples(path: Path, sample_ids: list[str], limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "sample_id",
        "source_dataset",
        "source_image_id",
        "group_id",
        "split",
        "sha256",
        "output_image",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest is empty or missing required fields: {path}")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Feasibility inference is restricted to training records")
    requested = set(sample_ids)
    if requested:
        rows = [row for row in rows if row["sample_id"] in requested]
        missing = requested - {row["sample_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        path = ROOT / row["output_image"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Image hash mismatch for {row['sample_id']}")
    return rows


def load_calibrations(path: Path | None) -> tuple[dict[str, Calibration], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("calibrations")
    if not isinstance(raw, dict):
        raise ValueError("Calibration manifest must contain a calibrations object")
    valid: dict[str, Calibration] = {}
    for sample_id, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Calibration for {sample_id} must be an object")
        if value.get("scale_status") == "valid":
            valid[str(sample_id)] = Calibration.from_mapping(value)
    return valid, raw


def should_escalate(result: dict[str, Any], policy: str, uncertain_below: float) -> bool:
    positive = bool(result["detections"])
    uncertain = float(result["sequence_likelihood_proxy"]) < uncertain_below
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "positive":
        return positive
    return positive or uncertain


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def cached_result(path: Path, resume: bool) -> dict[str, Any] | None:
    if not resume or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"florence-{uuid.uuid4()}"
    output_dir = (args.output_dir or ROOT / "runs" / "florence" / run_id).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.manifest.resolve(), args.sample_id, args.limit)
    calibrations, calibration_audit = load_calibrations(
        args.calibrations.resolve() if args.calibrations else None
    )
    unknown_calibrations = set(calibration_audit) - {row["sample_id"] for row in samples}
    if unknown_calibrations:
        raise ValueError(f"Calibration entries do not match selected samples: {sorted(unknown_calibrations)}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    started = time.perf_counter()
    generated_at = utc_now()
    checkpoint_hashes = {
        "florence-2-base-ft": directory_checkpoint_sha256(BASE_PATH),
        "florence-2-large-ft": directory_checkpoint_sha256(LARGE_PATH),
    }
    base_results: dict[str, dict[str, Any]] = {}
    print(f"Run {run_id}: Base inference for {len(samples)} sample(s)", flush=True)
    base = FlorenceRunner(BASE_PATH, "florence-2-base-ft", args.max_new_tokens)
    try:
        for index, sample in enumerate(samples, start=1):
            cache_path = output_dir / "cache" / "base" / f"{sample['sample_id']}.json"
            cached = cached_result(cache_path, args.resume)
            if cached is not None:
                base_results[sample["sample_id"]] = cached
                print(
                    f"BASE {index}/{len(samples)} {sample['sample_id']} cached "
                    f"detections={len(cached['detections'])}",
                    flush=True,
                )
                continue
            image_path = ROOT / sample["output_image"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            try:
                base_results[sample["sample_id"]] = base.infer(image)
            finally:
                image.close()
            write_json_atomic(cache_path, base_results[sample["sample_id"]])
            print(
                f"BASE {index}/{len(samples)} {sample['sample_id']} "
                f"detections={len(base_results[sample['sample_id']]['detections'])}",
                flush=True,
            )
    finally:
        base.close()
        del base
        gc.collect()

    escalated_ids = [
        sample["sample_id"]
        for sample in samples
        if should_escalate(base_results[sample["sample_id"]], args.escalation, args.uncertain_below)
    ]
    large_results: dict[str, dict[str, Any]] = {}
    if escalated_ids:
        print(f"Large escalation for {len(escalated_ids)} sample(s)", flush=True)
        large = FlorenceRunner(LARGE_PATH, "florence-2-large-ft", args.max_new_tokens)
        try:
            selected = [sample for sample in samples if sample["sample_id"] in set(escalated_ids)]
            for index, sample in enumerate(selected, start=1):
                cache_path = output_dir / "cache" / "large" / f"{sample['sample_id']}.json"
                cached = cached_result(cache_path, args.resume)
                if cached is not None:
                    large_results[sample["sample_id"]] = cached
                    print(
                        f"LARGE {index}/{len(selected)} {sample['sample_id']} cached "
                        f"detections={len(cached['detections'])}",
                        flush=True,
                    )
                    continue
                image_path = ROOT / sample["output_image"]
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                try:
                    large_results[sample["sample_id"]] = large.infer(image)
                finally:
                    image.close()
                write_json_atomic(cache_path, large_results[sample["sample_id"]])
                print(
                    f"LARGE {index}/{len(selected)} {sample['sample_id']} "
                    f"detections={len(large_results[sample['sample_id']]['detections'])}",
                    flush=True,
                )
        finally:
            large.close()
            del large
            gc.collect()

    findings: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        image_path = ROOT / sample["output_image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            size = image.size
            quality = image_quality(image)
        base_result = base_results[sample_id]
        large_result = large_results.get(sample_id)
        final_result = large_result if large_result and large_result["detections"] else base_result
        escalated = large_result is not None
        route = ["florence-2-base-ft"] + (["florence-2-large-ft"] if escalated else [])
        model_id = str(final_result["model_id"])
        calibration_record = calibration_audit.get(
            sample_id, {"scale_status": "not_provided", "scale_method": "none"}
        )
        sample_findings: list[dict[str, Any]] = []
        rejected_generated_geometries: list[dict[str, Any]] = []
        for detection in final_result["detections"]:
            try:
                finding = build_finding(
                    run_id=run_id,
                    sample=sample,
                    image_path=image_path,
                    image_size=size,
                    defect_family=detection["defect_family"],
                    subtype=detection["subtype"],
                    box_xyxy=detection["box_xyxy"],
                    confidence_proxy=detection["confidence_proxy"],
                    pipeline_route=route,
                    escalation_state="completed" if escalated else "not_required",
                    checkpoint_sha256=checkpoint_hashes[model_id],
                    model_id=model_id,
                    quality=quality,
                    calibration=calibrations.get(sample_id),
                    unavailable_scale_status=str(
                        calibration_record.get("scale_status", "not_provided")
                    ),
                )
            except ValueError as exc:
                rejected_generated_geometries.append(
                    {"box_xyxy": detection.get("box_xyxy"), "reason": str(exc)}
                )
                continue
            sample_findings.append(finding)
            findings.append(finding)
        image_results.append(
            {
                "sample_id": sample_id,
                "status": "completed",
                "candidate_count": len(sample_findings),
                "no_visible_target_defect_candidate": not sample_findings,
                "finding_ids": [item["finding_id"] for item in sample_findings],
                "rejected_generated_geometries": rejected_generated_geometries,
                "quality": quality,
                "calibration": calibration_record,
                "base": base_result,
                "large": large_result,
                "final_model_id": model_id,
                "limitations": [
                    "No-candidate output is not evidence of structural safety.",
                    "Sequence likelihood is an uncalibrated routing proxy, not defect probability.",
                ],
            }
        )

    validate_findings(findings)
    write_jsonl(output_dir / "findings.jsonl", findings)
    write_jsonl(output_dir / "image_results.jsonl", image_results)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "scope": "training_only_feasibility_not_accuracy_evaluation",
        "status": "completed",
        "samples_processed": len(samples),
        "findings_emitted": len(findings),
        "images_escalated": len(escalated_ids),
        "images_with_candidates": sum(bool(row["candidate_count"]) for row in image_results),
        "images_without_candidates": sum(not row["candidate_count"] for row in image_results),
        "schema_valid_findings": len(findings),
        "rejected_generated_geometries": sum(
            len(row["rejected_generated_geometries"]) for row in image_results
        ),
        "physical_findings": sum(
            finding["measurement"]["scale_status"] == "valid" for finding in findings
        ),
        "escalation_policy": args.escalation,
        "uncertain_below_sequence_proxy": args.uncertain_below,
        "manifest": args.manifest.resolve().relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hashes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "outputs": {"findings": "findings.jsonl", "image_results": "image_results.jsonl"},
        "limitations": [
            "All source samples are training-only; this run cannot establish accuracy.",
            "Florence sequence likelihood is not a calibrated per-box confidence.",
            "Physical dimensions are absent unless calibration is explicitly valid.",
            "Candidate visible conditions require manual review and are not structural-safety determinations.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
