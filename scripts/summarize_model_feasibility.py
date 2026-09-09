"""Aggregate per-model CUDA smoke reports without treating them as accuracy."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "artifacts" / "model-feasibility"
EXPECTED = (
    "yolo11n",
    "yolo11n_seg",
    "resnet50",
    "sam2_1_hiera_tiny",
    "florence_2_base_ft",
    "florence_2_large_ft",
)


def main() -> None:
    rows: list[dict[str, object]] = []
    source_reports: list[dict[str, object]] = []
    for model_id in EXPECTED:
        path = INPUT_ROOT / f"{model_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["model_id"] != model_id:
            raise ValueError(f"Model ID mismatch in {path}")
        if report["scope"] != "cuda_hardware_smoke_no_training_not_accuracy":
            raise ValueError(f"Unexpected report scope in {path}")
        source_reports.append(report)
        rows.append(
            {
                "model_id": model_id,
                "status": report["status"],
                "checkpoint": report.get("checkpoint", ""),
                "dtype": report.get("dtype", ""),
                "requested_image_size": json.dumps(report.get("requested_image_size")),
                "cold_inference_seconds": report.get("cold_inference_seconds", ""),
                "warm_inference_seconds": report.get("warm_inference_seconds", ""),
                "peak_allocated_vram_bytes": report.get("peak_allocated_vram_bytes", ""),
                "peak_reserved_vram_bytes": report.get("peak_reserved_vram_bytes", ""),
                "sample_id": report["sample"]["sample_id"],
                "split": report["sample"]["split"],
            }
        )

    sample_ids = {report["sample"]["sample_id"] for report in source_reports}
    splits = {report["sample"]["split"] for report in source_reports}
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "cuda_hardware_smoke_no_training_not_accuracy",
        "all_expected_reports_present": len(source_reports) == len(EXPECTED),
        "all_passed": all(report["status"] == "passed" for report in source_reports),
        "training_only_sample_confirmed": splits == {"train"},
        "sample_ids": sorted(sample_ids),
        "warning": (
            "These timings are single-image smoke measurements, not throughput, "
            "quality, model-selection, or accuracy results."
        ),
        "models": rows,
    }

    json_path = INPUT_ROOT / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = INPUT_ROOT / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["all_passed"] and summary["training_only_sample_confirmed"] else 1)


if __name__ == "__main__":
    main()
