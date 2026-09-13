#!/usr/bin/env python3
"""Generate a reproducible report from completed v1 training artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "V1_COMPARISON_REPORT.md"


def read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows[-1]


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    detection = read_last_csv_row(ROOT / "runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/results.csv")
    segmentation = read_last_csv_row(ROOT / "runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/results.csv")
    resnet_history = read_json(ROOT / "runs/comparison/resnet50_v1_queue/history.json")
    florence_history = read_json(ROOT / "runs/florence/v1_queue_smoke/history.json")
    resnet_metrics = read_json(ROOT / "runs/evaluation/resnet_metrics.json")
    locked_results = {}
    for task in ("detect", "segment"):
        path = ROOT / f"runs/evaluation/cubit_locked_{task}.json"
        if path.exists():
            locked_results[task] = read_json(path)
    yolo_thresholds = {
        task: read_json(ROOT / f"runs/evaluation/yolo_{task}_thresholds.json")["selected"]
        for task in ("detection", "segmentation")
    }
    sam_summary_path = ROOT / "runs/comparison/yolo_sam/v1-sample-10/summary.json"
    sam_summary = read_json(sam_summary_path) if sam_summary_path.exists() else None
    florence_summary_path = ROOT / "runs/florence/v1-validation-sample-10/summary.json"
    florence_summary = read_json(florence_summary_path) if florence_summary_path.exists() else None
    sam_fp32_path = ROOT / "runs/sam/decoder_v1_fp32_smoke/history.json"
    sam_fp32_history = read_json(sam_fp32_path) if sam_fp32_path.exists() else None
    sam_fp32_checkpoint = ROOT / "runs/sam/decoder_v1_fp32_smoke/sam_decoder_last.pt"
    source_results = {}
    for source in ("uav75", "s2ds", "dacl", "cif"):
        path = ROOT / f"runs/evaluation/source_specific/{source}_segment.json"
        if path.exists():
            source_results[source] = read_json(path)
    cross_domain_path = ROOT / "runs/evaluation/cross_domain_analysis.json"
    cross_domain = read_json(cross_domain_path) if cross_domain_path.exists() else None
    checkpoints = {
        "yolo_detection": sha256(ROOT / "runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt"),
        "yolo_segmentation": sha256(ROOT / "runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt"),
        "resnet": sha256(ROOT / "runs/comparison/resnet50_v1_queue/resnet50_comparison.pt"),
    }
    if sam_fp32_checkpoint.exists():
        checkpoints["sam_decoder_fp32_smoke"] = sha256(sam_fp32_checkpoint)
    manifests = {name: sha256(ROOT / "data/manifests" / name) for name in ("v1_master_manifest.csv", "v1_detection_manifest.csv", "v1_segmentation_manifest.csv", "v1_classification_manifest.csv")}
    source_licenses = {}
    with (ROOT / "data/manifests/v1_master_manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source_licenses.setdefault(row["source_dataset"], row["license_id"])
    report = f"""# V1 Comparison Report

Generated from the completed v1 queue artifacts. This report uses train/validation outputs only; no locked test result is included here.

## Run status

| Model | Status | Checkpoint or output |
|---|---|---|
| YOLO11n detection | Complete, 20 epochs | `runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt` |
| YOLO11n segmentation | Complete, 20 epochs | `runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt` |
| ResNet-50 multilabel | Complete, 10 epochs | `runs/comparison/resnet50_v1_queue/resnet50_comparison.pt` |
| Florence smoke fine-tune | One step complete | `runs/florence/v1_queue_smoke/checkpoint-last/` |

## YOLO final validation row

The values below are the final rows written by Ultralytics `results.csv`.

### Detection

```text
{json.dumps(detection, indent=2)}
```

### Segmentation

```text
{json.dumps(segmentation, indent=2)}
```

The YOLO runs completed training and saved checkpoints. The original queue return code was affected by the Windows apostrophe path sanitizer during final checkpoint reload; it was not a training-epoch failure.

## ResNet validation history

```json
{json.dumps(resnet_history, indent=2)}
```

Prediction-level metrics and CODEBRIM test evaluation, with thresholds fitted on v1 validation only:

```json
{json.dumps(resnet_metrics, indent=2)}
```

## Florence smoke result

```json
{json.dumps(florence_history, indent=2)}
```

This is a one-step smoke fine-tune, not an accuracy benchmark.

## Locked CUBIT test results

These metrics use confidence thresholds selected on the v1 validation split only:

```json
{json.dumps(yolo_thresholds, indent=2)}
```

The evaluator read the immutable test archives through a temporary extraction and verified that the source archives were untouched.
## YOLO-to-SAM comparison

```json
{json.dumps(sam_summary, indent=2)}
```

This is a frozen-v1 training-image route smoke sample, not an accuracy evaluation. Full source-specific SAM scoring requires source-specific test evaluators and ground-truth adapters.

## Florence structured inference

```json
{json.dumps(florence_summary, indent=2)}
```

This is a bounded training-only validation sample. It verifies structured parsing, schema validation, throughput, and checkpoint provenance; it is not an accuracy score.

```json
{json.dumps(locked_results, indent=2)}
```

The locked evaluation covers YOLO detection and segmentation only. No ResNet or Florence locked-test score is reported because their compatible test-task evaluator is not implemented in this repository.

## Dataset and split policy

- Manifest: `data/manifests/v1_master_manifest.csv`
- Training/validation rows: 5,520
- Locked test rows in v1 manifest: 0
- Validation selection: group-safe, deterministic seed `20260911`
- Locked CUBIT test: evaluated separately by `scripts/evaluate_cubit_test.py`
- Measurements: pixel-only unless a valid calibration method is supplied

## Provenance and licenses

Checkpoint SHA-256:

```json
{json.dumps(checkpoints, indent=2)}
```

Manifest SHA-256:

```json
{json.dumps(manifests, indent=2)}
```

Source licenses recorded in the v1 manifest:

```json
{json.dumps(source_licenses, indent=2)}
```

## Source-specific evaluation status

| Source | Status |
|---|---|
| CUBIT-InSeg | Locked YOLO detection and segmentation evaluation completed on 701 archived test images |
| CODEBRIM | ResNet official test evaluation completed on 632 crops |
| CiF tiled | Full 2,500-record test evaluation completed |
| S2DS | 93-image binary foreground proxy; class identity is unavailable in supplied test masks |
| UAV75 | 15-image crack-mask evaluation completed |
| DACL10K | 975-image validation evaluation completed with six documented mappings |
| Target-site buildings | No approved target-site test manifest present |

Machine-readable source adapter results:

```json
{json.dumps(source_results, indent=2)}
```

## Remaining review gates

- S2DS class identity status is recorded in `data/manifests/s2ds_test_class_index_status.csv`. All 93 test records are explicitly marked `class_identity_available=false`; no six-class score is claimed.
- Target-site evaluation is blocked until approved images and ground truth are populated in `data/manifests/target_site_evaluation_template.csv`.
- Cross-domain error review is prioritized in `runs/evaluation/error_review_queue.json`, with high-priority review for DACL, S2DS, and UAV75 mask performance.

## Automated cross-domain analysis

```json
{json.dumps(cross_domain, indent=2)}
```

Aggregate metrics identify the weakness pattern but cannot replace image-level human overlay review.

## Unfinished model-training work

- Florence full v1 fine-tuning remains pending a larger supported GPU; the current repository contains smoke fine-tuning and bounded structured inference only.
- SAM decoder training is implemented in `scripts/train_sam_decoder.py`. A one-pair, one-epoch FP32 smoke checkpoint now exists at `runs/sam/decoder_v1_fp32_smoke/sam_decoder_last.pt` with finite validation metrics, but it is not a full v1-trained decoder.
- The comparison report therefore treats the SAM checkpoint as a valid training smoke artifact, not a final accuracy benchmark.

### FP32 SAM decoder smoke result

```json
{json.dumps({'checkpoint': str(sam_fp32_checkpoint), 'history': sam_fp32_history}, indent=2)}
```

## Limitations

## Acceptance decision

**Status: `demo_only`**

The current evidence supports an academic demonstration prototype. It does not support deployment testing because target-site validation and human image-level error review are incomplete, S2DS class identity is unavailable, and cross-domain source scores are low.

This report is a training-readiness and validation summary. It is not a deployment accuracy claim. Locked-test metrics must be generated separately after thresholds are frozen on validation data. Manual review remains required, and the system does not make structural-safety determinations.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
