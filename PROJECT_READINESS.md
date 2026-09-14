# Building Defect Inspection FYP Readiness

**Status:** Complete for the accepted academic demonstration
**Completion date:** 2026-09-13

The repository contains a complete local browser demonstration for visible building-condition inspection. It runs real YOLO detection and segmentation, ResNet-50 multilabel classification, bounded Florence structured inference, and a YOLO-to-SAM mask route. Model predictions remain immutable while reviewer decisions are stored separately.

## Delivered evidence

- Frozen, versioned dataset manifests and six-class taxonomy
- Trained YOLO11n detection and segmentation checkpoints
- Trained ResNet-50 multilabel checkpoint
- Local Florence and SAM model assets
- Locked CUBIT and official CODEBRIM evaluations
- Source-specific CiF, S2DS, DACL10K, and UAV75 evaluations
- Human image-level error review record
- Browser upload, inference, review, persistence, and export workflow
- Offline Python installer, dependency wheels, model weights, and verification scripts
- Exact dataset acquisition and training-input generation workflow
- Consolidated FYP report at `artifacts/fyp.docx`

## Headline results

| Evaluation | Samples | Result |
|---|---:|---:|
| CUBIT locked YOLO detection | 701 | 63.165% mAP@50 |
| CUBIT locked YOLO detection | 701 | 54.107% mAP@50:95 |
| CUBIT locked YOLO segmentation masks | 701 | 58.341% mAP@50 |
| ResNet group-safe validation | 1,104 | 99.330% micro-F1 |
| CODEBRIM official ResNet test | 632 | 36.641% micro-F1 |

The validation and test values are intentionally kept separate. The detailed metrics, source-specific results, thresholds, and provenance are in [`docs/V1_COMPARISON_REPORT.md`](docs/V1_COMPARISON_REPORT.md).

## Run readiness

The dataset-free handoff can process uploaded JPEG and PNG images without network access. Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_offline_env.ps1
.\.venv\Scripts\python.exe harness\server.py
```

Raw datasets are required only for regeneration, retraining, or benchmark evaluation. Their complete workflow is documented in [`docs/DATASET_REPRODUCTION_GUIDE.md`](docs/DATASET_REPRODUCTION_GUIDE.md).

## Scope boundary

This is an academic demonstration, not a production inspection or structural-safety system. Every finding requires manual review. Physical measurements remain pixel-only unless the input contains valid calibration metadata.

Optional accuracy recommendations are recorded in [`docs/RECOMMENDED_IMPROVEMENTS.md`](docs/RECOMMENDED_IMPROVEMENTS.md) and do not block completion.
