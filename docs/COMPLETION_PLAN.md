# FYP Completion Record

**Status:** Complete  
**Accepted scope:** Demo-only academic FYP prototype  
**Recorded:** 2026-09-13

The approved FYP workflow is implemented and verified with real model execution. This file is retained as the completion record for scripts and links that used its earlier filename.

## Completed implementation

- [x] Dataset sources audited without modifying raw archives.
- [x] Dataset taxonomy and source-label mappings frozen.
- [x] Leakage-controlled v1 manifests generated and hashed.
- [x] YOLO11n detector trained and checkpointed.
- [x] YOLO11n segmenter trained and checkpointed.
- [x] ResNet-50 multilabel classifier trained and checkpointed.
- [x] Florence-2 local inference pipeline executed.
- [x] YOLO-box-to-SAM mask route executed.
- [x] Locked CUBIT and source-specific evaluations recorded.
- [x] Threshold-selection evidence recorded.
- [x] Browser upload, real inference, review, persistence, and exports verified.
- [x] Structured finding records validated against the schema.
- [x] Human image-level error review recorded.
- [x] Acceptance package generated as `artifacts/fyp.docx`.
- [x] Offline contributor bundle scripts, dependencies, weights, and verification manifest prepared.

## Evidence locations

- `data/manifests/`
- `artifacts/data-audit/`
- `artifacts/model-assets/`
- `artifacts/model-feasibility/`
- `runs/evaluation/`
- `runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt`
- `runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt`
- `runs/comparison/resnet50_v1_queue/resnet50_comparison.pt`
- `docs/V1_COMPARISON_REPORT.md`
- `docs/HUMAN_ERROR_REVIEW.md`
- `artifacts/fyp.docx`

## Acceptance boundary

The delivered system is an academic prototype for demonstration and evaluation. It does not provide autonomous structural conclusions, repair prescriptions, code-compliance certification, or proof of safety. Manual review is required for every finding, and physical units require valid calibration.
