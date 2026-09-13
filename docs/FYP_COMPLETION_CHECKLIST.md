# FYP Completion Checklist

**Project:** Building Defect Inspection  
**Status:** Complete  
**Handoff date:** 2026-09-13

## Product and interface

- [x] Local browser application starts from `harness/server.py`.
- [x] JPEG and PNG images can be uploaded.
- [x] Real YOLO detection, YOLO segmentation, ResNet classification, and Florence routes are selectable.
- [x] Empty results and model failures are represented honestly.
- [x] Predictions remain immutable while reviewer decisions are stored separately.
- [x] JSON, CSV, annotated-image, and browser-report exports work.
- [x] The interface identifies the system as a local FYP demo using real model inference.

## Models and evidence

- [x] All pinned base-model assets are stored under `weights/` with an inventory and hashes.
- [x] The trained YOLO detection checkpoint used by the harness is included.
- [x] The trained YOLO segmentation checkpoint used by the harness is included.
- [x] The trained ResNet-50 checkpoint used by the harness is included.
- [x] Florence and SAM local assets are included.
- [x] Locked-test, source-specific, cross-domain, and threshold-selection outputs are included.
- [x] The human error review is documented in `docs/HUMAN_ERROR_REVIEW.md`.

## Reproducibility and transfer

- [x] Source code, schemas, configuration, manifests, tests, and documentation are included.
- [x] Python 3.11 runtime dependencies are pinned.
- [x] Offline CUDA PyTorch and torchvision wheels are included.
- [x] The offline setup script creates and verifies a clean environment.
- [x] The dataset downloader recreates the source layout used by this project.
- [x] Restricted datasets require explicit terms acceptance and are not redistributed in the ZIP.
- [x] The archive contains a SHA-256 file manifest and a verification script.
- [x] The consolidated Word completion document is named `fyp.docx`.

## Safety and reporting

- [x] Outputs are described as visible-condition candidates requiring manual review.
- [x] No absence-of-detection result is presented as proof of structural safety.
- [x] Physical values are withheld unless calibration is valid.
- [x] Dataset licenses and provenance records are retained.
- [x] The accepted FYP scope is marked complete consistently in presentation-facing documentation.
