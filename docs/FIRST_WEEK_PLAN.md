# First Execution Gate — Three-Person Team

**Status:** Feasibility gate closed; full-data execution gate open. Comparison feasibility training is complete, but the FYP is not complete.

Use [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) as the current execution checklist. This file records the first gate; it does not claim that full-data manifests, locked evaluation, or the product workflow are finished.

## Objective

The first gate is complete. The next gate is to freeze v1 task views, run full-data baselines, integrate YOLO-to-SAM, evaluate locked sources, and finish the real-inference review/export demonstration.

## Data lead

- [Completed] Run and review `scripts/audit_datasets.py`.
- [Technical verification completed; legal approval pending] Verify archive readability and licenses.
- [Completed] Verify both CODEBRIM archives with portable 7-Zip and decoded image probes; standard ZIP readers remain disallowed for these legacy headers.
- [Completed] Acquire, CRC-test, hash, count, and pair all six official CUBIT image/label archives while keeping the test split locked.
- Draft the first selection budget by source, class, domain, annotation type, and negative type.
- Propose group IDs for parent images, buildings, bridges, UAV flights, and related frames.

## Model lead

- [Completed] Create the isolated CUDA-enabled Python 3.11 environment.
- [Completed] Verify CUDA on the Quadro T2000.
- [Completed] Confirm ResNet-50 rather than RetinaNet as the classification baseline.
- [Completed for feasibility] Review and acquire the model/checkpoint choices in `docs/EXPERIMENT_MATRIX.md`.
- Define the common detector and segmentation evaluation inputs.
- [Completed] Select and materialize 300 training-only feasibility images after provenance review.
- [Completed] Normalize the 300-sample detection, segmentation, and classification labels.
- [Completed] Build full CUBIT train/validation labels, exact-dedup decisions, and CODEBRIM classification labels for the acquired sources.
- [Completed] Run short YOLO11n and ResNet-50 comparison training jobs. All three 5-epoch runs finished on CUDA (FP32): YOLO11n detection (`runs/comparison/yolo/yolo11n_detect_fyp/weights/best.pt`), YOLO11n-seg (`runs/comparison/yolo/yolo11n_seg_fyp/weights/best.pt`), and ResNet-50 classification (`runs/comparison/resnet50_fyp/resnet50_comparison.pt`). SAM 2.1 remains pretrained for the first comparison.

## Product/evaluation lead

- Convert `docs/BETA_SPECIFICATION.md` into the first user-flow/wireframe checklist.
- Review the structured finding fields for the FYP demonstration.
- Draft the inspection PDF/CSV/JSON content contract.
- Define preliminary operational measurements: latency, false alarms per image, escalation rate, and review time.
- Plan target-building capture permissions and metadata.

## End-of-gate evidence

- Approved beta specification.
- Taxonomy and known limitations documented for the FYP demonstration.
- Dataset inventory and license table.
- Archive health report.
- Proposed curated subset and frozen test sources.
- Exact model/checkpoint matrix.
- CUDA-enabled environment verification.
- Feasibility sample manifest.
- Prototype workflow and output-record review.

## Blocking decisions for the owner/advisor

- [Completed] Confirm ResNet-50 rather than RetinaNet as the classification baseline.
- [Completed] Confirm open-source distribution under AGPL-compatible terms if Ultralytics YOLO remains in the beta.
- Identify demonstration images and record their provenance/permissions.
