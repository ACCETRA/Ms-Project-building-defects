# First Execution Gate — Historical Record

**Status:** Completed (Archived). All subsequent gates, full-data baselines, evaluations, and the FYP acceptance package have been completed.

See [`docs/FYP_COMPLETION_CHECKLIST.md`](FYP_COMPLETION_CHECKLIST.md) and [`docs/COMPLETION_PLAN.md`](COMPLETION_PLAN.md) for the final completion record. This file is retained as an archive of the initial planning gate.

## Objective

Initial gate completed; all deliverables across v1 task views, baseline training, YOLO-to-SAM, locked source evaluation, and browser review/export workflows have been delivered.

## Data lead

- [Completed] Run and review `scripts/audit_datasets.py`.
- [Completed] Verify archive readability and licenses.
- [Completed] Verify both CODEBRIM archives with portable 7-Zip and decoded image probes; standard ZIP readers remain disallowed for these legacy headers.
- [Completed] Acquire, CRC-test, hash, count, and pair all six official CUBIT image/label archives while keeping the test split locked.
- [Completed] Draft the first selection budget by source, class, domain, annotation type, and negative type.
- [Completed] Propose group IDs for parent images, buildings, bridges, UAV flights, and related frames.

## Model lead

- [Completed] Create the isolated CUDA-enabled Python 3.11 environment.
- [Completed] Verify CUDA on the Quadro T2000.
- [Completed] Confirm ResNet-50 rather than RetinaNet as the classification baseline.
- [Completed] Review and acquire the model/checkpoint choices in `docs/EXPERIMENT_MATRIX.md`.
- [Completed] Define the common detector and segmentation evaluation inputs.
- [Completed] Select and materialize 300 training-only feasibility images after provenance review.
- [Completed] Normalize the 300-sample detection, segmentation, and classification labels.
- [Completed] Build full CUBIT train/validation labels, exact-dedup decisions, and CODEBRIM classification labels for the acquired sources.
- [Completed] Run YOLO11n and ResNet-50 comparison training jobs. All runs finished on CUDA: YOLO11n detection, YOLO11n-seg, and ResNet-50 classification.

## Product/evaluation lead

- [Completed] Convert `docs/BETA_SPECIFICATION.md` into the user-flow checklist.
- [Completed] Review the structured finding fields for the FYP demonstration.
- [Completed] Define the inspection PDF/CSV/JSON content contract.
- [Completed] Define operational measurements: latency, false alarms per image, escalation rate, and review time.
- [Completed] Document target-domain boundary and metadata.

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
- [Completed] Identify demonstration images and record their provenance/permissions.
