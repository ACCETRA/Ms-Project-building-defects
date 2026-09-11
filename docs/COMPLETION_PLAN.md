# Full Dataset Completion Plan

**Status:** Active execution plan
**Updated:** 2026-09-12
**Scope:** Full-data labeling, training, evaluation, and FYP beta completion

## 1. Current truth

The repository has completed the feasibility tier, not the full-data release:

- 300 training-only samples are materialized under `data/feasibility_v0/`.
- Detection, segmentation, classification, and Florence manifests contain 300 rows each.
- The normalized feasibility view contains 667 positive annotations.
- YOLO11n detection and YOLO11n-segmentation completed five-epoch feasibility runs.
- ResNet-50 completed a five-epoch feasibility run using stable FP32 CUDA optimization.
- Florence Base/Large feasibility inference infrastructure and a 300-sample run exist.
- CUBIT source, exact-dedup, and CODEBRIM classification registries exist.
- SAM 2.1 Tiny is available as a pretrained checkpoint, but the YOLO-to-SAM adapter is not yet an end-to-end project script.

The feasibility checkpoints are evidence that the stack runs. They are not final accuracy results.

## 2. What full dataset means

Do not merge every raw file into one untraceable folder. The final release is a versioned logical dataset with separate task views. Raw archives remain immutable.

The proposed first full release is approximately 20,000-25,000 selected images/tiles before augmentation:

| Source | Training/validation plan | Locked evaluation |
|---|---:|---:|
| CUBIT-InSeg | 5,035 train and 678 validation after exact deduplication | 694 leakage-clean test records |
| CiF tiled | Up to 12,000 train and 2,500 validation tiles | Source-specific test subset |
| S2DS | 563 train and 87 validation | 93 official test pairs |
| UAV75 | 50 train and 10 validation | 15 official test pairs |
| DACL10K | Up to 2,000 selected official-train images | 975 official validation images as bridge stress test |
| CODEBRIM | Official parent-safe classification partitions | Official classification test partition |
| Target site | Approved train/validation buildings only | Entire unseen buildings/sessions |

These are selection targets, not permission to include low-quality or redundant data. CUBIT test, S2DS test, UAV75 test, DACL validation, and target-site test data stay locked.

## 3. Work ownership

### Project owner

- Approve the final taxonomy, source budget, license use, and target-site permissions.
- Decide whether physical measurements are demonstrated; provide a valid scale method or accept pixel-only output.
- Approve the final model route and record the decision in `docs/DECISION_LOG.md`.
- Review final false negatives, limitations, and the FYP acceptance package.

### Data contributor

- Implement `scripts/build_v1_manifest.py`.
- Build `data/manifests/v1_master_manifest.csv` and the detection, segmentation, classification, and Florence task views.
- Complete source mapping, group IDs, exact/near-duplicate review, verified negatives, and annotation QA.
- Preserve native labels, mapping strength, source hashes, license IDs, and annotation provenance.
- Do not turn CODEBRIM image/crop labels or boxes into segmentation ground truth.

Existing data inputs are in `data/manifests/`, `data/feasibility_v0/`, `datasets/`, and `artifacts/data-audit/`.

### Model contributor

- Implement or extend `scripts/build_v1_training_inputs.py` to validate all task views and write reproducible YOLO data files.
- Run `scripts/train_yolo_comparison.py` for detection and segmentation using the frozen v1 manifests.
- Run `scripts/train_resnet_comparison.py` using the frozen classification manifest.
- Implement the YOLO-box to SAM 2.1 Tiny adapter and write outputs through the finding schema.
- Record checkpoints, seeds, resolution, split counts, losses, metrics, latency, VRAM, and failure states.
- Run the Florence Base/Large comparison route using `scripts/run_florence_pipeline.py`; Florence fine-tuning is optional and is not currently implemented.

Model entry points are `scripts/train_yolo_comparison.py`, `scripts/train_resnet_comparison.py`, and `scripts/run_florence_pipeline.py`. Current feasibility outputs are under `runs/comparison/` and `runs/florence/`.

### Product/evaluation contributor

- Implement the locked-test evaluator, initially planned as `scripts/evaluate_cubit_test.py`.
- Freeze thresholds only on training/validation data, then evaluate CUBIT and source-specific tests.
- Review false negatives, false positives, empty results, rejected images, and out-of-domain cases.
- Build the local upload, processing, review, and JSON/CSV/annotated-image/PDF-style export workflow.
- Record target-site metadata, calibration, review identity, and model/data provenance.

Product contracts are `docs/BETA_SPECIFICATION.md`, `docs/OUTPUT_CONTRACT.md`, and `schemas/finding_record.schema.json`.

## 4. Required implementation sequence

1. Approve v1 source budget, taxonomy mappings, licenses, negatives, and target-site permissions.
2. Implement and run the full manifest builder.
3. Review per-source/per-class counts and complete annotation QA.
4. Freeze group-safe train/validation/test views and hashes.
5. Prepare YOLO detection and segmentation directories from the frozen views.
6. Train YOLO11n detection, YOLO11n-segmentation, and ResNet-50.
7. Implement and run YOLO boxes -> SAM 2.1 Tiny masks.
8. Run Florence Base and validation-defined Large escalation on the same approved inputs.
9. Select thresholds and checkpoints using validation only.
10. Evaluate locked CUBIT, source-specific tests, CODEBRIM classification test, and target-site buildings.
11. Build and test the browser workflow, reviewer actions, and exports.
12. Update metrics, limitations, licenses, model hashes, and the final FYP report.

## 5. Exact missing files and outputs

The following planned full-data files do not currently exist and must be created or explicitly replaced by an equivalent:

- `scripts/build_v1_manifest.py`
- `scripts/build_v1_training_inputs.py`
- `scripts/evaluate_cubit_test.py`
- `data/manifests/v1_master_manifest.csv`
- `data/manifests/v1_detection_manifest.csv`
- `data/manifests/v1_segmentation_manifest.csv`
- `data/manifests/v1_classification_manifest.csv`
- full-data outputs under `data/v1/`
- YOLO-to-SAM inference output under `runs/comparison/`
- locked-test metrics under `runs/evaluation/`

Do not edit raw archives under `datasets/` or locked test content to create these outputs.

## 6. Verification gate

Before calling the project complete, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_markdown_links.py
.\.venv\Scripts\python.exe scripts/verify_model_assets.py
.\.venv\Scripts\python.exe scripts/evaluate_cubit_test.py
 git diff --check
```

Also verify that every final report identifies the dataset manifest, split policy, checkpoint hash, taxonomy version, source licenses, and whether measurements are pixel-only or calibrated.

## 7. Six-hour feasibility with three machines

Six hours is enough for a coordinated smoke release, not for the full project as defined here.

Three machines can parallelize:

- manifest construction and label QA;
- YOLO/ResNet feasibility or reduced v1 training;
- Florence inference and evaluation preparation.

Six hours is not enough to guarantee all of the following on the current data: full v1 annotation QA, 20-epoch YOLO detection and segmentation, 10-epoch ResNet training, Florence Base/Large evaluation, YOLO-to-SAM integration, locked-test evaluation, target-site validation, and a tested browser product.

A truthful six-hour milestone is:

- freeze a smaller v1 subset;
- produce and validate task manifests;
- launch or complete selected baseline runs;
- implement a narrow YOLO-to-SAM smoke path;
- update reproducibility records and identify remaining blockers.

The project may be demonstrated in six hours only by explicitly calling it a feasibility/demo checkpoint. It cannot honestly be called full-dataset completion or final accuracy validation without the gates above.
