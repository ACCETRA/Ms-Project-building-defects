# Dataset Reproduction Guide

**FYP status:** Complete  
**Purpose:** Restore, configure, generate, and verify the exact project data layout without placing raw datasets in the handoff package

## Storage requirement

The acquired raw sources occupy approximately 60.9 GiB. Materialized training data can require approximately 44.7 GiB more. Allow at least 115 GiB of free space for a full restoration and regeneration run.

Keep the extracted project at a short path without an apostrophe, for example `D:\ProjectAlpha`. Every generated path is relative to the project root.

## One-command preparation

Run the following from the project root on a machine with internet access:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

The script performs the full sequence:

1. Downloads each approved source into its fixed repository-relative location.
2. Verifies available checksums, archive sizes, shard counts, and source structure.
3. Audits native class counts and source inventories.
4. Recreates the deterministic 300-sample feasibility materialization.
5. Recreates normalized annotations and feasibility task views.
6. Rebuilds the CUBIT source registry and duplicate audit.
7. Rebuilds the exact-dedup selection and CODEBRIM classification registry.
8. Recreates the frozen v1 manifests without locked-test records.
9. Generates YOLO detection, YOLO segmentation, ResNet classification, and Florence training inputs.

Use `-SkipDownload` when the raw sources already exist in the correct locations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -SkipDownload
```

Use `-ForceDownload` only when an existing source must be reacquired:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses -ForceDownload
```

The command above is intentionally destructive only to downloader-managed source targets selected for reacquisition. Do not use it when locally retained source files have not been backed up.

## Raw dataset locations

| Source | Storage path |
|---|---|
| CiF tiled | `datasets/CiF-tiled/` |
| DACL10K | `datasets/DACL10K/dacl10k_v2_devphase.zip` |
| CODEBRIM originals | `datasets/CODEBRIM/CODEBRIM_original_images.zip` |
| CODEBRIM classification | `datasets/CODEBRIM/CODEBRIM_classification_dataset.zip` |
| S2DS | `datasets/building-target/S2DS/s2ds.zip` |
| UAV75 | `datasets/UAV-candidates/UAV75/` |
| CUBIT-InSeg | `datasets/building-target/CUBIT-InSeg/CUBIT-InSeg/` |

CODEBRIM and S2DS require explicit acceptance of their source terms. The handoff package includes their metadata and license records but not the dataset bytes.

## Generated locations

| Output | Storage path |
|---|---|
| Dataset audit | `artifacts/data-audit/` |
| Frozen manifests | `data/manifests/` |
| Feasibility images and annotations | `data/feasibility_v0/` |
| Normalized v1 images and annotations | `data/v1/images/` and `data/v1/annotations/` |
| YOLO detection input | `data/v1/training_inputs/yolo_detection/` |
| YOLO segmentation input | `data/v1/training_inputs/yolo_segmentation/` |
| ResNet classification manifest | `data/v1/training_inputs/classification_manifest_v1_0_0.csv` |
| Florence targets | `data/v1/training_inputs/florence_targets_v1_0_0.jsonl` |

Raw downloads remain immutable. Generated crops, masks, labels, manifests, and training inputs are stored outside the raw-source locations.

## Manual commands

The exact equivalent commands are:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1 -AcceptLicenses
.\.venv\Scripts\python.exe scripts\audit_datasets.py
.\.venv\Scripts\python.exe scripts\build_feasibility_manifest.py
.\.venv\Scripts\python.exe scripts\materialize_feasibility_set.py
.\.venv\Scripts\python.exe scripts\build_feasibility_task_views.py
.\.venv\Scripts\python.exe scripts\build_cubit_registry.py
.\.venv\Scripts\python.exe scripts\audit_cubit_duplicates.py
.\.venv\Scripts\python.exe scripts\build_cubit_exact_dedup_manifest.py
.\.venv\Scripts\python.exe scripts\build_codebrim_registry.py
.\.venv\Scripts\python.exe scripts\build_v1_manifest.py
.\.venv\Scripts\python.exe scripts\build_v1_training_inputs.py
```

The reviewed CUBIT near-duplicate decision queue is included in `artifacts/data-audit/` and is consumed by `build_v1_manifest.py`. The v1 builder excludes unresolved or reviewed duplicate non-test records and never adds the locked CUBIT test to training or validation.

## Verification

After preparation, run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
Get-Content artifacts\data-audit\dataset_audit.json
Get-Content artifacts\data-audit\v1_manifest_report.json
Get-Content artifacts\data-audit\v1_training_inputs_report.json
```

The v1 manifest report must show zero locked-test records. Dataset source terms continue to govern use and redistribution after download.
