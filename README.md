# Building Defect Inspection FYP

This repository contains the completed demo-only master's FYP for visible building-condition inspection. The local browser workflow runs real model inference, keeps model predictions immutable, stores human review separately, and exports JSON, CSV, annotated-image, and report evidence.

## Completion status

Every item in the accepted FYP demo scope is complete:

- [x] Frozen, versioned dataset manifests and taxonomy.
- [x] YOLO11n detection and segmentation training checkpoints.
- [x] ResNet-50 multilabel classification checkpoint.
- [x] Florence-2 local inference route.
- [x] YOLO-to-SAM mask-generation route.
- [x] Locked and source-specific evaluation evidence.
- [x] Browser upload, inference, review, persistence, and export workflow.
- [x] Human image-level error review recorded by owner attestation.
- [x] Offline contributor package, dependency installer, model assets, trained harness checkpoints, and dataset-acquisition script.
- [x] Consolidated completion document at `artifacts/fyp.docx`.

The system is an academic demonstration. It does not make structural-safety determinations, and physical measurements are pixel-only unless valid calibration metadata is supplied.

## Run the demo

On the original workstation:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

Open `http://127.0.0.1:8765` and select a model route.

On an offline contributor machine, extract the transfer ZIP and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_offline_env.ps1
.\.venv\Scripts\python.exe harness\server.py
```

See [`docs/OFFLINE_BUNDLE_GUIDE.md`](docs/OFFLINE_BUNDLE_GUIDE.md) and [`docs/FYP_COMPLETION_CHECKLIST.md`](docs/FYP_COMPLETION_CHECKLIST.md).

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
powershell -ExecutionPolicy Bypass -File scripts\verify_offline_bundle.ps1
```

## Dataset restoration

Raw datasets are not placed in the transfer ZIP because CODEBRIM and S2DS prohibit unrestricted redistribution and the local source collection is 60.9 GiB. On a connected machine, accept the source terms and restore the exact project layout with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1 -AcceptLicenses
```

The downloader pins the same source revisions, filenames, split selection, sizes, and published/local checksums used by this project. Dataset terms are retained under `datasets/`.

## Key records

- [`docs/BETA_SPECIFICATION.md`](docs/BETA_SPECIFICATION.md): accepted product boundary.
- [`docs/FYP_COMPLETION_CHECKLIST.md`](docs/FYP_COMPLETION_CHECKLIST.md): completed handoff checklist.
- [`docs/V1_COMPARISON_REPORT.md`](docs/V1_COMPARISON_REPORT.md): measured model evidence.
- [`docs/HUMAN_ERROR_REVIEW.md`](docs/HUMAN_ERROR_REVIEW.md): human-review completion record.
- [`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md): finding and review schema boundary.
- [`artifacts/model-assets/model_asset_inventory.json`](artifacts/model-assets/model_asset_inventory.json): model asset hashes.
- [`artifacts/data-audit/dataset_audit.json`](artifacts/data-audit/dataset_audit.json): dataset inventory and audit evidence.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for repository guardrails and reproducibility rules.
