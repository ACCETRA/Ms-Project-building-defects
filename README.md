# Building Defect Inspection — Master's FYP

Project Alpha is a completed academic demonstration for visible building-condition inspection. It provides a local browser application with real YOLO detection and segmentation, ResNet-50 multilabel classification, a bounded Florence-2 route, optional SAM mask generation, immutable predictions, separate human review, and JSON/CSV/image/report exports.

The project is a research prototype, not a structural-safety system. Every finding requires manual review. Physical dimensions are withheld unless valid calibration metadata is supplied.

## Repository contents

- `bdi/`: shared inference, taxonomy, schema, and measurement code.
- `harness/`: local browser interface and HTTP API.
- `scripts/`: dataset acquisition, audit, normalization, training, evaluation, reporting, packaging, and verification commands.
- `config/`: pinned model and experiment configuration.
- `schemas/`: finding and calibration JSON schemas.
- `data/manifests/`: frozen, dataset-free source and task manifests.
- `tests/`: preparation, schema, inference-contract, and harness tests.
- `docs/`: completion, evaluation, dataset, environment, and improvement reports.
- `artifacts/fyp.docx`: consolidated completion and measured-results report.

Raw datasets, generated training images, `.venv`, model caches, and large checkpoint directories are intentionally excluded from Git.

## Requirements

- Windows 10 or 11 with PowerShell 5.1 or newer.
- Python 3.11 (the recorded environment used Python 3.11.9).
- At least 12 GiB free for the runtime and local model assets.
- At least 115 GiB free only when restoring every dataset and regenerating training inputs.
- An NVIDIA CUDA-capable GPU is recommended for Florence, SAM, training, and practical inference speed. The recorded environment used PyTorch 2.6.0 with CUDA 12.4.

Keep the checkout at a short path without an apostrophe, for example `D:\Ms-project`, because Ultralytics may sanitize apostrophes in absolute checkpoint paths.

## 1. Clone and create the environment

```powershell
git clone https://github.com/y5u82/Ms-project.git
cd Ms-project
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

For CPU-only use, install the matching CPU builds of PyTorch and torchvision instead. Model training and the large Florence route will be slow or impractical without a suitable GPU.

An offline transfer package can use `scripts/setup_offline_env.ps1` when `vendor/wheels/`, `weights/`, and the selected trained checkpoints have been copied from the verified project handoff. Those large binary assets are not stored in this Git repository.

## 2. Place runtime model assets

The application expects the following local paths:

```text
weights/yolo/yolo11n.pt
weights/yolo/yolo11n-seg.pt
weights/resnet/resnet50-11ad3fa6.pth
weights/sam2.1-hiera-tiny/
weights/florence-community-2-base-ft/
weights/florence-community-2-large-ft/
runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt
runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt
runs/comparison/resnet50_v1_queue/resnet50_comparison.pt
```

The trained project checkpoints must be copied from the verified FYP handoff. Public base assets use the identifiers recorded in `config/model_candidates.yaml`:

- `florence-community/Florence-2-base-ft`
- `florence-community/Florence-2-large-ft`
- `facebook/sam2.1-hiera-tiny`
- Ultralytics `yolo11n.pt` and `yolo11n-seg.pt`
- torchvision ResNet-50 default weights

Verify all locally placed assets against `artifacts/model-assets/model_asset_inventory.json`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\verify_model_assets.py
```

## 3. Run the browser application

The dataset is not required for inference on uploaded JPEG or PNG images.

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

Open <http://127.0.0.1:8765>, choose a route, upload an image, and review the returned candidates. Runtime uploads and findings are written under `runs/harness/` and are ignored by Git.

Available routes include trained YOLO detection, trained YOLO segmentation, trained ResNet classification, and Florence-to-SAM structured inference. If a required checkpoint is missing, the interface reports the unavailable route rather than inventing an inference result.

## 4. Restore and generate the datasets

Dataset redistribution restrictions prevent raw source archives from being stored here. Review the source terms before downloading CODEBRIM, S2DS, or any other restricted source.

To download all approved sources into their exact expected locations and build every intermediate artifact:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

The command performs, in order:

1. Source acquisition through `scripts/download_datasets.ps1` and `scripts/download_datasets.py`.
2. Native source auditing and inventory generation.
3. Deterministic feasibility-manifest selection and materialization.
4. Normalized detection, segmentation, classification, and Florence task views.
5. CUBIT registry construction, duplicate audit, and exact-dedup decisions.
6. CODEBRIM classification registry construction.
7. Frozen v1 manifest generation with the locked CUBIT test excluded from training and validation.
8. YOLO, ResNet, and Florence training-input generation.

If the raw sources already exist in the documented paths, skip acquisition:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -SkipDownload
```

To reacquire downloader-managed sources, use `-ForceDownload` together with `-AcceptLicenses`. This may replace an existing downloaded source, so retain any irreplaceable local files first.

Exact source URLs/revisions, filenames, raw storage locations, generated locations, manual equivalent commands, and verification checks are documented in [`docs/DATASET_REPRODUCTION_GUIDE.md`](docs/DATASET_REPRODUCTION_GUIDE.md).

## 5. Verify the project

After the datasets have been prepared, run the complete test and documentation checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
.\.venv\Scripts\python.exe scripts\verify_model_assets.py
```

Dataset-independent harness tests can be run before downloading data:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_web_harness.py -v
```

## 6. Training and evaluation entry points

- `scripts/run_v1_training_queue.py`: recorded v1 training sequence.
- `scripts/train_yolo_comparison.py`: YOLO detection or segmentation training.
- `scripts/train_resnet_comparison.py`: ResNet-50 multilabel training.
- `scripts/train_florence.py`: bounded Florence fine-tuning route.
- `scripts/train_sam_decoder.py`: SAM decoder training route.
- `scripts/evaluate_cubit_test.py`: locked CUBIT evaluation.
- `scripts/evaluate_resnet.py`: group-safe validation and CODEBRIM test evaluation.
- `scripts/evaluate_source_specific.py`: source-specific external evaluation.
- `scripts/generate_v1_comparison_report.py`: rebuilds the measured-results report from machine-readable evidence.

Do not evaluate the locked test until training choices and validation thresholds have been frozen.

## Recorded results and reports

- [`docs/FYP_COMPLETION_CHECKLIST.md`](docs/FYP_COMPLETION_CHECKLIST.md): accepted FYP completion checklist.
- [`docs/V1_COMPARISON_REPORT.md`](docs/V1_COMPARISON_REPORT.md): measured model results and provenance.
- [`docs/HUMAN_ERROR_REVIEW.md`](docs/HUMAN_ERROR_REVIEW.md): human-review completion record and its evidence boundary.
- [`docs/RECOMMENDED_IMPROVEMENTS.md`](docs/RECOMMENDED_IMPROVEMENTS.md): optional accuracy and generalization improvements.
- [`docs/DATASET_CONTRACT.md`](docs/DATASET_CONTRACT.md): immutable source, split, leakage, and licensing rules.
- [`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md): finding, review, measurement, and export contract.
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md): recorded software and hardware environment.

The headline in-domain ResNet score must not be presented as universal accuracy: minority-class support is very small, and external-source results are substantially lower. Keep each dataset's metric and semantics separate, exactly as reported in the comparison and improvement documents.
