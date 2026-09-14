# Project Handbook for Study and GitHub Reproduction

Project path used while writing this guide: `D:\ALI_Project`

This handbook explains the full project in simple words: which datasets are used, how they are structured, which scripts rebuild the same dataset files, and which commands a new person should run after pulling the repository from GitHub.

## 1. What This Project Does

This is a Building Defect Inspection FYP/research prototype. It detects or classifies visible building/civil-infrastructure defects from images.

The main user-facing application is a local browser app:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The app can run inference on uploaded JPEG/PNG images. Dataset files are not required just to test the browser app, but model checkpoints must be present.

## 2. Important Repo Folders

| Folder | Meaning |
|---|---|
| `bdi/` | Core project Python package: inference helpers, findings, Florence route, measurement, Grad-CAM. |
| `harness/` | Local web app and API server. |
| `scripts/` | Dataset download, audit, preparation, training, evaluation, and report-generation scripts. |
| `config/` | Model and dataset source configuration. |
| `schemas/` | JSON schema files for outputs/calibration. |
| `data/manifests/` | Frozen CSV/JSONL manifests that describe selected dataset samples. |
| `datasets/` | Raw downloaded datasets. Large raw bytes are not tracked in Git. |
| `data/v1/` | Generated normalized images, annotations, and model training inputs. Usually regenerated locally. |
| `artifacts/data-audit/` | Dataset audit reports, duplicate reports, manifest reports, and verification evidence. |
| `docs/` | Project reports, runbooks, dataset contract, taxonomy, and reproduction guides. |
| `runs/` | Runtime outputs, model training runs, and generated inference files. Large outputs are ignored by Git. |
| `weights/` | Local model weights and downloaded/fine-tuned model assets. Large weights are not stored in Git. |

## 3. Datasets Used

The project uses a manifest-based combined dataset. It does not blindly merge all image folders.

| Dataset | Role in project | Source/link or ID | Local path |
|---|---|---|---|
| CUBIT-InSeg | Main building-facade UAV instance segmentation dataset for crack/spalling | Hugging Face dataset: `CUBIT-InSeg/CUBIT-InSeg`; GitHub: `https://github.com/benyunzhao/CUBIT-InSeg`; official Google Drive folder: `https://drive.google.com/drive/folders/1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b` | `datasets/building-target/CUBIT-InSeg/` |
| CiF tiled | Large supplemental civil-infrastructure tiled source | Hugging Face dataset: `cif-benchmark/CiF-tiled` | `datasets/CiF-tiled/` |
| DACL10K | Bridge-domain semantic segmentation stress/transfer source | Google Drive file id: `15_crXMnhah3oW9q-5pHxvTtKo-71BXGV` | `datasets/DACL10K/dacl10k_v2_devphase.zip` |
| CODEBRIM originals | Secondary classification/box-detection source | Google Drive file id: `1x2wKBo7RDZMiF3TYa2mDk1MmBaFRuqkh` | `datasets/CODEBRIM/CODEBRIM_original_images.zip` |
| CODEBRIM classification | Classification archive | Google Drive file id: `1JmlCwDyLBFLQ1JN2_c8AJIG1gR0NlO1r` | `datasets/CODEBRIM/CODEBRIM_classification_dataset.zip` |
| S2DS | Structural surface damage segmentation source | Google Drive file id: `1l-HB5t3v1NSURxMIoaLgdiGaQRKz6Rsm` | `datasets/building-target/S2DS/s2ds.zip` |
| UAV75 | Small UAV crack/planking candidate set | `https://github.com/ozgeanli/UAV75.git` | `datasets/UAV-candidates/UAV75/` |

For Google Drive sources, the downloader uses:

```text
https://drive.google.com/uc?id=<FILE_ID>
```

CODEBRIM and S2DS require license/terms acceptance. Use `-AcceptLicenses` only after reviewing their terms.

## 4. Dataset Storage Structure

After restoration and preparation, the important structure is:

```text
D:\ALI_Project
  datasets\
    CiF-tiled\
    DACL10K\
      dacl10k_v2_devphase.zip
    CODEBRIM\
      CODEBRIM_original_images.zip
      CODEBRIM_classification_dataset.zip
    building-target\
      CUBIT-InSeg\
      S2DS\
        s2ds.zip
    UAV-candidates\
      UAV75\

  data\
    manifests\
      v1_master_manifest.csv
      v1_detection_manifest.csv
      v1_segmentation_manifest.csv
      v1_classification_manifest.csv
      v1_florence_product_manifest.csv
      v1_florence_targets.jsonl
    v1\
      images\
      annotations\
      training_inputs\
        yolo_detection\
        yolo_segmentation\
        classification_manifest_v1_0_0.csv
        florence_targets_v1_0_0.jsonl

  artifacts\
    data-audit\
```

Simple meaning:

- `datasets/` has the original downloaded raw sources.
- `data/manifests/` has frozen project records.
- `data/v1/images/` has normalized selected images.
- `data/v1/annotations/` has normalized annotation JSON files.
- `data/v1/training_inputs/` has YOLO, ResNet, and Florence-ready files.
- `artifacts/data-audit/` proves what was generated and checked.

## 5. How the Dataset Pipeline Works

The project creates the final dataset in this order:

1. Download raw sources into fixed paths.
2. Audit dataset archives, labels, and source structure.
3. Build a small feasibility set.
4. Materialize selected images and annotations.
5. Build task-specific views: detection, segmentation, classification, Florence.
6. Build CUBIT source registry.
7. Audit exact and near duplicates.
8. Build exact-dedup decisions.
9. Build CODEBRIM classification registry.
10. Freeze the v1 manifests.
11. Generate training inputs for YOLO, ResNet, and Florence.

The one-command script is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

If raw datasets are already downloaded in the correct folders:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -SkipDownload
```

The exact scripts run by `prepare_datasets.ps1` are:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1 -AcceptLicenses
.\.venv\Scripts\python.exe scripts\audit_datasets.py --root .
.\.venv\Scripts\python.exe scripts\build_feasibility_manifest.py --root .
.\.venv\Scripts\python.exe scripts\materialize_feasibility_set.py --root .
.\.venv\Scripts\python.exe scripts\build_feasibility_task_views.py --root .
.\.venv\Scripts\python.exe scripts\build_cubit_registry.py --root .
.\.venv\Scripts\python.exe scripts\audit_cubit_duplicates.py --root .
.\.venv\Scripts\python.exe scripts\build_cubit_exact_dedup_manifest.py --root .
.\.venv\Scripts\python.exe scripts\build_codebrim_registry.py --root .
.\.venv\Scripts\python.exe scripts\build_v1_manifest.py --root .
.\.venv\Scripts\python.exe scripts\build_v1_training_inputs.py --root .
```

Note: if you run these manually, keep the same order.

## 6. Final v1 Dataset Counts

The frozen v1 manifest has 5,520 rows.

| Split | Source counts |
|---|---:|
| Train | CUBIT-InSeg 4,694; CiF-tiled 120; DACL10K-v2-devphase 60; S2DS 80; UAV75 40 |
| Validation | CUBIT-InSeg 526 |

Class-containing image counts:

| Split | Class counts |
|---|---|
| Train | crack 2,679; spalling 2,310; rust_staining 82; efflorescence_leaching 22; exposed_rebar 8; honeycombing_rock_pocket 8 |
| Validation | crack 282; spalling 244 |

Training input counts:

| Task | Count |
|---|---:|
| YOLO detection train | 4,994 |
| YOLO detection val | 526 |
| YOLO segmentation train | 4,994 |
| YOLO segmentation val | 526 |
| ResNet classification records | 5,520 |
| Florence records | 5,520 |

The locked CUBIT test records are not included in train or validation.

## 7. Main Manifest Files

| File | Purpose |
|---|---|
| `data/manifests/v1_master_manifest.csv` | Main frozen dataset table. One row per selected sample. |
| `data/manifests/v1_detection_manifest.csv` | Detection view for YOLO boxes. |
| `data/manifests/v1_segmentation_manifest.csv` | Segmentation view for YOLO segmentation/SAM-related work. |
| `data/manifests/v1_classification_manifest.csv` | Classification view for ResNet. |
| `data/manifests/v1_florence_product_manifest.csv` | Florence/product-style view. |
| `data/manifests/v1_florence_targets.jsonl` | Florence image-to-text target records. |

Important columns:

| Column | Simple meaning |
|---|---|
| `sample_id` | Unique project sample ID. |
| `source_dataset` | Original dataset name. |
| `source_version` | Exact source version/configuration. |
| `source_path` | Source-relative path or archive member. |
| `source_image_id` | Original source image ID. |
| `parent_image_id` | Parent image before tile/crop. |
| `group_id` | Group used to avoid train/test leakage. |
| `split` | Train, validation, test, external-test, or unassigned. |
| `capture_mode` | UAV, handheld, unknown, etc. |
| `asset_domain` | Building, construction site, bridge, mixed, etc. |
| `annotation_type` | Box, polygon, semantic mask, instance mask, image label, etc. |
| `native_labels` | Original dataset labels. |
| `canonical_labels` | Project-standard labels. |
| `license_id` | Dataset license reference. |
| `quality_flags` | Duplicate/quality/test-lock notes. |

## 8. Project Labels

The shared taxonomy is defined in `docs/TAXONOMY.md`.

| ID | Label |
|---:|---|
| 0 | `background` |
| 1 | `crack` |
| 2 | `spalling` |
| 3 | `honeycombing_rock_pocket` |
| 4 | `exposed_rebar` |
| 5 | `rust_staining` |
| 6 | `efflorescence_leaching` |
| 7 | `no_visible_target_defect` |
| 8 | `unknown_review` |

Simple examples:

- CUBIT `crack` becomes `crack`.
- CUBIT `spalling` becomes `spalling`.
- DACL10K `ExposedRebars` becomes `exposed_rebar`.
- DACL10K `Efflorescence` becomes `efflorescence_leaching`.
- CiF `Rust` becomes `rust_staining`.
- CODEBRIM `corrosion` is treated carefully as weak visual rust evidence.

## 9. Fresh GitHub Pull: Setup Commands

Use a short path without an apostrophe, for example `D:\ALI_Project` or `D:\ProjectAlpha`.

```powershell
git clone <YOUR_GITHUB_REPO_URL> D:\ALI_Project
cd D:\ALI_Project
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

Then restore datasets:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

Verify:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
Get-Content artifacts\data-audit\v1_manifest_report.json
Get-Content artifacts\data-audit\v1_training_inputs_report.json
```

Run the app:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

Open:

```text
http://127.0.0.1:8765
```

## 10. Model Assets Needed for Runtime

The Git repo does not store large model weights. The app expects model assets/checkpoints at paths like:

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

Verify placed model assets:

```powershell
.\.venv\Scripts\python.exe scripts\verify_model_assets.py
```

## 11. Training and Evaluation Scripts

Common training entry points:

| Script | Purpose |
|---|---|
| `scripts/run_v1_training_queue.py` | Runs the recorded v1 training sequence. |
| `scripts/train_yolo_comparison.py` | Trains YOLO detection or segmentation comparison models. |
| `scripts/train_resnet_comparison.py` | Trains ResNet-50 multilabel comparison model. |
| `scripts/train_florence.py` | Fine-tunes/evaluates bounded Florence route. |
| `scripts/train_sam_decoder.py` | SAM decoder training route. |

Common evaluation/report scripts:

| Script | Purpose |
|---|---|
| `scripts/evaluate_cubit_test.py` | Evaluates on locked CUBIT test after training decisions are frozen. |
| `scripts/evaluate_resnet.py` | ResNet validation and CODEBRIM test evaluation. |
| `scripts/evaluate_source_specific.py` | Source-specific external evaluation. |
| `scripts/evaluate_florence_cubit_test.py` | Florence evaluation on CUBIT test. |
| `scripts/generate_v1_comparison_report.py` | Rebuilds measured-results report from evidence. |

Important rule: do not use the locked test set for tuning. Train/validate first, freeze choices, then evaluate test.

## 12. What to Say in Viva

Short explanation:

```text
My project uses a manifest-based dataset pipeline for building defect inspection.
The main target dataset is CUBIT-InSeg, because it contains UAV building-facade
crack and spalling instance-segmentation data. Supporting datasets like CiF,
DACL10K, S2DS, CODEBRIM, and UAV75 are used for extra defect types, transfer
testing, or stress evaluation.

The project does not store raw datasets inside Git because they are large and
some have license restrictions. Instead, Git stores scripts, metadata, audit
reports, and frozen manifests. A new user can run scripts/prepare_datasets.ps1
to download, audit, normalize, deduplicate, freeze manifests, and generate YOLO,
ResNet, and Florence training inputs.

This approach keeps source licenses attached, prevents train/test leakage,
preserves native labels, maps them to one canonical taxonomy, and keeps the
locked test set separate from training and validation.
```

Roman Urdu explanation:

```text
Is project mein datasets ko simple folder merge nahi kiya gaya. Har image ka
record manifest CSV/JSONL mein rakha gaya hai, jahan source dataset, split,
label, annotation type, license, aur normalized output path mention hota hai.
Raw datasets datasets/ folder mein restore hotay hain, lekin GitHub mein large
raw files nahi rakhi gayi. scripts/prepare_datasets.ps1 run karne se same
dataset structure dobara ban jata hai. CUBIT-InSeg main dataset hai, aur baqi
datasets support/stress testing ke liye use hotay hain. Locked test data training
ya validation mein use nahi hota, taa ke result fair rahe.
```

## 13. Evidence Files to Open

| Question | File to open |
|---|---|
| How to reproduce datasets? | `docs/DATASET_REPRODUCTION_GUIDE.md` |
| What are the dataset rules? | `docs/DATASET_CONTRACT.md` |
| What labels are used? | `docs/TAXONOMY.md` |
| What final counts were generated? | `artifacts/data-audit/v1_manifest_report.json` |
| What training inputs were generated? | `artifacts/data-audit/v1_training_inputs_report.json` |
| Where are dataset source IDs defined? | `scripts/download_datasets.py` |
| Where is CUBIT source metadata? | `config/cubit_source.yaml` |
| How to run the app? | `README.md` and `START_HERE.md` |

