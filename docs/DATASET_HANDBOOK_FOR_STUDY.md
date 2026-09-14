# Dataset Handbook for Study

Project path: `D:\ALI_Project`

This handbook explains which datasets are used in this project, where their links are recorded, how they are stored locally, and how they are converted into training-ready files. It is written as a simple study guide so you can explain the dataset part of the FYP easily.

## 1. Short Summary

This project is a Building Defect Inspection FYP. It does not keep all raw dataset files inside Git because the datasets are large and some have license restrictions. Instead, the repository stores:

- dataset download scripts;
- dataset metadata and licenses;
- frozen CSV/JSONL manifests;
- normalized annotations;
- generated training-input structure.

The raw datasets are restored by running the preparation script. The prepared v1 dataset contains 5,520 manifest rows:

| Split | Source counts |
|---|---:|
| Train | CUBIT-InSeg 4,694, CiF-tiled 120, DACL10K 60, S2DS 80, UAV75 40 |
| Validation | CUBIT-InSeg 526 |

The locked CUBIT test set is not included in training or validation.

## 2. Dataset Links Used by This Project

These links and IDs come from `scripts/download_datasets.py` and `config/cubit_source.yaml`.

| Dataset | Purpose in project | Source/link | Local path |
|---|---|---|---|
| CUBIT-InSeg | Main UAV building-facade crack/spalling dataset | Hugging Face dataset id: `CUBIT-InSeg/CUBIT-InSeg`; GitHub: `https://github.com/benyunzhao/CUBIT-InSeg`; official Google Drive folder: `https://drive.google.com/drive/folders/1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b` | `datasets/building-target/CUBIT-InSeg/` |
| CiF tiled | Supplemental civil-infrastructure detection/segmentation source | Project downloader id: `cif-benchmark/CiF-tiled`; dataset card also cites `https://huggingface.co/datasets/ibm-research/cif-dataset` | `datasets/CiF-tiled/` |
| DACL10K | Bridge-domain semantic segmentation stress/transfer source | Google Drive file id: `15_crXMnhah3oW9q-5pHxvTtKo-71BXGV` | `datasets/DACL10K/dacl10k_v2_devphase.zip` |
| CODEBRIM originals | Secondary classification/box-detection source | Google Drive file id: `1x2wKBo7RDZMiF3TYa2mDk1MmBaFRuqkh` | `datasets/CODEBRIM/CODEBRIM_original_images.zip` |
| CODEBRIM classification | Classification archive | Google Drive file id: `1JmlCwDyLBFLQ1JN2_c8AJIG1gR0NlO1r` | `datasets/CODEBRIM/CODEBRIM_classification_dataset.zip` |
| S2DS | Structural surface damage segmentation source | Google Drive file id: `1l-HB5t3v1NSURxMIoaLgdiGaQRKz6Rsm` | `datasets/building-target/S2DS/s2ds.zip` |
| UAV75 | Small UAV crack/planking candidate set | `https://github.com/ozgeanli/UAV75.git` | `datasets/UAV-candidates/UAV75/` |

For Google Drive files, the downloader builds URLs in this form:

```text
https://drive.google.com/uc?id=<FILE_ID>
```

CODEBRIM and S2DS are restricted in the project downloader, so use `-AcceptLicenses` only after reviewing and accepting their terms.

## 3. Main Local Folder Structure

The important dataset folders are:

```text
D:\ALI_Project
  datasets\
    CiF-tiled\
    DACL10K\
    CODEBRIM\
    building-target\
      CUBIT-InSeg\
      S2DS\
    UAV-candidates\
      UAV75\

  data\
    manifests\
    feasibility_v0\
    v1\
      images\
      annotations\
      training_inputs\

  artifacts\
    data-audit\
```

Meaning:

- `datasets/` contains raw or downloaded source data.
- `data/manifests/` contains frozen CSV/JSONL records that describe selected samples.
- `data/v1/images/` contains normalized images used by the project.
- `data/v1/annotations/` contains normalized annotation JSON files.
- `data/v1/training_inputs/` contains model-specific training folders/files.
- `artifacts/data-audit/` contains audit reports, duplicate reports, and verification summaries.

## 4. How Raw Datasets Are Converted

The project does not simply merge folders. It uses a manifest-based pipeline:

1. Download sources into fixed paths.
2. Audit source files and labels.
3. Build a curated feasibility set.
4. Normalize images and annotations.
5. Build source registries for CUBIT and CODEBRIM.
6. Detect exact and near duplicates.
7. Exclude unresolved/reviewed duplicate training records.
8. Freeze v1 manifests.
9. Generate training inputs for YOLO, ResNet, and Florence.

The one-command pipeline is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

If datasets are already downloaded in the correct folders:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -SkipDownload
```

## 5. Important Manifest Files

These are the most important dataset files for explanation:

| File | Meaning |
|---|---|
| `data/manifests/v1_master_manifest.csv` | Main frozen v1 dataset record. One row per selected sample. |
| `data/manifests/v1_detection_manifest.csv` | Detection task view for YOLO boxes. |
| `data/manifests/v1_segmentation_manifest.csv` | Segmentation task view for YOLO segmentation/SAM-related work. |
| `data/manifests/v1_classification_manifest.csv` | Classification task view for ResNet. |
| `data/manifests/v1_florence_product_manifest.csv` | Product-style view for Florence route. |
| `data/manifests/v1_florence_targets.jsonl` | Florence image-to-text target records. |

Common columns in the v1 manifests:

| Column | Simple meaning |
|---|---|
| `sample_id` | Unique project sample ID. |
| `source_dataset` | Original dataset name, for example CUBIT-InSeg. |
| `source_version` | Exact version or release used. |
| `source_image_id` | Original source image ID. |
| `parent_image_id` | Original parent image before crop/tile. |
| `group_id` | Group used to avoid leakage across splits. |
| `split` | `train`, `val`, test, or external-test related split. |
| `capture_mode` | How image was captured, for example UAV RGB. |
| `asset_domain` | Building facade, bridge, construction site, etc. |
| `annotation_type` | Box, polygon, semantic mask, instance polygon, image label, etc. |
| `native_labels` | Original dataset label. |
| `canonical_labels` | Project standard label. |
| `license_id` | License reference. |
| `quality_flags` | Duplicate/test-lock/quality notes. |
| `output_image` | Normalized image path. |
| `normalized_annotation` | Normalized JSON annotation path. |

## 6. Canonical Labels

The project maps different dataset labels into one shared taxonomy. The main target labels are:

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

Example mapping:

- CUBIT `crack` becomes project label `crack`.
- CUBIT `spalling` becomes project label `spalling`.
- DACL10K `ExposedRebars` becomes `exposed_rebar`.
- DACL10K `Efflorescence` becomes `efflorescence_leaching`.
- CiF `Rust` becomes `rust_staining`.
- CODEBRIM `corrosion` is treated carefully as weak visual rust evidence.

The full taxonomy is in `docs/TAXONOMY.md`.

## 7. Model Training Inputs

Generated training inputs are stored under:

```text
data/v1/training_inputs/
```

Important outputs:

| Training input | Path |
|---|---|
| YOLO detection | `data/v1/training_inputs/yolo_detection/` |
| YOLO segmentation | `data/v1/training_inputs/yolo_segmentation/` |
| ResNet classification | `data/v1/training_inputs/classification_manifest_v1_0_0.csv` |
| Florence | `data/v1/training_inputs/florence_targets_v1_0_0.jsonl` |

Recorded v1 training counts:

| Task | Train | Validation |
|---|---:|---:|
| Detection | 4,994 | 526 |
| Segmentation | 4,994 | 526 |
| Classification | 5,520 total records | Uses manifest split |
| Florence | 5,520 total records | Uses manifest split |

## 8. Key Rules Used in This Dataset

Use these points in your report or viva:

- Raw datasets are immutable. The project never edits the downloaded source archives.
- The project uses manifests instead of blindly mixing dataset folders.
- Native labels are preserved, then mapped to canonical labels.
- Test data is locked and is not used for training or validation.
- Exact and near duplicates are checked to reduce leakage.
- Masks can produce derived bounding boxes, but boxes are not converted into fake ground-truth masks.
- Unknown annotations are not treated as negative examples automatically.
- Licenses remain attached to every source dataset.

## 9. Files to Open When Explaining the Dataset

Use these files as evidence:

| Purpose | File |
|---|---|
| General project dataset explanation | `README.md` |
| Restore/download guide | `docs/DATASET_REPRODUCTION_GUIDE.md` |
| Dataset rules/contract | `docs/DATASET_CONTRACT.md` |
| Label mapping | `docs/TAXONOMY.md` |
| v1 manifest summary | `artifacts/data-audit/v1_manifest_report.json` |
| training input summary | `artifacts/data-audit/v1_training_inputs_report.json` |
| source downloader registry | `scripts/download_datasets.py` |
| CUBIT source metadata | `config/cubit_source.yaml` |

## 10. Simple Explanation for Presentation

You can explain it like this:

> My project uses multiple public and academic building/civil-infrastructure defect datasets. The main target dataset is CUBIT-InSeg, because it contains UAV building-facade crack and spalling instance segmentation data. Other datasets such as CiF, DACL10K, S2DS, CODEBRIM, and UAV75 are used as supporting sources for additional defect types and domain testing. The project does not directly merge all images into one folder. Instead, it creates frozen manifest files that record each sample's source, split, labels, annotation type, license, and normalized output path. This reduces leakage, keeps test data locked, and makes the dataset reproducible.

## 11. Prompt You Can Give to ChatGPT

If you want ChatGPT to explain this again in easier words, paste this:

```text
I am working on a Building Defect Inspection FYP located at D:\ALI_Project.
Explain my dataset pipeline in simple student-friendly language.
Use these project facts:
- Raw datasets are in datasets/
- Frozen manifests are in data/manifests/
- Normalized v1 images and annotations are in data/v1/
- Training inputs are in data/v1/training_inputs/
- Main datasets are CUBIT-InSeg, CiF-tiled, DACL10K, CODEBRIM, S2DS, and UAV75
- Main v1 manifest file is data/manifests/v1_master_manifest.csv
- Task views are detection, segmentation, classification, and Florence JSONL
- Canonical labels include crack, spalling, exposed_rebar, rust_staining, efflorescence_leaching, honeycombing_rock_pocket, background, no_visible_target_defect, and unknown_review
- Explain why manifests, duplicate checks, locked test split, and license tracking are important
Write it in Roman Urdu plus simple English, suitable for a viva.
```

## 12. Quick Commands

Download/prepare everything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

Prepare only if raw data already exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -SkipDownload
```

Run tests/checks after preparation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
Get-Content artifacts\data-audit\v1_manifest_report.json
Get-Content artifacts\data-audit\v1_training_inputs_report.json
```

