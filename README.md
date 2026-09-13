# Building Defect Inspection Beta

This workspace is being prepared as a master's FYP prototype for visible-condition inspection. Cracks are the primary target. The current execution sequence is model comparison, real-result capture, and then a local web demonstration.

## Start here

- [`docs/BETA_SPECIFICATION.md`](docs/BETA_SPECIFICATION.md): approved FYP prototype boundary and definition of done.
- [`docs/ROADMAP_V2.md`](docs/ROADMAP_V2.md): corrected execution roadmap for the functional building-inspection beta.
- [`docs/TAXONOMY.md`](docs/TAXONOMY.md): canonical defect labels and source mappings.
- [`docs/DATASET_CONTRACT.md`](docs/DATASET_CONTRACT.md): rules for the curated mega dataset.
- [`docs/EXPERIMENT_MATRIX.md`](docs/EXPERIMENT_MATRIX.md): approved Florence full pipeline and ResNet/YOLO/SAM comparison pipeline.
- [`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md): structured finding, review, measurement, and export boundary.
- [`docs/FLORENCE_PIPELINE.md`](docs/FLORENCE_PIPELINE.md): executable Base/Large feasibility pipeline and calibration format.
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md): observed local hardware and software state.
- [`docs/FIRST_WEEK_PLAN.md`](docs/FIRST_WEEK_PLAN.md): first execution gate for the three-person team.
- [`docs/COMPLETION_PLAN.md`](docs/COMPLETION_PLAN.md): authoritative full-dataset completion plan, ownership, file locations, verification gates, and six-hour feasibility boundary.
- [`docs/CONTRIBUTOR_GPU_SETUP.md`](docs/CONTRIBUTOR_GPU_SETUP.md): AMD/NVIDIA/CPU contributor setup, harness startup, and GPU work boundaries.
- [`docs/SAM_DECODER_TRAINING.md`](docs/SAM_DECODER_TRAINING.md): frozen-decoder SAM training design and larger-GPU run instructions.
- [`PROJECT_READINESS.md`](PROJECT_READINESS.md): detailed correction and readiness analysis of the supplied roadmap.

The original user-supplied roadmap is preserved unchanged at [`source/The RoadMap.md`](source/The%20RoadMap.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor guide, priorities, safe workstreams, and repo guardrails.

This repository is intentionally organized so GitHub contributors can keep making progress while owner/advisor decisions are pending. The guide separates blocked decisions from parallelizable tasks and lists the validation commands needed before merging any update.

## Current state

The repository is at the completed feasibility tier, not full-dataset completion. The remaining work is tracked in [`docs/COMPLETION_PLAN.md`](docs/COMPLETION_PLAN.md): freeze the v1 task views, complete annotation QA, train on the frozen full views, implement YOLO-to-SAM inference, evaluate locked tests, and finish the review/export workflow.

- Core public dataset acquisition and non-destructive archive inspection are complete; optional sources and real target-site data remain open.
- All six CUBIT-InSeg archives are downloaded and verified as 5,596 train, 699 validation, and 701 locked test image/label pairs. Hashes, CRC results, and pairing evidence are in [`artifacts/data-audit/cubit_archive_inventory.json`](artifacts/data-audit/cubit_archive_inventory.json).
- CUBIT's 6,295 train/validation pairs have a metadata-only source registry at [`data/manifests/cubit_train_val_source_v1.csv`](data/manifests/cubit_train_val_source_v1.csv); the test split is excluded and unavailable to training selection.
- Full CUBIT labeling is now materialized as 6,295 train/validation registry rows plus 6,996 exact-dedup decisions, retaining 5,035 train, 678 validation, and 694 test records after duplicate priority handling.
- CUBIT contains 589 byte-identical `SP0`/`SP1` image pairs, including 222 cross-split pairs. The auditable `test > val > train` decisions are in [`data/manifests/cubit_exact_dedup_decisions_v1.csv`](data/manifests/cubit_exact_dedup_decisions_v1.csv), yielding 5,035 train, 678 validation, and 694 test records. Perceptual candidates remain review-only.
- The local research-paper set has been title-checked. The correct 23-page CUBIT paper is installed; an unrelated file from the initial acquisition is retained with a `MISIDENTIFIED_` filename so it cannot silently support project claims.
- A deterministic 300-sample, training-only feasibility set has been materialized and verified: 120 CiF, 80 S2DS, 40 UAV75, and 60 DACL10K samples.
- The reproducible selection manifest is [`data/manifests/feasibility_v0.csv`](data/manifests/feasibility_v0.csv). The content-hashed form is [`data/manifests/feasibility_v0_materialized.csv`](data/manifests/feasibility_v0_materialized.csv), and verification results are in [`artifacts/data-audit/feasibility_materialization_report.json`](artifacts/data-audit/feasibility_materialization_report.json).
- All 300 feasibility labels are normalized under taxonomy `0.1.0-beta`; versioned master, detection, segmentation, classification, and Florence product views are in `data/manifests/feasibility_*_v0_1.csv`. Four malformed native polygon fragments were rejected without discarding their valid native boxes; the audit is [`artifacts/data-audit/feasibility_task_views_report.json`](artifacts/data-audit/feasibility_task_views_report.json).
- CODEBRIM's 7,729 official classification crops are fully indexed and labeled at [`data/manifests/codebrim_classification_source_v1.csv`](data/manifests/codebrim_classification_source_v1.csv); no parent image crosses its official splits.
- All three comparison training jobs on the 300-sample feasibility set are complete. YOLO11n detection: 5 epochs, mAP50 0.016, checkpoint at `runs/comparison/yolo/yolo11n_detect_fyp/weights/best.pt`. YOLO11n-seg segmentation: 5 epochs, seg_loss 3.88→3.25, checkpoint at `runs/comparison/yolo/yolo11n_seg_fyp/weights/best.pt` (6 MB). ResNet-50 classification: 5 epochs, final val loss 0.313, checkpoint at `runs/comparison/resnet50_fyp/resnet50_comparison.pt` (90 MB). Ultralytics' Windows path-sanitizer bug (apostrophe in `ALI's Project`) now caught and handled in `scripts/train_yolo_comparison.py`; all future runs exit cleanly. SAM 2.1 remains pretrained and uses YOLO boxes as prompts for the first comparison.
- The isolated Python 3.11 environment now has hash-verified PyTorch `2.6.0+cu124` and torchvision `0.21.0+cu124`; a real float16 CUDA operation passed on the Quadro T2000. The global Python remains CPU-only, so use `.venv\Scripts\python.exe` for model work.
- All six runtime candidates are downloaded and pass offline deserialization: YOLO11n, YOLO11n-seg, ResNet-50, SAM 2.1 Hiera Tiny, and the native-Transformers Florence-2 Base/Large FT conversions. See [`artifacts/model-assets/model_asset_inventory.json`](artifacts/model-assets/model_asset_inventory.json) and [`artifacts/model-assets/model_load_verification.json`](artifacts/model-assets/model_load_verification.json).
- All six also pass a real FP16 CUDA one-image preflight on the Quadro T2000, with no CPU fallback. The consolidated hardware-only result is [`artifacts/model-feasibility/summary.json`](artifacts/model-feasibility/summary.json); it is explicitly not an accuracy comparison.
- The Florence runtime uses `florence-community/Florence-2-*-ft`, the official Transformers-converted Microsoft checkpoints. The original Microsoft custom-code snapshots are retained only as origin references after their processor interface failed against the current native runtime.
- Both CODEBRIM archives are checksum-verified and pass full 7-Zip tests. Use `vendor/7zip-portable/x64/7za.exe` for their image payloads; standard Python/Windows ZIP readers are incompatible with the archives' oversized legacy headers.
- No result is to be treated as a structural-safety determination.
- Physical measurements remain pixel-only unless the source image has a valid calibration or scale method.

## Reproduce the completed data-preparation checkpoint

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-audit.txt
.\.venv\Scripts\python.exe scripts\inventory_cubit_archives.py
.\.venv\Scripts\python.exe scripts\audit_datasets.py
.\.venv\Scripts\python.exe scripts\build_cubit_registry.py
.\.venv\Scripts\python.exe scripts\audit_cubit_duplicates.py
.\.venv\Scripts\python.exe scripts\build_cubit_exact_dedup_manifest.py
.\.venv\Scripts\python.exe scripts\build_codebrim_registry.py
.\.venv\Scripts\python.exe scripts\build_feasibility_manifest.py
.\.venv\Scripts\python.exe scripts\materialize_feasibility_set.py
.\.venv\Scripts\python.exe scripts\build_feasibility_task_views.py
```

The last command is idempotent: it reuses identical derived files and refuses to replace differing outputs unless `--overwrite` is explicitly supplied.

Run the lightweight preparation checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run one end-to-end Florence Base/Large pipeline check with:

```powershell
.\.venv\Scripts\python.exe scripts\run_florence_pipeline.py --limit 1 --escalation all
```

Rebuild the checkpoint inventory and repeat the no-inference/no-training load test with:

```powershell
.\.venv\Scripts\python.exe scripts\inventory_model_assets.py
.\.venv\Scripts\python.exe scripts\verify_model_assets.py
```

Reproduce the comparison training runs (5-epoch feasibility tier, not final benchmarks):

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_comparison.py --task detect --epochs 5
.\.venv\Scripts\python.exe scripts\train_resnet_comparison.py --epochs 5
```

Note: use `--epochs 20` / `--epochs 10` for the full planned runs once a larger manifest is frozen. The YOLO post-training validation step on Windows paths containing apostrophes will error in Ultralytics' sanitizer; the checkpoint and `results.csv` are unaffected. Use `--no-val` or rename the working directory if a clean final-val pass is required.
"# Project-Alpha_Legal_Intra"
