# Building Defect Inspection Beta

This workspace is being prepared as a functional academic beta for visible-condition inspection of completed and under-construction buildings. Cracks are the primary target. The current phase establishes scope, data governance, model comparisons, and hardware feasibility before full implementation.

## Start here

- [`docs/BETA_SPECIFICATION.md`](docs/BETA_SPECIFICATION.md): approved product boundary and definition of done.
- [`docs/ROADMAP_V2.md`](docs/ROADMAP_V2.md): corrected execution roadmap for the functional building-inspection beta.
- [`docs/TAXONOMY.md`](docs/TAXONOMY.md): canonical defect labels and source mappings.
- [`docs/DATASET_CONTRACT.md`](docs/DATASET_CONTRACT.md): rules for the curated mega dataset.
- [`docs/EXPERIMENT_MATRIX.md`](docs/EXPERIMENT_MATRIX.md): fair Florence/YOLO/SAM/ResNet comparisons.
- [`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md): structured finding, review, measurement, and export boundary.
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md): observed local hardware and software state.
- [`docs/FIRST_WEEK_PLAN.md`](docs/FIRST_WEEK_PLAN.md): first execution gate for the three-person team.
- [`PROJECT_READINESS.md`](PROJECT_READINESS.md): detailed correction and readiness analysis of the supplied roadmap.

The original user-supplied roadmap is preserved unchanged at [`source/The RoadMap.md`](source/The%20RoadMap.md).

## Current state

- Dataset acquisition and non-destructive archive inspection are underway.
- A deterministic 300-sample, training-only feasibility set has been materialized and verified: 120 CiF, 80 S2DS, 40 UAV75, and 60 DACL10K samples.
- The reproducible selection manifest is [`data/manifests/feasibility_v0.csv`](data/manifests/feasibility_v0.csv). The content-hashed form is [`data/manifests/feasibility_v0_materialized.csv`](data/manifests/feasibility_v0_materialized.csv), and verification results are in [`artifacts/data-audit/feasibility_materialization_report.json`](artifacts/data-audit/feasibility_materialization_report.json).
- CODEBRIM's 7,729 official classification crops have a metadata/source registry at [`data/manifests/codebrim_classification_source_v1.csv`](data/manifests/codebrim_classification_source_v1.csv); no parent image crosses its official splits.
- No training run has been started.
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
.\.venv\Scripts\python.exe scripts\audit_datasets.py
.\.venv\Scripts\python.exe scripts\build_codebrim_registry.py
.\.venv\Scripts\python.exe scripts\build_feasibility_manifest.py
.\.venv\Scripts\python.exe scripts\materialize_feasibility_set.py
```

The last command is idempotent: it reuses identical derived files and refuses to replace differing outputs unless `--overwrite` is explicitly supplied.

Run the lightweight preparation checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Rebuild the checkpoint inventory and repeat the no-inference/no-training load test with:

```powershell
.\.venv\Scripts\python.exe scripts\inventory_model_assets.py
.\.venv\Scripts\python.exe scripts\verify_model_assets.py
```
"# Project-Alpha_Legal_Intra" 
