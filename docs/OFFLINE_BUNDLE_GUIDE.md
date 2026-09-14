# Offline Project Handoff

The handoff folder contains the complete runnable FYP demo without redistributing raw datasets. It includes source code, schemas, configurations, manifests, tests, documentation, model assets, trained harness checkpoints, evaluation evidence, a Python installer, offline dependency wheels, and `artifacts/fyp.docx`.

## First run on Windows

1. Copy the handoff folder to a short path without an apostrophe, such as `D:\ProjectAlpha`.
2. Install Python 3.11.9 from `vendor\installers\python-3.11.9-amd64.exe` if Python 3.11 is unavailable.
3. Open PowerShell in the extracted folder.
4. Create and verify the environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_offline_env.ps1
```

5. Start the application:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

6. Open `http://127.0.0.1:8765`.

No network connection is required for setup or inference. The NVIDIA CUDA wheels also support CPU execution when a compatible NVIDIA GPU is unavailable, although large-model routes can be slow and memory intensive on CPU.

## Archive verification

Run this before setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_offline_bundle.ps1
```

The script checks every listed file against `artifacts/offline_bundle_manifest.sha256`.

## Dataset restoration and generation on a connected machine

The project uses 60.9 GiB of raw sources. CODEBRIM and S2DS terms prohibit unrestricted redistribution, so those source archives must be acquired directly by each authorized user.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

The command restores the same project paths, validates the source revision, shard counts, byte counts, or checksums, and regenerates the manifests and training inputs. Full details are in `docs/DATASET_REPRODUCTION_GUIDE.md`.

The raw dataset collection is not needed for normal browser inference on uploaded images.

## Included runtime evidence

- `weights/`: all locally pinned base-model assets.
- `runs/detect/.../yolo11n_detect_v1_queue/`: trained detector checkpoint and logs.
- `runs/segment/.../yolo11n_seg_v1_queue/`: trained segmenter checkpoint and logs.
- `runs/comparison/resnet50_v1_queue/`: trained classifier checkpoint and history.
- `runs/evaluation/`: completed evaluation outputs.
- `runs/comparison/yolo_sam/v1-sample-10/`: YOLO-to-SAM output evidence.
- `runs/florence/v1-validation-sample-10/`: Florence output evidence.
- `runs/sam/decoder_v1_fp32_smoke/`: SAM decoder training evidence.
- `docs/RECOMMENDED_IMPROVEMENTS.md`: optional improvement recommendations that do not block FYP completion.
- `docs/DATASET_REPRODUCTION_GUIDE.md`: exact dataset download, configuration, generation, storage, and verification instructions.

The 44.7 GiB materialized training cache and 45+ GiB duplicate/experimental run cache are reproducible local products and are not duplicated in the transfer archive. Their manifests, selected checkpoints, reports, and verification evidence are included.
