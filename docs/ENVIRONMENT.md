# Local Environment Baseline

**Observed:** 2026-09-08  
**Machine:** HP ZBook-class Windows workstation

## Hardware

| Item | Observed value |
|---|---|
| GPU | NVIDIA Quadro T2000 |
| VRAM | 4,096 MiB |
| NVIDIA driver | 595.95 |
| System RAM | Approximately 32 GB |
| D: free space after feasibility-set materialization | Approximately 138 GiB |
| D: free space after model/runtime acquisition (2026-09-09) | Approximately 122 GiB |

If a separate T3000 system is intended, it must receive its own environment record and benchmark results.

## Python discovery

- Python 3.13, 3.12, and 3.11 are installed.
- `uv` is available for isolated environments.
- The current default/global Python has PyTorch `2.12.0+cpu`.
- `torch.cuda.is_available()` is false in the default/global environment; do not use it for this project.
- The NVIDIA driver sees the GPU correctly, so the present blocker is the CPU-only PyTorch build rather than missing hardware.

## Environment decision

Create a project-local virtual environment using Python 3.11. Install an official CUDA-enabled PyTorch/torchvision combination there, verify a small CUDA tensor operation, and record exact versions before adding model-specific dependencies. Do not replace the user's global Python packages.

The local `.venv` uses Python 3.11.9 with PyTorch `2.6.0+cu124` and torchvision `0.21.0+cu124`. Both wheel hashes match the official CUDA 12.4 index. `torch.cuda.is_available()` is true, and a 1024 x 1024 float16 CUDA matrix multiplication passed on the Quadro T2000 (compute capability 7.5). The machine-readable result is `artifacts/environment/cuda_verification.json`.

The exact wheel sizes and official-index SHA-256 hashes are recorded in `config/cuda_environment.yaml`. After both downloads complete, `scripts/setup_cuda_env.ps1` verifies the files, installs them only into `.venv`, runs a float16 CUDA matrix operation, and writes `artifacts/environment/cuda_verification.json`. A false `torch.cuda.is_available()` causes a hard failure.

Dataset-audit dependencies are recorded separately in `requirements-audit.txt`. Keeping audit/data tooling separate prevents it from silently determining the eventual training stack.

The model-feasibility runtime currently pins Ultralytics `8.4.143`, Transformers `5.16.1`, and Accelerate `1.14.0`. Runtime checkpoints and their local paths are recorded in `config/model_candidates.yaml`; file-level hashes are generated in `artifacts/model-assets/model_asset_inventory.json` and `.csv`.

Florence uses the `florence-community/Florence-2-base-ft` and `Florence-2-large-ft` native-Transformers conversions. The downloaded Microsoft custom-code snapshots are preserved only as origin references because their processor/config interface is incompatible with the current native runtime. This is a checkpoint-format distinction, not a change from Florence-2 Base/Large FT as the compared model family.

## Windows path caveat

Ultralytics `8.4.143` sanitizes the apostrophe in this workspace's absolute path when a checkpoint path is passed directly, causing it to miss the local file and download a duplicate under `D:\ALIs Project`. Project wrappers must change into the checkpoint directory and pass the bare filename, as `scripts/verify_model_assets.py` now does, or relocate the workspace to a path without an apostrophe. The fixed verification completed without recreating the sanitized directory.

## Feasibility implications

The environment completed the accepted FYP workload: dataset preparation, YOLO and ResNet training, bounded Florence and SAM execution, locked evaluation, source-specific evaluation, and the local browser demonstration. Optional accuracy improvements are documented separately in [`RECOMMENDED_IMPROVEMENTS.md`](RECOMMENDED_IMPROVEMENTS.md).

- Use `.venv\Scripts\python.exe`; the global Python remains CPU-only.
- CUDA installation and basic execution are verified. One-image model preflight memory/latency is recorded; representative distributions remain unmeasured.
- First GPU tests use batch size 1 and small representative inputs.
- The project must log out-of-memory failures rather than silently fall back to CPU.

## Verified preparation checkpoint

- `scripts/audit_datasets.py` completes against the locally acquired sources.
- `scripts/build_feasibility_manifest.py` deterministically selects 300 training-only samples.
- `scripts/materialize_feasibility_set.py` materializes and opens all 300 images and annotations successfully.
- CUDA execution is verified; controlled model feasibility runs may begin after the exact dependencies and checkpoint hashes are recorded.

## Model CUDA preflight

All six candidates passed a real FP16 CUDA inference preflight on hash-verified training sample `cif_0001`. The heaviest was Florence-2 Large FT at approximately 1.90 GB peak allocated VRAM and 4.49 seconds warm for a short `<OD>` generation. SAM 2.1 Tiny produced a full 1024 x 1024 box-prompted mask at approximately 286 MB peak allocated VRAM and 0.74 seconds warm. The consolidated result is `artifacts/model-feasibility/summary.json`.

This resolves basic local inference fit only. It does not establish defect quality, sustained throughput, thermal behavior, 640/tiled performance, training fit, or final architecture choice. Those require the documented representative-image study and later controlled experiments.
