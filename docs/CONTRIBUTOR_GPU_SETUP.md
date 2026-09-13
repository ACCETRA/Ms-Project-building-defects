# Contributor GPU Setup

This guide separates work that runs on any machine from model work that currently requires NVIDIA CUDA.

## First: clone and create the environment

From the repository root on Windows PowerShell:

```powershell
git clone https://github.com/ACCETRA/Project-Alpha_Legal_Intra.git
Set-Location Project-Alpha_Legal_Intra
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Always use the repository environment:

```powershell
.\.venv\Scripts\python.exe
```

Do not use the global Python interpreter for this project.

## Work that works on AMD, NVIDIA, or CPU

These tasks do not require CUDA:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
.\.venv\Scripts\python.exe scripts\prepare_source_review.py
.\.venv\Scripts\python.exe scripts\generate_v1_comparison_report.py
```

Dataset and documentation work can also be performed on any machine, provided the required source archives are present:

- review manifests and licenses;
- prepare S2DS and target-site review inputs;
- inspect `runs/evaluation/error_review_queue.json`;
- review false-positive and false-negative overlays;
- update reports and documentation;
- run the local browser harness.

## Run the web harness

The harness is inference-backed when the local checkpoints are present. Start it with:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

Open:

```text
http://127.0.0.1:8765
```

The harness supports project metadata, batch upload, YOLO detection, YOLO segmentation, ResNet route, Florence route, manual review, persistence, JSON/CSV export, annotated-image export, and an HTML inspection report.

On an AMD or CPU-only machine, model routes that require CUDA may fail or be unavailable. Contributors can still use the UI for review/export of existing finding records and work on the frontend/API without claiming new model metrics.

## NVIDIA setup

The validated project environment is:

- Python 3.11
- PyTorch `2.6.0+cu124`
- torchvision `0.21.0+cu124`
- CUDA-capable NVIDIA driver
- Ultralytics `8.4.143`
- Transformers `5.16.1`
- Accelerate `1.14.0`

The repository currently stores the validated CUDA wheels under `vendor/wheels/` and records their hashes in `config/cuda_environment.yaml`.

Install the pinned CUDA wheels only in `.venv` using the repository setup process, then verify:

```powershell
.\.venv\Scripts\python.exe scripts\setup_cuda_env.ps1
```

If PowerShell execution policy blocks the script for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

Expected result:

```text
torch.cuda.is_available() == True
```

NVIDIA contributors can run the completed evaluation commands from the project report and use `nvidia-smi` to record GPU memory and utilization.

## AMD reality check

The current training and evaluation entrypoints explicitly require CUDA. Examples include:

- `scripts/train_resnet_comparison.py` refuses CPU fallback.
- `scripts/run_florence_pipeline.py` requires CUDA.
- `scripts/run_yolo_sam_comparison.py` requires CUDA.
- The harness model routes assume CUDA-backed model inference.

Do not install the NVIDIA CUDA wheels on an AMD machine, and do not report CPU execution as equivalent GPU training.

### AMD Windows

ROCm is not a drop-in replacement for this Windows CUDA environment. The current project scripts are not validated with AMD DirectML or Windows ROCm. An AMD Windows contributor should use CPU mode for preparation, harness/UI, reports, and tests, or use an NVIDIA/Linux machine for the model runs.

### AMD Linux with ROCm

A real AMD training port requires a Linux ROCm environment matched to the GPU and supported PyTorch/ROCm versions. The contributor must:

1. Install the ROCm-supported PyTorch build from the official PyTorch/ROCm instructions for that GPU.
2. Verify `torch.cuda.is_available()` and `torch.version.hip` in the ROCm environment.
3. Add a device abstraction instead of hard-coding CUDA-only checks.
4. Replace CUDA-specific AMP and memory calls with device-aware equivalents.
5. Validate Ultralytics YOLO, Transformers Florence, and SAM independently.
6. Re-run every metric on the AMD environment; do not compare a CPU/DirectML result to CUDA without recording the runtime.

A minimal ROCm check is:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.hip)"
```

A successful ROCm check does not prove that this repository is ported. The scripts and model libraries still need validation.

## Recommended contributor split

- AMD/CPU contributors: harness, frontend, API, manifests, source review, target-site metadata, S2DS class-index work, error review, documentation, and tests.
- NVIDIA contributors: YOLO, ResNet, Florence, SAM training/inference, throughput, VRAM, and source-specific model evaluations.
- Any contributor: report review, licenses, taxonomy decisions, qualitative error labels, and acceptance documentation.

## Remaining acceptance inputs

These are project inputs, not GPU setup problems:

1. Provide `data/manifests/s2ds_test_class_index.csv` if six-class S2DS scoring is required. The current `s2ds_test_class_index_status.csv` deliberately records that class identity is unavailable.
2. Provide approved target-site images, building/site/session metadata, ground-truth labels, and evaluation permission using `data/manifests/target_site_evaluation_template.csv`.
3. Perform human image-level review of false positives, false negatives, empty predictions, thin defects, and out-of-domain cases. Start with `runs/evaluation/error_review_queue.json`.
4. Complete the qualitative acceptance decision: demo-only versus further deployment testing, then record limitations and the final model decision.

## Before submitting a contribution

```powershell
git status --short --branch
git diff --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
git pull --rebase origin main
git push origin <your-branch>
```

Never commit model checkpoints, generated `runs/` outputs, local uploads, caches, secrets, or private target-site images.
