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

## When the dataset arrives separately

Do not put the dataset beside the repository and do not commit the dataset to Git. Copy or extract each supplied package into the repository locations expected by the scripts, preserving the filenames and directory structure below.

Expected public-source locations:

```text
datasets/building-target/S2DS/s2ds.zip
datasets/CiF-tiled/data/test_tiled-*.parquet
datasets/UAV-candidates/UAV75/test_img/
datasets/UAV-candidates/UAV75/test_lab/
datasets/DACL10K/dacl10k_v2_devphase.zip
```

The contributor should receive the source license/readme files with each dataset. Do not rename archive members, edit labels, or recompress archives. The source archives must remain immutable.

After copying the public datasets, verify that the expected paths exist:

```powershell
@(
	'datasets/building-target/S2DS/s2ds.zip',
	'datasets/CiF-tiled/data',
	'datasets/UAV-candidates/UAV75/test_img',
	'datasets/UAV-candidates/UAV75/test_lab',
	'datasets/DACL10K/dacl10k_v2_devphase.zip'
) | ForEach-Object { "$_ : $(Test-Path $_)" }
```

Then run the non-destructive checks:

```powershell
.\.venv\Scripts\python.exe scripts\audit_datasets.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If a supplied dataset is unavailable, do not create empty placeholder folders and do not mark its evaluation complete. Record it as `blocked_pending_data`.

### S2DS handoff

The S2DS archive is needed to identify the 93 test images, but its test masks are binary. If six-class scoring is required, the dataset handoff must also include an authoritative class-index file or source evidence from which a reviewer can create one.

Create this file after review:

```text
data/manifests/s2ds_test_class_index.csv
```

Required columns:

```text
sample_id,class_label,source_evidence,reviewer_id,reviewed_at,decision,notes
```

Allowed `class_label` values:

```text
crack
spalling
rust_staining
efflorescence_leaching
no_visible_target_defect
unknown_review
```

There must be exactly one reviewed row per S2DS test image. Never infer the class from the binary mask color. If the separate handoff does not contain class evidence, leave six-class scoring blocked and report only the valid binary foreground result.

### Target-site handoff

Private target-site data must stay outside Git. Store it in a local path such as:

```text
D:/project-private/target-site/
```

Do not copy private images into `datasets/`, `data/`, `runs/`, or the repository unless the owner explicitly approves that storage and licensing arrangement.

The handoff must include:

- approved RGB images;
- boxes, masks, or reviewed-negative labels;
- building, site, facade/floor/zone, and capture-session metadata;
- permission/consent and license evidence;
- stable SHA-256 hashes for images and labels;
- a statement that the buildings/sessions are unseen during training and threshold selection.

Populate the repository manifest with metadata and external paths only:

```text
data/manifests/target_site_evaluation_template.csv
```

Set `approved_for_evaluation=true` only after the owner/advisor has approved the permission record. Do not run or report target-site metrics while approval or ground truth is missing.

### Dataset handoff checklist

The person receiving the data should confirm:

```text
[ ] Public archives copied to the expected paths
[ ] Source README/license files preserved
[ ] Archive hashes recorded where available
[ ] Raw archives not edited or recompressed
[ ] S2DS class evidence supplied, or six-class scoring marked blocked
[ ] Target-site permission supplied, or target-site evaluation marked blocked
[ ] Private images stored outside Git
[ ] No checkpoints, caches, or private images staged for commit
```

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

## Can an AMD contributor finish the remaining work?

Yes. All five remaining acceptance tasks are CPU/data/review tasks. They do not require ROCm, CUDA, model training, or an NVIDIA GPU:

| Task | AMD/CPU capable? | What is required |
|---|---|---|
| S2DS class-index manifest | Yes | Human/source documentation identifying the class for each test mask |
| Target-site evaluation data | Yes | Approved images, labels, metadata, and permission |
| Human error review | Yes | Browser or image viewer, source outputs, and reviewer decisions |
| Demo-versus-deployment decision | Yes | Team/advisor review of metrics, errors, and limitations |
| Final limitations/model decision | Yes | Update `docs/DECISION_LOG.md` and the acceptance report |

AMD contributors must not rerun CUDA-only model metrics and present CPU or DirectML output as equivalent. They can complete the acceptance package using the already-generated checkpoints and evaluation artifacts.

## Remaining acceptance work

These are project inputs, not GPU setup problems:

### 1. S2DS class-index manifest

Only do this if six-class S2DS scoring is required. The supplied S2DS test masks are binary, so a contributor must supply an authoritative class mapping rather than infer it from mask pixels.

Create:

```text
data/manifests/s2ds_test_class_index.csv
```

Required columns:

```text
sample_id,class_label,source_evidence,reviewer_id,reviewed_at,decision,notes
```

Allowed `class_label` values are `crack`, `spalling`, `rust_staining`, `efflorescence_leaching`, `no_visible_target_defect`, or `unknown_review`. Use `unknown_review` when the source evidence is insufficient. Do not use the binary mask color as a class label.

Before committing it, verify:

```powershell
Import-Csv data/manifests/s2ds_test_class_index.csv | Measure-Object
```

There must be one reviewed row per S2DS test image, with no duplicate `sample_id` values. Keep the original S2DS archive unchanged.

### 2. Approved target-site data

Populate the template:

```text
data/manifests/target_site_evaluation_template.csv
```

Required fields:

```text
image_id,building_id,site_id,capture_session,split,approved_for_evaluation,ground_truth_manifest,notes
```

Also provide, outside Git unless explicitly approved:

- source images;
- ground-truth boxes/masks or reviewed negative decisions;
- permission/consent record;
- capture session and building metadata;
- license or ownership evidence;
- a stable hash for every image and label file.

Rules:

- Target-site images must be unseen during training and threshold selection.
- Reserve complete buildings or capture sessions for evaluation.
- Do not commit private images, personal information, access credentials, or sensitive site coordinates.
- Set `approved_for_evaluation=true` only after the owner/advisor approves the permission record.

No target-site score may be generated while the template is empty or approval is false.

### 3. Human image-level error review

Start from:

```text
runs/evaluation/error_review_queue.json
```

Review the source outputs and, where available, prediction/ground-truth overlays. Record one decision per reviewed image using a local review CSV or the harness. Minimum review fields:

```text
source,sample_id,reviewer_id,review_state,error_type,affected_classes,notes,reviewed_at
```

Allowed `review_state` values:

```text
correct, false_positive, false_negative, mixed, unusable, out_of_domain
```

At minimum inspect:

- DACL: crack, spalling, rust, and mapped-class misses;
- S2DS: binary foreground misses and background confusion;
- UAV75: thin cracks and planking false positives;
- CiF: small cracks, crowded instances, and domain shift;
- empty predictions and rejected/low-quality images.

An AMD contributor can do this with the browser harness, an image viewer, and the JSON/CSV outputs. No GPU is needed.

### 4. Acceptance decision

Review the locked CUBIT, CODEBRIM, CiF, S2DS, UAV75, and DACL results together with the error review. Choose one explicit status:

```text
demo_only
further_deployment_testing
blocked_pending_data
```

For this current result set, do not choose `further_deployment_testing` without addressing the very low cross-domain segmentation results and completing human error review.

Record the decision in `docs/DECISION_LOG.md` with:

- decision ID and date;
- selected status;
- evidence reviewed;
- known failure modes;
- whether target-site validation is complete;
- conditions required before deployment testing.

### 5. Final limitations and model decision

Update `docs/V1_COMPARISON_REPORT.md` with:

- final acceptance status;
- per-source metrics and sample counts;
- S2DS binary-only limitation unless the class-index manifest is supplied;
- target-site status;
- human error-review totals and representative failures;
- manual-review requirement;
- pixel-only measurement limitation unless calibration is valid;
- explicit statement that the system is not a structural-safety determination;
- chosen model route and why it was selected.

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_v1_comparison_report.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
git diff --check
```

## Completion gate

The project can be called complete as an **academic FYP prototype** when:

- the S2DS limitation is either accepted in writing or the class-index manifest is supplied;
- target-site evaluation is either completed with permission and labels or explicitly marked out of scope;
- human error review is completed and summarized;
- the demo/deployment-testing decision is recorded;
- the final report contains the limitations, source metrics, hashes, licenses, and model decision;
- the harness upload/review/export workflow passes its smoke test;
- the regression and markdown-link checks pass.

This completion gate does not mean the model is production-ready or structurally safe. Deployment testing requires additional target-domain evidence, professional review, and resolution of the documented cross-domain failures.

## Tasks that cannot be completed without the handoff

Do not claim these tasks are complete from the repository alone:

| Task | Required external evidence | Valid completion claim |
|---|---|---|
| S2DS six-class manifest | Authoritative per-image class evidence | Six-class S2DS scoring is complete only after every test image has a reviewed class row |
| Target-site evaluation | Approved images, labels, metadata, permission, and hashes | Target-site evaluation is complete only for the supplied approved buildings/sessions |
| Reproduced training | The contributor's own environment, required data, checkpoints, logs, and hardware record | Say `training reproduced on contributor machine` only after the run and verification logs exist |
| Re-run source metrics | Raw source annotations and matching source archives | Report only the sources actually rerun, with source version and hash |
| Independent dataset/license verification | Supplied archives, license/readme files, and hash records | Mark each source `verified`, `missing`, or `not independently verified` |

The following are not acceptable substitutes:

- assigning S2DS classes from binary mask colors;
- using public images as target-site data without approval;
- claiming a contributor reproduced training because a checkpoint was downloaded;
- claiming source metrics were rerun from summary JSON files alone;
- claiming license or hash verification without the supplied archive and license files.

If any required evidence is missing, record the status as `blocked_pending_data` and continue with documentation, harness review, and other non-blocked work.

## Recommended order after receiving data

1. Copy the supplied public datasets to the expected paths and run the non-destructive checks.
2. Confirm the S2DS class-index evidence status.
3. Confirm target-site permission and keep private data outside Git.
4. Review `runs/evaluation/error_review_queue.json` using the existing evaluation outputs.
5. Record the review decisions and acceptance status.
6. Regenerate `docs/V1_COMPARISON_REPORT.md`.
7. Run tests and markdown-link checks.
8. Commit only manifests, review decisions, documentation, and code. Never commit the dataset or private target images.

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
