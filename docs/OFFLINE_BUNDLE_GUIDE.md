# Offline Contributor Bundle

This bundle contains the Project Alpha source code, manifests, documentation, trained/model assets, offline vendor wheels, and acceptance evidence. Raw datasets are intentionally excluded.

## Included

- Source code under `bdi/`, `scripts/`, and `harness/`.
- Frozen manifests under `data/manifests/`.
- Schemas, configs, examples, tests, reports, and decision records.
- Model checkpoints under `weights/`.
- Offline vendor wheels/tools under `vendor/`.
- Generated acceptance artifacts and the consolidated FYP DOCX.

## Excluded

- `datasets/` raw archives and images.
- `data/v1/` and other materialized dataset content.
- `.venv/`, `.git/`, caches, `runs/`, local uploads, and temporary files.
- Private target-site images and labels.

## Offline setup

Create the environment with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The included CUDA wheels are intended for the validated NVIDIA environment. AMD/CPU contributors can run tests, documentation, manifest review, and the harness structure, but the CUDA model routes require a supported runtime.

Start the harness:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

Open `http://127.0.0.1:8765`.

Run verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_markdown_links.py
```

The bundle is a demonstration and verification package. It does not include enough raw data to reproduce dataset construction or full retraining.