# V1 Training Runbook

This document is the operator guide for the frozen v1 training queue on the Windows CUDA workstation.

## Current run

The active queue runs jobs sequentially so only one model uses the Quadro T2000 at a time:

1. YOLO11n detection, 20 epochs
2. YOLO11n segmentation, 20 epochs
3. ResNet-50 multilabel classification, 10 epochs
4. Florence v1 dry-run
5. Florence one-step CUDA smoke fine-tune

The queue uses Windows-safe dataloader settings (`workers=0`), larger batches where possible, and RAM image caching. It does not include locked test data.

## Check whether the queue is running

From the repository root:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match 'run_v1_training_queue|train_yolo_comparison.py|train_resnet_comparison.py|train_florence.py' } |
  Select-Object ProcessId,ParentProcessId,CommandLine
```

Check the queue files:

```powershell
Get-ChildItem runs/v1_queue
Get-Content runs/v1_queue/summary.json
```

The queue writes one log per job:

```text
runs/v1_queue/01_yolo_detection_v1.log
runs/v1_queue/02_yolo_segmentation_v1.log
runs/v1_queue/03_resnet_classification_v1.log
runs/v1_queue/04_florence_v1_dry_run.log
runs/v1_queue/05_florence_v1_one_step.log
```

Follow the active detector log:

```powershell
Get-Content runs/v1_queue/01_yolo_detection_v1.log -Tail 20 -Wait
```

The queue has entered model training when the log contains:

```text
Starting training for 20 epochs...
```

An active epoch usually contains a line like:

```text
1/20 ... GPU_mem ...
```

## Check GPU activity

Use `nvidia-smi`, not only the Windows Task Manager 3D graph:

```powershell
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
```

Expected states:

- `0-100 MiB`, `0%`: preparing data, scanning labels, or idle.
- Around `1-2 GiB`, non-zero utilization: YOLO training is using CUDA.
- High utilization with a changing epoch/batch counter: active model computation.

## Start the queue if it has stopped

First check that no queue or training process is active. Then run:

```powershell
Set-Location "D:\ALI's Project"
.\.venv\Scripts\python.exe scripts\run_v1_training_queue.py
```

Run this in a terminal that remains open. The queue starts from job 1 and uses the configured output names. Existing completed output directories may be reused or overwritten by Ultralytics because `exist_ok=True` is enabled.

## If the queue stops during a job

1. Read the corresponding log in `runs/v1_queue`.
2. Check whether the job created weights or history.
3. Check the return code in `runs/v1_queue/summary.json` if the queue reached the summary step.
4. Do not start another GPU job while the failed process is still present.

For YOLO, check:

```powershell
Get-ChildItem runs/comparison/yolo/yolo11n_detect_v1_queue -Recurse
Get-ChildItem runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights
```

For ResNet, check:

```powershell
Get-ChildItem runs/comparison/resnet50_v1_queue
Get-Content runs/comparison/resnet50_v1_queue/history.json
```

For Florence, check:

```powershell
Get-ChildItem runs/florence/v1_queue_smoke -Recurse
```

The Windows Ultralytics path sanitizer may save YOLO output below `runs/detect/runs/comparison/...` instead of the expected project path. A valid YOLO run has `results.csv` and `weights/best.pt` plus `weights/last.pt`.

## If the queue must be stopped

Stop only the queue and its descendants after confirming the process IDs:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match 'run_v1_training_queue|train_yolo_comparison.py|train_resnet_comparison.py|train_florence.py' } |
  Select-Object ProcessId,ParentProcessId,CommandLine
```

Then stop the identified process IDs:

```powershell
Stop-Process -Id <queue-or-job-pid> -Force
```

Do not stop unrelated Python processes without checking their command line first.

## What to do after detection finishes

1. Confirm `results.csv`, `best.pt`, and `last.pt` exist.
2. Record the final epoch and metrics from `results.csv`.
3. Confirm that the queue automatically starts `yolo_segmentation_v1`.
4. Monitor the segmentation log with:

```powershell
Get-Content runs/v1_queue/02_yolo_segmentation_v1.log -Tail 20 -Wait
```

5. Do not run a second detection job at the same time.

After segmentation finishes, the queue should proceed to ResNet, then the Florence checks. Validate each output before using it in the final comparison report.

## After the queue completes

Run the regression tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Then inspect the queue summary:

```powershell
Get-Content runs/v1_queue/summary.json
```

The next project work is:

1. Compare v1 metrics against the 300-sample feasibility baseline.
2. Add multilabel ResNet precision, recall, F1, PR-AUC, and per-class results.
3. Evaluate YOLO detection and segmentation on locked test data only after thresholds are fixed using validation data.
4. Run the YOLO-to-SAM route and compare mask quality and latency.
5. Finish the browser upload, review, and export workflow.

## GitHub commands

Check, commit, synchronize, and push:

```powershell
Set-Location "D:\ALI's Project"
git status --short --branch
git diff --stat
git diff --check
git add docs/V1_TRAINING_RUNBOOK.md scripts/run_v1_training_queue.py
git commit -m "Add v1 training operations runbook"
git pull --rebase origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git push origin main
git status --short --branch
```

If rebase conflicts occur, run `git status`, resolve the listed files, then use `git add <resolved-file>` and `git rebase --continue`. Do not commit model checkpoints, cache files, or large training logs unless repository policy requires them.

## Dataset and safety rules

- Do not edit raw archives.
- Do not add CUBIT test records to training or threshold selection.
- Do not tune on locked evaluation outputs.
- Keep physical measurements pixel-only unless a valid scale method exists.
- Keep the manual-review and non-structural-safety limitations in every product report.