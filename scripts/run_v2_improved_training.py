#!/usr/bin/env python3
"""V2 improved training queue — better augmentation, more epochs, validation-based checkpointing.

Saves all outputs under runs/v2_improved/ to preserve the original v1 checkpoints.
Runs sequentially so only one model uses the Quadro T2000 at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_ROOT = ROOT / "runs" / "v2_improved"


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    description: str


JOBS = (
    Job(
        "yolo_detection_v2",
        (
            "scripts/train_yolo_v2.py", "--task", "detect", "--epochs", "50",
            "--batch", "8", "--workers", "0", "--cache",
            "--project", "runs/v2_improved/yolo",
            "--name", "yolo11n_detect_v2",
        ),
        "YOLO11n detection with augmentation, validation, and 50 epochs",
    ),
    Job(
        "yolo_segmentation_v2",
        (
            "scripts/train_yolo_v2.py", "--task", "segment", "--epochs", "50",
            "--batch", "8", "--workers", "0", "--cache",
            "--project", "runs/v2_improved/yolo",
            "--name", "yolo11n_seg_v2",
        ),
        "YOLO11n-seg segmentation with augmentation, validation, and 50 epochs",
    ),
    Job(
        "resnet_classification_v2",
        (
            "scripts/train_resnet_v2.py", "--epochs", "30", "--batch", "16",
            "--workers", "0", "--mixed-precision",
            "--output", "runs/v2_improved/resnet50_v2",
        ),
        "ResNet-50 with augmentation, cosine LR, class weighting, and 30 epochs",
    ),
)


def run_job(job: Job, index: int) -> dict[str, object]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{index:02d}_{job.name}.log"
    started = datetime.now(timezone.utc)
    command = [str(PYTHON), *job.command]
    header = {
        "job": job.name,
        "description": job.description,
        "command": command,
        "started_at": started.isoformat(),
    }
    print(f"\n{'='*60}")
    print(f"[{index}] Starting: {job.description}")
    print(f"    Command: {' '.join(job.command)}")
    print(f"    Log: {log_path}")
    print(f"    Started: {started.isoformat()}")
    print(f"{'='*60}\n", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(header) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)

    finished = datetime.now(timezone.utc)
    elapsed = (finished - started).total_seconds()
    record = {
        "job": job.name,
        "description": job.description,
        "command": command,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
        "return_code": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed",
        "log": str(log_path.relative_to(ROOT)),
    }
    print(f"\n  -> {job.name}: {record['status']} in {record['elapsed_human']}", flush=True)
    return record


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V2 IMPROVED TRAINING QUEUE")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Output: {LOG_ROOT}")
    print("=" * 60)

    queue_start = time.time()
    results = []
    for index, job in enumerate(JOBS, start=1):
        result = run_job(job, index)
        results.append(result)
        if result["status"] == "failed":
            print(f"\n  WARNING: {job.name} failed with code {result['return_code']}")
            print(f"  Check log: {result['log']}")

    total_elapsed = time.time() - queue_start
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_human": f"{int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m",
        "jobs": results,
    }
    (LOG_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"QUEUE COMPLETE — Total: {summary['total_elapsed_human']}")
    for r in results:
        print(f"  {r['job']}: {r['status']} ({r['elapsed_human']})")
    print(f"{'='*60}")

    return 0 if all(r.get("status") == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
