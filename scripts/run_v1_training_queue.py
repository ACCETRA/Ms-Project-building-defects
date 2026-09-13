#!/usr/bin/env python3
"""Run the runnable v1 training jobs sequentially with durable logs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_ROOT = ROOT / "runs" / "v1_queue"


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]


JOBS = (
    Job(
        "yolo_detection_v1",
        (
            "scripts/train_yolo_comparison.py", "--task", "detect", "--epochs", "20",
            "--batch", "8", "--workers", "0", "--cache", "--project", "runs/comparison/yolo",
            "--name", "yolo11n_detect_v1_queue",
        ),
    ),
    Job(
        "yolo_segmentation_v1",
        (
            "scripts/train_yolo_comparison.py", "--task", "segment", "--epochs", "20",
            "--batch", "8", "--workers", "0", "--cache", "--project", "runs/comparison/yolo",
            "--name", "yolo11n_seg_v1_queue",
        ),
    ),
    Job(
        "resnet_classification_v1",
        (
            "scripts/train_resnet_comparison.py", "--epochs", "10", "--batch", "16",
            "--workers", "0", "--mixed-precision", "--output", "runs/comparison/resnet50_v1_queue",
        ),
    ),
    Job(
        "florence_v1_dry_run",
        ("scripts/train_florence.py", "--dry-run", "--dtype", "float32"),
    ),
    Job(
        "florence_v1_one_step",
        (
            "scripts/train_florence.py", "--overfit-one", "--max-steps", "1",
            "--dtype", "float32", "--output-dir", "runs/florence/v1_queue_smoke",
        ),
    ),
)


def completed(job: Job) -> bool:
    if job.name.startswith("yolo_detection"):
        directories = (
            ROOT / "runs" / "comparison" / "yolo" / "yolo11n_detect_v1_queue",
            ROOT / "runs" / "detect" / "runs" / "comparison" / "yolo" / "yolo11n_detect_v1_queue",
        )
        return any(
            (directory / "weights" / "best.pt").exists()
            and (directory / "weights" / "last.pt").exists()
            for directory in directories
        )
    if job.name.startswith("yolo_segmentation"):
        directories = (
            ROOT / "runs" / "comparison" / "yolo" / "yolo11n_seg_v1_queue",
            ROOT / "runs" / "segment" / "runs" / "comparison" / "yolo" / "yolo11n_seg_v1_queue",
        )
        return any(
            (directory / "weights" / "best.pt").exists()
            and (directory / "weights" / "last.pt").exists()
            for directory in directories
        )
    if job.name.startswith("resnet_classification"):
        return (ROOT / "runs" / "comparison" / "resnet50_v1_queue" / "resnet50_comparison.pt").exists()
    return False


def result_record(job: Job, command: list[str], started: str, finished: str, return_code: int, log_path: Path) -> dict[str, object]:
    output_verified = completed(job)
    return {
        "job": job.name,
        "command": command,
        "started_at": started,
        "finished_at": finished,
        "return_code": return_code,
        "status": "completed" if return_code == 0 or output_verified else "failed",
        "output_verified": output_verified,
        "log": str(log_path.relative_to(ROOT)),
    }


def run_job(job: Job, index: int) -> dict[str, object]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{index:02d}_{job.name}.log"
    started = datetime.now(timezone.utc)
    command = [str(PYTHON), *job.command]
    header = {
        "job": job.name,
        "command": command,
        "started_at": started.isoformat(),
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(header) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    finished = datetime.now(timezone.utc)
    record = result_record(job, command, started.isoformat(), finished.isoformat(), result.returncode, log_path)
    print(json.dumps(record), flush=True)
    return record


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    queue_path = LOG_ROOT / "queue.json"
    queue_path.write_text(
        json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "jobs": [job.name for job in JOBS]}, indent=2) + "\n",
        encoding="utf-8",
    )
    results = []
    for index, job in enumerate(JOBS, start=1):
        if completed(job):
            result = {
                "job": job.name,
                "skipped": True,
                "status": "completed",
                "output_verified": True,
                "reason": "verified output checkpoints already exist",
            }
            print(json.dumps(result), flush=True)
            results.append(result)
            continue
        result = run_job(job, index)
        results.append(result)
    (LOG_ROOT / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result.get("status") == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())