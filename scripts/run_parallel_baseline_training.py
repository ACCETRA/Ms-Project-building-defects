#!/usr/bin/env python3
"""Launch the comparison and Florence baseline training jobs in parallel."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run(command: list[str]) -> None:
    print(f"RUNNING: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--florence-epochs", type=int, default=1)
    parser.add_argument("--florence-max-steps", type=int, default=1)
    parser.add_argument("--overfit-one", action="store_true")
    args = parser.parse_args()

    if not PYTHON.is_file():
        raise FileNotFoundError(f"Python env not found: {PYTHON}")

    commands = [
        [str(PYTHON), "scripts/train_yolo_comparison.py", "--task", "detect", "--epochs", str(args.epochs), "--project", "runs/comparison/yolo", "--name", "yolo11n_detect_fyp_parallel"],
        [str(PYTHON), "scripts/train_yolo_comparison.py", "--task", "segment", "--epochs", str(args.epochs), "--project", "runs/comparison/yolo", "--name", "yolo11n_seg_fyp_parallel"],
        [str(PYTHON), "scripts/train_resnet_comparison.py", "--epochs", str(args.epochs), "--output", "runs/comparison/resnet50_parallel"],
        [str(PYTHON), "scripts/train_florence.py", "--targets", "data/manifests/florence_targets_v0_1.jsonl", "--output-dir", "runs/florence/finetune-parallel", "--epochs", str(args.florence_epochs), "--batch-size", "1", "--gradient-accumulation", "8"] + (["--overfit-one", "--max-steps", str(args.florence_max_steps)] if args.overfit_one else []),
    ]

    processes = [subprocess.Popen(command, cwd=str(ROOT)) for command in commands]
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)

    print("All baseline training jobs completed.")


if __name__ == "__main__":
    main()
