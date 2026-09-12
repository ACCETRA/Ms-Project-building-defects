#!/usr/bin/env python3
"""Higher-throughput baseline improvement script for the feasibility comparison models."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run(command: list[str]) -> None:
    print(f"RUNNING: {' '.join(command)}")
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    # Strict but realistic improvement path: increase throughput while holding the same manifest policy.
    run([
        str(PYTHON), "scripts/train_yolo_comparison.py",
        "--task", "detect",
        "--epochs", "10",
        "--batch", "4",
        "--workers", "2",
        "--amp",
        "--cache",
        "--project", "runs/comparison/yolo",
        "--name", "yolo11n_detect_improved",
    ])
    run([
        str(PYTHON), "scripts/train_yolo_comparison.py",
        "--task", "segment",
        "--epochs", "10",
        "--batch", "4",
        "--workers", "2",
        "--amp",
        "--cache",
        "--project", "runs/comparison/yolo",
        "--name", "yolo11n_seg_improved",
    ])
    run([
        str(PYTHON), "scripts/train_resnet_comparison.py",
        "--epochs", "10",
        "--batch", "8",
        "--workers", "2",
        "--mixed-precision",
        "--output", "runs/comparison/resnet50_improved",
    ])


if __name__ == "__main__":
    main()
