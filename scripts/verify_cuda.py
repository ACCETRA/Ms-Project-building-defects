#!/usr/bin/env python3
"""Require a real CUDA operation and record the exact local environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback


def nvidia_smi() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = root / "artifacts" / "environment" / "cuda_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "status": "failed",
    }

    try:
        import torch
        import torchvision

        report.update(
            {
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "nvidia_smi": nvidia_smi(),
            }
        )
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see CUDA; CPU fallback is not accepted")

        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        left = torch.randn((1024, 1024), device=device, dtype=torch.float16)
        right = torch.randn((1024, 1024), device=device, dtype=torch.float16)
        product = left @ right
        checksum = float(product.float().sum().cpu())
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        report.update(
            {
                "device_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "device_total_memory_bytes": properties.total_memory,
                "float16_matmul_shape": [1024, 1024],
                "float16_matmul_checksum": checksum,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "free_memory_after_test_bytes": free_bytes,
                "total_memory_after_test_bytes": total_bytes,
                "status": "passed",
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(1) from exc

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
