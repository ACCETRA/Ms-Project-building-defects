"""Verify that every pinned checkpoint can be loaded without training or inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "model-assets" / "model_load_verification.json"


def load_yolo(path: Path) -> dict[str, object]:
    from ultralytics import YOLO

    # Ultralytics sanitizes apostrophes in absolute model paths. Load by bare
    # filename from the checkpoint directory so it cannot rewrite the path or
    # silently download a duplicate into ``D:\ALIs Project``.
    previous_directory = Path.cwd()
    try:
        os.chdir(path.parent)
        model = YOLO(path.name)
    finally:
        os.chdir(previous_directory)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    return {"parameter_count": parameter_count}


def load_resnet(path: Path) -> dict[str, object]:
    from torchvision.models import resnet50

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    result = model.load_state_dict(state_dict, strict=True)
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def load_sam(path: Path) -> dict[str, object]:
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(path, local_files_only=True)
    model = Sam2Model.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "processor": type(processor).__name__,
        "compatibility_notice": (
            "The source config is sam2_video; initial image prompting uses its "
            "Sam2Model-compatible subset and still requires an inference smoke test."
        ),
    }


def load_florence(path: Path) -> dict[str, object]:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Florence2ForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "processor": type(processor).__name__,
    }


LOADERS: dict[str, tuple[Path, Callable[[Path], dict[str, object]]]] = {
    "yolo11n": (ROOT / "weights" / "yolo" / "yolo11n.pt", load_yolo),
    "yolo11n_seg": (ROOT / "weights" / "yolo" / "yolo11n-seg.pt", load_yolo),
    "resnet50": (
        ROOT / "weights" / "resnet" / "resnet50-11ad3fa6.pth",
        load_resnet,
    ),
    "sam2_1_hiera_tiny": (ROOT / "weights" / "sam2.1-hiera-tiny", load_sam),
    "florence_2_base_ft": (
        ROOT / "weights" / "florence-community-2-base-ft",
        load_florence,
    ),
    "florence_2_large_ft": (
        ROOT / "weights" / "florence-community-2-large-ft",
        load_florence,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        metavar="MODEL_ID",
        help=f"Model IDs to verify; defaults to all. Choices: {', '.join(sorted(LOADERS))}",
    )
    args = parser.parse_args()
    selected_models = args.models or list(LOADERS)
    unknown = sorted(set(selected_models) - set(LOADERS))
    if unknown:
        parser.error(f"unknown model ID(s): {', '.join(unknown)}")

    records: list[dict[str, object]] = []
    for model_id in selected_models:
        path, loader = LOADERS[model_id]
        started = time.perf_counter()
        record: dict[str, object] = {
            "model_id": model_id,
            "local_path": path.relative_to(ROOT).as_posix(),
        }
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            record.update(loader(path))
            record["status"] = "passed"
        except Exception as exc:  # Preserve the exact per-model failure in the report.
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["load_seconds"] = round(time.perf_counter() - started, 3)
            records.append(record)
            gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "deserialization_only_no_training_no_inference",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "passed": all(record["status"] == "passed" for record in records),
        "models": records,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
