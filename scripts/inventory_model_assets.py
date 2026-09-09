"""Create a reproducible inventory of locally downloaded model assets."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = ROOT / "weights"
OUTPUT_ROOT = ROOT / "artifacts" / "model-assets"

MODEL_SOURCES = {
    "yolo11n": ("ultralytics/yolo11n.pt", WEIGHTS_ROOT / "yolo" / "yolo11n.pt"),
    "yolo11n_seg": (
        "ultralytics/yolo11n-seg.pt",
        WEIGHTS_ROOT / "yolo" / "yolo11n-seg.pt",
    ),
    "resnet50": (
        "torchvision/resnet50-11ad3fa6",
        WEIGHTS_ROOT / "resnet" / "resnet50-11ad3fa6.pth",
    ),
    "sam2_1_hiera_tiny": (
        "facebook/sam2.1-hiera-tiny",
        WEIGHTS_ROOT / "sam2.1-hiera-tiny",
    ),
    "florence_2_base_ft": (
        "florence-community/Florence-2-base-ft",
        WEIGHTS_ROOT / "florence-community-2-base-ft",
    ),
    "florence_2_large_ft": (
        "florence-community/Florence-2-large-ft",
        WEIGHTS_ROOT / "florence-community-2-large-ft",
    ),
    "florence_2_base_ft_original_reference": (
        "microsoft/Florence-2-base-ft",
        WEIGHTS_ROOT / "florence-2-base-ft",
    ),
    "florence_2_large_ft_original_reference": (
        "microsoft/Florence-2-large-ft",
        WEIGHTS_ROOT / "florence-2-large-ft",
    ),
}

REQUIRED_MODEL_FILES = {
    "yolo11n": "yolo11n.pt",
    "yolo11n_seg": "yolo11n-seg.pt",
    "resnet50": "resnet50-11ad3fa6.pth",
    "sam2_1_hiera_tiny": "model.safetensors",
    "florence_2_base_ft": "model.safetensors",
    "florence_2_large_ft": "model.safetensors",
    "florence_2_base_ft_original_reference": "model.safetensors",
    "florence_2_large_ft_original_reference": "model.safetensors",
}

REFERENCE_MODEL_IDS = {
    "florence_2_base_ft_original_reference",
    "florence_2_large_ft_original_reference",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and ".cache" not in candidate.parts
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for model_id, (source, local_path) in MODEL_SOURCES.items():
        artifact_role = "origin_reference" if model_id in REFERENCE_MODEL_IDS else "runtime"
        files = model_files(local_path)
        required_path = (
            local_path
            if local_path.is_file()
            else local_path / REQUIRED_MODEL_FILES[model_id]
        )
        ready = required_path.is_file() and required_path.stat().st_size > 0
        total_bytes = 0
        for path in files:
            size = path.stat().st_size
            total_bytes += size
            records.append(
                {
                    "model_id": model_id,
                    "artifact_role": artifact_role,
                    "source": source,
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "bytes": size,
                    "sha256": sha256(path),
                }
            )
        summaries.append(
            {
                "model_id": model_id,
                "artifact_role": artifact_role,
                "source": source,
                "local_path": local_path.relative_to(ROOT).as_posix(),
                "present": bool(files),
                "ready": ready,
                "required_file": required_path.relative_to(ROOT).as_posix(),
                "file_count": len(files),
                "total_bytes": total_bytes,
            }
        )

    missing = [
        item["model_id"]
        for item in summaries
        if item["artifact_role"] == "runtime" and not item["ready"]
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not missing,
        "missing_runtime_models": missing,
        "models": summaries,
        "files": records,
    }

    json_path = OUTPUT_ROOT / "model_asset_inventory.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = OUTPUT_ROOT / "model_asset_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model_id",
                "artifact_role",
                "source",
                "relative_path",
                "bytes",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps({"complete": not missing, "missing": missing, "files": len(records)}))


if __name__ == "__main__":
    main()
