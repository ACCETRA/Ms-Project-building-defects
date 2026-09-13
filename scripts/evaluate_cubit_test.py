#!/usr/bin/env python3
"""Evaluate completed YOLO checkpoints on immutable CUBIT test archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "datasets/building-target/CUBIT-InSeg/CUBIT-InSeg/test"
CLASSES = ["crack", "spalling", "honeycombing_rock_pocket", "exposed_rebar", "rust_staining", "efflorescence_leaching"]


def extract_test(destination: Path) -> None:
    for archive_name in ("images.zip", "labels.zip"):
        with zipfile.ZipFile(TEST_ROOT / archive_name) as archive:
            archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    if not weights.is_file():
        raise FileNotFoundError(weights)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cubit_test_") as temporary:
        test_root = Path(temporary)
        extract_test(test_root)
        image_paths = sorted((test_root / "images").glob("*"))
        if len(image_paths) != 701:
            raise RuntimeError(f"Expected 701 locked test images, found {len(image_paths)}")
        data_yaml = test_root / "data.yaml"
        data_yaml.write_text(
            "path: {}\ntrain: {}\nval: {}\nnames: {}\n".format(
                test_root.as_posix(),
                (test_root / "images").as_posix(),
                (test_root / "images").as_posix(),
                json.dumps(CLASSES),
            ),
            encoding="utf-8",
        )
        evaluation_weights = test_root / weights.name
        shutil.copy2(weights, evaluation_weights)
        from ultralytics import YOLO

        model = YOLO(str(evaluation_weights))
        metrics = model.val(
            data=str(data_yaml),
            task=args.task,
            split="val",
            project=str(args.output),
            name=f"cubit_locked_{args.task}",
            exist_ok=True,
            device=0,
            workers=0,
            conf=args.conf,
            plots=False,
        )
        result = {
            "dataset": "CUBIT-InSeg locked test",
            "test_images": len(image_paths),
            "task": args.task,
            "weights": str(weights),
            "confidence": args.conf,
            "metrics": metrics.results_dict,
            "source_archives_untouched": True,
        }
        result_path = args.output / f"cubit_locked_{args.task}.json"
        result_path.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
