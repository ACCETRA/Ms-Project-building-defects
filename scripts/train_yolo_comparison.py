"""Train the YOLO comparison detector or segmenter from a project manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def split_rows(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted({row["group_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(rows) * fraction))
    validation_groups: set[str] = set()
    validation_count = 0
    for group in groups:
        validation_groups.add(group)
        validation_count += sum(row["group_id"] == group for row in rows)
        if validation_count >= target:
            break
    train = [row for row in rows if row["group_id"] not in validation_groups]
    validation = [row for row in rows if row["group_id"] in validation_groups]
    if not train or not validation:
        raise ValueError("The manifest cannot produce non-empty group-safe train and validation splits")
    return train, validation


def label_lines(row: dict[str, str], task: str, class_ids: dict[str, int]) -> list[str]:
    annotation_path = ROOT / row["normalized_annotation"]
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    width = payload["image"]["width"]
    height = payload["image"]["height"]
    for annotation in payload.get("annotations", []):
        label = annotation.get("canonical_label")
        if label not in class_ids or annotation.get("annotation_status") != "positive":
            continue
        box = annotation.get("box_xywh")
        if not box:
            continue
        x, y, box_width, box_height = box
        values = [
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        ]
        if task == "segment":
            polygons = annotation.get("polygons_xy") or []
            polygon = next((item for item in polygons if len(item) >= 3), None)
            if polygon is None:
                continue
            values = [value for point in polygon for value in (point[0] / width, point[1] / height)]
            lines.append("{} {}".format(class_ids[label], " ".join(f"{value:.8f}" for value in values)))
        else:
            lines.append("{} {}".format(class_ids[label], " ".join(f"{value:.8f}" for value in values)))
    return lines


def prepare_split(rows: list[dict[str, str]], output: Path, split: str, task: str, class_ids: dict[str, int]) -> Path:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    list_path = output / f"{split}.txt"
    paths: list[str] = []
    for row in rows:
        source = ROOT / row["output_image"]
        destination = image_dir / f"{row['sample_id']}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        (label_dir / f"{row['sample_id']}.txt").write_text(
            "\n".join(label_lines(row, task, class_ids)) + "\n", encoding="utf-8"
        )
        paths.append(destination.as_posix())
    list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return list_path


# Sentinel so we can detect whether --manifest / --name were explicitly supplied.
_DETECT_MANIFEST = ROOT / "data/manifests/v1_detection_manifest.csv"
_SEGMENT_MANIFEST = ROOT / "data/manifests/v1_segmentation_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Override manifest path. Defaults to the detection or segmentation "
                             "view depending on --task.")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--model", type=Path, default=ROOT / "weights/yolo/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision when supported.")
    parser.add_argument("--cache", action="store_true", help="Cache image tensors for faster YOLO training on repeated runs.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/comparison/yolo")
    parser.add_argument("--name", default=None,
                        help="Run name. Defaults to 'yolo11n_detect_fyp' or 'yolo11n_seg_fyp' "
                             "based on --task.")
    args = parser.parse_args()

    # Apply task-aware defaults for manifest and run name.
    if args.task == "segment":
        if args.manifest is None:
            args.manifest = _SEGMENT_MANIFEST
        if args.model.name == "yolo11n.pt":
            args.model = ROOT / "weights/yolo/yolo11n-seg.pt"
        if args.name is None:
            args.name = "yolo11n_seg_fyp"
    else:
        if args.manifest is None:
            args.manifest = _DETECT_MANIFEST
        if args.name is None:
            args.name = "yolo11n_detect_fyp"

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    rows = read_rows((ROOT / args.manifest) if not args.manifest.is_absolute() else args.manifest)
    rows = [row for row in rows if row.get("eligibility") and row["geometry_qc"] == "passed"]
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    output = args.project / args.name / "prepared"
    class_ids = {"crack": 0, "spalling": 1, "honeycombing_rock_pocket": 2, "exposed_rebar": 3, "rust_staining": 4, "efflorescence_leaching": 5}
    train_list = prepare_split(train_rows, output, "train", args.task, class_ids)
    val_list = prepare_split(val_rows, output, "val", args.task, class_ids)
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: {}\nval: {}\nnames: {}\n".format(
            train_list.as_posix(), val_list.as_posix(), json.dumps(list(class_ids))
        ),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "task": args.task,
        "model": str(args.model),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "amp": args.amp,
        "cache": args.cache,
        "seed": args.seed,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "train_groups": len({row["group_id"] for row in train_rows}),
        "validation_groups": len({row["group_id"] for row in val_rows}),
        "manifest": str(args.manifest),
    }
    run_dir = args.project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    try:
        model.train(
            data=str(data_yaml),
            task=args.task,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            device=0,
            pretrained=True,
            amp=args.amp,
            cache=args.cache,
            # val=False skips per-epoch validation but Ultralytics always calls
            # final_eval() at the end of _do_train(). We catch its
            # FileNotFoundError below instead of fighting the trainer internals.
            val=False,
        )
    except FileNotFoundError as exc:
        # Ultralytics' path sanitizer strips the apostrophe from "ALI's Project"
        # when reloading best.pt for final_eval. Training and checkpoint saving
        # are already complete at this point. Verify the weights are present at
        # the real path and exit cleanly.
        weight_dirs = [run_dir / "weights"]
        if not args.project.is_absolute():
            run_kind = "segment" if args.task == "segment" else "detect"
            weight_dirs.append(ROOT / "runs" / run_kind / args.project / args.name / "weights")
        existing = next(
            (
                (directory / "best.pt", directory / "last.pt")
                for directory in weight_dirs
                if (directory / "best.pt").exists() and (directory / "last.pt").exists()
            ),
            None,
        )
        if existing is not None:
            best, last = existing
            print(
                f"\nWARNING: Ultralytics final_eval raised FileNotFoundError due to the "
                f"Windows path-sanitizer stripping apostrophes from the project path.\n"
                f"Training is complete. Checkpoints verified at real path:\n"
                f"  best.pt : {best} ({best.stat().st_size / 1e6:.1f} MB)\n"
                f"  last.pt : {last} ({last.stat().st_size / 1e6:.1f} MB)\n"
                f"Suppressed error: {exc}"
            )
        else:
            # Weights are genuinely missing — re-raise so the caller sees the failure.
            raise


if __name__ == "__main__":
    main()
