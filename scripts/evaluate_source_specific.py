#!/usr/bin/env python3
"""Evaluate v1 YOLO checkpoints through source-specific test adapters."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["crack", "spalling", "honeycombing_rock_pocket", "exposed_rebar", "rust_staining", "efflorescence_leaching"]
CIF_CLASSES = {2: "crack", 3: "crack", 4: "crack", 5: "rust_staining", 6: "spalling"}
DACL_CLASSES = {"Crack": "crack", "ACrack": "crack", "Spalling": "spalling", "Rust": "rust_staining", "Efflorescence": "efflorescence_leaching", "Rockpocket": "honeycombing_rock_pocket", "ExposedRebars": "exposed_rebar"}


def write_box(label: str, box: tuple[float, float, float, float], width: int, height: int) -> str:
    x, y, w, h = box
    return f"{CLASSES.index(label)} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}"


def write_polygon(label: str, points: list[list[float]], width: int, height: int) -> str:
    values = " ".join(f"{point[0] / width:.8f} {point[1] / height:.8f}" for point in points)
    return f"{CLASSES.index(label)} {values}"


def mask_objects(mask: np.ndarray, label: str) -> list[tuple[tuple[float, float, float, float], list[list[float]]]]:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        polygon = contour.reshape(-1, 2).astype(float).tolist()
        if len(polygon) >= 3:
            objects.append(((float(x), float(y), float(w), float(h)), polygon))
    return objects


def add_sample(root: Path, name: str, image: Image.Image, labels: list[str], task: str) -> None:
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{name}.jpg"
    image.convert("RGB").save(image_path, quality=95)
    lines = []
    for label, box, polygon in labels:
        lines.append(write_polygon(label, polygon, image.width, image.height) if task == "segment" else write_box(label, box, image.width, image.height))
    (label_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def adapt_uav(root: Path, task: str) -> int:
    image_root = ROOT / "datasets/UAV-candidates/UAV75/test_img"
    mask_root = ROOT / "datasets/UAV-candidates/UAV75/test_lab"
    count = 0
    for image_path in sorted(image_root.glob("*.jpg")):
        mask = np.asarray(Image.open(mask_root / f"{image_path.stem}.png").convert("RGB"))[:, :, 0]
        objects = mask_objects((mask == 255).astype(np.uint8), "crack")
        add_sample(root, image_path.stem, Image.open(image_path), [("crack", box, polygon) for box, polygon in objects], task)
        count += 1
    return count


def adapt_s2ds(root: Path, task: str) -> int:
    archive = ROOT / "datasets/building-target/S2DS/s2ds.zip"
    count = 0
    with zipfile.ZipFile(archive) as source:
        names = sorted(name for name in source.namelist() if name.startswith("test/") and name.endswith(".png") and not name.endswith("_lab.png"))
        for name in names:
            stem = Path(name).stem
            image = Image.open(io.BytesIO(source.read(name))).convert("RGB")
            mask = np.asarray(Image.open(io.BytesIO(source.read(f"test/{stem}_lab.png"))).convert("RGB"))
            foreground = np.any(mask > 0, axis=2).astype(np.uint8)
            objects = mask_objects(foreground, "crack")
            add_sample(root, f"s2ds_{stem}", image, [("crack", box, polygon) for box, polygon in objects], task)
            count += 1
    return count


def adapt_cif(root: Path, task: str, limit: int | None = None) -> int:
    import pyarrow.parquet as pq
    count = 0
    for shard in sorted((ROOT / "datasets/CiF-tiled/data").glob("test_*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                image_data = row["image"]["bytes"]
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                objects = row["objects"]
                labels = []
                for category, box, segmentation in zip(objects["category_id"], objects["bbox"], objects["segmentation"], strict=False):
                    label = CIF_CLASSES.get(int(category))
                    if label is None:
                        continue
                    polygon = segmentation[0] if segmentation and segmentation[0] else []
                    points = [[float(polygon[i]), float(polygon[i + 1])] for i in range(0, len(polygon) - 1, 2)]
                    if len(points) >= 3:
                        labels.append((label, tuple(float(v) for v in box), points))
                add_sample(root, f"cif_{row['image_id']}", image, labels, task)
                count += 1
                if limit is not None and count >= limit:
                    return count
    return count


def adapt_dacl(root: Path, task: str) -> int:
    archive = ROOT / "datasets/DACL10K/dacl10k_v2_devphase.zip"
    extractor = ROOT / "vendor/7zip-portable/x64/7za.exe"
    count = 0
    with tempfile.TemporaryDirectory(prefix="dacl_adapter_") as extracted:
        destination = Path(extracted)
        subprocess.run([str(extractor), "x", str(archive), "dacl10k_v2_devphase/images/validation/*", "dacl10k_v2_devphase/annotations/validation/*", f"-o{destination}", "-y"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        image_root = destination / "dacl10k_v2_devphase/images/validation"
        annotation_root = destination / "dacl10k_v2_devphase/annotations/validation"
        for annotation_path in sorted(annotation_root.glob("*.json")):
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_path = image_root / payload["imageName"]
            if not image_path.is_file():
                continue
            image = Image.open(image_path).convert("RGB")
            labels = []
            for shape in payload.get("shapes", []):
                label = DACL_CLASSES.get(shape.get("label"))
                points = shape.get("points", [])
                if label and len(points) >= 3:
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    labels.append((label, (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)), [[float(x), float(y)] for x, y in zip(xs, ys)]))
            add_sample(root, f"dacl_{annotation_path.stem}", image, labels, task)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("cif", "s2ds", "uav75", "dacl"), required=True)
    parser.add_argument("--task", choices=("detect", "segment"), default="segment")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation/source_specific")
    parser.add_argument("--limit", type=int, help="Limit adapted samples for a bounded smoke run.")
    args = parser.parse_args()
    adapters = {"cif": adapt_cif, "s2ds": adapt_s2ds, "uav75": adapt_uav, "dacl": adapt_dacl}
    with tempfile.TemporaryDirectory(prefix=f"source_{args.source}_") as temporary:
        dataset = Path(temporary) / "dataset"
        if args.source == "cif":
            count = adapt_cif(dataset, args.task, args.limit)
        else:
            count = adapters[args.source](dataset, args.task)
        if count == 0:
            raise RuntimeError(f"Adapter produced no {args.source} samples")
        data = dataset / "data.yaml"
        data.write_text(f"path: {dataset.as_posix()}\ntrain: images/val\nval: images/val\nnames: {json.dumps(CLASSES)}\n", encoding="utf-8")
        from ultralytics import YOLO
        safe_weights = Path(temporary) / args.weights.name
        shutil.copy2(args.weights.resolve(), safe_weights)
        model = YOLO(str(safe_weights))
        metrics = model.val(data=str(data), task=args.task, split="val", device=0, workers=0, plots=False, project=str(args.output), name=f"{args.source}_{args.task}", exist_ok=True)
        result = {"source": args.source, "task": args.task, "samples": count, "metrics": metrics.results_dict, "test_archive_untouched": True, "s2ds_semantics": "binary_foreground_proxy; class identity is unavailable" if args.source == "s2ds" else None}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.source}_{args.task}.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
