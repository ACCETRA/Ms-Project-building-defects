#!/usr/bin/env python3
"""Freeze leakage-safe v1 task views from audited registries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MAPPING_VERSION = "BDI-TAX-001@0.1.0-beta"
FIELDS = [
    "sample_id", "source_dataset", "source_version", "source_image_id",
    "parent_image_id", "group_id", "split", "capture_mode", "asset_domain",
    "annotation_type", "native_labels", "canonical_labels", "annotation_status",
    "mapping_version", "license_id", "quality_flags", "sha256", "output_image",
    "normalized_annotation", "normalized_annotation_sha256", "target_annotation_count",
    "segmentation_annotation_count", "geometry_qc", "test_locked",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def box_from_points(points: list[list[float]], width: int, height: int) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Degenerate polygon")
    return [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)]


def read_cubit_annotations(raw: bytes, width: int, height: int) -> tuple[list[dict], int]:
    labels = {0: ("crack", "direct"), 1: ("spalling", "direct")}
    annotations: list[dict] = []
    rejected = 0
    for index, line in enumerate(raw.decode("utf-8", errors="replace").splitlines()):
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        if class_id not in labels or len(values) < 6 or len(values) % 2:
            rejected += 1
            continue
        points = [
            [round(values[offset] * width, 4), round(values[offset + 1] * height, 4)]
            for offset in range(0, len(values), 2)
        ]
        label, strength = labels[class_id]
        try:
            box = box_from_points(points, width, height)
        except ValueError:
            rejected += 1
            continue
        annotations.append({
            "annotation_id": f"polygon-{index}",
            "native_label": label,
            "canonical_label": label,
            "mapping_strength": strength,
            "annotation_status": "positive",
            "box_xywh": box,
            "box_provenance": "derived_from_native_polygon",
            "polygons_xy": [points],
            "segmentation_provenance": "native_polygon",
        })
    if not annotations:
        raise ValueError("No valid CUBIT polygons remained after geometry QA")
    return annotations, rejected


def unresolved_non_test_ids(queue: list[dict[str, str]]) -> set[str]:
    blocked: set[str] = set()
    for row in queue:
        if (row.get("decision") or "").strip() not in {"pending_human_review", "duplicate"}:
            continue
        for side in ("left", "right"):
            record_id = (row.get(f"{side}_record_id") or "").strip()
            split = (row.get(f"{side}_split") or "").strip()
            if record_id and split != "test":
                blocked.add(record_id)
    return blocked


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS + ["task", "eligibility", "geometry"])
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def write_florence_targets(path: Path, rows: list[dict[str, str]], root: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            payload = json.loads((root / row["normalized_annotation"]).read_text(encoding="utf-8"))
            labels = sorted({
                item["canonical_label"]
                for item in payload["annotations"]
                if item.get("annotation_status") == "positive"
                and item.get("canonical_label") != "unknown_review"
            })
            if not labels:
                continue
            target = " ".join(labels)
            stream.write(json.dumps({
                "sample_id": row["sample_id"],
                "image": row["output_image"],
                "target": target,
                "source_dataset": row["source_dataset"],
                "split": row["split"],
                "manifest_version": "v1_0_0",
            }) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = root / "data" / "manifests"
    output_root = root / "data" / "v1"
    image_root = output_root / "images" / "cubit"
    annotation_root = output_root / "annotations" / "cubit"
    image_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)

    decisions = read_csv(manifests / "cubit_exact_dedup_decisions_v1.csv")
    registry = {row["record_id"]: row for row in read_csv(manifests / "cubit_train_val_source_v1.csv")}
    queue = read_csv(root / "artifacts" / "data-audit" / "cubit_cross_split_near_duplicate_review_queue.csv")
    blocked = unresolved_non_test_ids(queue)
    selected_cubit = [
        row for row in decisions
        if row["selected_after_exact_dedup"].casefold() == "true"
        and row["split"] in {"train", "val"}
        and row["record_id"] not in blocked
    ]
    if any(row["split"] == "test" for row in selected_cubit):
        raise RuntimeError("Locked CUBIT test data entered the v1 selection")

    rows: list[dict[str, str]] = []
    annotation_payloads: dict[str, dict] = {}
    source_root = root / "datasets" / "building-target" / "CUBIT-InSeg" / "CUBIT-InSeg"
    archives: dict[tuple[str, str], zipfile.ZipFile] = {}
    try:
        for decision in selected_cubit:
            source = registry[decision["record_id"]]
            key = (source["split"], source["image_archive"])
            if key not in archives:
                archives[key] = zipfile.ZipFile(root / source["image_archive"])
            label_key = (source["split"], source["annotation_archive"])
            if label_key not in archives:
                archives[label_key] = zipfile.ZipFile(root / source["annotation_archive"])
            image_bytes = archives[key].read(source["image_member"])
            label_bytes = archives[label_key].read(source["annotation_member"])
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                image_format = image.format or "JPEG"
            sample_id = f"cubit_{source['split']}_{source['source_image_id'].lower()}"
            image_path = image_root / f"{sample_id}.jpg"
            image_path.write_bytes(image_bytes)
            annotations, rejected_geometry = read_cubit_annotations(label_bytes, width, height)
            payload = {
                "schema_version": 1,
                "sample_id": sample_id,
                "mapping_version": MAPPING_VERSION,
                "image": {"uri": image_path.relative_to(root).as_posix(), "width": width, "height": height, "format": image_format},
                "source_annotation": {"uri": source["annotation_member"], "sha256": sha256(label_bytes)},
                "annotations": annotations,
                "geometry_qc": {"rejected_geometry_count": rejected_geometry},
            }
            payload_bytes = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
            annotation_path = annotation_root / f"{sample_id}.json"
            annotation_path.write_bytes(payload_bytes)
            annotation_payloads[sample_id] = payload
            rows.append({
                "sample_id": sample_id, "source_dataset": source["source"], "source_version": source["source_version"],
                "source_image_id": source["source_image_id"], "parent_image_id": source["parent_image_id"],
                "group_id": source["group_id"], "split": source["split"], "capture_mode": source["capture_mode"],
                "asset_domain": source["asset_domain"], "annotation_type": source["annotation_type"],
                "native_labels": source["native_labels"], "canonical_labels": source["canonical_labels"],
                "annotation_status": "positive", "mapping_version": MAPPING_VERSION, "license_id": source["license_id"],
                "quality_flags": "cubit_exact_dedup|unresolved_near_duplicates_excluded|test_locked",
                "sha256": sha256(image_bytes), "output_image": image_path.relative_to(root).as_posix(),
                "normalized_annotation": annotation_path.relative_to(root).as_posix(),
                "normalized_annotation_sha256": sha256(payload_bytes), "target_annotation_count": str(len(annotations)),
                "segmentation_annotation_count": str(len(annotations)), "geometry_qc": "passed", "test_locked": "false",
            })
    finally:
        for archive in archives.values():
            archive.close()

    feasibility = read_csv(manifests / "feasibility_master_v0_1.csv")
    for source in feasibility:
        source = dict(source)
        source.update({"test_locked": "false", "quality_flags": f"feasibility_training_only|{source['quality_flags']}"})
        rows.append({field: source.get(field, "") for field in FIELDS})
        payload = json.loads((root / source["normalized_annotation"]).read_text(encoding="utf-8"))
        annotation_payloads[source["sample_id"]] = payload

    rows.sort(key=lambda row: (row["split"], row["source_dataset"], row["sample_id"]))
    if any(row["test_locked"].casefold() == "true" for row in rows):
        raise RuntimeError("v1 manifest contains locked test data")
    if not rows:
        raise RuntimeError("No v1 rows were selected")

    common = {field: "" for field in FIELDS}
    task_paths: dict[str, Path] = {}
    for task, eligibility, geometry in (
        ("detection", "native_box_or_box_derived_from_genuine_mask", "box_xywh"),
        ("segmentation", "genuine_polygon_or_native_mask", "polygon_or_mask"),
        ("classification", "image_multilabel_from_reviewed_annotations", "image_labels"),
        ("florence_product", "rgb_phrase_grounding_input", "image"),
    ):
        view = []
        for row in rows:
            if task == "segmentation" and int(row["segmentation_annotation_count"]) == 0:
                continue
            view.append({**common, **row, "task": task, "eligibility": eligibility, "geometry": geometry})
        task_path = manifests / f"v1_{task}_manifest.csv"
        write_csv(task_path, view)
        task_paths[task] = task_path

    florence_path = manifests / "v1_florence_targets.jsonl"
    write_florence_targets(florence_path, rows, root)
    task_paths["florence_targets"] = florence_path

    master_path = manifests / "v1_master_manifest.csv"
    with master_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    task_paths["master"] = master_path

    counts = {split: dict(Counter(row["source_dataset"] for row in rows if row["split"] == split)) for split in ("train", "val")}
    class_counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    for row in rows:
        class_counts[row["split"]].update(filter(None, row["canonical_labels"].split("|")))
    report = {
        "manifest_version": "v1_0_0",
        "mapping_version": MAPPING_VERSION,
        "rows": len(rows),
        "rows_by_split_and_source": counts,
        "class_image_counts": {split: dict(sorted(values.items())) for split, values in class_counts.items()},
        "annotation_counts": {split: sum(int(row["target_annotation_count"]) for row in rows if row["split"] == split) for split in ("train", "val")},
        "frozen_file_sha256": {name: file_sha256(path) for name, path in task_paths.items()},
        "near_duplicate_queue_records": len(queue),
        "reviewed_duplicate_queue_records": sum(
            (row.get("decision") or "").strip() == "duplicate" for row in queue
        ),
        "unresolved_near_duplicate_candidates": sum(
            (row.get("decision") or "").strip() == "pending_human_review" for row in queue
        ),
        "excluded_reviewed_or_unresolved_non_test_records": len(blocked),
        "locked_test_records_in_manifest": 0,
        "policy": "Reviewed duplicates and unresolved candidates are excluded from train/validation; CUBIT test remains locked and absent.",
        "source_archives_modified": False,
    }
    report_path = root / "artifacts" / "data-audit" / "v1_manifest_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())