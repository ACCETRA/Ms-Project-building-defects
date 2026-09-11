#!/usr/bin/env python3
"""Normalize feasibility labels and build row-level task-view manifests.

The source annotations remain immutable. Normalized JSON files are derived data;
the versioned CSV views contain their paths, hashes, mapping strengths, and geometry
provenance so every task selection is auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"
NORMALIZED_ROOT = ROOT / "data" / "feasibility_v0" / "normalized"
MANIFEST_ROOT = ROOT / "data" / "manifests"
REPORT_PATH = ROOT / "artifacts" / "data-audit" / "feasibility_task_views_report.json"
REVIEWER_LEDGER = ROOT / "confirmation" / "annotation_qa_confirmation.csv"
MAPPING_VERSION = "BDI-TAX-001@0.1.0-beta"

CIF_CLASSES = {
    1: ("Algae", "unknown_review", "excluded"),
    2: ("Crack", "crack", "direct"),
    3: ("Net-Crack", "crack", "direct"),
    4: ("Crack with Precipitation", "crack", "direct"),
    5: ("Rust", "rust_staining", "direct_visual"),
    6: ("Spalling", "spalling", "direct"),
}

LABELS = {
    "crack": ("crack", "direct"),
    "Crack": ("crack", "direct"),
    "ACrack": ("crack", "direct"),
    "Spalling": ("spalling", "direct"),
    "spalling": ("spalling", "direct"),
    "Rockpocket": ("honeycombing_rock_pocket", "strong"),
    "Cavity": ("unknown_review", "unknown"),
    "Hollowareas": ("unknown_review", "unknown"),
    "Wetspot": ("unknown_review", "unknown"),
    "Weathering": ("unknown_review", "unknown"),
    "WConccor": ("unknown_review", "unknown"),
    "ExposedRebars": ("exposed_rebar", "direct"),
    "Rust": ("rust_staining", "direct_visual"),
    "corrosion": ("rust_staining", "weak"),
    "Efflorescence": ("efflorescence_leaching", "direct"),
    "efflorescence": ("efflorescence_leaching", "direct"),
}

S2DS_COLORS = {
    (255, 255, 255, 255): ("crack", "crack", "direct"),
    (255, 0, 0, 255): ("spalling", "spalling", "direct"),
    (255, 255, 0, 255): ("corrosion", "rust_staining", "weak"),
    (0, 255, 255, 255): ("efflorescence", "efflorescence_leaching", "direct"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_reviewer_decisions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if "annotation_id" not in (rows.fieldnames or []):
            raise ValueError("Reviewer ledger must include an annotation_id column")
        decisions: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows:
            sample_id = (row.get("sample_id") or "").strip()
            annotation_id = (row.get("annotation_id") or "").strip()
            if not sample_id or not annotation_id:
                continue
            key = (sample_id, annotation_id)
            decisions[key] = row
    return decisions


def clamp_box(box: Iterable[float], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = (float(value) for value in box)
    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + box_width, float(width)))
    y2 = max(0.0, min(y + box_height, float(height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate or out-of-bounds box: {list(box)}")
    return [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)]


def polygon_points(flat_or_points: Any, width: int, height: int) -> list[list[float]]:
    if not isinstance(flat_or_points, list):
        raise ValueError("Polygon is not a list")
    if flat_or_points and isinstance(flat_or_points[0], (int, float)):
        if len(flat_or_points) % 2:
            raise ValueError("Polygon has an odd coordinate count")
        points = [flat_or_points[index : index + 2] for index in range(0, len(flat_or_points), 2)]
    else:
        points = flat_or_points
    if len(points) < 3:
        raise ValueError("Polygon requires at least three points")
    normalized: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Polygon point must contain x and y")
        x = max(0.0, min(float(point[0]), float(width)))
        y = max(0.0, min(float(point[1]), float(height)))
        normalized.append([round(x, 4), round(y, 4)])
    return normalized


def box_from_points(points: list[list[float]], width: int, height: int) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return clamp_box([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], width, height)


def normalize_cif(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload["record"]["objects"]
    annotations: list[dict[str, Any]] = []
    count = len(objects.get("category_id", []))
    for index in range(count):
        native, canonical, strength = CIF_CLASSES.get(
            int(objects["category_id"][index]),
            (f"unknown_{objects['category_id'][index]}", "unknown_review", "unknown"),
        )
        if strength == "excluded":
            continue
        polygons: list[list[list[float]]] = []
        rejected_polygons = 0
        segmentations = objects.get("segmentation", [])
        if index < len(segmentations):
            for polygon in segmentations[index] or []:
                try:
                    polygons.append(polygon_points(polygon, width, height))
                except (TypeError, ValueError):
                    rejected_polygons += 1
        annotations.append(
            {
                "annotation_id": str(objects.get("id", [index] * count)[index]),
                "native_label": native,
                "canonical_label": canonical,
                "mapping_strength": strength,
                "annotation_status": "positive",
                "box_xywh": clamp_box(objects["bbox"][index], width, height),
                "box_provenance": "native_box",
                "polygons_xy": polygons,
                "segmentation_provenance": (
                    "native_polygon" if polygons else "unavailable_after_geometry_qc"
                ),
                "rejected_geometry_count": rejected_polygons,
            }
        )
    return annotations


def normalize_dacl(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations: list[dict[str, Any]] = []
    for index, shape in enumerate(payload.get("shapes", [])):
        native = str(shape.get("label", "unknown"))
        if native not in LABELS:
            continue
        canonical, strength = LABELS[native]
        points = polygon_points(shape.get("points"), width, height)
        annotations.append(
            {
                "annotation_id": f"shape-{index}",
                "native_label": native,
                "canonical_label": canonical,
                "mapping_strength": strength,
                "annotation_status": "unknown" if canonical == "unknown_review" else "positive",
                "box_xywh": box_from_points(points, width, height),
                "box_provenance": "derived_from_mask",
                "polygons_xy": [points],
                "segmentation_provenance": "native_polygon",
            }
        )
    return annotations


def bbox_from_mask(mask: np.ndarray, width: int, height: int) -> list[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot derive a box from an empty mask")
    return clamp_box(
        [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
        width,
        height,
    )


def normalize_s2ds(path: Path, width: int, height: int, relative_uri: str) -> list[dict[str, Any]]:
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA"))
    annotations: list[dict[str, Any]] = []
    for color, (native, canonical, strength) in S2DS_COLORS.items():
        mask = np.all(rgba == np.asarray(color, dtype=np.uint8), axis=-1)
        if not mask.any():
            continue
        annotations.append(
            {
                "annotation_id": f"color-{'-'.join(map(str, color))}",
                "native_label": native,
                "canonical_label": canonical,
                "mapping_strength": strength,
                "annotation_status": "positive",
                "box_xywh": bbox_from_mask(mask, width, height),
                "box_provenance": "derived_from_mask",
                "semantic_mask": {
                    "uri": relative_uri,
                    "rgba": list(color),
                    "pixel_count": int(mask.sum()),
                },
                "segmentation_provenance": "native_semantic_mask",
            }
        )
    return annotations


def normalize_uav(path: Path, width: int, height: int, relative_uri: str) -> list[dict[str, Any]]:
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA"))
    crack = np.all(rgba == np.asarray((255, 255, 255, 255), dtype=np.uint8), axis=-1)
    if not crack.any():
        return []
    return [
        {
            "annotation_id": "crack-line-white",
            "native_label": "crack",
            "canonical_label": "crack",
            "mapping_strength": "direct",
            "annotation_status": "positive",
            "box_xywh": bbox_from_mask(crack, width, height),
            "box_provenance": "derived_from_mask",
            "semantic_mask": {
                "uri": relative_uri,
                "rgba": [255, 255, 255, 255],
                "pixel_count": int(crack.sum()),
            },
            "segmentation_provenance": "native_crack_line_mask",
        }
    ]


def normalize(
    row: dict[str, str], reviewer_decisions: dict[tuple[str, str], dict[str, str]]
) -> dict[str, Any]:
    width = int(row["materialized_width"])
    height = int(row["materialized_height"])
    path = ROOT / row["output_annotation"]
    source = row["source_dataset"]
    if source == "CiF-tiled":
        annotations = normalize_cif(path, width, height)
    elif source == "DACL10K-v2-devphase":
        annotations = normalize_dacl(path, width, height)
    elif source == "S2DS":
        annotations = normalize_s2ds(path, width, height, row["output_annotation"])
    elif source == "UAV75":
        annotations = normalize_uav(path, width, height, row["output_annotation"])
    else:
        raise ValueError(f"Unsupported source: {source}")
    if not annotations:
        raise ValueError(f"No target annotations remain after mapping for {row['sample_id']}")
    for annotation in annotations:
        decision = reviewer_decisions.get((row["sample_id"], annotation["annotation_id"]))
        if decision is None:
            continue
        final_label = (decision.get("final_canonical_label") or "").strip()
        review_status = (decision.get("annotation_status") or "").strip()
        if not final_label or "|" in final_label:
            raise ValueError(
                f"Reviewer decision for {row['sample_id']}/{annotation['annotation_id']} "
                "must contain one final_canonical_label"
            )
        if review_status not in {"positive", "verified_negative", "unknown", "not_applicable"}:
            raise ValueError(
                f"Reviewer decision for {row['sample_id']}/{annotation['annotation_id']} "
                f"has invalid annotation_status: {review_status!r}"
            )
        annotation["canonical_label"] = final_label
        annotation["mapping_strength"] = "reviewed"
        annotation["annotation_status"] = review_status
        annotation["review_decision"] = (decision.get("review_decision") or "").strip()
        annotation["geometry_decision"] = (decision.get("geometry_decision") or "").strip()
        annotation["reviewer_id"] = (decision.get("reviewer_id") or "").strip()
        annotation["review_notes"] = (decision.get("review_notes") or "").strip()
    return {
        "schema_version": 1,
        "sample_id": row["sample_id"],
        "mapping_version": MAPPING_VERSION,
        "image": {"uri": row["output_image"], "width": width, "height": height},
        "source_annotation": {"uri": row["output_annotation"], "sha256": row["annotation_sha256"]},
        "annotations": annotations,
        "geometry_qc": {
            "rejected_geometry_count": sum(
                int(item.get("rejected_geometry_count", 0)) for item in annotations
            )
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    global ROOT, SOURCE_MANIFEST, NORMALIZED_ROOT, MANIFEST_ROOT, REPORT_PATH, REVIEWER_LEDGER
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    SOURCE_MANIFEST = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"
    NORMALIZED_ROOT = ROOT / "data" / "feasibility_v0" / "normalized"
    MANIFEST_ROOT = ROOT / "data" / "manifests"
    REPORT_PATH = ROOT / "artifacts" / "data-audit" / "feasibility_task_views_report.json"
    REVIEWER_LEDGER = ROOT / "confirmation" / "annotation_qa_confirmation.csv"

    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 300:
        raise ValueError(f"Expected 300 feasibility records, found {len(source_rows)}")
    reviewer_decisions = load_reviewer_decisions(REVIEWER_LEDGER)
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    master: list[dict[str, Any]] = []
    for row in source_rows:
        payload = normalize(row, reviewer_decisions)
        data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        normalized_path = NORMALIZED_ROOT / f"{row['sample_id']}.json"
        normalized_path.write_bytes(data)
        strengths = sorted({item["mapping_strength"] for item in payload["annotations"]})
        segmentation_count = sum(
            item.get("segmentation_provenance") != "unavailable_after_geometry_qc"
            for item in payload["annotations"]
        )
        master.append(
            {
                **row,
                "mapping_version": MAPPING_VERSION,
                "normalized_annotation": normalized_path.relative_to(ROOT).as_posix(),
                "normalized_annotation_sha256": digest(data),
                "target_annotation_count": len(payload["annotations"]),
                "segmentation_annotation_count": segmentation_count,
                "rejected_geometry_count": payload["geometry_qc"]["rejected_geometry_count"],
                "mapping_strengths": "|".join(strengths),
                "geometry_qc": "passed",
            }
        )

    write_csv(MANIFEST_ROOT / "feasibility_master_v0_1.csv", master)
    common = [
        "sample_id", "source_dataset", "split", "group_id", "output_image",
        "normalized_annotation", "normalized_annotation_sha256", "canonical_labels",
        "mapping_version", "license_id", "target_annotation_count", "mapping_strengths",
        "segmentation_annotation_count", "rejected_geometry_count", "geometry_qc",
    ]
    task_specs = {
        "detection": ("native_box_or_box_derived_from_genuine_mask", "box_xywh"),
        "segmentation": ("genuine_polygon_semantic_or_line_mask", "polygon_or_mask"),
        "classification": ("image_multilabel_from_source_annotations", "image_labels"),
        "florence_product": ("rgb_phrase_grounding_input", "image"),
    }
    counts: dict[str, int] = {}
    for name, (eligibility, geometry) in task_specs.items():
        eligible_master = (
            [row for row in master if int(row["segmentation_annotation_count"]) > 0]
            if name == "segmentation"
            else master
        )
        rows = [
            {**{field: row[field] for field in common}, "task": name, "eligibility": eligibility, "geometry": geometry}
            for row in eligible_master
        ]
        write_csv(MANIFEST_ROOT / f"feasibility_{name}_v0_1.csv", rows)
        counts[name] = len(rows)

    report = {
        "schema_version": 1,
        "manifest_version": "feasibility_v0_1",
        "mapping_version": MAPPING_VERSION,
        "source_samples": len(source_rows),
        "normalized_samples": len(master),
        "target_annotations": sum(int(row["target_annotation_count"]) for row in master),
        "geometry_qc_passed": True,
        "rejected_malformed_geometries": sum(
            int(row["rejected_geometry_count"]) for row in master
        ),
        "task_view_counts": counts,
        "policy": {
            "image_labels_promoted_to_boxes": False,
            "boxes_promoted_to_masks": False,
            "mask_derived_boxes_marked": True,
            "source_annotations_modified": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
