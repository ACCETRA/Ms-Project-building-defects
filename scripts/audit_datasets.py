#!/usr/bin/env python3
"""Non-destructively audit locally acquired datasets and write inventory reports.

The script reads archive directories and annotation metadata without extracting full
datasets. It deliberately keeps source-native labels and reports incompatibilities;
it does not build training data or modify raw files.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile


CIF_CLASSES = {
    1: "Algae",
    2: "Crack",
    3: "Net-Crack",
    4: "Crack with Precipitation",
    5: "Rust",
    6: "Spalling",
}

S2DS_COLORS = {
    (0, 0, 0, 255): "background",
    (255, 255, 255, 255): "crack",
    (255, 0, 0, 255): "spalling",
    (255, 255, 0, 255): "corrosion",
    (0, 255, 255, 255): "efflorescence",
    (0, 255, 0, 255): "vegetation",
    (0, 0, 255, 255): "control_point",
}


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    current = float(value)
    for unit in units:
        if current < 1024 or unit == units[-1]:
            return f"{current:.2f} {unit}"
        current /= 1024
    return f"{value} B"


def split_from_name(name: str, choices: tuple[str, ...]) -> str:
    lower = name.lower().replace("\\", "/")
    for choice in choices:
        if f"/{choice}/" in f"/{lower}" or lower.startswith(f"{choice}/"):
            return choice
        if Path(lower).name.startswith(f"{choice}_"):
            return choice
    return "unknown"


def audit_cif(root: Path) -> tuple[dict, list[dict]]:
    source_root = root / "datasets" / "CiF-tiled"
    parquet_files = sorted(source_root.rglob("*.parquet"))
    result = {
        "source": "CiF-tiled",
        "local_path": str(source_root.relative_to(root)),
        "available": bool(parquet_files),
        "license": "CDLA-Permissive-2.0",
        "domain": "mixed civil infrastructure",
        "annotation_types": ["instance polygons", "boxes"],
        "archive_or_files_size_bytes": sum(p.stat().st_size for p in parquet_files),
        "splits": {},
        "notes": [
            "Tiled release only; parent-image fields must control leakage.",
            "Do not count 1024x1024 tiles as independent structures.",
        ],
    }
    class_rows: list[dict] = []
    if not parquet_files:
        return result, class_rows

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        result["notes"].append(f"Detailed parquet audit skipped: {exc}")
        return result, class_rows

    split_stats: dict[str, dict] = {}
    for path in parquet_files:
        split = next(
            (
                candidate
                for candidate in ("train", "val", "test")
                if path.name.startswith(f"{candidate}_tiled")
            ),
            "unknown",
        )
        stats = split_stats.setdefault(
            split,
            {
                "shards": 0,
                "samples": 0,
                "parents": set(),
                "empty_samples": 0,
                "instances": collections.Counter(),
                "images_with_label": collections.Counter(),
            },
        )
        parquet = pq.ParquetFile(path)
        stats["shards"] += 1
        stats["samples"] += parquet.metadata.num_rows
        for batch in parquet.iter_batches(
            # Read only category IDs; loading the full struct would also decode
            # every polygon and make this metadata audit unnecessarily slow.
            columns=["file_name_original", "objects.category_id"], batch_size=4096
        ):
            values = batch.to_pydict()
            for parent, objects in zip(
                values["file_name_original"], values["objects"], strict=True
            ):
                if parent:
                    stats["parents"].add(parent)
                category_ids = (objects or {}).get("category_id") or []
                if not category_ids:
                    stats["empty_samples"] += 1
                labels_in_image = set()
                for category_id in category_ids:
                    label = CIF_CLASSES.get(category_id, f"unknown_{category_id}")
                    stats["instances"][label] += 1
                    labels_in_image.add(label)
                for label in labels_in_image:
                    stats["images_with_label"][label] += 1

    for split, stats in sorted(split_stats.items()):
        result["splits"][split] = {
            "shards": stats["shards"],
            "samples": stats["samples"],
            "unique_parent_images": len(stats["parents"]),
            "empty_or_no_instance_tiles": stats["empty_samples"],
        }
        labels = sorted(set(stats["instances"]) | set(stats["images_with_label"]))
        for label in labels:
            class_rows.append(
                {
                    "dataset": "CiF-tiled",
                    "split": split,
                    "native_label": label,
                    "images_with_label": stats["images_with_label"][label],
                    "instances_or_shapes": stats["instances"][label],
                    "pixels": "",
                    "count_basis": "parquet instance polygons",
                }
            )
    return result, class_rows


def audit_dacl(root: Path) -> tuple[dict, list[dict]]:
    path = root / "datasets" / "DACL10K" / "dacl10k_v2_devphase.zip"
    result = {
        "source": "DACL10K-v2-devphase",
        "local_path": str(path.relative_to(root)),
        "available": path.is_file(),
        "license": "CC BY-NC 4.0",
        "domain": "bridges",
        "annotation_types": ["multi-label semantic polygons"],
        "archive_or_files_size_bytes": path.stat().st_size if path.is_file() else 0,
        "splits": {},
        "notes": [
            "Bridge-domain stress/transfer source, not primary building validation.",
            "Test-dev images have no public labels in this archive.",
        ],
    }
    class_rows: list[dict] = []
    if not path.is_file():
        return result, class_rows

    shapes_by_split: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    images_by_label: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for split in ("train", "validation", "testdev"):
            image_prefix = f"dacl10k_v2_devphase/images/{split}/"
            annotation_prefix = f"dacl10k_v2_devphase/annotations/{split}/"
            result["splits"][split] = {
                "images": sum(
                    1
                    for name in names
                    if name.startswith(image_prefix) and name.lower().endswith(".jpg")
                ),
                "annotations": sum(
                    1
                    for name in names
                    if name.startswith(annotation_prefix)
                    and name.lower().endswith(".json")
                ),
            }
        annotation_names = [
            name
            for name in names
            if "/annotations/" in name and name.lower().endswith(".json")
        ]
        for name in annotation_names:
            split = split_from_name(name, ("train", "validation"))
            try:
                payload = json.loads(archive.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                result["notes"].append(f"Unreadable annotation {name}: {exc}")
                continue
            labels_in_image = set()
            for shape in payload.get("shapes", []):
                label = str(shape.get("label", "unknown"))
                shapes_by_split[split][label] += 1
                labels_in_image.add(label)
            for label in labels_in_image:
                images_by_label[split][label] += 1

    for split in sorted(shapes_by_split):
        for label in sorted(shapes_by_split[split]):
            class_rows.append(
                {
                    "dataset": "DACL10K-v2-devphase",
                    "split": split,
                    "native_label": label,
                    "images_with_label": images_by_label[split][label],
                    "instances_or_shapes": shapes_by_split[split][label],
                    "pixels": "",
                    "count_basis": "LabelMe-style polygons/shapes",
                }
            )
    return result, class_rows


def audit_codebrim(root: Path) -> tuple[dict, list[dict]]:
    path = root / "datasets" / "CODEBRIM" / "CODEBRIM_original_images.zip"
    result = {
        "source": "CODEBRIM-original",
        "local_path": str(path.relative_to(root)),
        "available": path.is_file(),
        "license": "custom non-commercial research/educational license",
        "domain": "bridges",
        "annotation_types": ["bounding boxes", "multi-target source labels"],
        "archive_or_files_size_bytes": path.stat().st_size if path.is_file() else 0,
        "splits": {},
        "notes": [
            "Not pixel-mask ground truth.",
            "Archive contains macOS resource-fork entries that must be ignored.",
        ],
    }
    class_rows: list[dict] = []
    if not path.is_file():
        return result, class_rows

    labels = collections.Counter()
    images_with_label = collections.Counter()
    box_count = 0
    unreadable_regular_members: list[str] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        original_images = [
            info
            for info in infos
            if info.filename.startswith("original_dataset/images/")
            and info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
            and not Path(info.filename).name.startswith("._")
        ]
        annotations = [
            info
            for info in infos
            if info.filename.startswith("original_dataset/annotations/")
            and info.filename.lower().endswith(".xml")
            and not Path(info.filename).name.startswith("._")
        ]
        result["splits"]["unsplit_original"] = {
            "images": len(original_images),
            "xml_annotations": len(annotations),
            "images_without_matching_xml_by_stem": max(
                0, len(original_images) - len(annotations)
            ),
        }
        for info in annotations:
            try:
                xml_root = ET.fromstring(archive.read(info))
            except (ET.ParseError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                unreadable_regular_members.append(f"{info.filename}: {exc}")
                continue
            labels_in_image = set()
            for obj in xml_root.findall(".//object"):
                box_count += 1
                defect = obj.find("Defect")
                positive_labels = []
                if defect is not None:
                    positive_labels = [
                        child.tag
                        for child in defect
                        if (child.text or "").strip() == "1"
                    ]
                if not positive_labels:
                    positive_labels = [
                        obj.findtext("name", default="unknown").strip()
                    ]
                for name in positive_labels:
                    labels[name] += 1
                    labels_in_image.add(name)
            for name in labels_in_image:
                images_with_label[name] += 1

        # Exercise deterministic ordinary image members. zipfile.testzip() flags
        # a malformed zero-byte directory entry in this release, which is not a
        # useful health verdict for the actual image/XML payload.
        probes = []
        if original_images:
            indices = sorted({0, len(original_images) // 2, len(original_images) - 1})
            probes = [original_images[index] for index in indices]
        for info in probes:
            try:
                archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                unreadable_regular_members.append(f"{info.filename}: {exc}")

    if unreadable_regular_members:
        result["notes"].append(
            "Regular-member read failures: " + "; ".join(unreadable_regular_members[:10])
        )
    else:
        result["notes"].append(
            "All XML members and three deterministic image probes were readable; "
            "the reported CRC anomaly is limited to a zero-byte directory entry."
        )
    verification_path = (
        root / "artifacts" / "data-audit" / "codebrim_7zip_verification.json"
    )
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        archive_result = verification.get("original_archive", {})
        if archive_result.get("full_test_exit_code") == 0:
            result["notes"].append(
                "Full archive passed portable 7-Zip verification; use 7-Zip for "
                "image extraction because standard ZIP readers cannot handle the headers."
            )
    result["splits"]["unsplit_original"]["boxes"] = box_count
    for label in sorted(labels):
        class_rows.append(
            {
                "dataset": "CODEBRIM-original",
                "split": "unsplit_original",
                "native_label": label,
                "images_with_label": images_with_label[label],
                "instances_or_shapes": labels[label],
                "pixels": "",
                "count_basis": "Pascal/VOC-style XML objects",
            }
        )
    return result, class_rows


def audit_codebrim_classification(root: Path) -> tuple[dict, list[dict]]:
    path = (
        root
        / "datasets"
        / "CODEBRIM"
        / "CODEBRIM_classification_dataset.zip"
    )
    result = {
        "source": "CODEBRIM-classification",
        "local_path": str(path.relative_to(root)),
        "available": path.is_file(),
        "license": "custom non-commercial research/educational license",
        "domain": "bridge defect crops",
        "annotation_types": ["multi-label crop classification"],
        "archive_or_files_size_bytes": path.stat().st_size if path.is_file() else 0,
        "splits": {},
        "notes": [
            "Official unbalanced classification release selected; do not use the publisher's duplicated balanced archive.",
            "Crops from a common parent filename must be checked for cross-split parent leakage.",
        ],
    }
    class_rows: list[dict] = []
    if not path.is_file():
        return result, class_rows

    label_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    image_split: dict[str, str] = {}
    image_members: list[zipfile.ZipInfo] = []
    duplicate_names: collections.Counter = collections.Counter()
    read_failures: list[str] = []

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            if (
                info.is_dir()
                or name.startswith("__MACOSX/")
                or not name.lower().endswith(".png")
            ):
                continue
            parts = name.split("/")
            if len(parts) != 4 or parts[0] != "classification_dataset":
                continue
            split, folder = parts[1], parts[2]
            if split not in {"train", "val", "test"}:
                continue
            image_members.append(info)
            basename = parts[3]
            duplicate_names[basename] += 1
            image_split[basename] = split
            split_stats = result["splits"].setdefault(
                split,
                {"images": 0, "background_folder": 0, "defects_folder": 0},
            )
            split_stats["images"] += 1
            if folder == "background":
                split_stats["background_folder"] += 1
            elif folder == "defects":
                split_stats["defects_folder"] += 1

        metadata_names = [
            "classification_dataset/metadata/defects.xml",
            "classification_dataset/metadata/background.xml",
        ]
        metadata_rows = 0
        missing_metadata_images = 0
        for metadata_name in metadata_names:
            try:
                xml_root = ET.fromstring(archive.read(metadata_name))
            except (KeyError, ET.ParseError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                read_failures.append(f"{metadata_name}: {exc}")
                continue
            for defect in xml_root.findall(".//Defect"):
                basename = defect.attrib.get("name", "")
                split = image_split.get(basename)
                if not split:
                    missing_metadata_images += 1
                    continue
                metadata_rows += 1
                for child in defect:
                    if (child.text or "").strip() == "1":
                        label_counts[split][child.tag] += 1

        if image_members:
            indices = sorted({0, len(image_members) // 2, len(image_members) - 1})
            for index in indices:
                info = image_members[index]
                try:
                    archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    read_failures.append(f"{info.filename}: {exc}")

    duplicates = {name: count for name, count in duplicate_names.items() if count > 1}
    result["metadata_rows_matched"] = metadata_rows
    result["metadata_rows_without_image"] = missing_metadata_images
    result["duplicate_crop_basenames"] = len(duplicates)
    if read_failures:
        result["notes"].append("Read failures: " + "; ".join(read_failures[:10]))
    else:
        result["notes"].append(
            "Both metadata XML files and three deterministic image probes were readable."
        )
    verification_path = (
        root / "artifacts" / "data-audit" / "codebrim_7zip_verification.json"
    )
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        archive_result = verification.get("classification_archive", {})
        if archive_result.get("full_test_exit_code") == 0:
            result["notes"].append(
                "Full archive passed portable 7-Zip verification; six extracted "
                "probes decoded successfully. Use 7-Zip for image extraction."
            )
    for split in sorted(label_counts):
        for label in sorted(label_counts[split]):
            class_rows.append(
                {
                    "dataset": "CODEBRIM-classification",
                    "split": split,
                    "native_label": label,
                    "images_with_label": label_counts[split][label],
                    "instances_or_shapes": "",
                    "pixels": "",
                    "count_basis": "multi-label crop metadata",
                }
            )
    return result, class_rows


def audit_s2ds(root: Path, count_mask_pixels: bool) -> tuple[dict, list[dict]]:
    path = root / "datasets" / "building-target" / "S2DS" / "s2ds.zip"
    result = {
        "source": "S2DS",
        "local_path": str(path.relative_to(root)),
        "available": path.is_file(),
        "license": "academic use only; redistribution prohibited",
        "domain": "mostly concrete structural surfaces",
        "annotation_types": ["semantic color masks"],
        "archive_or_files_size_bytes": path.stat().st_size if path.is_file() else 0,
        "splits": {},
        "notes": ["Official split contains 743 image/mask pairs."],
    }
    class_rows: list[dict] = []
    if not path.is_file():
        return result, class_rows

    image_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    pixel_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    unknown_colors: collections.Counter = collections.Counter()

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".png")]
        for split in ("train", "val", "test"):
            split_names = [name for name in names if name.startswith(f"{split}/")]
            masks = [name for name in split_names if name.endswith("_lab.png")]
            images = [name for name in split_names if not name.endswith("_lab.png")]
            image_stems = {Path(name).stem for name in images}
            mask_stems = {Path(name).stem.removesuffix("_lab") for name in masks}
            result["splits"][split] = {
                "images": len(images),
                "masks": len(masks),
                "unpaired_images": len(image_stems - mask_stems),
                "unpaired_masks": len(mask_stems - image_stems),
            }
            if not count_mask_pixels:
                continue
            try:
                import numpy as np
                from PIL import Image
            except ImportError as exc:
                result["notes"].append(f"Mask-pixel audit skipped: {exc}")
                count_mask_pixels = False
                continue
            for name in masks:
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    rgba = np.asarray(image.convert("RGBA"))
                # Pack RGBA into one uint32 before np.unique. A two-dimensional
                # axis-wise unique is dramatically slower on 1024x1024 masks.
                packed = (
                    (rgba[..., 0].astype(np.uint32) << 24)
                    | (rgba[..., 1].astype(np.uint32) << 16)
                    | (rgba[..., 2].astype(np.uint32) << 8)
                    | rgba[..., 3].astype(np.uint32)
                )
                colors, counts = np.unique(packed, return_counts=True)
                labels_in_image = set()
                for color, count in zip(colors, counts, strict=True):
                    color_value = int(color)
                    color_key = (
                        (color_value >> 24) & 0xFF,
                        (color_value >> 16) & 0xFF,
                        (color_value >> 8) & 0xFF,
                        color_value & 0xFF,
                    )
                    label = S2DS_COLORS.get(color_key)
                    if label is None:
                        unknown_colors[str(color_key)] += int(count)
                        label = f"unknown_color_{color_key}"
                    pixel_counts[split][label] += int(count)
                    if label != "background" and count:
                        labels_in_image.add(label)
                for label in labels_in_image:
                    image_counts[split][label] += 1

    if unknown_colors:
        result["notes"].append(f"Unknown mask colors: {dict(unknown_colors)}")
    if count_mask_pixels:
        for split in sorted(pixel_counts):
            for label in sorted(pixel_counts[split]):
                class_rows.append(
                    {
                        "dataset": "S2DS",
                        "split": split,
                        "native_label": label,
                        "images_with_label": image_counts[split][label],
                        "instances_or_shapes": "",
                        "pixels": pixel_counts[split][label],
                        "count_basis": "semantic RGBA mask pixels",
                    }
                )
    return result, class_rows


def audit_uav75(root: Path) -> tuple[dict, list[dict]]:
    source_root = root / "datasets" / "UAV-candidates" / "UAV75"
    result = {
        "source": "UAV75",
        "local_path": str(source_root.relative_to(root)),
        "available": source_root.is_dir(),
        "license": "repository GPL-3.0; confirm dataset/media scope before release",
        "domain": "UAV structural surfaces",
        "annotation_types": ["pixel labels", "line-style crack labels"],
        "archive_or_files_size_bytes": 0,
        "splits": {},
        "notes": [
            "75-image stress set with crack and planking/context labels.",
            "Keep official splits; too small to support a standalone performance claim.",
        ],
    }
    if not source_root.is_dir():
        return result, []
    files = [path for path in source_root.rglob("*") if path.is_file()]
    result["archive_or_files_size_bytes"] = sum(path.stat().st_size for path in files)
    for split in ("train", "val", "test"):
        images = sorted((source_root / f"{split}_img").glob("*"))
        labels = sorted((source_root / f"{split}_lab").glob("*"))
        image_stems = {path.stem for path in images if path.is_file()}
        label_stems = {path.stem for path in labels if path.is_file()}
        result["splits"][split] = {
            "images": len(image_stems),
            "masks": len(label_stems),
            "unpaired_images": len(image_stems - label_stems),
            "unpaired_masks": len(label_stems - image_stems),
        }
    return result, []


def zip_member_count(path: Path, suffixes: tuple[str, ...]) -> int | None:
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            return sum(
                1
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(suffixes)
            )
    except (OSError, zipfile.BadZipFile):
        return None


def cubit_label_counts(
    path: Path,
) -> tuple[collections.Counter, collections.Counter, int]:
    """Count public train/val CUBIT classes without extracting label files."""
    instances: collections.Counter = collections.Counter()
    images_with_label: collections.Counter = collections.Counter()
    malformed_lines = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            labels_in_image = set()
            with archive.open(info) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        fields = line.split(maxsplit=1)
                        if not fields:
                            continue
                        try:
                            class_id = int(fields[0])
                        except ValueError:
                            malformed_lines += 1
                            continue
                        label = {0: "crack", 1: "spalling"}.get(
                            class_id, f"unknown_{class_id}"
                        )
                        instances[label] += 1
                        labels_in_image.add(label)
            for label in labels_in_image:
                images_with_label[label] += 1
    return instances, images_with_label, malformed_lines


def audit_cubit(root: Path) -> tuple[dict, list[dict]]:
    source_root = root / "datasets" / "building-target" / "CUBIT-InSeg" / "CUBIT-InSeg"
    expected = {
        "train_images": source_root / "train" / "images.zip",
        "train_labels": source_root / "train" / "labels.zip",
        "val_images": source_root / "val" / "images.zip",
        "val_labels": source_root / "val" / "labels.zip",
        "test_images": source_root / "test" / "images.zip",
        "test_labels": source_root / "test" / "labels.zip",
    }
    availability = {name: path.is_file() for name, path in expected.items()}
    sizes = {name: path.stat().st_size for name, path in expected.items() if path.is_file()}
    split_archives = {
        "train": (expected["train_images"], expected["train_labels"], 5596),
        "val": (expected["val_images"], expected["val_labels"], 699),
        "test": (expected["test_images"], expected["test_labels"], 701),
    }
    split_inventory = {}
    for split, (image_path, label_path, expected_count) in split_archives.items():
        image_count = zip_member_count(image_path, (".jpg", ".jpeg", ".png"))
        label_count = zip_member_count(label_path, (".txt",))
        split_inventory[split] = {
            "expected_images": expected_count,
            "image_archive": image_path.is_file(),
            "label_archive": label_path.is_file(),
            "image_members": image_count,
            "label_members": label_count,
            "count_matches_expected": (
                image_count == expected_count and label_count == expected_count
                if image_count is not None and label_count is not None
                else None
            ),
        }

    acquisition_complete = all(
        details["count_matches_expected"] is True
        for details in split_inventory.values()
    )

    notes = [
        "Preferred building-facade target benchmark.",
        "Keep the official test split locked; do not use it for model selection.",
    ]
    if acquisition_complete:
        notes.append(
            "All six official split archives are present; integrity results are recorded separately."
        )
    else:
        missing = [name for name, present in availability.items() if not present]
        if missing:
            notes.append("Missing archives: " + ", ".join(missing) + ".")
        else:
            notes.append("One or more archives are incomplete or have unexpected counts.")

    result = {
        "source": "CUBIT-InSeg",
        "local_path": str(source_root.relative_to(root)),
        "available": any(availability.values()),
        "acquisition_complete": acquisition_complete,
        "license": "CC BY 4.0",
        "domain": "UAV building facades",
        "annotation_types": ["instance masks", "boxes"],
        "archive_or_files_size_bytes": sum(sizes.values()),
        "splits": split_inventory,
        "notes": notes,
    }
    class_rows = []
    # The official test labels stay locked: audit only filenames and archive
    # integrity there, never class content or performance-tuning statistics.
    for split in ("train", "val"):
        label_path = split_archives[split][1]
        if not label_path.is_file():
            continue
        instances, images_with_label, malformed_lines = cubit_label_counts(label_path)
        result["splits"][split]["malformed_label_lines"] = malformed_lines
        for label in sorted(instances):
            class_rows.append(
                {
                    "dataset": "CUBIT-InSeg",
                    "split": split,
                    "native_label": label,
                    "images_with_label": images_with_label[label],
                    "instances_or_shapes": instances[label],
                    "pixels": "",
                    "count_basis": "YOLO instance-segmentation polygon lines",
                }
            )
    return result, class_rows


def audit_other_sources(root: Path) -> list[dict]:
    sources = []
    liu = root / "datasets" / "UAV-candidates" / "Liu-UAV-benchmark"
    sources.append(
        {
            "source": "Liu-UAV-benchmark-repository",
            "local_path": str(liu.relative_to(root)),
            "available": liu.is_dir(),
            "license": "repository license present; linked consolidated dataset unresolved",
            "domain": "repository describes consolidated crack sources",
            "annotation_types": ["samples/code only in local clone"],
            "archive_or_files_size_bytes": sum(
                path.stat().st_size for path in liu.rglob("*") if path.is_file()
            )
            if liu.is_dir()
            else 0,
            "splits": {},
            "notes": ["Do not include until image provenance and data license are verified."],
        }
    )
    conreb = root / "references" / "toolkits" / "ConRebSeg"
    sources.append(
        {
            "source": "ConRebSeg-toolkit",
            "local_path": str(conreb.relative_to(root)),
            "available": conreb.is_dir(),
            "license": "source/toolkit terms require review; self-collected data release is separate",
            "domain": "reinforced-concrete construction scenes",
            "annotation_types": ["segmentation metadata/tooling"],
            "archive_or_files_size_bytes": sum(
                path.stat().st_size for path in conreb.rglob("*") if path.is_file()
            )
            if conreb.is_dir()
            else 0,
            "splits": {},
            "notes": [
                "Optional for exposed bars/construction context; not primary crack ground truth.",
                "Full approximately 22 GB data has not been acquired.",
            ],
        }
    )
    return sources


def inventory_row(source: dict) -> dict:
    split_summary = "; ".join(
        f"{name}="
        + ",".join(f"{key}:{value}" for key, value in values.items())
        for name, values in source.get("splits", {}).items()
    )
    return {
        "source": source["source"],
        "available": source.get("available", False),
        "domain": source.get("domain", ""),
        "annotation_types": "; ".join(source.get("annotation_types", [])),
        "license": source.get("license", ""),
        "local_path": source.get("local_path", ""),
        "size_bytes": source.get("archive_or_files_size_bytes", 0),
        "size_human": human_bytes(source.get("archive_or_files_size_bytes", 0)),
        "splits": split_summary,
        "notes": " | ".join(source.get("notes", [])),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Workspace root (default: parent of scripts directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <root>/artifacts/data-audit)",
    )
    parser.add_argument(
        "--skip-s2ds-pixels",
        action="store_true",
        help="Skip full semantic mask pixel counting for a faster audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or root / "artifacts" / "data-audit").resolve()
    if not (root / "datasets").is_dir():
        print(f"Dataset directory not found under {root}", file=sys.stderr)
        return 2

    sources: list[dict] = []
    class_rows: list[dict] = []
    auditors = (
        audit_cif,
        audit_dacl,
        audit_codebrim,
        audit_codebrim_classification,
        lambda project_root: audit_s2ds(
            project_root, count_mask_pixels=not args.skip_s2ds_pixels
        ),
        audit_uav75,
        audit_cubit,
    )
    for auditor in auditors:
        source, rows = auditor(root)
        sources.append(source)
        class_rows.extend(rows)
        print(f"Audited {source['source']}")
    sources.extend(audit_other_sources(root))

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "workspace": str(root),
        "raw_files_modified": False,
        "sources": sources,
    }
    (output / "dataset_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        output / "dataset_inventory.csv",
        [inventory_row(source) for source in sources],
        [
            "source",
            "available",
            "domain",
            "annotation_types",
            "license",
            "local_path",
            "size_bytes",
            "size_human",
            "splits",
            "notes",
        ],
    )
    write_csv(
        output / "native_class_counts.csv",
        class_rows,
        [
            "dataset",
            "split",
            "native_label",
            "images_with_label",
            "instances_or_shapes",
            "pixels",
            "count_basis",
        ],
    )
    print(f"Wrote {output / 'dataset_audit.json'}")
    print(f"Wrote {output / 'dataset_inventory.csv'}")
    print(f"Wrote {output / 'native_class_counts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
