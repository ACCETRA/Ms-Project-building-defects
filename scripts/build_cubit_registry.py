#!/usr/bin/env python3
"""Build a metadata-only CUBIT train/validation source registry.

The official test split is deliberately excluded. Image bytes are not extracted;
selected assets will be materialized and content-hashed only in a later curation
step. Because CUBIT filenames do not expose trustworthy building/flight IDs, the
registry records that grouping limitation instead of inventing provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import zipfile


CLASS_NAMES = {0: "crack", 1: "spalling"}
FIELDS = [
    "record_id",
    "source",
    "source_version",
    "split",
    "source_image_id",
    "parent_image_id",
    "group_id",
    "grouping_status",
    "filename_prefix",
    "capture_mode",
    "asset_domain",
    "native_labels",
    "canonical_labels",
    "polygon_instances",
    "annotation_status",
    "annotation_type",
    "image_archive",
    "image_member",
    "annotation_archive",
    "annotation_member",
    "license_id",
    "quality_flags",
]


def zip_members_by_stem(path: Path, suffix: str) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            Path(info.filename).stem.casefold(): info.filename
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.casefold() == suffix
        }


def label_summary(archive: zipfile.ZipFile, member: str) -> tuple[Counter, int]:
    counts: Counter = Counter()
    malformed = 0
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.split(maxsplit=1)
                if not fields:
                    continue
                try:
                    class_id = int(fields[0])
                except ValueError:
                    malformed += 1
                    continue
                counts[CLASS_NAMES.get(class_id, f"unknown_{class_id}")] += 1
    return counts, malformed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> int:
    root = parse_args().root.resolve()
    source_root = (
        root / "datasets" / "building-target" / "CUBIT-InSeg" / "CUBIT-InSeg"
    )
    rows: list[dict] = []
    unmatched: dict[str, dict[str, list[str]]] = {}
    images_with_class: dict[str, Counter] = {}
    instances_by_class: dict[str, Counter] = {}
    malformed_by_split: dict[str, int] = {}

    for split in ("train", "val"):
        image_archive = source_root / split / "images.zip"
        label_archive = source_root / split / "labels.zip"
        images = zip_members_by_stem(image_archive, ".jpg")
        labels = zip_members_by_stem(label_archive, ".txt")
        unmatched[split] = {
            "images_without_label": sorted(images.keys() - labels.keys()),
            "labels_without_image": sorted(labels.keys() - images.keys()),
        }
        image_class_counts: Counter = Counter()
        instance_counts: Counter = Counter()
        malformed_lines = 0
        with zipfile.ZipFile(label_archive) as archive:
            for stem in sorted(images.keys() & labels.keys()):
                counts, malformed = label_summary(archive, labels[stem])
                malformed_lines += malformed
                instance_counts.update(counts)
                image_class_counts.update(counts.keys())
                class_names = sorted(counts)
                display_stem = Path(images[stem]).stem
                prefix = display_stem.split("_", maxsplit=1)[0]
                rows.append(
                    {
                        "record_id": f"cubit_{split}_{stem}",
                        "source": "CUBIT-InSeg",
                        "source_version": "official_2026_release",
                        "split": split,
                        "source_image_id": display_stem,
                        "parent_image_id": display_stem,
                        "group_id": f"cubit:{split}:{stem}",
                        "grouping_status": "source_parent_or_flight_id_unavailable",
                        "filename_prefix": prefix,
                        "capture_mode": "uav_rgb",
                        "asset_domain": "building_facade",
                        "native_labels": "|".join(class_names),
                        "canonical_labels": "|".join(class_names),
                        "polygon_instances": sum(counts.values()),
                        "annotation_status": "publisher_annotation",
                        "annotation_type": "instance_polygon",
                        "image_archive": image_archive.relative_to(root).as_posix(),
                        "image_member": images[stem],
                        "annotation_archive": label_archive.relative_to(root).as_posix(),
                        "annotation_member": labels[stem],
                        "license_id": "CC-BY-4.0",
                        "quality_flags": "official_split|group_metadata_unavailable|dedup_pending",
                    }
                )
        images_with_class[split] = image_class_counts
        instances_by_class[split] = instance_counts
        malformed_by_split[split] = malformed_lines

    output = root / "data" / "manifests" / "cubit_train_val_source_v1.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "registry_version": "cubit_train_val_source_v1",
        "purpose": "source registry; not a balanced or frozen training selection",
        "test_split_included": False,
        "records": len(rows),
        "records_by_split": dict(Counter(row["split"] for row in rows)),
        "images_with_class": {
            split: dict(sorted(counts.items()))
            for split, counts in images_with_class.items()
        },
        "polygon_instances": {
            split: dict(sorted(counts.items()))
            for split, counts in instances_by_class.items()
        },
        "malformed_label_lines": malformed_by_split,
        "unmatched": unmatched,
        "grouping_limitation": (
            "Official filenames do not expose verified building, flight, or capture-session "
            "IDs. Per-image group IDs are placeholders; exact/near-duplicate analysis is "
            "required before a selected manifest is frozen."
        ),
    }
    report_path = (
        root / "artifacts" / "data-audit" / "cubit_train_val_registry_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} records to {output.relative_to(root)}")
    print(f"Wrote {report_path.relative_to(root)}")
    print(json.dumps(report["records_by_split"], indent=2))
    return 0 if not any(value for split in unmatched.values() for value in split.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
