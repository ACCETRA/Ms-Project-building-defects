#!/usr/bin/env python3
"""Build a metadata-only source registry for CODEBRIM classification crops.

The oversized CODEBRIM ZIP can expose its directory and XML metadata through
Python, but image extraction must use the verified portable 7-Zip executable.
This script records every official crop without extracting image payloads.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


CANONICAL = {
    "Background": "no_visible_target_defect",
    "Crack": "crack",
    "Spallation": "spalling",
    "Efflorescence": "efflorescence_leaching",
    "ExposedBars": "exposed_rebar",
    "CorrosionStain": "rust_staining",
}

FIELDS = [
    "record_id",
    "source",
    "split",
    "parent_id",
    "crop_name",
    "folder",
    "native_labels",
    "canonical_labels",
    "annotation_status",
    "annotation_type",
    "archive_path",
    "image_member",
    "metadata_member",
    "license",
]


def parent_from_crop(filename: str) -> str:
    return re.sub(r"_crop_\d+$", "", Path(filename).stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    archive_path = root / "datasets" / "CODEBRIM" / "CODEBRIM_classification_dataset.zip"
    archive_relative = archive_path.relative_to(root).as_posix()

    with zipfile.ZipFile(archive_path) as archive:
        members: dict[str, tuple[str, str, str]] = {}
        for name in archive.namelist():
            parts = name.split("/")
            if (
                len(parts) == 4
                and parts[0] == "classification_dataset"
                and parts[1] in {"train", "val", "test"}
                and parts[2] in {"background", "defects"}
                and name.lower().endswith(".png")
            ):
                members[parts[3]] = (parts[1], parts[2], name)

        labels_by_crop: dict[str, tuple[tuple[str, ...], str]] = {}
        metadata_without_image: list[str] = []
        for metadata_member in (
            "classification_dataset/metadata/background.xml",
            "classification_dataset/metadata/defects.xml",
        ):
            xml_root = ET.fromstring(archive.read(metadata_member))
            for defect in xml_root.findall(".//Defect"):
                crop_name = defect.attrib.get("name", "")
                labels = tuple(
                    child.tag for child in defect if (child.text or "").strip() == "1"
                )
                if crop_name not in members:
                    metadata_without_image.append(crop_name)
                    continue
                labels_by_crop[crop_name] = (labels, metadata_member)

    rows: list[dict[str, str]] = []
    parents_by_split: dict[str, set[str]] = defaultdict(set)
    native_counts: dict[str, Counter[str]] = defaultdict(Counter)
    images_without_metadata: list[str] = []
    for crop_name, (split, folder, member) in sorted(members.items(), key=lambda item: (item[1][0], item[0])):
        if crop_name not in labels_by_crop:
            images_without_metadata.append(crop_name)
            native_labels: tuple[str, ...] = ()
            metadata_member = ""
        else:
            native_labels, metadata_member = labels_by_crop[crop_name]
        canonical_labels = sorted(
            {CANONICAL[label] for label in native_labels if label in CANONICAL}
        )
        parent_id = parent_from_crop(crop_name)
        parents_by_split[split].add(parent_id)
        native_counts[split].update(native_labels)
        rows.append(
            {
                "record_id": f"codebrim_cls_{split}_{Path(crop_name).stem}",
                "source": "CODEBRIM-classification",
                "split": split,
                "parent_id": parent_id,
                "crop_name": crop_name,
                "folder": folder,
                "native_labels": "|".join(native_labels),
                "canonical_labels": "|".join(canonical_labels),
                "annotation_status": "verified_negative" if native_labels == ("Background",) else "positive",
                "annotation_type": "multi-label crop classification",
                "archive_path": archive_relative,
                "image_member": member,
                "metadata_member": metadata_member,
                "license": "custom non-commercial research/educational license",
            }
        )

    parent_splits: dict[str, set[str]] = defaultdict(set)
    for split, parents in parents_by_split.items():
        for parent in parents:
            parent_splits[parent].add(split)
    cross_split = {
        parent: sorted(splits)
        for parent, splits in parent_splits.items()
        if len(splits) > 1
    }

    registry_path = root / "data" / "manifests" / "codebrim_classification_source_v1.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "registry_version": "codebrim_classification_source_v1",
        "purpose": "source registry; not a balanced training selection",
        "image_records": len(rows),
        "records_by_split": dict(Counter(row["split"] for row in rows)),
        "parent_images_by_split": {
            split: len(parents) for split, parents in sorted(parents_by_split.items())
        },
        "cross_split_parent_count": len(cross_split),
        "cross_split_parent_examples": dict(list(sorted(cross_split.items()))[:20]),
        "images_without_metadata": images_without_metadata,
        "metadata_rows_without_image_count": len(metadata_without_image),
        "metadata_rows_without_image": sorted(metadata_without_image),
        "images_with_native_label": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(native_counts.items())
        },
        "extraction_requirement": "vendor/7zip-portable/x64/7za.exe",
    }
    report_path = root / "artifacts" / "data-audit" / "codebrim_classification_registry_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} records to {registry_path.relative_to(root)}")
    print(json.dumps(report["records_by_split"], indent=2))
    print(f"Cross-split parent groups: {len(cross_split)}")


if __name__ == "__main__":
    main()
