#!/usr/bin/env python3
"""Build a deterministic, training-only manifest for the first hardware smoke test.

This script does not extract images and does not alter source datasets. The output
is a 300-sample manifest spanning CiF, S2DS, UAV75, and DACL10K. It is intended to
test loading, transforms, inference, and VRAM use; it is not an accuracy benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random
from typing import Iterable
import zipfile


SEED = 20260908

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

CANONICAL = {
    "Crack": "crack",
    "Net-Crack": "crack",
    "Crack with Precipitation": "crack",
    "ACrack": "crack",
    "crack": "crack",
    "Spalling": "spalling",
    "spalling": "spalling",
    "Rockpocket": "honeycombing_rock_pocket",
    "ExposedRebars": "exposed_rebar",
    "Rust": "rust_staining",
    "corrosion": "rust_staining",
    "Efflorescence": "efflorescence_leaching",
    "efflorescence": "efflorescence_leaching",
}

FIELDS = [
    "sample_id",
    "source_dataset",
    "source_version",
    "source_path",
    "source_image_id",
    "parent_image_id",
    "group_id",
    "split",
    "capture_mode",
    "asset_domain",
    "annotation_type",
    "native_labels",
    "canonical_labels",
    "annotation_status",
    "mapping_version",
    "license_id",
    "quality_flags",
    "selection_reason",
    "sha256",
    "locator_type",
    "image_locator",
    "annotation_locator",
    "parquet_row_index",
    "output_image",
    "output_annotation",
]


@dataclass(frozen=True)
class Candidate:
    key: str
    source_image_id: str
    group_id: str
    native_labels: tuple[str, ...]
    image_locator: str
    annotation_locator: str
    locator_type: str
    parquet_row_index: int | None = None


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def canonical_labels(native_labels: Iterable[str]) -> list[str]:
    return sorted({CANONICAL[label] for label in native_labels if label in CANONICAL})


def quota_select(
    candidates: list[Candidate], quotas: list[tuple[str, int]], total: int
) -> list[tuple[Candidate, str]]:
    """Select unique candidates/groups, satisfying rare-label quotas first."""
    ranked = sorted(candidates, key=lambda item: stable_rank(item.key))
    selected: list[tuple[Candidate, str]] = []
    used_keys: set[str] = set()
    used_groups: set[str] = set()

    for label, count in quotas:
        matches = [candidate for candidate in ranked if label in candidate.native_labels]
        added = 0
        for candidate in matches:
            if candidate.key in used_keys or candidate.group_id in used_groups:
                continue
            selected.append((candidate, f"quota:{label}"))
            used_keys.add(candidate.key)
            used_groups.add(candidate.group_id)
            added += 1
            if added == count:
                break
        if added != count:
            raise RuntimeError(
                f"Could select only {added}/{count} unique candidates for {label}"
            )

    for candidate in ranked:
        if len(selected) == total:
            break
        if candidate.key in used_keys or candidate.group_id in used_groups:
            continue
        selected.append((candidate, "deterministic_fill"))
        used_keys.add(candidate.key)
        used_groups.add(candidate.group_id)

    if len(selected) != total:
        raise RuntimeError(f"Could select only {len(selected)}/{total} samples")
    return selected


def cif_candidates(root: Path) -> list[Candidate]:
    import pyarrow.parquet as pq

    shards = sorted((root / "datasets" / "CiF-tiled" / "data").glob("train_tiled-*.parquet"))
    if len(shards) < 12:
        raise FileNotFoundError("Expected the complete CiF train shard set")

    # A smoke test does not need all 84 shards. Twelve evenly distributed shards
    # retain source diversity while keeping later materialization I/O reasonable.
    shard_indices = sorted({round(index * (len(shards) - 1) / 11) for index in range(12)})
    selected_shards = [shards[index] for index in shard_indices]
    candidates: list[Candidate] = []
    for path in selected_shards:
        table = pq.read_table(
            path,
            columns=["file_name", "file_name_original", "objects.category_id"],
        )
        values = table.to_pydict()
        relative = path.relative_to(root).as_posix()
        for row_index, (file_name, parent, category_ids) in enumerate(
            zip(
                values["file_name"],
                values["file_name_original"],
                values["category_id"],
                strict=True,
            )
        ):
            native = tuple(
                sorted(
                    {
                        CIF_CLASSES.get(category_id, f"unknown_{category_id}")
                        for category_id in (category_ids or [])
                    }
                )
            )
            if not canonical_labels(native):
                continue
            key = f"{relative}#{row_index}"
            candidates.append(
                Candidate(
                    key=key,
                    source_image_id=file_name,
                    group_id=parent or file_name,
                    native_labels=native,
                    image_locator=relative,
                    annotation_locator=relative,
                    locator_type="parquet_row",
                    parquet_row_index=row_index,
                )
            )
    return candidates


def s2ds_candidates(root: Path) -> list[Candidate]:
    from PIL import Image

    archive_path = root / "datasets" / "building-target" / "S2DS" / "s2ds.zip"
    archive_relative = archive_path.relative_to(root).as_posix()
    candidates: list[Candidate] = []
    with zipfile.ZipFile(archive_path) as archive:
        mask_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith("_lab.png")
        )
        for mask_name in mask_names:
            image_name = mask_name.replace("_lab.png", ".png")
            with Image.open(io.BytesIO(archive.read(mask_name))) as image:
                rgba = image.convert("RGBA")
                color_counts = rgba.getcolors(maxcolors=rgba.width * rgba.height)
            if color_counts is None:
                raise RuntimeError(f"Unexpected high-color label mask: {mask_name}")
            colors = {tuple(color) for _, color in color_counts}
            native = tuple(
                sorted(
                    {
                        S2DS_COLORS[color]
                        for color in colors
                        if color in S2DS_COLORS and S2DS_COLORS[color] not in {"background", "vegetation", "control_point"}
                    }
                )
            )
            if not native:
                continue
            sample_stem = Path(image_name).stem
            candidates.append(
                Candidate(
                    key=f"{archive_relative}::{image_name}",
                    source_image_id=sample_stem,
                    group_id=f"S2DS:{sample_stem}",
                    native_labels=native,
                    image_locator=f"{archive_relative}::{image_name}",
                    annotation_locator=f"{archive_relative}::{mask_name}",
                    locator_type="zip_member",
                )
            )
    return candidates


def dacl_candidates(root: Path) -> list[Candidate]:
    archive_path = root / "datasets" / "DACL10K" / "dacl10k_v2_devphase.zip"
    archive_relative = archive_path.relative_to(root).as_posix()
    prefix = "dacl10k_v2_devphase/annotations/train/"
    candidates: list[Candidate] = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        annotation_names = sorted(
            name for name in archive_names if name.startswith(prefix) and name.endswith(".json")
        )
        for annotation_name in annotation_names:
            payload = json.loads(archive.read(annotation_name))
            native = tuple(sorted({str(shape.get("label", "unknown")) for shape in payload.get("shapes", [])}))
            if not canonical_labels(native):
                continue
            stem = Path(annotation_name).stem
            image_name = f"dacl10k_v2_devphase/images/train/{stem}.jpg"
            if image_name not in archive_names:
                continue
            candidates.append(
                Candidate(
                    key=f"{archive_relative}::{image_name}",
                    source_image_id=stem,
                    group_id=f"DACL10K:{stem}",
                    native_labels=native,
                    image_locator=f"{archive_relative}::{image_name}",
                    annotation_locator=f"{archive_relative}::{annotation_name}",
                    locator_type="zip_member",
                )
            )
    return candidates


def uav_candidates(root: Path) -> list[Candidate]:
    source_root = root / "datasets" / "UAV-candidates" / "UAV75"
    candidates: list[Candidate] = []
    for image_path in sorted((source_root / "train_img").glob("*.jpg")):
        annotation_path = source_root / "train_lab" / f"{image_path.stem}.png"
        if not annotation_path.is_file():
            continue
        candidates.append(
            Candidate(
                key=image_path.relative_to(root).as_posix(),
                source_image_id=image_path.stem,
                group_id=f"UAV75:{image_path.stem}",
                native_labels=("crack",),
                image_locator=image_path.relative_to(root).as_posix(),
                annotation_locator=annotation_path.relative_to(root).as_posix(),
                locator_type="file",
            )
        )
    return candidates


def make_rows(root: Path) -> list[dict[str, str | int]]:
    plans = [
        (
            "cif",
            "CiF-tiled",
            "tiled-1024-public",
            "mixed civil infrastructure",
            "UAV",
            "CDLA-Permissive-2.0",
            "instance polygons and boxes",
            cif_candidates(root),
            [
                ("Crack with Precipitation", 10),
                ("Net-Crack", 15),
                ("Rust", 20),
                ("Spalling", 20),
                ("Crack", 55),
            ],
            120,
        ),
        (
            "s2ds",
            "S2DS",
            "official-archive",
            "concrete structures",
            "unknown",
            "GPL-3.0 repository; verify dataset terms before distribution",
            "semantic mask",
            s2ds_candidates(root),
            [("efflorescence", 12), ("corrosion", 12), ("spalling", 16), ("crack", 40)],
            80,
        ),
        (
            "uav75",
            "UAV75",
            "repository-snapshot-2026-09-07",
            "UAV concrete crack imagery",
            "UAV",
            "GPL-3.0 repository; verify dataset terms before distribution",
            "crack line mask",
            uav_candidates(root),
            [("crack", 40)],
            40,
        ),
        (
            "dacl",
            "DACL10K-v2-devphase",
            "v2-devphase",
            "bridges (transfer stress source)",
            "unknown",
            "CC BY-NC 4.0",
            "multi-label polygons",
            dacl_candidates(root),
            [
                ("Rockpocket", 8),
                ("ExposedRebars", 8),
                ("Efflorescence", 5),
                ("Rust", 6),
                ("Spalling", 10),
                ("ACrack", 5),
                ("Crack", 18),
            ],
            60,
        ),
    ]

    rows: list[dict[str, str | int]] = []
    for (
        prefix,
        source,
        source_version,
        domain,
        capture_mode,
        license_name,
        annotation_type,
        candidates,
        quotas,
        total,
    ) in plans:
        chosen = quota_select(candidates, quotas, total)
        for index, (candidate, reason) in enumerate(chosen, start=1):
            image_suffix = Path(candidate.image_locator.split("::")[-1]).suffix.lower()
            annotation_suffix = Path(candidate.annotation_locator.split("::")[-1]).suffix.lower()
            if candidate.locator_type == "parquet_row":
                image_suffix = ".jpg"
                annotation_suffix = ".json"
            sample_id = f"{prefix}_{index:04d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source,
                    "source_version": source_version,
                    "source_path": candidate.image_locator,
                    "source_image_id": candidate.source_image_id,
                    "parent_image_id": candidate.group_id,
                    "group_id": candidate.group_id,
                    "split": "train",
                    "capture_mode": capture_mode,
                    "asset_domain": domain,
                    "annotation_type": annotation_type,
                    "native_labels": "|".join(candidate.native_labels),
                    "canonical_labels": "|".join(canonical_labels(candidate.native_labels)),
                    "annotation_status": "positive",
                    "mapping_version": "BDI-TAX-001@0.1.0-draft",
                    "license_id": license_name,
                    "quality_flags": "",
                    "selection_reason": reason,
                    "sha256": "",
                    "locator_type": candidate.locator_type,
                    "image_locator": candidate.image_locator,
                    "annotation_locator": candidate.annotation_locator,
                    "parquet_row_index": "" if candidate.parquet_row_index is None else candidate.parquet_row_index,
                    "output_image": f"data/feasibility_v0/images/{prefix}/{sample_id}{image_suffix}",
                    "output_annotation": f"data/feasibility_v0/annotations/{prefix}/{sample_id}{annotation_suffix}",
                }
            )
    return rows


def write_outputs(root: Path, rows: list[dict[str, str | int]]) -> None:
    output_path = root / "data" / "manifests" / "feasibility_v0.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_source: dict[str, int] = {}
    native_counts: dict[str, dict[str, int]] = {}
    canonical_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row["source_dataset"])
        by_source[source] = by_source.get(source, 0) + 1
        native_counts.setdefault(source, {})
        canonical_counts.setdefault(source, {})
        for label in str(row["native_labels"]).split("|"):
            native_counts[source][label] = native_counts[source].get(label, 0) + 1
        for label in str(row["canonical_labels"]).split("|"):
            canonical_counts[source][label] = canonical_counts[source].get(label, 0) + 1

    report = {
        "manifest_version": "feasibility_v0",
        "seed": SEED,
        "purpose": "hardware and data-pipeline smoke test only; not an accuracy benchmark",
        "policy": {
            "partitions_used": ["train"],
            "held_out_partitions_used": False,
            "cif_group_rule": "one selected tile per file_name_original",
        },
        "total_samples": len(rows),
        "samples_by_source": by_source,
        "images_with_native_label": native_counts,
        "images_with_canonical_label": canonical_counts,
    }
    report_path = root / "artifacts" / "data-audit" / "feasibility_manifest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} samples to {output_path.relative_to(root)}")
    print(json.dumps(by_source, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    random.seed(SEED)
    rows = make_rows(root)
    write_outputs(root, rows)


if __name__ == "__main__":
    main()
