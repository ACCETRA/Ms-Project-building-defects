#!/usr/bin/env python3
"""Materialize and validate the training-only feasibility manifest.

Raw sources remain untouched. Selected files are copied/extracted into
data/feasibility_v0 and a verification CSV/JSON is written alongside the audit
artifacts. Existing derived files are reused only when their bytes match.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile


EXPECTED_SAMPLES = 300


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_output(root: Path, relative: str) -> Path:
    output_root = (root / "data" / "feasibility_v0").resolve()
    path = (root / relative).resolve()
    if output_root != path and output_root not in path.parents:
        raise ValueError(f"Output escapes feasibility directory: {relative}")
    return path


def write_derived(path: Path, data: bytes, overwrite: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing == data:
            return "reused_identical"
        if not overwrite:
            raise FileExistsError(
                f"Derived output differs: {path}. Re-run with --overwrite only if intentional."
            )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return "written"


def image_metadata(data: bytes) -> dict[str, Any]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format or "unknown",
            "mode": image.mode,
        }


def validate_annotation(data: bytes, suffix: str) -> None:
    if suffix.lower() == ".json":
        json.loads(data)
        return
    if suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        image_metadata(data)
        return
    raise ValueError(f"Unsupported annotation format: {suffix}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} samples, found {len(rows)}")
    if {row["split"] for row in rows} != {"train"}:
        raise ValueError("Feasibility manifest must contain training partitions only")
    for field in ("sample_id", "group_id"):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field} values in manifest")
    for row in rows:
        locators = f"{row['image_locator']} {row['annotation_locator']}".lower()
        if "/test/" in locators or "\\test\\" in locators:
            raise ValueError(f"Held-out test path found in {row['sample_id']}")
    return rows


def serialize_cif_annotation(record: dict[str, Any]) -> bytes:
    image_field = record.pop("image", None) or {}
    payload = {
        "schema": "cif_native_row_v1",
        "source_image_path": image_field.get("path"),
        "record": record,
    }
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def materialize_parquet_rows(
    root: Path,
    rows: list[dict[str, str]],
    overwrite: bool,
    verification: list[dict[str, Any]],
) -> None:
    import pyarrow.parquet as pq

    rows_by_path: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["locator_type"] == "parquet_row":
            rows_by_path.setdefault(row["image_locator"], []).append(row)

    columns = [
        "image_id",
        "image",
        "file_name",
        "width",
        "height",
        "tile_row",
        "tile_col",
        "file_name_original",
        "width_original",
        "height_original",
        "objects",
    ]
    for relative, selected_rows in sorted(rows_by_path.items()):
        source_path = (root / relative).resolve()
        parquet = pq.ParquetFile(source_path)
        wanted = {int(row["parquet_row_index"]): row for row in selected_rows}
        group_start = 0
        for group_index in range(parquet.num_row_groups):
            group_size = parquet.metadata.row_group(group_index).num_rows
            group_end = group_start + group_size
            in_group = {
                index: row
                for index, row in wanted.items()
                if group_start <= index < group_end
            }
            if not in_group:
                group_start = group_end
                continue
            table = parquet.read_row_group(group_index, columns=columns)
            records = table.to_pylist()
            for global_index, manifest_row in in_group.items():
                record = records[global_index - group_start]
                image_field = record.get("image") or {}
                image_bytes = image_field.get("bytes")
                if not image_bytes:
                    raise ValueError(f"Missing CiF image bytes at {relative} row {global_index}")
                metadata = image_metadata(image_bytes)
                annotation_bytes = serialize_cif_annotation(record)
                add_outputs(
                    root,
                    manifest_row,
                    image_bytes,
                    annotation_bytes,
                    metadata,
                    overwrite,
                    verification,
                )
                wanted.pop(global_index)
            group_start = group_end
        if wanted:
            raise IndexError(f"Rows not found in {relative}: {sorted(wanted)}")


def add_outputs(
    root: Path,
    row: dict[str, str],
    image_bytes: bytes,
    annotation_bytes: bytes,
    metadata: dict[str, Any],
    overwrite: bool,
    verification: list[dict[str, Any]],
) -> None:
    image_output = safe_output(root, row["output_image"])
    annotation_output = safe_output(root, row["output_annotation"])
    image_status = write_derived(image_output, image_bytes, overwrite)
    annotation_status = write_derived(annotation_output, annotation_bytes, overwrite)
    validate_annotation(annotation_bytes, annotation_output.suffix)
    verification.append(
        {
            "sample_id": row["sample_id"],
            "source": row["source_dataset"],
            "image_path": image_output.relative_to(root).as_posix(),
            "annotation_path": annotation_output.relative_to(root).as_posix(),
            "image_bytes": len(image_bytes),
            "annotation_bytes": len(annotation_bytes),
            "image_sha256": sha256(image_bytes),
            "annotation_sha256": sha256(annotation_bytes),
            "width": metadata["width"],
            "height": metadata["height"],
            "image_format": metadata["format"],
            "image_mode": metadata["mode"],
            "image_status": image_status,
            "annotation_status": annotation_status,
            "readable": True,
        }
    )


def materialize_other_rows(
    root: Path,
    rows: list[dict[str, str]],
    overwrite: bool,
    verification: list[dict[str, Any]],
) -> None:
    with ExitStack() as stack:
        archives: dict[Path, zipfile.ZipFile] = {}

        def read_locator(locator: str, locator_type: str) -> bytes:
            if locator_type == "file":
                return (root / locator).read_bytes()
            if locator_type != "zip_member":
                raise ValueError(f"Unsupported locator type: {locator_type}")
            archive_relative, member = locator.split("::", 1)
            archive_path = (root / archive_relative).resolve()
            if archive_path not in archives:
                archives[archive_path] = stack.enter_context(zipfile.ZipFile(archive_path))
            return archives[archive_path].read(member)

        for row in rows:
            if row["locator_type"] == "parquet_row":
                continue
            image_bytes = read_locator(row["image_locator"], row["locator_type"])
            annotation_bytes = read_locator(row["annotation_locator"], row["locator_type"])
            metadata = image_metadata(image_bytes)
            add_outputs(
                root,
                row,
                image_bytes,
                annotation_bytes,
                metadata,
                overwrite,
                verification,
            )


def write_verification(
    root: Path,
    manifest_rows: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> None:
    rows.sort(key=lambda row: row["sample_id"])
    csv_path = root / "artifacts" / "data-audit" / "feasibility_materialization.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verification_by_id = {row["sample_id"]: row for row in rows}
    materialized_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        verified = verification_by_id[manifest_row["sample_id"]]
        combined: dict[str, Any] = dict(manifest_row)
        combined["sha256"] = verified["image_sha256"]
        combined.update(
            {
                "annotation_sha256": verified["annotation_sha256"],
                "materialized_width": verified["width"],
                "materialized_height": verified["height"],
                "materialized_image_format": verified["image_format"],
                "materialized_readable": verified["readable"],
            }
        )
        materialized_rows.append(combined)
    materialized_path = (
        root / "data" / "manifests" / "feasibility_v0_materialized.csv"
    )
    with materialized_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized_rows[0]))
        writer.writeheader()
        writer.writerows(materialized_rows)

    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    report = {
        "manifest_version": "feasibility_v0",
        "samples_verified": len(rows),
        "all_readable": all(row["readable"] for row in rows),
        "samples_by_source": by_source,
        "total_image_bytes": sum(row["image_bytes"] for row in rows),
        "total_annotation_bytes": sum(row["annotation_bytes"] for row in rows),
        "materialized_manifest": materialized_path.relative_to(root).as_posix(),
        "verification_csv": csv_path.relative_to(root).as_posix(),
    }
    report_path = root / "artifacts" / "data-audit" / "feasibility_materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only differing files inside data/feasibility_v0.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "data" / "manifests" / "feasibility_v0.csv"
    rows = read_manifest(manifest)
    verification: list[dict[str, Any]] = []
    materialize_parquet_rows(root, rows, args.overwrite, verification)
    materialize_other_rows(root, rows, args.overwrite, verification)
    if len(verification) != len(rows):
        raise RuntimeError(f"Verified {len(verification)}/{len(rows)} samples")
    write_verification(root, rows, verification)


if __name__ == "__main__":
    main()
