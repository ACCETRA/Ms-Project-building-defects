#!/usr/bin/env python3
"""Inventory and integrity-test the six immutable CUBIT-InSeg ZIP archives.

This script never extracts or changes dataset content. It records byte size,
SHA-256, ZIP member counts, image/label stem pairing, and portable 7-Zip's full
CRC result in JSON and CSV reports.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile


EXPECTED_SPLIT_COUNTS = {"train": 5596, "val": 699, "test": 701}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
LABEL_SUFFIXES = {".txt"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def archive_members(path: Path, kind: str) -> tuple[int, set[str]]:
    suffixes = IMAGE_SUFFIXES if kind == "images" else LABEL_SUFFIXES
    stems: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.suffix.lower() in suffixes:
                stems.add(member.stem.casefold())
    return len(stems), stems


def seven_zip_test(path: Path, executable: Path) -> dict:
    completed = subprocess.run(
        [str(executable), "t", str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout + completed.stderr
    result = {
        "exit_code": completed.returncode,
        "everything_ok": completed.returncode == 0 and "Everything is Ok" in output,
    }
    for key in ("Folders", "Files", "Size", "Compressed"):
        match = re.search(rf"^{key}:\s+(\d+)$", output, flags=re.MULTILINE)
        if match:
            result[key.lower()] = int(match.group(1))
    if not result["everything_ok"]:
        result["diagnostic_tail"] = output[-2000:]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "data-audit",
    )
    parser.add_argument(
        "--seven-zip",
        type=Path,
        default=project_root / "vendor" / "7zip-portable" / "x64" / "7za.exe",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    dataset_root = (
        root / "datasets" / "building-target" / "CUBIT-InSeg" / "CUBIT-InSeg"
    )
    seven_zip = args.seven_zip.resolve()
    if not seven_zip.is_file():
        print(f"7-Zip executable not found: {seven_zip}", file=sys.stderr)
        return 2

    archive_rows: list[dict] = []
    stem_sets: dict[tuple[str, str], set[str]] = {}
    missing: list[str] = []
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            path = dataset_root / split / f"{kind}.zip"
            relative_path = str(path.relative_to(root))
            if not path.is_file():
                missing.append(relative_path)
                continue
            print(f"Inventorying {relative_path}", flush=True)
            member_count, stems = archive_members(path, kind)
            stem_sets[(split, kind)] = stems
            archive_rows.append(
                {
                    "split": split,
                    "kind": kind,
                    "path": relative_path,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "member_count": member_count,
                    "expected_count": EXPECTED_SPLIT_COUNTS[split],
                    "count_matches_expected": member_count
                    == EXPECTED_SPLIT_COUNTS[split],
                    "integrity": seven_zip_test(path, seven_zip),
                }
            )

    split_rows = []
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        image_stems = stem_sets.get((split, "images"), set())
        label_stems = stem_sets.get((split, "labels"), set())
        split_rows.append(
            {
                "split": split,
                "expected_count": expected_count,
                "image_count": len(image_stems) if image_stems else None,
                "label_count": len(label_stems) if label_stems else None,
                "images_without_label": sorted(image_stems - label_stems),
                "labels_without_image": sorted(label_stems - image_stems),
                "paired": bool(image_stems)
                and image_stems == label_stems
                and len(image_stems) == expected_count,
            }
        )

    complete = (
        not missing
        and len(archive_rows) == 6
        and all(row["count_matches_expected"] for row in archive_rows)
        and all(row["integrity"]["everything_ok"] for row in archive_rows)
        and all(row["paired"] for row in split_rows)
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CUBIT-InSeg",
        "source_archives_modified": False,
        "test_split_extracted": False,
        "acquisition_complete_and_verified": complete,
        "missing_archives": missing,
        "archives": archive_rows,
        "splits": split_rows,
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cubit_archive_inventory.json"
    csv_path = output / "cubit_archive_inventory.csv"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "kind",
            "path",
            "size_bytes",
            "sha256",
            "member_count",
            "expected_count",
            "count_matches_expected",
            "integrity_exit_code",
            "integrity_ok",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in archive_rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames[:-2]},
                    "integrity_exit_code": row["integrity"]["exit_code"],
                    "integrity_ok": row["integrity"]["everything_ok"],
                }
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"CUBIT acquisition complete and verified: {complete}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
