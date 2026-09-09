#!/usr/bin/env python3
"""Audit exact and perceptual CUBIT image duplicates without extraction.

The locked test images participate only in deterministic hash comparison so a
train/test leak can be found. Test labels are never opened. Perceptual matches
are review candidates, not automatic deletion decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageOps


HASH_FIELDS = [
    "record_id",
    "split",
    "image_archive",
    "image_member",
    "source_image_id",
    "width",
    "height",
    "encoded_size_bytes",
    "sha256",
    "phash64",
]
PAIR_FIELDS = [
    "match_type",
    "phash_hamming_distance",
    "cross_split",
    "left_record_id",
    "left_split",
    "left_member",
    "right_record_id",
    "right_split",
    "right_member",
]


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(math.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


DCT32 = dct_matrix(32)


def perceptual_hash(image: Image.Image) -> int:
    grayscale = ImageOps.exif_transpose(image).convert("L")
    resized = grayscale.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    transformed = DCT32 @ pixels @ DCT32.T
    coefficients = transformed[:8, :8].reshape(-1)[1:]
    median = float(np.median(coefficients))
    value = 0
    for coefficient in coefficients:
        value = (value << 1) | int(coefficient > median)
    return value


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict[int, tuple]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            node_value, indices, children = node
            distance = self.distance(value, node_value)
            if distance == 0:
                indices.append(index)
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, [index], {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node_value, indices, children = stack.pop()
            distance = self.distance(value, node_value)
            if distance <= radius:
                matches.extend((index, distance) for index in indices)
            minimum = distance - radius
            maximum = distance + radius
            stack.extend(
                child
                for edge, child in children.items()
                if minimum <= edge <= maximum
            )
        return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--output", type=Path, default=project_root / "artifacts" / "data-audit")
    parser.add_argument(
        "--phash-radius",
        type=int,
        default=6,
        help="Maximum 64-bit pHash Hamming distance for a review candidate",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not 0 <= args.phash_radius <= 16:
        print("--phash-radius must be between 0 and 16", file=sys.stderr)
        return 2
    root = args.root.resolve()
    dataset_root = (
        root / "datasets" / "building-target" / "CUBIT-InSeg" / "CUBIT-InSeg"
    )
    rows: list[dict] = []
    failures: list[dict] = []
    for split in ("train", "val", "test"):
        archive_path = dataset_root / split / "images.zip"
        archive_relative = archive_path.relative_to(root).as_posix()
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and Path(info.filename).suffix.casefold() in {".jpg", ".jpeg", ".png"}
                ),
                key=lambda info: info.filename.casefold(),
            )
            for position, info in enumerate(members, start=1):
                try:
                    encoded = archive.read(info)
                    with Image.open(io.BytesIO(encoded)) as image:
                        image.load()
                        width, height = image.size
                        phash = perceptual_hash(image)
                except Exception as exc:  # report individual corrupt/unsupported assets
                    failures.append(
                        {"split": split, "member": info.filename, "error": str(exc)}
                    )
                    continue
                stem = Path(info.filename).stem
                rows.append(
                    {
                        "record_id": f"cubit_{split}_{stem.casefold()}",
                        "split": split,
                        "image_archive": archive_relative,
                        "image_member": info.filename,
                        "source_image_id": stem,
                        "width": width,
                        "height": height,
                        "encoded_size_bytes": len(encoded),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "phash64": f"{phash:016x}",
                    }
                )
                if position % 500 == 0 or position == len(members):
                    print(f"{split}: {position}/{len(members)}", flush=True)

    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_sha[row["sha256"]].append(index)
    exact_groups = [indices for indices in by_sha.values() if len(indices) > 1]

    pair_rows: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    tree = BKTree()
    for right_index, right in enumerate(rows):
        right_hash = int(right["phash64"], 16)
        for left_index, distance in tree.query(right_hash, args.phash_radius):
            pair = (left_index, right_index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            left = rows[left_index]
            exact = left["sha256"] == right["sha256"]
            pair_rows.append(
                {
                    "match_type": "exact_bytes" if exact else "perceptual_candidate",
                    "phash_hamming_distance": distance,
                    "cross_split": left["split"] != right["split"],
                    "left_record_id": left["record_id"],
                    "left_split": left["split"],
                    "left_member": left["image_member"],
                    "right_record_id": right["record_id"],
                    "right_split": right["split"],
                    "right_member": right["image_member"],
                }
            )
        tree.add(right_hash, right_index)

    pair_rows.sort(
        key=lambda row: (
            not row["cross_split"],
            row["match_type"] != "exact_bytes",
            row["phash_hamming_distance"],
            row["left_record_id"],
            row["right_record_id"],
        )
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hash_path = output / "cubit_image_hashes.csv"
    pair_path = output / "cubit_duplicate_candidates.csv"
    report_path = output / "cubit_duplicate_audit.json"
    write_csv(hash_path, rows, HASH_FIELDS)
    write_csv(pair_path, pair_rows, PAIR_FIELDS)

    exact_cross_split = [
        pair for pair in pair_rows if pair["match_type"] == "exact_bytes" and pair["cross_split"]
    ]
    perceptual_cross_split = [
        pair
        for pair in pair_rows
        if pair["match_type"] == "perceptual_candidate" and pair["cross_split"]
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CUBIT-InSeg",
        "source_images_modified_or_extracted": False,
        "test_labels_opened": False,
        "test_use": "deterministic leakage hashes only; no model or threshold tuning",
        "images_processed": len(rows),
        "images_by_split": dict(Counter(row["split"] for row in rows)),
        "decode_failures": failures,
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_images": sum(len(group) for group in exact_groups),
        "candidate_pair_count": len(pair_rows),
        "cross_split_exact_pair_count": len(exact_cross_split),
        "cross_split_perceptual_candidate_count": len(perceptual_cross_split),
        "phash_algorithm": "64-bit DCT pHash, 32x32 grayscale, top-left 8x8 excluding DC",
        "phash_hamming_radius": args.phash_radius,
        "interpretation": (
            "Exact cross-split matches are leakage. Perceptual candidates require human "
            "review and must not be removed automatically. This source-only audit does not "
            "replace cross-source duplicate analysis on the selected mega-dataset manifest."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {hash_path}")
    print(f"Wrote {pair_path}")
    print(f"Wrote {report_path}")
    print(json.dumps({key: report[key] for key in (
        "images_processed",
        "exact_duplicate_groups",
        "candidate_pair_count",
        "cross_split_exact_pair_count",
        "cross_split_perceptual_candidate_count",
    )}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
