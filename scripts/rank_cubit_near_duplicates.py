#!/usr/bin/env python3
"""Rank unresolved cross-split CUBIT perceptual matches for human review.

Exact-dedup exclusions are applied first. Remaining pairs are ranked using the
existing pHash distance plus grayscale thumbnail correlation/error. This script
does not exclude any image, alter any split, extract source files, or open test
labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageOps


FIELDS = [
    "review_rank",
    "phash_hamming_distance",
    "grayscale_correlation",
    "grayscale_mae_0_1",
    "grayscale_rmse_0_1",
    "same_dimensions",
    "left_record_id",
    "left_split",
    "left_member",
    "right_record_id",
    "right_split",
    "right_member",
    "decision",
    "review_notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grayscale_thumbnail(encoded: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(encoded)) as image:
        image.draft("L", (256, 256))
        grayscale = ImageOps.exif_transpose(image).convert("L")
        resized = grayscale.resize((128, 128), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32) / 255.0


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0:
        return 1.0 if np.array_equal(left_flat, right_flat) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> int:
    root = parse_args().root.resolve()
    audit_root = root / "artifacts" / "data-audit"
    hashes = {
        row["record_id"]: row for row in read_csv(audit_root / "cubit_image_hashes.csv")
    }
    selected = {
        row["record_id"]
        for row in read_csv(
            root / "data" / "manifests" / "cubit_exact_dedup_decisions_v1.csv"
        )
        if row["selected_after_exact_dedup"].casefold() == "true"
    }
    raw_candidates = [
        row
        for row in read_csv(audit_root / "cubit_duplicate_candidates.csv")
        if row["match_type"] == "perceptual_candidate"
        and row["cross_split"].casefold() == "true"
    ]
    candidates = [
        row
        for row in raw_candidates
        if row["left_record_id"] in selected and row["right_record_id"] in selected
    ]
    needed_ids = {
        record_id
        for row in candidates
        for record_id in (row["left_record_id"], row["right_record_id"])
    }
    by_archive: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(needed_ids):
        by_archive[hashes[record_id]["image_archive"]].append(record_id)

    thumbnails: dict[str, np.ndarray] = {}
    for archive_relative, record_ids in by_archive.items():
        with zipfile.ZipFile(root / archive_relative) as archive:
            for position, record_id in enumerate(record_ids, start=1):
                row = hashes[record_id]
                thumbnails[record_id] = grayscale_thumbnail(
                    archive.read(row["image_member"])
                )
                if position % 250 == 0 or position == len(record_ids):
                    print(
                        f"{Path(archive_relative).parent.name}: {position}/{len(record_ids)}",
                        flush=True,
                    )

    review_rows: list[dict] = []
    for row in candidates:
        left_id = row["left_record_id"]
        right_id = row["right_record_id"]
        left = thumbnails[left_id]
        right = thumbnails[right_id]
        difference = left - right
        left_meta = hashes[left_id]
        right_meta = hashes[right_id]
        review_rows.append(
            {
                "review_rank": 0,
                "phash_hamming_distance": int(row["phash_hamming_distance"]),
                "grayscale_correlation": round(correlation(left, right), 8),
                "grayscale_mae_0_1": round(float(np.mean(np.abs(difference))), 8),
                "grayscale_rmse_0_1": round(float(np.sqrt(np.mean(difference**2))), 8),
                "same_dimensions": (
                    left_meta["width"] == right_meta["width"]
                    and left_meta["height"] == right_meta["height"]
                ),
                "left_record_id": left_id,
                "left_split": row["left_split"],
                "left_member": row["left_member"],
                "right_record_id": right_id,
                "right_split": row["right_split"],
                "right_member": row["right_member"],
                "decision": "pending_human_review",
                "review_notes": "",
            }
        )
    review_rows.sort(
        key=lambda row: (
            row["phash_hamming_distance"],
            -row["grayscale_correlation"],
            row["grayscale_mae_0_1"],
            row["left_record_id"],
            row["right_record_id"],
        )
    )
    for rank, row in enumerate(review_rows, start=1):
        row["review_rank"] = rank

    queue_path = audit_root / "cubit_cross_split_near_duplicate_review_queue.csv"
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)

    report = {
        "schema_version": 1,
        "dataset": "CUBIT-InSeg",
        "raw_cross_split_perceptual_candidates": len(raw_candidates),
        "already_resolved_by_exact_dedup": len(raw_candidates) - len(candidates),
        "unresolved_pairs_ranked": len(review_rows),
        "unresolved_by_phash_distance": dict(
            sorted(Counter(row["phash_hamming_distance"] for row in review_rows).items())
        ),
        "images_thumbnail_decoded": len(thumbnails),
        "source_images_modified_or_extracted": False,
        "test_labels_opened": False,
        "automated_exclusions": 0,
        "ranking_only": True,
        "ranking_order": (
            "ascending pHash Hamming distance, descending grayscale correlation, "
            "ascending grayscale MAE, stable record IDs"
        ),
        "required_next_step": (
            "A reviewer must inspect candidate pairs and record keep/exclude decisions. "
            "No numerical threshold in this report is an automatic duplicate verdict."
        ),
    }
    report_path = audit_root / "cubit_near_duplicate_review_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {queue_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
