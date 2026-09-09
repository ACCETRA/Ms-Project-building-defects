#!/usr/bin/env python3
"""Create non-destructive exact-duplicate decisions for CUBIT-InSeg.

The immutable official archives remain unchanged. When the same encoded image is
present in multiple official splits, the deterministic priority is test > val >
train. Within one split, the lexicographically first record is retained. This
protects held-out evaluation while recording every exclusion for auditability.
Perceptual candidates are not removed automatically.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


FIELDS = [
    "record_id",
    "split",
    "source_image_id",
    "image_archive",
    "image_member",
    "annotation_archive",
    "annotation_member",
    "sha256",
    "selected_after_exact_dedup",
    "exact_group_size",
    "exact_keeper_record_id",
    "decision_reason",
    "test_locked",
]
PRIORITY = {"test": 0, "val": 1, "train": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    root = parse_args().root.resolve()
    hashes_path = root / "artifacts" / "data-audit" / "cubit_image_hashes.csv"
    candidates_path = (
        root / "artifacts" / "data-audit" / "cubit_duplicate_candidates.csv"
    )
    source_rows = read_csv(hashes_path)
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_hash[row["sha256"]].append(row)

    decisions: list[dict] = []
    for sha256, group in by_hash.items():
        ordered = sorted(group, key=lambda row: (PRIORITY[row["split"]], row["record_id"]))
        keeper = ordered[0]
        for row in ordered:
            selected = row is keeper
            same_split = row["split"] == keeper["split"]
            if len(group) == 1:
                reason = "unique_sha256"
            elif selected:
                reason = "kept_exact_group_by_split_priority"
            elif same_split:
                reason = "excluded_exact_duplicate_within_split"
            else:
                reason = f"excluded_exact_duplicate_of_higher_priority_{keeper['split']}"
            split = row["split"]
            stem = row["source_image_id"]
            split_root = f"datasets/building-target/CUBIT-InSeg/CUBIT-InSeg/{split}"
            decisions.append(
                {
                    "record_id": row["record_id"],
                    "split": split,
                    "source_image_id": stem,
                    "image_archive": row["image_archive"],
                    "image_member": row["image_member"],
                    "annotation_archive": f"{split_root}/labels.zip",
                    "annotation_member": f"labels/{stem}.txt",
                    "sha256": sha256,
                    "selected_after_exact_dedup": selected,
                    "exact_group_size": len(group),
                    "exact_keeper_record_id": keeper["record_id"],
                    "decision_reason": reason,
                    "test_locked": split == "test",
                }
            )
    decisions.sort(key=lambda row: (PRIORITY[row["split"]], row["record_id"]))

    selected = [row for row in decisions if row["selected_after_exact_dedup"]]
    selected_hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        selected_hash_splits[row["sha256"]].add(row["split"])
    residual_cross_split = {
        sha256: sorted(splits)
        for sha256, splits in selected_hash_splits.items()
        if len(splits) > 1
    }
    if residual_cross_split:
        raise RuntimeError("Exact hashes still cross selected splits")

    perceptual_candidates = read_csv(candidates_path)
    cross_split_perceptual = [
        row
        for row in perceptual_candidates
        if row["match_type"] == "perceptual_candidate"
        and row["cross_split"].casefold() == "true"
    ]

    output = root / "data" / "manifests" / "cubit_exact_dedup_decisions_v1.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(decisions)

    report = {
        "manifest_version": "cubit_exact_dedup_decisions_v1",
        "purpose": "exact-duplicate decisions; not the final frozen mega-dataset manifest",
        "raw_archives_modified": False,
        "selection_priority": ["test", "val", "train"],
        "within_split_tiebreaker": "lexicographically first record_id",
        "official_records_by_split": dict(Counter(row["split"] for row in decisions)),
        "selected_records_by_split": dict(Counter(row["split"] for row in selected)),
        "excluded_records_by_split": dict(
            Counter(row["split"] for row in decisions if not row["selected_after_exact_dedup"])
        ),
        "selected_total": len(selected),
        "excluded_total": len(decisions) - len(selected),
        "residual_selected_cross_split_exact_hashes": residual_cross_split,
        "cross_split_perceptual_candidates_pending_review": len(cross_split_perceptual),
        "near_duplicate_policy": "review only; no automatic exclusion",
        "evaluation_policy": (
            "Report publisher-comparable official test metrics separately from the "
            "leakage-clean exact-deduplicated test view. Do not train on excluded copies."
        ),
    }
    report_path = root / "artifacts" / "data-audit" / "cubit_exact_dedup_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(decisions)} decisions to {output.relative_to(root)}")
    print(f"Wrote {report_path.relative_to(root)}")
    print(json.dumps({
        "selected_records_by_split": report["selected_records_by_split"],
        "excluded_records_by_split": report["excluded_records_by_split"],
        "cross_split_perceptual_candidates_pending_review": len(cross_split_perceptual),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
