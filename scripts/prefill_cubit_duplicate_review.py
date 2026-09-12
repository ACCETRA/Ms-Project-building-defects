#!/usr/bin/env python3
"""Create a conservative, machine-assisted CUBIT duplicate-review worksheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "artifacts" / "data-audit" / "cubit_cross_split_near_duplicate_review_queue.csv"


def classify(row: dict[str, str]) -> tuple[str, str]:
    distance = int(row["phash_hamming_distance"])
    correlation = float(row["grayscale_correlation"])
    mae = float(row["grayscale_mae_0_1"])
    if distance == 0 and correlation >= 0.90 and mae <= 0.05:
        return (
            "same_duplicate",
            "Machine-assisted prefill: pHash distance=0, high grayscale correlation, and low grayscale MAE. Human must confirm visually.",
        )
    if distance <= 2 and correlation >= 0.90 and mae <= 0.08:
        return (
            "same_parent_near_duplicate",
            "Machine-assisted prefill: very small pHash distance, high grayscale correlation, and low grayscale MAE. Human must confirm shared scene/capture.",
        )
    return (
        "uncertain",
        "Machine-assisted prefill could not establish a safe duplicate decision from metrics alone. Human visual review required.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "hanzalah review" / "review_queue_prefilled.csv")
    args = parser.parse_args()
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = list(rows[0]) + ["machine_decision", "machine_review_notes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            decision, notes = classify(row)
            row["machine_decision"] = decision
            row["machine_review_notes"] = notes
            writer.writerow(row)
    counts = {decision: sum(classify(row)[0] == decision for row in rows) for decision in ("same_duplicate", "same_parent_near_duplicate", "different_image", "uncertain")}
    print({"rows": len(rows), "output": str(args.output), "counts": counts})


if __name__ == "__main__":
    main()
