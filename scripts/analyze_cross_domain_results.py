#!/usr/bin/env python3
"""Summarize automated cross-domain comparison weakness from saved metrics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("cif", "dacl", "s2ds", "uav75")


def main() -> None:
    results = {}
    for source in SOURCES:
        path = ROOT / "runs/evaluation/source_specific" / f"{source}_segment.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            results[source] = {
                "samples": payload["samples"],
                "box_map50": metrics.get("metrics/mAP50(B)"),
                "mask_map50": metrics.get("metrics/mAP50(M)"),
                "mask_map50_95": metrics.get("metrics/mAP50-95(M)"),
                "mask_recall": metrics.get("metrics/recall(M)"),
                "semantics": payload.get("s2ds_semantics"),
            }
    ordered = sorted(results.items(), key=lambda item: item[1]["mask_map50"] or 0.0, reverse=True)
    analysis = {
        "status": "automated_cross_domain_analysis",
        "results": results,
        "ranking_by_mask_map50": [source for source, _ in ordered],
        "findings": [
            "CUBIT is the primary in-domain locked benchmark; cross-domain scores must not be combined into one accuracy number.",
            "Low recall and mask AP across external sources indicate domain shift, label/task mismatch, resolution/thin-defect sensitivity, or threshold mismatch.",
            "S2DS is binary foreground only because class identity is unavailable; it is not a six-class comparison.",
            "Image-level false-positive and false-negative causes require human overlay review and cannot be inferred from aggregate metrics alone.",
        ],
        "required_next_checks": [
            "Review per-image overlays and empty predictions.",
            "Compare source image resolution and annotation geometry.",
            "Recheck validation thresholds against source-specific operating goals without tuning on test results.",
            "Consider source-balanced adaptation only after error review.",
        ],
    }
    output = ROOT / "runs/evaluation/cross_domain_analysis.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
