#!/usr/bin/env python3
"""Select YOLO confidence thresholds using validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-confidence", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    data = args.data if args.data.is_absolute() else ROOT / args.data
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="yolo_threshold_") as temporary:
        safe_weights = Path(temporary) / weights.name
        shutil.copy2(weights, safe_weights)
        model = YOLO(str(safe_weights))
        candidates = []
        confidence = args.min_confidence
        while confidence <= args.max_confidence + 1e-9:
            metrics = model.val(
                data=str(data), split="val", task=args.task, conf=confidence,
                device=0, workers=0, plots=False, verbose=False,
            ).results_dict
            precision = float(metrics.get("metrics/precision(B)", 0.0))
            recall = float(metrics.get("metrics/recall(B)", 0.0))
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            candidates.append({"confidence": round(confidence, 3), "precision": precision, "recall": recall, "f1": f1, "metrics": metrics})
            confidence += args.step
    selected = max(candidates, key=lambda row: (row["f1"], row["recall"], -row["confidence"]))
    result = {
        "task": args.task,
        "data": str(data),
        "weights": str(weights),
        "selection_source": "validation split only",
        "objective": "maximize box F1, then recall, then prefer lower confidence",
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
