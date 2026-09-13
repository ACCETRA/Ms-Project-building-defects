#!/usr/bin/env python3
"""Prepare explicit source-label and target-site review artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def prepare_s2ds_manifest() -> Path:
    output = ROOT / "data/manifests/s2ds_test_class_index_status.csv"
    archive = ROOT / "datasets/building-target/S2DS/s2ds.zip"
    with zipfile.ZipFile(archive) as source:
        images = sorted(Path(name).stem for name in source.namelist() if name.startswith("test/") and name.endswith(".png") and not name.endswith("_lab.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "source_member", "label_member", "class_identity_available", "foreground_semantics", "review_status"])
        writer.writeheader()
        for sample_id in images:
            writer.writerow({"sample_id": sample_id, "source_member": f"test/{sample_id}.png", "label_member": f"test/{sample_id}_lab.png", "class_identity_available": "false", "foreground_semantics": "binary foreground; source README lists multiple classes but test mask does not identify class", "review_status": "blocked_pending_class_index_manifest"})
    return output


def prepare_target_template() -> Path:
    output = ROOT / "data/manifests/target_site_evaluation_template.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text("image_id,building_id,site_id,capture_session,split,approved_for_evaluation,ground_truth_manifest,notes\n", encoding="utf-8")
    return output


def prepare_error_review() -> Path:
    output = ROOT / "runs/evaluation/error_review_queue.json"
    source_files = {
        name: ROOT / "runs/evaluation/source_specific" / f"{name}_segment.json"
        for name in ("uav75", "s2ds", "dacl", "cif")
    }
    queue = []
    for source, path in source_files.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        queue.append({"source": source, "priority": "high" if metrics.get("metrics/mAP50(M)", 1.0) < 0.1 else "review", "samples": payload["samples"], "mask_map50": metrics.get("metrics/mAP50(M)"), "mask_map50_95": metrics.get("metrics/mAP50-95(M)"), "required_review": ["false negatives", "false positives", "empty predictions", "small/thin defects", "out-of-domain/background confusion"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "prioritized_review_queue", "items": queue, "limitations": ["Aggregate metrics do not identify individual error images; per-image prediction/ground-truth overlays must be reviewed next.", "Manual review is required; results are not structural-safety determinations."]}, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    print(prepare_s2ds_manifest())
    print(prepare_target_template())
    print(prepare_error_review())


if __name__ == "__main__":
    main()
