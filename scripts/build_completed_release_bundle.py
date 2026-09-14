#!/usr/bin/env python3
"""Build the upgraded, self-contained Project-Alpha FYP verification bundle.

Includes:
- All source code (bdi, harness, tests, schemas, config)
- Complete documentation with all 5 new upgrade chapters (Ablation, Generalization, Field Test, Explainability, Florence)
- All 50+ automation, evaluation, and dataset creation/reproduction scripts
- Model checkpoints (YOLO, ResNet, SAM, Florence)
- Complete evaluation outputs (evaluation metrics, ablation results, Grad-CAM panels)
- Real-world field survey (20 field photos, annotated visuals, HTML report, findings.jsonl)
- Cryptographic SHA-256 verification manifest
- Excludes raw multi-gigabyte dataset partitions while providing 1-click dataset reproduction scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = Path(r"C:\Users\Z B O O K\OneDrive\Documents\LAST OF USE\Project-Alpha-FYP-Complete-V2")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file_safe(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete FYP upgraded release bundle")
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    out_dir = args.output
    if out_dir.exists():
        print(f"Removing existing target folder {out_dir}...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BUILDING UPGRADED FYP VERIFICATION BUNDLE")
    print(f"Destination: {out_dir}")
    print("=" * 70)

    # 1. Root files
    root_files = [
        ".gitattributes",
        ".gitignore",
        "README.md",
        "START_HERE.md",
        "PROJECT_READINESS.md",
        "requirements-audit.txt",
        "requirements-runtime.txt",
    ]
    for rf in root_files:
        src = ROOT / rf
        if src.is_file():
            copy_file_safe(src, out_dir / rf)

    # Add comprehensive Third-Party Verification Guide
    verification_guide = out_dir / "THIRD_PARTY_VERIFICATION_GUIDE.md"
    verification_guide.write_text(
        """# Third-Party Verification and Audit Guide
Building Defect Inspection (BDI) — Master's FYP Research Prototype

This package is a self-contained, reproducible distribution of the Building Defect Inspection research prototype. It contains all model weights, evaluation metrics, ablation studies, explainability maps, field test surveys, and software harnesses.

---

## 1. Quick Verification (Offline Mode)

All pre-trained and fine-tuned checkpoints are included in `weights/` and `runs/`:
- **YOLO11n Detector:** `runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt`
- **YOLO11n Segmenter:** `runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt`
- **ResNet-50 Classifier:** `runs/comparison/resnet50_v1_queue/resnet50_comparison.pt`
- **Florence-2 Base FT:** `weights/florence-community-2-base-ft/`
- **SAM 2.1 Tiny:** `weights/sam2.1-hiera-tiny/`

### Test Model Inference Instantly
To run a fast multi-model smoke inference test on the included field images:
```powershell
python scripts/smoke_model_inference.py
```

### Review Interactive Field Test Report
Open the included standalone HTML report in any web browser:
`data/field_test/field_inspection_report.html`

### View Model Explainability Panels
Inspect the Grad-CAM feature attribution heatmaps in:
`runs/explainability/`

---

## 2. Reproducing the Full Multi-Source Dataset

To preserve portable archive sizes, raw multi-gigabyte training image folders (`data/v1/images/`, `datasets/`) are excluded from this release bundle. 

All normalization manifests, label annotations, and reproduction scripts are included. To restore the complete dataset from authoritative public sources:

```powershell
# 1. Download source datasets (CUBIT-InSeg, CODEBRIM, DACL10K, CiF, S2DS)
powershell -ExecutionPolicy Bypass -File scripts/download_datasets.ps1

# 2. Extract, audit duplicates, and normalize into unified taxonomy
powershell -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1

# 3. Build frozen v1 task manifests and training inputs
python scripts/build_v1_manifest.py
python scripts/build_v1_training_inputs.py
```
See `docs/DATASET_REPRODUCTION_GUIDE.md` and `docs/DATASET_CONTRACT.md` for full dataset governance.

---

## 3. Reviewing Thesis Deliverables and Upgrades

Key research documentation and experimental findings:
- **Ablation Studies:** `docs/ABLATION_STUDIES.md` (Resolution 512 vs 640, augmentations, loss weighting, backbone depth)
- **Generalization Gap Analysis:** `docs/GENERALIZATION_GAP_ANALYSIS.md` (Mathematical dissection of in-domain 99.3% vs cross-domain 36.6%)
- **Field Inspection Protocol:** `docs/FIELD_TEST_PROTOCOL.md` (Engineering site survey standards and calibration)
- **Florence vs YOLO vs ResNet Comparison:** `docs/EXPERIMENT_MATRIX.md` (Locked-test empirical benchmark)
- **Interactive Review Dashboard:** `harness/index.html` (Browser workflow for uploading, defect detection, and PE sign-off)

---

## 4. Cryptographic Hash Verification

Every file in this bundle is hashed with SHA-256 in `artifacts/offline_bundle_manifest.sha256`.
To verify integrity:
```powershell
Get-FileHash -Algorithm SHA256 <filename>
```
""",
        encoding="utf-8",
    )

    # 2. Source directories to copy completely
    copy_dirs = [
        "bdi",
        "config",
        "confirmation",
        "docs",
        "examples",
        "harness",
        "schemas",
        "scripts",
        "tests",
        "vendor",
        "weights",
        "data/manifests",
        "data/field_test",
        "runs/evaluation",
        "runs/explainability",
    ]

    for rel_dir in copy_dirs:
        src_dir = ROOT / rel_dir
        if not src_dir.exists():
            continue
        print(f"Copying {rel_dir}...")
        for p in src_dir.rglob("*"):
            if p.is_file():
                # Skip cache and tmp files
                if "__pycache__" in p.parts or ".cache" in p.parts or p.suffix == ".tmp":
                    continue
                rel = p.relative_to(ROOT)
                copy_file_safe(p, out_dir / rel)

    # 3. Copy trained checkpoints in runs/
    checkpoint_paths = [
        "runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt",
        "runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/last.pt",
        "runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt",
        "runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/last.pt",
        "runs/comparison/resnet50_v1_queue/resnet50_comparison.pt",
        "runs/comparison/yolo_sam/v1-sample-10",
        "runs/florence/v1-validation-sample-10",
        "runs/sam/decoder_v1_fp32_smoke",
        "runs/v1_queue",
    ]

    for cp in checkpoint_paths:
        src = ROOT / cp
        if src.is_file():
            copy_file_safe(src, out_dir / cp)
        elif src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    copy_file_safe(p, out_dir / p.relative_to(ROOT))

    # 4. Copy dataset documentation stubs (without raw images)
    dataset_doc_files = [
        "datasets/README.md",
        "datasets/license.md",
        "datasets/CODEBRIM/README.md",
        "datasets/CODEBRIM/license.md",
        "datasets/DACL10K/README.md",
        "datasets/building-target/README.md",
        "datasets/building-target/S2DS/README.md",
        "datasets/building-target/S2DS/LICENSE",
        "datasets/CiF-tiled/README.md",
        "datasets/UAV-candidates/README.md",
    ]
    for df in dataset_doc_files:
        src = ROOT / df
        if src.is_file():
            copy_file_safe(src, out_dir / df)

    # 5. Generate SHA256 manifest
    print("\nGenerating SHA256 integrity manifest for all bundle files...")
    manifest_rel = Path("artifacts/offline_bundle_manifest.sha256")
    manifest_out = out_dir / manifest_rel
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p != manifest_out])
    manifest_lines = []
    total_bytes = 0

    for idx, f in enumerate(all_files, 1):
        rel = f.relative_to(out_dir).as_posix()
        h = file_sha256(f)
        manifest_lines.append(f"{h}  {rel}")
        total_bytes += f.stat().st_size
        if idx % 100 == 0 or idx == len(all_files):
            print(f"  Hashed {idx}/{len(all_files)} files...")

    manifest_out.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print("UPGRADED FYP VERIFICATION BUNDLE COMPLETED SUCCESSFULLY")
    print(f"Target Location: {out_dir}")
    print(f"Total Files:     {len(all_files) + 1}")
    print(f"Total Size:      {total_bytes / (1024**3):.2f} GiB")
    print(f"Integrity File:  {manifest_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
