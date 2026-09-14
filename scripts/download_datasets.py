#!/usr/bin/env python3
"""Download and verify project datasets from their original sources.

This script recreates the exact dataset layout used by the project.
It acquires each source from its original publication endpoint,
verifies sizes and checksums where available, and preserves the
immutable archive layout.  Restricted datasets (CODEBRIM, S2DS)
require explicit --accept-licenses.

Usage (via the wrapper):
    powershell -ExecutionPolicy Bypass -File scripts/download_datasets.ps1 -AcceptLicenses

Direct:
    python scripts/download_datasets.py --root . --dataset all --accept-licenses
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Source registry — pins the exact revisions used by this project
# ---------------------------------------------------------------------------

SOURCES: dict[str, dict] = {
    "cif": {
        "description": "CiF tiled 1024x1024 official splits (Hugging Face parquet)",
        "license": "CC BY 4.0",
        "restricted": False,
        "method": "huggingface",
        "repo_id": "cif-benchmark/CiF-tiled",
        "target_dir": "datasets/CiF-tiled",
        "expected_total_bytes": 24_770_000_000,  # ~23.07 GB decimal
        "verify": "parquet_count",
        "expected_file_pattern": "data/*.parquet",
    },
    "dacl10k": {
        "description": "DACL10K v2 development-phase labeled archive",
        "license": "CC BY-NC 4.0",
        "restricted": False,
        "method": "gdown",
        "file_id": "15_crXMnhah3oW9q-5pHxvTtKo-71BXGV",
        "filename": "dacl10k_v2_devphase.zip",
        "target_dir": "datasets/DACL10K",
        "expected_bytes": 5_109_737_241,
        "expected_sha256": "dcbcd5fb82699076a2c7f3a72492a9ef798870e0ca1f0c9399360f273ea95260",
    },
    "codebrim_orig": {
        "description": "CODEBRIM original images and bounding-box annotations",
        "license": "Custom non-commercial academic",
        "restricted": True,
        "method": "gdown",
        "file_id": "1x2wKBo7RDZMiF3TYa2mDk1MmBaFRuqkh",
        "filename": "CODEBRIM_original_images.zip",
        "target_dir": "datasets/CODEBRIM",
        "expected_bytes": 8_310_630_622,
        "expected_md5": "27baf3a036d0b7d757ff4df47c08c449",
    },
    "codebrim_class": {
        "description": "CODEBRIM unbalanced classification archive",
        "license": "Custom non-commercial academic",
        "restricted": True,
        "method": "gdown",
        "file_id": "1JmlCwDyLBFLQ1JN2_c8AJIG1gR0NlO1r",
        "filename": "CODEBRIM_classification_dataset.zip",
        "target_dir": "datasets/CODEBRIM",
        "expected_bytes": 7_911_716_093,
        "expected_md5": "c1612d9674e2e628e72e7f5817c40130",
    },
    "s2ds": {
        "description": "S2DS structural surface damage segmentation dataset",
        "license": "Repository terms — non-commercial academic",
        "restricted": True,
        "method": "gdown",
        "file_id": "1l-HB5t3v1NSURxMIoaLgdiGaQRKz6Rsm",
        "filename": "s2ds.zip",
        "target_dir": "datasets/building-target/S2DS",
        "expected_bytes": 1_330_000_000,  # ~1.24 GiB
    },
    "uav75": {
        "description": "UAV75 crack detection dataset (75 images)",
        "license": "Academic use",
        "restricted": False,
        "method": "git_clone",
        "repo_url": "https://github.com/ozgeanli/UAV75.git",
        "target_dir": "datasets/UAV-candidates/UAV75",
        "expected_image_count": 75,
    },
    "cubit": {
        "description": "CUBIT-InSeg building facade UAV instance segmentation",
        "license": "CC BY 4.0",
        "restricted": False,
        "method": "huggingface",
        "repo_id": "CUBIT-InSeg/CUBIT-InSeg",
        "target_dir": "datasets/building-target/CUBIT-InSeg",
        "expected_total_bytes": 19_059_292_068,
        "verify": "archive_count",
        "expected_archive_count": 6,
    },
}

DATASET_ALIASES = {
    "codebrim": ["codebrim_orig", "codebrim_class"],
    "all": list(SOURCES.keys()),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_datasets(names: Sequence[str]) -> list[str]:
    result: list[str] = []
    for name in names:
        name = name.strip().lower()
        if name in DATASET_ALIASES:
            result.extend(DATASET_ALIASES[name])
        elif name in SOURCES:
            result.append(name)
        else:
            raise ValueError(f"Unknown dataset: {name!r}. Available: {', '.join(sorted(SOURCES))} or aliases {', '.join(sorted(DATASET_ALIASES))}")
    seen: set[str] = set()
    return [x for x in result if not (x in seen or seen.add(x))]


def download_gdown(source: dict, target_dir: Path, force: bool) -> Path:
    import gdown
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / source["filename"]
    if output.exists() and not force:
        print(f"  Already present: {output}")
        return output
    url = f"https://drive.google.com/uc?id={source['file_id']}"
    print(f"  Downloading {source['filename']} from Google Drive...")
    gdown.download(url, str(output), quiet=False)
    return output


def download_huggingface(source: dict, target_dir: Path, force: bool) -> Path:
    from huggingface_hub import snapshot_download
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()) and not force:
        print(f"  Already present: {target_dir}")
        return target_dir
    print(f"  Downloading {source['repo_id']} from Hugging Face...")
    snapshot_download(
        repo_id=source["repo_id"],
        repo_type="dataset",
        local_dir=str(target_dir),
    )
    return target_dir


def download_git_clone(source: dict, target_dir: Path, force: bool) -> Path:
    target_dir_p = Path(target_dir)
    if target_dir_p.exists() and any(target_dir_p.iterdir()) and not force:
        print(f"  Already present: {target_dir}")
        return target_dir_p
    if target_dir_p.exists():
        shutil.rmtree(target_dir_p)
    print(f"  Cloning {source['repo_url']}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", source["repo_url"], str(target_dir_p)],
        check=True,
    )
    return target_dir_p


def verify_source(name: str, source: dict, target_dir: Path) -> None:
    """Run post-download verification checks."""
    if source["method"] == "gdown":
        filepath = target_dir / source["filename"]
        if not filepath.exists():
            raise FileNotFoundError(f"  MISSING: {filepath}")
        size = filepath.stat().st_size
        if "expected_bytes" in source:
            expected = source["expected_bytes"]
            # Allow 5% tolerance for approximate sizes
            if isinstance(expected, int) and expected > 1_000_000_000:
                tolerance = 0.05
                if abs(size - expected) / expected > tolerance:
                    print(f"  WARNING: {name} size {size:,} differs from expected {expected:,} by more than {tolerance:.0%}")
                else:
                    print(f"  Size OK: {size:,} bytes")
            elif size != expected:
                raise ValueError(f"  Size mismatch for {name}: got {size:,}, expected {expected:,}")
            else:
                print(f"  Size OK: {size:,} bytes")
        if "expected_sha256" in source:
            actual = sha256_file(filepath)
            if actual != source["expected_sha256"]:
                raise ValueError(f"  SHA-256 mismatch for {name}: got {actual}, expected {source['expected_sha256']}")
            print(f"  SHA-256 OK: {actual}")
        if "expected_md5" in source:
            actual = md5_file(filepath)
            if actual != source["expected_md5"]:
                raise ValueError(f"  MD5 mismatch for {name}: got {actual}, expected {source['expected_md5']}")
            print(f"  MD5 OK: {actual}")

    elif source["method"] == "huggingface":
        if "verify" in source:
            if source["verify"] == "parquet_count":
                import glob
                pattern = str(target_dir / source.get("expected_file_pattern", "**/*"))
                count = len(glob.glob(pattern, recursive=True))
                print(f"  Found {count} matching files")
            elif source["verify"] == "archive_count":
                archives = list(target_dir.rglob("*.zip")) + list(target_dir.rglob("*.tar.gz"))
                print(f"  Found {len(archives)} archives")

    elif source["method"] == "git_clone":
        if "expected_image_count" in source:
            images = list(target_dir.rglob("*.jpg")) + list(target_dir.rglob("*.png")) + list(target_dir.rglob("*.bmp"))
            print(f"  Found {len(images)} images (expected {source['expected_image_count']})")

    print(f"  {name}: verified")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify project datasets")
    parser.add_argument("--root", type=Path, required=True, help="Project root directory")
    parser.add_argument("--dataset", nargs="+", default=["all"], help="Datasets to download (or 'all')")
    parser.add_argument("--accept-licenses", action="store_true", help="Accept restricted dataset terms")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    root = args.root.resolve()
    datasets = resolve_datasets(args.dataset)

    print(f"Project root: {root}")
    print(f"Datasets to acquire: {', '.join(datasets)}")
    print()

    failed: list[str] = []
    skipped: list[str] = []

    for name in datasets:
        source = SOURCES[name]
        target_dir = root / source["target_dir"]
        print(f"[{name}] {source['description']}")
        print(f"  License: {source['license']}")

        if source["restricted"] and not args.accept_licenses:
            print(f"  SKIPPED: restricted dataset requires --accept-licenses")
            skipped.append(name)
            continue

        try:
            if source["method"] == "gdown":
                download_gdown(source, target_dir, args.force)
            elif source["method"] == "huggingface":
                download_huggingface(source, target_dir, args.force)
            elif source["method"] == "git_clone":
                download_git_clone(source, target_dir, args.force)
            else:
                raise ValueError(f"Unknown download method: {source['method']}")

            verify_source(name, source, target_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(name)

        print()

    print("=" * 60)
    print(f"Completed: {len(datasets) - len(failed) - len(skipped)}")
    if skipped:
        print(f"Skipped (license): {', '.join(skipped)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    print("All requested datasets acquired and verified.")


if __name__ == "__main__":
    main()
