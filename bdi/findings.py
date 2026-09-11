"""Construction and validation helpers for immutable finding records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from .measurement import Calibration, measure_box


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "finding_record.schema.json"
TAXONOMY_VERSION = "BDI-TAX-001@0.1.0-beta"
PIPELINE_VERSION = "0.1.0-florence-feasibility"
MANDATORY_LIMITATIONS = [
    "candidate_only",
    "manual_review_required",
    "not_structural_safety_determination",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_checkpoint_sha256(path: Path) -> str:
    """Stable digest of every regular checkpoint file and its relative name."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp_box(box_xyxy: list[float], width: int, height: int) -> list[float] | None:
    if len(box_xyxy) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)]


def build_finding(
    *,
    run_id: str,
    sample: dict[str, str],
    image_path: Path,
    image_size: tuple[int, int],
    defect_family: str,
    subtype: str,
    box_xyxy: list[float],
    confidence_proxy: float,
    pipeline_route: list[str],
    escalation_state: str,
    checkpoint_sha256: str,
    model_id: str,
    quality: dict[str, Any],
    calibration: Calibration | None,
    unavailable_scale_status: str = "not_provided",
) -> dict[str, Any]:
    width, height = image_size
    box_xywh = clamp_box(box_xyxy, width, height)
    if box_xywh is None:
        raise ValueError(f"Invalid generated box: {box_xyxy}")
    limitations = list(MANDATORY_LIMITATIONS)
    if calibration is None:
        limitations.append("uncalibrated_measurement")
    if quality["decision"] != "accepted":
        limitations.append("low_image_quality")

    finding_key = json.dumps(
        [run_id, sample["sample_id"], defect_family, subtype, box_xywh], separators=(",", ":")
    )
    finding_id = f"finding-{uuid.uuid5(uuid.NAMESPACE_URL, finding_key)}"
    capture_mode = sample.get("capture_mode", "unknown")
    if capture_mode not in {"handheld", "UAV", "fixed_camera", "unknown"}:
        capture_mode = "unknown"
    uri = image_path.relative_to(ROOT).as_posix()

    measurement = measure_box(box_xywh, calibration)
    if calibration is None and unavailable_scale_status in {"invalid", "out_of_plane"}:
        measurement["scale_status"] = unavailable_scale_status

    return {
        "schema_version": "1.0.0-beta",
        "finding_id": finding_id,
        "project_id": "feasibility-v0",
        "inspection_id": run_id,
        "created_at": utc_now(),
        "image": {
            "source_image_id": sample["source_image_id"],
            "parent_image_id": sample.get("parent_image_id") or None,
            "uri": uri,
            "sha256": sample["sha256"],
            "width_pixels": width,
            "height_pixels": height,
            "captured_at": None,
            "capture_mode": capture_mode,
            "camera_id": None,
            "quality": quality,
        },
        "location": {
            "site_id": f"dataset:{sample['source_dataset']}",
            "building_id": sample.get("group_id") or sample["source_image_id"],
            "elevation_or_facade": None,
            "floor": None,
            "room_or_zone": None,
            "element_type": "unknown",
            "gps": None,
        },
        "prediction": {
            "immutable": True,
            "defect_family": defect_family,
            "subtype": subtype,
            "confidence": round(max(0.0, min(float(confidence_proxy), 1.0)), 6),
            "geometry": {
                "coordinate_space": "source_image_pixels",
                "box_xywh": box_xywh,
                "centerline_uri": None,
            },
            "pipeline_route": pipeline_route,
            "escalation_state": escalation_state,
        },
        "measurement": measurement,
        "review": {
            "state": "unreviewed",
            "reviewer_id": None,
            "reviewed_at": None,
            "reviewer_label": None,
            "notes": None,
        },
        "provenance": {
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha256,
            "pipeline_version": PIPELINE_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "dataset_manifest_version": "feasibility_v0_materialized",
            "runtime": "windows-python311-cuda-fp16",
        },
        "evidence_links": [{"modality": "RGB", "uri": uri, "status": "source"}],
        "limitation_codes": limitations,
    }


def validate_findings(findings: list[dict[str, Any]]) -> None:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    for finding in findings:
        errors = sorted(validator.iter_errors(finding), key=lambda error: list(error.path))
        if errors:
            details = "; ".join(
                f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
                for error in errors[:5]
            )
            raise ValueError(f"Finding {finding.get('finding_id')} failed schema validation: {details}")
