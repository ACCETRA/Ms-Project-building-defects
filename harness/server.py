from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "harness" / "web"
RUNTIME_ROOT = ROOT / "runs" / "harness"
UPLOAD_ROOT = RUNTIME_ROOT / "uploads"
FINDING_ROOT = RUNTIME_ROOT / "findings"
TAXONOMY = (
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
    "no_visible_target_defect",
    "unknown_review",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finding_for_upload(filename: str, image_uri: str, image_hash: str, size: tuple[int, int]) -> dict:
    width, height = size
    finding_id = f"finding-harness-{image_hash[:16]}"
    box = [round(width * 0.2, 2), round(height * 0.2, 2), round(width * 0.45, 2), round(height * 0.35, 2)]
    return {
        "schema_version": "1.0.0-beta",
        "finding_id": finding_id,
        "project_id": "harness-local",
        "inspection_id": "harness-local",
        "created_at": now(),
        "image": {
            "source_image_id": filename,
            "parent_image_id": None,
            "uri": image_uri,
            "sha256": image_hash,
            "width_pixels": width,
            "height_pixels": height,
            "captured_at": None,
            "capture_mode": "unknown",
            "camera_id": None,
            "quality": {"decision": "accepted", "flags": [], "notes": None},
        },
        "location": {
            "site_id": "harness-local",
            "building_id": "unassigned",
            "elevation_or_facade": None,
            "floor": None,
            "room_or_zone": None,
            "element_type": "unknown",
            "gps": None,
        },
        "prediction": {
            "immutable": True,
            "defect_family": "crack",
            "subtype": "unspecified",
            "confidence": 0.0,
            "geometry": {
                "coordinate_space": "source_image_pixels",
                "box_xywh": box,
                "centerline_uri": None,
            },
            "pipeline_route": ["harness_placeholder"],
            "escalation_state": "completed",
        },
        "measurement": {
            "scale_status": "not_provided",
            "scale_method": "none",
            "pixel": {
                "length": max(box[2], box[3]),
                "max_width": min(box[2], box[3]),
                "area": box[2] * box[3],
            },
        },
        "review": {
            "state": "unreviewed",
            "reviewer_id": None,
            "reviewed_at": None,
            "reviewer_label": None,
            "notes": None,
        },
        "provenance": {
            "model_id": "harness-placeholder-no-model",
            "checkpoint_sha256": "0" * 64,
            "pipeline_version": "0.1.0-harness",
            "taxonomy_version": "BDI-TAX-001@0.1.0-beta",
            "dataset_manifest_version": "harness-local",
            "runtime": "local-python",
        },
        "evidence_links": [{"modality": "RGB", "uri": image_uri, "status": "source"}],
        "limitation_codes": ["candidate_only", "manual_review_required", "not_structural_safety_determination", "uncalibrated_measurement"],
    }


def flatten(finding: dict) -> dict[str, str | float | int | None]:
    prediction = finding["prediction"]
    geometry = prediction["geometry"]
    review = finding["review"]
    return {
        "finding_id": finding["finding_id"],
        "image": finding["image"]["source_image_id"],
        "prediction_label": prediction["defect_family"],
        "prediction_subtype": prediction["subtype"],
        "confidence": prediction["confidence"],
        "box_x": geometry["box_xywh"][0],
        "box_y": geometry["box_xywh"][1],
        "box_width": geometry["box_xywh"][2],
        "box_height": geometry["box_xywh"][3],
        "review_state": review["state"],
        "reviewer_id": review["reviewer_id"],
        "reviewer_label": review["reviewer_label"],
        "review_notes": review["notes"],
    }


def annotated_image(finding: dict) -> bytes:
    path = ROOT / finding["image"]["uri"]
    with Image.open(path).convert("RGB") as image:
        output = image.copy()
        draw = ImageDraw.Draw(output)
        x, y, width, height = finding["prediction"]["geometry"]["box_xywh"]
        draw.rectangle((x, y, x + width, y + height), outline="#e4513f", width=max(3, image.width // 300))
        label = finding["review"].get("reviewer_label") or finding["prediction"]["defect_family"]
        draw.text((x, max(0, y - 18)), f"{label} | {finding['review']['state']}", fill="#e4513f")
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        return buffer.getvalue()


class HarnessHandler(BaseHTTPRequestHandler):
    server_version = "BDI-Harness/0.1"

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload, indent=2).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.send_json({"status": "ready", "findings": list(self.server.findings.values())})
            return
        if parsed.path.startswith("/api/findings/"):
            if parsed.path.endswith("/image"):
                finding_id = parsed.path.split("/")[-2]
                finding = self.server.findings.get(finding_id)
                if finding is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    image_path = ROOT / finding["image"]["uri"]
                    if image_path.is_file():
                        self.send_bytes(image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "image/jpeg")
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                return
            finding = self.server.findings.get(parsed.path.rsplit("/", 1)[-1])
            if finding is None:
                self.send_json({"error": "Finding not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(finding)
            return
        if parsed.path.startswith("/api/export/"):
            self.export(parsed.path.rsplit("/", 1)[-1])
            return
        relative = "index.html" if parsed.path in {"", "/"} else parsed.path.removeprefix("/")
        path = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in path.parents and path != WEB_ROOT:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.upload()
            return
        if parsed.path.startswith("/api/findings/"):
            self.review(parsed.path.rsplit("/", 1)[-1])
            return
        self.send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)

    def upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "Upload must be multipart/form-data"}, HTTPStatus.BAD_REQUEST)
            return
        boundary_match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
        if boundary_match is None:
            self.send_json({"error": "Multipart boundary is missing"}, HTTPStatus.BAD_REQUEST)
            return
        boundary = (boundary_match.group(1) or boundary_match.group(2)).encode("ascii")
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        image_data = None
        filename = ""
        for part in body.split(b"--" + boundary):
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0 or b'name="image"' not in part[:header_end]:
                continue
            headers = part[:header_end].decode("utf-8", errors="replace")
            filename_match = re.search(r'filename="([^"]*)"', headers)
            filename = filename_match.group(1) if filename_match else ""
            image_data = part[header_end + 4 :].rstrip(b"\r\n-")
            break
        if image_data is None or not filename:
            self.send_json({"error": "Choose an image"}, HTTPStatus.BAD_REQUEST)
            return
        data = image_data
        try:
            image = Image.open(io.BytesIO(data))
            size = image.size
            image.verify()
        except Exception as exc:
            self.send_json({"error": f"Invalid image: {exc}"}, HTTPStatus.BAD_REQUEST)
            return
        file_hash = sha256(data)
        safe_name = f"{file_hash[:16]}-{Path(filename).name}"
        destination = UPLOAD_ROOT / safe_name
        destination.write_bytes(data)
        finding = finding_for_upload(filename, destination.relative_to(ROOT).as_posix(), file_hash, size)
        self.server.findings[finding["finding_id"]] = finding
        self.persist(finding)
        self.send_json({"status": "completed", "finding": finding}, HTTPStatus.CREATED)

    def review(self, finding_id: str) -> None:
        finding = self.server.findings.get(finding_id)
        if finding is None:
            self.send_json({"error": "Finding not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        action = payload.get("action")
        reviewer_id = str(payload.get("reviewer_id") or "local-reviewer").strip()
        notes = str(payload.get("notes") or "").strip() or None
        label = payload.get("label")
        if action == "approve":
            finding["review"]["state"] = "approved"
        elif action == "reject":
            finding["review"]["state"] = "rejected"
        elif action == "relabel" and label in TAXONOMY:
            finding["review"]["state"] = "relabelled"
            finding["review"]["reviewer_label"] = label
        else:
            self.send_json({"error": "Use approve, reject, or a valid relabel action"}, HTTPStatus.BAD_REQUEST)
            return
        finding["review"]["reviewer_id"] = reviewer_id
        finding["review"]["reviewed_at"] = now()
        finding["review"]["notes"] = notes
        self.persist(finding)
        self.send_json(finding)

    def export(self, kind: str) -> None:
        findings = list(self.server.findings.values())
        if kind == "json":
            self.send_bytes(json.dumps(findings, indent=2).encode("utf-8"), "application/json", HTTPStatus.OK)
            return
        if kind == "csv":
            buffer = io.StringIO()
            rows = [flatten(finding) for finding in findings]
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else ["finding_id"])
            writer.writeheader()
            writer.writerows(rows)
            self.send_bytes(buffer.getvalue().encode("utf-8"), "text/csv", HTTPStatus.OK)
            return
        if kind == "image" and findings:
            self.send_bytes(annotated_image(findings[0]), "image/png", HTTPStatus.OK)
            return
        self.send_json({"error": "No findings or unsupported export"}, HTTPStatus.NOT_FOUND)

    @staticmethod
    def persist(finding: dict) -> None:
        FINDING_ROOT.mkdir(parents=True, exist_ok=True)
        (FINDING_ROOT / f"{finding['finding_id']}.json").write_text(json.dumps(finding, indent=2) + "\n", encoding="utf-8")

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{now()}] {format % args}")


class HarnessServer(ThreadingHTTPServer):
    findings: dict[str, dict]


def main() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    FINDING_ROOT.mkdir(parents=True, exist_ok=True)
    server = HarnessServer(("127.0.0.1", 8765), HarnessHandler)
    server.findings = {}
    print("BDI harness running at http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping harness")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
