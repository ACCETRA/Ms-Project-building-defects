from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB_ROOT = ROOT / "harness" / "web"
RUNTIME_ROOT = ROOT / "runs" / "harness"
UPLOAD_ROOT = RUNTIME_ROOT / "uploads"
FINDING_ROOT = RUNTIME_ROOT / "findings"
PROJECT_ROOT = RUNTIME_ROOT / "project.json"
DETECTOR_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "comparison" / "yolo" / "yolo11n_detect_v1_queue" / "weights" / "best.pt"
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


def finding_for_upload(filename: str, image_uri: str, image_hash: str, size: tuple[int, int], label: str = "crack", confidence: float = 0.0, box: list[float] | None = None, model_id: str = "harness-placeholder-no-model", checkpoint_hash: str = "0" * 64, mask: dict | None = None, route: str = "harness_placeholder") -> dict:
    width, height = size
    finding_id = f"finding-harness-{image_hash[:16]}"
    box = box or [round(width * 0.2, 2), round(height * 0.2, 2), round(width * 0.45, 2), round(height * 0.35, 2)]
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
            "defect_family": label,
            "subtype": "unspecified",
            "confidence": confidence,
            "geometry": {
                "coordinate_space": "source_image_pixels",
                "box_xywh": box,
                **({"mask": mask} if mask else {}),
                "centerline_uri": None,
            },
            "pipeline_route": [route],
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
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_hash,
            "pipeline_version": "0.1.0-harness",
            "taxonomy_version": "BDI-TAX-001@0.1.0-beta",
            "dataset_manifest_version": "harness-local",
            "runtime": "local-python",
        },
        "evidence_links": [{"modality": "RGB", "uri": image_uri, "status": "source"}],
        "limitation_codes": ["candidate_only", "manual_review_required", "not_structural_safety_determination", "uncalibrated_measurement"],
    }


def detector_findings(filename: str, image_uri: str, image_hash: str, data: bytes, size: tuple[int, int], route: str = "detect") -> list[dict]:
    weights = DETECTOR_WEIGHTS
    task = "detect"
    if route == "segment":
        weights = ROOT / "runs" / "segment" / "runs" / "comparison" / "yolo" / "yolo11n_seg_v1_queue" / "weights" / "best.pt"
        task = "segment"
    if not weights.is_file():
        return [finding_for_upload(filename, image_uri, image_hash, size)]
    from ultralytics import YOLO
    from PIL import Image

    checkpoint_hash = sha256(weights.read_bytes())
    with tempfile.TemporaryDirectory(prefix="bdi_model_") as temporary:
        safe_checkpoint = Path(temporary) / weights.name
        shutil.copy2(weights, safe_checkpoint)
        model = YOLO(str(safe_checkpoint))
        with Image.open(io.BytesIO(data)) as image:
            result = model.predict(source=image.convert("RGB"), conf=0.4, device=0, task=task, verbose=False)[0]
    boxes = result.boxes
    findings = []
    for index, coordinates in enumerate(boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = coordinates
        label = result.names[int(boxes.cls[index].item())]
        box = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        mask = None
        if task == "segment" and result.masks is not None:
            polygons = result.masks.xy[index].tolist()
            if len(polygons) >= 3:
                mask = {"encoding": "polygon_xy", "points": [[round(float(point[0]), 2), round(float(point[1]), 2)] for point in polygons]}
        finding = finding_for_upload(
            filename, image_uri, image_hash, size, label,
            round(float(boxes.conf[index].item()), 6), box, f"yolo11n-{task}-v1", checkpoint_hash, mask, f"yolo11n_{task}_v1",
        )
        finding["finding_id"] = f"finding-harness-{image_hash[:12]}-{index:03d}"
        findings.append(finding)
    return findings


def no_finding(filename: str, image_uri: str, image_hash: str, size: tuple[int, int], route: str) -> dict:
    finding = finding_for_upload(filename, image_uri, image_hash, size, "no_visible_target_defect", 1.0, [0, 0, 0, 0], f"{route}-v1", sha256((ROOT / image_uri).read_bytes()), route)
    finding["prediction"]["escalation_state"] = "completed"
    finding["limitation_codes"].append("out_of_domain")
    return finding


def classification_findings(filename: str, image_uri: str, image_hash: str, data: bytes, size: tuple[int, int]) -> list[dict]:
    checkpoint = ROOT / "runs" / "comparison" / "resnet50_v1_queue" / "resnet50_comparison.pt"
    if not checkpoint.is_file():
        return [no_finding(filename, image_uri, image_hash, size, "resnet")]
    import torch
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=None)
    model.fc = __import__("torch").nn.Linear(model.fc.in_features, len(TAXONOMY) - 2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    with Image.open(io.BytesIO(data)) as image:
        tensor = ResNet50_Weights.DEFAULT.transforms()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        scores = torch.sigmoid(model(tensor))[0].tolist()
    checkpoint_hash = sha256(checkpoint.read_bytes())
    findings = []
    for index, confidence in enumerate(scores):
        if confidence < 0.4:
            continue
        label = TAXONOMY[index]
        findings.append(finding_for_upload(filename, image_uri, image_hash, size, label, round(float(confidence), 6), [0, 0, size[0], size[1]], "resnet50-v1", checkpoint_hash, route="resnet_v1"))
    return findings or [no_finding(filename, image_uri, image_hash, size, "resnet")]


def florence_findings(filename: str, image_uri: str, image_hash: str, data: bytes, size: tuple[int, int]) -> list[dict]:
    checkpoint = ROOT / "weights" / "florence-community-2-base-ft"
    if not checkpoint.is_dir():
        return [no_finding(filename, image_uri, image_hash, size, "florence")]
    from bdi.florence import FlorenceRunner
    with Image.open(io.BytesIO(data)) as image:
        runner = FlorenceRunner(checkpoint, "florence-community-2-base-ft")
        try:
            result = runner.infer(image.convert("RGB"))
        finally:
            runner.close()
    checkpoint_hash = sha256(next(checkpoint.rglob("*.safetensors")).read_bytes())
    findings = []
    for index, detection in enumerate(result.get("detections", [])):
        x1, y1, x2, y2 = detection["box_xyxy"]
        findings.append(finding_for_upload(filename, image_uri, image_hash, size, detection["defect_family"], float(detection["confidence_proxy"]), [x1, y1, x2 - x1, y2 - y1], "florence-community-2-base-ft", checkpoint_hash, route="florence_v1"))
    return findings or [no_finding(filename, image_uri, image_hash, size, "florence")]


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
        query = parse_qs(parsed.query)
        if parsed.path == "/api/project":
            self.send_json(self.server.project)
            return
        if parsed.path == "/api/processing":
            self.send_json({"status": "completed", "items": list(self.server.processing.values())})
            return
        if parsed.path == "/api/state":
            self.send_json({"status": "ready", "project": self.server.project, "processing": list(self.server.processing.values()), "findings": list(self.server.findings.values())})
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
        if parsed.path == "/api/project":
            self.update_project()
            return
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
        parts = []
        for part in body.split(b"--" + boundary):
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0 or b'name="image"' not in part[:header_end]:
                continue
            headers = part[:header_end].decode("utf-8", errors="replace")
            filename_match = re.search(r'filename="([^"]*)"', headers)
            filename = filename_match.group(1) if filename_match else ""
            parts.append((filename, part[header_end + 4 :].rstrip(b"\r\n-")))
        if not parts:
            self.send_json({"error": "Choose an image"}, HTTPStatus.BAD_REQUEST)
            return
        route = "detect"
        fields = re.findall(rb'name="([^"]+)"\r\n\r\n([^\r]*)', body)
        for name, value in fields:
            if name == b"route":
                route = value.decode("utf-8", errors="replace").strip()
        results = []
        for filename, data in parts:
            try:
                image = Image.open(io.BytesIO(data))
                size = image.size
                image.verify()
            except Exception as exc:
                results.append({"filename": filename, "status": "failed", "error": f"Invalid image: {exc}"})
                continue
            file_hash = sha256(data)
            safe_name = f"{file_hash[:16]}-{Path(filename).name}"
            destination = UPLOAD_ROOT / safe_name
            destination.write_bytes(data)
            try:
                if route in {"detect", "segment"}:
                    findings = detector_findings(filename, destination.relative_to(ROOT).as_posix(), file_hash, data, size, route)
                elif route == "resnet":
                    findings = classification_findings(filename, destination.relative_to(ROOT).as_posix(), file_hash, data, size)
                elif route == "florence":
                    findings = florence_findings(filename, destination.relative_to(ROOT).as_posix(), file_hash, data, size)
                else:
                    findings = [no_finding(filename, destination.relative_to(ROOT).as_posix(), file_hash, size, route)]
            except Exception as exc:
                results.append({"filename": filename, "status": "failed", "route": route, "error": str(exc)})
                self.server.processing[file_hash] = {"image_hash": file_hash, "filename": filename, "status": "failed", "route": route, "error": str(exc)}
                continue
            for finding in findings:
                self.server.findings[finding["finding_id"]] = finding
                self.persist(finding)
            results.append({"filename": filename, "status": "completed", "route": route, "findings": findings})
            self.server.processing[file_hash] = {"image_hash": file_hash, "filename": filename, "status": "completed", "route": route, "finding_count": len(findings)}
        self.send_json({"status": "completed", "items": results}, HTTPStatus.CREATED)

    def update_project(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_json({"error": "Project body must be valid JSON"}, HTTPStatus.BAD_REQUEST)
            return
        self.server.project.update({key: str(payload[key]).strip() for key in ("project_id", "inspection_id", "site_id", "building_id", "facade", "floor", "zone") if key in payload})
        PROJECT_ROOT.parent.mkdir(parents=True, exist_ok=True)
        PROJECT_ROOT.write_text(json.dumps(self.server.project, indent=2) + "\n", encoding="utf-8")
        self.send_json(self.server.project)

    def review(self, finding_id: str) -> None:
        finding = self.server.findings.get(finding_id)
        if finding is None:
            self.send_json({"error": "Finding not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_json({"error": "Review body must be valid JSON"}, HTTPStatus.BAD_REQUEST)
            return
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
        if kind == "report":
            body = "<html><head><meta charset='utf-8'><title>BDI Inspection Report</title><style>body{font-family:Arial;margin:40px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}small{color:#555}</style></head><body>"
            body += f"<h1>Inspection report: {self.server.project.get('inspection_id', 'local')}</h1><p>{self.server.project.get('site_id', 'Unspecified site')} / {self.server.project.get('building_id', 'Unspecified building')}</p><table><tr><th>Image</th><th>Prediction</th><th>Confidence</th><th>Review</th></tr>"
            for finding in findings:
                body += f"<tr><td>{finding['image']['source_image_id']}</td><td>{finding['prediction']['defect_family']}</td><td>{finding['prediction']['confidence']:.3f}</td><td>{finding['review']['state']}</td></tr>"
            body += "</table><p><small>Candidate findings require manual review. This is not a structural safety determination. Measurements are pixel-only unless calibration is provided.</small></p></body></html>"
            self.send_bytes(body.encode("utf-8"), "text/html; charset=utf-8", HTTPStatus.OK)
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
    processing: dict[str, dict]
    project: dict[str, str]


def main() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    FINDING_ROOT.mkdir(parents=True, exist_ok=True)
    server = HarnessServer(("127.0.0.1", 8765), HarnessHandler)
    server.findings = {}
    server.processing = {}
    server.project = {
        "project_id": "harness-local", "inspection_id": "inspection-local", "site_id": "Unspecified site",
        "building_id": "Unspecified building", "facade": "", "floor": "", "zone": "",
    }
    if PROJECT_ROOT.is_file():
        try:
            server.project.update(json.loads(PROJECT_ROOT.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    for record_path in FINDING_ROOT.glob("*.json"):
        try:
            finding = json.loads(record_path.read_text(encoding="utf-8"))
            server.findings[finding["finding_id"]] = finding
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    print("BDI harness running at http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping harness")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
