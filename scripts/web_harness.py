#!/usr/bin/env python3
"""Minimal local web harness for project status and Florence inference smoke tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "weights" / "florence-community-2-base-ft"
MANIFEST = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"


def summarize_harness_status() -> dict[str, Any]:
    checkpoint_exists = CHECKPOINT.is_dir()
    manifest_exists = MANIFEST.is_file()
    ready = checkpoint_exists and manifest_exists
    status = "ready" if ready else "blocked"
    if ready:
        status = "warning"
    return {
        "checkpoint": str(CHECKPOINT),
        "manifest": str(MANIFEST),
        "status": status,
        "ready": ready,
        "notes": (
            "The completed FYP harness is ready for local model inference and review."
            if ready
            else "Missing checkpoint or manifest; the project is not yet ready for the full v1 route."
        ),
    }


def _predict_from_image(image_path: str) -> dict[str, Any]:
    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Missing Florence checkpoint at {CHECKPOINT}")
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")

    from PIL import Image

    from bdi.florence import FlorenceRunner

    runner = FlorenceRunner(CHECKPOINT, "florence-2-base-ft", max_new_tokens=128)
    try:
        with Image.open(image_file) as image:
            return runner.infer(image.convert("RGB"))
    finally:
        runner.close()


class HarnessHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/status"}:
            payload = summarize_harness_status()
            self._send_json(payload)
            return
        if parsed.path == "/predict":
            self._send_json({"error": "Use POST /predict with a JSON body containing an image_path field."}, status=400)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self._send_json({"error": "Not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON."}, status=400)
            return

        image_path = payload.get("image_path")
        if not image_path:
            self._send_json({"error": "Missing JSON field: image_path"}, status=400)
            return
        try:
            result = _predict_from_image(image_path)
        except Exception as exc:  # pragma: no cover - defensive network error path
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json({"status": "ok", "result": result})

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--status", action="store_true", help="Print the harness status summary and exit.")
    parser.add_argument("--image", type=str, help="Run a single-image Florence smoke inference and print the JSON result.")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(summarize_harness_status(), indent=2, ensure_ascii=False))
        return
    if args.image:
        print(json.dumps(_predict_from_image(args.image), indent=2, ensure_ascii=False))
        return

    server = ThreadingHTTPServer((args.host, args.port), HarnessHandler)
    print(json.dumps({"status": "serving", "host": args.host, "port": args.port, "summary": summarize_harness_status()}, ensure_ascii=False))
    server.serve_forever()


if __name__ == "__main__":
    main()
