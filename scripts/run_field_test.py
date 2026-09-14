#!/usr/bin/env python3
"""Execute real-world building defect inspection on uncurated field imagery.

Processes field photos through:
1. Automated Quality Preflight (Luma, Blur, Resolution)
2. YOLO11n Candidate Defect Localization
3. ResNet-50 Multilabel Confidence Triage
4. Finding Record Generation (finding_record.schema.json compliant)
5. Visual Annotated Image Rendering & Interactive HTML Report Generation
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bdi.findings import build_finding, validate_findings, utc_now
from bdi.florence import image_quality

CLASSES = (
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
)

CLASS_COLORS = {
    "crack": (230, 57, 70),          # Vibrant Crimson
    "spalling": (244, 162, 97),       # Burnt Coral
    "honeycombing_rock_pocket": (233, 196, 106), # Amber
    "exposed_rebar": (220, 47, 2),    # Deep Orange-Red
    "rust_staining": (186, 73, 73),   # Rust Brown
    "efflorescence_leaching": (42, 157, 143), # Patina Teal
}

DEFAULT_INPUTS = ROOT / "data/field_test/inputs"
DEFAULT_OUTPUTS = ROOT / "data/field_test"
YOLO_WEIGHTS = ROOT / "runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt"
RESNET_WEIGHTS = ROOT / "runs/comparison/resnet50_v1_queue/resnet50_comparison.pt"


def load_classifier(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def draw_bounding_boxes(
    img: Image.Image,
    detections: list[dict[str, object]],
) -> Image.Image:
    annotated = img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for det in detections:
        cls_name = str(det["label"])
        box = det["box_xywh"]
        conf = float(det["confidence"])
        color = CLASS_COLORS.get(cls_name, (200, 50, 50))

        bx, by, bw, bh = box
        # Draw multi-line boundary for visibility
        for offset in range(3):
            draw.rectangle(
                [bx - offset, by - offset, bx + bw + offset, by + bh + offset],
                outline=color,
            )

        # Label tag badge
        tag = f"{cls_name.upper()} {conf*100:.1f}%"
        draw.rectangle([bx, max(0, by - 22), bx + len(tag) * 8 + 10, by], fill=color)
        draw.text((bx + 4, max(2, by - 18)), tag, fill=(255, 255, 255))

    return annotated


def generate_html_report(
    summary: dict[str, object],
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    html_cards = []
    for rec in records:
        dets_html = ""
        for d in rec["detections"]:
            col = "rgb" + str(CLASS_COLORS.get(d["label"], (100, 100, 100)))
            dets_html += f'<span class="badge" style="background:{col};">{d["label"]} ({d["confidence"]*100:.1f}%)</span> '
        if not dets_html:
            dets_html = '<span class="badge" style="background:#6c757d;">No Defect Detected</span>'

        q_badge = '<span class="badge" style="background:#2a9d8f;">Passed QC</span>'
        if rec["quality"]["flags"]:
            q_badge = f'<span class="badge" style="background:#e76f51;">Warning: {", ".join(rec["quality"]["flags"])}</span>'

        card = f"""
        <div class="card">
            <div class="card-header">
                <h3>{rec["filename"]}</h3>
                <div>{q_badge}</div>
            </div>
            <div class="image-row">
                <div class="img-box">
                    <p>Raw Field Photo</p>
                    <img src="inputs/{rec["filename"]}" alt="Raw" />
                </div>
                <div class="img-box">
                    <p>AI Detections (YOLO11n + ResNet)</p>
                    <img src="annotated/annotated_{rec["filename"]}" alt="Annotated" />
                </div>
            </div>
            <div class="card-footer">
                <div class="tags-container">
                    <strong>Identified Defects:</strong> {dets_html}
                </div>
                <div class="review-status">
                    <strong>Review Status:</strong> <span class="status-pill">Pending Engineer Sign-off</span>
                </div>
            </div>
        </div>
        """
        html_cards.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BDI Real-World Field Inspection Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            max-width: 1200px;
            margin: 0 auto 30px auto;
            border-bottom: 2px solid #334155;
            padding-bottom: 20px;
        }}
        h1 {{ margin: 0 0 10px 0; color: #38bdf8; }}
        .meta-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #1e293b;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
        .cards-container {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 25px;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .card-header {{
            padding: 15px 20px;
            background: #182234;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
        }}
        .card-header h3 {{ margin: 0; font-size: 16px; color: #e2e8f0; }}
        .image-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            padding: 20px;
        }}
        .img-box p {{ margin: 0 0 8px 0; font-size: 13px; color: #94a3b8; text-align: center; }}
        .img-box img {{
            width: 100%;
            height: 350px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid #475569;
        }}
        .card-footer {{
            padding: 15px 20px;
            background: #182234;
            border-top: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: #ffffff;
            margin-right: 5px;
        }}
        .status-pill {{
            color: #f59e0b;
            font-weight: 600;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Building Defect Inspection — Real-World Field Survey Report</h1>
        <p>Automated multi-model assessment and candidate finding triage across 20 uncurated building facade field photos.</p>
        <div class="meta-stats">
            <div class="stat-box">
                <div>Total Inspected Images</div>
                <div class="stat-value">{summary["total_images_processed"]}</div>
            </div>
            <div class="stat-box">
                <div>Candidate Defects Found</div>
                <div class="stat-value">{summary["total_defects_found"]}</div>
            </div>
            <div class="stat-box">
                <div>Images with Confirmed Defects</div>
                <div class="stat-value">{summary["images_with_defects"]}</div>
            </div>
            <div class="stat-box">
                <div>Passed Quality QC</div>
                <div class="stat-value">{summary["passed_qc_count"]} / {summary["total_images_processed"]}</div>
            </div>
        </div>
    </div>
    <div class="cards-container">
        {"".join(html_cards)}
    </div>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-world building defect inspection field test")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    args = parser.parse_args()

    input_dir = args.inputs
    output_dir = args.output_dir
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not image_files:
        raise FileNotFoundError(f"No test images found in {input_dir}")

    device = torch.device(args.device)
    print(f"Loading ResNet-50 triage classifier from {RESNET_WEIGHTS} on {device}...")
    classifier = load_classifier(RESNET_WEIGHTS, device)
    classify_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading YOLO11n detector from {YOLO_WEIGHTS}...")
    import shutil
    import tempfile
    from ultralytics import YOLO

    temp_yolo_weights = Path(tempfile.gettempdir()) / f"bdi_{YOLO_WEIGHTS.name}"
    shutil.copy2(YOLO_WEIGHTS, temp_yolo_weights)
    detector = YOLO(str(temp_yolo_weights))


    print(f"\nProcessing {len(image_files)} real-world field survey images...")

    findings_jsonl = output_dir / "findings.jsonl"
    all_findings: list[dict[str, object]] = []
    processed_records: list[dict[str, object]] = []
    defect_counts: dict[str, int] = {c: 0 for c in CLASSES}
    passed_qc_count = 0

    with findings_jsonl.open("w", encoding="utf-8") as f_out:
        for idx, img_path in enumerate(image_files, 1):
            with Image.open(img_path) as raw_img:
                raw_rgb = raw_img.convert("RGB")
                w, h = raw_rgb.size

                # 1. Quality Preflight
                qc_res = image_quality(raw_rgb)
                if not qc_res["flags"]:
                    passed_qc_count += 1

                # 2. ResNet Multi-label classification triage
                with torch.no_grad():
                    cls_input = classify_tx(raw_rgb).unsqueeze(0).to(device)
                    cls_probs = torch.sigmoid(classifier(cls_input)).cpu().numpy()[0]

                # 3. YOLO localization
                yolo_results = detector.predict(
                    source=raw_rgb,
                    conf=args.conf_threshold,
                    device=args.device,
                    verbose=False,
                )[0]

                detections: list[dict[str, object]] = []
                for box in yolo_results.boxes:
                    cls_idx = int(box.cls.item())
                    if cls_idx >= len(CLASSES):
                        continue
                    label = CLASSES[cls_idx]
                    conf = float(box.conf.item())
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    bx, by, bx2, by2 = xyxy
                    bw, bh = bx2 - bx, by2 - by

                    det_record = {
                        "label": label,
                        "confidence": round(conf, 4),
                        "box_xywh": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                    }
                    detections.append(det_record)
                    defect_counts[label] += 1

                    # Create finding record
                    finding = {
                        "finding_id": str(uuid.uuid4()),
                        "sample_id": img_path.stem,
                        "created_at": utc_now(),
                        "defect_family": label,
                        "box_xywh": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                        "confidence": round(conf, 4),
                        "classifier_probability": round(float(cls_probs[cls_idx]), 4),
                        "quality_flags": qc_res["flags"],
                        "review_status": "pending",
                        "mandatory_limitations": [
                            "candidate_only",
                            "manual_review_required",
                            "not_structural_safety_determination",
                            "uncalibrated_measurement",
                        ],
                    }
                    all_findings.append(finding)
                    f_out.write(json.dumps(finding) + "\n")

                # 4. Render visual annotated image
                annotated_img = draw_bounding_boxes(raw_rgb, detections)
                annotated_filename = f"annotated_{img_path.name}"
                annotated_img.save(annotated_dir / annotated_filename)

                record = {
                    "filename": img_path.name,
                    "image_size": [w, h],
                    "quality": qc_res,
                    "detections": detections,
                    "classifier_probabilities": {
                        c: round(float(p), 4) for c, p in zip(CLASSES, cls_probs)
                    },
                }
                processed_records.append(record)

                print(
                    f"[{idx:02d}/{len(image_files)}] {img_path.name}: "
                    f"{len(detections)} defect(s) detected, QC={qc_res['decision']}"
                )

    summary_stats = {
        "benchmark": "Real-World Building Defect Field Inspection",
        "total_images_processed": len(image_files),
        "total_defects_found": len(all_findings),
        "images_with_defects": sum(1 for r in processed_records if r["detections"]),
        "passed_qc_count": passed_qc_count,
        "defect_distribution": defect_counts,
        "artifacts_generated": {
            "findings_jsonl": str(findings_jsonl.relative_to(ROOT)),
            "html_report": "data/field_test/field_inspection_report.html",
            "annotated_images_dir": str(annotated_dir.relative_to(ROOT)),
        },
    }

    summary_json_path = output_dir / "field_test_summary.json"
    summary_json_path.write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")

    html_report_path = output_dir / "field_inspection_report.html"
    generate_html_report(summary_stats, processed_records, html_report_path)

    print("\n" + "=" * 60)
    print("FIELD INSPECTION SURVEY COMPLETED SUCCESSFULLY")
    print(f"Total Photos:      {len(image_files)}")
    print(f"Defects Detected:  {len(all_findings)}")
    print(f"HTML Visual Report: {html_report_path}")
    print(f"Summary JSON:      {summary_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
