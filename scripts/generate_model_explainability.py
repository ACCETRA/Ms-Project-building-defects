#!/usr/bin/env python3
"""Generate Grad-CAM visual explainability maps for ResNet-50 defect predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bdi.gradcam import GradCAM, apply_colormap_on_image

CLASSES = (
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching",
)

DEFAULT_CHECKPOINT = ROOT / "runs/comparison/resnet50_v1_queue/resnet50_comparison.pt"
DEFAULT_OUTPUT = ROOT / "runs/explainability"
MANIFEST_PATH = ROOT / "data/manifests/v1_classification_manifest.csv"


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def get_image_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_panel(
    original_img: Image.Image,
    cam_heatmap: np.ndarray,
    overlay_img: Image.Image,
    sample_id: str,
    target_class: str,
    prob: float,
    gt_labels: str,
) -> Image.Image:
    """Create a high-quality 3-column composite panel with diagnostic annotations."""
    size = 300
    orig_resized = original_img.resize((size, size), Image.BILINEAR)

    # Convert normalized heatmap to RGB Jet
    r = np.clip(1.5 - np.abs(cam_heatmap * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(cam_heatmap * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(cam_heatmap * 4.0 - 1.0), 0.0, 1.0)
    jet_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    cam_img = Image.fromarray(jet_rgb).resize((size, size), Image.BILINEAR)

    overlay_resized = overlay_img.resize((size, size), Image.BILINEAR)

    # Panel dimensions: 3 images wide + border + text banner
    header_height = 60
    panel_w = size * 3 + 40
    panel_h = size + header_height + 20
    panel = Image.new("RGB", (panel_w, panel_h), color=(250, 250, 252))
    draw = ImageDraw.Draw(panel)

    # Text header
    title_text = f"Sample: {sample_id}  |  Target Class: {target_class.upper()}  |  Model P({target_class}) = {prob*100:.1f}%"
    sub_text = f"Ground Truth: {gt_labels}  |  Grad-CAM Target Layer: ResNet-50 layer4[-1]"
    draw.text((15, 12), title_text, fill=(20, 20, 30))
    draw.text((15, 34), sub_text, fill=(80, 80, 100))

    # Paste images
    panel.paste(orig_resized, (10, header_height))
    panel.paste(cam_img, (size + 20, header_height))
    panel.paste(overlay_resized, (size * 2 + 30, header_height))

    # Sub-labels
    draw.text((15, header_height + size - 20), "Raw Inspection Image", fill=(255, 255, 255))
    draw.text((size + 25, header_height + size - 20), "Grad-CAM Activation Map", fill=(255, 255, 255))
    draw.text((size * 2 + 35, header_height + size - 20), "Attribution Overlay", fill=(255, 255, 255))

    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for defect models")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-class", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")  # CPU safe to run while GPU trains
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading ResNet-50 from {args.checkpoint} on {device}...")
    model = load_model(args.checkpoint, device)

    # Target the last bottleneck of layer4
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    transform = get_image_transform(224)

    # Load manifest and find representative samples for each class
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    selected_samples: list[tuple[dict[str, str], str, int]] = []
    for class_idx, class_name in enumerate(CLASSES):
        matching = [r for r in rows if class_name in r.get("canonical_labels", "").split("|")]
        for row in matching[: args.limit_per_class]:
            selected_samples.append((row, class_name, class_idx))

    print(f"Generating Grad-CAM explainability for {len(selected_samples)} cases across classes...")
    results_meta: list[dict[str, object]] = []

    try:
        for idx, (row, target_class, class_idx) in enumerate(selected_samples, 1):
            img_path = ROOT / row["output_image"]
            if not img_path.is_file():
                continue

            with Image.open(img_path) as raw_img:
                raw_rgb = raw_img.convert("RGB")
                tensor_input = transform(raw_rgb).unsqueeze(0).to(device)

                # Compute model output probabilities
                with torch.no_grad():
                    logits = model(tensor_input)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                # Generate heatmap
                heatmap = cam.generate_heatmap(tensor_input, target_class=class_idx)
                overlay = apply_colormap_on_image(raw_rgb.resize((224, 224)), heatmap, alpha=0.45)

                sample_id = row["sample_id"]
                target_prob = float(probs[class_idx])

                panel = create_panel(
                    original_img=raw_rgb,
                    cam_heatmap=heatmap,
                    overlay_img=overlay,
                    sample_id=sample_id,
                    target_class=target_class,
                    prob=target_prob,
                    gt_labels=row["canonical_labels"],
                )

                panel_filename = f"gradcam_{idx:02d}_{sample_id}_{target_class}.png"
                panel.save(output_dir / panel_filename)

                meta_record = {
                    "sample_id": sample_id,
                    "target_class": target_class,
                    "class_index": class_idx,
                    "model_probability": target_prob,
                    "all_probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
                    "ground_truth": row["canonical_labels"],
                    "image_path": str(img_path.relative_to(ROOT)),
                    "artifact_image": panel_filename,
                }
                results_meta.append(meta_record)
                print(f"[{idx}/{len(selected_samples)}] Saved {panel_filename} (prob={target_prob:.3f})")

        summary_file = output_dir / "explainability_summary.json"
        summary_file.write_text(json.dumps(results_meta, indent=2), encoding="utf-8")
        print(f"\nAll Grad-CAM panels generated in {output_dir}")

    finally:
        cam.remove_hooks()


if __name__ == "__main__":
    main()
