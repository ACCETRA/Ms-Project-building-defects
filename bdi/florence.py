"""CUDA-only Florence-2 phrase-grounding runtime for defect candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat


TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"
GROUNDING_TEXT = (
    "A concrete or masonry surface with a crack, spalling, a honeycombing rock "
    "pocket, exposed rebar, rust staining, or efflorescence leaching."
)


@dataclass(frozen=True)
class Detection:
    defect_family: str
    subtype: str
    box_xyxy: list[float]
    confidence_proxy: float
    raw_label: str


def canonicalize_label(label: str) -> tuple[str, str] | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    if "crack" in normalized:
        if "alligator" in normalized or "network" in normalized:
            return "crack", "network_alligator"
        if "deposit" in normalized or "precipitation" in normalized:
            return "crack", "with_deposit"
        return "crack", "unspecified"
    if "spall" in normalized:
        return "spalling", "not_applicable"
    if "honeycomb" in normalized or "rock pocket" in normalized or "rockpocket" in normalized:
        return "honeycombing_rock_pocket", "not_applicable"
    if "exposed rebar" in normalized or "exposed reinforcement" in normalized:
        return "exposed_rebar", "not_applicable"
    if "rust" in normalized or "corrosion stain" in normalized:
        return "rust_staining", "not_applicable"
    if "efflorescence" in normalized or "leaching" in normalized:
        return "efflorescence_leaching", "not_applicable"
    return None


def image_quality(image: Image.Image) -> dict[str, Any]:
    """Deterministic warnings only; thresholds are feasibility defaults."""
    gray = image.convert("L")
    mean_brightness = float(ImageStat.Stat(gray).mean[0])
    edge_variance = float(np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var())
    flags: list[str] = []
    if min(image.size) < 512:
        flags.append("low_resolution")
    if mean_brightness < 35:
        flags.append("underexposed")
    elif mean_brightness > 220:
        flags.append("overexposed")
    if edge_variance < 45:
        flags.append("blur")
    decision = "accepted_with_warning" if flags else "accepted"
    notes = None
    if flags:
        notes = (
            "Automated feasibility warning; thresholds require target-domain calibration. "
            f"mean_luma={mean_brightness:.2f}, edge_variance={edge_variance:.2f}"
        )
    return {"decision": decision, "flags": flags, "notes": notes}


def box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate(detections: list[Detection], threshold: float = 0.85) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence_proxy, reverse=True):
        if any(
            detection.defect_family == other.defect_family
            and box_iou(detection.box_xyxy, other.box_xyxy) >= threshold
            for other in kept
        ):
            continue
        kept.append(detection)
    return kept


class FlorenceRunner:
    """Loads one Florence checkpoint and provides deterministic generation."""

    def __init__(self, checkpoint: Path, model_id: str, max_new_tokens: int = 128):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval().to("cuda")
        if next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("Florence silently failed to load on CUDA")

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        self.torch.cuda.empty_cache()

    def infer(self, image: Image.Image) -> dict[str, Any]:
        torch = self.torch
        prompt = TASK_PROMPT + GROUNDING_TEXT
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            "cuda", torch.float16
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        confidence_proxy = self._sequence_likelihood(output)
        sequence = output.sequences.detach().cpu()
        decoded = self.processor.batch_decode(sequence, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            decoded, task=TASK_PROMPT, image_size=image.size
        )
        payload = parsed.get(TASK_PROMPT, {}) if isinstance(parsed, dict) else {}
        boxes = payload.get("bboxes", []) if isinstance(payload, dict) else []
        labels = payload.get("labels", []) if isinstance(payload, dict) else []
        detections: list[Detection] = []
        for box, label in zip(boxes, labels, strict=False):
            mapped = canonicalize_label(str(label))
            if mapped is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            values = [float(item) for item in box]
            if not all(math.isfinite(item) for item in values):
                continue
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            family, subtype = mapped
            detections.append(
                Detection(family, subtype, values, confidence_proxy, str(label))
            )
        detections = deduplicate(detections)
        return {
            "model_id": self.model_id,
            "task": TASK_PROMPT,
            "grounding_text": GROUNDING_TEXT,
            "decoded": decoded,
            "sequence_likelihood_proxy": round(confidence_proxy, 6),
            "confidence_kind": "generated_token_geometric_mean_not_calibrated_probability",
            "elapsed_seconds": round(elapsed, 4),
            "detections": [
                {
                    "defect_family": item.defect_family,
                    "subtype": item.subtype,
                    "box_xyxy": item.box_xyxy,
                    "confidence_proxy": round(item.confidence_proxy, 6),
                    "raw_label": item.raw_label,
                }
                for item in detections
            ],
        }

    def _sequence_likelihood(self, output: Any) -> float:
        """Geometric mean generated-token likelihood, used only as a routing proxy."""
        scores = list(output.scores or [])
        if not scores:
            return 0.0
        token_ids = output.sequences[:, -len(scores) :]
        log_probabilities = []
        for index, logits in enumerate(scores):
            token = token_ids[:, index].to(logits.device)
            value = logits.float().log_softmax(dim=-1).gather(1, token[:, None]).mean()
            log_probabilities.append(float(value.detach().cpu()))
        return max(0.0, min(math.exp(sum(log_probabilities) / len(log_probabilities)), 1.0))
