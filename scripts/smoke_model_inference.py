"""Run one CUDA-only model smoke test on a training-only feasibility image.

This is a hardware/runtime gate. It is not an accuracy evaluation and does not
alter weights, thresholds, data, or annotations.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"
OUTPUT_ROOT = ROOT / "artifacts" / "model-feasibility"
MODEL_IDS = (
    "yolo11n",
    "yolo11n_seg",
    "resnet50",
    "sam2_1_hiera_tiny",
    "florence_2_base_ft",
    "florence_2_large_ft",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_sample(sample_id: str | None) -> dict[str, str]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if sample_id is None or row["sample_id"] == sample_id]
    if not matches:
        raise ValueError(f"Sample not found in feasibility manifest: {sample_id}")
    sample = matches[0]
    if sample["split"] != "train":
        raise ValueError("Smoke inference is restricted to training-only samples.")
    image_path = ROOT / sample["output_image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    actual_hash = file_sha256(image_path)
    if actual_hash != sample["sha256"]:
        raise ValueError(f"Image hash mismatch for {sample['sample_id']}")
    return sample


def timed_cuda(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = call()
    torch.cuda.synchronize()
    return output, time.perf_counter() - started


def run_yolo(model_id: str, image_path: Path, image_size: int) -> dict[str, Any]:
    from ultralytics import YOLO

    filename = "yolo11n.pt" if model_id == "yolo11n" else "yolo11n-seg.pt"
    model_path = ROOT / "weights" / "yolo" / filename
    previous_directory = Path.cwd()
    try:
        os.chdir(model_path.parent)
        model = YOLO(model_path.name)
    finally:
        os.chdir(previous_directory)

    source = str(image_path.relative_to(ROOT))

    def infer() -> Any:
        return model.predict(
            source=source,
            imgsz=image_size,
            device=0,
            half=True,
            batch=1,
            verbose=False,
        )

    cold_output, cold_seconds = timed_cuda(infer)
    warm_output, warm_seconds = timed_cuda(infer)
    result = warm_output[0]
    parameter_device = model.predictor.device.type
    if parameter_device != "cuda":
        raise RuntimeError(f"YOLO silently used {parameter_device} instead of CUDA")
    return {
        "checkpoint": f"weights/yolo/{filename}",
        "dtype": "float16",
        "requested_image_size": image_size,
        "cold_inference_seconds": round(cold_seconds, 4),
        "warm_inference_seconds": round(warm_seconds, 4),
        "output": {
            "detections": int(len(result.boxes)) if result.boxes is not None else 0,
            "has_masks": result.masks is not None,
        },
    }


def run_resnet(image: Image.Image) -> dict[str, Any]:
    from torchvision.models import ResNet50_Weights, resnet50

    checkpoint = ROOT / "weights" / "resnet" / "resnet50-11ad3fa6.pth"
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device="cuda", dtype=torch.float16)
    tensor = ResNet50_Weights.DEFAULT.transforms()(image).unsqueeze(0).to("cuda", torch.float16)

    def infer() -> torch.Tensor:
        with torch.inference_mode():
            return model(tensor)

    cold_output, cold_seconds = timed_cuda(infer)
    warm_output, warm_seconds = timed_cuda(infer)
    return {
        "checkpoint": "weights/resnet/resnet50-11ad3fa6.pth",
        "dtype": "float16",
        "requested_image_size": 224,
        "processed_tensor_shape": list(tensor.shape),
        "cold_inference_seconds": round(cold_seconds, 4),
        "warm_inference_seconds": round(warm_seconds, 4),
        "output": {
            "logit_shape": list(warm_output.shape),
            "imagenet_top1_index": int(warm_output.argmax(dim=-1).item()),
            "interpretation": "runtime_only_not_a_defect_prediction",
        },
    }


def first_cif_box(annotation_path: Path) -> list[float]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes = annotation["record"]["objects"]["bbox"]
    if not boxes:
        raise ValueError("Selected CiF sample has no prompt box")
    x, y, width, height = (float(value) for value in boxes[0])
    image_width = float(annotation["record"]["width"])
    image_height = float(annotation["record"]["height"])
    return [x, y, min(x + width, image_width - 1), min(y + height, image_height - 1)]


def run_sam(image: Image.Image, annotation_path: Path) -> dict[str, Any]:
    from transformers import Sam2Model, Sam2Processor

    checkpoint = ROOT / "weights" / "sam2.1-hiera-tiny"
    processor = Sam2Processor.from_pretrained(checkpoint, local_files_only=True)
    model = Sam2Model.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    prompt_box = first_cif_box(annotation_path)
    inputs = processor(
        images=image,
        input_boxes=[[prompt_box]],
        return_tensors="pt",
    ).to("cuda", torch.float16)

    def infer() -> Any:
        with torch.inference_mode():
            return model(**inputs, multimask_output=False)

    cold_output, cold_seconds = timed_cuda(infer)
    warm_output, warm_seconds = timed_cuda(infer)
    masks = processor.post_process_masks(
        warm_output.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
    )[0]
    return {
        "checkpoint": "weights/sam2.1-hiera-tiny",
        "dtype": "float16",
        "requested_image_size": 1024,
        "processed_tensor_shape": list(inputs["pixel_values"].shape),
        "prompt_type": "ground_truth_box_hardware_smoke_only",
        "prompt_box_xyxy": prompt_box,
        "cold_inference_seconds": round(cold_seconds, 4),
        "warm_inference_seconds": round(warm_seconds, 4),
        "output": {
            "mask_shape": list(masks.shape),
            "mask_pixels": int(masks.sum().item()),
            "iou_score": float(warm_output.iou_scores.detach().float().cpu().max().item()),
        },
    }


def run_florence(model_id: str, image: Image.Image) -> dict[str, Any]:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    directory = (
        "florence-community-2-base-ft"
        if model_id == "florence_2_base_ft"
        else "florence-community-2-large-ft"
    )
    checkpoint = ROOT / "weights" / directory
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = Florence2ForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    inputs = processor(text="<OD>", images=image, return_tensors="pt").to("cuda", torch.float16)

    def infer() -> torch.Tensor:
        with torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=32, num_beams=1, do_sample=False)

    cold_output, cold_seconds = timed_cuda(infer)
    warm_output, warm_seconds = timed_cuda(infer)
    decoded = processor.batch_decode(warm_output.detach().cpu(), skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(decoded, task="<OD>", image_size=image.size)
    labels = parsed.get("<OD>", {}).get("labels", []) if isinstance(parsed, dict) else []
    boxes = parsed.get("<OD>", {}).get("bboxes", []) if isinstance(parsed, dict) else []
    return {
        "checkpoint": f"weights/{directory}",
        "dtype": "float16",
        "requested_image_size": list(image.size),
        "processed_tensor_shape": list(inputs["pixel_values"].shape),
        "task_prompt": "<OD>",
        "max_new_tokens": 32,
        "cold_inference_seconds": round(cold_seconds, 4),
        "warm_inference_seconds": round(warm_seconds, 4),
        "output": {
            "generated_token_count": int(warm_output.shape[-1]),
            "parsed_box_count": len(boxes),
            "parsed_labels": labels[:10],
            "interpretation": "pretrained_runtime_smoke_not_defect_accuracy",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=MODEL_IDS)
    parser.add_argument("--sample-id", default="cif_0001")
    parser.add_argument("--image-size", type=int, default=512, choices=(512, 640))
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    if args.output_tag and not args.output_tag.replace("-", "").replace("_", "").isalnum():
        parser.error("--output-tag may contain only letters, numbers, hyphens, and underscores")

    sample = select_sample(args.sample_id)
    image_path = ROOT / sample["output_image"]
    annotation_path = ROOT / sample["output_annotation"]
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    record: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "cuda_hardware_smoke_no_training_not_accuracy",
        "model_id": args.model_id,
        "sample": {
            "sample_id": sample["sample_id"],
            "source_dataset": sample["source_dataset"],
            "split": sample["split"],
            "group_id": sample["group_id"],
            "image": sample["output_image"],
            "sha256": sample["sha256"],
            "width": image.width,
            "height": image.height,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        record["environment"].update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if args.model_id.startswith("yolo"):
            details = run_yolo(args.model_id, image_path, args.image_size)
        elif args.model_id == "resnet50":
            details = run_resnet(image)
        elif args.model_id == "sam2_1_hiera_tiny":
            details = run_sam(image, annotation_path)
        else:
            details = run_florence(args.model_id, image)
        record.update(details)
        record["total_seconds"] = round(time.perf_counter() - started, 4)
        record["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        record["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
        record["status"] = "passed"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        record["status"] = "out_of_memory" if "out of memory" in message.lower() else "failed"
        record["error"] = message
        if torch.cuda.is_available():
            record["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        image.close()
        gc.collect()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    output_path = OUTPUT_ROOT / f"{args.model_id}{suffix}.json"
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
