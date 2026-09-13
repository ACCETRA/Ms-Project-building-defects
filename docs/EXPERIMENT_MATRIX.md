# Model Comparison and Feasibility Matrix

**Experiment plan ID:** BDI-EXP-001  
**Status:** Model matrix approved; 5-epoch comparison feasibility runs complete; full-data training and evaluation remain

The authoritative remaining-work and ownership plan is [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md). Current checkpoints prove execution only. Full comparison results require frozen v1 task views, full-data runs, YOLO-to-SAM inference, and locked-test metrics.

## 1. Core model candidates

| Role | Primary candidate | Reason | Initial local status |
|---|---|---|---|
| Florence detector/triage | `florence-community/Florence-2-base-ft` | Native-Transformers conversion of Microsoft's 0.23B task-tuned checkpoint | FP16 CUDA preflight passed; broader feasibility pending |
| Florence escalation | `florence-community/Florence-2-large-ft` | Native-Transformers conversion of Microsoft's 0.77B task-tuned checkpoint | FP16 CUDA preflight passed; inference only for now |
| Specialist detector | Ultralytics `yolo11n.pt` | Stable nano baseline, 2.6M parameters, 640-pixel reference input | FP16 CUDA preflight passed at 512 px |
| Autonomous segmentation | Ultralytics `yolo11n-seg.pt` | Same model family and size tier as detector | FP16 CUDA preflight passed at 512 px |
| Prompted segmentation | `facebook/sam2.1-hiera-tiny` | 31.4M image-model subset loaded from the official SAM 2.1 Tiny checkpoint | FP16 1024 px box-prompt preflight passed; no fine-tuning |
| Classification baseline | Torchvision ResNet-50 with current default ImageNet weights | Recognizable, reproducible ResNet baseline | FP16 CUDA preflight passed at 224 px |

The Florence runtime identifiers intentionally use the `florence-community` conversions. Hugging Face identifies these as the official Transformers-converted Microsoft checkpoints. The original `microsoft/Florence-2-*-ft` repositories use custom code and do not load correctly through the current native processor; their original snapshots are retained locally for provenance, not used as the runtime artifacts.

YOLO26n is outside the accepted FYP scope; YOLO11n is the evaluated baseline.

ResNet-50 remains the classification baseline. RetinaNet is deferred because Florence and YOLO already cover the detector comparison track.

## 2. Comparison tracks

The project has two explicit pipelines:

1. **Full pipeline:** Florence-2 Base/Large detection/triage with segmentation.
2. **Comparison pipeline:** ResNet-50 classification + YOLO11n detection + SAM 2.1 Tiny segmentation.
3. Build the local web demonstration around real outputs from both pipelines.
4. Defer broader ablations, field validation, physical measurement validation, and production features.

The Florence segmentation adapter must convert Florence-localized regions into segmentation prompts or an equivalent segmentation stage and preserve the originating Florence model decision in the finding record. The comparison route uses YOLO11n boxes as prompts for SAM 2.1 Tiny.

Training entry points:

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_comparison.py --task detect --epochs 20
.\.venv\Scripts\python.exe scripts\train_yolo_comparison.py --task segment --epochs 20
.\.venv\Scripts\python.exe scripts\train_resnet_comparison.py --epochs 10
```

These defaults create a deterministic group-safe validation split from the current manifest. The current 300-sample manifest is the immediate FYP feasibility/demo training set, not a final accuracy benchmark; use a larger frozen manifest for reported comparison results. SAM 2.1 is used pretrained with YOLO boxes for the first comparison and is not fine-tuned in this short run.

### Completed 5-epoch feasibility training results (2026-09-12)

**YOLO11n detection** — 240 train / 60 val, 512 px, batch 1, FP32 CUDA, AMP disabled (Quadro T2000 stability), seed 20260911

| Epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1 | 1.946 | 4.218 | 1.903 | 0.0091 | 0.0046 |
| 2 | 2.015 | 4.195 | 1.881 | 0.0142 | 0.0060 |
| 3 | 1.988 | 4.098 | 1.897 | 0.0073 | 0.0030 |
| 4 | 2.009 | 4.095 | 1.917 | 0.0115 | 0.0038 |
| 5 | 1.834 | 4.007 | 1.830 | 0.0159 | 0.0049 |

Checkpoints: `runs/comparison/yolo/yolo11n_detection_fyp/weights/best.pt` and `last.pt` (5.4 MB each). Ultralytics' Windows post-training validation step failed due to a path-sanitizer bug on paths containing apostrophes; the training checkpoint and metrics CSV are unaffected.

**ResNet-50 classification** — 240 train / 60 val, 224 px, batch 4, FP32 CUDA, LR 1e-4, seed 20260911

| Epoch | train_loss | val_loss |
|---|---|---|
| 1 | 0.5050 | 0.3695 |
| 2 | 0.3491 | 0.3364 |
| 3 | 0.2757 | 0.3285 |
| 4 | 0.2343 | 0.3430 |
| 5 | 0.2055 | 0.3125 |

Checkpoint: `runs/comparison/resnet50_fyp/resnet50_comparison.pt` (90 MB). A previous FP16 run produced NaN losses on the Quadro T2000; the corrected FP32 run produced finite losses throughout. These losses are feasibility evidence, not final reported metrics.

**YOLO11n-seg segmentation** — 240 train / 60 val (146 polygon labels in train, 35 in val), 512 px, batch 1, FP32 CUDA, AMP disabled, seed 20260911

| Epoch | box_loss | seg_loss | cls_loss | Box mAP50 | Mask mAP50 |
|---|---|---|---|---|---|
| 1 | 1.915 | 3.879 | 4.954 | 0 | 0 |
| 2 | 1.897 | 3.523 | 4.896 | 0 | 0 |
| 3 | 1.871 | 3.599 | 4.799 | 0 | 0 |
| 4 | 1.981 | 3.516 | 4.709 | 0 | 0 |
| 5 | 1.808 | 3.246 | 4.825 | 0.0011 | 0.000007 |

Checkpoints: `runs/comparison/yolo/yolo11n_seg_fyp/weights/best.pt` and `last.pt` (6.0 MB each). Near-zero mAP is expected at 5 epochs on 146 polygon samples; the seg_loss trend downward (3.88 → 3.25) confirms the model is learning mask geometry. The Ultralytics `final_eval` step raised the same Windows path-sanitizer `FileNotFoundError` as the detection run; the script now catches it, verifies the real-path weights, and exits cleanly (exit code 0).

### Track A: fast detection

- Full pipeline: Florence-2 Base-FT with Florence-2 Large escalation versus comparison pipeline YOLO11n detection.
- Same selected images, canonical labels, group-safe split, tiling policy, and box evaluation.
- Report AP50, mAP50:95, per-class precision/recall, crack recall at operating threshold, latency, and VRAM.

### Track B: Florence cascade

- Florence-2 Base alone versus Base→Large on validation-defined uncertain cases.
- Report accuracy gain, error recovery, escalation rate, latency, and peak memory.
- A cascade is retained only if Large recovers meaningful errors without an unacceptable compute penalty.

### Track C: segmentation

- Full pipeline: Florence-localized regions followed by segmentation.
- Comparison pipeline: YOLO11n detection boxes followed by SAM 2.1 Tiny masks.
- Report an oracle-box diagnostic for the comparison pipeline to separate detector errors from segmenter errors.
- Report Dice/IoU, mask AP where compatible, boundary quality, crack recall, latency, and VRAM.

### Track D: classification

- ResNet-50 multi-label baseline versus an equivalent image/crop-level Florence classification/triage result.
- Report per-class precision/recall/F1, macro/micro F1, PR-AUC, and calibration.
- Do not compare classifier F1 directly with detector mAP or mask IoU.

### Track E: full versus comparison web demonstration

- Connect both the Florence full pipeline and the ResNet + YOLO + SAM comparison pipeline to a local browser workflow only after real outputs exist.
- Demonstrate upload, inference, result display, manual review, and export.
- Do not present the demonstration as a production inspection system.

## 3. SAM adaptation ladder

1. Ground-truth-box→SAM oracle diagnostic.
2. YOLO-box→SAM without fine-tuning.
3. Mask-selection, box expansion, tile reconciliation, and post-processing rules calibrated on validation data.
4. Only if needed: frozen encoder with cached embeddings and a trainable mask decoder.
5. Compare the tuned decoder against the untouched checkpoint; retain only measured improvement.

No result should say SAM “detects cracks.” The detector assigns the class and prompt; SAM proposes the mask.

Initial SAM prompt inference uses the built-in Transformers `Sam2Model` path on Windows. Use Meta's repository under WSL/Linux only if later decoder tuning or repository-specific features require it; the optional custom CUDA extension is not part of the first feasibility gate.

## 4. Local feasibility protocol

### Completed one-image CUDA preflight

All six candidates executed on the Quadro T2000 using the same hash-verified training-only CiF sample. Warm single-image times were approximately 0.044 s (YOLO11n), 0.051 s (YOLO11n-seg), 0.028 s (ResNet-50), 0.737 s (SAM 2.1 Tiny), 1.262 s (Florence-2 Base FT), and 4.487 s (Florence-2 Large FT). Peak allocated VRAM ranged from approximately 52 MB to 1.90 GB. These numbers prove execution only: they are not throughput, accuracy, or model-selection evidence. The machine-readable records are in `artifacts/model-feasibility/summary.json` and `.csv`.

### Representative feasibility study still required

Use 200–500 representative images after the first manifest draft. Test 512 and 640 input sizes where supported, batch size 1 first, and mixed precision only after checking numerical support.

For each model record:

- exact checkpoint and software commit/version;
- image and tile resolution;
- dtype and batch size;
- cold-start and warmed latency;
- peak allocated/reserved GPU memory;
- CPU RAM and disk cache impact;
- successful, out-of-memory, or fallback status;
- output schema compatibility and a small qualitative error review.

## 5. Stop/fallback rules

- If a model cannot run reliably within 4 GB after batch-size and resolution reduction, do not distort the entire evaluation to force it locally; mark it for optional larger-GPU evaluation.
- If 512 resizing erases thin cracks, use overlapping tiles rather than declaring the model incapable from a destructive resize.
- If a stage adds no statistically/operationally meaningful value, remove it from the beta while retaining the comparison result.
- Keep test partitions inaccessible during model and threshold selection.
- For CUBIT, use the recorded exact-deduplicated split views for model selection and the primary public score. If the full publisher test is also reported, label it separately and disclose the official split's exact-duplicate contamination.

## 6. Licensing gate

Ultralytics software/models require an AGPL-compatible open-source project or an appropriate commercial license for proprietary/commercial deployment. CODEBRIM, DACL10K, and S2DS impose non-commercial/academic restrictions. CUBIT-InSeg states CC BY 4.0 and CiF states CDLA-Permissive-2.0. Final release and model-training provenance must be reviewed against the exact source terms.
