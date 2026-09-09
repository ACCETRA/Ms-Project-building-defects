# Model Comparison and Feasibility Matrix

**Experiment plan ID:** BDI-EXP-001  
**Status:** Provisional model names frozen for feasibility; final freeze follows local hardware tests

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

YOLO26n may be evaluated later as a product-improvement experiment, but it is not part of the first baseline matrix. Holding it back prevents a new model generation from expanding the initial experiment count before the basic pipeline is proven.

If the advisor confirms that “REZNEK” meant RetinaNet rather than ResNet, replace the classification row with a torchvision RetinaNet detector and revise the comparison contract before training.

## 2. Comparison tracks

### Track A: fast detection

- Florence-2 Base-FT versus YOLO11n detection.
- Same selected images, canonical labels, group-safe split, tiling policy, and box evaluation.
- Report AP50, mAP50:95, per-class precision/recall, crack recall at operating threshold, latency, and VRAM.

### Track B: Florence cascade

- Florence-2 Base alone versus Base→Large on validation-defined uncertain cases.
- Report accuracy gain, error recovery, escalation rate, latency, and peak memory.
- A cascade is retained only if Large recovers meaningful errors without an unacceptable compute penalty.

### Track C: segmentation

- YOLO11n-seg autonomous masks.
- YOLO11n detection boxes→SAM 2.1 Tiny masks.
- For SAM evaluation, use the same detector boxes; also report an oracle-box diagnostic using ground-truth boxes to separate detector errors from segmenter errors.
- Report Dice/IoU, mask AP where compatible, boundary quality, crack recall, latency, and VRAM.

### Track D: classification

- ResNet-50 multi-label baseline versus an equivalent image/crop-level Florence classification/triage result.
- Report per-class precision/recall/F1, macro/micro F1, PR-AUC, and calibration.
- Do not compare classifier F1 directly with detector mAP or mask IoU.

### Track E: end-to-end beta route

- Best defensible Florence-based route versus best specialist route (`YOLO→SAM` or YOLO segmentation).
- Report complete image-to-reviewed-finding latency, crack false-negative rate, false alarms per image, mask quality, escalation/review rate, and peak VRAM.

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

## 6. Licensing gate

Ultralytics software/models require an AGPL-compatible open-source project or an appropriate commercial license for proprietary/commercial deployment. CODEBRIM, DACL10K, and S2DS impose non-commercial/academic restrictions. CUBIT-InSeg states CC BY 4.0 and CiF states CDLA-Permissive-2.0. Final release and model-training provenance must be reviewed against the exact source terms.
