# SAM Decoder Training

The repository now includes `scripts/train_sam_decoder.py` for decoder-only SAM 2 training.

## Design

- SAM image encoder: frozen.
- SAM prompt encoder: frozen.
- SAM mask decoder: trainable.
- Prompt: ground-truth annotation box from the v1 segmentation manifest.
- Target: polygon rasterized on demand from the normalized annotation.
- Split: group-safe train/validation split using seed `20260911`.
- Loss: binary cross-entropy plus soft Dice loss.
- Metrics: validation loss, BCE, Dice, and IoU.
- Checkpoint: `sam_decoder_last.pt` containing model state, trainable-module declaration, history, and seed.
- Modes: `--dtype float32` for stable full decoder training on a larger GPU, or optional `--peft` LoRA adapters when the `peft` package is installed.
- Pair index: `pair_manifest.json`.

This trains the decoder to improve mask quality for the supplied box prompts. It does not train SAM to detect classes; YOLO or another detector still supplies the class and box.

## Inspect pairs without a GPU

```powershell
.\.venv\Scripts\python.exe scripts\train_sam_decoder.py `
  --inspect-only `
  --max-train-pairs 4 `
  --max-val-pairs 2 `
  --output-dir runs/sam/decoder_v1_inspect
```

## Train on a supported GPU

Use a larger NVIDIA GPU with enough VRAM for the image encoder forward pass and decoder gradients:

```powershell
.\.venv\Scripts\python.exe scripts\train_sam_decoder.py `
  --epochs 5 `
  --batch-size 1 `
  --dtype float32 `
  --learning-rate 1e-5 `
  --output-dir runs/sam/decoder_v1
```

Optional LoRA/PEFT mode:

```powershell
\.\.venv\Scripts\python.exe scripts\train_sam_decoder.py `
  --epochs 5 `
  --batch-size 1 `
  --dtype float16 `
  --peft `
  --lora-rank 4 `
  --output-dir runs/sam/decoder_v1_lora
```

Install `peft` only in the project `.venv` before using `--peft`; the base adapter does not require it.

For an initial controlled run:

```powershell
.\.venv\Scripts\python.exe scripts\train_sam_decoder.py `
  --epochs 1 `
  --max-train-pairs 32 `
  --max-val-pairs 16 `
  --output-dir runs/sam/decoder_v1_smoke
```

## Current workstation boundary

The Quadro T2000 has 4 GB VRAM. The decoder-only FP16 optimizer update became non-finite even on a one-pair smoke update, so this machine must not be used to claim a trained SAM decoder. The adapter aborts before writing a corrupted checkpoint. Run the actual training in FP32 or PEFT mode on a larger supported GPU.

## Validation rules

- Do not evaluate on locked test data during training or decoder selection.
- Keep YOLO detector thresholds fixed from validation-only selection.
- Compare prompted pretrained SAM against the trained decoder using the same boxes and held-out validation groups.
- Report Dice/IoU, mask AP where compatible, latency, peak VRAM, and failure cases separately by source.
- Keep manual review and the non-structural-safety limitation in every product report.
