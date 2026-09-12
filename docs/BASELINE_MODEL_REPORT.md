# Baseline model report

## Scope

This report documents the validated baseline training results currently available in the project repository. The numbers below are for the 300-sample feasibility manifest and should be treated as a demo baseline, not proof of final dataset readiness or production accuracy.

## Executive grade

Overall grade: D

Reasoning:
- YOLO detection is barely above zero on the current 300-sample feasibility set.
- YOLO segmentation is effectively not learning in this short baseline.
- ResNet-50 reduces loss substantially, but it is still a weak benchmark on the limited feasibility set.
- Florence fine-tuning has only a single-step smoke run and cannot yet be scored as a meaningful accuracy result.

## Data and setup

- Manifest: `data/manifests/feasibility_v0_materialized.csv`
- Split: group-safe training/validation split with 240 train / 60 validation records for the comparison models
- Data policy: feasibility demo only; not a frozen v1 release
- Hardware: local CUDA-capable Windows workstation with Quadro T2000-class GPU

## Comparison model metrics

### 1) YOLO11n detection

Final epoch metrics from `runs/comparison/yolo/yolo11n_detect_fyp/results.csv`:
- Epoch 5 precision(B): 0.84588
- Epoch 5 recall(B): 0.03323
- Epoch 5 mAP50(B): 0.01561
- Epoch 5 mAP50-95(B): 0.00435
- Validation box loss: 2.71090
- Validation cls loss: 4.76202
- Validation dfl loss: 2.44248

Train loss progression:
- Epoch 1: 1.94612
- Epoch 2: 2.01572
- Epoch 3: 1.98399
- Epoch 4: 2.00069
- Epoch 5: 1.84613

Interpretation:
- Precision is high only because detections are sparse, but recall is near zero.
- The detector is not yet production-ready on this feasibility set.
- Grade: E for detection quality in the current baseline.

### 2) YOLO11n segmentation

Final epoch metrics from `runs/comparison/yolo/yolo11n_seg_fyp/results.csv`:
- Epoch 5 precision(B): 0.00095
- Epoch 5 recall(B): 0.24672
- Epoch 5 mAP50(B): 0.00120
- Epoch 5 mAP50-95(B): 0.00031
- Epoch 5 precision(M): 0.00015
- Epoch 5 recall(M): 0.02095
- Epoch 5 mAP50(M): 0.00007
- Epoch 5 mAP50-95(M): 0.00001
- Validation seg loss: 2.87738
- Train seg loss: 3.26179

Interpretation:
- Segmentation is not learning usable mask quality in the present run.
- This is not a valid final model choice.
- Grade: F for segmentation quality in this baseline.

### 3) ResNet-50 classification baseline

Loss values from `runs/comparison/resnet50_fyp/history.json`:
- Epoch 1 train loss: 0.50497, val loss: 0.36949
- Epoch 2 train loss: 0.34911, val loss: 0.33645
- Epoch 3 train loss: 0.27572, val loss: 0.32848
- Epoch 4 train loss: 0.23429, val loss: 0.34299
- Epoch 5 train loss: 0.20550, val loss: 0.31247

Interpretation:
- The model clearly improved during training.
- Loss is decreasing and stable, but no calibrated accuracy or per-class AP metrics were computed for this feasibility baseline.
- Grade: C for the current baseline; stronger than detection/segmentation, but not yet sufficient for a final research claim.

## Florence fine-tuning status

The Florence training entry point in `scripts/train_florence.py` was validated in smoke mode and one-step CUDA execution:
- Dry-run readiness check passed
- One-step fine-tune executed successfully on the feasibility manifest
- Validation loss at step 1: 7.8279

This is not enough to assign an accuracy grade because it is a smoke-only run with a single sample overfit and no end-to-end validation metric. The result is a technical validation of the training loop, not a performance result.

Grade for Florence baseline: provisional C for pipeline readiness, but not for accuracy.

## Summary table

| Model | Result | Grade | Notes |
|---|---:|---:|---|
| YOLO detection | mAP50 0.01561 | E | Barely learning on current demo set |
| YOLO segmentation | mAP50 0.00120 | F | Not usable in this baseline |
| ResNet-50 | val loss 0.31247 | C | Best of the three, but early-stage |
| Florence | smoke-test pass only | C (pipeline) | Accuracy not yet meaningful |

## Recommendation

1. Treat all numbers in this report as baseline-only evidence.
2. Do not present them as final model performance.
3. Freeze the v1 manifest before claiming a serious accuracy benchmark.
4. Run the longer training set on the approved v1 split before selecting a final model.
5. Use this report only as a checkpoint for engineering decisions and next-step planning.

## Files used

- `runs/comparison/yolo/yolo11n_detect_fyp/results.csv`
- `runs/comparison/yolo/yolo11n_seg_fyp/results.csv`
- `runs/comparison/resnet50_fyp/history.json`
- `runs/florence/finetune-smoke/history.json`
- `scripts/train_florence.py`
- `scripts/train_yolo_comparison.py`
- `scripts/train_resnet_comparison.py`

## Final assessment

The current project status is functional baseline evidence, not final model completion. The only honest overall grade for the available metrics is D, because the comparison models are not yet strong enough to support a credible final demonstration or real deployment claim.
