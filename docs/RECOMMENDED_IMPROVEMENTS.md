# Recommended Improvements

**FYP status:** Complete  
**Purpose:** Optional recommendations for improving accuracy and generalization after the accepted FYP demonstration

These recommendations do not block the completed FYP. They explain how a later research run could improve performance on new buildings and external datasets while preserving the current evidence.

## Current evidence

The strongest directly comparable result is YOLO11n detection on the 701-image locked CUBIT-InSeg test: 89.997% precision, 64.198% recall, 63.165% mAP@50, and 54.107% mAP@50:95. The segmentation route reached 58.341% mask mAP@50 and 40.496% mask mAP@50:95 on the same test.

ResNet-50 reached 99.330% micro-F1 and 99.374% micro PR-AUC on the 1,104-image group-safe validation split. This is not a universal 99% accuracy claim. The official 632-image CODEBRIM test produced 36.641% micro-F1 and 36.856% micro PR-AUC, showing that performance falls when the image and label distribution changes.

External segmentation results show the same domain-shift problem. Mask mAP@50 was 2.462% on CiF, 0.698% on S2DS, 0.339% on DACL10K, and 0.0047% on UAV75. These values must remain separate because the sources use different imagery, annotation rules, and label semantics.

## Highest value improvements

### Collect representative target-site labels

Create a reviewed dataset from the same cameras, distances, materials, lighting, resolutions, and defect severities expected in use. Annotate every visible target defect consistently. Split data by building, flight, video, or capture session so related frames and near duplicates cannot cross from training into validation or test.

This is the most important improvement because the current CUBIT result is much stronger than the external-source results. That difference indicates that domain match matters more than changing architecture alone.

### Increase minority-class support

The ResNet validation split contains 590 crack positives and 515 spalling positives, but only 12 rust-staining positives, 2 efflorescence positives, 1 honeycombing positive, and no exposed-rebar positives. Add reviewed positives for the rare classes and ensure each class has meaningful support in validation and the locked test.

Report per-class precision, recall, F1, PR-AUC, and support. Do not use a high micro-average as evidence that every class performs equally well.

### Harmonize labels before combining datasets

Maintain a written mapping for every native label. Do not silently treat rock pockets as honeycombing, rust staining as exposed rebar, or binary foreground as a six-class mask. Keep weak or ambiguous mappings out of strict class-level evaluation.

### Improve thin-defect segmentation

Use high-resolution or overlapping tiles for fine cracks, retain narrow polygons during resizing, and audit masks after conversion. Compare mask loss, tile size, overlap, and augmentation settings through controlled validation experiments. Keep the locked test untouched until the final configuration is frozen.

### Calibrate thresholds on validation only

Select confidence thresholds separately for detection, segmentation, and classification using the validation split. Freeze those thresholds before locked-test evaluation. Include precision-recall curves and per-class operating points when false negatives and false positives have different costs.

### Use error review to select training changes

Record false positives, false negatives, small or thin defects, background texture confusion, truncation, blur, and occlusion. Change the data or training configuration only when a repeated error category supports the change. Keep reviewer decisions separate from immutable model predictions.

### Measure improvements against fixed baselines

For every optional experiment, retain the current checkpoints and evaluate the candidate with the same split, evaluator, IoU definition, and metric names. Report both mAP@50 and mAP@50:95 for localization, and report mask metrics separately from box metrics.

## Recommended acceptance rule for an improved model

Replace a current checkpoint only when the candidate improves the intended validation metric without materially damaging minority-class recall, passes the same regression tests, and is evaluated once on an untouched locked test. A result on a different dataset or a different metric is not a direct improvement comparison.

## Claims that remain appropriate

- The completed FYP runs real local classification, detection, segmentation, and structured vision-language inference.
- YOLO11n detection achieved 63.165% mAP@50 on the 701-image locked CUBIT test.
- ResNet-50 achieved 99.330% micro-F1 on group-safe in-domain validation.
- Cross-domain results are materially lower and are reported separately.
- The system is an academic demonstration and does not make structural-safety determinations.

