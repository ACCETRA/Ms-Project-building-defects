# Building Defect Inspection Beta - Roadmap v2

**Roadmap ID:** BDI-ROADMAP-002  
**Status:** Approved scope; preparation in progress  
**Updated:** 2026-09-08  
**Supersedes for execution:** `source/The RoadMap.md`  
**Original preserved:** yes

## 1. Outcome

Build a fully functional academic beta for visible-condition inspection of completed and under-construction buildings. It should look and behave like a small real product, not a collection of notebooks. A user must be able to create an inspection, upload RGB images, follow processing, review model findings, and export a traceable report.

Cracks are the primary target. Secondary visible targets are spalling, honeycombing/rock pockets, exposed rebar, rust/corrosion staining, and efflorescence/leaching. The beta identifies candidate visible conditions for qualified review. It does not declare a structure safe, diagnose hidden damage, prescribe repairs, or certify code compliance.

The first release uses handheld and UAV still images of visible concrete and masonry surfaces. Live video, operational thermal/GPR/3D fusion, and autonomous engineering conclusions are later phases.

## 2. Why the system uses small models

The smaller variants are a deliberate product choice, not a sign that the project is weak:

- Every uploaded image needs a fast, inexpensive first pass.
- The available local GPU is an NVIDIA Quadro T2000 with 4 GB VRAM.
- A staged system can reserve more expensive processing for positive or uncertain regions.
- The comparison is more defensible when models are evaluated under the same realistic compute budget.
- Product reliability depends on data quality, calibration, review, and failure handling as much as parameter count.

The project may use a larger model temporarily when it provides measurable value, but model size is not the goal. The selected system must meet recall, mask quality, latency, memory, and review-workload targets on the intended hardware or on an explicitly documented deployment target.

## 3. Product boundary

### Included

- Building facades, walls, slabs, beams, columns, ceilings, foundations, and visible masonry/concrete elements.
- JPEG/PNG RGB images from handheld cameras and UAVs.
- Single and batch upload.
- Image quality checks and honest rejection/warning states.
- Defect boxes, masks where supported, confidence, pixel measurements, review status, and provenance.
- Physical measurements only when a valid scale method exists.
- Human approval, rejection, relabelling, and escalation without overwriting the raw prediction.
- Reopenable inspections and JSON, CSV, annotated-image, and PDF-style exports.

### Deferred

- Thermal, GPR, 3D reconstruction, and historical-record fusion as an operational feature.
- Live video and temporal tracking.
- Hidden/internal damage diagnosis.
- Severity grades without engineer-approved rules and calibrated measurements.
- Repair recommendations and structural-safety decisions.

The structured finding schema already contains evidence links so future modalities can be attached without redesigning every visual output. In the current beta, only real RGB evidence is shown as integrated.

## 4. Data strategy: a curated mega dataset

The mega dataset is a versioned logical dataset, not an anonymous folder merge. Raw archives remain immutable. A manifest records each selected image, its source/version, parent/group, split, native labels, canonical labels, annotation type, mapping version, license, quality flags, selection reason, and content hash when materialized.

One registry produces different task-compatible views:

1. **Detection view:** native boxes plus boxes derived from genuine masks.
2. **Segmentation view:** genuine masks, polygons, or approved crack lines only.
3. **Classification view:** image/crop-level multi-label targets, including CODEBRIM.
4. **Product/field view:** intended building images, verified negatives, and difficult construction-site negatives.

An image-level label never becomes a claimed pixel mask. A box can prompt SAM, but SAM's output is a model-generated mask unless a human mask is available as ground truth. Weak and pseudo labels remain separately identified.

### Source roles

| Source | Role in this beta | Important restriction |
|---|---|---|
| CUBIT-InSeg | Primary public building-facade UAV crack/spalling source and locked public building test | Train/validation archives are still pending; do not tune on the official test images |
| CiF tiled | Large supplemental crack, spalling, and rust instance-segmentation/detection source | Group by original parent image; all available tiles are positive, so it cannot supply verified whole-image negatives |
| S2DS | Small multi-class segmentation source for cracks, spalling, corrosion appearance, and efflorescence | Do not present it as a large standalone benchmark |
| UAV75 | Tiny UAV crack/line-mask feasibility and stress source | Too small for the main product claim |
| CODEBRIM classification | ResNet and image-level multi-label baseline | No genuine masks; custom non-commercial terms; use portable 7-Zip for extraction |
| CODEBRIM original | Optional box-detection/classification transfer source | Bridge-domain; no segmentation claim |
| DACL10K | Capped bridge-domain transfer source and separately reported bridge robustness test | Not the main proof for a building product |
| Target-site data | Required final proof, hard negatives, and deployment calibration | Must have permission and whole buildings/sessions reserved for test |

### First curated release target

The provisional training universe is approximately 20,000-25,000 selected images/tiles before augmentation:

- CUBIT-InSeg: all official train/validation pairs when available; lock the official test set.
- CiF: up to 12,000 train and 2,500 validation tiles, sampled across classes and distinct parent images.
- S2DS: all 563 train and 87 validation pairs; lock all 93 test pairs.
- UAV75: all 50 train and 10 validation pairs; lock all 15 test pairs.
- DACL10K: cap training selection near 2,000 relevant official-train images; lock official validation as bridge stress evidence.
- CODEBRIM classification: retain official parent-safe train/validation/test partitions for the classifier track.
- Target site: aim for 500-1,000 reviewed images across at least three buildings/capture sessions, including verified negatives and hard negatives.

This is a starting budget, not a reason to include low-quality or redundant samples.

## 5. Taxonomy and annotation

The canonical top-level labels are:

- `crack` (primary), with optional `linear`, `network_alligator`, `with_deposit`, or `unspecified` subtype;
- `spalling`;
- `honeycombing_rock_pocket`;
- `exposed_rebar`;
- `rust_staining`;
- `efflorescence_leaching`;
- `no_visible_target_defect`;
- `unknown_review`.

Rust-colored evidence is recorded as rust staining, not proof of hidden reinforcement corrosion. Ambiguous data remains unknown instead of being forced into a class. Each image/class relationship distinguishes positive, verified negative, unknown, and not applicable.

A qualified civil/structural reviewer must approve the annotation guide, resolve edge cases, and review representative examples before dataset v0.1 is frozen.

## 6. Leakage, duplicates, and balancing

The order is mandatory:

1. identify parent images, buildings, structures, flights, and capture sessions;
2. detect exact and near duplicates;
3. assign complete groups to train, validation, and locked test partitions;
4. create tiles/crops only after group assignment;
5. profile class counts at image, instance, and pixel levels;
6. apply training-only sampling/augmentation experiments.

Cracks remain the majority because they are the primary task. The goal is not identical class counts. Start with an unweighted baseline, then compare controlled class-aware sampling, moderate minority augmentation, and task-appropriate weighted/focal/Dice/Tversky losses. Never oversample, augment, or rebalance validation/test data. Keep verified no-defect and hard-negative images so false alarms can be measured.

Hard negatives must include joints, formwork seams, tie holes, edges, conduits/cables, chalk/paint, shadows, wet patches, dust, scaffolding, texture, and occlusion.

## 7. Model comparison and final pipeline decision

YOLO, SAM, ResNet, and Florence do different jobs. They are not four interchangeable entries on one leaderboard.

### Track A - image triage and box detection

- Florence-2 Base FT (`florence-community/Florence-2-base-ft`), the native-Transformers conversion of Microsoft's checkpoint.
- YOLO11n detection (`yolo11n.pt`).
- ResNet-50 as an image/crop multi-label classifier baseline, not as a detector.

Compare on the same input policy and appropriate task view. Report crack recall, precision, PR-AUC/F1 for image triage, box AP50/mAP50:95 for detection, latency, and peak VRAM.

### Track B - segmentation

- YOLO11n-seg as an autonomous segmentation baseline.
- YOLO11n boxes passed to SAM 2.1 Hiera Tiny as prompts.

SAM does not know what a crack is and does not discover defects by itself. The adapter converts detector boxes into prompts, selects/merges masks, restores source-image coordinates, and records failures. Begin with prompt-only evaluation. Fine-tune a lightweight decoder only if prompt-only results are inadequate and the experiment fits the hardware/time budget.

### Track C - Florence escalation

- Florence-2 Base FT alone.
- Florence-2 Base FT followed by Florence-2 Large FT only for validation-defined positive/uncertain cases.

This ablation determines whether the large stage adds enough quality to justify memory, latency, and complexity. Florence-2 Large is not assumed to fit local fine-tuning on 4 GB VRAM.

### End-to-end decision

Compare the best Florence route with the best specialist route (likely YOLO -> SAM or YOLO segmentation) using crack recall, mask quality, false alarms per image, escalation rate, latency, peak VRAM, failure rate, and reviewer workload.

`REZNEK` is provisionally interpreted as **ResNet**. The advisor must confirm this. If the intended term was **RetinaNet**, it becomes another detector beside YOLO and changes the experiment matrix.

YOLO26n may be tested only after the YOLO11 baseline is reproducible; it must not expand the first matrix by default.

## 8. Image resolution and crack measurement

Thin cracks can disappear during resize. Each model track must test a documented full-image/overlapping-tile policy. Comparisons use the same source images and comparable resolution budgets.

The default output is pixel geometry. Millimetres/centimetres require one validated method:

- reference marker in the surface plane;
- camera and planar-surface calibration;
- trustworthy UAV ground-sample distance with acceptable viewing geometry; or
- registered 3D scale.

Physical measurement evaluation needs field-measured ground truth and reports error/uncertainty. A precise-looking mask does not automatically produce a valid physical crack width.

## 9. Validation strategy

Three kinds of evidence are reported separately:

1. **Development validation:** group-safe validation partitions for model/threshold selection.
2. **Public locked building test:** CUBIT-InSeg official test after the pipeline is frozen.
3. **Target-site field test:** entire unseen buildings/capture sessions, including verified negatives and image-quality failures.

DACL10K official validation is a separately labelled bridge-domain robustness test if DACL train contributes to training. It is not presented as the main building generalization score. CiF/S2DS/UAV official tests are reported by source and remain untouched during tuning.

Cross-dataset testing means a complete source/domain is excluded from training and threshold selection. It is not the same as randomly withholding a slice after all sources are mixed.

### Required metrics

- Classification/triage: precision, recall, F1, PR-AUC, confusion matrix, calibration.
- Detection: AP50, mAP50:95, and crack recall at the chosen operating threshold.
- Segmentation: class IoU/Dice and mask AP where the annotation supports it.
- Measurement: pixel error; physical MAE/RMSE only with calibrated field ground truth.
- Product: latency, throughput, peak VRAM, failure/rejection rate, escalation rate, false alarms per image, and review time.
- Reliability: per-source/per-class results, confidence intervals, and false-negative review.

## 10. Functional beta architecture

Recommended first deployment is a local, browser-based application on the project workstation:

- responsive web interface for inspection setup, upload, status, review, and export;
- API/service layer for projects, inspections, images, findings, and reviewer actions;
- background inference worker so large batches do not freeze the interface;
- local relational database for projects, states, reviews, and provenance;
- immutable source-image storage plus separate derived overlays/masks/reports;
- versioned model runner behind a stable finding-record interface.

The exact web framework can be selected at the implementation gate. The important boundary is that model execution is isolated from the product records and that failed inference, no-findings, rejected images, and uncalibrated measurements are first-class states.

### Required user flow

1. Create/reopen a project and inspection.
2. Record site, building, facade/floor/zone, and element when known.
3. Upload one image or a batch.
4. See queued, processing, completed, rejected, or failed status.
5. Inspect boxes/masks, confidence, quality warnings, route, and measurement status.
6. Approve, reject, relabel, or escalate each finding.
7. Export versioned JSON/CSV, annotated images, and a PDF-style summary.
8. Reopen the exact inspection with original model and review provenance.

The machine-readable record is `schemas/finding_record.schema.json`. The model prediction is immutable; reviewer decisions are separate.

## 11. Hardware and environment

Observed local hardware:

- NVIDIA Quadro T2000, 4,096 MiB VRAM;
- NVIDIA driver 595.95;
- approximately 32 GB system RAM;
- approximately 138 GiB free on D: after current acquisition/preparation.

There is no blanket 24 GB minimum and no immediate 1 TB SSD requirement for the curated plan. Start with batch size 1, mixed precision where supported, controlled tiles, gradient accumulation for small training jobs, and explicit out-of-memory logging. Occasional 12-24 GB GPU access is a schedule accelerator for Florence-2 Large or SAM adaptation, not a prerequisite for validating the smaller candidates.

CUDA-enabled PyTorch must pass a real tensor operation before model acquisition/execution. CPU fallback is not accepted as evidence that the GPU pipeline works.

## 12. Three-person work split

### Person 1 - data and governance

- licenses, citations, checksums, source registry, and immutable raw data;
- taxonomy mapping, annotation conversion, group-safe splits, duplicates, and class/pixel audits;
- target-site import and annotation workflow.

### Person 2 - models and efficiency

- CUDA environment and reproducible baselines;
- YOLO, ResNet, SAM adapter, and Florence experiments;
- resolution/tiling, balancing ablations, latency, and VRAM profiling.

### Person 3 - product and evaluation

- inspection workflow, persistence, review UI, structured findings, and exports;
- locked evaluation, confidence/calibration, measurement protocol, field pilot, and demo.

A qualified civil/structural reviewer is additionally required unless one team member is qualified and formally takes that role.

## 13. Delivery schedule

Workstreams overlap rather than run one after another.

| Workstream | Typical effort |
|---|---:|
| Scope, workflow, taxonomy, acceptance rules | 1-2 weeks |
| Source audit and curated manifest | 2-3 weeks |
| Conversion plus targeted annotation/QA | 2-5 weeks |
| YOLO and ResNet baselines | 2-3 weeks |
| YOLO-to-SAM adapter and evaluation | 2-4 weeks |
| Florence comparison and cascade ablation | 3-5 weeks |
| Functional beta application and exports | 5-7 weeks in parallel |
| Locked tests, field pilot, fixes, handoff | 2-4 weeks |

Expected total is 14-20 calendar weeks on the current workstation with a tightly curated dataset. Occasional larger GPU access and ready target-site labels can reduce this to 12-16 weeks. New annotation for every secondary class or local fine-tuning of every large model can extend it to 18-26 weeks. Real multimodal fusion is a separate phase.

## 14. Delivery gates

### Gate 1 - scope and licenses

- beta specification approved;
- ResNet versus RetinaNet confirmed;
- engineer/reviewer named;
- target sites and image permissions identified;
- Ultralytics AGPL/enterprise implications accepted for the intended distribution.

### Gate 2 - data v0.1

- sources and checksums recorded;
- CUBIT train/validation acquired or formally deferred;
- taxonomy reviewed;
- exact/near-duplicate and group rules executed;
- task-specific selected manifests frozen;
- verified negative strategy accepted.

### Gate 3 - hardware and simple baselines

- CUDA environment verified;
- 300-sample feasibility set loads end to end;
- one simple result per task before balancing or cascades;
- latency and peak VRAM recorded with no silent CPU fallback.

### Gate 4 - controlled comparison

- fixed splits, resolution, budgets, thresholds, and stopping rules;
- balancing and cascade ablations completed;
- per-source/per-class metrics and error review completed;
- final model route selected from evidence.

### Gate 5 - product and external validation

- complete upload/process/review/export/reopen workflow;
- CUBIT locked test and target-site unseen-building test;
- false-negative and hard-negative review;
- measurement, safety, and license limitations visible;
- reproducible demo uses real inference, not hard-coded findings.

## 15. Current checkpoint

Completed:

- corrected beta scope, taxonomy, dataset contract, experiment matrix, and environment plan;
- source audit for CiF, DACL10K, CODEBRIM, S2DS, UAV75, and partial CUBIT acquisition;
- checksum and 7-Zip verification for both CODEBRIM archives;
- CODEBRIM classification registry with 7,729 records and zero cross-split parent groups;
- deterministic 300-sample feasibility selection and materialization;
- 300/300 image/annotation pairs readable and content hashed;
- versioned finding-record JSON schema and validated example;
- automated preparation tests.

Completed after the data checkpoint:

- official PyTorch 2.6.0/torchvision 0.21.0 CUDA 12.4 wheels downloaded and hash-verified;
- isolated Python 3.11 installation completed;
- real float16 CUDA matrix operation passed on the Quadro T2000;
- all six selected runtime checkpoints downloaded and offline-deserialized;
- one-image FP16 CUDA hardware smoke tests passed for YOLO11n, YOLO11n-seg, ResNet-50, SAM 2.1 Tiny, and Florence-2 Base/Large, with no CPU fallback.

Not started:

- representative multi-image quality, latency, and memory evaluation;
- label conversion into final task views;
- duplicate/near-duplicate image analysis for curated v0.1;
- model training/comparison;
- beta application implementation;
- target-site capture and final evaluation.

## 16. Actions required from the project owner

1. Ask the advisor to confirm whether the comparison is **ResNet** or **RetinaNet**.
2. Name the civil/structural reviewer and schedule taxonomy plus sample-image review.
3. Identify at least three candidate buildings/construction sites and obtain written image-use permission.
4. Decide whether the delivered source must be open under AGPL-compatible terms or whether Ultralytics enterprise licensing/another implementation is needed.
5. Decide whether physical crack dimensions are required in beta 1; if yes, choose and field-test the scale/calibration method.
6. Provide any separate T3000/larger-GPU machine details if it exists; otherwise all claims remain tied to the observed T2000.
7. Keep the official test sets and target-site test buildings unavailable to day-to-day model/UI tuning.

These decisions do not block documentation and data auditing, but they do block a defensible final dataset freeze and product acceptance claim.
