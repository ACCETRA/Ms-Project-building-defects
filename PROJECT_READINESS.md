# Infrastructure Defect Inspection — Master's FYP Prototype Readiness Plan

Prepared: 2026-09-07; preparation checkpoint updated 2026-09-08  
Status: feasibility training is complete; full-data labeling, training, evaluation, and product completion remain. See [`docs/COMPLETION_PLAN.md`](docs/COMPLETION_PLAN.md) for the authoritative execution plan.

## Recommended project boundary

This document describes the intended beta. The actual repository checkpoint is narrower: the 300-sample feasibility tier has run, while the proposed 20,000-25,000 sample v1 mixture has not yet been materialized into task-specific manifests or trained. Do not describe the feasibility checkpoints as final accuracy results.

Build a **fully functional academic beta for RGB-image inspection of buildings and construction works, with cracks as the primary defect**. The initial material scope is visible concrete and masonry surfaces on façades, walls, slabs, columns, beams, ceilings, and foundations. It should behave like a coherent product, not a loose collection of model notebooks. The beta should:

1. find crack candidates with high recall;
2. localize them with boxes and masks;
3. report pixel geometry and physical dimensions only when capture scale is known;
4. detect a small set of secondary visible conditions where labels permit;
5. provide a complete inspection workflow: project creation, image intake, processing status, results, review, and export;
6. produce structured, reviewable findings rather than claim an autonomous engineering assessment.

Thermal, GPR, full 3D fusion, and historical-record fusion are outside the accepted FYP scope. The current implementation preserves the identifiers, calibration, coordinates, timestamps, and provenance that additional modalities would require.

## Decisions the project owner must make before implementation

- **Use case:** a master's FYP prototype with a coherent demonstration workflow, released under AGPL-compatible open-source terms. It is not a production inspection product and has no professional engineering validation. DACL10K and CODEBRIM can support the academic prototype under their non-commercial terms, but a later commercial release will require permission or retraining on commercially usable/private data. Ultralytics YOLO remains subject to its AGPL obligations.
- **Target asset:** completed and under-construction buildings, initially limited to visible concrete and masonry elements. General workmanship defects, MEP defects, steel fabrication, fire damage, and hidden/internal structural diagnosis are outside the first beta.
- **Capture mode:** handheld/mobile still images for interiors and accessible elements, plus UAV still images for exterior façades. Extracted video frames are deferred until frame grouping and duplicate handling are designed.
- **Primary deliverable:** an end-to-end reviewed inspection package containing annotated images, crack boxes/masks, confidence and quality warnings, structured JSON/CSV findings, and a generated summary report. Crack measurement remains a separate validation problem, not an automatic by-product of segmentation.
- **Secondary visible classes:** recommended first-beta set is spalling, honeycombing/rock pockets, exposed rebar, rust/corrosion staining, efflorescence/leaching, and dampness/moisture staining. Public datasets do not cover these evenly, so the owner must either reduce this list or fund target-site annotation. Algae and general weathering can wait.
- **Deployment target:** local browser-based demonstration on this Windows workstation.
- **Acceptance criteria:** complete real-inference demo by Sunday 2026-09-13, schema-valid outputs, honest failure/empty-result handling, recorded latency/VRAM, and no unsupported structural or physical-measurement claims.
- **Named UAV dataset:** use building-façade-focused CUBIT-InSeg as the preferred UAV source under its repository's CC BY 4.0 terms. Retain UAV75 only as a tiny stress set. The larger Liu repository is a consolidated crack benchmark and is not equivalent to 11,298 independently captured UAV images.

## Corrections to the supplied roadmap

1. **CiF volume:** the public release contains 12,896 unique full-resolution source images and 148,642 derived 1024×1024 tiles. Full images and their tiles must never cross data splits or be counted as independent structures.
2. **CiF classes:** Algae, Crack, Net-Crack, Crack with Precipitation, Rust, and Spalling. CiF does not directly provide CODEBRIM's exposed-rebar class.
3. **DACL10K classes:** 13 damage classes plus 6 component/object classes, not 12 plus 6.
4. **DACL10K test access:** public pixel labels cover 6,935 train and 975 validation images. The test-dev/challenge labels are withheld; official challenge evaluation requires submission. If all DACL10K labeled images are untouched during development, the 7,910 public labeled images can serve as a local external test set, but that must not be described as the hidden challenge test.
5. **CODEBRIM incompatibility:** CODEBRIM supplies multi-label crops and original-image bounding boxes, not pixel-accurate masks. It can train/evaluate classification and detection, but it cannot be pooled as equal segmentation ground truth without new human masks or explicitly labeled weak/pseudo masks.
6. **Label semantics:** rust staining is evidence associated with corrosion, not proof of internal corrosion; DACL10K's washout/concrete-corrosion label is not steel corrosion. Net-crack/alligator crack and crack-with-precipitation should remain subtypes under a crack family rather than be silently erased.
7. **Measurement:** pixels cannot be reported as millimetres without a scale, camera calibration, known ground-sample distance, planar geometry, a reference marker, or registered 3D geometry.
8. **Safety:** “minimal human review” is not an appropriate claim for a safety-related inspection system. The prototype can look polished and operate end to end while still labelling outputs as candidate findings that require manual review.

## Data strategy

The project may use a smaller, balanced **curated mega dataset** rather than every image from every source. Implement it as a versioned manifest/training view, not as an untraceable physical folder merge and not by deleting unused source files. Maintain immutable raw sources and a registry that maps every selected annotation into a shared taxonomy while preserving `dataset`, `source_asset`, `source_image`, `parent_image`, annotation type, split, and license.

One registry should produce separate task-compatible views:

- a **detection view** containing boxes derived from native boxes or masks;
- a **segmentation view** containing only genuine pixel masks, with weak or pseudo masks clearly identified if they are ever added;
- a **classification view** containing image/crop-level multi-label targets;
- a **product-demo view** containing representative building and construction-site images plus difficult no-defect examples.

This matters because CODEBRIM classification labels cannot silently become SAM segmentation ground truth, and an unlabeled class in a partially annotated dataset must not automatically be treated as a confirmed negative. Selection should be stratified by source, building/site, capture mode, defect family, image quality, and negative/context type. Balance the training sampler after the splits are frozen; do not force every class to have an identical count merely to make the folder look balanced.

| Source | Recommended use now | Do not use it for |
|---|---|---|
| CUBIT-InSeg | Preferred building-façade UAV instance-segmentation source for crack and spalling; target-domain benchmark | Assuming equal-distance capture provides metrically calibrated imagery; omitting the attribution required by CC BY 4.0 |
| CiF tiled official splits | Primary supplemental crack/condition detection and instance-segmentation training, validation, and in-domain test | Counting tiles as unique source images; physical measurement without calibration |
| S2DS | Small multi-class structural-surface validation/transfer source for crack, spalling, corrosion, and efflorescence | A large standalone benchmark or commercial use |
| CODEBRIM original images/boxes | Secondary multi-label classification and box-detection transfer track; ResNet baseline | Pixel-mask training; treating bridge imagery as proof of building performance; commercial product training under its current license |
| DACL10K public labeled set | Out-of-domain bridge robustness test after the building model is locked | Serving as the primary external test for a building product |
| UAV75 | Small building/drone-domain stress test or smoke dataset | A statistically strong standalone training claim |
| Liu UAV/crack benchmark | Candidate source pending exact provenance, license review, and duplicate audit | Assuming every consolidated source image is UAV-captured |
| ConRebSeg | Optional construction-context source if exposed rebar/person/equipment segmentation becomes a beta requirement | Primary crack training; its labels target exposed bars and scene objects rather than crack masks |

Recommended shared visible taxonomy:

- `Crack`, with optional subtype `linear`, `network/alligator`, or `with mineral deposit`;
- `Spalling`;
- `Honeycombing/rock pockets`;
- `Rust/corrosion staining`;
- `Efflorescence/leaching`;
- `Exposed rebar`;
- `Dampness/moisture staining`;
- optional surface conditions such as `Algae` and `Weathering`.

The project team must document this mapping and mark disputed cases as `unknown_review`; no professional engineering approval is assumed.

### Splitting and leakage controls

- Respect CiF's official parent-image splits.
- Start from CUBIT-InSeg's official train/validation/test boundaries, but apply the recorded exact-dedup decision manifest with priority `test > val > train`. Keep its test images unavailable to model, threshold, and user-interface tuning. The immutable publisher archives stay unchanged.
- For CODEBRIM, split by bridge/source structure, not by crop or box.
- Group UAV video frames or overlapping photographs by flight/scene before splitting.
- Detect exact and near duplicates before assigning any custom split.
- Include construction-site hard negatives such as joints, formwork seams, tie holes, concrete edges, conduits/cables, chalk and paint marks, dust, shadows, wet patches, scaffolding, and partial occlusion so they are not routinely mistaken for cracks.
- If DACL10K is retained as an out-of-domain stress test, keep it inaccessible to model and threshold selection and evaluate it only after the building-domain experiment is locked.
- Create a separate target-building/site field test set, grouped by building and capture session. A public cross-dataset score does not replace validation on the actual deployment domain.

## Imbalance handling for a crack-primary system

Profile imbalance at four levels: source images, images containing each class, instances, and annotated pixels. Image counts alone are misleading for multi-label segmentation.

The acquired CUBIT labels demonstrate why: its official training split contains 2,535 crack images and 3,061 spalling images, but 41,811 crack polygon lines versus 8,224 spalling polygon lines. Validation contains 299 crack and 400 spalling images, with 5,029 and 1,106 polygon lines respectively. Thin/fragmented crack geometry produces many more instances without implying more crack images. Balance exposure at the image/group level first and treat instance or pixel weighting as an experiment, not an automatic inverse-frequency formula.

Start with an unweighted baseline. Then compare a controlled correction using:

- class-aware sampling/cropping for secondary classes;
- moderate minority-class augmentation that preserves thin crack geometry;
- a task-appropriate loss: weighted binary cross-entropy/focal plus Dice/Tversky for overlapping segmentation, or class-weighted detection losses for detector heads.

Do not automatically apply inverse-frequency weights, oversampling, and heavy augmentation at maximum strength together; they can over-correct and overfit repeated minority examples. Never oversample or augment validation/test data. Preserve adequate no-defect/background examples to measure false alarms.

## Fair model comparison

YOLO, SAM, ResNet, and Florence-2 do different jobs, so a single leaderboard would be misleading.

Recommended comparison, aligned with the supplied Florence-2 roadmap:

1. **Detection/triage track:** compare Florence-2 Base with a named YOLO nano/small detector on the same detection view, split, resolution policy, and box metrics. A ResNet classifier may be compared with image-level Florence/YOLO triage on the classification view, but cannot be scored as a detector unless a detector such as RetinaNet was actually intended.
2. **Segmentation track:** compare an autonomous YOLO segmentation model with `YOLO detector → SAM 2 Tiny` using exactly the same YOLO boxes as SAM prompts. SAM is a prompted mask generator, not an autonomous crack classifier or detector.
3. **Florence cascade ablation:** compare Florence-2 Base alone with `Base → Large for uncertain cases` to test whether the Large stage adds enough recall/precision to justify its latency and memory. “Uncertain” must be defined and calibrated on validation data.
4. **End-to-end systems:** compare the best Florence-based route against the best specialist route, such as `YOLO → SAM`, using crack recall, end-to-end mask quality, latency, escalation rate, and peak VRAM. This is the product-level comparison that answers whether Florence should remain in the beta.

“YOLO is the small model and SAM is the large model” is a useful verbal simplification, not a technical definition. Both families have multiple sizes; for example, SAM 2.1 Tiny is much smaller than SAM 2.1 Large. The correct distinction is that YOLO performs autonomous detection on every image, while SAM performs prompt-driven pixel segmentation only for selected detections.

Use **ResNet-50** as the separate image/crop classification baseline. It is not the backbone used by YOLO or SAM. RetinaNet is deferred because Florence and YOLO already cover the detector comparison track.

A pure semantic-mask baseline is outside the accepted FYP scope; the matrix remains bounded to the delivered models.

For any cascade, send both positive and uncertain images forward, tune the first stage for high recall, calibrate its uncertainty threshold on validation data, and audit a random sample of “confident negatives.” A negative image may skip SAM but must still be preserved as a recorded no-defect result. Otherwise the first stage's missed cracks can never be recovered by the larger model.

SAM adaptation should begin with box prompting and evaluation before any fine-tuning. The adaptation layer must convert YOLO boxes to SAM prompts, select/merge masks, reconnect results to tiles and source images, and reject poor masks. If this is insufficient, test a frozen image encoder with cached embeddings and a trainable lightweight mask decoder. Decoder-only fine-tuning is an experiment to validate, not a guaranteed requirement or guaranteed improvement for thin cracks.

Cracks only a few pixels wide can disappear during resize. Evaluate overlapped tiling and resolution explicitly; use the same source images, splits, input resolution policy, compute budget, and stopping rule for fair comparisons.

## Evaluation contract

Report results per class, per source dataset, and for cracks separately from macro averages.

- Classification/triage: precision, recall, F1, PR-AUC, confusion matrix, and calibrated confidence.
- Detection: AP50 and mAP50:95 for boxes, plus crack recall at the selected operating threshold.
- Segmentation: class IoU/Dice and instance mask AP where annotations support it.
- Measurement: pixel error first; MAE/RMSE in physical units only against field-measured, calibrated ground truth.
- Operations: latency, throughput, peak VRAM, rejected-image rate, and percentage sent for review/escalation.
- Reliability: confidence intervals, performance by image quality/capture domain, and an error review of false negatives.

Model choice and thresholds are locked using only leakage-clean training/validation sources. Run CUBIT-InSeg's 694-image exact-deduplicated building-façade test view and the target-site field test after locking. The 701-image publisher test metric may be reported separately for comparability, with its known duplication disclosed. Report DACL10K separately, if used, as out-of-domain bridge robustness rather than the beta's principal generalization result. A later leave-one-dataset-out study is useful but not required for the first beta.

## Structured finding record

Each prediction should eventually produce a versioned record containing:

- project/site, building, inspection, source-image, and parent-image IDs;
- building elevation or façade, floor, room/zone, and element type (wall, slab, column, beam, ceiling, or foundation) when known;
- capture timestamp, camera metadata, GPS/asset component, coordinate reference/frame, pose, and calibration/GSD when known;
- defect family and subtype;
- bounding box and mask/polygon or RLE;
- pixel length, width, and area;
- physical dimensions only when calibration is valid, including units and scale method;
- confidence, uncertainty/calibration method, image-quality flags, and escalation/review state;
- model, weights, taxonomy, dataset, and pipeline versions;
- links to the immutable source image and any additional thermal, GPR, 3D, or historical evidence.

This is the interface for additional evidence integration. It is not necessary to build the integration engine in this project.

## Functional beta definition of done

The beta is complete when a reviewer can use it without opening a notebook or reading developer instructions:

1. create or open an inspection project and identify the site, building, floor/elevation, zone, and structural element;
2. upload one image or a batch and see validation errors for corrupt, unsupported, blurred, or badly exposed inputs;
3. follow clear queued/processing/completed/failed states;
4. view each image with defect boxes and masks, filter findings, and inspect confidence and image-quality warnings;
5. approve, reject, relabel, or flag a finding for engineer review without changing the immutable model prediction;
6. see pixel measurements and calibrated physical measurements only when the scale status is valid;
7. export annotated images, a PDF-style inspection summary, and versioned JSON/CSV findings;
8. reopen prior inspections and retain model, taxonomy, dataset, timestamp, and review provenance;
9. display limitations, academic-use terms, and a visible “decision support—not a structural safety determination” notice;
10. handle empty results and failures honestly, with no fake fusion controls or placeholder claims presented as working features.

The beta should include a scripted demonstration dataset and walkthrough, but the displayed results must come from the actual pipeline rather than hard-coded examples. Product polish should come from a reliable workflow, not from hiding uncertainty.

## Resources required

Current machine observed on 2026-09-07:

- NVIDIA Quadro T2000 with 4 GB VRAM;
- approximately 32 GB system RAM;
- approximately 184 GB free on D: before source acquisition.

The laptop is suitable for dataset engineering, the beta application, YOLO nano/small and ResNet experiments on tiled images, Florence-2 Base inference experiments, and SAM 2 Tiny prompt-based inference. Local training must use controlled image sizes, mixed precision where supported, small batches, and gradient accumulation. Full-resolution end-to-end fine-tuning of Florence-2 Large or SAM 2 is not a realistic expectation on 4 GB VRAM. Meta's official SAM 2 training example assumes 80 GB A100 GPUs, but that example is not a minimum requirement for inference or for a carefully constrained decoder-only experiment.

Provide:

- no immediate 1 TB purchase is required for the curated-subset plan. The current D: drive is usable if raw archives stay compressed, working subsets are controlled, caches are cleaned deliberately, and free space is monitored; an external 500 GB or larger SSD is a convenience/backup recommendation rather than a beta prerequisite;
- no fixed 24 GB GPU minimum. First benchmark the chosen model variants on the installed Quadro T2000 4 GB. Occasional access to a 12–24 GB cloud/workstation GPU is recommended if Florence-2 Large or SAM adaptation cannot fit locally, and is primarily a schedule accelerator rather than proof that the smaller models are invalid;
- experiment storage/backups separate from the laptop;
- a qualified structural/civil engineer for taxonomy, annotation QA, severity rules, field measurement, and acceptance review;
- a target-domain pilot capture with camera calibration/scale information and permission to use the imagery.

## Three-person division of work

- **Data lead:** licensing, provenance, immutable raw data, taxonomy mapping, group splits, duplicate/leakage audit, annotation statistics, and annotation tooling.
- **Model lead:** reproducible baselines, controlled imbalance experiments, resolution/tiling study, model comparison, and efficiency profiling.
- **Evaluation/product lead:** structured record, calibration and uncertainty, field-measurement protocol, locked external evaluation, review workflow, and final report/demo.

The civil/structural engineer is an additional required reviewer unless one of the three team members is qualified for that role.

## Realistic three-person schedule

The workstreams overlap; the durations below are not added end to end.

| Workstream | Typical calendar effort | Main output |
|---|---:|---|
| Scope, user workflow, taxonomy, and acceptance rules | 1–2 weeks | Signed beta specification and annotation guide |
| Source audit and curated-mega-dataset manifest | 2–3 weeks | Provenance, licenses, selection rules, duplicate report, and group-safe splits |
| Label conversion and targeted annotation/QA | 2–5 weeks | Detection, segmentation, and classification views |
| YOLO and ResNet baselines | 2–3 weeks | Reproducible local baselines and first hardware timings |
| YOLO-to-SAM prompt adapter and mask evaluation | 2–4 weeks | Box-prompted masks, tile merging, failure analysis, and decoder-tuning decision |
| Florence-2 Base/Large comparison and cascade ablation | 3–5 weeks | Stage and end-to-end comparison results |
| Functional beta application, review workflow, persistence, and reports | 5–7 weeks in parallel | Usable browser-based inspection beta |
| Locked tests, target-site pilot, fixes, and handoff | 2–4 weeks | Final metrics, limitations, demo, and documentation |

With the installed Quadro T2000 and a tightly curated dataset, plan for **14–20 calendar weeks**. Occasional larger-GPU access and already-available target-site labels can bring this closer to **12–16 weeks**. Starting target-site annotation from zero, requiring every secondary class, or insisting on local fine-tuning of every large variant pushes it toward **18–26 weeks**. Implementing real 3D/thermal/GPR fusion is outside the accepted FYP scope and is not included in these estimates.

## Delivery gates

1. **Scope/license gate:** signed one-page MVP, AGPL-compatible academic distribution decision, named target asset/capture mode, selected datasets/models, and license approval.
2. **Data gate:** immutable source manifest and checksums, audited labels, approved taxonomy, group-safe splits, duplicate report, and class/pixel statistics.
3. **Baseline gate:** one simple result per task before imbalance correction or a cascade.
4. **Comparison gate:** fixed splits/budgets, calibrated thresholds, ablations for sampling/loss/augmentation, and efficiency results.
5. **External-validation gate:** one locked, leakage-clean CUBIT-InSeg building-façade test, a target-site field test, and a false-negative review; DACL10K is an optional out-of-domain bridge stress test.
6. **Measurement gate:** calibrated capture protocol and engineer-verified physical ground truth; otherwise release pixel measurements only.
7. **Functional-beta gate:** end-to-end inspection workflow, review queue, structured export, generated report, persisted history, reproducible model/data versions, error states, limitations, and handoff documentation.

A production inspection product would require additional field collection, professional safety validation, monitoring, security, and regulatory/contractual work; those are outside this FYP scope.

## Source acquisition layout

- `source/The RoadMap.md`: preserved copy of the supplied roadmap.
- `references/papers/`: CiF, DACL10K, CODEBRIM, UAV inspection, CUBIT-InSeg, Florence-2, SAM 2, and ResNet papers downloaded and title-checked. The initial file mislabeled as CUBIT was identified as an unrelated Taylor-Aris dispersion paper and preserved under `MISIDENTIFIED_Shear_alignment_Taylor-Aris_2026.pdf`; the correct 23-page CUBIT publisher PDF is now installed with SHA-256 `efad2118799d9762a35d1e27a4a666d7a260f136f911e600ea0b1f58e18895ab`.
- `references/toolkits/`: official DACL10K, CODEBRIM, CUBIT-InSeg, and ConRebSeg supporting repositories; downloaded for reference only and not installed.
- `datasets/CiF-tiled/`: selected 1024×1024 official splits; download completed (approximately 23.07 GB decimal).
- `datasets/DACL10K/`: labeled development archive downloaded and verified: 5,109,737,241 bytes, SHA-256 `dcbcd5fb82699076a2c7f3a72492a9ef798870e0ca1f0c9399360f273ea95260`.
- `datasets/CODEBRIM/`: original images/box annotations downloaded and verified: 8,310,630,622 bytes, MD5 `27baf3a036d0b7d757ff4df47c08c449`; official unbalanced classification archive also downloaded and verified: 7,911,716,093 bytes, MD5 `c1612d9674e2e628e72e7f5817c40130`. Both archives pass complete 7-Zip 26.02 tests and 11 extracted image probes decode correctly. Their legacy headers trigger a 32-bit-overflow warning and break Python `zipfile`/Windows `tar`, so all image extraction must use the project-local portable 7-Zip path.
- `datasets/building-target/CUBIT-InSeg/`: all six official train/validation/test image and label archives downloaded (19,059,292,068 bytes total), CRC-tested, SHA-256 recorded, and paired by filename: 5,596 train, 699 validation, and 701 locked test pairs. The definitive report is `artifacts/data-audit/cubit_archive_inventory.json`; test content remains unavailable to tuning.
- CUBIT duplicate audit: all 6,996 images decoded and hashed without extraction. It found 589 exact `SP0`/`SP1` duplicate pairs, including 222 cross-split pairs. The non-destructive exact-dedup manifest retains 5,035 train, 678 validation, and 694 test records with no selected exact hash crossing splits; 521 perceptual cross-split candidates remain pending review.
- `datasets/building-target/S2DS/`: repository and 1.24 GB official dataset archive downloaded.
- `data/manifests/feasibility_v0.csv`: deterministic training-only smoke-test manifest completed with 120 CiF, 80 S2DS, 40 UAV75, and 60 DACL10K samples.
- `data/feasibility_v0/`: all 300 selected image/annotation pairs materialized and verified readable; this set is for data-loading, inference, and VRAM feasibility only, not model-selection or accuracy claims.

The project-local Python 3.11 environment now contains hash-verified PyTorch `2.6.0+cu124` and torchvision `0.21.0+cu124`. A real float16 CUDA matrix operation passed on the observed Quadro T2000; the detailed result is recorded at `artifacts/environment/cuda_verification.json`. All six selected runtime checkpoints pass offline deserialization and one-image FP16 CUDA inference, including Florence-2 Large at approximately 1.90 GB peak allocated VRAM. Representative speed distributions, defect quality, training fit, and end-to-end compatibility are still unmeasured.
- `datasets/UAV-candidates/UAV75/`: complete 75-image UAV dataset.
- `datasets/UAV-candidates/Liu-UAV-benchmark/`: repository copy for provenance review; the linked external consolidated dataset has not been assumed to be the selected UAV source.
- `weights/`: local runtime assets for YOLO11n, YOLO11n-seg, ResNet-50, SAM 2.1 Hiera Tiny, and native-Transformers Florence-2 Base/Large FT conversions. File-level hashes and load results are in `artifacts/model-assets/`.

The comparison roles and exact feasibility versions are now provisionally frozen in `config/model_candidates.yaml`, so the required weights have been downloaded and audited. No training has started. A version change after GPU smoke testing must create a new decision-log entry rather than silently replacing a checkpoint.
