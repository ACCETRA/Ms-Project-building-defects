# Curated Mega Dataset Contract

**Contract ID:** BDI-DATA-001  
**Status:** Active for data preparation

## 1. Definition

The “mega dataset” is a versioned logical collection of selected samples from approved sources. It is represented by manifests and task views. It is not an anonymous folder merge, and it does not require every source image to participate in training.

## 2. Raw-source rules

- Raw downloads are immutable.
- Archive checksums, source URL, retrieval date, version, citation, and license are recorded.
- Selection never deletes the raw source.
- Generated masks, boxes, tiles, embeddings, or pseudo labels live outside raw-source directories.
- Dataset-specific terms govern every derivative and trained model that uses the source.

## 3. Required manifest fields

| Field | Purpose |
|---|---|
| `sample_id` | Stable project identifier |
| `source_dataset` | Dataset of origin |
| `source_version` | Exact release/configuration |
| `source_path` | Immutable source-relative path or archive member |
| `source_image_id` | Native source identifier |
| `parent_image_id` | Full image before tiling/cropping |
| `group_id` | Building, structure, flight, video, or capture-session grouping |
| `split` | `train`, `validation`, `test`, `external_test`, or `unassigned` |
| `capture_mode` | handheld, UAV, unknown, or other controlled value |
| `asset_domain` | building, construction site, bridge, mixed, unknown |
| `annotation_type` | image label, box, polygon, semantic mask, instance mask, line |
| `native_labels` | Unmodified source labels |
| `canonical_labels` | Approved taxonomy targets |
| `annotation_status` | positive, verified negative, unknown, partial |
| `mapping_version` | Taxonomy/mapping version |
| `license_id` | Machine-readable license reference |
| `quality_flags` | Blur, exposure, resolution, occlusion, duplication, etc. |
| `selection_reason` | Why the sample is in the curated release |
| `sha256` | Content identity when materialized as a standalone file |

## 4. Task views

### Detection view

Uses native boxes plus boxes derived from genuine masks. Derived boxes retain their provenance. Image-level classification tags are not converted into boxes.

### Segmentation view

Uses genuine polygons/masks/lines that have an approved interpretation. Native boxes and image tags are excluded from ground-truth pixel evaluation. Any pseudo-mask experiment uses a separate weak-label view.

### Classification view

Uses image/crop-level multi-label targets. Crops from one parent image remain within one split.

### Product and field-test view

Represents intended building/construction inputs, including no-defect and difficult-negative cases. This view is not used for threshold selection once designated as a final test.

## 5. Split order

1. Identify parent images, buildings/structures, flights/videos, and capture sessions.
2. Detect exact and near duplicates.
3. Assign whole groups to splits.
4. Reserve official and target-site test groups.
5. Select/curate train and validation samples.
6. Tile/crop only after group assignment.
7. Augment only the training split.

No tile, crop, overlapping UAV frame, or duplicate of a test parent may enter training.

CUBIT-InSeg's publisher split contains exact `SP0`/`SP1` duplicates, including cross-split copies. Preserve the raw archives, but derive selected views with `test > validation > train` priority and one deterministic keeper per exact-hash group. Report the publisher test and the leakage-clean test separately. Perceptual-hash candidates require review and are never auto-deleted.

## 6. Balancing policy

- Profile image, class-containing-image, instance, and annotated-pixel counts.
- Preserve cracks as the primary class without erasing minority conditions.
- Preserve enough verified/background and hard-negative images to measure false alarms.
- Begin with an unweighted baseline.
- Evaluate class-aware sampling, targeted augmentation, and loss weighting as controlled ablations.
- Do not balance validation or test data.
- Do not require equal counts across all classes.

## 7. Initial source roles

| Source | Initial role |
|---|---|
| CUBIT-InSeg | Primary target-domain UAV building-façade crack/spalling benchmark |
| CiF tiled | Large supplemental instance-segmentation/detection source |
| S2DS | Small multi-class structural-surface semantic segmentation source |
| UAV75 | Tiny UAV crack/planking stress set |
| CODEBRIM originals | Secondary classification/box-detection source; non-commercial educational/research use |
| DACL10K | Bridge-domain semantic segmentation transfer/stress source; not primary building proof |
| Liu repository | Provenance candidate only until the linked dataset and license are verified |
| ConRebSeg | Optional construction-context/exposed-rebar source, not primary crack ground truth |

## 8. Release acceptance

A curated dataset version may be frozen only when it has:

- source and license manifest;
- approved taxonomy mapping;
- duplicate/group audit;
- split report;
- per-class counts by source and annotation type;
- documented selection procedure and random seed;
- representative image review by the engineering reviewer;
- explicit list of known missing classes and weak mappings.

## 9. Feasibility v0.1 implementation

The training-only 300-sample feasibility collection now has a normalized master
manifest and row-level detection, segmentation, classification, and Florence
product views at `data/manifests/feasibility_*_v0_1.csv`. Normalized annotations
are reproducibly generated under `data/feasibility_v0/normalized/` by
`scripts/build_feasibility_task_views.py`.

The conversion preserves native labels and source-annotation hashes, identifies
mapping strength, marks boxes derived from genuine masks, and does not create masks
from boxes or image labels. The first full conversion retained 667 target
annotations and rejected four malformed native polygon fragments while retaining
the valid native boxes associated with those objects. This is a feasibility view,
not the final curated mega-dataset release.
