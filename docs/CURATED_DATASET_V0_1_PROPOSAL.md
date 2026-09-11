# Curated Mega Dataset v0.1 — Selection Proposal

**Status:** Source budget draft; feasibility sample completed; full v1 selection and QA remain
**Audit evidence:** `artifacts/data-audit/`  
**Purpose:** bound the first detector, segmenter, classifier, and hardware-feasibility datasets

## 1. Audit findings that affect selection

- CiF tiled is complete: 105,139 train, 22,368 validation, and 21,135 test tiles. The local audit found 8,874/1,900/1,806 unique parent filenames respectively. All audited tiles contain at least one annotated instance, so CiF alone cannot provide verified whole-image negatives.
- DACL10K is complete for public train/validation labels: 6,935 train and 975 validation images. It is bridge-domain and should be capped in the building training mixture.
- S2DS is complete with 563 train, 87 validation, and 93 test image/mask pairs. It is small enough to keep its official train/validation partitions intact.
- UAV75 is complete with 50 train, 10 validation, and 15 test image/mask pairs. Its planking context is useful for crack false positives, but its size is too small for a primary claim.
- CUBIT-InSeg is complete and verified: 5,596 train, 699 validation, and 701 test image/label pairs. All six archives pass CRC checks, their SHA-256 values are recorded, and no image/label stem is unpaired. Keep the official test split locked and do not inspect or tune on it.
- CUBIT's available train labels contain 2,535 crack images/41,811 crack polygons and 3,061 spalling images/8,224 spalling polygons; validation contains 299/5,029 and 400/1,106 respectively. This is image-level spalling prevalence but instance-level crack prevalence, so polygon-count equality is not a sensible balancing target.
- CUBIT's 6,996 images contain 589 exact `SP0`/`SP1` duplicate pairs: 352 within train, 8 within validation, 7 within test, 99 across train/validation, 110 across train/test, and 13 across validation/test. Apply the recorded `test > val > train` exact-dedup decisions; 521 cross-split perceptual candidates still require review.
- CODEBRIM originals contain 1,590 images and 1,052 XML files with 5,233 multi-label boxes. The official unbalanced classification release contains 7,729 crops. Both published checksums match, both full archives pass 7-Zip tests, and 11 extracted probes decode. Python/Windows standard ZIP readers remain incompatible with the oversized headers, so use the project-local portable 7-Zip extractor. The classification registry confirms zero parent groups crossing the official train/validation/test partitions.
- The current public sources do not supply enough verified no-defect building/construction images. A target-site negative collection is required.

## 2. Proposed source budgets

These are maximum first-release budgets, not quotas that must be filled regardless of quality.

| Source | Training/validation proposal | Locked evaluation | Rationale |
|---|---|---|---|
| CUBIT-InSeg | Use the exact-deduplicated 5,035 train and 678 validation records, subject to near-duplicate review | 694-record leakage-clean locked test; report the full 701-image publisher test separately | Primary building-façade UAV domain; official split contains exact duplicates |
| CiF tiled | Select up to 12,000 train and 2,500 validation tiles across parent images and native classes | Select up to 2,500 official test tiles for source-specific reporting | Limit dominant source while preserving rare native classes and parent diversity |
| S2DS | All 563 train and 87 validation pairs | All 93 official test pairs | Small building/structural surface source with useful secondary classes |
| UAV75 | All 50 train and 10 validation pairs | All 15 official test pairs | Tiny crack/planking stress source |
| DACL10K | Select up to 2,000 official-train images containing canonical target classes; derive an internal group-safe validation subset from official train | All 975 official validation labels as bridge-domain stress test | Prevent bridge imagery from dominating a building product |
| CODEBRIM classification | Use the official unbalanced train/validation partitions after archive verification | Official test partition | ResNet/image-level comparison only; no segmentation claims |
| Target-site building data | Add approved train/validation buildings only after capture and review | Reserve entire unseen buildings/capture sessions | Essential deployment evidence and verified negatives |

The likely first training universe is approximately 20,000–25,000 selected source images/tiles before augmentation, subject to final CODEBRIM and cross-source curation. It is intentionally much smaller than the total downloaded corpus.

## 3. Feasibility sample v0 â€” completed

Build a 300-image manifest from training/validation sources only:

- 120 CiF train tiles spanning crack subtypes, rust, and spalling across distinct parent images;
- 80 S2DS train images spanning all in-scope native classes;
- 40 UAV75 train images spanning crack/planking appearance;
- 60 DACL10K train images spanning crack, alligator crack, spalling, rock pocket, exposed rebar, rust, and efflorescence.

Do not use CUBIT test, S2DS test, UAV75 test, DACL validation, or future target-building test images for hardware tuning.

The initial 300 images measure loading, tiling, inference compatibility, latency, and VRAM. They are not an accuracy benchmark and must not be reported as one.

Completed evidence:

- selection manifest: `data/manifests/feasibility_v0.csv`;
- CUBIT train/validation source registry: `data/manifests/cubit_train_val_source_v1.csv` (6,295 records; locked test excluded);
- CUBIT image hashes/candidates: `artifacts/data-audit/cubit_image_hashes.csv` and `cubit_duplicate_candidates.csv`;
- CUBIT exact-dedup decisions: `data/manifests/cubit_exact_dedup_decisions_v1.csv` (5,035 train/678 validation/694 test retained);
- materialized/content-hashed manifest: `data/manifests/feasibility_v0_materialized.csv`;
- materialized data: `data/feasibility_v0/`;
- verification: 300/300 images and annotations readable, 300 unique sample IDs, 300 unique group IDs, and no held-out partition locators;
- report: `artifacts/data-audit/feasibility_materialization_report.json`.

## 4. Training mixture policy

- Sample by canonical class, source, and parent/group rather than by raw file frequency.
- Keep cracks primary while establishing minimum exposure for every secondary class.
- Retain native multi-label combinations.
- Avoid repeatedly showing the same rare instance through many nearly identical tiles.
- Use a source cap so CiF cannot dominate every batch.
- Add verified no-defect/hard-negative images as a dedicated sampling stratum.
- Establish an unweighted baseline before activating loss weighting.

Exact sampler weights will be calculated from the selected manifest. They must not be guessed from global headline counts.

## 5. Target-site collection requirement

Before beta acceptance, collect permission-cleared RGB images from at least three distinct completed/under-construction buildings or clearly document why fewer sites were possible. Include:

- confirmed target defects reviewed by the project team;
- verified no-visible-target-defect surfaces;
- joints, formwork seams, tie holes, concrete edges, conduits/cables, chalk/paint marks, dust, shadows, wet patches, scaffolding, and occlusion;
- interior and exterior conditions where intended;
- capture metadata and scale/calibration evidence when physical measurement is desired.

A practical pilot target is 500–1,000 reviewed site images, with whole buildings/capture sessions reserved for final testing. This is a planning target, not a statistical guarantee; rare-class availability may require more collection.

## 6. Promotion gates

The proposal is not yet a frozen full-data training manifest. The next step is to create the v1 master and task views, review the source/class counts, and record the freeze before full training. See [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) for ownership, exact output paths, and the required sequence.

Version 0.1 cannot be frozen until:

- CUBIT train/validation masks are available and archive-verified; **completed**;
- CODEBRIM classification archive is checksum-verified and readable through the recorded 7-Zip extraction path; **completed**;
- canonical class mappings receive engineering review;
- verified negative strategy is approved;
- exact/near-duplicate and group-ID rules are implemented;
- license compatibility and intended source-code disclosure are decided;
- the selected-manifest class/source counts are reviewed.
