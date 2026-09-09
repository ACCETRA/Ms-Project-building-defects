# Functional Beta Specification

**Specification ID:** BDI-BETA-001  
**Status:** Approved for preparation  
**Date:** 2026-09-07

## 1. Product outcome

Deliver a functional, mobile-friendly inspection beta that accepts RGB still images of completed and under-construction buildings, finds visible surface-condition candidates, supports human review, and produces a traceable inspection package. It must operate end to end without requiring a notebook or hard-coded demonstration results.

This is an academic project with product-quality workflow and architecture. It is a decision-support beta, not an autonomous structural-safety system.

## 2. Intended users

- Site or building inspector who creates an inspection and uploads images.
- Civil/structural engineer or qualified reviewer who verifies findings.
- Project administrator who manages inspections, users, model versions, and exports.

## 3. Target assets and materials

Included:

- completed buildings and active building-construction sites;
- visible concrete and masonry façades, walls, slabs, beams, columns, ceilings, and foundations;
- exterior UAV still images and accessible interior/exterior handheld still images.

Excluded from the first beta:

- bridges and roads as primary deployment targets;
- MEP inspection, general workmanship, steel fabrication, fire assessment, and site-safety monitoring;
- hidden/internal damage diagnosis;
- live video processing and temporal tracking;
- operational 3D, thermal, GPR, or historical-record fusion;
- automated structural conclusions, repair prescriptions, or code-compliance certification.

## 4. Defect scope

Primary:

- Crack, with optional visual subtypes: linear, network/alligator, and crack with deposit.

Secondary:

- Spalling.
- Honeycombing/rock pockets.
- Exposed rebar.
- Rust/corrosion staining.
- Efflorescence/leaching.

Control labels:

- No visible target defect.
- Unknown/other/review required.

The detailed annotation definitions are controlled by `docs/TAXONOMY.md`.

## 5. Accepted input

- JPEG or PNG RGB still images.
- Single-image and batch upload.
- Original image retained immutably.
- Optional metadata: site, building, floor/elevation, room/zone, element type, capture time, camera, GPS, UAV flight, calibration, and ground-sample distance.

The intake process must flag corrupt, unsupported, badly exposed, severely blurred, or extremely low-resolution images. A warning does not automatically mean rejection; the rejection policy will be calibrated with field data.

## 6. Required workflow

1. Create or reopen an inspection project.
2. Identify the site/building and optional floor, façade, zone, or element.
3. Upload one image or a batch.
4. Observe queued, processing, completed, or failed status.
5. Review candidate boxes/masks, class, confidence, and image-quality warnings.
6. Approve, reject, relabel, or escalate a finding without overwriting the immutable raw model prediction.
7. Export annotated images, versioned JSON/CSV records, and a PDF-style summary.
8. Reopen prior inspections with their source, model, taxonomy, and review provenance intact.

## 7. Required finding record

Every model result must be convertible to a structured record containing:

- project, site, building, inspection, source-image, and parent-image IDs;
- floor/elevation, room/zone, façade, and element when known;
- source URI/path, capture timestamp, camera, GPS/pose, calibration/GSD status;
- defect family, subtype, box, mask/polygon or RLE;
- pixel length, width, and area;
- physical value, unit, and scale method only when calibration is valid;
- model confidence, uncertainty/quality flags, and escalation state;
- immutable model decision plus separate reviewer decision;
- model, checkpoint, pipeline, taxonomy, and dataset-manifest versions.

This record is the future integration boundary for 3D, thermal, GPR, and historical evidence.

## 8. Measurement rule

The default beta reports pixel geometry. Millimetres or centimetres may only be shown when the image has a validated reference marker, camera/plane calibration, UAV GSD with acceptable geometry, or registered 3D scale. The interface must show the scale status and never present uncalibrated pixels as physical measurements.

## 9. Safety and human review

- Display “candidate visible condition—qualified review required.”
- Do not label absence of a model finding as proof that an element is safe.
- Retain no-defect decisions for audit.
- Escalate uncertain, low-quality, and out-of-distribution images.
- Preserve reviewer identity, timestamp, and changes.

## 10. Definition of done

The beta is complete when the workflow in section 6 functions with real inference results; the selected model is reproducible; failure states are honest; target-building validation is reported; outputs are exportable and reopenable; and limitations/licensing are visible in both the application and report.

Numerical acceptance thresholds remain a pre-release decision because they require target-site data. At minimum they must cover crack recall, smallest evaluated crack width in pixels, false alarms per image, mask quality, latency, peak VRAM, image rejection rate, and reviewer workload.

