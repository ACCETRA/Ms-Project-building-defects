# Finding and Export Contract

**Contract ID:** BDI-OUTPUT-001  
**Status:** Implemented for the demo-only FYP prototype

The schema is the handoff boundary for both routes: Florence findings and YOLO -> SAM findings use the same record shape. The local browser workflow implements upload, real inference, separate human review, persistence, and JSON/CSV/annotated-image/report exports.
**Machine schema:** `schemas/finding_record.schema.json`

## Core rule

The beta stores the model prediction and the human review as separate facts. Reviewers may approve, reject, or relabel a finding, but they do not overwrite the original prediction or its model/checkpoint provenance.

## Coordinate and measurement rules

- Boxes use source-image pixel coordinates in `[x, y, width, height]` order.
- Masks use COCO RLE or explicit polygon points and identify their encoding.
- Pixel length, maximum width, and area may be emitted when the geometry supports them.
- A physical measurement object is forbidden unless `scale_status` is `valid` and a non-`none` scale method is recorded.
- A valid scale method still requires uncertainty to be reported; it does not make a model result an engineering conclusion.

## Product exports

The same stored records drive all output formats:

- JSON preserves the complete schema and provenance.
- CSV flattens one finding per row and links mask geometry as a file or stable URI.
- Annotated images render accepted display labels without modifying the source image.
- The PDF-style report summarizes inspection metadata, reviewed findings, rejected/failed images, measurement status, limitations, and model/dataset versions.

## Evidence links

`evidence_links` can retain references to RGB, thermal, GPR, 3D, and historical records. The completed FYP demo processes RGB still images; other evidence-link fields are provenance metadata only and are not presented as multimodal inference.

## Safety wording

Every user-facing report must include both `manual_review_required` and `not_structural_safety_determination` limitations. An image with no finding is not evidence that the structure or element is safe.
