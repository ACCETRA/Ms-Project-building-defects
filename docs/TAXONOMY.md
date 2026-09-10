# Canonical Defect Taxonomy and Annotation Guide

**Taxonomy ID:** BDI-TAX-001  
**Version:** 0.1.0-beta  
**Approval:** accepted for the academic Florence-first beta; engineering review is deferred

## Core principle

Labels describe visible evidence in an RGB image. They do not assert an unseen failure mechanism. In particular, a visible rust-colored stain is not sufficient proof of internal reinforcement corrosion.

## Canonical labels

| ID | Canonical label | Include | Exclude or flag |
|---:|---|---|---|
| 0 | `background` | Pixels outside a labelled target region in a comprehensively annotated mask | Do not interpret as a verified defect-free image when the source is only partially annotated |
| 1 | `crack` | Visible linear separation/fracture in concrete or masonry | Joints, formwork seams, cables, edges, shadows, paint/chalk marks |
| 2 | `spalling` | Local surface material loss, flaking, or detached concrete/masonry | Deliberate openings, rough unfinished edges, shallow staining |
| 3 | `honeycombing_rock_pocket` | Voids/coarse aggregate exposure caused by incomplete consolidation or rock-pocket appearance | Ordinary texture, exposed aggregate finish, deep spalling when the distinction is uncertain |
| 4 | `exposed_rebar` | Reinforcement visibly exposed at the surface | Cables, conduits, tie wire, scaffolding, unrelated metalwork |
| 5 | `rust_staining` | Visible rust-colored staining or oxidation evidence | Do not rename as confirmed internal corrosion without additional evidence |
| 6 | `efflorescence_leaching` | Visible mineral/salt deposit or leaching product | Paint, dust, glare, ordinary light-colored texture |
| 7 | `no_visible_target_defect` | Image reviewed as containing none of labels 1–6 | An image with missing/partial annotations is not automatically negative |
| 8 | `unknown_review` | Ambiguous candidate, unsupported condition, or uncertain boundary | Must not be silently forced into a known class |

## Crack subtypes

Subtypes are optional attributes under `crack`; they are not independent mutually exclusive top-level classes.

| Subtype | Meaning |
|---|---|
| `linear` | One or more predominantly linear crack traces |
| `network_alligator` | Interconnected polygonal/network cracking |
| `with_deposit` | Crack associated with visible precipitation/mineral deposit |
| `unspecified` | Source label or image does not support a reliable subtype |

## Source mapping

| Source label | Canonical target | Mapping strength | Note |
|---|---|---|---|
| CiF `Crack` | `crack` + `linear/unspecified` | Direct | Preserve native instance and source label |
| CiF `Net-Crack` | `crack` + `network_alligator` | Direct | Do not flatten away the subtype in source metadata |
| CiF `Crack with Precipitation` | `crack` + `with_deposit` | Direct | Does not equal generic efflorescence everywhere in the image |
| CiF `Rust` | `rust_staining` | Direct visual mapping | Evidence label only |
| CiF `Spalling` | `spalling` | Direct | — |
| CiF `Algae` | `unknown_review` or excluded auxiliary | Out of v0 core | May be restored in a later taxonomy |
| DACL10K `Crack` | `crack` + `linear/unspecified` | Direct | Multi-label semantic polygon |
| DACL10K `ACrack` | `crack` + `network_alligator` | Direct | — |
| DACL10K `Rockpocket` | `honeycombing_rock_pocket` | Strong | Reviewer to confirm terminology |
| DACL10K `ExposedRebars` | `exposed_rebar` | Direct | — |
| DACL10K `Rust` | `rust_staining` | Direct visual mapping | — |
| DACL10K `Efflorescence` | `efflorescence_leaching` | Direct | — |
| DACL10K `Spalling` | `spalling` | Direct | — |
| DACL10K `WConccor` | `unknown_review` | Do not map to rebar corrosion | Object/washout label, not the same concept |
| CODEBRIM `crack` | `crack` | Direct at native annotation level | Box/image labels are not pixel masks |
| CODEBRIM `spallation`/spalling equivalent | `spalling` | Review native XML spelling | Preserve original string |
| CODEBRIM `exposed bars` equivalent | `exposed_rebar` | Review native XML spelling | Preserve original string |
| CODEBRIM `efflorescence` | `efflorescence_leaching` | Direct at native annotation level | — |
| CODEBRIM `corrosion` | `rust_staining` | Weak semantic mapping | Visible evidence only; record mapping as weak |
| S2DS `crack` | `crack` | Direct | Semantic mask |
| S2DS `spalling` | `spalling` | Direct | Semantic mask |
| S2DS `corrosion` | `rust_staining` | Weak semantic mapping | Verify examples during QA |
| S2DS `efflorescence` | `efflorescence_leaching` | Direct | Semantic mask |
| S2DS `vegetation` | `unknown_review` or excluded auxiliary | Out of v0 core | — |
| S2DS `control point` | context metadata | Not a defect | Useful for calibration/georeferencing |
| CUBIT-InSeg `crack` | `crack` | Direct | Building-façade UAV instance mask |
| CUBIT-InSeg `spalling` | `spalling` | Direct | Building-façade UAV instance mask |
| UAV75 crack line | `crack` | Direct with geometry flag | Line-style ground truth may need width handling |
| UAV75 planking | hard-negative/context label | Not a target defect | Retain to study false positives |

## Annotation status

Every image/class combination must distinguish:

- `positive`: explicitly annotated target exists;
- `verified_negative`: reviewer/source guarantees absence;
- `unknown`: the class may be present but was not annotated;
- `not_applicable`: label does not apply to the task/view.

Unknown must never be converted to negative merely because no annotation was supplied.

## Geometry policy

- Preserve native polygon/mask geometry and native boxes.
- A mask may generate a derived box, labelled `derived_from_mask`.
- A box must not generate a claimed ground-truth mask.
- Pseudo masks must be stored separately from human ground truth and must identify model/checkpoint/threshold.
- Tile coordinates must be reversible to the parent image.
- Crack skeletons/centerlines and width estimates are derived outputs, not replacements for the source mask.

## Severity policy

Severity grades are excluded from taxonomy v0.1. A reviewer-approved engineering rule and calibrated measurements are required before introducing severity labels.

## Beta acceptance thresholds

These are provisional release gates for the Florence-first beta, not claims of engineering adequacy. They must be measured on locked validation and unseen target-building data before release:

- crack recall: at least 0.90 on the locked building-domain test;
- crack false-negative rate: at most 0.10 on the same test, with a documented review of every sampled false negative;
- false alarms: at most 2 candidate findings per image on the target-building test median;
- physical measurement coverage: 100% of findings presented with physical dimensions must have a valid recorded scale method and uncertainty;
- uncalibrated images: never display millimetres or centimetres; retain pixel geometry and the `uncalibrated_measurement` limitation;
- no severity, repair, or structural-safety conclusion may be emitted by the beta.

