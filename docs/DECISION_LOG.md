# Decision Log

## 2026-09-07

| ID | Decision | Status | Consequence |
|---|---|---|---|
| D-001 | Build a functional academic beta rather than a notebook-only prototype | Approved | Product workflow, persistence, review, exports, and failure handling are first-class requirements |
| D-002 | Primary deployment domain is completed and under-construction buildings | Approved | Building/façade and target-site tests replace bridge-only evidence as the main validation |
| D-003 | Cracks are primary; spalling, honeycombing/rock pockets, exposed rebar, rust staining, and efflorescence are secondary | Approved | Canonical taxonomy and reporting language are frozen for the FYP demonstration |
| D-004 | Use a curated, balanced mega dataset rather than every source image | Approved | Task manifests select subsets while raw sources remain immutable |
| D-005 | Keep separate detection, segmentation, and classification views | Approved by data compatibility | Prevents boxes/image tags from being presented as genuine masks |
| D-006 | Compare Florence, YOLO, SAM, and a ResNet baseline by task and end to end | Approved | Provisional exact model matrix created |
| D-007 | Treat “REZNEK” as ResNet-50 | Approved on 2026-09-11 | ResNet-50 is the classification baseline; RetinaNet is deferred |
| D-008 | Exclude operational 3D/thermal/GPR/historical fusion | Completed scope boundary | Structured visual finding schema standardizes RGB defect inspection; external modalities are excluded from the FYP |
| D-009 | Report pixel measurements unless calibration is valid | Approved default | Prevents unsupported millimetre claims |
| D-010 | Use the installed GPU first; larger GPU is optional | Approved correction | Local environment/hardware feasibility precedes any cloud requirement |
| D-011 | Treat local GPU as Quadro T2000 4 GB unless a separate T3000 machine is identified | Observed | All local benchmarks record this exact device |
| D-012 | Use official unbalanced CODEBRIM classification data rather than the publisher's pre-oversampled release | In progress | Balancing remains a controlled training decision |
| D-013 | Use a 300-sample training-only feasibility set before model acquisition | Completed | 120 CiF, 80 S2DS, 40 UAV75, and 60 DACL10K samples are materialized; this set cannot be reported as an accuracy benchmark |
| D-014 | Use portable 7-Zip for the verified CODEBRIM archives | Resolved | Both full archives pass 7-Zip tests and 11 extracted image probes decode; Python `zipfile` and Windows `tar` remain disallowed because the archives use oversized legacy ZIP headers |
| D-015 | Do not substitute CPU PyTorch for the required CUDA test | Completed | Official CUDA wheels were hash-verified, installed in `.venv`, and passed a float16 GPU operation on the Quadro T2000 |
| D-016 | Run Florence from the official native-Transformers converted checkpoints | Resolved | `florence-community/Florence-2-*-ft` is the runtime source; Microsoft custom-code snapshots are retained as provenance references after a reproducible processor incompatibility was found |
| D-017 | Require a CUDA-only one-image preflight before the representative feasibility study | Completed | All six candidates execute in FP16 on the Quadro T2000; this proves runtime fit only and does not authorize accuracy claims or training |
| D-018 | Acquire CUBIT-InSeg completely while preserving its official test lock | Completed | Six official archives totaling 19,059,292,068 bytes pass CRC, member-count, and image/label pairing checks; SHA-256 values are recorded without profiling locked test-label contents |
| D-019 | Derive leakage-clean CUBIT views instead of trusting the publisher split unchanged | Completed for exact duplicates | 589 byte-identical `SP0`/`SP1` pairs include 222 cross-split pairs; `test > val > train` decisions retain 5,035/678/694 records with no cross-split exact hash, while 521 perceptual candidates remain review-only |
| D-020 | Use Florence Base/Large with segmentation as the full FYP pipeline | Approved for implementation phase | Florence-2 Base is the primary route and Florence-2 Large is the escalation route; the full route must produce localized findings and segmentation masks |
| D-021 | Accept the v0.1 visible-condition taxonomy for the academic FYP prototype without external engineering review | Approved for FYP prototype | Use the documented labels and `unknown_review`; do not introduce severity, repair, or structural-safety claims |
| D-022 | Require physical measurements in beta 1 | Approved with calibration constraint | Physical dimensions are emitted only for images with a valid reference marker, planar camera calibration, valid UAV GSD, or registered 3D scale; otherwise pixel geometry is retained and physical output is withheld |
| D-023 | Use royalty-free or explicitly licensed target images | Approved policy | Record source URL, license terms, retrieval date, and attribution requirements; this does not change the separate license obligations of public training datasets |
| D-024 | Release the academic beta as open source under AGPL-compatible terms | Approved on 2026-09-11 | Any delivered YOLO component must remain compatible with AGPL obligations; commercial distribution requires separate dataset permissions and any required commercial model licensing |
| D-025 | Use ResNet-50 + YOLO11n + SAM 2.1 as the comparison pipeline | Approved on 2026-09-11 | Compare the Florence full pipeline against a conventional classification, detection, and segmentation route; build the web demo only after real model results are available |
| D-026 | Treat the 300-sample runs as feasibility evidence, not full-data completion | Completed | Full v1 task manifests, full-data training, locked evaluations, and browser workflow completed; see `docs/V1_COMPARISON_REPORT.md` and `docs/FYP_COMPLETION_CHECKLIST.md` |
| D-027 | Approve the v1 source/license set, group-safe official-split policy, and `unknown_review` treatment | Approved on 2026-09-12 | Train only on approved source-train records; use source validation records or group-safe internal validation; keep locked test records isolated; preserve unknown geometry but exclude it from positive targets |
| D-028 | Treat the current 300-sample comparison and Florence runs as demo baselines | Approved on 2026-09-12 | Their metrics must not be presented as final accuracy or harness-readiness evidence |
| D-029 | Use staged compute for real training | Recommended for execution on 2026-09-12 | Quadro T2000 remains for smoke tests; use a supported 24 GB-class GPU for v1 YOLO/ResNet training and Florence Base LoRA fine-tuning; Florence Large or full-parameter tuning may require 40-48 GB; SAM remains prompted unless decoder adaptation is explicitly approved |

## Scope boundaries

- Target-building/site access is excluded from the accepted FYP scope; demonstration uses public test partitions and sample images.
- Physical scale measurement requires valid per-image calibration metadata; otherwise pixel coordinates are emitted.

## 2026-09-13

| ID | Decision | Status | Consequence |
|---|---|---|---|
| D-030 | Accept the current system as a demo-only academic FYP prototype, not a deployment-ready inspection system | Recorded | The browser harness, trained v1 checkpoints, source evaluations, review/export workflow, and documented limitations are sufficient for demonstration. Target-site validation and S2DS six-class labeling remain required before any deployment-testing claim. |
| D-031 | Formally accept binary-only S2DS reporting for this FYP unless an authoritative class-index manifest is later supplied | Recorded | The supplied S2DS test masks do not encode class identity. No six-class S2DS accuracy claim will be made. |
| D-032 | Record the human image-level error review as complete for FYP demo acceptance | Completed on 2026-09-13 | The completion and evidence boundary are documented in `docs/HUMAN_ERROR_REVIEW.md`. No unrecorded per-image counts or reviewer details are inferred. Human review is no longer an open demo gate, but manual review remains mandatory during use. |
