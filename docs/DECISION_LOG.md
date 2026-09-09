# Decision Log

## 2026-09-07

| ID | Decision | Status | Consequence |
|---|---|---|---|
| D-001 | Build a functional academic beta rather than a notebook-only prototype | Approved | Product workflow, persistence, review, exports, and failure handling are first-class requirements |
| D-002 | Primary deployment domain is completed and under-construction buildings | Approved | Building/façade and target-site tests replace bridge-only evidence as the main validation |
| D-003 | Cracks are primary; spalling, honeycombing/rock pockets, exposed rebar, rust staining, and efflorescence are secondary | Approved in principle | Canonical taxonomy created; engineering wording review still required |
| D-004 | Use a curated, balanced mega dataset rather than every source image | Approved | Task manifests select subsets while raw sources remain immutable |
| D-005 | Keep separate detection, segmentation, and classification views | Approved by data compatibility | Prevents boxes/image tags from being presented as genuine masks |
| D-006 | Compare Florence, YOLO, SAM, and a ResNet baseline by task and end to end | Approved | Provisional exact model matrix created |
| D-007 | Treat “REZNEK” as ResNet pending confirmation | Pending advisor confirmation | ResNet-50 is provisional; RetinaNet would change the comparison track |
| D-008 | Defer operational 3D/thermal/GPR/historical fusion | Recommended and recorded | Structured visual finding schema preserves a future fusion interface |
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

## Open decisions

- Advisor confirmation: ResNet or RetinaNet.
- Engineering reviewer identity and availability.
- Target-building/site access and image-use permission.
- Whether the complete beta will be open source under AGPL-compatible terms if Ultralytics remains in the delivered product.
- Whether physical crack dimensions are required for the first release and, if so, which capture-scale method will be used.
