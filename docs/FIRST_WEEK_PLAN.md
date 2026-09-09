# First Execution Gate — Three-Person Team

**Status:** Technical preparation substantially complete; owner/advisor and field-access decisions remain open.

## Objective

Finish the evidence needed to approve the first curated-data release and begin a truthful hardware feasibility test. This gate does not include full training.

## Data lead

- [Completed] Run and review `scripts/audit_datasets.py`.
- [Technical verification completed; legal approval pending] Verify archive readability and licenses.
- [Completed] Verify both CODEBRIM archives with portable 7-Zip and decoded image probes; standard ZIP readers remain disallowed for these legacy headers.
- Retry missing CUBIT image/label archives after Google Drive quota reset.
- Draft the first selection budget by source, class, domain, annotation type, and negative type.
- Propose group IDs for parent images, buildings, bridges, UAV flights, and related frames.

## Model lead

- [Completed] Create the isolated CUDA-enabled Python 3.11 environment.
- [Completed] Verify CUDA on the Quadro T2000.
- Confirm the advisor meant ResNet rather than RetinaNet.
- [Completed for feasibility] Review and acquire the model/checkpoint choices in `docs/EXPERIMENT_MATRIX.md`.
- Define the common detector and segmentation evaluation inputs.
- [Completed] Select and materialize 300 training-only feasibility images after provenance review.

## Product/evaluation lead

- Convert `docs/BETA_SPECIFICATION.md` into the first user-flow/wireframe checklist.
- Review the structured finding fields with the engineering reviewer.
- Draft the inspection PDF/CSV/JSON content contract.
- Define preliminary operational measurements: latency, false alarms per image, escalation rate, and review time.
- Plan target-building capture permissions and metadata.

## End-of-gate evidence

- Approved beta specification.
- Engineer-reviewed taxonomy or a logged list of disputed classes.
- Dataset inventory and license table.
- Archive health report.
- Proposed curated subset and frozen test sources.
- Exact model/checkpoint matrix.
- CUDA-enabled environment verification.
- Feasibility sample manifest.
- Product workflow and output-record review.

## Blocking decisions for the owner/advisor

- Confirm ResNet versus RetinaNet.
- Confirm whether the complete project will be open-source under AGPL-compatible terms if Ultralytics YOLO remains in the beta.
- Identify the qualified engineering reviewer.
- Identify target buildings/construction sites and image-use permissions.
