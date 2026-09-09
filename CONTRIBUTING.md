# Contributing to the Building Defect Inspection Beta

This repository is meant to keep project work moving even when some decisions are intentionally waiting on the project owner or external stakeholders. The goal is to prevent lost productivity while preserving evidence quality.

## 1. Project status and guardrails

The current workspace is in the data-preparation and feasibility phase. The project has already verified:

- all six CUBIT archives were acquired, CRC-tested, hashed, and paired;
- the exact duplicate leak and leakage-clean test policy are documented;
- the Python environment and model runtime smoke tests are complete;
- the feasibility set and source registries are materialized.

Important guardrails:

- Do not start model training before the dataset freeze is approved.
- Do not treat any one-image smoke test as accuracy evidence.
- Do not inspect or tune on the official locked test split.
- Do not modify source archives or raw dataset files.
- Keep all derived manifests and reports versioned and reproducible.

## 2. What is blocked vs. what is safe to do now

### Blocked until owner/advisor decisions are made

These are not optional technical tasks; they need a user decision before finalization:

1. ResNet vs RetinaNet
2. Engineering reviewer identity and review schedule
3. Target buildings and site image-use permissions
4. YOLO licensing/open-source direction
5. Whether physical crack dimensions are required for beta 1 and how they will be calibrated

These decisions are recorded in:

- [docs/FIRST_WEEK_PLAN.md](docs/FIRST_WEEK_PLAN.md)
- [docs/ROADMAP_V2.md](docs/ROADMAP_V2.md)
- [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

### Safe work contributors can do immediately

These tasks can proceed in parallel without risking the final model route:

- review the duplicate/near-duplicate queue and rank candidates for human review;
- finalize the curated v0.1 selection logic and inclusion/exclusion rules;
- define group IDs and source provenance rules for the mega dataset;
- draft task-specific dataset manifests for classification, detection, and segmentation views;
- build the negative-example and verified-negative policy;
- document the target-site capture plan by whole building/session;
- prepare the beta application workflow and output contract review checklist;
- validate script reproducibility and documentation links;
- prepare issue lists and contributor notes for the next delivery gate.

## 3. Current repo map

Start here:

- [README.md](README.md)
- [docs/ROADMAP_V2.md](docs/ROADMAP_V2.md)
- [docs/BETA_SPECIFICATION.md](docs/BETA_SPECIFICATION.md)
- [docs/DATASET_CONTRACT.md](docs/DATASET_CONTRACT.md)
- [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md)
- [docs/TAXONOMY.md](docs/TAXONOMY.md)
- [docs/OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md)
- [PROJECT_READINESS.md](PROJECT_READINESS.md)

Useful generated artifacts:

- [artifacts/data-audit/cubit_archive_inventory.json](artifacts/data-audit/cubit_archive_inventory.json)
- [data/manifests/cubit_train_val_source_v1.csv](data/manifests/cubit_train_val_source_v1.csv)
- [data/manifests/cubit_exact_dedup_decisions_v1.csv](data/manifests/cubit_exact_dedup_decisions_v1.csv)
- [artifacts/data-audit/cubit_near_duplicate_review_report.json](artifacts/data-audit/cubit_near_duplicate_review_report.json)
- [artifacts/model-assets/model_asset_inventory.json](artifacts/model-assets/model_asset_inventory.json)
- [artifacts/model-feasibility/summary.json](artifacts/model-feasibility/summary.json)

## 4. Recommended contributor workflow

### Step A — repo setup

```powershell
# from the repo root
.
.
.
.
```

Use the project virtual environment and the repo's pinned audit dependencies:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-audit.txt
```

Then run the preparation checks:

```powershell
.
.
```

```powershell
.
```

```powershell
.
```

### Step B — choose a workstream

Pick a lane that does not depend on the missing owner decisions.

#### Data curation lane

- Review [docs/CURATED_DATASET_V0_1_PROPOSAL.md](docs/CURATED_DATASET_V0_1_PROPOSAL.md)
- Update manifests under [data/manifests](data/manifests)
- Validate the exact-dedup policy in [scripts/build_cubit_exact_dedup_manifest.py](scripts/build_cubit_exact_dedup_manifest.py)
- Rank unresolved perceptual candidates for review in [scripts/rank_cubit_near_duplicates.py](scripts/rank_cubit_near_duplicates.py)

#### Documentation and governance lane

- Update contract and policy docs
- Capture open assumptions and unresolved decisions in issue tracker notes
- Ensure the README and docs remain consistent with the actual state

#### Product and workflow lane

- Review [docs/BETA_SPECIFICATION.md](docs/BETA_SPECIFICATION.md)
- Review [docs/OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md)
- Draft the end-to-end beta workflow and reviewer experience

#### Validation lane

- Re-run the project tests after any documentation or manifest update

```powershell
.
```

## 5. Validation commands to run before opening a PR

Use the project checks before merging any update:

```powershell
.
.
```

```powershell
.
```

These are the repository's current baseline checks:

- unit tests from [tests/test_preparation.py](tests/test_preparation.py)
- dataset manifest integrity checks
- JSON/YAML validity checks
- link checks as defined by the project tooling
- no accidental modifications to raw archive files

## 6. What not to do

- Do not train on the locked publisher test split.
- Do not include target-site or unseen-building images in tuning sets.
- Do not claim model accuracy from one-image smoke tests.
- Do not rewrite the CUBIT archive inventory or alter raw archives.
- Do not call a result “final” until the relevant owner decisions are recorded.

## 7. Contributor priorities for the next delivery gate

When contributors are idle or blocked, the next best tasks are:

1. complete the curated v0.1 subset logic;
2. finalize the task-specific manifests;
3. produce the near-duplicate review queue and triage notes;
4. draft the target-site capture checklist and whole-building/session capture policy;
5. prepare the final beta app workflow and output record examples;
6. document the licensing review for YOLO and any deployment assumptions.

This is the highest-productivity path while owner decisions are pending.

## 8. Communication expectations

When you open or update a task, include:

- what was verified;
- what remains blocked by an owner decision;
- what is ready for review;
- which command or artifact proves the status.

This keeps the backlog evidence-based and reduces rework.

## 9. Final note

A contributor should assume this repo is in a disciplined research-prep stage: the data and model feasibility groundwork is complete, but the project is still waiting on governance and product decisions. The right contributor contribution is not to guess the final direction; it is to keep the repo ready, reproducible, and decision-ready while those decisions are pending.
