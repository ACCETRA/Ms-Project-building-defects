# Florence-First Feasibility Pipeline

**Pipeline version:** `0.1.0-florence-feasibility`  
**Scope:** CUDA-only, training-only feasibility execution

## What it does

`scripts/run_florence_pipeline.py` reads the hash-verified 300-sample feasibility
manifest, runs Florence-2 Base phrase grounding, and unloads Base before optionally
loading Florence-2 Large. Large is the escalation route for Base positives or low
sequence-likelihood outputs by default. Each emitted finding is validated against
`schemas/finding_record.schema.json` before any output files are finalized.

The generated-token geometric-mean likelihood is a routing proxy. Florence-2 does
not provide a calibrated per-box defect probability, so the value must not be
interpreted as one. Thresholds need a held-out building-domain validation set.

## Run

Use the isolated CUDA environment. A one-sample end-to-end verification is:

```powershell
.\.venv\Scripts\python.exe scripts\run_florence_pipeline.py --limit 1 --escalation all
```

Run the complete training-only feasibility set with:

```powershell
.\.venv\Scripts\python.exe scripts\run_florence_pipeline.py
```

Outputs go to `runs/florence/<run-id>/`:

- `findings.jsonl`: one schema-valid immutable record per candidate;
- `image_results.jsonl`: one audit record per image, including zero-candidate cases,
  raw parsed generations, Base/Large routing, quality warnings, and calibration state;
- `summary.json`: run scope, counts, environment, checkpoint digests, and limitations.

The run refuses non-training manifest records, missing or hash-mismatched images,
silent CPU fallback, unknown calibration sample IDs, and non-empty output folders.
Model results are cached after every image under the run directory. If a process is
interrupted, repeat the same command with the same `--run-id` and add `--resume`;
cached Base/Large results are reused. Malformed generated boxes are recorded and
excluded instead of aborting the run or entering the finding store.

## Calibration

Calibration is optional at the CLI but mandatory before physical dimensions can be
emitted. Supply `--calibrations <json>` using
`schemas/calibration_manifest.schema.json`. The pipeline accepts:

- an explicitly validated local `millimeters_per_pixel` value for a reference
  marker, UAV GSD, or registered 3D scale; or
- a non-singular 3 x 3 pixel-to-plane homography for planar camera calibration.

`uncertainty` is an absolute linear uncertainty in the selected output unit. Area
is emitted in the corresponding squared unit. Box geometry is only a localization
measurement; it is not a crack centerline/width measurement. Uncalibrated findings
retain pixels and the `uncalibrated_measurement` limitation, with no physical object.

The example calibration file contains demonstration numbers only. It is not valid
evidence for any real image.

## Interpretation boundary

This pipeline produces candidate visible conditions for qualified review. The
300 samples are drawn only from training partitions, so a run is not an accuracy
evaluation and cannot satisfy the release thresholds in `docs/TAXONOMY.md`.
No-candidate output is not evidence that a structure or element is safe.
