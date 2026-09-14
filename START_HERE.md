# Project Alpha Offline Handoff

**FYP status:** Complete  
**Package type:** Dataset-free offline runnable handoff

## Run the completed demonstration

1. Keep this folder at a short path without an apostrophe, such as `D:\ProjectAlpha`.
2. Install Python 3.11 from `vendor\installers\python-3.11.9-amd64.exe` if required.
3. Open PowerShell in this folder.
4. Verify the handoff:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_offline_bundle.ps1
```

5. Create the offline environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_offline_env.ps1
```

6. Start the local application:

```powershell
.\.venv\Scripts\python.exe harness\server.py
```

7. Open `http://127.0.0.1:8765`.

The model checkpoints and runtime dependencies required for inference are included. The application does not need a dataset to process uploaded JPEG or PNG images.

## Reports

- `artifacts/fyp.docx`: consolidated FYP completion and results report.
- `docs/V1_COMPARISON_REPORT.md`: detailed measured results and provenance.
- `docs/RECOMMENDED_IMPROVEMENTS.md`: optional accuracy and generalization improvements.
- `docs/FYP_COMPLETION_CHECKLIST.md`: completed handoff checklist.
- `docs/HUMAN_ERROR_REVIEW.md`: completed human-review record.

## Restore datasets

Datasets are excluded because the raw collection is approximately 60.9 GiB and some sources restrict redistribution. On an authorized internet-connected machine, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_datasets.ps1 -AcceptLicenses
```

See `docs/DATASET_REPRODUCTION_GUIDE.md` for exact source locations, generated storage paths, manual commands, validation, and licensing boundaries.
