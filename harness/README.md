# BDI Web Harness

A local review harness for the pre-training integration stage. It supports image upload, processing status, candidate finding display, manual approve/reject/relabel, and JSON/CSV/annotated-image exports.

The harness intentionally creates a placeholder candidate because model training/inference happens elsewhere. It preserves the output contract: `prediction` is immutable and reviewer actions are stored under `review`.

## Run

From the repository root:

```powershell
python harness\server.py
```

Open <http://127.0.0.1:8765>.

Uploaded images and reviewed finding records are stored locally under `runs/harness/` (ignored by Git). Stop the server with `Ctrl+C`.

## API

- `GET /api/state`
- `POST /api/upload` with multipart field `image`
- `POST /api/findings/<finding_id>` with `{ "action": "approve" | "reject" | "relabel", "label": "crack" }`
- `GET /api/export/json`
- `GET /api/export/csv`
- `GET /api/export/image`
