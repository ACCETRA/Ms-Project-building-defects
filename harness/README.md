# BDI Web Harness

A local FYP demonstration harness with real inference routes. It supports image upload, processing status, candidate finding display, manual approve/reject/relabel, and JSON/CSV/annotated-image/report exports.

The harness runs the available v1 YOLO detection, YOLO segmentation, ResNet classification, or Florence route selected in the interface. If a required checkpoint is unavailable, the route returns an explicit fallback/no-finding record rather than presenting a fabricated model result. The output contract is preserved: `prediction` is immutable and reviewer actions are stored under `review`.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe harness\server.py
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
- `GET /api/export/report`
