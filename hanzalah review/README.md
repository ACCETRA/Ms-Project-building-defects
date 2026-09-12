# Hanzalah Review Packet

## Purpose

This packet is for human review of the **405 unresolved CUBIT-InSeg cross-split perceptual-duplicate candidates**. These pairs may show the same image, the same scene, or a near-duplicate captured from a different view. The review must be completed before the final v1 train/validation/test manifests are frozen.

The packet is intentionally metadata-only. It does not contain raw CUBIT archives or extracted test images. This protects the locked test set and avoids creating a second uncontrolled copy of the dataset.

## What the reviewer must decide

For every row in `review_queue.csv`, inspect both referenced CUBIT image members and set:

- `decision` to exactly one of:
  - `same_duplicate`: same underlying image or an effectively identical duplicate; exclude the lower-priority copy from the selected view.
  - `same_parent_near_duplicate`: clearly the same capture/scene/parent but not pixel-identical; keep the pair in one split and exclude or reassign the lower-priority record.
  - `different_image`: visually different images; retain both if all other rules permit.
  - `uncertain`: evidence is insufficient; do not make an automatic exclusion. Escalate to the project owner.
- `review_notes`: concise visual evidence, including what is shared and what differs.
- `reviewer_id`: the reviewer name or approved identifier.
- `reviewed_at`: an ISO date such as `2026-09-12`.

Do not change the record IDs, split names, archive members, rank, or image-analysis measurements.

## Files in this packet

- `review_queue.csv`: the 405 rows requiring review. This is the working file.
- `review_queue_original.csv`: untouched copy of the generated queue.
- `review_decisions_template.csv`: empty decision form with the required fields.
- `cubit_exact_dedup_decisions_v1.csv`: completed exact-duplicate decisions for context; do not rewrite it.
- `cubit_near_duplicate_review_report.json`: audit summary explaining why these pairs remain unresolved.
- `cubit_duplicate_audit.json`: duplicate and test-lock audit evidence.
- `cubit_train_val_source_v1.csv`: source registry for the CUBIT train/validation records.
- `TAXONOMY.md`: canonical label and annotation rules for context.
- `source_manifest.csv`: source inventory and license notes.

## Nine careful review steps

1. Open the first and second image identified by `left_member` and `right_member`, using the archive paths and split shown in the row.
2. Compare stable visual content: building or structure, facade geometry, cracks, spalling, camera angle, shadows, background, and distinctive marks.
3. Ignore ordinary compression, small brightness changes, resizing, and minor framing changes when deciding whether two images represent the same capture.
4. Mark `same_duplicate` only when the images are effectively the same image or an exact/near-exact duplicate of the same view.
5. Mark `same_parent_near_duplicate` when the images are different frames/views of the same underlying capture group or scene and placing them in separate splits would make validation/test optimistic.
6. Mark `different_image` only when there is clear evidence that the images are unrelated or sufficiently independent; a different filename alone is not evidence.
7. Mark `uncertain` when the image cannot be inspected, the view is ambiguous, or the evidence does not support a confident decision. Never guess.
8. Write the reason in `review_notes`, identify yourself in `reviewer_id`, and add `reviewed_at` for every row.
9. Run the completion checks below and report every `uncertain` row or disagreement to the project owner before the v1 freeze.

## Split and exclusion policy

The project uses this priority for exact duplicates:

`test > validation > train`

For a duplicate spanning splits, retain the highest-priority record in its split and exclude the lower-priority copy from the selected leakage-clean view. Do not move or edit the official source archives.

For perceptual near-duplicates, human review determines whether the pair must be kept in one split. A pair marked `same_parent_near_duplicate` must not remain split across train and validation/test. The project owner must record the final keep/exclude or reassignment decision in the v1 manifest.

The locked CUBIT test set must not be used for model training, threshold selection, annotation tuning, or repeated visual inspection beyond this duplicate review. This packet contains locators for the locked records only because duplicate-leakage review requires knowing whether a candidate crosses the test boundary.

## How to access the images

The queue provides archive members, for example:

- `left_split=train`, `left_member=images/SP0_1224.JPG`
- `right_split=test`, `right_member=images/SP1_1210.JPG`

Use the corresponding read-only archives under:

`datasets/building-target/CUBIT-InSeg/CUBIT-InSeg/<split>/images.zip`

Do not extract or modify label archives for this review. If an image cannot be safely inspected, mark `uncertain` and record why. Do not open test labels for this task.

## Completion checks

From the repository root, after saving the completed worksheet:

```powershell
$rows = Import-Csv 'hanzalah review/review_queue.csv'
$valid = @('same_duplicate','same_parent_near_duplicate','different_image','uncertain')
$missing = @($rows | Where-Object { $_.decision -notin $valid -or -not $_.reviewer_id -or -not $_.reviewed_at -or -not $_.review_notes })
"Rows: $($rows.Count)"
"Missing or invalid decisions: $($missing.Count)"
"Uncertain decisions: $(@($rows | Where-Object decision -eq 'uncertain').Count)"
```

Expected result:

- Rows: `405`
- Missing or invalid decisions: `0`
- Uncertain decisions: ideally `0`; any remaining uncertain rows require project-owner resolution.

After review, the project owner must:

1. Preserve this completed worksheet as review evidence.
2. Apply decisions to a new v1 selection manifest, never to raw archives.
3. Re-run exact-hash and group/split leakage checks.
4. Confirm that no selected train/validation/test records share a reviewed duplicate group.
5. Freeze the manifest with a version, reviewer identity, decision counts, and worksheet hash.

## Important boundaries

- These are duplicate decisions, not defect-label decisions.
- Do not relabel cracks, spalling, or other defects in this worksheet.
- Do not infer a building ID when the evidence does not support one.
- Do not delete files from `datasets/`.
- Do not call a `different_image` decision a proof of independent buildings or capture sessions.
- Do not use this review packet as model-accuracy evidence.
