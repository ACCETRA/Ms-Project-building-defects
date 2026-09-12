# Third-Party Label Review Packet

## Purpose

This packet contains the 43 feasibility samples currently recorded in the Florence/comparison label ledger. Review the images against the native and normalized annotations, then record independent decisions in `review_decisions.csv`.

The packet is a review copy. Do not edit files under the original `data/` or `datasets/` directories, and do not inspect or modify locked test data.

## Contents

- `review_decisions.csv`: working copy of the label ledger.
- `packet_manifest.csv`: exact sample list, provenance, and SHA-256 hashes.
- `images/`: 43 images, named by `sample_id`.
- `native_annotations/`: native JSON or mask annotations, named by `sample_id`.
- `normalized_annotations/`: project-normalized annotations used by downstream preparation.
- `TAXONOMY.md`: canonical classes and review rules.
- `source_manifest.csv`: source manifest rows used to construct this packet.

## Nine review instructions

1. Open every image in `images/` and compare it with the matching files in `native_annotations/` and `normalized_annotations/` using the same `sample_id`.
2. Confirm that each marked region is visibly present in the image and that the polygon, mask, or box covers the intended region rather than a seam, edge, shadow, cable, graffiti, or ordinary texture.
3. Confirm the native label first; do not replace or erase it. Native labels are evidence of what the source dataset supplied.
4. Check the proposed canonical label against `TAXONOMY.md`. Use only the approved canonical names and preserve multi-label combinations with `|` separators.
5. Treat `S2DS corrosion -> rust_staining` as a weak visible-evidence mapping only. Do not interpret it as confirmed internal reinforcement corrosion.
6. Treat `DACL10K Rockpocket -> honeycombing_rock_pocket` as a strong mapping that still requires terminology confirmation from the reviewer.
7. Set `review_decision` to `accept`, `relabel`, or `reject`. If relabeling, enter the corrected value in `final_canonical_label`; if rejecting, explain why in `review_notes`.
8. Set `geometry_decision` to `accept`, `relabel`, or `reject`, and set `annotation_status` to `positive`, `unknown`, or another value allowed by the taxonomy. Do not turn incomplete annotation into a negative label.
9. Enter your real reviewer name, date or review batch identifier, and concise evidence in `review_notes`; after completing all rows, report disagreements and sign off the packet without changing source files.

## Decision rules

- Review the 43 rows in this packet only; they are training-partition feasibility samples, not final accuracy evidence.
- A source polygon or mask may support a derived box. A source box must not be converted into a ground-truth mask.
- If the image and annotation disagree, choose `relabel` or `reject` and explain the evidence. Do not silently repair geometry in place.
- `positive` means the source explicitly annotates a target. It does not mean the image has been independently certified as free from all other defects.
- Do not assign severity, structural safety, repair advice, or physical dimensions.

## Review outcome

The review is complete only when every row has a named independent reviewer, a decision, a geometry decision, an annotation status, and notes. The project owner should compare the completed ledger with the original `confirmation/annotation_qa_confirmation.csv` before freezing any training manifest.
