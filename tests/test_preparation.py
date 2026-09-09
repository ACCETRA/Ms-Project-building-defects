from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PreparationTests(unittest.TestCase):
    def test_feasibility_selection_contract(self) -> None:
        path = ROOT / "data" / "manifests" / "feasibility_v0.csv"
        rows = read_csv(path)
        required = {
            "sample_id",
            "source_dataset",
            "source_version",
            "source_path",
            "source_image_id",
            "parent_image_id",
            "group_id",
            "split",
            "capture_mode",
            "asset_domain",
            "annotation_type",
            "native_labels",
            "canonical_labels",
            "annotation_status",
            "mapping_version",
            "license_id",
            "quality_flags",
            "selection_reason",
            "sha256",
        }
        self.assertEqual(len(rows), 300)
        self.assertTrue(required.issubset(rows[0]))
        self.assertEqual(len({row["sample_id"] for row in rows}), 300)
        self.assertEqual(len({row["group_id"] for row in rows}), 300)
        self.assertEqual({row["split"] for row in rows}, {"train"})
        self.assertFalse(
            any(
                "/test/" in f"{row['image_locator']} {row['annotation_locator']}".lower()
                for row in rows
            )
        )

    def test_materialized_feasibility_hashes(self) -> None:
        path = ROOT / "data" / "manifests" / "feasibility_v0_materialized.csv"
        rows = read_csv(path)
        self.assertEqual(len(rows), 300)
        for row in rows:
            image_path = ROOT / row["output_image"]
            annotation_path = ROOT / row["output_annotation"]
            self.assertTrue(image_path.is_file())
            self.assertTrue(annotation_path.is_file())
            self.assertEqual(
                hashlib.sha256(image_path.read_bytes()).hexdigest(), row["sha256"]
            )
            self.assertEqual(
                hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                row["annotation_sha256"],
            )
            self.assertEqual(row["materialized_readable"], "True")

    def test_codebrim_registry_is_parent_safe(self) -> None:
        path = ROOT / "data" / "manifests" / "codebrim_classification_source_v1.csv"
        rows = read_csv(path)
        self.assertEqual(len(rows), 7729)
        split_counts: dict[str, int] = {}
        parent_splits: dict[str, set[str]] = {}
        for row in rows:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            parent_splits.setdefault(row["parent_id"], set()).add(row["split"])
        self.assertEqual(split_counts, {"test": 632, "train": 6481, "val": 616})
        self.assertFalse([parent for parent, splits in parent_splits.items() if len(splits) > 1])

    def test_cubit_registry_excludes_locked_test(self) -> None:
        path = ROOT / "data" / "manifests" / "cubit_train_val_source_v1.csv"
        rows = read_csv(path)
        self.assertEqual(len(rows), 6295)
        self.assertEqual(
            dict(Counter(row["split"] for row in rows)),
            {"train": 5596, "val": 699},
        )
        self.assertNotIn("test", {row["split"] for row in rows})
        self.assertTrue(all(row["annotation_type"] == "instance_polygon" for row in rows))
        self.assertTrue(all("dedup_pending" in row["quality_flags"] for row in rows))

    def test_cubit_exact_dedup_manifest_is_split_safe(self) -> None:
        path = ROOT / "data" / "manifests" / "cubit_exact_dedup_decisions_v1.csv"
        rows = read_csv(path)
        selected = [
            row for row in rows if row["selected_after_exact_dedup"] == "True"
        ]
        self.assertEqual(len(rows), 6996)
        self.assertEqual(
            dict(Counter(row["split"] for row in selected)),
            {"test": 694, "val": 678, "train": 5035},
        )
        split_by_hash: dict[str, set[str]] = {}
        for row in selected:
            split_by_hash.setdefault(row["sha256"], set()).add(row["split"])
        self.assertFalse(
            {sha256: splits for sha256, splits in split_by_hash.items() if len(splits) > 1}
        )
        duplicate_audit = json.loads(
            (ROOT / "artifacts" / "data-audit" / "cubit_duplicate_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(duplicate_audit["images_processed"], 6996)
        self.assertEqual(duplicate_audit["exact_duplicate_groups"], 589)
        self.assertEqual(duplicate_audit["cross_split_exact_pair_count"], 222)
        self.assertFalse(duplicate_audit["test_labels_opened"])

    def test_finding_schema_and_example(self) -> None:
        try:
            import jsonschema
        except ModuleNotFoundError:
            self.skipTest("jsonschema is not installed")
        schema = json.loads(
            (ROOT / "schemas" / "finding_record.schema.json").read_text(encoding="utf-8")
        )
        example = json.loads(
            (ROOT / "examples" / "finding_record.example.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(example, schema)

    def test_audit_reports_are_current(self) -> None:
        materialization = json.loads(
            (ROOT / "artifacts" / "data-audit" / "feasibility_materialization_report.json").read_text(encoding="utf-8")
        )
        codebrim = json.loads(
            (ROOT / "artifacts" / "data-audit" / "codebrim_7zip_verification.json").read_text(encoding="utf-8")
        )
        self.assertEqual(materialization["samples_verified"], 300)
        self.assertTrue(materialization["all_readable"])
        self.assertEqual(codebrim["classification_archive"]["full_test_exit_code"], 0)
        self.assertEqual(codebrim["original_archive"]["full_test_exit_code"], 0)

        cubit = json.loads(
            (ROOT / "artifacts" / "data-audit" / "cubit_archive_inventory.json").read_text(encoding="utf-8")
        )
        self.assertTrue(cubit["acquisition_complete_and_verified"])
        self.assertFalse(cubit["source_archives_modified"])
        self.assertFalse(cubit["test_split_extracted"])
        self.assertEqual(len(cubit["archives"]), 6)
        self.assertTrue(all(row["integrity"]["everything_ok"] for row in cubit["archives"]))
        self.assertTrue(all(row["count_matches_expected"] for row in cubit["archives"]))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in cubit["archives"]))
        self.assertTrue(all(row["paired"] for row in cubit["splits"]))


if __name__ == "__main__":
    unittest.main()
