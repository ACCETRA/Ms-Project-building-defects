from __future__ import annotations

import json
import csv
from pathlib import Path
import unittest

import jsonschema

from bdi.findings import build_finding, validate_findings
from bdi.florence import canonicalize_label, deduplicate, Detection
from bdi.measurement import Calibration, measure_box


ROOT = Path(__file__).resolve().parents[1]


class FlorencePipelineUnitTests(unittest.TestCase):
    def test_normalized_task_view_manifests(self) -> None:
        names = ("master", "detection", "segmentation", "classification", "florence_product")
        for name in names:
            path = ROOT / "data" / "manifests" / f"feasibility_{name}_v0_1.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 300, name)
            self.assertTrue(all(row["mapping_version"] == "BDI-TAX-001@0.1.0-beta" for row in rows))
            self.assertTrue(all(row["geometry_qc"] == "passed" for row in rows))

        report = json.loads(
            (ROOT / "artifacts" / "data-audit" / "feasibility_task_views_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["normalized_samples"], 300)
        self.assertEqual(report["target_annotations"], 812)
        self.assertEqual(report["rejected_malformed_geometries"], 4)
        self.assertTrue(report["geometry_qc_passed"])
        self.assertFalse(report["policy"]["image_labels_promoted_to_boxes"])
        self.assertFalse(report["policy"]["boxes_promoted_to_masks"])

    def test_unmapped_dacl_labels_are_preserved_as_unknown(self) -> None:
        payload = json.loads(
            (ROOT / "data" / "feasibility_v0" / "normalized" / "dacl_0001.json").read_text(
                encoding="utf-8"
            )
        )
        unknown = [
            annotation for annotation in payload["annotations"]
            if annotation["canonical_label"] == "unknown_review"
        ]
        self.assertEqual(len(unknown), 10)
        unmapped = [
            annotation for annotation in unknown
            if annotation["native_label"] != "Rockpocket"
        ]
        self.assertEqual(
            {annotation["native_label"] for annotation in unmapped},
            {"Cavity", "Hollowareas", "Wetspot", "Weathering", "WConccor"},
        )
        self.assertTrue(all(annotation["annotation_status"] == "unknown" for annotation in unknown))

    def test_reviewer_decision_is_reflected_in_normalized_annotation(self) -> None:
        payload = json.loads(
            (ROOT / "data" / "feasibility_v0" / "normalized" / "dacl_0001.json").read_text(
                encoding="utf-8"
            )
        )
        reviewed = next(
            annotation for annotation in payload["annotations"] if annotation["annotation_id"] == "shape-19"
        )
        self.assertEqual(reviewed["native_label"], "Rockpocket")
        self.assertEqual(reviewed["canonical_label"], "unknown_review")
        self.assertEqual(reviewed["annotation_status"], "unknown")
        self.assertEqual(reviewed["mapping_strength"], "reviewed")
        self.assertEqual(reviewed["reviewer_id"], "shah231")

    def test_dacl_unknown_source_labels_are_not_dropped(self) -> None:
        expected = {"Cavity", "Hollowareas", "Wetspot", "Weathering", "WConccor"}
        source_labels: set[str] = set()
        normalized_labels: set[str] = set()
        for path in (ROOT / "data" / "feasibility_v0" / "annotations" / "dacl").glob("*.json"):
            source = json.loads(path.read_text(encoding="utf-8"))
            normalized = json.loads(
                (ROOT / "data" / "feasibility_v0" / "normalized" / path.name).read_text(
                    encoding="utf-8"
                )
            )
            source_labels.update(
                shape["label"] for shape in source.get("shapes", []) if shape.get("label") in expected
            )
            normalized_labels.update(
                annotation["native_label"]
                for annotation in normalized.get("annotations", [])
                if annotation.get("native_label") in expected
            )
        self.assertEqual(source_labels, expected)
        self.assertEqual(normalized_labels, expected)

    def test_calibration_schema_and_example(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "calibration_manifest.schema.json").read_text(encoding="utf-8")
        )
        example = json.loads(
            (ROOT / "examples" / "calibration_manifest.example.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(example, schema)

    def test_uncalibrated_measurement_withholds_physical_values(self) -> None:
        measurement = measure_box([10, 20, 30, 80])
        self.assertEqual(measurement["scale_status"], "not_provided")
        self.assertNotIn("physical", measurement)
        self.assertEqual(measurement["pixel"]["length"], 80)

    def test_isotropic_and_homography_measurements(self) -> None:
        scale = Calibration.from_mapping(
            {
                "scale_status": "valid",
                "scale_method": "reference_marker",
                "unit": "mm",
                "uncertainty": 0.5,
                "millimeters_per_pixel": 0.25,
            }
        )
        physical = measure_box([0, 0, 20, 40], scale)["physical"]
        self.assertEqual(physical["length"], 10)
        self.assertEqual(physical["max_width"], 5)
        self.assertEqual(physical["area"], 50)

        homography = Calibration.from_mapping(
            {
                "scale_status": "valid",
                "scale_method": "camera_planar_calibration",
                "unit": "cm",
                "uncertainty": 2,
                "homography_pixel_to_mm": [[2, 0, 0], [0, 3, 0], [0, 0, 1]],
            }
        )
        physical = measure_box([0, 0, 20, 40], homography)["physical"]
        self.assertEqual(physical["length"], 12)
        self.assertEqual(physical["max_width"], 4)
        self.assertEqual(physical["area"], 48)
        self.assertEqual(physical["uncertainty"], 0.2)

    def test_label_mapping_and_deduplication(self) -> None:
        self.assertEqual(canonicalize_label("network crack"), ("crack", "network_alligator"))
        self.assertEqual(
            canonicalize_label("exposed reinforcement"), ("exposed_rebar", "not_applicable")
        )
        detections = [
            Detection("crack", "unspecified", [0, 0, 10, 10], 0.8, "crack"),
            Detection("crack", "unspecified", [0, 0, 10, 10], 0.7, "crack"),
        ]
        self.assertEqual(len(deduplicate(detections)), 1)

    def test_built_finding_is_schema_valid(self) -> None:
        sample = {
            "sample_id": "test_0001",
            "source_dataset": "unit-test",
            "source_image_id": "image-1",
            "parent_image_id": "parent-1",
            "group_id": "group-1",
            "capture_mode": "unknown",
            "sha256": "0" * 64,
        }
        finding = build_finding(
            run_id="unit-test-run",
            sample=sample,
            image_path=ROOT / "data" / "dummy.jpg",
            image_size=(100, 80),
            defect_family="crack",
            subtype="unspecified",
            box_xyxy=[10, 10, 40, 60],
            confidence_proxy=0.75,
            pipeline_route=["florence-2-base-ft"],
            escalation_state="not_required",
            checkpoint_sha256="1" * 64,
            model_id="florence-2-base-ft",
            quality={"decision": "accepted", "flags": [], "notes": None},
            calibration=None,
        )
        validate_findings([finding])
        self.assertNotIn("physical", finding["measurement"])
        self.assertIn("uncalibrated_measurement", finding["limitation_codes"])


if __name__ == "__main__":
    unittest.main()
