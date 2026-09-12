import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WebHarnessSmokeTests(unittest.TestCase):
    def test_status_summary_uses_real_repos_and_model_paths(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "web_harness",
            ROOT / "scripts" / "web_harness.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        status = module.summarize_harness_status()
        self.assertIn("checkpoint", status)
        self.assertIn("manifest", status)
        self.assertIn("status", status)
        self.assertIn("ready", status)
        self.assertTrue(status["checkpoint"].endswith("florence-community-2-base-ft"))
        self.assertTrue(status["manifest"].endswith("feasibility_v0_materialized.csv"))

    def test_status_summary_forwards_model_readiness(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "web_harness",
            ROOT / "scripts" / "web_harness.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        status = module.summarize_harness_status()
        self.assertIsInstance(status["ready"], bool)
        self.assertIsInstance(status["status"], str)
        self.assertIn(status["status"].lower(), {"ready", "warning", "blocked"})


if __name__ == "__main__":
    unittest.main()
