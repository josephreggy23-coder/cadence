import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed_robustness import summarize_runs  # noqa: E402


class SeedRobustnessTests(unittest.TestCase):
    def test_summary_preserves_direction_and_range(self):
        runs = [
            {
                "smoothed_accuracy": 0.90,
                "causal_accuracy": 0.83,
                "causal_refractory_recall": 0.70,
                "causal_intact_b1": 0.70,
                "causal_blocked_b1": -0.02,
                "causal_b1_contrast": 0.72,
                "causal_b1_contrast_ci95": [0.30, 1.10],
            },
            {
                "smoothed_accuracy": 0.88,
                "causal_accuracy": 0.80,
                "causal_refractory_recall": 0.66,
                "causal_intact_b1": 0.60,
                "causal_blocked_b1": 0.03,
                "causal_b1_contrast": 0.57,
                "causal_b1_contrast_ci95": [0.10, 0.90],
            },
        ]
        summary = summarize_runs(runs)
        self.assertEqual(summary["direction_checks"]["positive_causal_contrasts"], 2)
        self.assertEqual(
            summary["direction_checks"]["contrast_intervals_excluding_zero"], 2
        )
        self.assertEqual(summary["direction_checks"]["total_seed_replicates"], 2)
        contrast = summary["descriptive_statistics"]["causal_b1_contrast"]
        self.assertAlmostEqual(contrast["minimum"], 0.57)
        self.assertAlmostEqual(contrast["maximum"], 0.72)

    def test_summary_requires_at_least_one_run(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_runs([])

    def test_versioned_five_seed_benchmark_keeps_its_documented_guardrails(self):
        result_path = ROOT / "results" / "seed_robustness.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        runs = payload["per_seed"]
        self.assertEqual(len(runs), 5)
        self.assertTrue(all(float(run["causal_accuracy"]) >= 0.70 for run in runs))
        self.assertTrue(all(float(run["causal_b1_contrast"]) > 0 for run in runs))
        self.assertGreaterEqual(
            sum(float(run["causal_b1_contrast_ci95"][0]) > 0 for run in runs), 4
        )


if __name__ == "__main__":
    unittest.main()
