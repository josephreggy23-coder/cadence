import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_synthetic import (
    generate_dataset,
    load_dataset,
    simulation_metadata,
    state_occupancy,
    simulate_trace,
    validate_dataset,
)


class SimulateTraceTests(unittest.TestCase):
    def test_rejects_invalid_frame_count(self):
        with self.assertRaisesRegex(ValueError, "n_frames"):
            simulate_trace(np.random.default_rng(1), 0, -3.0, 0.9)

    def test_rejects_invalid_load_decay(self):
        with self.assertRaisesRegex(ValueError, "load_decay"):
            simulate_trace(np.random.default_rng(1), 10, -3.0, 0.9, load_decay=1.1)

    def test_returns_a_load_value_for_each_frame(self):
        _, states, time, load = simulate_trace(np.random.default_rng(1), 10, -3.0, 0.9)
        self.assertEqual(states.shape, time.shape)
        self.assertEqual(states.shape, load.shape)
        self.assertTrue(np.all(load >= 0))

    def test_state_occupancy_is_a_normalized_summary(self):
        summary = state_occupancy(np.array([0, 0, 2, 3]))
        self.assertEqual(summary["QUIESCENT"], 0.5)
        self.assertAlmostEqual(sum(summary.values()), 1.0)

    def test_dataset_generation_is_deterministic_and_labeled(self):
        first = generate_dataset("intact", n_traces=2, n_frames=3, seed=4)
        second = generate_dataset("intact", n_traces=2, n_frames=3, seed=4)
        self.assertEqual(len(first), 6)
        self.assertListEqual(
            list(first.columns),
            ["condition", "trace_id", "time_s", "calcium", "true_state", "load"],
        )
        self.assertTrue(first.equals(second))

    def test_manifest_records_the_feedback_condition(self):
        manifest = simulation_metadata("blocked", 2, 3, 4, 0.8)
        self.assertEqual(manifest["beta1"], 0.02)
        self.assertEqual(manifest["frame_interval_s"], 0.5)

    def test_dataset_validation_rejects_unknown_hidden_state(self):
        dataset = generate_dataset("intact", n_traces=1, n_frames=2)
        dataset.loc[0, "true_state"] = 99
        with self.assertRaisesRegex(ValueError, "hidden-state"):
            validate_dataset(dataset)

    def test_dataset_loader_validates_a_saved_csv(self):
        dataset = generate_dataset("blocked", n_traces=1, n_frames=2, seed=3)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            dataset.to_csv(path, index=False)
            restored = load_dataset(path)
        pd.testing.assert_frame_equal(restored, dataset, check_exact=False)


if __name__ == "__main__":
    unittest.main()
