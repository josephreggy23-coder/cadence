import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_synthetic import simulate_trace


class SimulateTraceTests(unittest.TestCase):
    def test_rejects_invalid_frame_count(self):
        with self.assertRaisesRegex(ValueError, "n_frames"):
            simulate_trace(np.random.default_rng(1), 0, -3.0, 0.9)

    def test_rejects_invalid_load_decay(self):
        with self.assertRaisesRegex(ValueError, "load_decay"):
            simulate_trace(np.random.default_rng(1), 10, -3.0, 0.9, load_decay=1.1)


if __name__ == "__main__":
    unittest.main()
