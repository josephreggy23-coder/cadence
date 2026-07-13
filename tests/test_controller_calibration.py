import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controller import AdaptiveCadence, HIGH, REFRACTORY  # noqa: E402


class AdaptiveCadenceCalibrationTests(unittest.TestCase):
    @staticmethod
    def _record_transition(policy, load, switched, stimulus):
        policy.prev_state = HIGH
        policy.prev_L = load
        policy.prev_u = stimulus
        next_state = REFRACTORY if switched else HIGH
        policy._record(next_state, load)

    def test_stimulated_transitions_do_not_enter_or_change_baseline_fit(self):
        policy = AdaptiveCadence(-3.0, 0.9, min_events=2)
        self._record_transition(policy, load=0.5, switched=True, stimulus=0.0)
        self._record_transition(policy, load=2.0, switched=False, stimulus=0.0)
        policy._refit_b0()

        baseline_sample = (list(policy.obs_L), list(policy.obs_y))
        baseline_b0 = policy.b0

        for load in (0.2, 0.4, 0.8, 1.6):
            self._record_transition(policy, load=load, switched=True, stimulus=2.0)
        policy._refit_b0()

        self.assertEqual((policy.obs_L, policy.obs_y), baseline_sample)
        self.assertEqual(policy.b0, baseline_b0)
        self.assertEqual(policy.n_stim, 4)
        self.assertEqual(policy.k_stim, 4)

    def test_unstimulated_transitions_update_baseline_sample_and_fit(self):
        policy = AdaptiveCadence(-3.0, 0.9, min_events=2)
        self._record_transition(policy, load=0.5, switched=True, stimulus=0.0)
        self._record_transition(policy, load=2.0, switched=False, stimulus=0.0)
        policy._refit_b0()
        initial_b0 = policy.b0

        self._record_transition(policy, load=0.5, switched=True, stimulus=0.0)
        self._record_transition(policy, load=0.7, switched=True, stimulus=0.0)
        policy._refit_b0()

        self.assertEqual(policy.obs_L, [0.5, 2.0, 0.5, 0.7])
        self.assertEqual(policy.obs_y, [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(policy.n_quiet, 4)
        self.assertEqual(policy.k_quiet, 3.0)
        self.assertNotAlmostEqual(policy.b0, initial_b0, places=6)


if __name__ == "__main__":
    unittest.main()
