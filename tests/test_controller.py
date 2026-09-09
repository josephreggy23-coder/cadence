"""Tests for the CADENCE controller module.

Covers the sigmoid utility, transition row construction, static Cadence
policy dosing logic, and AdaptiveCadence safety interlocks (futility
abort, budget exhaustion, calibration gating).

The controller's feedback law converts a constant-hazard horizon
approximation into a clipped dose:

    u* = (logit(p_needed) - b0 - b1*L) / (b1*kappa)

where p_needed = 1 - (1 - target)^(1/horizon) is the per-frame
escape probability required to achieve the target cumulative escape
probability over the planning horizon.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controller import (
    CADENCE_U_MAX,
    HIGH,
    KAPPA,
    LEAK_TO_OSC,
    OSCILLATORY,
    QUIESCENT,
    REFRACTORY,
    AdaptiveCadence,
    Cadence,
    NoControl,
    OpenLoop,
    challenge_transitions,
    high_row,
    sigmoid,
)


# ------------------------------------------------------------------
# Sigmoid
# ------------------------------------------------------------------
class TestSigmoid:
    def test_at_zero(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_saturates(self) -> None:
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative_saturates(self) -> None:
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self) -> None:
        """sigmoid(x) + sigmoid(-x) = 1 for all x."""
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0, abs=1e-12)

    def test_monotone_increasing(self) -> None:
        xs = np.linspace(-10, 10, 100)
        ys = np.array([sigmoid(x) for x in xs])
        assert np.all(np.diff(ys) >= 0)

    def test_no_overflow_at_extreme_values(self) -> None:
        """Should not produce nan or inf even at extreme inputs."""
        for x in [-1000.0, 1000.0, -1e10, 1e10]:
            y = sigmoid(x)
            assert np.isfinite(y)


# ------------------------------------------------------------------
# high_row: transition row out of SUSTAINED_HIGH
# ------------------------------------------------------------------
class TestHighRow:
    def test_sums_to_one(self) -> None:
        for p_off in [0.0, 0.1, 0.5, 0.9, 0.99]:
            row = high_row(p_off)
            assert row.sum() == pytest.approx(1.0, abs=1e-12)

    def test_nonnegative(self) -> None:
        for p_off in [0.0, 0.5, 0.98]:
            row = high_row(p_off)
            assert np.all(row >= 0)

    def test_no_transition_to_quiescent(self) -> None:
        """The model forbids direct HIGH -> QUIESCENT transitions."""
        row = high_row(0.5)
        assert row[QUIESCENT] == pytest.approx(0.0)

    def test_refractory_receives_p_off(self) -> None:
        """Most of p_off should go to the REFRACTORY state."""
        row = high_row(0.5)
        assert row[REFRACTORY] > 0.4

    def test_leak_to_oscillatory(self) -> None:
        """There should always be a small leak to OSCILLATORY."""
        row = high_row(0.3)
        assert row[OSCILLATORY] > 0

    def test_clips_extreme_p_off(self) -> None:
        """p_off > 1 - LEAK_TO_OSC should be clipped, not crash."""
        row = high_row(1.5)
        assert row.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(row >= 0)


# ------------------------------------------------------------------
# challenge_transitions
# ------------------------------------------------------------------
class TestChallengeTransitions:
    def test_non_high_rows_sum_to_one(self) -> None:
        """Non-HIGH rows should be proper probability distributions.
        The HIGH row is filled dynamically by high_row() at runtime."""
        T = challenge_transitions()
        for i in [QUIESCENT, OSCILLATORY, REFRACTORY]:
            assert T[i].sum() == pytest.approx(1.0, abs=1e-10)

    def test_nonnegative(self) -> None:
        T = challenge_transitions()
        assert np.all(T >= 0)

    def test_oscillatory_entry_boosted(self) -> None:
        """The challenge should increase P(OSC -> HIGH)."""
        T = challenge_transitions(entry_boost=0.3)
        assert T[OSCILLATORY, HIGH] > 0.12 + 0.25  # > base + most of boost


# ------------------------------------------------------------------
# NoControl and OpenLoop
# ------------------------------------------------------------------
class TestSimplePolicies:
    def test_no_control_always_zero(self) -> None:
        policy = NoControl()
        for state in [QUIESCENT, OSCILLATORY, HIGH, REFRACTORY]:
            assert policy.act(state, L=5.0) == 0.0

    def test_open_loop_constant_dose(self) -> None:
        policy = OpenLoop(dose=2.5)
        for state in [QUIESCENT, OSCILLATORY, HIGH, REFRACTORY]:
            assert policy.act(state, L=0.0) == 2.5


# ------------------------------------------------------------------
# Cadence (static model-based policy)
# ------------------------------------------------------------------
class TestCadence:
    def test_no_action_outside_high_state(self) -> None:
        """The policy only acts when the estimated state is HIGH."""
        policy = Cadence(b0=-3.0, b1=0.9)
        for state in [QUIESCENT, OSCILLATORY, REFRACTORY]:
            assert policy.act(state, L=0.0) == 0.0

    def test_acts_in_high_state(self) -> None:
        """In the HIGH state with low load, should prescribe a dose."""
        policy = Cadence(b0=-3.0, b1=0.9)
        u = policy.act(HIGH, L=0.5)
        assert u > 0.0

    def test_dose_clipped_to_u_max(self) -> None:
        """Dose should never exceed u_max."""
        policy = Cadence(b0=-10.0, b1=0.01, u_max=5.0)
        u = policy.act(HIGH, L=0.0)
        assert u <= 5.0

    def test_dose_nonnegative(self) -> None:
        policy = Cadence(b0=-3.0, b1=0.9)
        for L in [0.0, 1.0, 5.0, 10.0]:
            u = policy.act(HIGH, L=L)
            assert u >= 0.0

    def test_higher_load_means_less_intervention(self) -> None:
        """Higher accumulated load raises endogenous escape probability,
        so less intervention should be needed."""
        policy = Cadence(b0=-3.0, b1=0.9)
        u_low_load = policy.act(HIGH, L=0.5)
        u_high_load = policy.act(HIGH, L=5.0)
        assert u_high_load <= u_low_load

    def test_zero_b1_gives_max_dose(self) -> None:
        """When b1 ~ 0, no finite dose can act; should return u_max."""
        policy = Cadence(b0=-3.0, b1=0.0, u_max=8.0)
        u = policy.act(HIGH, L=1.0)
        assert u == pytest.approx(8.0)

    def test_sufficient_endogenous_escape_gives_zero(self) -> None:
        """If the cell is already likely to escape on its own, u = 0."""
        # b0=2 with b1=0.9 and L=5 gives very high endogenous p_off
        policy = Cadence(b0=2.0, b1=0.9, horizon=6, target=0.9)
        u = policy.act(HIGH, L=5.0)
        assert u == 0.0


# ------------------------------------------------------------------
# AdaptiveCadence: calibration and safety
# ------------------------------------------------------------------
class TestAdaptiveCadence:
    @staticmethod
    def _force_transition(policy, load, switched, stimulus=0.0):
        """Simulate one HIGH -> next transition for calibration testing."""
        policy.prev_state = HIGH
        policy.prev_L = load
        policy.prev_u = stimulus
        next_state = REFRACTORY if switched else HIGH
        policy._record(next_state, load)

    def test_observation_window_returns_zero(self) -> None:
        """During calibration frames, the policy should not intervene."""
        policy = AdaptiveCadence(-3.0, 0.9, calib_frames=100)
        for t in range(100):
            policy.t = t
            u = policy.act(HIGH, L=1.0)
            assert u == 0.0

    def test_refit_changes_b0(self) -> None:
        """Feeding transitions with high switch rate should raise b0."""
        policy = AdaptiveCadence(-3.0, 0.9, min_events=3)
        initial_b0 = policy.b0
        # Several high-load transitions that all switch off
        for _ in range(10):
            self._force_transition(policy, load=2.0, switched=True)
        policy._refit_b0()
        assert policy.calibrated is True
        assert policy.b0 != initial_b0

    def test_futility_abort(self) -> None:
        """If stimulated rate <= quiet rate, policy should abort."""
        policy = AdaptiveCadence(
            -3.0, 0.9, min_stim_trials=5, futility_margin=0.0
        )
        # Quiet transitions: 3/5 switch
        for _ in range(3):
            self._force_transition(policy, load=1.0, switched=True, stimulus=0.0)
        for _ in range(2):
            self._force_transition(policy, load=1.0, switched=False, stimulus=0.0)

        # Stimulated transitions: only 1/5 switch (worse than quiet)
        for _ in range(1):
            self._force_transition(policy, load=1.0, switched=True, stimulus=1.0)
        for _ in range(4):
            self._force_transition(policy, load=1.0, switched=False, stimulus=1.0)

        policy._check_futility()
        assert policy.aborted is True

    def test_budget_exhaustion(self) -> None:
        """Cumulative dose should stop at max_cumulative_dose."""
        policy = AdaptiveCadence(
            -3.0, 0.9,
            calib_frames=0,
            min_events=1,
            max_cumulative_dose=10.0,
        )
        # Force calibration
        self._force_transition(policy, load=1.0, switched=True)
        policy._refit_b0()

        total_dose = 0.0
        for _ in range(100):
            u = policy.act(HIGH, L=0.5)
            total_dose += u
            if policy.budget_exhausted:
                break

        assert total_dose <= 10.0 + 1e-6

    def test_stimulated_transitions_excluded_from_baseline(self) -> None:
        """Transitions under stimulation should not enter obs_L / obs_y."""
        policy = AdaptiveCadence(-3.0, 0.9, min_events=2)
        # Two unstimulated observations
        self._force_transition(policy, load=0.5, switched=True, stimulus=0.0)
        self._force_transition(policy, load=2.0, switched=False, stimulus=0.0)

        n_baseline = len(policy.obs_L)

        # Three stimulated observations
        for _ in range(3):
            self._force_transition(policy, load=1.0, switched=True, stimulus=2.0)

        assert len(policy.obs_L) == n_baseline  # unchanged
        assert policy.n_stim == 3

    def test_invalid_max_dose_raises(self) -> None:
        with pytest.raises(ValueError, match="max_cumulative_dose"):
            AdaptiveCadence(-3.0, 0.9, max_cumulative_dose=-1.0)
