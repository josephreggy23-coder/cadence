"""Tests for Markov chain diagnostic utilities.

Covers stationary distribution, entropy rate, spectral gap,
mixing time bounds, and mean dwell times using transition matrices
with known analytical properties.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from markov_diagnostics import (
    MarkovDiagnostics,
    _validate_transition_matrix,
    diagnose,
    entropy_rate,
    mean_dwell_times,
    mixing_time_bound,
    spectral_gap,
    stationary_distribution,
)


# ------------------------------------------------------------------
# Fixtures: transition matrices with known properties
# ------------------------------------------------------------------
def _uniform_chain(K: int) -> np.ndarray:
    """Doubly stochastic matrix: stationary = uniform."""
    return np.full((K, K), 1.0 / K)


def _two_state_chain(p01: float, p10: float) -> np.ndarray:
    """2-state chain with known stationary = [p10, p01] / (p01 + p10)."""
    return np.array([
        [1.0 - p01, p01],
        [p10, 1.0 - p10],
    ])


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
class TestValidation:
    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _validate_transition_matrix(np.ones((2, 3)))

    def test_negative_entries_raise(self) -> None:
        A = np.array([[1.5, -0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            _validate_transition_matrix(A)

    def test_rows_not_summing_to_one_raise(self) -> None:
        A = np.array([[0.5, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="sum to 1"):
            _validate_transition_matrix(A)

    def test_single_state_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _validate_transition_matrix(np.array([[1.0]]))


# ------------------------------------------------------------------
# Stationary distribution
# ------------------------------------------------------------------
class TestStationaryDistribution:
    def test_uniform_chain(self) -> None:
        """Doubly stochastic matrix has uniform stationary distribution."""
        pi = stationary_distribution(_uniform_chain(4))
        np.testing.assert_allclose(pi, 0.25, atol=1e-12)

    def test_two_state_analytical(self) -> None:
        """For a 2-state chain, pi = [p10, p01] / (p01 + p10)."""
        p01, p10 = 0.3, 0.7
        pi = stationary_distribution(_two_state_chain(p01, p10))
        expected = np.array([p10, p01]) / (p01 + p10)
        np.testing.assert_allclose(pi, expected, atol=1e-12)

    def test_sums_to_one(self) -> None:
        A = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
        ])
        pi = stationary_distribution(A)
        assert pi.sum() == pytest.approx(1.0, abs=1e-12)

    def test_nonnegative(self) -> None:
        A = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ])
        pi = stationary_distribution(A)
        assert np.all(pi >= 0)

    def test_is_left_eigenvector(self) -> None:
        """pi @ A should equal pi."""
        A = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.4, 0.1, 0.5],
        ])
        pi = stationary_distribution(A)
        np.testing.assert_allclose(pi @ A, pi, atol=1e-10)

    def test_identity_chain(self) -> None:
        """Identity matrix: every state is absorbing, but the solver
        should still return a valid distribution."""
        # The system is not irreducible, but the linear solve still works.
        pi = stationary_distribution(np.eye(3))
        assert pi.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(pi >= 0)


# ------------------------------------------------------------------
# Entropy rate
# ------------------------------------------------------------------
class TestEntropyRate:
    def test_deterministic_chain_zero_entropy(self) -> None:
        """A deterministic cycle has zero entropy rate."""
        # 0 -> 1 -> 2 -> 0 deterministic cycle.
        A = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        h = entropy_rate(A)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_uniform_chain_max_entropy(self) -> None:
        """Uniform chain has entropy rate = log2(K)."""
        K = 4
        h = entropy_rate(_uniform_chain(K))
        assert h == pytest.approx(np.log2(K), rel=1e-10)

    def test_entropy_nonnegative(self) -> None:
        A = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
        ])
        assert entropy_rate(A) >= 0.0

    def test_high_self_loop_low_entropy(self) -> None:
        """Strong self-loops (long dwells) should give lower entropy."""
        A_high = np.array([[0.99, 0.01], [0.01, 0.99]])
        A_low = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert entropy_rate(A_high) < entropy_rate(A_low)


# ------------------------------------------------------------------
# Spectral gap
# ------------------------------------------------------------------
class TestSpectralGap:
    def test_uniform_chain_gap_is_one(self) -> None:
        """The uniform chain mixes in one step: lambda_2 = 0, gap = 1."""
        gap, eigs = spectral_gap(_uniform_chain(3))
        assert gap == pytest.approx(1.0, abs=1e-10)

    def test_near_identity_small_gap(self) -> None:
        """A chain close to the identity should have a near-zero gap."""
        eps = 0.01
        A = (1 - eps) * np.eye(3) + eps * np.ones((3, 3)) / 3
        gap, _ = spectral_gap(A)
        assert gap < 0.05

    def test_gap_in_zero_one(self) -> None:
        A = np.array([[0.7, 0.3], [0.4, 0.6]])
        gap, _ = spectral_gap(A)
        assert 0.0 < gap <= 1.0

    def test_eigenvalues_sorted_by_magnitude(self) -> None:
        A = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ])
        _, eigs = spectral_gap(A)
        mags = np.abs(eigs)
        assert all(mags[i] >= mags[i + 1] - 1e-12 for i in range(len(mags) - 1))

    def test_leading_eigenvalue_is_one(self) -> None:
        A = np.array([[0.8, 0.2], [0.3, 0.7]])
        _, eigs = spectral_gap(A)
        assert np.abs(eigs[0]) == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------------
# Mixing time
# ------------------------------------------------------------------
class TestMixingTime:
    def test_uniform_chain_fast_mixing(self) -> None:
        """Uniform chain should have a very short mixing time."""
        t = mixing_time_bound(_uniform_chain(4), epsilon=0.25)
        assert t < 10.0

    def test_near_identity_slow_mixing(self) -> None:
        """Near-identity chain should mix slowly."""
        eps = 0.001
        A = (1 - eps) * np.eye(3) + eps * np.ones((3, 3)) / 3
        t = mixing_time_bound(A, epsilon=0.25)
        assert t > 100.0

    def test_invalid_epsilon_raises(self) -> None:
        A = _two_state_chain(0.3, 0.7)
        with pytest.raises(ValueError, match="epsilon"):
            mixing_time_bound(A, epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            mixing_time_bound(A, epsilon=1.0)


# ------------------------------------------------------------------
# Mean dwell times
# ------------------------------------------------------------------
class TestMeanDwellTimes:
    def test_two_state_formula(self) -> None:
        """E[dwell_i] = 1 / (1 - A[i,i])."""
        A = _two_state_chain(0.2, 0.4)
        d = mean_dwell_times(A)
        assert d[0] == pytest.approx(1.0 / 0.2)  # 1/(1-0.8) = 5
        assert d[1] == pytest.approx(1.0 / 0.4)  # 1/(1-0.6) = 2.5

    def test_absorbing_state_infinite(self) -> None:
        """An absorbing state (A[i,i] = 1) should have infinite dwell."""
        A = np.array([[1.0, 0.0], [0.5, 0.5]])
        d = mean_dwell_times(A)
        assert np.isinf(d[0])
        assert d[1] == pytest.approx(2.0)

    def test_all_positive(self) -> None:
        A = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
        ])
        d = mean_dwell_times(A)
        assert np.all(d > 0)


# ------------------------------------------------------------------
# Combined diagnostics
# ------------------------------------------------------------------
class TestDiagnose:
    def test_returns_dataclass(self) -> None:
        result = diagnose(_two_state_chain(0.3, 0.7))
        assert isinstance(result, MarkovDiagnostics)

    def test_all_fields_finite_for_ergodic_chain(self) -> None:
        result = diagnose(_two_state_chain(0.3, 0.7))
        assert np.all(np.isfinite(result.stationary))
        assert np.isfinite(result.entropy_rate)
        assert np.isfinite(result.spectral_gap)
        assert np.isfinite(result.mixing_time)

    def test_four_state_cadence_like(self) -> None:
        """A transition matrix resembling the CADENCE 4-state system."""
        A = np.array([
            [0.90, 0.08, 0.01, 0.01],  # QUIESCENT
            [0.02, 0.85, 0.12, 0.01],  # OSCILLATORY
            [0.00, 0.05, 0.80, 0.15],  # SUSTAINED_HIGH
            [0.15, 0.05, 0.00, 0.80],  # REFRACTORY
        ])
        result = diagnose(A)
        # Stationary should be a proper distribution.
        assert result.stationary.sum() == pytest.approx(1.0, abs=1e-10)
        # With these self-loop probabilities, entropy is moderate.
        assert 0.0 < result.entropy_rate < np.log2(4)
        # The chain should mix, but not instantly.
        assert result.mixing_time > 1.0
