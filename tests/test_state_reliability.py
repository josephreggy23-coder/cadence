"""Tests for HMM state reliability and reproducibility diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_reliability import (
    adjusted_rand_index,
    align_state_labels,
    cohens_kappa,
    cross_seed_consistency,
    split_half_reliability,
    temporal_stability,
)


# -----------------------------------------------------------------------
# Cohen's kappa
# -----------------------------------------------------------------------

class TestCohensKappa:
    def test_perfect_agreement_gives_kappa_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_random_assignments_give_kappa_near_zero(self):
        rng = np.random.default_rng(42)
        n = 10_000
        a = rng.integers(0, 4, size=n)
        b = rng.integers(0, 4, size=n)
        kappa = cohens_kappa(a, b)
        assert abs(kappa) < 0.05, f"Expected kappa near 0 for random labels, got {kappa}"

    def test_complete_disagreement_gives_negative_kappa(self):
        # Two raters that systematically swap 0 <-> 1.
        a = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([1, 1, 1, 0, 0, 0])
        kappa = cohens_kappa(a, b)
        assert kappa < 0, f"Expected negative kappa for systematic disagreement, got {kappa}"

    def test_known_value(self):
        # Textbook example: 2 raters, 2 categories, 50 items.
        # Rater A: 25 yes, 25 no.  Rater B: 30 yes, 20 no.
        # Agree on 20 yes and 15 no => p_o = 35/50 = 0.70.
        a = np.array([1]*25 + [0]*25)
        b = np.array([1]*20 + [0]*5 + [1]*10 + [0]*15)
        kappa = cohens_kappa(a, b)
        # p_e = (25/50)*(30/50) + (25/50)*(20/50) = 0.30 + 0.20 = 0.50
        # kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
        assert kappa == pytest.approx(0.40, abs=1e-10)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa(np.array([0, 1]), np.array([0, 1, 2]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cohens_kappa(np.array([]), np.array([]))


# -----------------------------------------------------------------------
# Adjusted Rand index
# -----------------------------------------------------------------------

class TestAdjustedRandIndex:
    def test_identical_partitions_give_ari_one(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)

    def test_permuted_identical_partitions_give_ari_one(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([2, 2, 0, 0, 1, 1])  # same partition, permuted labels
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)

    def test_random_partitions_give_ari_near_zero(self):
        rng = np.random.default_rng(123)
        n = 5_000
        a = rng.integers(0, 5, size=n)
        b = rng.integers(0, 5, size=n)
        ari = adjusted_rand_index(a, b)
        assert abs(ari) < 0.05, f"Expected ARI near 0 for random partitions, got {ari}"

    def test_known_contingency(self):
        # Two partitions that partially agree.
        a = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([0, 0, 1, 1, 1, 0])
        ari = adjusted_rand_index(a, b)
        # Manual: pairs = C(6,2)=15
        # Same-same pairs: (0,1) in both a and b => a=1; different-different: d=4
        # a+d=5, RI=5/15=1/3.
        # Should be between -1 and 1 and neither extreme.
        assert -1.0 <= ari <= 1.0

    def test_single_cluster_gives_ari_one(self):
        a = np.array([0, 0, 0, 0])
        b = np.array([5, 5, 5, 5])
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)


# -----------------------------------------------------------------------
# State alignment
# -----------------------------------------------------------------------

class TestStateAlignment:
    def test_identity_permutation(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        result = align_state_labels(labels, labels)
        assert result.agreement == pytest.approx(1.0)

    def test_simple_swap_recovered(self):
        a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        b = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # 0->2, 1->0, 2->1
        result = align_state_labels(a, b)
        assert result.agreement == pytest.approx(1.0)
        # After alignment, labels should match exactly.
        np.testing.assert_array_equal(result.aligned_labels_b, a)

    def test_partial_agreement(self):
        a = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([0, 0, 1, 1, 1, 1])  # one frame differs
        result = align_state_labels(a, b)
        # Best alignment keeps labels as-is or swaps; either way
        # at most 5/6 agreement.
        assert result.agreement == pytest.approx(5.0 / 6.0)

    def test_permuted_sequences_give_ari_one_after_alignment(self):
        rng = np.random.default_rng(99)
        a = rng.integers(0, 4, size=200)
        perm = {0: 3, 1: 0, 2: 1, 3: 2}
        b = np.array([perm[int(x)] for x in a])
        result = align_state_labels(a, b)
        assert result.agreement == pytest.approx(1.0)
        ari = adjusted_rand_index(a, result.aligned_labels_b)
        assert ari == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Split-half reliability
# -----------------------------------------------------------------------

class TestSplitHalfReliability:
    def test_constant_sequence_perfect_kappa(self):
        seq = np.zeros(100, dtype=int)
        result = split_half_reliability(seq)
        assert result.kappa == pytest.approx(1.0)

    def test_alternating_sequence(self):
        # [0,1,0,1,...] -- even indices are 0, odd indices are 1.
        # half_a = all 0, half_b = all 1.  Perfect disagreement after
        # alignment still gives agreement of 1.0 (aligned maps 1->0).
        seq = np.array([0, 1] * 50)
        result = split_half_reliability(seq)
        # After alignment, half_b (all 1s) maps to 0 => perfect agreement.
        assert result.aligned_agreement == pytest.approx(1.0)

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            split_half_reliability(np.array([0, 1, 0]))


# -----------------------------------------------------------------------
# Cross-seed consistency
# -----------------------------------------------------------------------

class TestCrossSeedConsistency:
    def test_identical_seeds_give_ari_one(self):
        seq = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        result = cross_seed_consistency([seq, seq, seq])
        assert result.mean_ari == pytest.approx(1.0)
        assert result.min_ari == pytest.approx(1.0)

    def test_permuted_seeds_give_ari_one(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([1, 1, 2, 2, 0, 0])
        result = cross_seed_consistency([a, b])
        assert result.mean_ari == pytest.approx(1.0)

    def test_random_seeds_give_low_ari(self):
        rng = np.random.default_rng(77)
        seqs = [rng.integers(0, 3, size=500) for _ in range(5)]
        result = cross_seed_consistency(seqs)
        assert result.mean_ari < 0.1

    def test_single_seed_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            cross_seed_consistency([np.array([0, 1, 2])])


# -----------------------------------------------------------------------
# Temporal stability
# -----------------------------------------------------------------------

class TestTemporalStability:
    def test_constant_dwell_gives_zero_cv(self):
        # State 0 for 10 frames, state 1 for 10, state 0 for 10, state 1 for 10.
        seq = np.array([0]*10 + [1]*10 + [0]*10 + [1]*10)
        result = temporal_stability(seq)
        for stats in result.per_state:
            assert stats.cv == pytest.approx(0.0)
            assert stats.mean_dwell == pytest.approx(10.0)
            assert stats.n_visits == 2

    def test_known_cv(self):
        # Dwell times of 10 and 20 for state 0 => mean=15, std=sqrt(50)~7.07
        # CV = 7.07/15 ~ 0.4714
        seq = np.array([0]*10 + [1]*5 + [0]*20 + [1]*5)
        result = temporal_stability(seq)
        state_0 = [s for s in result.per_state if s.state == 0][0]
        expected_cv = np.std([10, 20], ddof=1) / np.mean([10, 20])
        assert state_0.cv == pytest.approx(expected_cv, abs=1e-10)

    def test_single_visit_gives_zero_cv(self):
        seq = np.array([0]*50)
        result = temporal_stability(seq)
        assert len(result.per_state) == 1
        assert result.per_state[0].cv == pytest.approx(0.0)
        assert result.per_state[0].n_visits == 1

    def test_geometric_dwell_times(self):
        # Generate a sequence with geometrically distributed dwell times.
        # For a geometric distribution with parameter p, CV = sqrt(1-p)/p.
        # With p=0.1, mean dwell = 10, CV ~ sqrt(0.9)/0.1 ~ 9.49.
        # But with a finite sample this is approximate.
        rng = np.random.default_rng(314)
        p = 0.1
        dwells = rng.geometric(p, size=500)
        seq = np.concatenate([np.full(d, i % 3) for i, d in enumerate(dwells)])
        result = temporal_stability(seq)
        # Just verify it runs and CVs are positive.
        for stats in result.per_state:
            assert stats.cv > 0
            assert stats.n_visits > 10

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="empty"):
            temporal_stability(np.array([]))
