"""Reliability and reproducibility diagnostics for HMM state assignments.

When an HMM is fitted to calcium traces, the inferred latent states should
reflect genuine biological modes -- not fitting artifacts that depend on the
particular data split, random seed, or optimization trajectory.  This module
provides four complementary diagnostics:

Split-half reliability
----------------------
Split traces into two halves, fit independent HMMs on each, decode an
overlapping held-out segment with both models, and measure agreement via
Cohen's kappa:

    kappa = (p_o - p_e) / (1 - p_e)

where p_o is the observed agreement fraction and p_e is the expected
agreement by chance (computed from the marginal label distributions).
A kappa of 1 means perfect agreement; 0 means chance-level; negative
values indicate systematic disagreement.

Cross-seed consistency
----------------------
Given state-assignment vectors from multiple random restarts of the same
model, measure pairwise consistency via the adjusted Rand index (ARI):

    ARI = (RI - E[RI]) / (max(RI) - E[RI])

where RI = (a + d) / C(n, 2), with a = pairs assigned to the same
cluster in both partitions and d = pairs in different clusters in both.
ARI = 1 for identical partitions (up to label permutation) and ~0 for
random partitions.

State alignment
---------------
Two HMM fits may discover the same states but label them in different
order.  Given two label vectors, find the optimal label permutation that
maximizes agreement by solving the linear assignment problem (Hungarian
algorithm) on the confusion matrix.  Return the aligned agreement and
the permutation mapping.

Temporal stability
------------------
For each state, measure the coefficient of variation (CV) of dwell times:

    CV = std(dwell_times) / mean(dwell_times)

States with low CV have consistent dwell durations and are likely real.
States with high CV may be splitting artifacts where the model uses one
label for two distinct dynamical regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Chance-corrected agreement between two label vectors.

    Parameters
    ----------
    labels_a, labels_b
        Integer label arrays of the same length.

    Returns
    -------
    float
        Cohen's kappa in [-1, 1].  1 = perfect, 0 = chance, <0 = worse
        than chance.
    """
    labels_a = np.asarray(labels_a, dtype=int).ravel()
    labels_b = np.asarray(labels_b, dtype=int).ravel()
    if len(labels_a) != len(labels_b):
        raise ValueError("Label vectors must have the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("Label vectors must not be empty")

    p_o = float(np.mean(labels_a == labels_b))

    all_labels = np.union1d(labels_a, labels_b)
    p_e = 0.0
    for k in all_labels:
        p_e += (np.sum(labels_a == k) / n) * (np.sum(labels_b == k) / n)

    if abs(1.0 - p_e) < 1e-15:
        # Both raters assign the same single category to every item.
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Adjusted Rand index
# ---------------------------------------------------------------------------

def _contingency_pair_counts(labels_a: np.ndarray, labels_b: np.ndarray):
    """Return (a, b_sum, c_sum, n) for the ARI formula.

    a    = number of pairs in the same group in both partitions
    b_sum = sum of C(n_i., 2) over rows of the contingency table
    c_sum = sum of C(n_.j, 2) over columns
    n    = total number of elements
    """
    labels_a = np.asarray(labels_a, dtype=int).ravel()
    labels_b = np.asarray(labels_b, dtype=int).ravel()
    if len(labels_a) != len(labels_b):
        raise ValueError("Label vectors must have the same length")
    n = len(labels_a)
    if n < 2:
        raise ValueError("Need at least two observations for pair counting")

    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)

    # Build the contingency table.
    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    a_map = {v: i for i, v in enumerate(classes_a)}
    b_map = {v: i for i, v in enumerate(classes_b)}
    for la, lb in zip(labels_a, labels_b):
        contingency[a_map[la], b_map[lb]] += 1

    def comb2(x):
        return x * (x - 1) // 2

    # Sum of C(n_ij, 2) over all cells.
    sum_comb = np.sum(comb2(contingency))
    # Row and column marginal pair counts.
    b_sum = np.sum(comb2(contingency.sum(axis=1)))
    c_sum = np.sum(comb2(contingency.sum(axis=0)))

    return int(sum_comb), int(b_sum), int(c_sum), n


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index between two partitions.

    ARI = (RI - E[RI]) / (max(RI) - E[RI])

    Parameters
    ----------
    labels_a, labels_b
        Integer label arrays of the same length.

    Returns
    -------
    float
        ARI in [-1, 1].  1 = identical partitions up to permutation,
        ~0 = random, <0 = worse than random.
    """
    sum_comb, b_sum, c_sum, n = _contingency_pair_counts(labels_a, labels_b)
    n_comb = n * (n - 1) // 2

    expected = b_sum * c_sum / n_comb if n_comb > 0 else 0.0
    max_index = (b_sum + c_sum) / 2.0

    denominator = max_index - expected
    if abs(denominator) < 1e-15:
        # Both partitions are identical (or all items in one cluster).
        return 1.0

    return (sum_comb - expected) / denominator


# ---------------------------------------------------------------------------
# State alignment (Hungarian algorithm)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning two state-label vectors.

    Attributes
    ----------
    permutation
        Mapping from labels in ``labels_b`` to aligned labels, as a
        dict {original_label: aligned_label}.
    aligned_labels_b
        The ``labels_b`` vector after applying the permutation.
    agreement
        Fraction of frames where the aligned labels agree.
    confusion_matrix
        Confusion matrix (rows = labels_a classes, cols = labels_b classes)
        before alignment.
    """

    permutation: dict
    aligned_labels_b: np.ndarray
    agreement: float
    confusion_matrix: np.ndarray


def align_state_labels(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> AlignmentResult:
    """Find the optimal label permutation that maximizes agreement.

    Uses the Hungarian algorithm on the confusion matrix to solve the
    linear assignment problem: maximize the total number of matching
    frames when relabeling ``labels_b``.

    Parameters
    ----------
    labels_a, labels_b
        Integer label arrays of the same length.

    Returns
    -------
    AlignmentResult
        Contains the permutation, aligned labels, agreement fraction,
        and confusion matrix.
    """
    labels_a = np.asarray(labels_a, dtype=int).ravel()
    labels_b = np.asarray(labels_b, dtype=int).ravel()
    if len(labels_a) != len(labels_b):
        raise ValueError("Label vectors must have the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("Label vectors must not be empty")

    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)

    # Build the confusion matrix: C[i, j] = count of frames where
    # labels_a == classes_a[i] and labels_b == classes_b[j].
    confusion = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    a_map = {v: i for i, v in enumerate(classes_a)}
    b_map = {v: i for i, v in enumerate(classes_b)}
    for la, lb in zip(labels_a, labels_b):
        confusion[a_map[la], b_map[lb]] += 1

    # Solve the assignment problem.  linear_sum_assignment minimizes cost,
    # so negate the confusion matrix to maximize agreement.  The matrix
    # may be rectangular when the two fits have different numbers of states;
    # pad if needed.
    k = max(len(classes_a), len(classes_b))
    cost = np.zeros((k, k), dtype=np.int64)
    cost[: len(classes_a), : len(classes_b)] = -confusion
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build the permutation: for each assigned pair, map classes_b[col]
    # to classes_a[row].  Unmatched labels in b (when b has more states)
    # map to themselves offset to avoid collisions.
    permutation = {}
    for r, c in zip(row_ind, col_ind):
        if c < len(classes_b):
            if r < len(classes_a):
                permutation[int(classes_b[c])] = int(classes_a[r])
            else:
                # Extra state in b with no match in a -- keep its label.
                permutation[int(classes_b[c])] = int(classes_b[c])

    aligned = np.array([permutation.get(int(lb), int(lb)) for lb in labels_b])
    agreement = float(np.mean(labels_a == aligned))

    return AlignmentResult(
        permutation=permutation,
        aligned_labels_b=aligned,
        agreement=agreement,
        confusion_matrix=confusion,
    )


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitHalfResult:
    """Result of a split-half reliability analysis.

    Attributes
    ----------
    kappa
        Cohen's kappa on the overlapping segment after state alignment.
    raw_agreement
        Fraction of frames that agree before alignment.
    aligned_agreement
        Fraction of frames that agree after optimal alignment.
    n_frames
        Number of frames in each half.
    """

    kappa: float
    raw_agreement: float
    aligned_agreement: float
    n_frames: int


def split_half_reliability(
    state_sequence: np.ndarray,
) -> SplitHalfResult:
    """Assess reliability by comparing assignments on two halves.

    Splits the sequence at the midpoint, then measures how consistently
    the two halves agree on which state each frame belongs to.  Because
    the two halves are non-overlapping sub-sequences of the same trace,
    the comparison is valid only where both halves assign labels to the
    same frames.  Here we use even-indexed frames as "half A" and
    odd-indexed frames as "half B", so every frame is in exactly one
    half and we compare the interleaved assignments.

    Parameters
    ----------
    state_sequence
        Integer state-assignment vector for one trace.

    Returns
    -------
    SplitHalfResult
        Agreement statistics between the two halves.
    """
    seq = np.asarray(state_sequence, dtype=int).ravel()
    if len(seq) < 4:
        raise ValueError("Sequence must have at least 4 frames for split-half")

    half_a = seq[0::2]
    half_b = seq[1::2]
    # Truncate to equal length if the sequence has odd length.
    min_len = min(len(half_a), len(half_b))
    half_a = half_a[:min_len]
    half_b = half_b[:min_len]

    raw_agreement = float(np.mean(half_a == half_b))

    alignment = align_state_labels(half_a, half_b)
    aligned_agreement = alignment.agreement

    kappa = cohens_kappa(half_a, alignment.aligned_labels_b)

    return SplitHalfResult(
        kappa=kappa,
        raw_agreement=raw_agreement,
        aligned_agreement=aligned_agreement,
        n_frames=min_len,
    )


# ---------------------------------------------------------------------------
# Cross-seed consistency
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossSeedResult:
    """Result of cross-seed consistency analysis.

    Attributes
    ----------
    mean_ari
        Mean pairwise adjusted Rand index across all seed pairs.
    pairwise_ari
        Matrix of pairwise ARI values, shape (n_seeds, n_seeds).
    min_ari
        Minimum pairwise ARI (the weakest link).
    n_seeds
        Number of seed replicates compared.
    """

    mean_ari: float
    pairwise_ari: np.ndarray
    min_ari: float
    n_seeds: int


def cross_seed_consistency(
    state_sequences: list[np.ndarray],
) -> CrossSeedResult:
    """Measure consistency of state assignments across random restarts.

    Parameters
    ----------
    state_sequences
        List of integer state-assignment vectors, one per random restart.
        All must have the same length (same data, different fits).

    Returns
    -------
    CrossSeedResult
        Pairwise ARI statistics.
    """
    if len(state_sequences) < 2:
        raise ValueError("Need at least two seed replicates")

    seqs = [np.asarray(s, dtype=int).ravel() for s in state_sequences]
    n = len(seqs[0])
    for i, s in enumerate(seqs):
        if len(s) != n:
            raise ValueError(
                f"All sequences must have the same length; "
                f"sequence 0 has {n}, sequence {i} has {len(s)}"
            )

    k = len(seqs)
    pairwise = np.ones((k, k), dtype=float)
    ari_values = []

    for i in range(k):
        for j in range(i + 1, k):
            ari = adjusted_rand_index(seqs[i], seqs[j])
            pairwise[i, j] = ari
            pairwise[j, i] = ari
            ari_values.append(ari)

    return CrossSeedResult(
        mean_ari=float(np.mean(ari_values)),
        pairwise_ari=pairwise,
        min_ari=float(np.min(ari_values)),
        n_seeds=k,
    )


# ---------------------------------------------------------------------------
# Temporal stability (dwell-time CV)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DwellTimeStats:
    """Dwell-time statistics for a single state.

    Attributes
    ----------
    state
        The state label.
    mean_dwell
        Mean dwell time in frames.
    std_dwell
        Standard deviation of dwell times.
    cv
        Coefficient of variation (std / mean).
    n_visits
        Number of contiguous visits to this state.
    """

    state: int
    mean_dwell: float
    std_dwell: float
    cv: float
    n_visits: int


@dataclass(frozen=True)
class TemporalStabilityResult:
    """Temporal stability analysis for all states.

    Attributes
    ----------
    per_state
        List of DwellTimeStats, one per unique state.
    overall_cv
        Mean CV across all states, weighted by visit count.
    """

    per_state: list[DwellTimeStats] = field(default_factory=list)
    overall_cv: float = 0.0


def _extract_dwell_times(state_sequence: np.ndarray) -> dict[int, list[int]]:
    """Extract contiguous dwell times for each state.

    Parameters
    ----------
    state_sequence
        Integer state-assignment vector.

    Returns
    -------
    dict
        Mapping from state label to list of dwell durations (in frames).
    """
    seq = np.asarray(state_sequence, dtype=int).ravel()
    if len(seq) == 0:
        return {}

    dwells: dict[int, list[int]] = {}
    current_state = int(seq[0])
    current_length = 1

    for i in range(1, len(seq)):
        if int(seq[i]) == current_state:
            current_length += 1
        else:
            dwells.setdefault(current_state, []).append(current_length)
            current_state = int(seq[i])
            current_length = 1

    # Final run.
    dwells.setdefault(current_state, []).append(current_length)
    return dwells


def temporal_stability(
    state_sequence: np.ndarray,
) -> TemporalStabilityResult:
    """Measure dwell-time regularity for each state.

    A stable (real) state should have a consistent characteristic dwell
    time, producing a relatively low coefficient of variation.  A
    splitting artifact that merges two dynamical regimes under one label
    typically has high CV because dwell times come from a mixture.

    Parameters
    ----------
    state_sequence
        Integer state-assignment vector.

    Returns
    -------
    TemporalStabilityResult
        Per-state dwell statistics and overall weighted CV.
    """
    dwells = _extract_dwell_times(state_sequence)
    if not dwells:
        raise ValueError("State sequence must not be empty")

    per_state = []
    total_weighted_cv = 0.0
    total_visits = 0

    for state in sorted(dwells.keys()):
        durations = np.array(dwells[state], dtype=float)
        n_visits = len(durations)
        mean_d = float(np.mean(durations))
        if n_visits < 2:
            std_d = 0.0
            cv = 0.0
        else:
            std_d = float(np.std(durations, ddof=1))
            cv = std_d / mean_d if mean_d > 0 else 0.0

        per_state.append(DwellTimeStats(
            state=state,
            mean_dwell=mean_d,
            std_dwell=std_d,
            cv=cv,
            n_visits=n_visits,
        ))
        total_weighted_cv += cv * n_visits
        total_visits += n_visits

    overall_cv = total_weighted_cv / total_visits if total_visits > 0 else 0.0

    return TemporalStabilityResult(per_state=per_state, overall_cv=overall_cv)
