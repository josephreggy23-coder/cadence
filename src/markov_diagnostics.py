"""Diagnostic quantities for finite discrete-time Markov chains.

A fitted HMM produces a transition matrix A whose rows are conditional
distributions: A[i, j] = P(s_{t+1} = j | s_t = i).  This module
extracts three diagnostics from that matrix:

Stationary distribution
-----------------------
The unique row vector pi satisfying pi @ A = pi and sum(pi) = 1.
It gives the long-run fraction of time the chain spends in each
state, regardless of where it starts.  We find it as the left
eigenvector of A corresponding to eigenvalue 1.

Entropy rate
------------
For a stationary Markov chain with transition matrix A and
stationary distribution pi, the entropy rate is

    h = - sum_i pi_i  sum_j A[i,j] log2 A[i,j]

This is the expected per-step uncertainty about the next state,
given that the chain has been running long enough to reach
stationarity.  Lower entropy means the chain is more predictable
(long dwells in each state); higher entropy means frequent
switching.

Mixing time
-----------
The number of steps needed before the chain's distribution is
within total-variation distance epsilon of the stationary
distribution, regardless of starting state.  We bound it via the
spectral gap:

    t_mix(epsilon) <= (1 / gap) * log(1 / (epsilon * pi_min))

where gap = 1 - |lambda_2| is one minus the second-largest
eigenvalue magnitude.  A small gap means slow mixing (the chain
remembers its initial state for a long time); a large gap means
fast forgetting.

These diagnostics help characterize the fitted calcium dynamics:
a chain with high dwell times (low entropy rate, slow mixing)
behaves differently under stimulation than one that switches
rapidly on its own.

References
----------
Levin DA, Peres Y. Markov Chains and Mixing Times. 2nd ed.
American Mathematical Society; 2017.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarkovDiagnostics:
    """Summary diagnostics for a finite Markov chain.

    Attributes
    ----------
    stationary
        Stationary distribution, shape (K,).
    entropy_rate
        Shannon entropy rate in bits per step.
    spectral_gap
        1 - |lambda_2|, the gap that controls mixing speed.
    mixing_time
        Upper bound on steps to reach within epsilon of stationarity.
    eigenvalues
        All eigenvalues of A, sorted by descending magnitude.
    """

    stationary: np.ndarray
    entropy_rate: float
    spectral_gap: float
    mixing_time: float
    eigenvalues: np.ndarray


def _validate_transition_matrix(A: np.ndarray) -> None:
    """Check that A is a valid row-stochastic matrix."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Transition matrix must be square")
    if A.shape[0] < 2:
        raise ValueError("Transition matrix must have at least 2 states")
    if np.any(A < -1e-12):
        raise ValueError("Transition matrix entries must be non-negative")
    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"Rows must sum to 1; got sums {row_sums}"
        )


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a row-stochastic matrix.

    The stationary distribution pi satisfies pi @ A = pi, which is
    equivalent to finding the left eigenvector of A for eigenvalue 1.
    We solve (A^T - I) @ pi = 0 with the constraint sum(pi) = 1 by
    replacing the last equation with the normalization constraint.

    Parameters
    ----------
    A
        Row-stochastic transition matrix, shape (K, K).

    Returns
    -------
    np.ndarray
        Stationary distribution, shape (K,), summing to 1.
    """
    _validate_transition_matrix(A)
    K = A.shape[0]

    # Solve (A^T - I) pi = 0 with sum(pi) = 1.
    # Replace the last row of (A^T - I) with the normalization.
    M = A.T - np.eye(K)
    M[-1, :] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        # Fallback for reducible chains (e.g., the identity matrix).
        # Use the eigendecomposition: the left eigenvector for
        # eigenvalue 1 of A^T is a right eigenvector of A^T.
        evals, evecs = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(evals - 1.0))
        pi = np.real(evecs[:, idx])
        if pi.sum() < 0:
            pi = -pi

    # Clamp tiny negatives from numerical noise.
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


def entropy_rate(A: np.ndarray, pi: np.ndarray | None = None) -> float:
    """Shannon entropy rate of a stationary Markov chain.

    Parameters
    ----------
    A
        Row-stochastic transition matrix.
    pi
        Stationary distribution (computed if not provided).

    Returns
    -------
    float
        Entropy rate in bits per step.
    """
    _validate_transition_matrix(A)
    if pi is None:
        pi = stationary_distribution(A)

    h = 0.0
    for i in range(A.shape[0]):
        if pi[i] < 1e-15:
            continue
        for j in range(A.shape[1]):
            if A[i, j] > 1e-15:
                h -= pi[i] * A[i, j] * np.log2(A[i, j])
    return float(h)


def spectral_gap(A: np.ndarray) -> tuple[float, np.ndarray]:
    """Spectral gap and eigenvalues of the transition matrix.

    The spectral gap is 1 - |lambda_2|, where lambda_2 is the
    eigenvalue with the second-largest magnitude.  For an irreducible
    aperiodic chain the largest eigenvalue is exactly 1.

    Parameters
    ----------
    A
        Row-stochastic transition matrix.

    Returns
    -------
    gap
        The spectral gap, in (0, 1] for an ergodic chain.
    eigenvalues
        All eigenvalues, sorted by descending magnitude.
    """
    _validate_transition_matrix(A)
    eigs = np.linalg.eigvals(A)
    mags = np.abs(eigs)
    order = np.argsort(-mags)
    eigs_sorted = eigs[order]

    # The second-largest magnitude determines the gap.
    if len(eigs_sorted) < 2:
        return 1.0, eigs_sorted

    gap = 1.0 - float(np.abs(eigs_sorted[1]))
    return max(gap, 0.0), eigs_sorted


def mixing_time_bound(
    A: np.ndarray,
    epsilon: float = 0.25,
    pi: np.ndarray | None = None,
) -> float:
    """Upper bound on the mixing time from the spectral gap.

    The standard bound for a reversible chain is

        t_mix(epsilon) <= (1 / gap) * log(1 / (epsilon * pi_min))

    where pi_min is the smallest stationary probability and gap
    is the spectral gap.  This is conservative but dimension-free.

    Parameters
    ----------
    A
        Row-stochastic transition matrix.
    epsilon
        Total-variation distance threshold.
    pi
        Stationary distribution (computed if not provided).

    Returns
    -------
    float
        Upper bound on the mixing time in steps.
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be in (0, 1)")

    _validate_transition_matrix(A)
    if pi is None:
        pi = stationary_distribution(A)

    gap, _ = spectral_gap(A)
    if gap < 1e-15:
        return float("inf")

    pi_min = pi[pi > 0].min()
    return float(np.log(1.0 / (epsilon * pi_min)) / gap)


def diagnose(
    A: np.ndarray,
    epsilon: float = 0.25,
    state_names: list[str] | None = None,
) -> MarkovDiagnostics:
    """Compute all diagnostics for a transition matrix.

    Parameters
    ----------
    A
        Row-stochastic transition matrix.
    epsilon
        Threshold for the mixing time bound.
    state_names
        Optional list of state labels (for display; not stored).

    Returns
    -------
    MarkovDiagnostics
        Collected diagnostics.
    """
    _validate_transition_matrix(A)
    pi = stationary_distribution(A)
    h = entropy_rate(A, pi)
    gap, eigs = spectral_gap(A)
    t_mix = mixing_time_bound(A, epsilon=epsilon, pi=pi)

    return MarkovDiagnostics(
        stationary=pi,
        entropy_rate=h,
        spectral_gap=gap,
        mixing_time=t_mix,
        eigenvalues=eigs,
    )


def mean_dwell_times(A: np.ndarray) -> np.ndarray:
    """Expected dwell time in each state before transitioning out.

    For a Markov chain, the dwell time in state i is geometrically
    distributed with parameter 1 - A[i,i], so the expected dwell is

        E[dwell_i] = 1 / (1 - A[i,i])

    Parameters
    ----------
    A
        Row-stochastic transition matrix.

    Returns
    -------
    np.ndarray
        Expected dwell time per state, shape (K,).
    """
    _validate_transition_matrix(A)
    diag = np.diag(A)
    # Avoid division by zero for absorbing states.
    return np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)
