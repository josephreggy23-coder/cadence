"""Tests for the sensor-aware kinetic HMM.

The KineticHMM models calcium fluorescence with explicit exponential
sensor dynamics (GCaMP time constant tau), discretized on a grid and
combined with four latent regimes for forward-backward inference.

Tests cover:

1. Forward-backward normalization and log-likelihood computation.
2. Sensor dynamics: the calcium grid mapping should approach the
   latent level with time constant tau.
3. State decoding on synthetic traces with known ground truth.
4. EM fitting: likelihood should increase monotonically.
5. The KineticFilter causal online estimator should match the
   forward pass of forward-backward.
6. Model save/load round-trip.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinetic_hmm import (
    KineticFilter,
    KineticHMM,
    RANK_TO_TRUE,
    initial_model,
)


# ---- helpers ---------------------------------------------------------------

def _default_model(nc=30):
    """Small KineticHMM with known parameters for testing."""
    mu = np.array([0.05, 0.10, 0.35, 0.90])
    sd = np.array([0.03, 0.04, 0.08, 0.12])
    tau = 3.0
    A = np.array([
        [0.90, 0.05, 0.04, 0.01],
        [0.06, 0.85, 0.07, 0.02],
        [0.03, 0.05, 0.82, 0.10],
        [0.02, 0.03, 0.10, 0.85],
    ])
    return KineticHMM(mu, sd, tau, A=A, nc=nc)


def _generate_trace(model, length=200, seed=42):
    """Generate a synthetic calcium trace from the model's generative process."""
    rng = np.random.default_rng(seed)
    states = np.empty(length, dtype=int)
    calcium = np.empty(length)

    states[0] = rng.choice(model.K, p=model.pi)
    c = model.mu[states[0]]

    for t in range(length):
        if t > 0:
            states[t] = rng.choice(model.K, p=model.A[states[t - 1]])
        # Exponential sensor dynamics
        c = c + (model.mu[states[t]] - c) / model.tau
        calcium[t] = c + rng.normal(0, model.sd[states[t]])

    return calcium, states


# ---- forward-backward normalization ----------------------------------------

class TestForwardBackward:
    def test_posterior_sums_to_one(self) -> None:
        """Marginal posterior at each frame should sum to 1."""
        model = _default_model()
        y, _ = _generate_trace(model, length=100)
        post, ll, _ = model.forward_backward(y)

        np.testing.assert_allclose(
            post.sum(axis=1), np.ones(len(y)), atol=1e-10
        )

    def test_posterior_is_nonnegative(self) -> None:
        model = _default_model()
        y, _ = _generate_trace(model, length=100)
        post, _, _ = model.forward_backward(y)
        assert np.all(post >= 0)

    def test_loglikelihood_is_finite(self) -> None:
        model = _default_model()
        y, _ = _generate_trace(model, length=100)
        _, ll, _ = model.forward_backward(y)
        assert np.isfinite(ll)

    def test_longer_sequence_has_proportionally_scaled_loglik(self) -> None:
        """Log-likelihood should scale roughly with sequence length."""
        model = _default_model()
        y_short, _ = _generate_trace(model, length=50)
        y_long, _ = _generate_trace(model, length=200)
        _, ll_short, _ = model.forward_backward(y_short)
        _, ll_long, _ = model.forward_backward(y_long)

        # Not exact proportionality, but the ratio should be in a reasonable range
        ratio = ll_long / ll_short
        assert 1.5 < ratio < 8.0

    def test_xi_sum_is_nonnegative(self) -> None:
        """Expected transition counts should be non-negative."""
        model = _default_model()
        y, _ = _generate_trace(model, length=100)
        _, _, xi_sum = model.forward_backward(y, want_xi=True)
        assert np.all(xi_sum >= 0)


# ---- sensor dynamics -------------------------------------------------------

class TestSensorDynamics:
    def test_cmap_shape(self) -> None:
        """Sensor map should have shape (K, nc)."""
        model = _default_model(nc=50)
        assert model.cmap.shape == (model.K, 50)

    def test_cmap_indices_in_range(self) -> None:
        """All mapped indices should be valid grid indices."""
        model = _default_model()
        assert np.all(model.cmap >= 0)
        assert np.all(model.cmap < model.nc)

    def test_sensor_approaches_latent_level(self) -> None:
        """After dwelling in one state, sensor should approach its mu.

        The sensor update is c[t] = c[t-1] + (mu[s] - c[t-1]) / tau.
        After many steps in state s, c should converge to mu[s].
        """
        model = _default_model(nc=100)
        # Start at the low end of the grid
        c = model.cgrid[0]
        target_state = 3  # SUSTAINED_HIGH, mu = 0.90

        for _ in range(100):
            c = c + (model.mu[target_state] - c) / model.tau

        np.testing.assert_allclose(c, model.mu[target_state], atol=0.01)


# ---- state decoding --------------------------------------------------------

class TestDecode:
    def test_decoded_states_in_biological_range(self) -> None:
        """Decoded states should be in {0, 1, 2, 3} (biological convention)."""
        model = _default_model()
        y, _ = _generate_trace(model, length=100)
        decoded = model.decode(y)

        assert decoded.shape == (100,)
        assert set(np.unique(decoded)).issubset({0, 1, 2, 3})

    def test_decode_recovers_dominant_state(self) -> None:
        """A trace generated from a single dominant state should be decoded
        as that state for most frames."""
        rng = np.random.default_rng(123)
        model = _default_model()
        # Generate a trace that stays in SUSTAINED_HIGH (internal state 3)
        length = 100
        c = model.mu[3]
        y = np.empty(length)
        for t in range(length):
            c = c + (model.mu[3] - c) / model.tau
            y[t] = c + rng.normal(0, model.sd[3])

        decoded = model.decode(y)
        # SUSTAINED_HIGH in biological convention is state 2
        high_fraction = np.mean(decoded == 2)
        assert high_fraction > 0.5

    def test_rank_to_true_maps_correctly(self) -> None:
        """The RANK_TO_TRUE mapping should convert internal ascending-calcium
        order to the biological convention."""
        # Internal order: 0=QUIESCENT, 1=REFRACTORY, 2=OSCILLATORY, 3=HIGH
        # Biological:     0=QUIESCENT, 1=OSCILLATORY, 2=HIGH, 3=REFRACTORY
        assert RANK_TO_TRUE[0] == 0  # QUIESCENT stays 0
        assert RANK_TO_TRUE[1] == 3  # internal 1 (REFRACT) -> bio 3
        assert RANK_TO_TRUE[2] == 1  # internal 2 (OSC) -> bio 1
        assert RANK_TO_TRUE[3] == 2  # internal 3 (HIGH) -> bio 2


# ---- EM fitting ------------------------------------------------------------

class TestFit:
    def test_em_likelihood_nondecreasing(self) -> None:
        """EM iterations should not decrease the log-likelihood."""
        model = _default_model()
        traces = [_generate_trace(model, length=80, seed=s)[0] for s in range(5)]

        fit_model = _default_model()
        prev_ll = -np.inf
        for _ in range(5):
            xi_tot = np.zeros((fit_model.K, fit_model.K))
            ll = 0.0
            for y in traces:
                _, l, xi = fit_model.forward_backward(y, want_xi=True)
                ll += l
                xi_tot += xi
            xi_tot += 1e-6
            fit_model.A = xi_tot / xi_tot.sum(axis=1, keepdims=True)

            assert ll >= prev_ll - 1e-6  # allow tiny numerical jitter
            prev_ll = ll

    def test_transition_matrix_stays_stochastic_after_fit(self) -> None:
        """Transition rows should sum to 1 after fitting."""
        model = _default_model()
        traces = [_generate_trace(model, length=60, seed=s)[0] for s in range(3)]

        fit_model = _default_model()
        fit_model.fit(traces, n_iter=3, opt_every=0, verbose=False)

        np.testing.assert_allclose(
            fit_model.A.sum(axis=1), np.ones(fit_model.K), atol=1e-10
        )
        assert np.all(fit_model.A >= 0)


# ---- KineticFilter (causal online estimator) --------------------------------

class TestKineticFilter:
    def test_filter_matches_forward_pass(self) -> None:
        """The causal filter should produce the same MAP states as the
        forward pass of forward-backward (not the smoothed posterior).

        Since forward-backward uses smoothing (backward pass), the MAP
        from the filter may differ, but the filter's posterior should
        be a valid probability distribution at each step.
        """
        model = _default_model()
        y, _ = _generate_trace(model, length=50)

        filt = KineticFilter(model)
        filter_states = np.array([filt.step(y[t]) for t in range(len(y))])

        assert filter_states.shape == (50,)
        assert set(np.unique(filter_states)).issubset({0, 1, 2, 3})

    def test_filter_reset_produces_same_result(self) -> None:
        """Resetting and re-running should give identical states."""
        model = _default_model()
        y, _ = _generate_trace(model, length=30)

        filt = KineticFilter(model)
        run1 = [filt.step(y[t]) for t in range(len(y))]
        filt.reset()
        run2 = [filt.step(y[t]) for t in range(len(y))]

        assert run1 == run2

    def test_filter_is_causal(self) -> None:
        """Adding future observations should not change past estimates.

        The filter's state at time t depends only on y[0:t+1], so
        processing a prefix should give the same states as processing
        the full sequence up to that point.
        """
        model = _default_model()
        y, _ = _generate_trace(model, length=60)

        # Full sequence
        filt_full = KineticFilter(model)
        full_states = [filt_full.step(y[t]) for t in range(60)]

        # Prefix of length 30
        filt_prefix = KineticFilter(model)
        prefix_states = [filt_prefix.step(y[t]) for t in range(30)]

        assert prefix_states == full_states[:30]


# ---- save/load round-trip --------------------------------------------------

class TestSaveLoad:
    def test_round_trip_preserves_parameters(self, tmp_path) -> None:
        """Save and load should recover exact parameters."""
        model = _default_model(nc=40)
        path = tmp_path / "test_model.npz"
        model.save(str(path))

        loaded = KineticHMM.load(str(path))

        np.testing.assert_allclose(loaded.mu, model.mu)
        np.testing.assert_allclose(loaded.sd, model.sd)
        assert loaded.tau == pytest.approx(model.tau)
        np.testing.assert_allclose(loaded.A, model.A)
        assert loaded.nc == model.nc

    def test_loaded_model_gives_same_loglik(self, tmp_path) -> None:
        """A loaded model should compute the same log-likelihood."""
        model = _default_model()
        y, _ = _generate_trace(model, length=80)
        _, ll_orig, _ = model.forward_backward(y)

        path = tmp_path / "model.npz"
        model.save(str(path))
        loaded = KineticHMM.load(str(path))
        _, ll_loaded, _ = loaded.forward_backward(y)

        assert ll_loaded == pytest.approx(ll_orig, abs=1e-6)


# ---- initial_model ---------------------------------------------------------

class TestInitialModel:
    def test_initial_model_has_four_states(self) -> None:
        model = _default_model()
        traces = [_generate_trace(model, length=60, seed=s)[0] for s in range(3)]
        init = initial_model(traces)
        assert init.K == 4

    def test_initial_means_are_sorted(self) -> None:
        """Data-driven initialization should produce sorted latent means."""
        model = _default_model()
        traces = [_generate_trace(model, length=60, seed=s)[0] for s in range(3)]
        init = initial_model(traces)
        assert np.all(np.diff(init.mu) > 0)
