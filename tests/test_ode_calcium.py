"""Tests for the continuous-time ODE calcium dynamics module.

Covers the fixed-step RK4 integrator, adaptive Dormand-Prince integrator,
steady-state solver, conservation properties, and bifurcation detection
in the Li-Rinzel model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ode_calcium import (
    CalciumODE,
    adaptive_rk45,
    bifurcation_scan,
    rk4_step,
    simulate_calcium,
    steady_state,
)


# --------------------------------------------------------------------------- #
# RK4 on known ODEs
# --------------------------------------------------------------------------- #

class TestRK4ExponentialDecay:
    """RK4 on dy/dt = -y, y(0) = 1.  Exact solution: y(t) = exp(-t)."""

    def test_accuracy_at_t10(self) -> None:
        f = lambda t, y: -y
        y = np.array([1.0])
        dt = 0.01
        t = 0.0
        for _ in range(1000):
            y = rk4_step(f, y, t, dt)
            t += dt
        # Global error should be O(dt^4) ~ 1e-8
        assert abs(y[0] - np.exp(-10.0)) < 1e-9

    def test_step_halving_improves_accuracy(self) -> None:
        """Halving dt should reduce the global error by ~16x (order 4)."""
        f = lambda t, y: -y

        errors = []
        for dt in [0.02, 0.01]:
            y = np.array([1.0])
            t = 0.0
            n = int(5.0 / dt)
            for _ in range(n):
                y = rk4_step(f, y, t, dt)
                t += dt
            errors.append(abs(y[0] - np.exp(-5.0)))

        ratio = errors[0] / errors[1]
        # For a 4th-order method, halving dt gives 2^4 = 16x improvement
        assert ratio > 12.0


class TestRK4HarmonicOscillator:
    """RK4 on [dx/dt, dv/dt] = [v, -x].  Exact: x = cos(t), v = -sin(t)."""

    def test_one_period_return(self) -> None:
        f = lambda t, y: np.array([y[1], -y[0]])
        y = np.array([1.0, 0.0])
        dt = 0.0001
        t = 0.0
        n = int(round(2.0 * np.pi / dt))
        for _ in range(n):
            y = rk4_step(f, y, t, dt)
            t += dt
        # Compare against exact solution at the actual endpoint t = n*dt
        # (which may differ slightly from 2*pi)
        assert abs(y[0] - np.cos(t)) < 1e-7
        assert abs(y[1] - (-np.sin(t))) < 1e-7

    def test_mid_period_accuracy(self) -> None:
        f = lambda t, y: np.array([y[1], -y[0]])
        y = np.array([1.0, 0.0])
        dt = 0.0001
        t = 0.0
        n = int(round(np.pi / dt))
        for _ in range(n):
            y = rk4_step(f, y, t, dt)
            t += dt
        # Compare against exact solution at actual endpoint
        assert abs(y[0] - np.cos(t)) < 1e-7
        assert abs(y[1] - (-np.sin(t))) < 1e-7


# --------------------------------------------------------------------------- #
# Conservation laws
# --------------------------------------------------------------------------- #

class TestConservation:
    """Test that the integrators preserve conserved quantities."""

    def test_harmonic_oscillator_energy(self) -> None:
        """Harmonic oscillator energy E = x^2 + v^2 should be nearly conserved.

        This tests the integrator's symplectic-like behaviour on a
        Hamiltonian system, analogous to testing total-calcium conservation
        in a closed astrocyte model.
        """
        f = lambda t, y: np.array([y[1], -y[0]])
        y = np.array([1.0, 0.0])
        E0 = y[0] ** 2 + y[1] ** 2
        dt = 0.01
        t = 0.0
        for _ in range(10_000):  # 100 seconds
            y = rk4_step(f, y, t, dt)
            t += dt
        E = y[0] ** 2 + y[1] ** 2
        assert abs(E - E0) / E0 < 1e-6

    def test_calcium_flux_balance_at_steady_state(self) -> None:
        """At steady state, net flux into the cytosol must be zero.

        With the plasma membrane pump, the balance is:
            J_release + J_leak = J_SERCA + J_pm
        """
        params = CalciumODE()
        for ip3 in [0.1, 0.3, 0.8]:
            C, h = steady_state(params, ip3)
            inward = params.j_release(C, ip3, h) + params.j_leak(C)
            outward = params.j_serca(C) + params.j_pm(C)
            assert abs(inward - outward) < 1e-9

    def test_closed_system_total_calcium(self) -> None:
        """With no PM pump, total cell calcium C + c1*C_ER is conserved.

        This is guaranteed by the model construction (C_ER is computed from
        conservation) but verifying it confirms the implementation couples
        the variables correctly.
        """
        params = CalciumODE(v_pm=0.0)
        t, C = simulate_calcium(params, 0.4, (0.0, 20.0), dt=0.005)
        # Total calcium should be c_total at every time step
        total = C + params.c1 * np.array([params.c_er(c) for c in C])
        np.testing.assert_allclose(total, params.c_total, atol=1e-12)


# --------------------------------------------------------------------------- #
# Adaptive integrator
# --------------------------------------------------------------------------- #

class TestAdaptiveRK45:

    def test_exponential_decay_accuracy(self) -> None:
        """Adaptive RK45 on dy/dt = -y should match exp(-t) closely."""
        f = lambda t, y: -y
        t, y = adaptive_rk45(f, np.array([1.0]), (0.0, 10.0))
        assert abs(y[-1, 0] - np.exp(-10.0)) < 1e-8

    def test_reaches_endpoint(self) -> None:
        """Integration should reach the requested final time."""
        f = lambda t, y: -y
        t, y = adaptive_rk45(f, np.array([1.0]), (0.0, 5.0))
        assert abs(t[-1] - 5.0) < 1e-12

    def test_adaptive_more_accurate_than_fixed(self) -> None:
        """With comparable effort, adaptive should beat fixed-step RK4."""
        # A moderately fast decay: dy/dt = -10y, y(0) = 1
        f = lambda t, y: np.array([-10.0 * y[0]])

        # Fixed-step RK4 with dt = 0.01, integrated to t = 1
        y_fixed = np.array([1.0])
        t_cur = 0.0
        dt = 0.01
        for _ in range(100):
            y_fixed = rk4_step(f, y_fixed, t_cur, dt)
            t_cur += dt
        err_fixed = abs(y_fixed[0] - np.exp(-10.0))

        # Adaptive with tight tolerance
        t_adap, y_adap = adaptive_rk45(
            f, np.array([1.0]), (0.0, 1.0), rtol=1e-10, atol=1e-12,
        )
        err_adap = abs(y_adap[-1, 0] - np.exp(-10.0))

        assert err_adap < err_fixed

    def test_adapts_to_stiffness(self) -> None:
        """Step size should shrink in stiff regions and grow in smooth ones."""
        # Exponential decay: solution is smooth, steps should grow
        f = lambda t, y: -y
        t, y = adaptive_rk45(f, np.array([1.0]), (0.0, 5.0), max_step=1.0)
        # Should use fewer steps than a fixed dt=0.01 method (500 steps)
        assert len(t) < 200

    def test_multidimensional_system(self) -> None:
        """Adaptive solver handles vector-valued ODEs (harmonic oscillator)."""
        f = lambda t, y: np.array([y[1], -y[0]])
        t, y = adaptive_rk45(
            f, np.array([1.0, 0.0]), (0.0, 2 * np.pi),
            rtol=1e-10, atol=1e-12,
        )
        assert abs(y[-1, 0] - 1.0) < 1e-6
        assert abs(y[-1, 1] - 0.0) < 1e-6


# --------------------------------------------------------------------------- #
# Steady-state solver
# --------------------------------------------------------------------------- #

class TestSteadyState:

    def test_residual_is_small(self) -> None:
        """Steady state should satisfy dC/dt ~ 0 and dh/dt ~ 0."""
        params = CalciumODE()
        for ip3 in [0.05, 0.1, 0.2, 0.8, 1.0]:
            C, h = steady_state(params, ip3)
            dydt = params.derivatives(0.0, np.array([C, h]), ip3)
            assert np.max(np.abs(dydt)) < 1e-9, (
                f"Residual too large at IP3={ip3}: {dydt}"
            )

    def test_h_equals_h_inf(self) -> None:
        """At steady state, h must equal h_inf(C, IP3)."""
        params = CalciumODE()
        for ip3 in [0.05, 0.1, 0.3, 0.5, 1.0]:
            C, h = steady_state(params, ip3)
            h_expected = params.h_inf(C, ip3)
            assert abs(h - h_expected) < 1e-8, (
                f"h mismatch at IP3={ip3}: h={h}, h_inf={h_expected}"
            )

    def test_physically_reasonable_values(self) -> None:
        """Steady-state C and h should be in physically valid ranges."""
        params = CalciumODE()
        for ip3 in [0.01, 0.1, 0.5, 1.0, 2.0]:
            C, h = steady_state(params, ip3)
            assert 0 < C < params.c_total, f"C={C} out of range at IP3={ip3}"
            assert 0 < h < 1, f"h={h} out of range at IP3={ip3}"

    def test_C_increases_with_ip3(self) -> None:
        """Steady-state calcium should generally increase with IP3."""
        params = CalciumODE()
        ip3_values = [0.01, 0.05, 0.1, 0.5, 1.0]
        C_values = [steady_state(params, ip3)[0] for ip3 in ip3_values]
        # Not strictly monotonic everywhere (bistability possible),
        # but the endpoints should follow the expected trend
        assert C_values[-1] > C_values[0]

    def test_convergence_from_different_initial_guesses(self) -> None:
        """Newton should converge to the same solution from varied guesses."""
        params = CalciumODE()
        ip3 = 0.1
        results = []
        for C0 in [0.01, 0.1, 0.3, 0.5]:
            C, h = steady_state(params, ip3, C0=C0)
            results.append((C, h))
        # All should converge to the same steady state
        for C, h in results:
            assert abs(C - results[0][0]) < 1e-6
            assert abs(h - results[0][1]) < 1e-6


# --------------------------------------------------------------------------- #
# Calcium simulation
# --------------------------------------------------------------------------- #

class TestSimulateCalcium:

    def test_output_shapes(self) -> None:
        params = CalciumODE()
        t, C = simulate_calcium(params, 0.1, (0.0, 10.0), dt=0.01)
        assert t.shape == C.shape
        assert len(t) == 1001  # ceil(10/0.01) + 1

    def test_calcium_stays_physical(self) -> None:
        """Calcium should remain non-negative and below c_total."""
        params = CalciumODE()
        t, C = simulate_calcium(params, 0.5, (0.0, 50.0), dt=0.005)
        assert np.all(C >= -1e-10), f"Negative calcium: min = {C.min()}"
        assert np.all(C < params.c_total + 1e-10), (
            f"Calcium exceeds total: max = {C.max()}"
        )

    def test_callable_ip3(self) -> None:
        """IP3 can be a time-varying callable."""
        params = CalciumODE()
        # Step function: IP3 jumps from 0.01 to 0.5 at t = 5
        ip3_fn = lambda t: 0.01 if t < 5.0 else 0.5
        t, C = simulate_calcium(params, ip3_fn, (0.0, 20.0), dt=0.01)
        # After the step, calcium should increase from its resting level
        C_before = C[t < 4.0]
        C_after = C[t > 15.0]
        assert np.mean(C_after) > np.mean(C_before)

    def test_constant_ip3_approaches_steady_state(self) -> None:
        """With constant sub-threshold IP3, calcium should approach C*."""
        params = CalciumODE()
        ip3 = 0.1
        C_ss, _ = steady_state(params, ip3)
        t, C = simulate_calcium(params, ip3, (0.0, 100.0), dt=0.01)
        # Final calcium should be close to the steady state
        assert abs(C[-1] - C_ss) < 0.01


# --------------------------------------------------------------------------- #
# Bifurcation detection
# --------------------------------------------------------------------------- #

class TestBifurcationScan:

    def test_detects_hopf_bifurcation(self) -> None:
        """The Li-Rinzel model has a known Hopf bifurcation in the IP3 range."""
        params = CalciumODE()
        ip3_range = np.linspace(0.01, 1.0, 300)
        result = bifurcation_scan(params, ip3_range)

        assert len(result["hopf_indices"]) >= 1, (
            "No Hopf bifurcation detected; expected at least one"
        )

    def test_hopf_at_reasonable_ip3(self) -> None:
        """Hopf bifurcation should occur at a biologically plausible IP3."""
        params = CalciumODE()
        ip3_range = np.linspace(0.01, 1.0, 300)
        result = bifurcation_scan(params, ip3_range)

        for idx in result["hopf_indices"]:
            ip3_bif = result["ip3"][idx]
            assert 0.05 < ip3_bif < 0.95, (
                f"Hopf bifurcation at IP3={ip3_bif} is outside expected range"
            )

    def test_stability_transition(self) -> None:
        """There should be both stable and unstable regions in the scan."""
        params = CalciumODE()
        ip3_range = np.linspace(0.01, 1.0, 300)
        result = bifurcation_scan(params, ip3_range)

        assert np.any(result["stable"]), "No stable region found"
        assert np.any(~result["stable"]), "No unstable region found"

    def test_output_shapes(self) -> None:
        """All output arrays should have consistent shapes."""
        params = CalciumODE()
        ip3_range = np.linspace(0.1, 0.8, 50)
        result = bifurcation_scan(params, ip3_range)

        assert result["ip3"].shape == (50,)
        assert result["C_ss"].shape == (50,)
        assert result["h_ss"].shape == (50,)
        assert result["eigenvalues"].shape == (50, 2)
        assert result["stable"].shape == (50,)
        assert result["hopf_indices"].dtype == int

    def test_low_ip3_is_stable(self) -> None:
        """At very low IP3, the quiescent steady state should be stable."""
        params = CalciumODE()
        ip3_range = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = bifurcation_scan(params, ip3_range)
        assert np.all(result["stable"]), (
            "Expected all-stable at low IP3"
        )


# --------------------------------------------------------------------------- #
# CalciumODE dataclass
# --------------------------------------------------------------------------- #

class TestCalciumODE:

    def test_frozen(self) -> None:
        params = CalciumODE()
        with pytest.raises(AttributeError):
            params.v_release = 10.0  # type: ignore[misc]

    def test_custom_parameters(self) -> None:
        params = CalciumODE(v_release=12.0, K_SERCA=0.05)
        assert params.v_release == 12.0
        assert params.K_SERCA == 0.05
        # Other parameters should keep defaults
        assert params.v_leak == 0.11

    def test_c_er_conservation(self) -> None:
        """C + c1 * C_ER should equal c_total for any C."""
        params = CalciumODE()
        for C in [0.01, 0.1, 0.5, 1.0, 1.9]:
            total = C + params.c1 * params.c_er(C)
            assert abs(total - params.c_total) < 1e-12

    def test_h_inf_bounds(self) -> None:
        """h_inf should always be in (0, 1)."""
        params = CalciumODE()
        for ip3 in [0.01, 0.1, 1.0, 10.0]:
            for C in [0.001, 0.1, 0.5, 2.0]:
                h = params.h_inf(C, ip3)
                assert 0 < h < 1, f"h_inf={h} out of (0,1) at C={C}, IP3={ip3}"

    def test_j_serca_hill_shape(self) -> None:
        """SERCA flux should saturate at v_SERCA for large C."""
        params = CalciumODE()
        # At very large C, J_SERCA -> v_SERCA
        assert abs(params.j_serca(100.0) - params.v_SERCA) < 0.001
        # At C = K_SERCA, J_SERCA should be v_SERCA / 2
        j_half = params.j_serca(params.K_SERCA)
        assert abs(j_half - params.v_SERCA / 2.0) < 1e-10
