"""Continuous-time ODE model for astrocyte calcium dynamics.

This module implements the Li-Rinzel reduction of the De Young-Keizer IP3
receptor model, providing a continuous differential equation description of
cytosolic calcium dynamics in astrocytes.  It bridges the gap between the
discrete Markov state model in ``kinetic_hmm.py`` and the underlying
biophysical mechanisms.

Model equations
---------------
The state vector is (C, h), where C is cytosolic calcium concentration (uM)
and h is the IP3 receptor inactivation gating variable (dimensionless, 0--1).

    dC/dt = J_release(C, IP3, h) - J_SERCA(C) + J_leak(C) - J_pm(C)
    dh/dt = (h_inf(C, IP3) - h) / tau_h(C, IP3)

Flux terms:

    J_release = c1 * v_release * m_inf^3 * n_inf^3 * h^3 * (C_ER - C)
        IP3 receptor channel flux.  m_inf = IP3 / (IP3 + K_IP3) is the IP3
        activation gate; n_inf = C / (C + K_act) is the calcium activation
        gate; h is the slow inactivation gate; (C_ER - C) is the driving
        force from the ER-cytosol concentration gradient.  The c1 factor
        converts from ER-volume to cytosol-volume units.

    J_SERCA = v_SERCA * C^2 / (K_SERCA^2 + C^2)
        Sarco/endoplasmic reticulum Ca-ATPase pump.  Hill coefficient of 2
        reflects the two calcium ions transported per ATP hydrolysed.

    J_leak = c1 * v_leak * (C_ER - C)
        Passive leak from the ER lumen to the cytosol, scaled by c1 for
        the same ER-to-cytosol volume conversion as J_release.

    J_pm = v_pm * C / (K_pm + C)
        Plasma membrane calcium pump (PMCA).  Extrudes calcium to the
        extracellular space.

Conservation:

    C_ER = (c_total - C) / c1

    where c_total is the total cell calcium and c1 is the ER-to-cytosol
    volume ratio.  This enforces conservation of total calcium within the
    cell (modulo plasma membrane fluxes).

IP3R inactivation kinetics (derived from the De Young-Keizer reduction):

    Q2   = d2 * (IP3 + K_IP3) / (IP3 + d3)
    h_inf = Q2 / (Q2 + C)
    tau_h = 1 / (a2 * (Q2 + C))

    Q2 captures the IP3-dependent shift of the inactivation curve: higher
    IP3 lowers Q2, shifting the h_inf curve to the left and allowing
    oscillations at lower calcium concentrations.

Numerical methods
-----------------
Two integrators are provided:

- ``rk4_step``: fixed-step classical fourth-order Runge-Kutta, O(dt^4) global
  error.  Simple, efficient for smooth dynamics at known time scales.

- ``adaptive_rk45``: Dormand-Prince RK4(5) with embedded error estimation and
  automatic step-size control.  The 4th-order solution advances the state; the
  difference from the 5th-order solution estimates local truncation error.
  The step is accepted when the L2-normed scaled error is below 1.

References
----------
Li, Y.-X. & Rinzel, J. (1994).  Equations for InsP3 receptor-mediated
[Ca2+]i oscillations derived from a detailed kinetic model: a
Hodgkin-Huxley like formalism.  J. Theor. Biol. 166, 461--473.

De Young, G. W. & Keizer, J. (1992).  A single-pool inositol
1,4,5-trisphosphate-receptor-based model for agonist-stimulated
oscillations in Ca2+ concentration.  PNAS 89, 9895--9899.

Dormand, J. R. & Prince, P. J. (1980).  A family of embedded
Runge-Kutta formulae.  J. Comput. Appl. Math. 6, 19--26.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------- #
# Dormand-Prince RK4(5) Butcher tableau coefficients
# --------------------------------------------------------------------------- #
#
# Nodes (c_i):
#   [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
#
# Coupling matrix (a_ij), lower-triangular -- row i uses stages 1..i-1:
#   Row 2: [1/5]
#   Row 3: [3/40,        9/40]
#   Row 4: [44/45,      -56/15,       32/9]
#   Row 5: [19372/6561, -25360/2187,  64448/6561,  -212/729]
#   Row 6: [9017/3168,  -355/33,      46732/5247,   49/176,   -5103/18656]
#   Row 7: [35/384,      0,           500/1113,     125/192,  -2187/6784,  11/84]
#
# 4th-order weights (b_i, used to advance the solution):
#   [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
#
# 5th-order weights (b*_i, used for error estimation):
#   [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
#
# Error coefficients e_i = b*_i - b_i (used to compute the error vector
# without forming the 5th-order solution explicitly):

_DP_E = np.array([
    -71.0 / 57600,       # e1
     0.0,                # e2  (both b and b* are zero)
     71.0 / 16695,       # e3
    -71.0 / 1920,        # e4
     17253.0 / 339200,   # e5
    -22.0 / 525,         # e6
     1.0 / 40,           # e7
])


# --------------------------------------------------------------------------- #
# Calcium dynamics model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalciumODE:
    """Parameter set for the Li-Rinzel calcium dynamics model.

    All concentrations are in micromolar (uM) and all times in seconds (s).
    Default values reproduce the standard Li-Rinzel parameterisation with
    the addition of a plasma membrane pump for biological completeness.

    Attributes
    ----------
    v_release : float
        Maximal IP3R channel flux rate (s^-1).
    K_IP3 : float
        IP3 dissociation constant for IP3R activation (uM).  Denoted d1
        in the De Young-Keizer model.
    K_act : float
        Calcium dissociation constant for IP3R activation (uM).  Denoted
        d5 in the De Young-Keizer model.
    d2 : float
        Calcium dissociation constant for IP3R inactivation (uM).
    d3 : float
        IP3 dissociation constant for the inhibitory calcium binding site
        (uM).
    a2 : float
        IP3R inactivation rate constant (uM^-1 s^-1).
    v_SERCA : float
        Maximal SERCA pump rate (uM/s).
    K_SERCA : float
        SERCA half-activation calcium concentration (uM).
    v_leak : float
        ER-to-cytosol leak rate constant (s^-1).
    v_pm : float
        Maximal plasma membrane pump rate (uM/s).
    K_pm : float
        Plasma membrane pump half-activation concentration (uM).
    c_total : float
        Total cell calcium, cytosol + ER (uM).
    c1 : float
        ER-to-cytosol volume ratio (dimensionless).
    """

    # IP3 receptor channel parameters
    v_release: float = 6.0       # s^-1
    K_IP3: float = 0.13          # uM  (d1 in De Young-Keizer)
    K_act: float = 0.082         # uM  (d5 in De Young-Keizer)
    d2: float = 1.049            # uM
    d3: float = 0.9434           # uM
    a2: float = 0.2              # uM^-1 s^-1

    # SERCA pump
    v_SERCA: float = 0.9         # uM/s
    K_SERCA: float = 0.1         # uM

    # ER passive leak
    v_leak: float = 0.11         # s^-1

    # Plasma membrane pump
    v_pm: float = 0.07           # uM/s
    K_pm: float = 0.3            # uM

    # Conservation / cell geometry
    c_total: float = 2.0         # uM  (total Ca in cytosol + ER)
    c1: float = 0.185            # dimensionless (V_ER / V_cyt)

    # ------------------------------------------------------------------ #
    # Derived quantities and flux functions
    # ------------------------------------------------------------------ #

    def c_er(self, C: float) -> float:
        """ER lumen calcium from total-calcium conservation.

        C_ER = (c_total - C) / c1

        At rest C ~ 0.05--0.5 uM gives C_ER ~ 8--10 uM, consistent with
        measured ER calcium concentrations in astrocytes.
        """
        return (self.c_total - C) / self.c1

    def j_release(self, C: float, ip3: float, h: float) -> float:
        """IP3R channel release flux (uM/s).

        J_release = c1 * v_release * (IP3 / (IP3 + K_IP3))^3
                                    * (C / (C + K_act))^3
                                    * h^3
                                    * (C_ER - C)

        The c1 factor converts the flux from ER-volume units to
        cytosol-volume units (dC/dt is per cytosol volume).  The three
        cubic gating terms correspond to the De Young-Keizer receptor
        subunit model: each of the three identical subunits must have IP3
        bound (m_inf), calcium bound at the activating site (n_inf), and
        be free of calcium at the inhibitory site (h).
        """
        m_inf = ip3 / (ip3 + self.K_IP3)
        n_inf = C / (C + self.K_act)
        return (self.c1 * self.v_release
                * m_inf ** 3 * n_inf ** 3 * h ** 3
                * (self.c_er(C) - C))

    def j_serca(self, C: float) -> float:
        """SERCA pump flux (uM/s).

        J_SERCA = v_SERCA * C^2 / (K_SERCA^2 + C^2)

        Hill coefficient 2 reflects cooperative binding of two Ca ions.
        """
        return self.v_SERCA * C ** 2 / (self.K_SERCA ** 2 + C ** 2)

    def j_leak(self, C: float) -> float:
        """ER passive leak flux (uM/s).

        J_leak = c1 * v_leak * (C_ER - C)

        A small constitutive leak from the ER lumen through non-specific
        channels balances the SERCA pump at rest.  The c1 factor converts
        from ER-volume to cytosol-volume units, matching J_release.
        """
        return self.c1 * self.v_leak * (self.c_er(C) - C)

    def j_pm(self, C: float) -> float:
        """Plasma membrane pump flux (uM/s).

        J_pm = v_pm * C / (K_pm + C)

        First-order Michaelis-Menten kinetics for calcium extrusion across
        the plasma membrane.
        """
        return self.v_pm * C / (self.K_pm + C)

    def h_inf(self, C: float, ip3: float) -> float:
        """Steady-state IP3R inactivation.

        Q2    = d2 * (IP3 + K_IP3) / (IP3 + d3)
        h_inf = Q2 / (Q2 + C)

        As cytosolic calcium rises, h_inf decreases: the receptor
        inactivates, providing negative feedback that terminates the
        calcium puff and enables oscillatory dynamics.
        """
        Q2 = self.d2 * (ip3 + self.K_IP3) / (ip3 + self.d3)
        return Q2 / (Q2 + C)

    def tau_h(self, C: float, ip3: float) -> float:
        """Time constant for IP3R inactivation (s).

        tau_h = 1 / (a2 * (Q2 + C))

        The time constant decreases with rising calcium, so inactivation
        accelerates during a calcium transient -- the system self-limits.
        """
        Q2 = self.d2 * (ip3 + self.K_IP3) / (ip3 + self.d3)
        return 1.0 / (self.a2 * (Q2 + C))

    def derivatives(self, t: float, y: np.ndarray, ip3: float) -> np.ndarray:
        """Right-hand side of the calcium ODE system.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous dynamics; included for
            compatibility with ODE integrator interfaces).
        y : ndarray, shape (2,)
            State vector [C, h].
        ip3 : float
            Current IP3 concentration (uM).

        Returns
        -------
        dydt : ndarray, shape (2,)
            Time derivatives [dC/dt, dh/dt].
        """
        C, h = float(y[0]), float(y[1])
        dCdt = (self.j_release(C, ip3, h)
                - self.j_serca(C)
                + self.j_leak(C)
                - self.j_pm(C))
        dhdt = (self.h_inf(C, ip3) - h) / self.tau_h(C, ip3)
        return np.array([dCdt, dhdt])


# --------------------------------------------------------------------------- #
# Fixed-step classical RK4 integrator
# --------------------------------------------------------------------------- #

def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    """Advance one step with the classic fourth-order Runge-Kutta method.

    Given dy/dt = f(t, y), advance from (t, y) to (t + dt, y_next):

        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt * k1 / 2)
        k3 = f(t + dt/2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)
        y_next = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    The method is fourth-order accurate: local truncation error is O(dt^5)
    and global error is O(dt^4).

    Parameters
    ----------
    f : callable (t, y) -> dy/dt
        Right-hand side of the ODE system.
    y : ndarray
        Current state vector.
    t : float
        Current time.
    dt : float
        Time step.

    Returns
    -------
    y_next : ndarray
        State at t + dt.
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + dt * k1 / 2.0)
    k3 = f(t + dt / 2.0, y + dt * k2 / 2.0)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# --------------------------------------------------------------------------- #
# Adaptive Dormand-Prince RK4(5) integrator
# --------------------------------------------------------------------------- #

def adaptive_rk45(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE with adaptive step-size control (Dormand-Prince).

    Uses the Dormand-Prince RK4(5) pair: the 4th-order solution advances the
    state while an embedded 5th-order solution estimates local truncation
    error.  The step size is adjusted so the scaled error satisfies

        sc_i   = atol + rtol * max(|y_i|, |y_next_i|)
        E      = sqrt( (1/n) * sum_i (err_i / sc_i)^2 )

    A step is accepted when E <= 1, and the next step size is

        dt_new = dt * clamp(0.9 * E^{-1/5}, 0.2, 5.0)

    The method exploits the FSAL (first-same-as-last) property: the 7th
    stage of one step equals the 1st stage of the next, saving one function
    evaluation per accepted step.

    See the module-level ``_DP_E`` array and Butcher tableau comment for the
    full Dormand-Prince coefficients.

    Parameters
    ----------
    f : callable (t, y) -> dy/dt
        Right-hand side of the ODE system.
    y0 : ndarray
        Initial state vector.
    t_span : (t_start, t_end)
        Integration interval.
    rtol, atol : float
        Relative and absolute error tolerances.
    max_step : float
        Maximum allowed step size.

    Returns
    -------
    t : ndarray, shape (N,)
        Accepted time points (non-uniform spacing).
    y : ndarray, shape (N, n_vars)
        Solution at each accepted time point.
    """
    t0, tf = t_span
    y = np.asarray(y0, dtype=float).copy()
    t = t0

    # Initial step size heuristic
    dt = min(max_step, (tf - t0) * 1e-3)
    min_step = 1e-14 * abs(tf - t0)

    ts: List[float] = [t]
    ys: List[np.ndarray] = [y.copy()]

    # Precompute the first stage (FSAL reuse on subsequent accepted steps)
    k1 = f(t, y)
    max_iter = 2_000_000

    for _ in range(max_iter):
        if t >= tf:
            break

        dt = min(dt, tf - t, max_step)
        dt = max(dt, min_step)

        # ---- Seven Dormand-Prince stages -------------------------------- #
        k2 = f(t + dt / 5.0,
               y + dt * (k1 / 5.0))

        k3 = f(t + 3.0 * dt / 10.0,
               y + dt * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2))

        k4 = f(t + 4.0 * dt / 5.0,
               y + dt * (44.0 / 45.0 * k1
                         - 56.0 / 15.0 * k2
                         + 32.0 / 9.0 * k3))

        k5 = f(t + 8.0 * dt / 9.0,
               y + dt * (19372.0 / 6561.0 * k1
                         - 25360.0 / 2187.0 * k2
                         + 64448.0 / 6561.0 * k3
                         - 212.0 / 729.0 * k4))

        k6 = f(t + dt,
               y + dt * (9017.0 / 3168.0 * k1
                         - 355.0 / 33.0 * k2
                         + 46732.0 / 5247.0 * k3
                         + 49.0 / 176.0 * k4
                         - 5103.0 / 18656.0 * k5))

        # 4th-order solution (b7 = 0, so k7 is not needed here)
        y_new = y + dt * (35.0 / 384.0 * k1
                          + 500.0 / 1113.0 * k3
                          + 125.0 / 192.0 * k4
                          - 2187.0 / 6784.0 * k5
                          + 11.0 / 84.0 * k6)

        # 7th stage at the 4th-order solution (needed for error and FSAL)
        k7 = f(t + dt, y_new)

        # ---- Error estimate --------------------------------------------- #
        err_vec = dt * (_DP_E[0] * k1
                        + _DP_E[2] * k3
                        + _DP_E[3] * k4
                        + _DP_E[4] * k5
                        + _DP_E[5] * k6
                        + _DP_E[6] * k7)

        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.mean((err_vec / sc) ** 2))

        # ---- Accept / reject ------------------------------------------- #
        if err_norm <= 1.0:
            t += dt
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            k1 = k7              # FSAL: reuse 7th stage as next 1st stage
        # On rejection, k1 stays valid (t, y unchanged)

        # ---- Step-size control ------------------------------------------ #
        if err_norm == 0.0:
            dt = min(dt * 5.0, max_step)
        else:
            factor = 0.9 * err_norm ** (-0.2)
            dt *= min(5.0, max(0.2, factor))

    return np.array(ts), np.array(ys)


# --------------------------------------------------------------------------- #
# High-level simulation driver
# --------------------------------------------------------------------------- #

def simulate_calcium(
    params: CalciumODE,
    ip3_signal: Union[float, Callable[[float], float]],
    t_span: Tuple[float, float],
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the full calcium dynamics model with fixed-step RK4.

    Parameters
    ----------
    params : CalciumODE
        Model parameters.
    ip3_signal : float or callable
        IP3 input.  A float gives a constant concentration; a callable
        ``ip3(t)`` allows time-varying stimulation protocols.
    t_span : (t_start, t_end)
        Simulation interval in seconds.
    dt : float
        Fixed integration time step (s).  A value of 0.01 s is adequate
        for most calcium oscillation frequencies (0.01--0.1 Hz).

    Returns
    -------
    t : ndarray, shape (N,)
        Time points.
    calcium : ndarray, shape (N,)
        Cytosolic calcium concentration at each time point (uM).
    """
    t0, tf = t_span
    ip3_fn: Callable[[float], float] = (
        ip3_signal if callable(ip3_signal) else lambda _t: ip3_signal
    )

    # Initial conditions: resting calcium with the inactivation gate at
    # its equilibrium value for the starting IP3 level.
    ip3_0 = ip3_fn(t0)
    C0 = 0.05
    h0 = params.h_inf(C0, ip3_0)
    y = np.array([C0, h0])

    # Build the RHS closure that evaluates IP3 at each sub-step time
    def rhs(t_loc: float, y_loc: np.ndarray) -> np.ndarray:
        return params.derivatives(t_loc, y_loc, ip3_fn(t_loc))

    n_steps = int(np.ceil((tf - t0) / dt))
    t_arr = np.empty(n_steps + 1)
    c_arr = np.empty(n_steps + 1)
    t_arr[0] = t0
    c_arr[0] = C0

    t_cur = t0
    for i in range(1, n_steps + 1):
        y = rk4_step(rhs, y, t_cur, dt)
        t_cur = t0 + i * dt
        t_arr[i] = t_cur
        c_arr[i] = y[0]

    return t_arr, c_arr


# --------------------------------------------------------------------------- #
# Steady-state finder (Newton's method)
# --------------------------------------------------------------------------- #

def steady_state(
    params: CalciumODE,
    ip3: float,
    C0: float = 0.05,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[float, float]:
    """Find the steady state (C*, h*) for a given constant IP3 level.

    Solves F(C, h) = [dC/dt, dh/dt] = 0 using Newton's method with a
    numerically approximated 2x2 Jacobian (central differences).  A damped
    line search ensures the iterate stays in the physically admissible
    region (C > 0, 0 < h < 1).

    Parameters
    ----------
    params : CalciumODE
    ip3 : float
        Constant IP3 concentration (uM).
    C0 : float
        Initial guess for cytosolic calcium (uM).
    tol : float
        Convergence tolerance on max(|F|).
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    C_ss, h_ss : float
        Steady-state calcium and inactivation variable.

    Raises
    ------
    RuntimeError
        If Newton's method does not converge within *max_iter* iterations.
    """
    h0 = params.h_inf(C0, ip3)
    x = np.array([C0, h0])

    def F(xv: np.ndarray) -> np.ndarray:
        return params.derivatives(0.0, xv, ip3)

    eps_jac = 1e-8

    for _iteration in range(max_iter):
        Fx = F(x)
        if np.max(np.abs(Fx)) < tol:
            return float(x[0]), float(x[1])

        # Numerical Jacobian via central differences
        J = np.empty((2, 2))
        for j in range(2):
            dx = np.zeros(2)
            dx[j] = eps_jac * max(abs(x[j]), 1.0)
            J[:, j] = (F(x + dx) - F(x - dx)) / (2.0 * dx[j])

        # Solve J @ delta = -Fx via 2x2 explicit inverse
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            # Near-singular Jacobian: perturb and retry
            x += np.array([eps_jac, eps_jac])
            continue

        delta = np.array([
            J[1, 1] * (-Fx[0]) - J[0, 1] * (-Fx[1]),
            -J[1, 0] * (-Fx[0]) + J[0, 0] * (-Fx[1]),
        ]) / det

        # Damped step with backtracking line search
        alpha = 1.0
        for _ in range(20):
            x_try = x + alpha * delta
            if x_try[0] > 0.0 and 0.0 < x_try[1] < 1.0:
                if np.max(np.abs(F(x_try))) < np.max(np.abs(Fx)):
                    break
            alpha *= 0.5
        x = x + alpha * delta

    raise RuntimeError(
        f"Newton's method did not converge after {max_iter} iterations "
        f"(residual = {np.max(np.abs(F(x))):.2e})"
    )


# --------------------------------------------------------------------------- #
# Bifurcation analysis
# --------------------------------------------------------------------------- #

def _jacobian(
    params: CalciumODE,
    C: float,
    h: float,
    ip3: float,
) -> np.ndarray:
    """Numerical Jacobian of the calcium ODE at a given state.

    Returns the 2x2 matrix J[i,j] = d(F_i)/d(y_j) evaluated at (C, h)
    using central finite differences with step eps = 1e-8 * max(|y_j|, 1).
    """
    x = np.array([C, h])
    eps = 1e-8
    J = np.empty((2, 2))
    for j in range(2):
        dx = np.zeros(2)
        dx[j] = eps * max(abs(x[j]), 1.0)
        fp = params.derivatives(0.0, x + dx, ip3)
        fm = params.derivatives(0.0, x - dx, ip3)
        J[:, j] = (fp - fm) / (2.0 * dx[j])
    return J


def bifurcation_scan(
    params: CalciumODE,
    ip3_range: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Scan IP3 levels and characterise steady-state stability.

    For each IP3 value, finds the steady state via Newton's method and
    evaluates the Jacobian eigenvalues.  A **Hopf bifurcation** is detected
    where the real part of the dominant eigenvalue crosses zero while the
    eigenvalues form a complex-conjugate pair (nonzero imaginary part).

    At a Hopf bifurcation the stable steady state exchanges stability with
    a limit-cycle oscillation.  This is the biophysical mechanism by which
    astrocytes transition between quiescent and oscillatory calcium
    signalling as IP3 rises in response to synaptic activity.

    Supercritical Hopf:  a small stable limit cycle appears as IP3 crosses
    the bifurcation from below, producing low-amplitude oscillations that
    grow with further IP3 increase.

    Subcritical Hopf (if present):  the unstable limit cycle collides with
    the steady state and oscillations cease abruptly, restoring the elevated
    steady state at high IP3.

    Parameters
    ----------
    params : CalciumODE
    ip3_range : ndarray
        1-D array of IP3 concentrations to scan (uM).  Should be
        monotonically ordered for clean continuation.

    Returns
    -------
    result : dict with keys
        ``ip3``          : ndarray -- scanned IP3 values
        ``C_ss``         : ndarray -- steady-state calcium at each IP3
        ``h_ss``         : ndarray -- steady-state h at each IP3
        ``eigenvalues``  : complex ndarray, shape (N, 2) -- Jacobian eigenvalues
        ``stable``       : bool ndarray -- True where the steady state is
                           asymptotically stable (all eigenvalue real parts < 0)
        ``hopf_indices`` : int ndarray -- indices in *ip3_range* where a Hopf
                           bifurcation is detected (sign change of the dominant
                           eigenvalue real part with nonzero imaginary part)
    """
    ip3_range = np.asarray(ip3_range, dtype=float)
    n = len(ip3_range)
    C_ss = np.empty(n)
    h_ss = np.empty(n)
    eigs = np.empty((n, 2), dtype=complex)
    stable = np.empty(n, dtype=bool)

    # Use continuation: seed each Newton solve with the previous solution
    C_guess = 0.05
    for i, ip3_val in enumerate(ip3_range):
        try:
            C_ss[i], h_ss[i] = steady_state(params, ip3_val, C0=C_guess)
            C_guess = C_ss[i]
        except RuntimeError:
            # If Newton fails, carry forward the last known solution
            C_ss[i] = C_ss[i - 1] if i > 0 else 0.05
            h_ss[i] = h_ss[i - 1] if i > 0 else 0.5
            C_guess = C_ss[i]

        J = _jacobian(params, C_ss[i], h_ss[i], ip3_val)
        eig_vals = np.linalg.eigvals(J)

        # Sort by real part, most positive first
        idx = np.argsort(-eig_vals.real)
        eigs[i] = eig_vals[idx]
        stable[i] = bool(np.all(eig_vals.real < 0))

    # Detect Hopf bifurcations: sign change in the dominant real part
    # accompanied by nonzero imaginary part (complex-conjugate pair).
    max_real = np.max(eigs.real, axis=1)
    hopf_list: List[int] = []
    for i in range(1, n):
        if max_real[i - 1] * max_real[i] < 0:
            # Sign change in the dominant eigenvalue's real part
            if np.any(np.abs(eigs[i].imag) > 1e-10):
                hopf_list.append(i)

    return {
        "ip3": ip3_range,
        "C_ss": C_ss,
        "h_ss": h_ss,
        "eigenvalues": eigs,
        "stable": stable,
        "hopf_indices": np.array(hopf_list, dtype=int),
    }
