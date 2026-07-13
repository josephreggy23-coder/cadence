"""
test_pipeline.py  —  CADENCE pipeline, Module 6
===============================================
Regression tests for the documented SYNTHETIC BENCHMARK, not just code paths.

The point of this file is that a future refactor cannot silently break the
central results. Each test corresponds to a claim the project makes in public:

  1. b1 is recovered significantly > 0 when feedback is intact.
  2. The oracle blocked slope is near zero; causal hard labels retain a small,
     explicitly documented null bias.
  3. b1_intact > b1_blocked in the fixed simulation contrast.
  4. CADENCE reduces synthetic high-state occupancy versus no control.
  5. Structural checks enforce the assumed coupling and hard exposure budget.

Runs with plain `python tests/test_pipeline.py` (no pytest required, keeping
inside the numpy/scipy/pandas/matplotlib/hmmlearn dependency budget), and is also
importable by pytest if you have it.

PREREQUISITE: run the pipeline first, so data/decoded_*.csv and models/ exist:
    python src/run_all.py --quick --skip-existing

A NOTE ON WHICH STATES EACH TEST USES
------------------------------------
Oracle-state checks isolate the hazard estimator. End-to-end checks use inferred
states and exclude traces used to fit the state model. Both views are kept so
state-decoding error cannot be mistaken for hazard-model performance.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from estimate_feedback_law import (  # noqa: E402
    build_transition_dataset, fit_logistic, bootstrap_b1)
from controller import (  # noqa: E402
    OnlineStateEstimator, NoControl, OpenLoop, Cadence, AdaptiveCadence,
    evaluate, TRUE_B1_INTACT, TRUE_B1_BLOCKED, TRUE_B0)

DECODED = {c: os.path.join(ROOT, "data", f"decoded_{c}.csv")
           for c in ("intact", "blocked")}
MODEL = os.path.join(ROOT, "models", "kinetic_model.npz")
ESTIMATES = os.path.join(ROOT, "results", "feedback_estimates.json")

# Kept small so the whole suite runs in well under a minute; large enough that
# the conclusions are stable under reseeding.
N_BOOT = 120
CTRL_TRACES = 6
CTRL_FRAMES = 600   # long enough for the self-calibrating arm to calibrate
SEED = 0

B0_SHIFT = -2.5
B0_CHALLENGE, B1_REFERENCE = TRUE_B0 + B0_SHIFT, 0.9


def _require_artifacts():
    missing = [
        p for p in list(DECODED.values()) + [MODEL, ESTIMATES]
        if not os.path.exists(p)
    ]
    if missing:
        raise SystemExit(
            "Missing pipeline artifacts:\n  " + "\n  ".join(missing) +
            "\n\nRun the pipeline first:\n"
            "  python src/run_all.py --quick --skip-existing")


def _fit_b1(condition, state_col, decay=0.90):
    df = pd.read_csv(DECODED[condition])
    L, y, _ = build_transition_dataset(df, state_col, decay)
    beta, _, _ = fit_logistic(L, y)
    bootstrap_seed = SEED + (0 if condition == "intact" else 1_000)
    boots = bootstrap_b1(
        df, state_col, decay, n_boot=N_BOOT, seed=bootstrap_seed
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return beta[1], lo, hi, boots


def _learned_law():
    with open(ESTIMATES, encoding="utf-8") as handle:
        values = json.load(handle)["causal"]["intact"]
    return float(values["b0"]), float(values["b1"])


def _learned_decay():
    with open(ESTIMATES, encoding="utf-8") as handle:
        return float(json.load(handle)["shared_decay"])


def _adaptive_from_estimates():
    b0, b1 = _learned_law()
    return AdaptiveCadence(b0, b1)


def _fixed_from_estimates():
    b0, b1 = _learned_law()
    return Cadence(b0, b1)


# --------------------------------------------------------------------------- #
# CLAIM 1: feedback is recovered when it is present
# --------------------------------------------------------------------------- #
def test_b1_intact_significantly_positive():
    """
    Asserted on TRUE states: this defends the ESTIMATOR, showing it recovers a
    known positive b1. (End-to-end attenuation is covered by test 3.)
    """
    b1, lo, hi, _ = _fit_b1("intact", "true_state")
    assert b1 > 0, f"intact b1 should be positive, got {b1:+.4f}"
    assert lo > 0, (f"intact b1 CI should exclude zero, got [{lo:+.4f}, {hi:+.4f}]")
    # ground truth is +0.9; allow generous slack for load-scale differences
    assert 0.5 < b1 < 1.8, f"intact b1 {b1:+.4f} implausibly far from truth +0.9"


# --------------------------------------------------------------------------- #
# CLAIM 2: no feedback is "recovered" when it has been removed
# --------------------------------------------------------------------------- #
def test_b1_blocked_approximately_zero():
    """
    On TRUE synthetic states the estimator must not invent a slope where the
    generator encoded approximately zero; its CI has to contain zero.
    """
    b1, lo, hi, _ = _fit_b1("blocked", "true_state")
    assert lo <= 0 <= hi, (
        f"blocked b1 CI [{lo:+.4f}, {hi:+.4f}] should contain zero (truth +0.02)")
    assert abs(b1) < 0.25, f"blocked b1 {b1:+.4f} should be near zero"


def test_blocked_causal_b1_bias_is_small_and_documented():
    """
    Causal hard-state decoding produces a small negative slope in a near-null
    arm. Pin its size and direction so it cannot be mistaken for feedback.
    """
    b1, lo, hi, _ = _fit_b1("blocked", "causal_state")
    assert -0.10 < b1 < 0, f"blocked causal b1 {b1:+.4f} bias is too large"
    assert hi < 0, (
        f"documented causal null-bias interval changed: [{lo:+.4f}, {hi:+.4f}]")


def test_feedback_result_matches_decoded_fit():
    """Prevent the machine-readable controller input from becoming stale."""
    b1, _, _, _ = _fit_b1("intact", "causal_state")
    _, saved_b1 = _learned_law()
    assert np.isclose(saved_b1, b1, atol=1e-8), (
        f"saved b1 {saved_b1:+.6f} does not match decoded-data fit {b1:+.6f}")


# --------------------------------------------------------------------------- #
# CLAIM 3: the falsifiable comparison, end to end
# --------------------------------------------------------------------------- #
def test_b1_intact_greater_than_blocked_end_to_end():
    """
    Asserted on INFERRED states, because this is the claim the actual pipeline
    makes. The bootstrap difference must exclude zero.
    """
    _, _, _, boots_i = _fit_b1("intact", "causal_state")
    _, _, _, boots_b = _fit_b1("blocked", "causal_state")
    n = min(len(boots_i), len(boots_b))
    diff = boots_i[:n] - boots_b[:n]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    assert lo > 0, (
        f"b1_intact - b1_blocked CI [{lo:+.4f}, {hi:+.4f}] must exclude zero")


# --------------------------------------------------------------------------- #
# Controller claims
# --------------------------------------------------------------------------- #
def _run_controller(b1_true, policy_factory):
    est = OnlineStateEstimator(MODEL)
    return evaluate(policy_factory, est, CTRL_TRACES, SEED,
                    n_frames=CTRL_FRAMES, b1_true=b1_true, b0_shift=B0_SHIFT,
                    est_load_decay=_learned_decay())


def test_cadence_reduces_high_state_occupancy_vs_no_control():
    """Self-calibrating CADENCE is the deliverable, so it is what gets tested."""
    none = _run_controller(TRUE_B1_INTACT, lambda: NoControl())
    cad = _run_controller(TRUE_B1_INTACT, _adaptive_from_estimates)
    assert cad["high_state_frac"] < none["high_state_frac"], (
        f"CADENCE {cad['high_state_frac']*100:.1f}% should beat no-control "
        f"{none['high_state_frac']*100:.1f}%")


def test_fixed_law_under_treats_the_shifted_plant():
    """
    PINS A SCIENTIFIC POINT, not a bug.

    A fixed law learned from the unshifted simulation makes the controller stay
    silent on a shifted synthetic plant. This is why the adaptive simulation
    includes online b0 calibration; it is not biological transfer evidence.
    """
    none = _run_controller(TRUE_B1_INTACT, lambda: NoControl())
    cad = _run_controller(TRUE_B1_INTACT, _fixed_from_estimates)
    assert cad["high_state_frac"] > 0.9 * none["high_state_frac"], (
        "the fixed law now changes the shifted plant substantially; revisit "
        "the calibration scenario")


def test_causal_b1_is_positive_but_attenuated():
    """
    Online filtering is stricter than offline smoothing. The causal slope should
    preserve the encoded positive direction while transparently showing some
    attenuation relative to the simulator's true +0.9.
    """
    b1, lo, hi, _ = _fit_b1("intact", "causal_state")
    assert lo > 0, f"intact end-to-end b1 CI [{lo:+.3f}, {hi:+.3f}] must exclude zero"
    assert 0.4 < b1 < 0.9, (
        f"causal b1={b1:+.3f} should remain positive but attenuated vs +0.9")


def test_hard_budget_prevents_runaway_exposure():
    """
    The causal-law audit exposed that the heuristic futility check can fail and
    spend more than continuous open loop. The adaptive policy therefore has an
    explicit hard cumulative budget. This is a software constraint, not evidence
    that the policy can identify biological non-response.
    """
    ol = _run_controller(TRUE_B1_BLOCKED, lambda: OpenLoop(1.0))
    cad = _run_controller(TRUE_B1_BLOCKED, _adaptive_from_estimates)
    assert cad["cost"] <= 180.0 + 1e-9, (
        f"adaptive cost {cad['cost']:.1f} exceeded its hard 180-unit budget")
    assert cad["cost"] <= 0.30 * ol["cost"] + 1e-9


def test_plant_parameter_reference_improves_high_state_occupancy():
    """A reference policy given the shifted plant parameters should improve occupancy."""
    ol = _run_controller(TRUE_B1_INTACT, lambda: OpenLoop(1.0))
    cad = _run_controller(
        TRUE_B1_INTACT, lambda: Cadence(B0_CHALLENGE, B1_REFERENCE)
    )
    none = _run_controller(TRUE_B1_INTACT, lambda: NoControl())
    assert cad["high_state_frac"] < none["high_state_frac"]
    assert cad["high_state_frac"] <= ol["high_state_frac"]


# --------------------------------------------------------------------------- #
# STRUCTURAL CHECK: BLOCKADE DISABLES CONTROL BY CONSTRUCTION
# --------------------------------------------------------------------------- #
def test_structural_blockade_disables_cadence():
    """
    The simulator multiplies intervention by b1, so b1 near zero must disable
    control. This validates implementation consistency, not a biological claim.
    """
    none = _run_controller(TRUE_B1_BLOCKED, lambda: NoControl())
    cad = _run_controller(TRUE_B1_BLOCKED, _adaptive_from_estimates)
    # "fails to restore" = does not meaningfully beat doing nothing
    assert cad["high_state_frac"] >= 0.90 * none["high_state_frac"], (
        f"STRUCTURAL CHECK FAILED: CADENCE restored under b1 blockade "
        f"({none['high_state_frac']*100:.1f}% -> "
        f"{cad['high_state_frac']*100:.1f}%). The controller is not acting "
        f"through the endogenous law.")


def test_structural_blockade_disables_open_loop_coupling():
    """
    Fixed-dose input uses the same assumed b1 coupling and must also be disabled.
    """
    none = _run_controller(TRUE_B1_BLOCKED, lambda: NoControl())
    ol = _run_controller(TRUE_B1_BLOCKED, lambda: OpenLoop(1.0))
    assert ol["cost"] > 0, "open-loop should be spending stimulation"
    assert ol["high_state_frac"] >= 0.90 * none["high_state_frac"], (
        "open-loop unexpectedly restored under the b1-coupled plant")


# --------------------------------------------------------------------------- #
# Minimal runner so no test framework is required
# --------------------------------------------------------------------------- #
def _main():
    _require_artifacts()
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} pipeline tests...\n")
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            failures.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
            failures.append(name)

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All documented synthetic benchmark checks hold.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
