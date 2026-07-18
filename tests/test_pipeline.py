"""
test_pipeline.py  —  CADENCE pipeline, Module 6
===============================================
Regression tests for the SCIENTIFIC CLAIMS, not just the code paths.

The point of this file is that a future refactor cannot silently break the
central results. Each test corresponds to a claim the project makes in public:

  1. b1 is recovered significantly > 0 when feedback is intact.
  2. b1 is ~ 0 when feedback is blocked (the pharmacological control).
  3. b1_intact > b1_blocked - the falsifiable comparison.
  4. CADENCE reduces pathological time versus no control.
  5. CADENCE costs less than open-loop stimulation.
  6. KILL-SHOT: under blocked feedback, CADENCE FAILS to restore.

Runs with plain `python tests/test_pipeline.py` (no pytest required, keeping
inside the numpy/scipy/pandas/matplotlib/hmmlearn dependency budget), and is also
importable by pytest if you have it.

PREREQUISITE: run the pipeline first, so data/decoded_*.csv and models/ exist:
    python src/run_all.py --quick --skip-existing

A NOTE ON WHICH STATES EACH TEST USES — please read before "fixing" a test
--------------------------------------------------------------------------
Module 3 established that state-estimation error attenuates b1 and, in the
blocked condition, pushes the INFERRED estimate significantly NEGATIVE when the
truth is ~0. That is a known, documented artifact, not a bug in these tests.

So the tests are deliberately split by which claim is being defended:
  - Claims about the ESTIMATOR being correct (tests 1, 2) are asserted on TRUE
    states, where the estimator is known to recover the ground truth.
  - Claims about the END-TO-END PIPELINE (test 3) are asserted on INFERRED
    states, because that is what the pipeline actually delivers.
  - test_2b explicitly PINS the known artifact, so that if a future improvement
    to state recovery fixes it, this test fails loudly and tells you to update
    the README rather than letting a real advance go unnoticed.
"""

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

# Kept small so the whole suite runs in well under a minute; large enough that
# the conclusions are stable under reseeding.
N_BOOT = 120
CTRL_TRACES = 6
CTRL_FRAMES = 600   # long enough for the self-calibrating arm to calibrate
SEED = 0

# Laws recovered by Module 3 (see README). Used to configure the controller.
B0_LEARNED, B1_LEARNED = -2.703, 0.832
B0_SHIFT = -2.5
B0_DISEASE, B1_DISEASE = TRUE_B0 + B0_SHIFT, 0.9


def _require_artifacts():
    missing = [p for p in list(DECODED.values()) + [MODEL] if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing pipeline artifacts:\n  " + "\n  ".join(missing) +
            "\n\nRun the pipeline first:\n"
            "  python src/run_all.py --quick --skip-existing")


def _fit_b1(condition, state_col, decay=0.90):
    df = pd.read_csv(DECODED[condition])
    L, y, _ = build_transition_dataset(df, state_col, decay)
    beta, _, _ = fit_logistic(L, y)
    boots = bootstrap_b1(df, state_col, decay, n_boot=N_BOOT, seed=SEED)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return beta[1], lo, hi, boots


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
    The pharmacological control. On TRUE states the estimator must NOT invent a
    feedback signal where none exists - its CI has to contain zero.
    """
    b1, lo, hi, _ = _fit_b1("blocked", "true_state")
    assert lo <= 0 <= hi, (
        f"blocked b1 CI [{lo:+.4f}, {hi:+.4f}] should contain zero (truth +0.02)")
    assert abs(b1) < 0.25, f"blocked b1 {b1:+.4f} should be near zero"


def test_blocked_inferred_b1_artifact_is_still_present():
    """
    PINS A KNOWN ARTIFACT (see README, Step 3).

    From INFERRED states the blocked condition yields a significantly NEGATIVE
    b1 where the truth is ~0, because spurious refractory calls cluster at the
    start of long high runs. This is documented as a limitation.

    If this test FAILS, that is potentially GOOD NEWS: state recovery may have
    improved enough to remove the artifact. Do not silently delete this test -
    verify the improvement, then update the README's limitations section.
    """
    b1, lo, hi, _ = _fit_b1("blocked", "inferred_state")
    assert b1 < 0 and hi < 0, (
        "The documented negative-b1 artifact is GONE (blocked inferred "
        f"b1={b1:+.4f}, CI [{lo:+.4f}, {hi:+.4f}]). If state recovery improved, "
        "update README limitations and this test.")


# --------------------------------------------------------------------------- #
# CLAIM 3: the falsifiable comparison, end to end
# --------------------------------------------------------------------------- #
def test_b1_intact_greater_than_blocked_end_to_end():
    """
    Asserted on INFERRED states, because this is the claim the actual pipeline
    makes. The bootstrap difference must exclude zero.
    """
    _, _, _, boots_i = _fit_b1("intact", "inferred_state")
    _, _, _, boots_b = _fit_b1("blocked", "inferred_state")
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
                    n_frames=CTRL_FRAMES, b1_true=b1_true, b0_shift=B0_SHIFT)


def test_cadence_reduces_pathological_time_vs_no_control():
    """Self-calibrating CADENCE is the deliverable, so it is what gets tested."""
    none = _run_controller(TRUE_B1_INTACT, lambda: NoControl())
    cad = _run_controller(TRUE_B1_INTACT,
                          lambda: AdaptiveCadence(B0_LEARNED, B1_LEARNED))
    assert cad["pathological_frac"] < none["pathological_frac"], (
        f"CADENCE {cad['pathological_frac']*100:.1f}% should beat no-control "
        f"{none['pathological_frac']*100:.1f}%")


def test_cadence_costs_less_than_open_loop():
    ol = _run_controller(TRUE_B1_INTACT, lambda: OpenLoop(1.0))
    cad = _run_controller(TRUE_B1_INTACT,
                          lambda: AdaptiveCadence(B0_LEARNED, B1_LEARNED))
    assert cad["cost"] < 0.5 * ol["cost"], (
        f"CADENCE cost {cad['cost']:.0f} should be well under open-loop "
        f"{ol['cost']:.0f}")


def test_healthy_law_alone_under_treats_the_diseased_cell():
    """
    PINS A SCIENTIFIC POINT, not a bug.

    An ACCURATE law learned from healthy cells makes the controller stay silent
    on a diseased cell - it correctly believes healthy cells self-suppress and
    wrongly concludes this one will too. This is precisely why online b0
    calibration exists. If this ever starts restoring, the disease model or the
    law has changed and the calibration argument in the README needs revisiting.
    """
    none = _run_controller(TRUE_B1_INTACT, lambda: NoControl())
    cad = _run_controller(TRUE_B1_INTACT, lambda: Cadence(B0_LEARNED, B1_LEARNED))
    assert cad["pathological_frac"] > 0.9 * none["pathological_frac"], (
        "the uncalibrated healthy law now restores the diseased cell; the "
        "motivation for online b0 calibration needs re-examining")


def test_end_to_end_b1_recovers_ground_truth():
    """
    The kinetic estimator removed the ~5x attenuation: the END-TO-END b1 (from
    inferred states) should now bracket the simulator's true +0.9.
    """
    b1, lo, hi, _ = _fit_b1("intact", "inferred_state")
    assert lo > 0, f"intact end-to-end b1 CI [{lo:+.3f}, {hi:+.3f}] must exclude zero"
    assert lo <= 0.9 <= hi or abs(b1 - 0.9) < 0.25, (
        f"end-to-end b1={b1:+.3f} CI [{lo:+.3f}, {hi:+.3f}] no longer recovers "
        f"the ground truth +0.9")


def test_safety_interlock_stands_down_when_pathway_is_dead():
    """
    Under blockade the self-calibrating controller observes a cell that never
    switches off, infers a very low b0, and would otherwise escalate the dose
    forever - it spent MORE than continuous open-loop (829 vs 600) before the
    futility interlock was added. It must now stand down instead.
    """
    ol = _run_controller(TRUE_B1_BLOCKED, lambda: OpenLoop(1.0))
    cad = _run_controller(TRUE_B1_BLOCKED,
                          lambda: AdaptiveCadence(B0_LEARNED, B1_LEARNED))
    assert cad["cost"] < ol["cost"], (
        f"SAFETY: self-calibrating CADENCE spent {cad['cost']:.0f} under blockade "
        f"vs open-loop {ol['cost']:.0f} - the futility interlock is not firing")


def test_disease_calibrated_cadence_is_much_cheaper_than_open_loop():
    """The headline efficiency claim: comparable restoration at a fraction of the cost."""
    ol = _run_controller(TRUE_B1_INTACT, lambda: OpenLoop(1.0))
    cad = _run_controller(TRUE_B1_INTACT, lambda: Cadence(B0_DISEASE, B1_DISEASE))
    assert cad["cost"] < 0.5 * ol["cost"], (
        f"disease-calibrated CADENCE cost {cad['cost']:.0f} should be well under "
        f"half of open-loop {ol['cost']:.0f}")
    # and it must still actually help
    none = _run_controller(TRUE_B1_INTACT, lambda: NoControl())
    assert cad["pathological_frac"] < none["pathological_frac"]


# --------------------------------------------------------------------------- #
# CLAIM 6: THE KILL-SHOT
# --------------------------------------------------------------------------- #
def test_killshot_cadence_fails_to_restore_when_feedback_blocked():
    """
    The most important test in the file.

    With the feedback pathway blocked, CADENCE must NOT restore rhythm, because
    its intervention acts THROUGH that pathway. If this test ever passes in the
    sense of "CADENCE restored the cell", the controller is brute-forcing the
    system and the project's central claim is invalid.
    """
    none = _run_controller(TRUE_B1_BLOCKED, lambda: NoControl())
    cad = _run_controller(TRUE_B1_BLOCKED,
                          lambda: AdaptiveCadence(B0_LEARNED, B1_LEARNED))
    # "fails to restore" = does not meaningfully beat doing nothing
    assert cad["pathological_frac"] >= 0.90 * none["pathological_frac"], (
        f"KILL-SHOT VIOLATED: CADENCE restored the cell under blocked feedback "
        f"({none['pathological_frac']*100:.1f}% -> "
        f"{cad['pathological_frac']*100:.1f}%). The controller is not acting "
        f"through the endogenous law.")


def test_killshot_open_loop_also_fails_despite_full_cost():
    """
    The sharpest form of the kill-shot: open-loop spends the maximum and still
    achieves nothing, because the pathway its stimulus acts through is gone.
    """
    none = _run_controller(TRUE_B1_BLOCKED, lambda: NoControl())
    ol = _run_controller(TRUE_B1_BLOCKED, lambda: OpenLoop(1.0))
    assert ol["cost"] > 0, "open-loop should be spending stimulation"
    assert ol["pathological_frac"] >= 0.90 * none["pathological_frac"], (
        "open-loop restored the cell under blocked feedback, which would mean "
        "the stimulus bypasses the feedback pathway")


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
    print("All scientific claims hold.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
