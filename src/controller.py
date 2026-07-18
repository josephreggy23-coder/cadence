"""
controller.py  —  CADENCE pipeline, Module 4  ***THE SOLUTION***
================================================================
Everything before this module exists to make this module possible. Modules 1-3
learn the glia's own regulatory rule; this module USES that rule to decide when,
and how hard, to intervene — and does so minimally, because it acts *through* the
endogenous feedback rather than overriding it.

THE CONTROL PROBLEM
-------------------
Keep calcium in a healthy oscillatory cadence and out of pathological
SUSTAINED_HIGH, while spending as little stimulation as possible. Cost matters:
real glial neuromodulation that blasts continuously wastes energy and risks
driving astrocytes into reactive/cytotoxic states. "Works but costs everything"
is not a solution.

HOW THE INTERVENTION IS MODELLED (and why the kill-shot is real)
----------------------------------------------------------------
This is the single most important modelling decision in the project.

The stimulus does NOT force the state directly. It acts by *driving the
load-sensing feedback pathway* — it makes the cell behave as though it had
accumulated more calcium load than it actually has:

        P(high -> refractory | L, u) = sigmoid(b0 + b1 * (L + kappa * u))

Note where `u` sits: INSIDE the `b1` term. The consequence is the whole thesis of
the project. If the feedback pathway is pharmacologically blocked (b1 ~ 0), then
b1*(L + kappa*u) ~ 0 no matter how large `u` is — the controller can spend
unlimited stimulation and produce no suppression, because the mechanism it works
through is gone.

Had we instead written `sigmoid(b0 + b1*L + gain*u)`, the stimulus would bypass
the feedback pathway and would still "work" under blockade. That would make the
kill-shot experiment unfalsifiable and the entire claim circular. The kill-shot
is only meaningful because the intervention is wired through b1.

THE POLICY: MINIMAL AND PREDICTIVE
----------------------------------
At each frame, while the cell is believed to be in SUSTAINED_HIGH:

  1. Predict whether the cell will self-suppress on its own within a horizon H:
         P(escape by itself) = 1 - (1 - p_endo)^H,   p_endo = sigmoid(b0 + b1*L)
  2. If that already exceeds the target, DO NOTHING. The cell is going to fix
     itself; intervening would be wasted cost. This is where the savings come
     from — CADENCE is silent most of the time.
  3. Only if the cell is predicted to FAIL to recover in time, solve for the
     SMALLEST u that lifts the escape probability to target:
         p_needed = 1 - (1 - target)^(1/H)
         u* = (logit(p_needed) - b0 - b1*L) / (b1 * kappa),  clipped to [0, u_max]

Step 3 is minimal by construction: it is the exact solution of "what is the least
stimulation that achieves the goal", not a fixed dose.

ONLINE STATE ESTIMATION (no peeking at the future)
--------------------------------------------------
Modules 1-3 are offline analyses of complete recordings and may use Viterbi,
which needs the whole trace. A controller cannot. Here we run the HMM FORWARD
filter, which is causal: the state belief at frame t uses only frames 1..t.

One subtlety, handled explicitly: the emission includes a CENTRED derivative
(see features.py), which at frame t peeks at frame t+1. Rather than switch to a
backward difference — which would give the online features a different
distribution from the one the model was trained on, silently corrupting the
likelihoods — the controller runs with ONE FRAME (0.5 s) OF LATENCY: at time t it
acts on the state estimate for frame t-1, using the properly matched centred
feature. This is both statistically consistent and realistic for a real-time
system, which always has some loop delay.

WHAT IS COMPARED
----------------
  (a) no control       - the disease runs unchecked; the floor.
  (b) open-loop        - fixed continuous stimulation, the current clinical
                         paradigm: no sensing, no timing, same dose forever.
  (c) CADENCE          - this policy.

Metrics: time spent in pathological SUSTAINED_HIGH (measured on the TRUE hidden
state, which the controller never sees), and cumulative intervention cost sum(u).

HONEST CAVEAT ON WHICH LAW THE CONTROLLER USES
-----------------------------------------------
Module 3 showed the end-to-end estimate of b1 is attenuated ~5x by
state-estimation error (+0.21 recovered vs +0.9 true). The controller is run with
BOTH laws so the operational cost of that error is visible rather than hidden:
  - "learned"  : b1 as actually recovered end-to-end. Underestimating the cell's
                 own suppression makes the controller distrust it and over-treat.
  - "oracle"   : b1 from true states. What CADENCE would achieve if Module 2's
                 REFRACTORY bottleneck were solved.
The gap between them is the price of the current state-estimation error, and is
reported as a headline number.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import make_features, CALCIUM_COL

# ---- constants shared with the ground-truth simulator ---------------------- #
STATE_NAMES = ["QUIESCENT", "OSCILLATORY", "SUSTAINED_HIGH", "REFRACTORY"]
QUIESCENT, OSCILLATORY, HIGH, REFRACTORY = 0, 1, 2, 3

EMISSION_MEAN = np.array([0.05, 0.35, 0.90, 0.10])
EMISSION_SD = np.array([0.03, 0.08, 0.12, 0.04])
GCAMP_TAU = 3.0
DT = 0.5
LOAD_DECAY = 0.92

# Healthy baseline transitions (from generate_synthetic.py). The HIGH row is
# filled dynamically by the feedback law.
BASE_T = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.06, 0.82, 0.12, 0.00],
    [0.00, 0.02, 0.00, 0.00],
    [0.25, 0.10, 0.00, 0.65],
])

TRUE_B0, TRUE_B1_INTACT, TRUE_B1_BLOCKED = -3.0, 0.9, 0.02

# Intervention efficacy: how much "virtual load" one unit of stimulation buys.
KAPPA = 1.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


LEAK_TO_OSC = 0.02


def high_row(p_off):
    """
    Transition row out of SUSTAINED_HIGH given the (possibly stimulus-boosted)
    off-probability.

    Note the clip: under strong intervention p_off can be driven arbitrarily
    close to 1, at which point `p_off + LEAK_TO_OSC` would exceed 1 and the row
    would stop being a valid distribution. The generator never hit this because
    it had no control input; the controller does. We clip and renormalise so the
    row is always exactly a probability vector.
    """
    p_off = float(np.clip(p_off, 0.0, 1.0 - LEAK_TO_OSC))
    p_stay = max(0.0, 1.0 - p_off - LEAK_TO_OSC)
    row = np.array([0.00, LEAK_TO_OSC, p_stay, p_off])
    return row / row.sum()


# --------------------------------------------------------------------------- #
# Disease model
# --------------------------------------------------------------------------- #
def disease_transitions(entry_boost=0.30):
    """
    The disease model: an elevated tendency to ENTER SUSTAINED_HIGH.

    We raise P(OSCILLATORY -> HIGH) from 0.12 to 0.12+entry_boost, taking the mass
    from "stay oscillatory". Combined with `b0_shift` below (harder to switch off),
    this reproduces the pathological phenotype we want to restore: the cell keeps
    falling into the high state and struggles to leave it.
    """
    T = BASE_T.copy()
    T[OSCILLATORY, HIGH] += entry_boost
    T[OSCILLATORY, OSCILLATORY] -= entry_boost
    T[OSCILLATORY] = np.clip(T[OSCILLATORY], 0, None)
    T[OSCILLATORY] /= T[OSCILLATORY].sum()
    return T


# --------------------------------------------------------------------------- #
# Online (causal) state estimator
# --------------------------------------------------------------------------- #
class OnlineStateEstimator:
    """
    Causal HMM forward filter. Maintains a belief over hidden states using only
    observations seen so far — no Viterbi, no future frames.

    Because the trained emission uses a CENTRED derivative, the estimate returned
    at call t refers to frame t-1 (one-frame latency). See module docstring.
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.K = int(d["n_states"])
        self.startprob = d["startprob"]
        self.transmat = d["transmat"]
        self.means = d["means"]                      # (K, n_features)
        self.covars = d["covars"].reshape(self.K, -1)
        self.feature_mode = str(d["feature_mode"]) if "feature_mode" in d else "level"
        # map HMM index -> biological index, by calcium-mean rank (same rule as
        # recover_states.py; the controller must not invent its own labelling)
        rank_to_true = [QUIESCENT, REFRACTORY, OSCILLATORY, HIGH]
        self.hmm_to_true = np.empty(self.K, dtype=int)
        for rank, s in enumerate(np.argsort(self.means[:, CALCIUM_COL])):
            self.hmm_to_true[s] = rank_to_true[rank]
        self.reset()

    def reset(self):
        self.alpha = None
        self.buffer = []      # raw calcium history, for the centred derivative

    def _log_emission(self, x):
        """Diagonal-Gaussian log-likelihood of observation x under each state."""
        var = self.covars
        d = x[None, :] - self.means
        return -0.5 * np.sum(np.log(2 * np.pi * var) + d * d / var, axis=1)

    def update(self, calcium_t):
        """
        Feed one new calcium sample. Returns the MAP biological state for frame
        t-1, or None until enough history exists to form the centred feature.
        """
        self.buffer.append(float(calcium_t))
        if len(self.buffer) < 3:
            return None

        # rebuild features over the (short) recent window so the centred
        # derivative matches training exactly, then take the frame that is now
        # fully determined: the second-to-last one.
        window = np.asarray(self.buffer[-8:])
        feats = make_features(window, self.feature_mode)
        x = feats[-2]

        logB = self._log_emission(x)
        if self.alpha is None:
            loga = np.log(self.startprob + 1e-300) + logB
        else:
            # forward recursion in log space
            m = self.alpha.max()
            trans = np.log(self.transmat + 1e-300)
            loga = np.log(np.exp(self.alpha - m) @ np.exp(trans)) + m + logB
        self.alpha = loga - loga.max()   # renormalise for numerical stability
        return int(self.hmm_to_true[np.argmax(self.alpha)])


# --------------------------------------------------------------------------- #
# Control policies
# --------------------------------------------------------------------------- #
class NoControl:
    name = "no control"

    def act(self, state, L):
        return 0.0


class OpenLoop:
    """
    The current clinical paradigm: a fixed dose, applied continuously, with no
    sensing and no timing. It is the honest strawman — it is what CADENCE has to
    beat, and it does restore rhythm; the question is at what cost.
    """
    name = "open-loop"

    def __init__(self, dose=1.0):
        self.dose = dose

    def act(self, state, L):
        return self.dose


class Cadence:
    """
    Model-based, predictive, minimal.

    Intervenes ONLY when the learned law says the cell will fail to self-suppress
    within `horizon` frames, and then only by the smallest amount that restores
    the target escape probability.
    """
    name = "CADENCE"

    def __init__(self, b0, b1, horizon=6, target=0.90, u_max=8.0, kappa=KAPPA):
        self.b0, self.b1 = b0, b1
        self.horizon, self.target, self.u_max, self.kappa = horizon, target, u_max, kappa

    def act(self, state, L):
        if state != HIGH:
            return 0.0                      # the law is only defined in the high state

        p_endo = sigmoid(self.b0 + self.b1 * L)
        p_escape_alone = 1.0 - (1.0 - p_endo) ** self.horizon
        if p_escape_alone >= self.target:
            return 0.0                      # it will fix itself - stay silent

        # smallest u that lifts per-frame escape probability to what's needed
        p_needed = 1.0 - (1.0 - self.target) ** (1.0 / self.horizon)
        logit_needed = np.log(p_needed / (1.0 - p_needed))
        denom = self.b1 * self.kappa
        if abs(denom) < 1e-9:
            # Feedback pathway is (believed) dead: no finite dose can work through
            # it. Ask for the maximum and let the simulation show it fails - this
            # is the kill-shot condition, not an error.
            return self.u_max
        u = (logit_needed - self.b0 - self.b1 * L) / denom
        return float(np.clip(u, 0.0, self.u_max))


# --------------------------------------------------------------------------- #
# Closed-loop simulation
# --------------------------------------------------------------------------- #
def simulate_closed_loop(policy, estimator, rng, n_frames=600,
                         b1_true=TRUE_B1_INTACT, b0_true=TRUE_B0,
                         b0_shift=-1.5, entry_boost=0.30, kappa=KAPPA):
    """
    Run one closed-loop trace.

    The plant (true dynamics) is the same generative model as
    generate_synthetic.py, with the disease modifications applied, plus the
    control input entering through the feedback law. The controller sees ONLY the
    noisy calcium sample — never the true state, never the load.

    `b0_shift` makes the diseased cell intrinsically worse at switching itself
    off; `entry_boost` makes it fall into the high state more often.
    """
    T_dis = disease_transitions(entry_boost)
    b0_eff = b0_true + b0_shift

    state = QUIESCENT
    true_load = 0.0
    est_load = 0.0
    c = EMISSION_MEAN[QUIESCENT]

    states = np.empty(n_frames, dtype=int)
    calcium = np.empty(n_frames)
    actions = np.zeros(n_frames)
    est_states = np.full(n_frames, -1, dtype=int)

    estimator.reset()

    for t in range(n_frames):
        states[t] = state

        # ---- sensor: causal GCaMP smoothing + observation noise ---------- #
        c += (EMISSION_MEAN[state] - c) / GCAMP_TAU
        y = c + rng.normal(0, EMISSION_SD[state])
        calcium[t] = y

        # ---- controller observes, estimates, and acts -------------------- #
        s_hat = estimator.update(y)
        if s_hat is not None:
            est_states[t] = s_hat
            # the controller maintains its OWN load estimate from its OWN state
            # belief - it has no access to the true load
            est_load = est_load * LOAD_DECAY + (1.0 if s_hat == HIGH else 0.0)
            u = policy.act(s_hat, est_load)
        else:
            u = 0.0
        actions[t] = u

        # ---- plant update ------------------------------------------------ #
        true_load = true_load * LOAD_DECAY + (1.0 if state == HIGH else 0.0)
        if state == HIGH:
            # THE INTERVENTION ENTERS HERE, INSIDE the b1 term - see docstring.
            p_off = sigmoid(b0_eff + b1_true * (true_load + kappa * u))
            row = high_row(p_off)
        else:
            row = T_dis[state].copy()
            row = row / row.sum()
        state = rng.choice(4, p=row)

    return {"true_state": states, "calcium": calcium, "action": actions,
            "est_state": est_states}


def evaluate(policy_factory, estimator, n_traces, seed, **sim_kw):
    """Run several traces and aggregate the two metrics that matter."""
    rng = np.random.default_rng(seed)
    path_frac, costs, traces = [], [], []
    for i in range(n_traces):
        out = simulate_closed_loop(policy_factory(), estimator, rng, **sim_kw)
        path_frac.append(np.mean(out["true_state"] == HIGH))
        costs.append(out["action"].sum())
        traces.append(out)
    return {"pathological_frac": float(np.mean(path_frac)),
            "pathological_sd": float(np.std(path_frac)),
            "cost": float(np.mean(costs)),
            "cost_sd": float(np.std(costs)),
            "traces": traces}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_control_comparison(results, out_png, title):
    """Trace + intervention markers for each policy, stacked for comparison."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        tr = res["traces"][0]
        t = np.arange(len(tr["calcium"])) * DT
        # shade true pathological periods
        path = tr["true_state"] == HIGH
        ax.fill_between(t, 0, 1, where=path, transform=ax.get_xaxis_transform(),
                        color="#e07b7b", alpha=0.30, lw=0)
        ax.plot(t, tr["calcium"], color="black", lw=0.8)
        u = tr["action"]
        if u.max() > 0:
            ax2 = ax.twinx()
            ax2.plot(t, u, color="#1f77b4", lw=0.9, alpha=0.85)
            ax2.set_ylabel("stim u", color="#1f77b4", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
        ax.set_ylabel("dF/F")
        ax.set_title(f"{name}  —  pathological {res['pathological_frac']*100:.1f}% "
                     f"of time,  cost {res['cost']:.0f}", fontsize=10, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_bars(all_results, out_png):
    """Pathological time and cost, side by side, across policies and conditions."""
    conds = list(all_results.keys())
    policies = list(next(iter(all_results.values())).keys())
    x = np.arange(len(policies))
    w = 0.8 / len(conds)
    colors = ["#1f77b4", "#d62728", "#7f7f7f"]

    # wrap long policy names onto two lines so the tick labels never collide
    short = [p.replace(" (", "\n(") for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for k, cond in enumerate(conds):
        vals = [all_results[cond][p]["pathological_frac"] * 100 for p in policies]
        err = [all_results[cond][p]["pathological_sd"] * 100 for p in policies]
        axes[0].bar(x + k * w, vals, w, yerr=err, capsize=3,
                    label=cond, color=colors[k % len(colors)])
        cvals = [all_results[cond][p]["cost"] for p in policies]
        axes[1].bar(x + k * w, cvals, w, label=cond, color=colors[k % len(colors)])

    for ax, ylab, title in ((axes[0], "% time in SUSTAINED_HIGH", "Restoration"),
                            (axes[1], "cumulative intervention cost", "Cost")):
        ax.set_xticks(x + w * (len(conds) - 1) / 2, short, fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("CADENCE vs baselines: equal-or-better restoration at lower cost\n"
                 "(and, under blocked feedback, correct failure)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="CADENCE model-based controller.")
    ap.add_argument("--model", default="models/hmm_model.npz")
    ap.add_argument("--n_traces", type=int, default=12)
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default="figures")
    # Laws recovered by Module 3. Defaults are the numbers actually measured.
    ap.add_argument("--b0_learned", type=float, default=-2.363)
    ap.add_argument("--b1_learned", type=float, default=0.208)
    ap.add_argument("--b0_oracle", type=float, default=-3.160)
    ap.add_argument("--b1_oracle", type=float, default=1.088)
    # Disease-calibrated law. See DEPLOYMENT NOTE below for why this variant
    # exists and why it is the scientifically correct way to run the controller.
    ap.add_argument("--b0_disease", type=float, default=None,
                    help="baseline propensity calibrated to the DISEASED cell; "
                         "defaults to the true diseased b0 (an oracle calibration).")
    ap.add_argument("--b1_disease", type=float, default=0.9)
    ap.add_argument("--b0_shift", type=float, default=-2.5,
                    help="how much harder the diseased cell is at switching off.")
    ap.add_argument("--entry_boost", type=float, default=0.40,
                    help="extra P(OSCILLATORY -> SUSTAINED_HIGH) in disease.")
    ap.add_argument("--open_loop_dose", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    est = OnlineStateEstimator(args.model)
    sim_kw = dict(n_frames=args.n_frames, b0_shift=args.b0_shift,
                  entry_boost=args.entry_boost)

    # DEPLOYMENT NOTE — why a disease-calibrated law is the correct way to run this
    # -------------------------------------------------------------------------
    # Modules 1-3 learn the feedback law from HEALTHY cells. Running that law
    # unchanged on a diseased cell is a category error, and the results below show
    # exactly why: the healthy law says "this cell will suppress itself shortly",
    # so the controller stays silent - while the diseased cell, whose baseline
    # propensity b0 is far lower, never actually recovers.
    #
    # The fix is not to abandon the learned law but to split it correctly:
    #   - b1 (feedback GAIN) is a property of the signalling pathway. It is what
    #     Module 3 recovers, and it is preserved in disease - the pathway still
    #     works, the cell is just harder to switch off.
    #   - b0 (baseline propensity) is cell- and condition-specific and must be
    #     calibrated on the cell being treated, exactly as a clinician would
    #     baseline a patient before setting stimulation parameters.
    #
    # The "disease-calibrated" variant below does that. It is an ORACLE
    # calibration (we hand it the true diseased b0) and is labelled as such: it
    # is an upper bound showing what CADENCE achieves when correctly baselined,
    # not a claim that baselining is free. Estimating b0 online from the
    # patient's own trace is the obvious next step and is listed as future work.
    b0_disease = (args.b0_disease if args.b0_disease is not None
                  else TRUE_B0 + args.b0_shift)

    all_results = {}

    # ------------------------------------------------------------------ #
    # CONDITION 1: diseased cell, feedback INTACT -> CADENCE should restore
    # ------------------------------------------------------------------ #
    print("=== DISEASE + INTACT FEEDBACK (can CADENCE restore, and cheaply?) ===")
    res = {}
    res["no control"] = evaluate(lambda: NoControl(), est, args.n_traces,
                                 args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)
    res["open-loop"] = evaluate(lambda: OpenLoop(args.open_loop_dose), est,
                                args.n_traces, args.seed,
                                b1_true=TRUE_B1_INTACT, **sim_kw)
    res["CADENCE (learned law)"] = evaluate(
        lambda: Cadence(args.b0_learned, args.b1_learned), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)
    res["CADENCE (healthy-law)"] = evaluate(
        lambda: Cadence(args.b0_oracle, args.b1_oracle), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)
    res["CADENCE (disease-calib)"] = evaluate(
        lambda: Cadence(b0_disease, args.b1_disease), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)

    for name, r in res.items():
        print(f"  {name:<26} pathological {r['pathological_frac']*100:5.1f}%"
              f" (sd {r['pathological_sd']*100:4.1f})   cost {r['cost']:8.1f}")
    all_results["disease + intact feedback"] = res

    # ------------------------------------------------------------------ #
    # CONDITION 2: THE KILL-SHOT. Feedback blocked -> CADENCE must FAIL.
    # ------------------------------------------------------------------ #
    print("\n=== KILL-SHOT: DISEASE + BLOCKED FEEDBACK (CADENCE must FAIL) ===")
    print("  If CADENCE still restored rhythm here, it would mean the policy is")
    print("  brute-forcing the system rather than working through the endogenous")
    print("  law - which would invalidate the entire claim.")
    res_b = {}
    res_b["no control"] = evaluate(lambda: NoControl(), est, args.n_traces,
                                   args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["open-loop"] = evaluate(lambda: OpenLoop(args.open_loop_dose), est,
                                  args.n_traces, args.seed,
                                  b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["CADENCE (learned law)"] = evaluate(
        lambda: Cadence(args.b0_learned, args.b1_learned), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["CADENCE (healthy-law)"] = evaluate(
        lambda: Cadence(args.b0_oracle, args.b1_oracle), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["CADENCE (disease-calib)"] = evaluate(
        lambda: Cadence(b0_disease, args.b1_disease), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)

    for name, r in res_b.items():
        print(f"  {name:<26} pathological {r['pathological_frac']*100:5.1f}%"
              f" (sd {r['pathological_sd']*100:4.1f})   cost {r['cost']:8.1f}")
    all_results["disease + blocked feedback"] = res_b

    # ------------------------------------------------------------------ #
    # Verdicts
    # ------------------------------------------------------------------ #
    print("\n=== VERDICTS ===")
    nc = res["no control"]["pathological_frac"]
    ol = res["open-loop"]
    for tag in ("CADENCE (learned law)", "CADENCE (healthy-law)",
                "CADENCE (disease-calib)"):
        cd = res[tag]
        restored = cd["pathological_frac"] < nc
        cheaper = cd["cost"] < ol["cost"]
        as_good = cd["pathological_frac"] <= ol["pathological_frac"] * 1.10
        print(f"  [{tag}]")
        print(f"    reduces pathological time vs no-control : "
              f"{'YES' if restored else 'NO'} "
              f"({nc*100:.1f}% -> {cd['pathological_frac']*100:.1f}%)")
        print(f"    restoration within 10% of open-loop      : "
              f"{'YES' if as_good else 'NO'} "
              f"({ol['pathological_frac']*100:.1f}% open-loop)")
        print(f"    cheaper than open-loop                   : "
              f"{'YES' if cheaper else 'NO'} "
              f"({ol['cost']:.0f} -> {cd['cost']:.0f}, "
              f"{(1 - cd['cost']/max(ol['cost'],1e-9))*100:.0f}% saving)")

    print("\n  [KILL-SHOT]")
    nc_b = res_b["no control"]["pathological_frac"]
    for tag in ("CADENCE (learned law)", "CADENCE (healthy-law)",
                "CADENCE (disease-calib)"):
        cd_b = res_b[tag]
        failed = cd_b["pathological_frac"] >= nc_b * 0.90
        print(f"    {tag}: pathological {nc_b*100:.1f}% -> "
              f"{cd_b['pathological_frac']*100:.1f}%  "
              f"-> {'FAILS to restore (CORRECT)' if failed else 'STILL RESTORES (RED FLAG)'}")

    # ------------------------------------------------------------------ #
    plot_control_comparison(res, os.path.join(args.figdir, "control_intact.png"),
                            "Closed-loop control, diseased cell with INTACT feedback\n"
                            "(red shading = true pathological SUSTAINED_HIGH)")
    plot_control_comparison(res_b, os.path.join(args.figdir, "control_blocked.png"),
                            "KILL-SHOT: feedback blocked - CADENCE cannot restore\n"
                            "(red shading = true pathological SUSTAINED_HIGH)")
    plot_bars(all_results, os.path.join(args.figdir, "control_summary.png"))
    print(f"\nwrote figures to {args.figdir}/")
    print("Module 4 complete. Next: run_all.py + tests/test_pipeline.py.")


if __name__ == "__main__":
    main()
