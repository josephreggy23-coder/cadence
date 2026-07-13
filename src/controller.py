"""CADENCE Module 4: a synthetic controller stress test.

The policy acts on a four-state simulated plant using

    P(high -> refractory | L, u) = sigmoid(b0 + b1 * (L + kappa * u)).

This equation is an assumption, not a discovered astrocyte mechanism. Because
the same b1 multiplies both recent high-state exposure and the intervention,
setting b1 near zero algebraically disables both. The blocked-pathway experiment
is therefore a structural consistency check for this simulator, not an
independent biological falsification.

While the estimated state is HIGH, the policy converts a constant-hazard horizon
approximation into a clipped one-step action. It is the smallest action under
that approximation; it is not a solved finite-horizon optimal-control problem,
because future load and state probabilities are not propagated.

The online estimator is causal. Outcomes are reported as synthetic high-state
occupancy and intervention cost in arbitrary units. No result in this module is
evidence of safety, efficacy, or mechanism in tissue, animals, or people. The
simulated challenge shift and transfer of b1 across conditions are assumptions
that require independent biological testing.
"""

import argparse
import json
import os

import numpy as np
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
CADENCE_U_MAX = 8.0


def sigmoid(x):
    # clip before exp: under strong intervention the logit can run large enough
    # to overflow, which is harmless numerically but produces warning noise that
    # would mask real problems in the logs.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


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
# Synthetic challenge model
# --------------------------------------------------------------------------- #
def challenge_transitions(entry_boost=0.30):
    """
    Synthetic challenge: an elevated tendency to enter SUSTAINED_HIGH.

    We raise P(OSCILLATORY -> HIGH) from 0.12 to 0.12+entry_boost, taking the mass
    from "stay oscillatory". Combined with `b0_shift` below (harder to switch off),
    this creates the high-occupancy behavior used to stress-test the policies.
    It is not calibrated to a named disease or biological preparation.
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

    The preferred kinetic model is exactly causal. The retained Gaussian-HMM
    ablation uses a centred derivative and therefore returns a one-frame-delayed
    estimate.
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)

        # Preferred path: the kinetic model, whose forward recursion is already
        # exactly causal, so no latency workaround is needed.
        if "model_type" in d and str(d["model_type"]) == "kinetic":
            from kinetic_hmm import KineticHMM, KineticFilter
            self.kind = "kinetic"
            self._filter = KineticFilter(KineticHMM.load(npz_path))
            self.reset()
            return

        self.kind = "gaussian"
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
        if getattr(self, "kind", None) == "kinetic":
            self._filter.reset()
            return
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
        if self.kind == "kinetic":
            # zero latency: the belief at frame t uses frames 1..t and nothing more
            return self._filter.step(float(calcium_t))

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
    """Fixed-dose synthetic comparator with no sensing or timing."""
    name = "open-loop"

    def __init__(self, dose=1.0):
        self.dose = dose

    def act(self, state, L):
        return self.dose


class Cadence:
    """
    Model-based controller under a constant-hazard horizon approximation.

    The returned dose is minimal only for the policy's local algebraic
    approximation; this class does not solve a dynamic optimal-control problem.
    """
    name = "CADENCE"

    def __init__(self, b0, b1, horizon=6, target=0.90,
                 u_max=CADENCE_U_MAX, kappa=KAPPA):
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
            # Under the assumed coupling, no finite dose can act when b1 is zero.
            return self.u_max
        u = (logit_needed - self.b0 - self.b1 * L) / denom
        return float(np.clip(u, 0.0, self.u_max))


class AdaptiveCadence(Cadence):
    """
    Synthetic policy that updates b0 from unstimulated inferred transitions.

    The simulator assumes b1 transfers while b0 changes across conditions. This
    class tests adaptation under that assumption; it does not establish that the
    same parameter split holds biologically.

    HOW
      1. OBSERVE-ONLY WINDOW. For the first `calib_frames` frames the controller
         applies no stimulation and simply records (load, did-it-switch-off)
         pairs from its own inferred states. Intervening during this window would
         contaminate the estimate - the whole point is to measure the cell's
         UNAIDED behaviour.
      2. Fit b0 by 1-D maximum likelihood with b1 held fixed.
      3. Refit periodically as more evidence accumulates, then control as usual.

    Limitation: the observe-only window delays intervention, and the estimate is
    only as good as the online state inference feeding it.
    """
    name = "CADENCE (self-calibrating)"

    def __init__(self, b0_init, b1, calib_frames=150, refit_every=50,
                 min_events=8, min_stim_trials=40, futility_margin=0.0,
                 max_cumulative_dose=180.0, **kw):
        super().__init__(b0_init, b1, **kw)
        self.calib_frames = calib_frames
        self.refit_every = refit_every
        self.min_events = min_events
        self.t = 0
        self.prev_state = None
        self.prev_L = None
        self.prev_u = 0.0
        self.obs_L, self.obs_y = [], []
        self.calibrated = False

        # --- simulation exposure constraints (see _check_futility) -------- #
        self.min_stim_trials = min_stim_trials
        self.futility_margin = futility_margin
        self.max_cumulative_dose = float(max_cumulative_dose)
        if self.max_cumulative_dose <= 0:
            raise ValueError("max_cumulative_dose must be positive")
        self.n_stim = self.k_stim = 0
        self.n_quiet = self.k_quiet = 0
        self.aborted = False
        self.cumulative_dose = 0.0
        self.budget_exhausted = False

    def _record(self, state, L):
        """Record an inferred transition without contaminating calibration.

        Baseline ``b0`` describes the cell's *unaided* switch-off propensity, so
        only transitions following zero stimulation belong in ``obs_L`` and
        ``obs_y``.  Stimulated outcomes are still retained in the separate
        counters used by the futility interlock.
        """
        if self.prev_state == HIGH:
            switched = 1.0 if state == REFRACTORY else 0.0
            # bucket the same outcome by whether we were stimulating, so the
            # interlock can ask "is the stimulus doing anything at all?"
            if self.prev_u > 0:
                self.n_stim += 1
                self.k_stim += switched
            else:
                self.obs_L.append(self.prev_L)
                self.obs_y.append(switched)
                self.n_quiet += 1
                self.k_quiet += switched
        self.prev_state, self.prev_L = state, L

    def _check_futility(self):
        """
        Simulation safeguard that stops after an unpromising response history.

        It compares non-random stimulated and quiet frames, so confounding by
        indication remains. This heuristic is not a validated statistical test
        or a claim of real-world safety.
        """
        if self.aborted or self.n_stim < self.min_stim_trials:
            return
        rate_stim = self.k_stim / max(self.n_stim, 1)
        rate_quiet = self.k_quiet / max(self.n_quiet, 1)
        if rate_stim <= rate_quiet + self.futility_margin:
            self.aborted = True

    def _refit_b0(self):
        """
        1-D MLE for b0 with b1 fixed. Newton steps on the logistic
        log-likelihood; cheap enough to run online.
        """
        L = np.asarray(self.obs_L)
        y = np.asarray(self.obs_y)
        if len(L) < self.min_events or y.sum() < 1:
            return
        b0 = self.b0
        for _ in range(40):
            z = b0 + self.b1 * L
            p = sigmoid(z)
            grad = np.sum(y - p)
            hess = -np.sum(p * (1 - p))
            if abs(hess) < 1e-9:
                break
            step = grad / hess
            b0 -= step
            if abs(step) < 1e-6:
                break
        if np.isfinite(b0):
            self.b0 = float(np.clip(b0, -12.0, 4.0))
            self.calibrated = True

    def act(self, state, L):
        self.t += 1
        self._record(state, L)
        self._check_futility()

        if self.aborted:
            self.prev_u = 0.0
            return 0.0                      # pathway unresponsive: stand down
        if self.cumulative_dose >= self.max_cumulative_dose:
            self.budget_exhausted = True
            self.prev_u = 0.0
            return 0.0                      # hard exposure budget reached
        if self.t <= self.calib_frames:
            self.prev_u = 0.0
            return 0.0                      # observe only: do not contaminate
        if self.t == self.calib_frames + 1 or self.t % self.refit_every == 0:
            self._refit_b0()
        if not self.calibrated:
            self.prev_u = 0.0
            return 0.0
        u = super().act(state, L)
        u = min(u, self.max_cumulative_dose - self.cumulative_dose)
        self.cumulative_dose += u
        if self.cumulative_dose >= self.max_cumulative_dose - 1e-12:
            self.budget_exhausted = True
        self.prev_u = u
        return u


# --------------------------------------------------------------------------- #
# Closed-loop simulation
# --------------------------------------------------------------------------- #
def simulate_closed_loop(policy, estimator, rng, n_frames=600,
                         b1_true=TRUE_B1_INTACT, b0_true=TRUE_B0,
                         b0_shift=-1.5, entry_boost=0.30, kappa=KAPPA,
                         est_load_decay=LOAD_DECAY):
    """
    Run one closed-loop trace.

    The plant (true dynamics) is the same generative model as
    generate_synthetic.py, with the challenge modifications applied, plus the
    control input entering through the feedback law. The controller sees ONLY the
    noisy calcium sample — never the true state, never the load.

    `b0_shift` lowers the synthetic plant's switch-off propensity;
    `entry_boost` makes it enter the high state more often.
    """
    T_dis = challenge_transitions(entry_boost)
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
            est_load = (
                est_load * est_load_decay + (1.0 if s_hat == HIGH else 0.0)
            )
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
    high_frac, costs, traces = [], [], []
    for i in range(n_traces):
        out = simulate_closed_loop(policy_factory(), estimator, rng, **sim_kw)
        high_frac.append(np.mean(out["true_state"] == HIGH))
        costs.append(out["action"].sum())
        traces.append(out)
    return {"high_state_frac": float(np.mean(high_frac)),
            "high_state_sd": float(np.std(high_frac)),
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
        # Shade true SUSTAINED_HIGH periods in the synthetic plant.
        high_mask = tr["true_state"] == HIGH
        ax.fill_between(t, 0, 1, where=high_mask, transform=ax.get_xaxis_transform(),
                        color="#e07b7b", alpha=0.30, lw=0)
        ax.plot(t, tr["calcium"], color="black", lw=0.8)
        u = tr["action"]
        if u.max() > 0:
            ax2 = ax.twinx()
            ax2.plot(t, u, color="#1f77b4", lw=0.9, alpha=0.85)
            ax2.set_ylabel("stim u", color="#1f77b4", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
        ax.set_ylabel("dF/F")
        ax.set_title(f"{name}  —  high-state {res['high_state_frac']*100:.1f}% "
                     f"of time,  cost {res['cost']:.0f}", fontsize=10, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_bars(all_results, out_png):
    """High-state occupancy and cost across policies and conditions."""
    conds = list(all_results.keys())
    policies = list(next(iter(all_results.values())).keys())
    x = np.arange(len(policies))
    w = 0.8 / len(conds)
    colors = ["#1f77b4", "#d62728", "#7f7f7f"]

    # wrap long policy names onto two lines so the tick labels never collide
    short = [p.replace(" (", "\n(") for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for k, cond in enumerate(conds):
        vals = [all_results[cond][p]["high_state_frac"] * 100 for p in policies]
        err = [all_results[cond][p]["high_state_sd"] * 100 for p in policies]
        axes[0].bar(x + k * w, vals, w, yerr=err, capsize=3,
                    label=cond, color=colors[k % len(colors)])
        cvals = [all_results[cond][p]["cost"] for p in policies]
        axes[1].bar(x + k * w, cvals, w, label=cond, color=colors[k % len(colors)])

    for ax, ylab, title in ((axes[0], "% time in SUSTAINED_HIGH", "High-state occupancy"),
                            (axes[1], "cumulative intervention cost", "Cost")):
        ax.set_xticks(x + w * (len(conds) - 1) / 2, short, fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic controller benchmark\n"
                 "occupancy and intervention cost under the assumed plant")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="CADENCE model-based controller.")
    ap.add_argument("--model", default="models/kinetic_model.npz")
    ap.add_argument("--n_traces", type=int, default=12)
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--result_out", default="results/controller_benchmark.json",
                    help="machine-readable synthetic benchmark output")
    ap.add_argument("--estimates", default="results/feedback_estimates.json",
                    help="machine-readable Module 3 output")
    # Explicit values override the versioned Module 3 artifact.
    ap.add_argument("--b0_learned", type=float, default=None)
    ap.add_argument("--b1_learned", type=float, default=None)
    ap.add_argument("--est_load_decay", type=float, default=None,
                    help="controller exposure decay; defaults to Module 3's value")
    # Oracle challenge calibration used only as a simulation reference arm.
    ap.add_argument("--b0_challenge", "--b0_disease", dest="b0_challenge",
                    type=float, default=None,
                    help="reference baseline propensity for the shifted plant")
    ap.add_argument("--b1_reference", "--b1_disease", dest="b1_reference",
                    type=float, default=0.9)
    ap.add_argument("--b0_shift", type=float, default=-2.5,
                    help="synthetic decrease in baseline switch-off propensity")
    ap.add_argument("--entry_boost", type=float, default=0.40,
                    help="synthetic increase in P(OSCILLATORY -> SUSTAINED_HIGH)")
    ap.add_argument("--open_loop_dose", type=float, default=1.0)
    ap.add_argument("--adaptive_budget_fraction", type=float, default=0.30,
                    help="hard adaptive-policy cumulative budget as a fraction "
                         "of continuous open-loop exposure")
    args = ap.parse_args()

    learned_b0 = learned_b1 = learned_decay = None
    if args.estimates and os.path.exists(args.estimates):
        with open(args.estimates, encoding="utf-8") as handle:
            estimate_payload = json.load(handle)
        intact = estimate_payload.get("causal", {}).get("intact", {})
        if not intact:
            # Backward compatibility with pre-v2 result artifacts.
            intact = estimate_payload.get("inferred", {}).get("intact", {})
        if intact:
            learned_b0 = float(intact["b0"])
            learned_b1 = float(intact["b1"])
        learned_decay = float(estimate_payload["shared_decay"])
        print(f"Loaded learned law from {args.estimates}: "
              f"b0={learned_b0:+.3f}, b1={learned_b1:+.3f}, "
              f"decay={learned_decay:.3f}")
    else:
        print("Module 3 estimate artifact not found; explicit learned-law "
              "coefficients and decay are required.")
    if args.b0_learned is not None:
        learned_b0 = args.b0_learned
    if args.b1_learned is not None:
        learned_b1 = args.b1_learned
    if args.est_load_decay is not None:
        learned_decay = args.est_load_decay
    if learned_b0 is None or learned_b1 is None or learned_decay is None:
        ap.error(
            "run Module 3 first, or provide --b0_learned, --b1_learned, "
            "and --est_load_decay"
        )
    if not 0.0 < learned_decay <= 1.0:
        ap.error("controller exposure decay must be in (0, 1]")
    if not 0.0 < args.adaptive_budget_fraction <= 1.0:
        ap.error("--adaptive_budget_fraction must be in (0, 1]")

    os.makedirs(args.figdir, exist_ok=True)
    est = OnlineStateEstimator(args.model)
    sim_kw = dict(n_frames=args.n_frames, b0_shift=args.b0_shift,
                  entry_boost=args.entry_boost,
                  est_load_decay=learned_decay)
    adaptive_budget = (
        args.adaptive_budget_fraction * args.open_loop_dose * args.n_frames
    )

    # MODEL-MISMATCH NOTE
    # The plant below is deliberately shifted in b0 while b1 is preserved. That
    # split is part of the simulated scenario, not a biological finding. The
    # oracle arm receives the true shifted b0; the adaptive arm estimates it from
    # unstimulated inferred transitions.
    b0_challenge = (args.b0_challenge if args.b0_challenge is not None
                    else TRUE_B0 + args.b0_shift)

    all_results = {}

    # ------------------------------------------------------------------ #
    # CONDITION 1: shifted synthetic plant with intact feedback
    # ------------------------------------------------------------------ #
    print("=== SHIFTED SYNTHETIC PLANT + INTACT FEEDBACK ===")
    res = {}
    res["no control"] = evaluate(lambda: NoControl(), est, args.n_traces,
                                 args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)
    res["open-loop"] = evaluate(lambda: OpenLoop(args.open_loop_dose), est,
                                args.n_traces, args.seed,
                                b1_true=TRUE_B1_INTACT, **sim_kw)
    res["CADENCE (learned law)"] = evaluate(
        lambda: Cadence(learned_b0, learned_b1), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)
    res["CADENCE (plant-param ref)"] = evaluate(
        lambda: Cadence(b0_challenge, args.b1_reference), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)
    res["CADENCE (self-calib)"] = evaluate(
        lambda: AdaptiveCadence(
            learned_b0, learned_b1, max_cumulative_dose=adaptive_budget
        ), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_INTACT, **sim_kw)

    for name, r in res.items():
        print(f"  {name:<26} high-state {r['high_state_frac']*100:5.1f}%"
              f" (sd {r['high_state_sd']*100:4.1f})   cost {r['cost']:8.1f}")
    all_results["shifted synthetic plant + intact feedback"] = res

    # ------------------------------------------------------------------ #
    # CONDITION 2: structural blockade check. With u multiplied by b1, setting
    # b1 near zero disables intervention by construction.
    # ------------------------------------------------------------------ #
    print("\n=== STRUCTURAL CHECK: BLOCKED COEFFICIENT DISABLES CONTROL ===")
    print("  This is expected from the assumed equation; it is a code/model")
    print("  consistency check, not independent biological evidence.")
    res_b = {}
    res_b["no control"] = evaluate(lambda: NoControl(), est, args.n_traces,
                                   args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["open-loop"] = evaluate(lambda: OpenLoop(args.open_loop_dose), est,
                                  args.n_traces, args.seed,
                                  b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["CADENCE (learned law)"] = evaluate(
        lambda: Cadence(learned_b0, learned_b1), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["CADENCE (plant-param ref)"] = evaluate(
        lambda: Cadence(b0_challenge, args.b1_reference), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)
    res_b["CADENCE (self-calib)"] = evaluate(
        lambda: AdaptiveCadence(
            learned_b0, learned_b1, max_cumulative_dose=adaptive_budget
        ), est,
        args.n_traces, args.seed, b1_true=TRUE_B1_BLOCKED, **sim_kw)

    for name, r in res_b.items():
        print(f"  {name:<26} high-state {r['high_state_frac']*100:5.1f}%"
              f" (sd {r['high_state_sd']*100:4.1f})   cost {r['cost']:8.1f}")
    all_results["shifted synthetic plant + blocked feedback"] = res_b

    result_payload = {
        "schema_version": 3,
        "analysis_scope": (
            "synthetic policy stress test; not evidence of biological efficacy or safety"
        ),
        "seed": int(args.seed),
        "n_traces": int(args.n_traces),
        "n_frames_per_trace": int(args.n_frames),
        "assumptions": {
            "intervention_equation": "sigmoid(b0 + b1 * (L + kappa * u))",
            "true_b1_intact": float(TRUE_B1_INTACT),
            "true_b1_blocked": float(TRUE_B1_BLOCKED),
            "b0_shift": float(args.b0_shift),
            "entry_boost": float(args.entry_boost),
            "plant_load_decay": float(LOAD_DECAY),
            "controller_exposure_decay": float(learned_decay),
            "intervention_cost_units": "arbitrary",
            "open_loop_dose_per_frame": float(args.open_loop_dose),
            "cadence_peak_dose_limit": float(CADENCE_U_MAX),
            "adaptive_cumulative_budget": float(adaptive_budget),
            "adaptive_budget_fraction_of_open_loop": float(
                args.adaptive_budget_fraction
            ),
        },
        "learned_law": {"b0": float(learned_b0), "b1": float(learned_b1)},
        "scenarios": {
            scenario: {
                policy: {
                    metric: float(value)
                    for metric, value in metrics.items()
                    if metric != "traces"
                }
                for policy, metrics in policies.items()
            }
            for scenario, policies in all_results.items()
        },
    }
    result_dir = os.path.dirname(args.result_out)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(args.result_out, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)
        handle.write("\n")
    print(f"\nwrote {args.result_out}")

    # ------------------------------------------------------------------ #
    # Verdicts
    # ------------------------------------------------------------------ #
    print("\n=== DESCRIPTIVE CHECKS ===")
    print("  Costs are not efficacy-matched, units are arbitrary, peak doses differ,")
    print("  and the adaptive policy has an explicit cumulative budget.")
    nc = res["no control"]["high_state_frac"]
    ol = res["open-loop"]
    for tag in ("CADENCE (learned law)", "CADENCE (plant-param ref)",
                "CADENCE (self-calib)"):
        cd = res[tag]
        restored = cd["high_state_frac"] < nc
        cheaper = cd["cost"] < ol["cost"]
        as_good = cd["high_state_frac"] <= ol["high_state_frac"] * 1.10
        print(f"  [{tag}]")
        print(f"    reduces high-state time vs no-control   : "
              f"{'YES' if restored else 'NO'} "
              f"({nc*100:.1f}% -> {cd['high_state_frac']*100:.1f}%)")
        print(f"    occupancy within 10% of open-loop        : "
              f"{'YES' if as_good else 'NO'} "
              f"({ol['high_state_frac']*100:.1f}% open-loop)")
        print(f"    lower cumulative cost than open-loop     : "
              f"{'YES' if cheaper else 'NO'} "
              f"({ol['cost']:.0f} -> {cd['cost']:.0f}, "
              f"{(1 - cd['cost']/max(ol['cost'],1e-9))*100:.0f}% saving)")

    print("\n  [STRUCTURAL BLOCKADE CHECK]")
    nc_b = res_b["no control"]["high_state_frac"]
    for tag in ("CADENCE (learned law)", "CADENCE (plant-param ref)",
                "CADENCE (self-calib)"):
        cd_b = res_b[tag]
        failed = cd_b["high_state_frac"] >= nc_b * 0.90
        print(f"    {tag}: high-state {nc_b*100:.1f}% -> "
              f"{cd_b['high_state_frac']*100:.1f}%  "
              f"-> {'fails as constructed' if failed else 'unexpected restoration'}")

    # ------------------------------------------------------------------ #
    plot_control_comparison(res, os.path.join(args.figdir, "control_intact.png"),
                            "Closed-loop control, shifted plant with INTACT feedback\n"
                            "(red shading = true SUSTAINED_HIGH)")
    plot_control_comparison(res_b, os.path.join(args.figdir, "control_blocked.png"),
                            "Structural check: b1 near zero disables control by construction\n"
                            "(red shading = true SUSTAINED_HIGH)")
    plot_bars(all_results, os.path.join(args.figdir, "control_summary.png"))
    print(f"\nwrote figures to {args.figdir}/")
    print("Module 4 complete. Next: run_all.py + tests/test_pipeline.py.")


if __name__ == "__main__":
    main()
