"""
run_all.py  —  CADENCE pipeline, Module 5
=========================================
One command reproduces the synthetic benchmark and, when the versioned compact
H1R export is present, the real-data secondary analysis: generated traces,
fitted model, held-out state recovery, hazard-law recovery, policy stress tests,
machine-readable results, and figures.

    python src/run_all.py                 # full run (model-order sweep included)
    python src/run_all.py --quick         # fast rerun, narrower sweep
    python src/run_all.py --skip-existing # don't regenerate data that's present

WHY A SUBPROCESS ORCHESTRATOR RATHER THAN IMPORTS
--------------------------------------------------
Each stage is invoked exactly as the README documents it, through its own CLI.
That means this script cannot silently diverge from the documented commands — if
`python src/fit_hmm.py` breaks, `run_all.py` breaks in the same way, rather than
succeeding through some private code path a reader can't reproduce by hand.

REPRODUCIBILITY
---------------
Every stage takes an explicit --seed and all of them default to the same value,
so a full run is deterministic end to end. The only figure this script produces
directly is the transition-matrix comparison (below); everything else is written
by the module that owns it, so a figure can never drift from the analysis that
generated it.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
STATE_NAMES = ["QUIESCENT", "OSCILLATORY", "SUSTAINED_HIGH", "REFRACTORY"]


def run(cmd, label):
    """Run one pipeline stage, echoing the exact command so it can be re-run by hand."""
    print(f"\n{'=' * 74}\n>>> {label}\n    $ {' '.join(cmd)}\n{'=' * 74}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
    res = subprocess.run(cmd, cwd=ROOT, env=env)
    if res.returncode != 0:
        print(f"\n!!! stage failed: {label} (exit {res.returncode})")
        sys.exit(res.returncode)
    print(f"    [{label} done in {time.time() - t0:.1f}s]")


# --------------------------------------------------------------------------- #
# The one figure this script owns: transition-matrix comparison
# --------------------------------------------------------------------------- #
def empirical_transitions(states):
    """Row-normalised empirical transition matrix from a decoded state sequence."""
    counts = np.zeros((4, 4))
    np.add.at(counts, (states[:-1], states[1:]), 1)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def plot_transition_matrices(decoded_paths, out_png):
    """
    Transition-matrix heatmaps, intact vs blocked, built from the INFERRED states.

    This summarizes decoded transitions without fitting the hazard regression.
    It remains a synthetic, model-dependent diagnostic because the labels come
    from the fitted state estimator.
    """
    fig, axes = plt.subplots(1, len(decoded_paths), figsize=(5.8 * len(decoded_paths), 5.0))
    if len(decoded_paths) == 1:
        axes = [axes]

    for ax, (cond, path) in zip(axes, decoded_paths.items()):
        df = pd.read_csv(path)
        if "model_split" in df and (df["model_split"] == "held_out").any():
            df = df[df["model_split"] != "fit"]
        # concatenate per-trace so no transition straddles two cells
        mats = []
        for _, g in df.groupby("trace_id", sort=True):
            s = g.sort_values("time_s")["inferred_state"].to_numpy()
            mats.append(empirical_transitions(s))
        M = np.mean(mats, axis=0)

        im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(4), STATE_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(4), STATE_NAMES, fontsize=8)
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")
        ax.set_title(f"{cond}")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.6 else "black")
        # highlight the transition the whole project is about
        ax.add_patch(plt.Rectangle((3 - 0.5, 2 - 0.5), 1, 1, fill=False,
                                   edgecolor="#33dd88", lw=2.5))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Inferred transition matrices (row-normalised)\n"
                 "green box = SUSTAINED_HIGH -> REFRACTORY, the feedback transition")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run the complete CADENCE pipeline.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_traces", type=int, default=60)
    ap.add_argument("--quick", action="store_true",
                    help="fewer EM restarts and a narrower K sweep; for reruns.")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="reuse data/*.csv if already present.")
    ap.add_argument("--control_traces", type=int, default=12)
    ap.add_argument("--fit_traces", type=int, default=15,
                    help="traces used to fit the kinetic model.")
    ap.add_argument("--with-ablation", dest="with_ablation", action="store_true",
                    help="also run the Gaussian-HMM baseline and its K sweep "
                         "(slow; regenerates the model-order figure).")
    args = ap.parse_args()

    py = sys.executable
    for d in ("data", "models", "figures", "results"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)

    t_start = time.time()

    # -- 0. ground-truth synthetic data --------------------------------- #
    for cond in ("intact", "blocked"):
        out = os.path.join("data", f"{cond}.csv")
        if args.skip_existing and os.path.exists(os.path.join(ROOT, out)):
            print(f"[skip] {out} exists")
            continue
        condition_seed = args.seed + (0 if cond == "intact" else 1)
        run([py, "src/generate_synthetic.py", "--condition", cond,
             "--n_traces", str(args.n_traces), "--seed", str(condition_seed),
             "--out", out], f"generate {cond} data")

    # -- 1. fit the state estimator ------------------------------------- #
    # The kinetic model (Module 1b) is the production estimator: it models the
    # GCaMP sensor explicitly instead of assuming y_t depends only on s_t, which
    # took REFRACTORY recall from 38% to 82% and removed the b1 attenuation.
    run([py, "src/kinetic_hmm.py", "--fit_traces", str(args.fit_traces),
         "--out", "models/kinetic_model.npz"],
        "Module 1b: fit kinetic (sensor-aware) state model")

    # The Gaussian HMM is retained as a reproducible ABLATION - it is what
    # produces the model-order ("why 4 states") figure and the baseline the
    # kinetic model is measured against. Off by default because its K-sweep
    # dominates runtime.
    if args.with_ablation:
        fit_cmd = [py, "src/fit_hmm.py", "--seed", str(args.seed)]
        if args.quick:
            fit_cmd += ["--n_restarts", "2", "--k_min", "3", "--k_max", "5"]
        run(fit_cmd, "Module 1 (ablation): Gaussian HMM + model-order selection")
        run([py, "src/recover_states.py", "--model", "models/hmm_model.npz"],
            "Module 2 (ablation): recovery scoring with the Gaussian HMM")

    # -- 2. decode + validate against ground truth ---------------------- #
    run([py, "src/recover_states.py", "--model", "models/kinetic_model.npz"],
        "Module 2: decode + recovery scoring")

    # -- transition-matrix figure (owned by this script) ---------------- #
    plot_transition_matrices(
        {"intact": os.path.join(ROOT, "data", "decoded_intact.csv"),
         "blocked": os.path.join(ROOT, "data", "decoded_blocked.csv")},
        os.path.join(ROOT, "figures", "transition_matrices.png"))

    # -- 3. recover the feedback law ------------------------------------ #
    fb_cmd = [py, "src/estimate_feedback_law.py", "--seed", str(args.seed)]
    if args.quick:
        fb_cmd += ["--n_boot", "150"]
    run(fb_cmd, "Module 3: recover feedback law (b1) with CIs")

    # -- 4. the controller ---------------------------------------------- #
    run([py, "src/controller.py", "--seed", str(args.seed),
         "--model", "models/kinetic_model.npz",
         "--n_traces", str(args.control_traces)],
        "Module 4: synthetic controller stress test")

    # -- 5. public H1R astrocyte secondary analysis -------------------- #
    h1r_export = os.path.join(ROOT, "data", "processed", "h1r_astrocytes_v1.csv.gz")
    if os.path.exists(h1r_export):
        run([py, "src/analyze_h1r_astrocytes.py"],
            "Module 5: reproduce H1R astrocyte secondary analysis")
    else:
        print("[skip] compact H1R export not present; see docs/real_data.md")

    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 74}")
    print(f"PIPELINE COMPLETE in {time.time() - t_start:.1f}s")
    print(f"{'=' * 74}")
    print("Figures written to figures/:")
    for f in sorted(os.listdir(os.path.join(ROOT, "figures"))):
        if f.endswith(".png"):
            print(f"  - {f}")
    print("\nTo verify the documented synthetic benchmark checks:")
    print("  python tests/test_pipeline.py")


if __name__ == "__main__":
    main()
