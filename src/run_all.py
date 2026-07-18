"""
run_all.py  —  CADENCE pipeline, Module 5
=========================================
One command reproduces the entire project from nothing: synthetic data, fitted
model, decoded states, recovered feedback law, controller benchmarks, and every
publication figure.

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

    This is the figure that shows the feedback difference as raw structure rather
    than as a fitted coefficient: in the intact condition the SUSTAINED_HIGH row
    carries real mass into REFRACTORY (the cell switches itself off), whereas in
    the blocked condition that mass collapses onto SUSTAINED_HIGH itself (the
    cell gets stuck). It is the same story `b1` tells, visible without any model.
    """
    fig, axes = plt.subplots(1, len(decoded_paths), figsize=(5.8 * len(decoded_paths), 5.0))
    if len(decoded_paths) == 1:
        axes = [axes]

    for ax, (cond, path) in zip(axes, decoded_paths.items()):
        df = pd.read_csv(path)
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
    args = ap.parse_args()

    py = sys.executable
    for d in ("data", "models", "figures"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)

    t_start = time.time()

    # -- 0. ground-truth synthetic data --------------------------------- #
    for cond in ("intact", "blocked"):
        out = os.path.join("data", f"{cond}.csv")
        if args.skip_existing and os.path.exists(os.path.join(ROOT, out)):
            print(f"[skip] {out} exists")
            continue
        run([py, "src/generate_synthetic.py", "--condition", cond,
             "--n_traces", str(args.n_traces), "--seed", str(args.seed),
             "--out", out], f"generate {cond} data")

    # -- 1. fit the hidden-state model ---------------------------------- #
    fit_cmd = [py, "src/fit_hmm.py", "--seed", str(args.seed)]
    if args.quick:
        fit_cmd += ["--n_restarts", "2", "--k_min", "3", "--k_max", "5"]
    run(fit_cmd, "Module 1: fit HMM + model-order selection")

    # -- 2. decode + validate against ground truth ---------------------- #
    run([py, "src/recover_states.py"], "Module 2: Viterbi decode + recovery scoring")

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
         "--n_traces", str(args.control_traces)],
        "Module 4: CADENCE controller vs baselines + kill-shot")

    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 74}")
    print(f"PIPELINE COMPLETE in {time.time() - t_start:.1f}s")
    print(f"{'=' * 74}")
    print("Figures written to figures/:")
    for f in sorted(os.listdir(os.path.join(ROOT, "figures"))):
        if f.endswith(".png"):
            print(f"  - {f}")
    print("\nTo verify the scientific claims still hold:")
    print("  python tests/test_pipeline.py")


if __name__ == "__main__":
    main()
