"""Compact multi-seed robustness benchmark for CADENCE's synthetic pipeline.

The reference benchmark is deliberately deterministic so its figures and
machine-readable results can be reproduced exactly. That does not answer a
different question: whether its qualitative synthetic findings depend on one
fortunate random draw. This script repeats the end-to-end estimator over a
small, fixed set of independent generator seeds and reports descriptive
between-seed variation.

It measures only synthetic quantities:

* held-out state recovery under the same four-state simulator;
* causal-decoder b1 in intact and blocked synthetic conditions; and
* their within-seed contrast.

The seed distribution is not a biological sample, a confidence interval, or
independent validation. It is a reproducibility sensitivity check within the
assumed simulator family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estimate_feedback_law import (
    bootstrap_coefficients,
    build_transition_dataset,
    fit_logistic,
)
from generate_synthetic import generate_dataset
from kinetic_hmm import initial_model, select_tau
from recover_states import decode_dataframe


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
SHARED_DECAY = 0.90


def _sequences(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return calcium traces in the same ordering used by the CLI fitter."""
    return [
        group.sort_values("time_s")["calcium"].to_numpy()
        for _, group in frame.groupby("trace_id", sort=True)
    ]


def _accuracy(frame: pd.DataFrame, state_col: str) -> float:
    return float((frame[state_col].to_numpy() == frame["true_state"].to_numpy()).mean())


def _recall(frame: pd.DataFrame, state_col: str, target_state: int) -> float:
    truth = frame["true_state"].to_numpy() == target_state
    if not truth.any():
        return float("nan")
    return float((frame.loc[truth, state_col].to_numpy() == target_state).mean())


def _fit_slope(
    frame: pd.DataFrame,
    state_col: str,
    decay: float,
    *,
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[float, tuple[float, float], int, int, np.ndarray]:
    """Fit one slope and its whole-trace bootstrap interval for a seed replicate."""
    load, outcome, _ = build_transition_dataset(frame, state_col, decay)
    if len(load) < 20 or outcome.sum() < 3:
        raise RuntimeError("insufficient high-state transitions for a slope estimate")
    beta, _, _ = fit_logistic(load, outcome)
    bootstrap = bootstrap_coefficients(
        frame, state_col, decay, n_boot=n_boot, seed=bootstrap_seed
    )
    if len(bootstrap) < max(20, n_boot // 2):
        raise RuntimeError("insufficient successful trace-bootstrap fits for a slope interval")
    lower, upper = np.percentile(bootstrap[:, 1], [2.5, 97.5])
    return (
        float(beta[1]),
        (float(lower), float(upper)),
        int(len(load)),
        int(outcome.sum()),
        bootstrap[:, 1],
    )


def run_seed(
    seed: int,
    *,
    n_traces: int,
    fit_traces: int,
    n_frames: int,
    n_iter: int,
    decay: float,
    n_boot: int,
) -> dict[str, object]:
    """Run one independent, fully in-memory synthetic benchmark replicate."""
    if not 0 < fit_traces < n_traces:
        raise ValueError("fit_traces must be greater than zero and smaller than n_traces")

    # Keep every random stream distinct across conditions and replicates. The
    # kinetic fitter itself is deterministic given the generated observations.
    base_seed = int(seed)
    intact_seed = 2 * base_seed
    blocked_seed = intact_seed + 1
    intact = generate_dataset(
        "intact", n_traces=n_traces, n_frames=n_frames, seed=intact_seed
    )
    blocked = generate_dataset(
        "blocked", n_traces=n_traces, n_frames=n_frames, seed=blocked_seed
    )

    train = _sequences(intact[intact["trace_id"] < fit_traces])
    selected_tau = select_tau(train, verbose=False)
    model = initial_model(train, tau0=selected_tau)
    model.fit(train, n_iter=n_iter, verbose=False)

    intact_decoded = decode_dataframe(intact, model, None, "kinetic")
    blocked_decoded = decode_dataframe(blocked, model, None, "kinetic")
    intact_decoded["model_split"] = np.where(
        intact_decoded["trace_id"] < fit_traces, "fit", "held_out"
    )
    blocked_decoded["model_split"] = "condition_transfer"
    held_out = intact_decoded[intact_decoded["model_split"] == "held_out"].copy()

    intact_b1, intact_ci, intact_frames, intact_events, intact_bootstrap = _fit_slope(
        held_out,
        "causal_state",
        decay,
        n_boot=n_boot,
        bootstrap_seed=100_000 + intact_seed,
    )
    blocked_b1, blocked_ci, blocked_frames, blocked_events, blocked_bootstrap = _fit_slope(
        blocked_decoded,
        "causal_state",
        decay,
        n_boot=n_boot,
        bootstrap_seed=100_000 + blocked_seed,
    )
    # The two independent bootstrap streams approximate the distribution of
    # a contrast between unpaired synthetic conditions. Pair only by draw index
    # after independently generating the streams; do not reuse an RNG stream.
    draws = min(len(intact_bootstrap), len(blocked_bootstrap))
    if draws < max(20, n_boot // 2):
        raise RuntimeError("insufficient bootstrap draws for the within-seed contrast")
    contrast_ci = tuple(
        float(value)
        for value in np.percentile(
            intact_bootstrap[:draws] - blocked_bootstrap[:draws], [2.5, 97.5]
        )
    )
    return {
        "base_seed": base_seed,
        "intact_generator_seed": intact_seed,
        "blocked_generator_seed": blocked_seed,
        "selected_tau": float(selected_tau),
        "fitted_tau": float(model.tau),
        "smoothed_accuracy": _accuracy(held_out, "inferred_state"),
        "causal_accuracy": _accuracy(held_out, "causal_state"),
        "causal_refractory_recall": _recall(held_out, "causal_state", 3),
        "causal_intact_b1": intact_b1,
        "causal_intact_b1_ci95": list(intact_ci),
        "causal_blocked_b1": blocked_b1,
        "causal_blocked_b1_ci95": list(blocked_ci),
        "causal_b1_contrast": intact_b1 - blocked_b1,
        "causal_b1_contrast_ci95": list(contrast_ci),
        "contrast_bootstrap_draws": draws,
        "intact_hazard_frames": intact_frames,
        "intact_switch_events": intact_events,
        "intact_bootstrap_draws": int(len(intact_bootstrap)),
        "blocked_hazard_frames": blocked_frames,
        "blocked_switch_events": blocked_events,
        "blocked_bootstrap_draws": int(len(blocked_bootstrap)),
    }


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
    }


def summarize_runs(runs: list[dict[str, float | int]]) -> dict[str, object]:
    """Produce descriptive, JSON-safe summaries without inferential language."""
    if not runs:
        raise ValueError("at least one seed replicate is required")
    summaries = {
        key: _summary_stats([float(run[key]) for run in runs])
        for key in (
            "smoothed_accuracy",
            "causal_accuracy",
            "causal_refractory_recall",
            "causal_intact_b1",
            "causal_blocked_b1",
            "causal_b1_contrast",
        )
    }
    contrasts = np.asarray([float(run["causal_b1_contrast"]) for run in runs])
    return {
        "descriptive_statistics": summaries,
        "direction_checks": {
        "positive_causal_contrasts": int((contrasts > 0).sum()),
        "contrast_intervals_excluding_zero": int(
            sum(float(run["causal_b1_contrast_ci95"][0]) > 0 for run in runs)
        ),
            "total_seed_replicates": len(runs),
        },
    }


def plot_runs(runs: list[dict[str, object]], output_path: Path) -> None:
    """Render a compact figure that preserves each seed rather than hiding it."""
    seeds = np.asarray([int(run["base_seed"]) for run in runs])
    smoothed = np.asarray([float(run["smoothed_accuracy"]) for run in runs])
    causal = np.asarray([float(run["causal_accuracy"]) for run in runs])
    contrast = np.asarray([float(run["causal_b1_contrast"]) for run in runs])
    contrast_ci = np.asarray([run["causal_b1_contrast_ci95"] for run in runs], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    fig.patch.set_facecolor("#F7F9FC")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E7EBF0", linewidth=0.8)

    axes[0].plot(seeds, smoothed, "o-", color="#386CB0", label="offline smoothed")
    axes[0].plot(seeds, causal, "o-", color="#E67E22", label="causal filter")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("independent generator seed")
    axes[0].set_ylabel("held-out frame accuracy")
    axes[0].set_title("State recovery across synthetic replicates", loc="left", weight="bold")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        seeds,
        contrast,
        yerr=np.vstack([contrast - contrast_ci[:, 0], contrast_ci[:, 1] - contrast]),
        fmt="o",
        color="#1B998B",
        ecolor="#86C8BC",
        capsize=3,
    )
    axes[1].axhline(0, color="#B42318", linestyle="--", linewidth=1)
    axes[1].set_xlabel("independent generator base seed")
    axes[1].set_ylabel("causal intact − blocked b1")
    axes[1].set_title("Within-seed synthetic hazard contrast", loc="left", weight="bold")

    fig.suptitle(
        "CADENCE multi-seed synthetic sensitivity benchmark",
        x=0.055, ha="left", fontsize=14, weight="bold", color="#172033",
    )
    fig.text(
        0.055,
        0.015,
        "Error bars: per-seed whole-trace bootstrap 95% intervals. Across-seed variation is descriptive, not biological validation.",
        fontsize=8.8,
        color="#5B6472",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-traces", type=int, default=30)
    parser.add_argument("--fit-traces", type=int, default=10)
    parser.add_argument("--n-frames", type=int, default=600)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-boot", type=int, default=150)
    parser.add_argument("--decay", type=float, default=SHARED_DECAY)
    parser.add_argument("--result-out", type=Path, default=Path("results/seed_robustness.json"))
    parser.add_argument("--figure-out", type=Path, default=Path("figures/seed_robustness.png"))
    args = parser.parse_args()
    if not args.seeds:
        parser.error("provide at least one seed")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if not 0 < args.fit_traces < args.n_traces:
        parser.error("--fit-traces must be greater than zero and smaller than --n-traces")
    if args.n_frames < 50 or args.n_iter < 1 or args.n_boot < 20:
        parser.error("--n-frames must be at least 50, --n-iter positive, and --n-boot at least 20")
    if not 0 < args.decay < 1:
        parser.error("--decay must be between zero and one")
    return args


def main() -> None:
    args = parse_args()
    runs = []
    for seed in args.seeds:
        print(f"\n=== synthetic seed {seed} ===", flush=True)
        result = run_seed(
            seed,
            n_traces=args.n_traces,
            fit_traces=args.fit_traces,
            n_frames=args.n_frames,
            n_iter=args.n_iter,
            decay=args.decay,
            n_boot=args.n_boot,
        )
        runs.append(result)
        print(
            "  held-out accuracy: "
            f"smoothed={result['smoothed_accuracy']:.3f}, "
            f"causal={result['causal_accuracy']:.3f}",
            flush=True,
        )
        print(
            "  causal b1: "
            f"intact={result['causal_intact_b1']:+.3f}, "
            f"blocked={result['causal_blocked_b1']:+.3f}, "
            f"contrast={result['causal_b1_contrast']:+.3f} "
            f"[{result['causal_b1_contrast_ci95'][0]:+.3f}, "
            f"{result['causal_b1_contrast_ci95'][1]:+.3f}]",
            flush=True,
        )

    summary = summarize_runs(runs)
    payload = {
        "schema_version": 1,
        "analysis_scope": (
            "compact multi-seed synthetic sensitivity analysis; descriptive within "
            "the assumed simulator family, not a confidence interval, independent "
            "validation, or biological result"
        ),
        "configuration": {
            "base_seeds": [int(seed) for seed in args.seeds],
            "generator_seed_mapping": "intact=2*base_seed; blocked=2*base_seed+1",
            "n_traces_per_condition": args.n_traces,
            "fit_traces": args.fit_traces,
            "held_out_intact_traces": args.n_traces - args.fit_traces,
            "frames_per_trace": args.n_frames,
            "kinetic_em_iterations": args.n_iter,
            "trace_cluster_bootstrap_draws_per_arm": args.n_boot,
            "shared_exposure_decay": args.decay,
        },
        "per_seed": runs,
        "summary": summary,
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_runs(runs, args.figure_out)
    checks = summary["direction_checks"]
    print(
        f"\npositive causal contrasts: {checks['positive_causal_contrasts']}/"
        f"{checks['total_seed_replicates']}"
    )
    print(
        "contrast intervals excluding zero: "
        f"{checks['contrast_intervals_excluding_zero']}/"
        f"{checks['total_seed_replicates']}"
    )
    print(f"wrote {args.result_out} and {args.figure_out}")


if __name__ == "__main__":
    main()
