"""
generate_synthetic.py
----------------------
Ground-truth simulator for astrocyte-like glial calcium dynamics.

WHY THIS EXISTS
The project's central claim is that glial calcium is governed by a hidden,
load-dependent feedback law: the longer/harder the system sits in a high-calcium
state, the more likely it is to switch itself OFF (into a refractory/suppressed
state). To prove our inference pipeline can recover such a law from real data,
we first generate data from a KNOWN law and show recovery. This is validation,
not a result: it demonstrates the estimator works before we trust it on wet-lab data.

MODEL
Four hidden states:
    0 = QUIESCENT      (low baseline calcium)
    1 = OSCILLATORY    (baseline waves)
    2 = SUSTAINED_HIGH (pathological / activated elevation)
    3 = REFRACTORY     (self-suppressed, the feedback "off" state)

Emissions: continuous calcium fluorescence (dF/F-like), state-specific mean +
noise, with slow GCaMP-like temporal smoothing so traces resemble real imaging.

THE FEEDBACK LAW (the thing to be discovered):
The transition SUSTAINED_HIGH -> REFRACTORY is NOT constant. Its probability
rises with an accumulated "calcium load" variable L that integrates recent time
spent in the high state. This makes the process semi-Markov / load-dependent:
    P(high -> refractory | L) = sigmoid(beta0 + beta1 * L)
with beta1 > 0. beta1 is the quantitative signature of negative feedback.
A control condition (feedback "blocked") sets beta1 ~ 0, flattening the law -
this mirrors the pharmacological kill-shot experiment.

USAGE
    python generate_synthetic.py --condition intact  --n_traces 60 --out data/intact.csv
    python generate_synthetic.py --condition blocked --n_traces 60 --out data/blocked.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----- fixed biology-inspired parameters (documented, not magic numbers) -----
STATE_NAMES = ["QUIESCENT", "OSCILLATORY", "SUSTAINED_HIGH", "REFRACTORY"]
EMISSION_MEAN = np.array([0.05, 0.35, 0.90, 0.10])   # dF/F-like mean per state
EMISSION_SD   = np.array([0.03, 0.08, 0.12, 0.04])   # noise per state
GCAMP_TAU = 3.0        # frames; slow sensor decay to mimic GCaMP kinetics
DT = 0.5               # seconds per frame (2 Hz imaging)

# Base transition tendencies (rows sum handled in code). The high->refractory
# entry is overwritten dynamically by the feedback law below.
BASE_T = np.array([
    # to:  QUI   OSC   HIGH  REF
    [0.90, 0.08, 0.02, 0.00],  # from QUIESCENT
    [0.06, 0.82, 0.12, 0.00],  # from OSCILLATORY
    [0.00, 0.02, 0.00, 0.00],  # from SUSTAINED_HIGH  (row filled by law)
    [0.25, 0.10, 0.00, 0.65],  # from REFRACTORY  (slowly recovers)
])

def high_row(p_off):
    """Build the SUSTAINED_HIGH transition row given the dynamic off-probability."""
    p_stay = max(0.0, 1.0 - p_off - 0.02)   # small leak to OSC
    return np.array([0.00, 0.02, p_stay, p_off])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def state_occupancy(states):
    """Return the fraction of frames assigned to each hidden state."""
    counts = np.bincount(states, minlength=len(STATE_NAMES))
    return {name: float(count / states.size) for name, count in zip(STATE_NAMES, counts)}

def simulate_trace(rng, n_frames, beta0, beta1, load_decay=0.92):
    """
    Simulate one calcium trace under a load-dependent feedback law.
    Returns calcium, hidden state, time, and accumulated load arrays.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    if not 0.0 <= load_decay <= 1.0:
        raise ValueError("load_decay must be between 0 and 1")
    state = 0
    load = 0.0
    states = np.empty(n_frames, dtype=int)
    loads = np.empty(n_frames, dtype=float)
    latent = np.empty(n_frames)   # noise-free target the sensor chases
    for t in range(n_frames):
        states[t] = state
        # accumulate load while in HIGH, decay otherwise
        if state == 2:
            load = load * load_decay + 1.0
        else:
            load = load * load_decay
        loads[t] = load
        # dynamic feedback law only affects the HIGH row
        if state == 2:
            p_off = sigmoid(beta0 + beta1 * load)
            row = high_row(p_off)
        else:
            row = BASE_T[state].copy()
            row = row / row.sum()
        latent[t] = EMISSION_MEAN[state]
        state = rng.choice(4, p=row)
    # GCaMP-like smoothing (causal exponential) + observation noise
    calcium = np.empty(n_frames)
    c = latent[0]
    for t in range(n_frames):
        c += (latent[t] - c) / GCAMP_TAU
        calcium[t] = c + rng.normal(0, EMISSION_SD[states[t]])
    return calcium, states, np.arange(n_frames) * DT, loads


def generate_dataset(condition, n_traces=60, n_frames=600, seed=0, load_decay=0.92):
    """Generate a labeled synthetic dataset for one feedback condition."""
    if condition not in {"intact", "blocked"}:
        raise ValueError("condition must be 'intact' or 'blocked'")
    if n_traces < 1:
        raise ValueError("n_traces must be at least 1")
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")

    beta0 = -3.0
    beta1 = 0.9 if condition == "intact" else 0.02
    rng = np.random.default_rng(seed)
    rows = []
    for trace_id in range(n_traces):
        calcium, states, time, load = simulate_trace(
            rng, n_frames, beta0, beta1, load_decay=load_decay
        )
        rows.extend(
            (condition, trace_id, time_index, calcium[time_index], states[time_index], load[time_index])
            for time_index in range(n_frames)
        )
    return pd.DataFrame(
        rows,
        columns=["condition", "trace_id", "time_s", "calcium", "true_state", "load"],
    )


def simulation_metadata(condition, n_traces, n_frames, seed, load_decay):
    """Describe the parameters used to generate a synthetic dataset."""
    beta1 = 0.9 if condition == "intact" else 0.02
    return {
        "condition": condition,
        "n_traces": n_traces,
        "n_frames": n_frames,
        "seed": seed,
        "load_decay": load_decay,
        "beta0": -3.0,
        "beta1": beta1,
        "frame_interval_s": DT,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["intact", "blocked"], default="intact")
    ap.add_argument("--n_traces", type=int, default=60)
    ap.add_argument("--n_frames", type=int, default=600)   # 5 min at 2 Hz
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--load-decay",
        type=float,
        default=0.92,
        help="Per-frame retention of accumulated high-calcium load (0 to 1).",
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--metadata-out", type=str, help="Optional JSON manifest path.")
    args = ap.parse_args()

    if args.n_traces < 1:
        ap.error("--n_traces must be at least 1")
    if args.n_frames < 1:
        ap.error("--n_frames must be at least 1")
    if not 0.0 <= args.load_decay <= 1.0:
        ap.error("--load-decay must be between 0 and 1")

    df = generate_dataset(
        args.condition,
        n_traces=args.n_traces,
        n_frames=args.n_frames,
        seed=args.seed,
        load_decay=args.load_decay,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            simulation_metadata(
                args.condition, args.n_traces, args.n_frames, args.seed, args.load_decay
            ),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[{args.condition}] wrote {len(df)} rows "
          f"({args.n_traces} traces x {args.n_frames} frames) to {output_path}")
    beta1 = 0.9 if args.condition == "intact" else 0.02
    print(f"  ground-truth beta1 = {beta1}  (feedback strength)")
    print(f"  load decay = {args.load_decay}")
    print(f"  manifest = {metadata_path}")
    last_trace = df.loc[df["trace_id"] == args.n_traces - 1, "true_state"].to_numpy()
    print(f"  state occupancy (last trace) = {state_occupancy(last_trace)}")

if __name__ == "__main__":
    main()
