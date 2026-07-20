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
    args = ap.parse_args()

    if args.n_traces < 1:
        ap.error("--n_traces must be at least 1")
    if args.n_frames < 1:
        ap.error("--n_frames must be at least 1")
    if not 0.0 <= args.load_decay <= 1.0:
        ap.error("--load-decay must be between 0 and 1")

    # THE ONE KNOB THAT ENCODES THE HYPOTHESIS:
    # intact  -> strong positive beta1 (feedback present)
    # blocked -> beta1 ~ 0 (feedback pharmacologically removed) -> flat law
    beta0 = -3.0
    beta1 = 0.9 if args.condition == "intact" else 0.02

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n_traces):
        cal, st, tt, load = simulate_trace(
            rng, args.n_frames, beta0, beta1, load_decay=args.load_decay
        )
        for t in range(args.n_frames):
            rows.append((args.condition, i, tt[t], cal[t], st[t], load[t]))
    df = pd.DataFrame(rows, columns=["condition", "trace_id", "time_s",
                                     "calcium", "true_state", "load"])
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[{args.condition}] wrote {len(df)} rows "
          f"({args.n_traces} traces x {args.n_frames} frames) to {output_path}")
    print(f"  ground-truth beta1 = {beta1}  (feedback strength)")
    print(f"  load decay = {args.load_decay}")

if __name__ == "__main__":
    main()
