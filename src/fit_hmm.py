"""Fit the Gaussian-HMM baseline used in the CADENCE ablation.

The model uses calcium level and slope while excluding simulator state labels.
Model order is assessed with trace-level held-out likelihood and BIC for
``K=2..6``. Four states are retained as an interpretable benchmark, although the
information criteria continue to improve slightly beyond four because sensor
smoothing makes the observed signal more continuous than the latent process.

The final model is fitted on intact traces, saved as plain arrays, and evaluated
by ``recover_states.py``. The sensor-aware kinetic model in ``kinetic_hmm.py`` is
the primary estimator.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: we only ever save figures, never display
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from features import make_features, CALCIUM_COL


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_traces(csv_path, feature_mode="level+slope"):
    """
    Load a condition CSV and return the observed calcium in hmmlearn's
    ``(X, lengths)`` representation for independent sequences:

        X       : shape (total_frames, n_features), with traces stacked vertically
        lengths : per-trace frame counts that preserve sequence boundaries

    Only the ``calcium`` column is used. Simulator labels are reserved for
    downstream evaluation.
    """
    df = pd.read_csv(csv_path)
    lengths = []
    chunks = []
    # group by trace so each cell is its own Markov sequence. Features are built
    # PER TRACE so the derivative never straddles a boundary between two cells.
    for _, g in df.groupby("trace_id", sort=True):
        g = g.sort_values("time_s")
        chunks.append(make_features(g["calcium"].to_numpy(), feature_mode))
        lengths.append(len(g))
    X = np.vstack(chunks)  # (total_frames, n_features)
    return X, lengths


def split_traces(X, lengths, test_frac=0.3, seed=0):
    """
    Split into train/test at the TRACE level (never mid-trace), so held-out
    scoring measures generalization to unseen cells, not memorization of the tail
    of a cell we already trained on. Returns (X_tr, len_tr, X_te, len_te).
    """
    rng = np.random.default_rng(seed)
    n = len(lengths)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_ids = set(idx[:n_test].tolist())

    # rebuild contiguous X for each split from the per-trace slices
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    tr_chunks, te_chunks, tr_len, te_len = [], [], [], []
    for i, L in enumerate(lengths):
        seg = X[bounds[i]:bounds[i + 1]]
        if i in test_ids:
            te_chunks.append(seg)
            te_len.append(L)
        else:
            tr_chunks.append(seg)
            tr_len.append(L)
    X_tr = np.concatenate(tr_chunks)
    X_te = np.concatenate(te_chunks)
    return X_tr, tr_len, X_te, te_len


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_one(X, lengths, n_states, seed=0, n_restarts=5, n_iter=200, tol=1e-3):
    """
    Fit a diagonal-covariance Gaussian HMM with several random EM restarts and
    keep the one with the highest TRAINING log-likelihood. EM is guaranteed only
    to reach a *local* optimum, so restarts are how we defend against a bad init.

    covariance_type="diag": with a 1-D signal this is just a per-state variance —
    the simplest emission that still lets noisy states (SUSTAINED_HIGH) be wider
    than quiet ones (QUIESCENT).
    """
    best_model = None
    best_ll = -np.inf
    for r in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=tol,
            random_state=seed + r,
            init_params="stmc",  # let EM initialize start/trans/means/covars
        )
        # hmmlearn can emit convergence warnings on hard inits; that's expected
        # across restarts, so we simply keep whichever restart scores best.
        model.fit(X, lengths)
        ll = model.score(X, lengths)
        if ll > best_ll:
            best_ll = ll
            best_model = model
    return best_model, best_ll


def n_free_params(n_states, n_features=1):
    """
    Free-parameter count p for BIC of a diagonal Gaussian HMM:
      - startprob : (K - 1)              (rows sum to 1)
      - transmat  : K · (K - 1)          (each row sums to 1)
      - means     : K · d
      - variances : K · d                (diagonal covariance)
    """
    K, d = n_states, n_features
    return (K - 1) + K * (K - 1) + K * d + K * d


def bic(log_likelihood, n_states, n_obs, n_features=1):
    """BIC = -2·logL + p·ln(N). Lower is better."""
    p = n_free_params(n_states, n_features)
    return -2.0 * log_likelihood + p * np.log(n_obs)


# --------------------------------------------------------------------------- #
# Model-order selection
# --------------------------------------------------------------------------- #
def select_model_order(X, lengths, k_values=(2, 3, 4, 5, 6), seed=0, n_restarts=5):
    """
    For each K, report:
      - held-out log-likelihood PER FRAME (comparable across splits/K), and
      - BIC on the full dataset.
    Returns a list of dict rows.
    """
    X_tr, len_tr, X_te, len_te = split_traces(X, lengths, test_frac=0.3, seed=seed)
    n_test_frames = sum(len_te)
    n_all_frames = sum(lengths)

    rows = []
    for K in k_values:
        # held-out: fit on train traces only, score untouched test traces
        model_tr, _ = fit_one(X_tr, len_tr, K, seed=seed, n_restarts=n_restarts)
        heldout_ll = model_tr.score(X_te, len_te)
        heldout_ll_per_frame = heldout_ll / n_test_frames

        # BIC: fit on ALL data (BIC already penalizes complexity, no split needed).
        # n_features must reflect the actual observation dimension, otherwise the
        # complexity penalty is understated and BIC silently favours bigger models.
        model_all, ll_all = fit_one(X, lengths, K, seed=seed, n_restarts=n_restarts)
        model_bic = bic(ll_all, K, n_all_frames, n_features=X.shape[1])

        rows.append({
            "K": K,
            "heldout_ll_per_frame": heldout_ll_per_frame,
            "bic": model_bic,
        })

    # Report absolute scores AND marginal gains, so the diminishing-returns knee
    # (not a nonexistent global minimum) is what justifies the chosen K.
    print(f"  {'K':>2}  {'held-out LL/frame':>18}  {'d LL/frame':>11}  "
          f"{'BIC':>12}  {'d BIC':>10}")
    prev_ll = prev_bic = None
    for r in rows:
        dll = "" if prev_ll is None else f"{r['heldout_ll_per_frame'] - prev_ll:+.4f}"
        dbic = "" if prev_bic is None else f"{r['bic'] - prev_bic:+.1f}"
        print(f"  {r['K']:>2}  {r['heldout_ll_per_frame']:>18.4f}  {dll:>11}  "
              f"{r['bic']:>12.1f}  {dbic:>10}")
        prev_ll, prev_bic = r["heldout_ll_per_frame"], r["bic"]
    return rows


def plot_selection(rows, out_png):
    """Plot held-out log-likelihood and BIC by model order."""
    Ks = [r["K"] for r in rows]
    ho = [r["heldout_ll_per_frame"] for r in rows]
    bics = [r["bic"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    c1 = "#1f77b4"
    ax1.plot(Ks, ho, "o-", color=c1, label="held-out LL / frame")
    ax1.set_xlabel("number of hidden states K")
    ax1.set_ylabel("held-out log-likelihood / frame", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(Ks)

    ax2 = ax1.twinx()
    c2 = "#d62728"
    ax2.plot(Ks, bics, "s--", color=c2, label="BIC")
    ax2.set_ylabel("BIC (lower = better)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    # mark the chosen order
    ax1.axvline(4, color="gray", ls=":", lw=1)
    ax1.text(4, ax1.get_ylim()[0], "  chosen K=4", color="gray", va="bottom")

    fig.suptitle("HMM model-order selection: why 4 states")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Save fitted parameters
# --------------------------------------------------------------------------- #
def diag_covars(model):
    """Return per-state diagonal variances as (K, n_features), whatever hmmlearn
    hands back (it fills diag covars out to (K, d, d))."""
    c = model.covars_
    if c.ndim == 3:
        return np.array([np.diag(ci) for ci in c])
    return np.asarray(c)


def save_model(model, out_npz, source_csv, heldout_rows, feature_mode):
    """Save fitted parameters as inspectable, version-stable arrays."""
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(
        out_npz,
        n_states=model.n_components,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,          # shape (K, n_features)
        covars=diag_covars(model),   # shape (K, n_features)
        feature_mode=np.array(feature_mode),
        source_csv=np.array(source_csv),
        heldout_k=np.array([r["K"] for r in heldout_rows]),
        heldout_ll_per_frame=np.array([r["heldout_ll_per_frame"] for r in heldout_rows]),
        heldout_bic=np.array([r["bic"] for r in heldout_rows]),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Fit + justify a Gaussian HMM on calcium.")
    ap.add_argument("--data", default="data/intact.csv",
                    help="CSV to fit the emission/state model on (default: intact).")
    ap.add_argument("--n_states", type=int, default=4,
                    help="final state count to fit and save (default: 4).")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", default="level+slope",
                    choices=["level+slope", "level"],
                    help="observation vector; 'level' is the ablation baseline.")
    ap.add_argument("--n_restarts", type=int, default=5,
                    help="EM restarts per fit; lower is faster, riskier init.")
    ap.add_argument("--skip_selection", action="store_true",
                    help="skip the K sweep and fit only --n_states (fast path).")
    ap.add_argument("--model_out", default="models/hmm_model.npz")
    ap.add_argument("--fig_out", default="figures/hmm_model_selection.png")
    args = ap.parse_args()

    print(f"Loading calcium traces from {args.data} (calcium only; true_state untouched)...")
    X, lengths = load_traces(args.data, args.features)
    print(f"  {len(lengths)} traces, {sum(lengths)} total frames, "
          f"features = {args.features} ({X.shape[1]}-D).\n")

    if args.skip_selection:
        # Fast path for reruns/tests. The K-sweep is the 'why 4 states' evidence
        # and should NOT be skipped when producing results for publication.
        print("Skipping model-order selection (--skip_selection).")
        rows = []
    else:
        print("Model-order selection (fitting K = "
              f"{args.k_min}..{args.k_max}); this is the 'why 4 states' evidence:")
        k_values = tuple(range(args.k_min, args.k_max + 1))
        rows = select_model_order(X, lengths, k_values=k_values, seed=args.seed,
                                  n_restarts=args.n_restarts)
        os.makedirs(os.path.dirname(args.fig_out), exist_ok=True)
        plot_selection(rows, args.fig_out)
        print(f"\nSaved model-order figure -> {args.fig_out}")

    print(f"\nFitting FINAL {args.n_states}-state model on all data...")
    final_model, final_ll = fit_one(X, lengths, args.n_states, seed=args.seed,
                                    n_restarts=args.n_restarts)
    print(f"  final training LL = {final_ll:.1f}")

    # Report the learned states sorted by mean calcium — a first sanity check that
    # the model found distinct, interpretable levels (Module 2 assigns biological
    # labels formally).
    means = final_model.means_[:, CALCIUM_COL]
    covs = diag_covars(final_model)
    order = np.argsort(means)
    print("  learned emission means (sorted, low -> high calcium):")
    for rank, s in enumerate(order):
        sd = np.sqrt(covs[s, CALCIUM_COL])
        extra = ""
        if X.shape[1] > 1:  # report the slope dimension too, when present
            extra = f"   slope_mean={final_model.means_[s, 1]:+.4f}"
        print(f"    state {s}:  mean={means[s]:.3f}  sd={sd:.3f}{extra}")

    save_model(final_model, args.model_out, args.data, rows, args.features)
    print(f"\nSaved fitted {args.n_states}-state model -> {args.model_out}")
    print("Module 1 complete. Next: recover_states.py (Viterbi decode + validate).")


if __name__ == "__main__":
    main()
