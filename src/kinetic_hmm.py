"""Sensor-aware switching model for synthetic calcium traces.

The observation model includes exponential sensor dynamics,

    c[t] = c[t-1] + (mu[s[t]] - c[t-1]) / tau
    y[t] = c[t] + noise,

so fluorescence depends on both the latent regime and prior sensor level. The
continuous sensor state is discretized on a grid and combined with the four
regimes for forward-backward inference. Profile likelihood selects ``tau``;
EM updates the transition matrix, latent levels, and observation noise.

Internal states are ordered by latent calcium level and mapped to the simulator's
label convention only during evaluation.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Internal state order is ASCENDING CALCIUM, which for this system is
#   0 = QUIESCENT, 1 = REFRACTORY, 2 = OSCILLATORY, 3 = SUSTAINED_HIGH
# The biological index convention used by true_state / the rest of the pipeline
# is 0=QUIESCENT, 1=OSCILLATORY, 2=SUSTAINED_HIGH, 3=REFRACTORY.
RANK_TO_TRUE = np.array([0, 3, 1, 2])
STATE_NAMES = ["QUIESCENT", "OSCILLATORY", "SUSTAINED_HIGH", "REFRACTORY"]

NC_DEFAULT = 100


class KineticHMM:
    """Switching state-space model with an explicit exponential sensor."""

    def __init__(self, mu, sd, tau, A=None, pi=None, nc=NC_DEFAULT,
                 c_lo=-0.2, c_hi=1.2):
        self.mu = np.asarray(mu, float)
        self.sd = np.asarray(sd, float)
        self.tau = float(tau)
        self.K = len(self.mu)
        self.A = np.full((self.K, self.K), 1.0 / self.K) if A is None else np.asarray(A, float)
        self.pi = np.full(self.K, 1.0 / self.K) if pi is None else np.asarray(pi, float)
        self.cgrid = np.linspace(c_lo, c_hi, nc)
        self.nc = nc
        self._rebuild_map()

    # ---------------------------------------------------------------- #
    def _rebuild_map(self):
        """cmap[s, i] = grid index of the sensor after one step into state s."""
        self.cmap = np.empty((self.K, self.nc), dtype=int)
        for s in range(self.K):
            nxt = self.cgrid + (self.mu[s] - self.cgrid) / self.tau
            self.cmap[s] = np.clip(np.searchsorted(self.cgrid, nxt), 0, self.nc - 1)

    def _emission(self, y):
        """Return rescaled emission probabilities and their log offsets.

        The forward recursion needs the offsets to recover the log-likelihood
        after per-frame rescaling.
        """
        z = (y[:, None, None] - self.cgrid[None, None, :]) / self.sd[None, :, None]
        logB = -0.5 * z * z - np.log(self.sd)[None, :, None]
        offset = logB.max(axis=(1, 2), keepdims=True)
        return np.exp(logB - offset), offset.reshape(-1)

    # ---------------------------------------------------------------- #
    def forward_backward(self, y, want_xi=False):
        """
        Exact inference over the joint (state, sensor-grid) space.

        Returns (posterior over discrete states per frame, loglik, xi_sum),
        where xi_sum is the expected transition-count matrix for the EM M-step.
        """
        T = len(y)
        B, log_offset = self._emission(y)
        K, NC = self.K, self.nc

        alpha = np.empty((T, K, NC))
        a = (self.pi[:, None] / NC) * B[0]
        z0 = a.sum()
        a /= z0
        alpha[0] = a
        loglik = np.log(z0) + log_offset[0]

        for t in range(1, T):
            pre = np.einsum("pi,pt->ti", alpha[t - 1], self.A)
            new = np.zeros((K, NC))
            for s in range(K):
                np.add.at(new[s], self.cmap[s], pre[s])
            new *= B[t]
            z = new.sum()
            if z <= 0:
                new[:] = 1.0 / (K * NC)
                z = 1.0
            new /= z
            alpha[t] = new
            loglik += np.log(z) + log_offset[t]

        beta = np.empty((T, K, NC))
        beta[-1] = 1.0
        xi_sum = np.zeros((K, K))
        for t in range(T - 2, -1, -1):
            nb = beta[t + 1] * B[t + 1]  # rescaling cancels in the normalised xi
            gathered = np.empty((K, NC))
            for s in range(K):
                gathered[s] = nb[s][self.cmap[s]]
            if want_xi:
                # xi[p, s] = sum_i alpha[t][p, i] * A[p, s] * gathered[s, i]
                xi = self.A * (alpha[t] @ gathered.T)
                tot = xi.sum()
                if tot > 0:
                    xi_sum += xi / tot
            b = np.einsum("ps,si->pi", self.A, gathered)
            mx = b.max()
            beta[t] = b / mx if mx > 0 else b
        post = alpha * beta
        post_s = post.sum(axis=2)
        post_s /= np.maximum(post_s.sum(axis=1, keepdims=True), 1e-300)
        return post_s, loglik, xi_sum

    def loglik(self, sequences):
        return sum(self.forward_backward(y)[1] for y in sequences)

    def decode(self, y):
        """MAP discrete state per frame, in BIOLOGICAL index convention."""
        post_s, _, _ = self.forward_backward(y)
        return RANK_TO_TRUE[post_s.argmax(axis=1)]

    # ---------------------------------------------------------------- #
    def fit(self, sequences, n_iter=12, opt_every=3, verbose=True):
        """
        Alternating EM:
          E-step  : exact forward-backward over (state, sensor)
          M-step A: closed form from expected transition counts
          M-step continuous: numerical optimisation of (mu, sd, tau) on the
                             marginal likelihood, every `opt_every` iterations
                             (it is far more expensive than the A update)
        """
        prev = -np.inf
        for it in range(n_iter):
            xi_tot = np.zeros((self.K, self.K))
            ll = 0.0
            for y in sequences:
                _, l, xi = self.forward_backward(y, want_xi=True)
                ll += l
                xi_tot += xi
            xi_tot += 1e-6
            self.A = xi_tot / xi_tot.sum(axis=1, keepdims=True)

            if verbose:
                print(f"    EM iter {it + 1:2d}  loglik = {ll:12.1f}")
            if it and abs(ll - prev) < 1e-3 * abs(prev):
                break
            prev = ll

            if opt_every and (it + 1) % opt_every == 0 and it + 1 < n_iter:
                self._optimise_continuous(sequences[:max(1, len(sequences) // 3)])
        return self

    def _optimise_continuous(self, sequences):
        """
        Numerically refine (mu, sd, tau) on the marginal likelihood.

        Parameterised so the optimiser cannot wander into invalid regions:
        mu is kept sorted-ascending via cumulative softplus offsets, sd and tau
        are optimised in log space. Nelder-Mead because the grid discretisation
        makes the objective mildly non-smooth.
        """
        def unpack(p):
            mu0 = p[0]
            gaps = np.exp(p[1:4])                 # strictly positive gaps
            mu = np.concatenate([[mu0], mu0 + np.cumsum(gaps)])
            sd = np.exp(p[4:8])
            tau = 1.0 + np.exp(p[8])              # tau > 1 frame
            return mu, sd, tau

        # Guard rails. With the likelihood computed correctly these should not
        # bind, but a runaway sd/tau is the signature of exactly the scaling bug
        # fixed above, so we keep them as a tripwire rather than trusting silence.
        SD_MAX, TAU_MAX = 0.40, 10.0

        def pack():
            gaps = np.diff(self.mu)
            gaps = np.maximum(gaps, 1e-3)
            return np.concatenate([[self.mu[0]], np.log(gaps),
                                   np.log(self.sd), [np.log(max(self.tau - 1, 1e-3))]])

        mu0, sd0, tau0, A0 = self.mu.copy(), self.sd.copy(), self.tau, self.A.copy()

        def nll(p):
            mu, sd, tau = unpack(p)
            if not np.all(np.isfinite(mu)) or mu[-1] > 3.0 or mu[0] < -0.5:
                return 1e12
            self.mu, self.sd, self.tau = mu, np.clip(sd, 1e-3, SD_MAX), min(tau, TAU_MAX)
            self._rebuild_map()
            try:
                return -self.loglik(sequences)
            except Exception:  # noqa: BLE001
                return 1e12

        res = minimize(nll, pack(), method="Nelder-Mead",
                       options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-2})
        if np.isfinite(res.fun):
            self.mu, self.sd, self.tau = unpack(res.x)
            self.sd = np.clip(self.sd, 1e-3, SD_MAX)
            self.tau = min(self.tau, TAU_MAX)
        else:
            self.mu, self.sd, self.tau = mu0, sd0, tau0
        self.A = A0
        self._rebuild_map()

    # ---------------------------------------------------------------- #
    def save(self, path, *, fit_trace_count=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, model_type=np.array("kinetic"), mu=self.mu, sd=self.sd,
                 tau=np.array(self.tau), A=self.A, pi=self.pi,
                 nc=np.array(self.nc),
                 c_lo=np.array(self.cgrid[0]), c_hi=np.array(self.cgrid[-1]),
                 fit_trace_count=np.array(
                     -1 if fit_trace_count is None else int(fit_trace_count)
                 ))

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        return KineticHMM(d["mu"], d["sd"], float(d["tau"]), A=d["A"], pi=d["pi"],
                          nc=int(d["nc"]), c_lo=float(d["c_lo"]), c_hi=float(d["c_hi"]))


class KineticFilter:
    """
    Causal online filter for the controller.

    This is the same forward recursion as `forward_backward`, exposed one frame
    at a time. It is genuinely causal - the belief at frame t uses only frames
    1..t - which removes the one-frame latency hack the Gaussian estimator needed
    (that model's centred-derivative feature peeked at t+1, so the controller had
    to act on a delayed estimate to keep the online features matched to training).
    Modelling the sensor properly makes the workaround unnecessary.
    """

    def __init__(self, model):
        self.m = model
        self.reset()

    def reset(self):
        self.alpha = None

    def step(self, y):
        """Feed one calcium sample; return the MAP biological state for THIS frame."""
        m = self.m
        z = (y - m.cgrid) / m.sd[:, None]
        logB = -0.5 * z * z - np.log(m.sd)[:, None]
        B = np.exp(logB - logB.max())

        if self.alpha is None:
            a = (m.pi[:, None] / m.nc) * B
        else:
            pre = np.einsum("pi,pt->ti", self.alpha, m.A)
            a = np.zeros((m.K, m.nc))
            for s in range(m.K):
                np.add.at(a[s], m.cmap[s], pre[s])
            a *= B
        tot = a.sum()
        if tot <= 0:
            a = np.ones((m.K, m.nc)) / (m.K * m.nc)
            tot = 1.0
        self.alpha = a / tot
        return int(RANK_TO_TRUE[self.alpha.sum(axis=1).argmax()])


# --------------------------------------------------------------------------- #
def load_sequences(csv_path, n_traces=None):
    df = pd.read_csv(csv_path)
    if n_traces:
        df = df[df.trace_id < n_traces]
    seqs = []
    for _, g in df.groupby("trace_id", sort=True):
        seqs.append(g.sort_values("time_s")["calcium"].to_numpy())
    return seqs


def initial_model(sequences, tau0=3.0):
    """
    Data-driven initialisation - no ground truth used.

    Latent means start at spread quantiles of the observed signal. Long dwells in
    the extreme states let the sensor equilibrate, so low/high quantiles are
    reasonable proxies for the extreme latent levels; the middle two are simply
    placed between them and refined by the optimiser.
    """
    y = np.concatenate(sequences)
    qs = np.quantile(y, [0.05, 0.25, 0.55, 0.97])
    sd0 = np.full(4, max(np.std(y) * 0.35, 0.02))
    A0 = np.full((4, 4), 0.05) + np.eye(4) * 0.8
    A0 /= A0.sum(axis=1, keepdims=True)
    return KineticHMM(qs, sd0, tau0, A=A0)


def select_tau(sequences, grid=(1.5, 2.0, 2.5, 3.0, 4.0, 5.0), verbose=True):
    """Profile likelihood over the sensor time constant - never assumed."""
    best, best_ll = None, -np.inf
    for tau in grid:
        m = initial_model(sequences, tau0=tau)
        m.fit(sequences[:6], n_iter=3, opt_every=0, verbose=False)
        ll = m.loglik(sequences[:6])
        if verbose:
            print(f"    tau = {tau:4.1f}  loglik = {ll:10.1f}")
        if ll > best_ll:
            best, best_ll = tau, ll
    return best


def main():
    ap = argparse.ArgumentParser(description="Fit the kinetic (sensor-aware) model.")
    ap.add_argument("--data", default="data/intact.csv")
    ap.add_argument("--fit_traces", type=int, default=20,
                    help="traces used for fitting (inference is O(T*K*NC)).")
    ap.add_argument("--n_iter", type=int, default=12)
    ap.add_argument("--out", default="models/kinetic_model.npz")
    args = ap.parse_args()

    print(f"Loading {args.data} (calcium only; true_state untouched)...")
    seqs = load_sequences(args.data, n_traces=args.fit_traces)
    print(f"  {len(seqs)} traces, {sum(len(s) for s in seqs)} frames.\n")

    print("Selecting sensor time constant tau by profile likelihood:")
    tau = select_tau(seqs)
    print(f"  selected tau = {tau}  (generator used 3.0)\n")

    print("Fitting kinetic model (EM over the joint state/sensor space):")
    model = initial_model(seqs, tau0=tau)
    model.fit(seqs, n_iter=args.n_iter)

    print("\n  fitted latent means (ascending = QUIESCENT, REFRACTORY, "
          "OSCILLATORY, SUSTAINED_HIGH):")
    for r, name in enumerate(["QUIESCENT", "REFRACTORY", "OSCILLATORY", "SUSTAINED_HIGH"]):
        print(f"    {name:<16} mu={model.mu[r]:6.3f}  sd={model.sd[r]:5.3f}")
    print(f"  fitted tau = {model.tau:.2f}")

    model.save(args.out, fit_trace_count=len(seqs))
    print(f"\nSaved kinetic model -> {args.out}")


if __name__ == "__main__":
    main()
