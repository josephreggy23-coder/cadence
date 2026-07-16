# CADENCE

**C**alcium **A**daptive **D**ynamics **EN**gine for **C**ontrolled **E**quilibrium

A model-based controller that restores healthy rhythm to astrocyte-like glial
calcium dynamics by learning the system's own feedback grammar and intervening
minimally, at exactly the right moment.

> The name is the point: healthy glia keep calcium in a steady *cadence*. In
> disease that rhythm breaks. CADENCE reads the rhythm, predicts when it is about
> to fail, and restores it with the smallest effective nudge — not brute-force
> stimulation.

## The problem

Glial calcium dysregulation drives major CNS disorders (epilepsy, stroke,
neurodegeneration). Current glial neuromodulation is **open-loop and
neuron-centric**: it applies fixed stimulation and hopes for the right calcium
response, with no principled way to know *when* or *how much* to intervene. That
wastes energy, risks pushing astrocytes into reactive/cytotoxic states, and
cannot adapt to a system that is actively failing.

## The solution

CADENCE is a closed-loop controller whose intelligence lives in a **generative
model of the glia's own regulatory rules**, not in a fixed threshold. It:

1. **Reads** a live calcium trace and infers the hidden dynamical state
   (`QUIESCENT`, `OSCILLATORY`, `SUSTAINED_HIGH`, `REFRACTORY`).
2. **Predicts** how close the system is to losing self-regulation, using the
   learned load-dependent feedback law
   `P(high → refractory | L) = sigmoid(b0 + b1·L)`.
3. **Prescribes** the minimal, best-timed intervention to steer calcium back to a
   healthy cadence, working *with* the endogenous feedback rather than overriding
   it.

Because the controller acts through the system's own law, it is minimal and
well-timed by design — the key advantage over open-loop blasting.

## Why this is new

Existing closed-loop neuromodulation is reactive and neuron-focused: sense a
threshold crossing, fire a fixed stimulus. CADENCE is **predictive and
model-based, targeting glia**: it uses a formal model of astrocyte calcium's
regulatory grammar to anticipate failure and choose the smallest effective
action. No existing tool controls *glial* calcium through a learned feedback law.

## The validation logic (built-in kill-shot)

- **Recovery:** CADENCE's model must recover a known feedback law from synthetic
  data before it is trusted on real recordings (estimator validation).
- **Restoration:** tuned on the healthy feedback law, the controller should
  restore healthy calcium dynamics in a disrupted (disease-model) system.
- **Kill-shot:** if the feedback pathway is pharmacologically blocked, CADENCE
  should *fail* to restore — proving it works **through** the endogenous law, not
  by brute force. A controller that still "works" with feedback blocked would be
  a red flag, and testing for that is the honest control.

## Repository structure

```
cadence/
├── README.md
├── src/
│   ├── generate_synthetic.py     # ground-truth simulator (known feedback law)
│   ├── plot_example.py           # example trace + the target law
│   ├── fit_hmm.py                # fit 4-state HMM to calcium
│   ├── recover_states.py         # decode + validate vs ground truth
│   ├── estimate_feedback_law.py  # recover b1 (feedback strength) per condition
│   └── controller.py             # CADENCE: model-based control policy
├── data/                         # synthetic csvs (regenerable)
├── figures/
└── tests/                        # asserts feedback recovered; controller restores
```

## Quickstart

```bash
python -m pip install -r requirements.txt
python generate_synthetic.py --condition intact  --n_traces 60 --out data/intact.csv
python generate_synthetic.py --condition blocked --n_traces 60 --out data/blocked.csv
```

Additional scientific packages used by the planned fitting and visualization
steps are introduced with those components, rather than required to generate
and inspect the synthetic ground-truth data.

For sensitivity experiments, `--load-decay` controls how quickly accumulated
high-calcium load fades between frames. It defaults to `0.92`; values must be
between `0` and `1`.

## Roadmap

- [x] Ground-truth synthetic simulator with load-dependent feedback
- [ ] HMM fitting + model-order selection (held-out likelihood, BIC)
- [ ] Hidden-state recovery validated against ground truth
- [ ] Feedback-law estimation (`b1` with CI): intact vs blocked
- [ ] CADENCE controller: model-based intervention policy (in silico)
- [ ] In-silico restoration + kill-shot (blocked feedback ⇒ no restoration)
- [ ] Wet-lab: fit + control on real glial calcium recordings

## Known limitations (addressed, not hidden)

- A plain HMM assumes memoryless transitions; the feedback is load-dependent
  (semi-Markov). Handled by estimating the feedback as an explicit function of
  accumulated load, not a constant transition matrix.
- Hidden-state models can overfit; defended with held-out likelihood, BIC, and
  the requirement that recovered states map to interpretable calcium levels.
- In-silico control is a model of control, not proof in tissue; the wet-lab step
  is what closes that gap, and the pipeline runs unchanged on real recordings.


