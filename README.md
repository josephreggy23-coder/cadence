# CADENCE

**Calcium Adaptive Dynamics Engine for Controlled Equilibrium**

CADENCE is an early-stage research prototype for studying model-based,
closed-loop control of astrocyte-like calcium dynamics. It begins with a
transparent synthetic system whose feedback law is known, so every later
inference or control step can be checked against ground truth.

## The idea

Glial calcium dysregulation is relevant to epilepsy, stroke, and
neurodegeneration. CADENCE models calcium activity as a four-state process:
`QUIESCENT`, `OSCILLATORY`, `SUSTAINED_HIGH`, and `REFRACTORY`.

The key mechanism is load-dependent negative feedback. The longer the system
remains in a sustained high-calcium state, the more likely it is to transition
to a refractory state:

`P(high -> refractory | load) = sigmoid(beta0 + beta1 * load)`

The intact condition uses a positive `beta1`; the blocked control makes it
nearly flat. This lets the project test whether a future estimator and
controller respond to the intended feedback pathway rather than a shortcut.

## What works today

- Deterministic synthetic trace generation for intact and feedback-blocked conditions
- Reusable `generate_dataset(...)` Python API for later fitting and validation stages
- Dataset-contract validation before output is written, protecting downstream fitting code
- Validated `load_dataset(...)` API for safely reusing generated CSV files
- Configurable trace count, duration, random seed, and load retention
- CSV output with condition, trace identifier, time, calcium signal, hidden state, and accumulated load
- Console summary of the hidden-state occupancy for a quick sanity check
- Unit tests for simulator input validation and continuous integration on GitHub Actions

## Quickstart

```bash
python -m pip install -r requirements.txt
python generate_synthetic.py --condition intact --n_traces 60 --out data/intact.csv
python generate_synthetic.py --condition blocked --n_traces 60 --out data/blocked.csv
python -m unittest discover -s tests -v
```

Each CSV is accompanied by a JSON manifest (for example, `data/intact.json`)
that records the random seed, feedback parameters, duration, and load-decay
setting used to generate it. Use `--metadata-out` to choose a different path.

## Real zebrafish calcium data

CADENCE now includes a loader for a real `Danio rerio` calcium-imaging asset
from [DANDI:001076](https://dandiarchive.org/dandiset/001076):
`sub-nan_ses-20230123T192927_obj-17bhudf_ophys.nwb`. The selected public asset
contains a fluorescence response matrix with 1,416 frames and 667 ROIs.

```python
from real_data import load_zebrafish_recording

recording = load_zebrafish_recording(
    "data/real/dandi-001076-zebrafish-ophys.nwb",
    max_rois=64,
)
```

`summarize_recording(recording)` returns label-free quality-control metrics
(duration, ROI count, fluorescence level, and per-ROI variability) before any
model fitting is attempted.

The file is ignored by Git because it is source data, not project code. This
asset is an open, single-subject DANDI draft and is suitable for exploratory
ingestion work only. It does not yet validate CADENCE's control hypothesis;
the synthetic system remains the ground-truth benchmark for estimator tests.

For sensitivity experiments, `--load-decay` controls how quickly accumulated
high-calcium load fades between frames. It defaults to `0.92`; values must be
between `0` and `1`.

### Output schema

| Column | Meaning |
| --- | --- |
| `condition` | `intact` or feedback-`blocked` simulation condition |
| `trace_id` | Identifier for an independently simulated trace |
| `time_s` | Frame time in seconds |
| `calcium` | Smoothed, noisy calcium observation |
| `true_state` | Ground-truth hidden-state index used to generate the frame |
| `load` | Accumulated high-calcium load used by the feedback law |

## Repository structure

```
cadence/
+-- generate_synthetic.py   # ground-truth simulator
+-- requirements.txt        # minimal runtime dependencies
+-- tests/                  # simulator tests
+-- docs/                   # design and validation notes
+-- examples/               # example usage material
```

## Validation path

1. Recover the known feedback law from synthetic data.
2. Show that a model-based controller restores healthy dynamics in a disrupted simulation.
3. Verify the controller fails when the feedback pathway is blocked.

The final failure condition is important: a controller that still succeeds
under the blocked control would not demonstrate that it works through the
proposed biological mechanism.

## Roadmap

- [x] Ground-truth synthetic simulator with load-dependent feedback
- [ ] HMM fitting and model-order selection
- [ ] Hidden-state recovery against synthetic ground truth
- [ ] Feedback-law estimation with uncertainty intervals
- [ ] Model-based intervention policy in silico
- [ ] In-silico restoration and blocked-feedback control
- [ ] Fit and control on real glial calcium recordings

## Scope

CADENCE is research software, not a clinical device or a validated biological
controller. Its current output is synthetic and intended for estimator and
workflow validation.
