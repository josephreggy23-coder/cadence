# CADENCE Tutorial

This walkthrough takes you from a fresh checkout to the main CADENCE results.
It also explains what each stage does, what files it creates, and what the
results do and do not mean.

## 1. Set up Python

CADENCE is tested with Python 3.11 and works with newer compatible Python
versions. From the repository root, create an isolated environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment keeps CADENCE's scientific packages separate from the rest of
your computer.

## 2. Run the complete compact pipeline

```bash
python src/run_all.py --quick --n_traces 30 --fit_traces 10 --control_traces 6
```

This command runs the project's five main stages in order:

1. Generate labelled synthetic calcium recordings.
2. Fit a sensor-aware four-state model.
3. Recover hidden states and estimate the feedback law.
4. Stress-test the CADENCE controller in simulation.
5. Reproduce the exploratory public astrocyte-data analysis.

The compact run normally takes a few minutes on a laptop. Existing result files
are replaced with deterministic reproductions.

## 3. Understand the synthetic data

The simulator represents four latent states:

| State | Meaning inside the simulator |
| --- | --- |
| `QUIESCENT` | Low baseline activity |
| `OSCILLATORY` | Moderate calcium activity |
| `SUSTAINED_HIGH` | Prolonged high activity |
| `REFRACTORY` | A temporary suppressed or recovery state |

The state labels are modeling choices for a controlled benchmark. They are not
claims that real astrocytes always occupy four discrete biological states.

Two datasets are generated:

- `data/intact.csv`: recent high-state exposure raises the probability of
  entering the refractory state.
- `data/blocked.csv`: that exposure-dependent relationship is nearly flat.

Every row contains a trace identifier, time, observed calcium-like
fluorescence, the simulator's true state, and the simulator's exposure value.
The true labels let the project measure whether state inference works.

To generate a smaller dataset by itself:

```bash
python src/generate_synthetic.py --condition intact --n_traces 5 --out data/intact_demo.csv
```

## 4. Understand the state model

Real calcium indicators have memory: the fluorescence signal fades gradually
after calcium changes. A model that ignores this can mistake sensor decay for a
new biological state.

`src/kinetic_hmm.py` fits a hidden-state model that includes this sensor
response. It selects a time constant, estimates the signal level and noise of
each state, and writes the fitted parameters to:

```text
models/kinetic_model.npz
```

`src/recover_states.py` then evaluates two forms of inference:

- Offline-smoothed inference can use the complete trace, including later data.
- Causal-filtered inference can use only the current and previous frames.

The causal result is the relevant one for a future real-time controller. The
offline result is useful as an upper-bound diagnostic.

## 5. Understand the feedback estimate

CADENCE asks whether recent inferred high-state exposure predicts an exit from
the high state. The fitted relationship is:

```text
P(exit from high state) = sigmoid(b0 + b1 * exposure)
```

`b1` is the important contrast:

- A positive `b1` means exit probability increases with recent exposure.
- A value near zero means the relationship is approximately flat.

`src/estimate_feedback_law.py` estimates this slope for the intact and blocked
conditions. It resamples whole traces to calculate uncertainty without treating
every frame as an independent experiment.

Results are written to `results/feedback_estimates.json` and visualized in
`figures/feedback_law.png`.

Because the simulator deliberately contains this contrast, recovering it is a
software and estimator validation. It is not evidence that the same law exists
in living astrocytes.

## 6. Understand the controller benchmark

`src/controller.py` compares several policies on a shifted synthetic plant:

- No control provides the untreated reference.
- Open-loop control applies a fixed schedule.
- Learned-law CADENCE uses the estimated synthetic relationship.
- Plant-parameter reference uses privileged simulator parameters as a
  diagnostic upper reference.
- Self-calibrating CADENCE updates its baseline estimate from unstimulated
  transitions.

The benchmark reports high-state occupancy and cumulative intervention cost.
It also enforces a hard intervention budget.

In the structurally blocked simulation, the same coefficient controls both
natural recovery and intervention coupling. Setting it near zero therefore
disables CADENCE by construction. This is a model consistency check, not
independent proof of a biological mechanism.

Results appear in `results/controller_benchmark.json` and the
`figures/control_*.png` files.

## 7. Understand the real-data analysis

`src/analyze_h1r_astrocytes.py` reproduces an exploratory secondary analysis of
public mouse astrocyte recordings. It aggregates regions of interest within
each slice before comparing paired slice-level signals. It also runs
offset-free and specification-sensitivity checks.

These recordings provide biological context, but they do not contain the
animal identifiers or intervention design needed to validate the CADENCE
controller. The synthetic and real-data analyses therefore answer different
questions and are not combined into a treatment claim.

Detailed methods are in [real_data.md](real_data.md).

## 8. Run the tests

```bash
python -m unittest discover -s tests -v
python tests/test_pipeline.py
```

The first command checks individual functions, validation rules, provenance,
and analysis behavior. The second checks the documented end-to-end synthetic
claims against the versioned benchmark artifacts.

Passing tests show that the current code reproduces its intended computational
behavior. They do not establish clinical safety, treatment effectiveness, or
biological truth.

## 9. Find the important outputs

| Location | Contents |
| --- | --- |
| `data/` | Generated and processed datasets |
| `models/` | Fitted sensor-aware model |
| `results/` | Machine-readable estimates and benchmarks |
| `figures/` | State, feedback, controller, and public-data plots |
| `docs/` | Methods, references, limitations, and reproducibility notes |
| `tests/` | Unit and scientific-invariant checks |

For a first pass, read `results/feedback_estimates.json`, inspect
`figures/state_recovery_confusion.png`, and then compare the intact and blocked
controller figures. Those three views tell the main computational story.
