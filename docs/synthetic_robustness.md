# Multi-seed synthetic robustness benchmark

This document records a sensitivity check across independent randomizations of
the same synthetic simulator. It is not an independent dataset, a confidence
interval across biological samples, or validation of an astrocyte mechanism.

## Fixed design

The benchmark repeats the published compact synthetic configuration five times:

| Setting | Value |
| --- | ---: |
| Base seeds | `0, 1, 2, 3, 4` |
| Generator streams | intact = `2 × base seed`; blocked = `2 × base seed + 1` |
| Traces per condition | 30 |
| Frames per trace | 600 |
| Intact model-fit traces | 10 |
| Intact held-out scoring traces | 20 |
| Kinetic-model EM iterations | 12 |
| Exposure decay | 0.90, fixed across conditions |
| Bootstrap draws per arm | 150 whole-trace resamples |

Each replicate refits the kinetic state model from its own intact training
traces. It then scores causal states only on the held-out intact traces and
estimates the intact-minus-blocked causal `b1` contrast with independent,
whole-trace bootstrap streams. No intermediate CSV, model, or decoded-state
artifact is reused across seeds.

## Versioned result

Run:

```bash
python src/seed_robustness.py
```

The versioned output is
[`results/seed_robustness.json`](../results/seed_robustness.json), with a
per-seed figure at
[`figures/seed_robustness.png`](../figures/seed_robustness.png).

| Endpoint | Median across five seeds | Range across five seeds |
| --- | ---: | ---: |
| Held-out offline-smoothed accuracy | 85.5% | 83.0–89.0% |
| Held-out causal-filter accuracy | 79.6% | 76.0–83.4% |
| Held-out causal refractory recall | 70.7% | 47.0–73.1% |
| Causal intact-minus-blocked `b1` | +0.725 | +0.637 to +0.891 |

All five seed-specific causal contrasts were positive, and every one of their
trace-bootstrap 95% intervals excluded zero. These are descriptive consistency
checks over fixed simulator randomizations, not pooled inferential tests.

## Interpretation boundary

The causal blocked-arm estimate was negative in all five seed runs. This is
consistent with the hard-label null bias documented in the reference run, and
is why CADENCE treats the intact-minus-blocked contrast—not the blocked slope
alone—as the relevant synthetic endpoint. This analysis does not eliminate the
need for parameter-mismatch, drift, indicator, and animal-identified biological
validation.
