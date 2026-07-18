# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Glial calcium dysregulation drives epilepsy, stroke, and neurodegeneration, yet
existing neuromodulation is open-loop and neuron-centric: it delivers fixed
stimulation with no principled account of when or how much to intervene, risking
reactive astrogliosis. I asked whether astrocyte calcium is
governed by a recoverable, load-dependent negative-feedback law, and whether a
controller acting through that law could restore healthy rhythm with minimal
intervention.

I model calcium as a four-state hidden process (quiescent, oscillatory,
sustained-high, refractory). Because the refractory "off" state emits calcium
only 0.05 dF/F above baseline — below the noise floor — level alone recovers it
at 24% recall. Observing that refractory is not low but mid-level and *falling*,
I added the signal's temporal derivative to the emission, raising sustained-high
recall from 78% to 93%. Feedback is modeled explicitly against a causal
exponential "calcium load" L, with P(high→refractory | L) = sigmoid(b0 + b1·L);
b1 > 0 is the feedback signature. Confidence intervals use a cluster bootstrap
over cells, since frames are autocorrelated.

On synthetic data from a known law, the estimator recovers b1 = +1.09
(truth +0.9) for intact feedback and +0.02 (truth +0.02) when feedback is
blocked. End-to-end, state-estimation error attenuates intact b1 fivefold and
drives blocked b1 spuriously negative — so absolute b1 is not yet trustworthy,
though the intact-blocked contrast holds decisively (p < 0.0001).
Refractory recovery (38%) is therefore the identified bottleneck, not a rounding
error.

The controller, under construction, will be benchmarked against no-control and
open-loop stimulation on pathological time and cumulative intervention cost.
The design includes its own falsification test: when feedback
is blocked, a controller acting through the endogenous law must *fail* to
restore rhythm. A controller that still succeeded would indicate brute force,
not mechanism. Work is in silico; wet-lab validation follows.
