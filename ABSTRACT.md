# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Glial calcium dysregulation drives epilepsy, stroke, and neurodegeneration, yet
existing neuromodulation is open-loop and neuron-centric: it delivers fixed
stimulation with no principled account of when or how much to intervene, risking
reactive astrogliosis. I asked whether astrocyte calcium is governed by a
recoverable, load-dependent negative-feedback law, and whether a controller
acting *through* that law could restore rhythm minimally.

I model calcium as four hidden states and estimate the feedback as
P(high→refractory | L) = sigmoid(b0 + b1·L), where L accumulates time spent activated; b1 > 0 is the feedback signature. Confidence intervals use
a cluster bootstrap over cells, since frames are autocorrelated.

The decisive result was recognising that a hidden Markov model is misspecified
here. It assumes fluorescence depends only on the current state, but a GCaMP
sensor integrates state *history*, so refractory frames — which follow high
calcium — are misread as oscillatory. Feature engineering and constrained
transitions both failed; one made accuracy collapse. Modelling the sensor
explicitly, by tracking the joint state of regime and sensor level, raised
refractory recall from 38% to 82% and recovered the simulator's hidden emission
means and time constant from unlabelled data.

That corrected estimation: recovered b1 rose from +0.208 to +0.832 against a true
+0.9. It also revealed a subtler problem: an accurate healthy-cell law makes the
controller stay silent on diseased ones, because feedback gain transfers across
disease but baseline propensity does not. Calibrating that
baseline online restored function at 89% below open-loop cost. Testing the
blocked condition exposed a safety failure: against a dead pathway the controller
escalated dose indefinitely for no benefit, until a futility interlock was added.

Under blockade CADENCE correctly fails to restore, confirming it works through
the endogenous law. Work is in silico; wet-lab validation follows.
