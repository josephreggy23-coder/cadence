# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Glial calcium dysregulation drives epilepsy, stroke, and neurodegeneration, yet
neuromodulation remains open-loop and neuron-centric, delivering fixed
stimulation with no principled account of when or how much to intervene. I asked
whether astrocyte calcium obeys a recoverable load-dependent feedback law,
P(high→refractory | L) = sigmoid(b0 + b1·L), and whether a controller acting
through that law could restore rhythm minimally. Validating against a simulator
with known ground truth, I found the standard approach misspecified: a hidden
Markov model assumes fluorescence reflects only the current state, but GCaMP
integrates state history, so refractory frames — which follow high calcium — are
systematically misread as oscillatory. Feature engineering and constrained
transitions both failed, one collapsing accuracy from 69% to 45%. Modelling the
sensor explicitly, by tracking the joint state of dynamical regime and sensor
level, raised refractory recall from 38% to 82% and recovered the simulator's
hidden emission means and time constant from unlabelled data. This corrected
everything downstream: recovered b1 rose from +0.208 to +0.832 against a true
+0.9. It also exposed a subtler failure — an accurate law learned from healthy
cells leaves the controller silent on diseased ones, because feedback gain
transfers across disease while baseline propensity does not — so the controller
now calibrates that baseline online from the cell it treats, restoring rhythm at
89% below open-loop cost. Testing the blocked condition revealed a safety flaw:
against a dead pathway the controller escalated dose indefinitely, spending more
than continuous stimulation for no benefit, until a futility interlock was added.
Crucially, when feedback is blocked CADENCE fails to restore, confirming it acts
through the endogenous law rather than brute force; a controller that still
worked would falsify the premise. Work is in silico; wet-lab validation follows.
