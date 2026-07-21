# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Astrocytes maintain the calcium rhythms that keep neural circuits stable, and
their disruption contributes to epilepsy, stroke, and neurodegeneration. Existing
neuromodulation is open-loop, delivering fixed stimulation with no principled
basis for deciding when or how much to intervene, which wastes energy and can
drive astrocytes into harmful reactive states. This study examines whether
astrocyte calcium follows a recoverable feedback law, in which longer residence
in a high-calcium state raises the probability that the cell switches itself off,
and whether a controller built on that law can restore healthy rhythm using far
less stimulation. To evaluate the approach against ground truth before applying it
to tissue, a simulator was constructed with a known feedback law and two
conditions, one intact and one in which the feedback is pharmacologically blocked.
A central finding is that the standard hidden Markov model fails here for a
specific reason: it assumes each fluorescence sample reflects only the current
state, whereas calcium indicators blur the signal over time, so the recovering
"off" state, which follows high calcium, is consistently mistaken for activity.
After feature engineering and constrained models proved inadequate, the sensor
itself was modeled, tracking the hidden state and sensor level jointly. This
raised recovery of the off state from 38 to 82 percent and recovered the
simulator's true parameters from unlabeled data, improving the estimated feedback
strength from 0.21 to 0.83 against a true value of 0.90. The resulting controller
intervenes only when the model predicts that a cell cannot recover unaided,
restoring rhythm at 89 percent lower cost than fixed stimulation after a brief
per-cell calibration. When feedback is blocked, restoration correctly fails,
demonstrating that control operates through endogenous biology rather than brute
force. The work is computational, and validation in living tissue is the next
step.
