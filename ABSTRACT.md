# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Astrocytes maintain the calcium rhythms that keep neural circuits stable,
and disruptions to these rhythms contribute to epilepsy, stroke, and
neurodegeneration. Current neuromodulation is open-loop, delivering fixed
stimulation with no principled way to decide when or how much to intervene, which
wastes energy and can drive astrocytes into harmful reactive states. I asked
whether astrocyte calcium follows a recoverable feedback law, in which the longer
a cell remains in a high-calcium state the more likely it is to switch itself off,
and whether a controller using that law could restore healthy rhythm with far
less stimulation. To test the idea against ground truth before
working with tissue, I built a simulator with a known feedback law and two
conditions, one intact and one with the feedback pharmacologically blocked. The
central result was that the standard hidden Markov model fails here for a specific
reason. It assumes each fluorescence sample reflects only the current state, but
calcium indicators blur the signal over time, so the recovering "off" state,
which follows high calcium, is consistently mistaken for activity. After
feature engineering and constrained models failed, I modeled the sensor itself,
tracking the hidden state and sensor level together. This raised recovery of
the off state from 38 to 82 percent and recovered the simulator's true parameters
from unlabeled data, improving the estimated feedback strength from 0.21 to 0.83
against a true value of 0.90. The controller intervenes only when the model
predicts a cell cannot recover on its own, restoring rhythm at 89 percent lower
cost than fixed stimulation after briefly calibrating to each cell. When feedback
is blocked, it correctly fails to restore rhythm, showing that it acts through the
cell's own biology rather than by force. This work is computational, and
validation in living tissue is the next step.
