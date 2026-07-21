# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Astrocytes maintain the calcium rhythms that keep neural circuits stable, and
when that rhythm breaks it contributes to epilepsy, stroke, and
neurodegeneration. Yet clinical neuromodulation is open-loop: it delivers fixed
stimulation with no principled sense of when or how much to intervene, wasting
energy and risking reactive astrogliosis. I asked whether astrocyte calcium obeys
a recoverable, load-dependent feedback law—P(high→refractory | L) =
sigmoid(β0 + β1·L), where β1 > 0 marks negative feedback—and whether a controller
acting *through* that law could restore rhythm with minimal stimulation. To test
this before touching tissue, I built a simulator with a known feedback law and an
intact-versus-blocked design mirroring a pharmacological experiment, so every
inference could be checked against ground truth. The decisive finding was that
the standard hidden Markov model is misspecified here: it assumes fluorescence
reflects only the current state, but a calcium indicator integrates state
history, so the refractory "off" state—which follows high calcium—is
systematically misread as active. Feature engineering and constrained transitions
both failed, one collapsing accuracy from 69% to 45%. Modeling the sensor
explicitly, by tracking the joint hidden regime and sensor level, raised
refractory recovery from 38% to 82% and recovered the simulator's hidden
parameters from unlabeled data, lifting the estimated feedback strength from
0.208 to 0.832 against a true 0.9. My controller then intervenes only when the
learned law predicts the cell cannot self-correct, restoring rhythm at 89% below
fixed-stimulation cost after calibrating each cell's baseline online; testing the
blocked condition even exposed—and let me fix—a dangerous tendency to overstimulate
an unresponsive pathway. Critically, when feedback is blocked the controller fails
to restore rhythm, confirming it works through the endogenous law rather than
brute force—a built-in falsification test. Work is in silico; validation on real
recordings follows.
