# CADENCE: Restoring Glial Calcium Rhythm Through a Learned Endogenous Feedback Law

Astrocytes govern the calcium rhythms that keep neural circuits stable, and the
breakdown of these rhythms is a common driver of epilepsy, stroke, and
neurodegeneration. Present neuromodulation remains blunt and open-loop, applying
fixed stimulation with no principled sense of timing or dose, squandering energy
and risking the reactive astrocyte states it should prevent. This project
reframes the challenge as one of control: rather than overriding the cell, it
learns the cell's own regulatory rule and steers the system through it. The
governing hypothesis is a load-dependent feedback law, in which prolonged
residence in a high-calcium state progressively increases the probability that a
cell suppresses itself. To earn trust before any tissue is involved, a simulator
with a known feedback law was constructed, paired with intact and
pharmacologically blocked conditions that render every inference checkable against
ground truth. A pivotal insight followed: the standard hidden Markov model is
fundamentally misspecified for calcium imaging, because it treats each fluorescent
sample as a snapshot of the present state, whereas real indicators accumulate
signal across time. As a result the recovering "off" state, which always trails
high activity, is systematically misread. Modeling the sensor itself, and tracking
hidden state and sensor level jointly, resolved the failure. Recovery of the off
state rose from 38 to 82 percent, and the simulator's hidden parameters were
reconstructed from unlabeled data, sharpening the estimated feedback strength from
0.21 to 0.83 against a true value of 0.90. The resulting controller acts only when
the model predicts a cell cannot recover unaided, restoring rhythm at 89 percent
lower cost than fixed stimulation once briefly calibrated to each cell. Decisively,
when the feedback pathway is blocked, restoration fails by design, confirming that
control flows through endogenous biology rather than force. This computational
foundation is built to transfer to living tissue.
