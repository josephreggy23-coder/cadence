# CADENCE: Sensor-Aware State Inference and Assumption-Aware Control

Dysregulated astrocytic Ca2+ signaling is a key driver of high-burden CNS
disorders, including epilepsy, stroke, and neurodegeneration, yet existing
neuromodulation remains open-loop and neuron-centric, delivering fixed stimulation
with no principled control of timing or dose. Here we present
CADENCE, a closed-loop computational framework that learns an astrocyte's own
regulatory law and restores healthy Ca2+ rhythm through minimal intervention. We hypothesized that astrocytic Ca2+ obeys a load-dependent negative feedback
law, in which prolonged residence in a high-Ca2+ state raises the probability that
the cell transitions into a refractory, self-suppressed state, and asked whether
it could be recovered from imaging and exploited for control. Validation used a
ground-truth simulator with paired intact and pharmacologically blocked feedback,
rendering every inference falsifiable. We found that a standard hidden
Markov model is misspecified for Ca2+ imaging: it treats each fluorescent sample
as instantaneous, whereas real indicators integrate signal over time, causing the
recovering state to be systematically misread. Embedding the sensor within the
state estimator resolved this, raising recovery of the suppressed state from 38%
to 82% and reconstructing the generative parameters from unlabeled data, which
sharpened the recovered feedback coefficient from 0.21 to 0.83 against a true 0.90.
The controller then intervenes only when the model predicts a cell cannot
self-correct, restoring rhythm at 89% lower cost than fixed stimulation after
per-cell calibration. Unexpectedly, improving state estimation degraded a
naive controller, revealing that feedback gain generalizes across disease while
baseline excitability does not, a dissociation that motivated online calibration.
Critically, blocking the feedback pathway abolished restoration by design,
confirming mechanism-dependent control rather than brute force. Together, these
results show CADENCE excels at recovering hidden regulatory rules from
imaging and delivering precisely timed, low-dose, mechanism-specific intervention
that resists fooling itself, establishing a foundation for efficient, adaptive
closed-loop neuromodulation of glial calcium in CNS disease.
