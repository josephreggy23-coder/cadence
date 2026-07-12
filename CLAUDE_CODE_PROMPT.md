# Claude Code build prompt for CADENCE

Paste the block below into Claude Code, run from inside the `cadence/` repo.

---

I'm building CADENCE: a model-based controller that restores healthy rhythm to
astrocyte-like glial calcium dynamics by learning the system's own feedback law
and intervening minimally at the right moment. This is a neuroscience research
project (ISEF/AAN). The SOLUTION is the controller; the feedback-law model is its
engine.

I already have a synthetic data generator (src/generate_synthetic.py) that
produces calcium traces from a KNOWN load-dependent feedback law in two
conditions: "intact" (feedback present, beta1=0.9) and "blocked" (control,
beta1~0). Data: data/intact.csv, data/blocked.csv. Columns: condition, trace_id,
time_s, calcium, true_state (true_state is for validation only — never used in
fitting).

Build a clean, reproducible pipeline (numpy/scipy/hmmlearn/pandas/matplotlib
only), heavily commented so a PhD judge asking "why did you do it that way" gets
an answer from the code:

1) src/fit_hmm.py — Fit a 4-state Gaussian HMM to the calcium signal. Justify the
   state count with held-out log-likelihood and BIC. Save fitted parameters.

2) src/recover_states.py — Viterbi-decode the hidden states, map them to
   biological labels by emission mean, and MEASURE recovery vs true_state
   (accuracy + confusion matrix). This proves the estimator works.

3) src/estimate_feedback_law.py — Compute a causal "calcium load" L (exponential
   accumulator over time in the inferred high state). Estimate
   P(high→refractory | L) = sigmoid(b0 + b1·L) via logistic regression; report b1
   with a confidence interval. Run separately for intact vs blocked and test
   b1_intact > b1_blocked. b1>0 is the feedback signature.

4) src/controller.py — THE SOLUTION. Implement CADENCE as a model-based control
   policy that, given the live inferred state and load L, decides whether and how
   much to intervene to keep calcium in the healthy oscillatory cadence and out of
   pathological SUSTAINED_HIGH. Requirements:
   - Use the learned feedback law to act MINIMALLY: intervene only when the model
     predicts the system will fail to self-suppress (i.e., when endogenous
     P(high→refractory | L) is too low to recover in time).
   - Compare against two baselines: (a) no control, (b) open-loop fixed
     stimulation. Report cumulative intervention "cost" and time spent in the
     pathological state for each. CADENCE should achieve equal-or-better
     restoration at LOWER cost.
   - Include a disease-model trace (elevated tendency to enter/stay in
     SUSTAINED_HIGH) to demonstrate restoration.
   - KILL-SHOT: when feedback is blocked (beta1~0), CADENCE's minimal policy
     should FAIL to restore, proving it works through the endogenous law. Show this.

5) src/run_all.py — one command runs the full pipeline and writes publication-
   quality figures: state-annotated trace; recovered feedback curve with CI
   (intact vs blocked); transition-matrix heatmaps; and a control comparison
   (calcium trace + intervention markers) for no-control vs open-loop vs CADENCE,
   plus a bar chart of pathological-time and intervention-cost across the three.

6) tests/test_pipeline.py — assert: recovered b1 significantly >0 for intact and
   ≈0 for blocked; CADENCE reduces pathological time vs no-control; CADENCE cost
   < open-loop cost; CADENCE fails to restore under blocked feedback.

Design constraints:
- Validate on synthetic ground truth FIRST; recovery before trust.
- Everything reproducible from run_all.py; fixed random seeds.
- Explain each modeling and control-policy choice in comments.
