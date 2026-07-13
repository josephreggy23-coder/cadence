"""
features.py  —  CADENCE pipeline, shared preprocessing
======================================================
ONE definition of the observation vector, imported by BOTH fit_hmm.py (training)
and recover_states.py (decoding).

WHY THIS FILE EXISTS AT ALL
If training and decoding ever computed features differently — even by a
one-frame shift in the derivative window — the decoded states would be silently
garbage while every printed diagnostic still looked fine. Centralising it makes
that class of bug impossible rather than merely unlikely.

WHY (level, slope) AND NOT LEVEL ALONE
Empirically measured, not assumed. Fitting on calcium level alone recovers the
REFRACTORY state at only 24% recall: half of all refractory frames are read as
OSCILLATORY. The reason is physical, not statistical —

  - REFRACTORY's true emission mean (0.10) sits just 0.05 above QUIESCENT (0.05),
    which is at or below the observation noise (sd 0.03-0.04). By LEVEL, the two
    are essentially the same measurement.
  - REFRACTORY always follows SUSTAINED_HIGH, so the GCaMP decay tail is still
    coasting down through the OSCILLATORY range (~0.35) during the frames when
    the underlying state has already switched off.

So the discriminating information is not "how high" but "how high AND which way
is it moving": refractory is mid-level *and falling steeply*; oscillatory is
mid-level and not systematically falling; quiescent is low and flat.

MEASURED EFFECT (intact synthetic condition, used to learn the exit law):
    feature set     overall acc    SUSTAINED_HIGH recall    REFRACTORY recall
    level only         69.6%              77.8%                   24.1%
    level + slope      68.5%              93.2%                   37.8%

Overall accuracy is deliberately NOT the target metric: it is dominated by
QUIESCENT (~48% of frames), so it rewards getting the easy state right. The two
states that Module 3's feedback law actually depends on — SUSTAINED_HIGH and its
exit into REFRACTORY — both improve substantially. That is the trade we want.

HONEST CAVEAT: 37.8% refractory recall is BETTER, not GOOD. The feedback-law
estimate in Module 3 is therefore attenuated by state-estimation error, and
Module 3 reports that explicitly rather than pretending the states are clean.

REJECTED ALTERNATIVE (kept here because the negative result is informative):
Deconvolving the GCaMP kernel (inverting c[t] = c[t-1] + (latent[t]-c[t-1])/tau)
was tested across post-smoothing widths 1/3/5/7. It trades overall accuracy
against refractory recall and never resolves the confusion — the best-accuracy
setting (76.3%) made REFRACTORY *worse* (18.8%). It also requires knowing tau,
which on real recordings must itself be estimated. Slope needs no such constant.
"""

import numpy as np

# Width (frames) of the moving average applied before differentiating. The raw
# frame-to-frame difference of a noisy trace is dominated by observation noise;
# a short pre-smooth makes the derivative reflect the sensor's actual trend
# without blurring away the fast HIGH->REFRACTORY fall we need to detect.
DERIV_SMOOTH_FRAMES = 3

FEATURE_NAMES = ["calcium", "d(calcium)/dt"]


def _moving_average(x, w):
    """Centred moving average; w<=1 is a no-op."""
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def make_features(calcium, mode="level+slope", deriv_smooth=DERIV_SMOOTH_FRAMES):
    """
    Build the observation matrix for ONE trace.

    Parameters
    ----------
    calcium : 1-D array of the observed signal for a single trace.
    mode    : "level+slope" (default, validated) or "level" (ablation baseline).

    Returns
    -------
    (n_frames, n_features) float array, ready for hmmlearn.

    The derivative is computed with np.gradient on the pre-smoothed signal, which
    is a centred difference. Note this makes the feature mildly NON-CAUSAL (frame
    t uses t+1). That is acceptable here because Modules 1-3 are OFFLINE analysis
    of complete recordings. The Module 4 controller runs online and must not peek
    at the future, so it uses a causal backward difference instead - see
    controller.py, which documents that deliberate divergence.
    """
    calcium = np.asarray(calcium, dtype=float)
    if mode == "level":
        return calcium.reshape(-1, 1)
    if mode == "level+slope":
        slope = np.gradient(_moving_average(calcium, deriv_smooth))
        return np.column_stack([calcium, slope])
    raise ValueError(f"unknown feature mode: {mode!r}")


def n_features(mode="level+slope"):
    return 1 if mode == "level" else 2


# The calcium level is ALWAYS feature column 0. Both the label-assignment step in
# recover_states.py and the emission-mean reporting in fit_hmm.py rank states by
# this column, so the biological ordering never depends on the slope feature.
CALCIUM_COL = 0
