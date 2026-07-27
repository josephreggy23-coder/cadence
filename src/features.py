"""Shared observation features for Gaussian-HMM fitting and decoding.

The baseline model uses calcium level and a smoothed temporal derivative.
Centralizing this transformation prevents training and decoding from drifting
apart. The derivative is an offline, centered feature; the real-time controller
uses a separate causal estimator.
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
    Build the observation matrix for one trace.

    Parameters
    ----------
    calcium : 1-D array of the observed signal for a single trace.
    mode    : "level+slope" (default, validated) or "level" (ablation baseline).

    Returns
    -------
    (n_frames, n_features) float array, ready for hmmlearn.

    The derivative uses a centered difference on the smoothed signal, so this
    feature is limited to offline analysis. The controller uses a causal state
    estimator instead.
    """
    calcium = np.asarray(calcium, dtype=float)
    if mode == "level":
        return calcium.reshape(-1, 1)
    if mode == "level+slope":
        slope = np.gradient(_moving_average(calcium, deriv_smooth))
        return np.column_stack([calcium, slope])
    raise ValueError(f"unknown feature mode: {mode!r}")


# State mapping and reporting always rank the calcium-level column.
CALCIUM_COL = 0
