"""Frequency-domain features for calcium trace characterization.

Astrocyte calcium dynamics span a range of timescales: fast transients
(< 1 s), sustained plateaus (seconds to tens of seconds), and slow
oscillations (0.01--0.1 Hz). The time-domain features in ``features.py``
capture level and slope; this module adds spectral descriptors that
quantify which timescales dominate a trace.

All routines work on single traces (1-D arrays) sampled at the imaging
frame rate (default 2 Hz, i.e. DT = 0.5 s).

Mathematical background
-----------------------
Power spectral density is estimated via Welch's method: the trace is
split into overlapping segments, each segment is windowed (Hann) and
FFT'd, and the periodograms are averaged. This reduces variance at the
cost of frequency resolution, which is the right trade-off for short,
noisy calcium recordings.

    S(f) = (2 / (fs * N)) * |FFT(x * w)|^2     (one-sided, averaged)

Band power ratios are defined as

    R_band = integral(S, f_lo, f_hi) / integral(S, 0, f_Nyquist)

which is dimensionless and comparable across traces with different
absolute amplitudes.
"""

from __future__ import annotations

import numpy as np

# NumPy >= 2.0 renamed trapz to trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# Default sampling parameters matching generate_synthetic.py.
DEFAULT_FS = 2.0  # Hz (DT = 0.5 s)

# Spectral bands for calcium dynamics, in Hz.
# These boundaries are chosen relative to the 2 Hz sampling rate:
#   slow:     0.01 -- 0.1 Hz  (periods 10--100 s, baseline fluctuations)
#   medium:   0.1  -- 0.4 Hz  (periods 2.5--10 s, oscillatory dynamics)
#   fast:     0.4  -- 1.0 Hz  (up to Nyquist, transient events)
BAND_SLOW = (0.01, 0.1)
BAND_MEDIUM = (0.1, 0.4)
BAND_FAST = (0.4, 1.0)


def welch_psd(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided power spectral density via Welch's method.

    Parameters
    ----------
    signal
        1-D calcium trace.
    fs
        Sampling frequency in Hz.
    nperseg
        Segment length for Welch averaging. Defaults to min(256, len(signal)).
        Shorter segments give lower variance but coarser frequency resolution.
    noverlap
        Number of overlapping samples between segments. Defaults to nperseg // 2.

    Returns
    -------
    freqs
        Frequency axis in Hz, from 0 to fs/2.
    psd
        Power spectral density estimate, in signal-units^2 / Hz.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("signal must be a 1-D array with at least 4 samples")
    if fs <= 0:
        raise ValueError("fs must be positive")

    n = x.size
    if nperseg is None:
        nperseg = min(256, n)
    if nperseg > n:
        nperseg = n
    if noverlap is None:
        noverlap = nperseg // 2

    # Hann window: zero at endpoints, which avoids spectral leakage from
    # abrupt segment boundaries.
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(nperseg) / nperseg))
    win_norm = float(np.sum(window ** 2))  # window power for normalization

    # Segment the signal with overlap.
    step = nperseg - noverlap
    starts = range(0, n - nperseg + 1, step)
    if len(starts) == 0:
        starts = [0]

    # One-sided FFT: keep frequencies 0 to fs/2.
    nfft = nperseg
    n_onesided = nfft // 2 + 1
    psd_accum = np.zeros(n_onesided)

    for start in starts:
        segment = x[start:start + nperseg] * window
        spectrum = np.fft.rfft(segment, n=nfft)
        periodogram = np.abs(spectrum) ** 2
        psd_accum += periodogram[:n_onesided]

    # Normalize: divide by number of segments, window power, and fs.
    psd = psd_accum / (len(starts) * win_norm * fs)
    # Double one-sided PSD (except DC and Nyquist) to account for
    # negative frequencies folded in.
    psd[1:-1] *= 2.0

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, psd


def band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    """Integrate PSD over a frequency band using the trapezoidal rule.

    Parameters
    ----------
    freqs, psd
        Output of ``welch_psd``.
    f_low, f_high
        Band boundaries in Hz (inclusive).

    Returns
    -------
    Power in the specified band, in signal-units^2.
    """
    if f_low >= f_high:
        raise ValueError("f_low must be strictly less than f_high")
    mask = (freqs >= f_low) & (freqs <= f_high)
    if mask.sum() < 2:
        return 0.0
    return float(_trapezoid(psd[mask], freqs[mask]))


def band_power_ratio(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    """Fraction of total power in a frequency band.

    Returns a value in [0, 1]. Useful for comparing spectral shape across
    traces with different absolute amplitudes.
    """
    total = float(_trapezoid(psd, freqs))
    if total <= 0:
        return np.nan
    return band_power(freqs, psd, f_low, f_high) / total


def dominant_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_min: float = 0.01,
) -> float:
    """Frequency of the PSD peak above ``f_min`` Hz.

    The DC component (f=0) is excluded by default via ``f_min``, since
    the mean calcium level is not a dynamical feature.
    """
    mask = freqs >= f_min
    if mask.sum() == 0:
        return np.nan
    idx = np.argmax(psd[mask])
    return float(freqs[mask][idx])


def spectral_centroid(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_min: float = 0.01,
) -> float:
    """Power-weighted mean frequency (spectral centroid).

    This is analogous to the "center of mass" of the spectrum. A trace
    dominated by slow oscillations will have a low centroid; one with
    rapid transients will have a higher centroid.

        f_c = sum(f * S(f)) / sum(S(f))    for f >= f_min
    """
    mask = freqs >= f_min
    f = freqs[mask]
    s = psd[mask]
    total = float(np.sum(s))
    if total <= 0:
        return np.nan
    return float(np.sum(f * s) / total)


def spectral_entropy(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_min: float = 0.01,
) -> float:
    """Normalized spectral entropy of the PSD.

    Measures how uniformly power is distributed across frequencies.
    A pure sinusoid (all power at one frequency) gives entropy near 0;
    white noise (uniform power) gives entropy near 1.

        H = -sum(p * log(p)) / log(N)

    where p(f) = S(f) / sum(S) is the normalized PSD treated as a
    probability distribution, and the denominator log(N) normalizes
    to [0, 1].
    """
    mask = freqs >= f_min
    s = psd[mask].copy()
    total = float(np.sum(s))
    if total <= 0 or mask.sum() < 2:
        return np.nan
    p = s / total
    # Guard against log(0).
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p)))
    return h / np.log(len(s))


def spectral_features(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
) -> dict[str, float]:
    """Compute all spectral descriptors for one calcium trace.

    Returns a flat dictionary suitable for adding as columns to a
    DataFrame alongside the time-domain features from ``features.py``.
    """
    freqs, psd = welch_psd(signal, fs=fs)
    return {
        "dominant_freq_hz": dominant_frequency(freqs, psd),
        "spectral_centroid_hz": spectral_centroid(freqs, psd),
        "spectral_entropy": spectral_entropy(freqs, psd),
        "band_power_slow": band_power_ratio(freqs, psd, *BAND_SLOW),
        "band_power_medium": band_power_ratio(freqs, psd, *BAND_MEDIUM),
        "band_power_fast": band_power_ratio(freqs, psd, *BAND_FAST),
    }
