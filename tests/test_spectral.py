"""Tests for spectral analysis utilities.

These tests verify that the Welch PSD estimator, band-power calculations,
and derived spectral descriptors behave correctly on signals with known
frequency content.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# The source modules live under src/ without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral import (
    BAND_FAST,
    BAND_MEDIUM,
    BAND_SLOW,
    band_power,
    band_power_ratio,
    dominant_frequency,
    spectral_centroid,
    spectral_entropy,
    spectral_features,
    welch_psd,
)


FS = 2.0  # 2 Hz sampling, matching the CADENCE imaging rate


def _sine(freq: float, duration_s: float = 300.0, fs: float = FS) -> np.ndarray:
    """Generate a pure sinusoid at the given frequency."""
    t = np.arange(0, duration_s, 1.0 / fs)
    return np.sin(2.0 * np.pi * freq * t)


# ---- Welch PSD basics -------------------------------------------------------

def test_welch_psd_returns_correct_shapes():
    signal = np.random.default_rng(0).normal(size=200)
    freqs, psd = welch_psd(signal, fs=FS)

    assert freqs.ndim == 1
    assert psd.ndim == 1
    assert freqs.shape == psd.shape
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(FS / 2.0)


def test_welch_psd_peak_at_signal_frequency():
    """A pure sinusoid should produce a PSD peak at its frequency."""
    target_freq = 0.2  # Hz
    signal = _sine(target_freq, duration_s=300.0)
    freqs, psd = welch_psd(signal, fs=FS)

    peak_freq = freqs[np.argmax(psd)]
    assert peak_freq == pytest.approx(target_freq, abs=0.02)


def test_welch_psd_nonnegative():
    signal = np.random.default_rng(1).normal(size=300)
    _, psd = welch_psd(signal, fs=FS)
    assert np.all(psd >= 0)


def test_welch_psd_rejects_too_short():
    with pytest.raises(ValueError, match="at least 4"):
        welch_psd(np.array([1.0, 2.0, 3.0]), fs=FS)


def test_welch_psd_rejects_multidimensional():
    with pytest.raises(ValueError, match="1-D"):
        welch_psd(np.zeros((10, 2)), fs=FS)


def test_welch_psd_rejects_nonpositive_fs():
    with pytest.raises(ValueError, match="fs must be positive"):
        welch_psd(np.zeros(100), fs=0.0)


# ---- Band power -------------------------------------------------------------

def test_band_power_captures_sinusoid():
    """Most power of a 0.2 Hz sinusoid should land in the medium band."""
    signal = _sine(0.2, duration_s=300.0)
    freqs, psd = welch_psd(signal, fs=FS)

    medium = band_power_ratio(freqs, psd, *BAND_MEDIUM)
    slow = band_power_ratio(freqs, psd, *BAND_SLOW)
    fast = band_power_ratio(freqs, psd, *BAND_FAST)

    assert medium > 0.8
    assert slow < 0.1
    assert fast < 0.1


def test_band_power_ratio_sums_to_near_one():
    """Band ratios covering the full spectrum should sum to approximately 1."""
    rng = np.random.default_rng(2)
    signal = rng.normal(size=600)
    freqs, psd = welch_psd(signal, fs=FS)

    total_ratio = (
        band_power_ratio(freqs, psd, *BAND_SLOW)
        + band_power_ratio(freqs, psd, *BAND_MEDIUM)
        + band_power_ratio(freqs, psd, *BAND_FAST)
    )
    # Won't be exactly 1 because DC (0 to 0.01 Hz) is excluded.
    assert total_ratio == pytest.approx(1.0, abs=0.05)


def test_band_power_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="f_low must be strictly less"):
        band_power(np.array([0.0, 0.5, 1.0]), np.array([1.0, 1.0, 1.0]), 0.5, 0.1)


# ---- Dominant frequency ------------------------------------------------------

def test_dominant_frequency_matches_sinusoid():
    signal = _sine(0.3, duration_s=300.0)
    freqs, psd = welch_psd(signal, fs=FS)

    assert dominant_frequency(freqs, psd) == pytest.approx(0.3, abs=0.02)


def test_dominant_frequency_picks_stronger_component():
    """When two sinusoids are mixed, the dominant should be the louder one."""
    t = np.arange(0, 300.0, 1.0 / FS)
    signal = 2.0 * np.sin(2 * np.pi * 0.15 * t) + 0.5 * np.sin(2 * np.pi * 0.4 * t)
    freqs, psd = welch_psd(signal, fs=FS)

    assert dominant_frequency(freqs, psd) == pytest.approx(0.15, abs=0.02)


# ---- Spectral centroid -------------------------------------------------------

def test_spectral_centroid_near_signal_frequency():
    signal = _sine(0.25, duration_s=300.0)
    freqs, psd = welch_psd(signal, fs=FS)

    assert spectral_centroid(freqs, psd) == pytest.approx(0.25, abs=0.05)


def test_spectral_centroid_shifts_with_frequency():
    """A higher-frequency signal should have a higher centroid."""
    low = _sine(0.1, duration_s=300.0)
    high = _sine(0.5, duration_s=300.0)
    f_lo, p_lo = welch_psd(low, fs=FS)
    f_hi, p_hi = welch_psd(high, fs=FS)

    assert spectral_centroid(f_lo, p_lo) < spectral_centroid(f_hi, p_hi)


# ---- Spectral entropy --------------------------------------------------------

def test_spectral_entropy_low_for_sinusoid():
    """A pure sinusoid concentrates power at one frequency: low entropy."""
    signal = _sine(0.2, duration_s=300.0)
    freqs, psd = welch_psd(signal, fs=FS)

    assert spectral_entropy(freqs, psd) < 0.5


def test_spectral_entropy_high_for_noise():
    """White noise distributes power uniformly: high entropy."""
    signal = np.random.default_rng(3).normal(size=600)
    freqs, psd = welch_psd(signal, fs=FS)

    assert spectral_entropy(freqs, psd) > 0.8


def test_spectral_entropy_bounded():
    """Normalized entropy must lie in [0, 1]."""
    for seed in range(5):
        signal = np.random.default_rng(seed).normal(size=400)
        freqs, psd = welch_psd(signal, fs=FS)
        h = spectral_entropy(freqs, psd)
        assert 0.0 <= h <= 1.0


# ---- Convenience wrapper -----------------------------------------------------

def test_spectral_features_returns_all_keys():
    signal = np.random.default_rng(4).normal(size=300)
    feats = spectral_features(signal, fs=FS)

    expected_keys = {
        "dominant_freq_hz",
        "spectral_centroid_hz",
        "spectral_entropy",
        "band_power_slow",
        "band_power_medium",
        "band_power_fast",
    }
    assert set(feats.keys()) == expected_keys
    for v in feats.values():
        assert np.isfinite(v)
