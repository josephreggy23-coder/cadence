"""Tests for the features module (observation features for Gaussian-HMM)."""

import sys
import os

import numpy as np
import pytest

# Ensure the src directory is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from features import (
    _moving_average,
    make_features,
    DERIV_SMOOTH_FRAMES,
    FEATURE_NAMES,
    CALCIUM_COL,
)


# ---------------------------------------------------------------------------
# _moving_average
# ---------------------------------------------------------------------------


class TestMovingAverage:
    """Tests for the centred moving-average helper."""

    def test_w_zero_returns_input_unchanged(self):
        """w=0 should be a no-op, returning the original array."""
        x = np.array([1.0, 2.0, 3.0])
        result = _moving_average(x, 0)
        assert result is x

    def test_w_one_returns_input_unchanged(self):
        """w=1 should be a no-op, returning the original array."""
        x = np.array([1.0, 2.0, 3.0])
        result = _moving_average(x, 1)
        assert result is x

    def test_w_negative_returns_input_unchanged(self):
        """Negative w should also be a no-op."""
        x = np.array([4.0, 5.0, 6.0])
        result = _moving_average(x, -2)
        assert result is x

    def test_w3_known_sequence(self):
        """w=3 on [1, 2, 3, 4, 5] should produce known centred averages.

        Interior values are exact averages of three neighbours.
        Boundary behaviour comes from np.convolve with mode='same'.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _moving_average(x, 3)
        # Interior: (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        np.testing.assert_allclose(result[1], 2.0)
        np.testing.assert_allclose(result[2], 3.0)
        np.testing.assert_allclose(result[3], 4.0)

    def test_preserves_length(self):
        """Output length must equal input length (mode='same')."""
        x = np.arange(20.0)
        for w in [2, 3, 5, 7]:
            result = _moving_average(x, w)
            assert len(result) == len(x), f"length mismatch for w={w}"

    def test_constant_input_stays_constant_interior(self):
        """Interior of a constant trace should remain constant after smoothing.

        Boundary elements are affected by np.convolve zero-padding in
        'same' mode, so only the interior (fully overlapping) region is
        checked.
        """
        w = 5
        x = np.full(20, 42.0)
        result = _moving_average(x, w)
        # Interior elements where the kernel fully overlaps.
        np.testing.assert_allclose(result[w // 2 : -(w // 2)], 42.0)


# ---------------------------------------------------------------------------
# make_features — mode="level"
# ---------------------------------------------------------------------------


class TestMakeFeaturesLevel:
    """Tests for make_features with mode='level'."""

    def test_output_shape(self):
        """Output should be (n, 1)."""
        calcium = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = make_features(calcium, mode="level")
        assert result.shape == (5, 1)

    def test_values_match_input(self):
        """The single column must contain the original calcium values."""
        calcium = np.array([10.0, 20.0, 30.0])
        result = make_features(calcium, mode="level")
        np.testing.assert_array_equal(result[:, 0], calcium)

    def test_accepts_list_input(self):
        """A plain Python list should be coerced to float array."""
        calcium = [1, 2, 3]
        result = make_features(calcium, mode="level")
        assert result.shape == (3, 1)
        assert result.dtype == float


# ---------------------------------------------------------------------------
# make_features — mode="level+slope"
# ---------------------------------------------------------------------------


class TestMakeFeaturesLevelSlope:
    """Tests for make_features with mode='level+slope'."""

    def test_output_shape(self):
        """Output should be (n, 2)."""
        calcium = np.arange(10.0)
        result = make_features(calcium, mode="level+slope")
        assert result.shape == (10, 2)

    def test_first_column_is_calcium(self):
        """Column 0 must be the unmodified calcium values."""
        calcium = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        result = make_features(calcium, mode="level+slope")
        np.testing.assert_array_equal(result[:, 0], calcium)

    def test_derivative_of_linear_ramp(self):
        """A perfect ramp f(t)=t has slope=1 everywhere (interior).

        After smoothing a perfect ramp the ramp is unchanged, so
        np.gradient of a linear signal returns a constant 1.0.
        """
        calcium = np.arange(50.0)  # long enough that boundary effects are negligible
        result = make_features(calcium, mode="level+slope")
        slope = result[:, 1]
        # Interior points (away from edges) should be very close to 1.0.
        np.testing.assert_allclose(slope[5:-5], 1.0, atol=1e-12)

    def test_zero_slope_for_constant_interior(self):
        """A constant trace should produce zero slope in the interior.

        Boundary elements may be non-zero because np.convolve with
        mode='same' zero-pads, creating edge artefacts in the smoothed
        signal that np.gradient then picks up.
        """
        calcium = np.full(50, 7.0)
        result = make_features(calcium, mode="level+slope")
        # Check well inside the boundaries.
        np.testing.assert_allclose(result[5:-5, 1], 0.0, atol=1e-14)

    def test_default_mode_is_level_slope(self):
        """Calling without mode should default to 'level+slope'."""
        calcium = np.arange(10.0)
        result = make_features(calcium)
        assert result.shape[1] == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestMakeFeaturesErrors:
    """Tests for invalid arguments to make_features."""

    def test_unknown_mode_raises_value_error(self):
        """An unrecognised mode string must raise ValueError."""
        with pytest.raises(ValueError, match="unknown feature mode"):
            make_features(np.zeros(5), mode="bogus")

    def test_unknown_mode_message_includes_mode_name(self):
        """The error message should contain the offending mode name."""
        with pytest.raises(ValueError, match="bogus"):
            make_features(np.zeros(5), mode="bogus")


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_feature_names_length(self):
        """FEATURE_NAMES should list two feature names (level + slope)."""
        assert len(FEATURE_NAMES) == 2

    def test_feature_names_are_strings(self):
        """Each feature name should be a non-empty string."""
        for name in FEATURE_NAMES:
            assert isinstance(name, str) and len(name) > 0

    def test_calcium_col_is_zero(self):
        """CALCIUM_COL must be 0."""
        assert CALCIUM_COL == 0

    def test_deriv_smooth_frames_is_positive(self):
        """DERIV_SMOOTH_FRAMES should be a positive integer."""
        assert isinstance(DERIV_SMOOTH_FRAMES, int)
        assert DERIV_SMOOTH_FRAMES > 0


# ---------------------------------------------------------------------------
# Edge cases — very short traces
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for single-element and very short traces."""

    def test_single_element_level(self):
        """A single-element trace in level mode should return shape (1, 1)."""
        result = make_features(np.array([3.14]), mode="level")
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(3.14)

    def test_single_element_level_slope_raises(self):
        """A single-element trace in level+slope mode fails.

        np.convolve with a kernel wider than the input (w=3 > n=1)
        produces an output longer than the input even in 'same' mode,
        causing a shape mismatch in np.column_stack. This documents the
        current behaviour; traces shorter than DERIV_SMOOTH_FRAMES are
        not expected in practice.
        """
        with pytest.raises(ValueError):
            make_features(np.array([3.14]), mode="level+slope")

    def test_two_element_level(self):
        """A two-element trace should produce (2, 1) in level mode."""
        result = make_features(np.array([1.0, 2.0]), mode="level")
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0])

    def test_two_element_level_slope_raises(self):
        """A two-element trace in level+slope mode fails.

        Same root cause as the single-element case: np.convolve with
        w=3 on a length-2 input produces a length-3 output in 'same'
        mode, which cannot be column-stacked with the length-2 calcium.
        """
        with pytest.raises(ValueError):
            make_features(np.array([1.0, 2.0]), mode="level+slope")
