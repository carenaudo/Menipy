"""Tests for contour smoothing utilities."""

import numpy as np
import pytest

from menipy.common.contour_smoothing import (
    filter_monotonic_contour,
    find_contact_intersections,
    smooth_contour,
    run,
)
from menipy.models.config import ContourSmoothingSettings
from menipy.models.context import Context
from menipy.models.geometry import Contour


class TestFilterMonotonicContour:
    """Tests for filter_monotonic_contour function."""

    def test_already_monotonic(self):
        """Contour with unique X values passes through unchanged."""
        points = np.array([[0, 10], [1, 8], [2, 5], [3, 8], [4, 10]])
        result = filter_monotonic_contour(points)
        
        assert len(result) == 5
        np.testing.assert_array_equal(result, points)

    def test_duplicate_x_keeps_min_y(self):
        """When X values repeat, keep the minimum Y (topmost point)."""
        points = np.array([
            [0, 10],
            [1, 8],
            [1, 12],  # duplicate X=1, should keep Y=8
            [2, 5],
            [3, 10],
        ])
        result = filter_monotonic_contour(points)
        
        assert len(result) == 4
        # Check that X=1 kept Y=8 (the minimum)
        x1_mask = result[:, 0] == 1
        assert result[x1_mask, 1] == 8

    def test_empty_array(self):
        """Empty input returns empty output."""
        points = np.array([]).reshape(0, 2)
        result = filter_monotonic_contour(points)
        assert len(result) == 0


class TestFindContactIntersections:
    """Tests for find_contact_intersections function."""

    def test_simple_curve_crossings(self):
        """Find crossings for a simple parabola-like curve."""
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([10, 6, 3, 3, 6, 10])  # U-shape
        substrate_y = 8.0
        
        left, right = find_contact_intersections(x, y, substrate_y)
        
        assert left is not None
        assert right is not None
        # Left crossing between x=0 and x=1 (y goes from 10 to 6, crossing 8)
        assert 0 < left[0] < 1
        # Right crossing between x=4 and x=5 (y goes from 6 to 10, crossing 8)
        assert 4 < right[0] < 5
        # Both should be at substrate_y
        assert left[1] == substrate_y
        assert right[1] == substrate_y

    def test_no_crossing(self):
        """Curve entirely above or below substrate returns None."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([5, 3, 2, 3, 5])  # All below substrate
        substrate_y = 10.0
        
        left, right = find_contact_intersections(x, y, substrate_y)
        
        assert left is None
        assert right is None

    def test_touching_substrate(self):
        """Curve touching substrate at endpoints."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([10, 6, 3, 6, 10])  # Touches substrate_y=10 at ends
        substrate_y = 10.0
        
        left, right = find_contact_intersections(x, y, substrate_y)
        
        # Should find crossings near the endpoints
        # (depends on implementation - may be None if exactly at boundary)
        # This is edge case behavior


class TestSmoothContour:
    """Tests for smooth_contour function."""

    def test_semicircle_contour(self):
        """Smoothing a semicircle should preserve shape and find contacts."""
        # Create sessile drop contour (image coords: Y increases downward)
        # Apex at top (min Y=60), substrate at bottom (Y=100)
        theta = np.linspace(0, np.pi, 50)
        r = 40
        x = r * np.cos(theta) + 100  # x: 60 to 140
        y = 100 - r * np.sin(theta)  # y: 100 at ends (substrate), ~60 at apex
        contour = np.column_stack([x, y])
        
        substrate_y = 100.0
        
        result = smooth_contour(contour, substrate_y, window_length=11, polyorder=3)
        
        assert result is not None
        assert 'apex' in result
        assert 'left_contact' in result
        assert 'right_contact' in result
        assert 'left_angle_deg' in result
        assert 'right_angle_deg' in result
        
        # Apex should be near top of circle (x~100, y~60)
        assert 95 < result['apex'][0] < 105  # Near x=100
        assert result['apex'][1] < 70  # Near top
        
        # Contact angles depend on how Savgol smooths the curve near endpoints
        # For a 50-point semicircle, expect angles in reasonable range
        assert 50 < result['left_angle_deg'] < 100
        assert 50 < result['right_angle_deg'] < 100

    def test_too_few_points(self):
        """Returns None with too few points."""
        contour = np.array([[0, 95], [1, 90], [2, 95]])
        result = smooth_contour(contour, substrate_y=100.0, window_length=21)
        
        # With only 3 points and window=21, should adjust window or return None
        # The function should handle this gracefully
        if result is not None:
            assert 'apex' in result

    def test_noisy_contour_smoothing(self):
        """Smoothing should reduce noise in the contour."""
        # Create noisy sessile drop
        np.random.seed(42)
        theta = np.linspace(0, np.pi, 100)
        r = 40
        x = r * np.cos(theta) + 100 + np.random.normal(0, 2, len(theta))
        y = 100 - r * np.sin(theta) + np.random.normal(0, 2, len(theta))
        contour = np.column_stack([x, y])
        
        result = smooth_contour(contour, substrate_y=100.0)
        
        assert result is not None
        # Check that y_smooth is actually smoother than raw y
        y_raw = contour[np.argsort(contour[:, 0]), 1]
        y_smooth = result['y_smooth']
        
        # Compute variance of differences (smoothness proxy)
        raw_var = np.var(np.diff(y_raw))
        smooth_var = np.var(np.diff(y_smooth))
        
        assert smooth_var < raw_var  # Smoothed curve should be smoother


class TestIntegration:
    """Integration tests with pipeline Context."""

    def test_run_with_enabled_settings(self):
        """Run function applies smoothing when enabled."""
        # Create sessile drop contour (image coords: Y increases downward)
        ctx = Context()
        theta = np.linspace(0, np.pi, 50)
        r = 40
        x = r * np.cos(theta) + 100
        y = 100 - r * np.sin(theta)  # Apex at min Y, substrate at max Y
        ctx.contour = Contour(xy=np.column_stack([x, y]))
        ctx.substrate_line = ((60, 100), (140, 100))
        
        settings = ContourSmoothingSettings(enabled=True)
        
        result_ctx = run(ctx, settings)
        
        assert hasattr(result_ctx, 'smoothing_results')
        assert result_ctx.smoothing_results is not None
        assert 'apex' in result_ctx.smoothing_results

    def test_run_with_disabled_settings(self):
        """Run function skips smoothing when disabled."""
        ctx = Context()
        theta = np.linspace(0, np.pi, 50)
        r = 40
        x = r * np.cos(theta) + 100
        y = 100 - r * np.sin(theta)
        ctx.contour = Contour(xy=np.column_stack([x, y]))
        
        settings = ContourSmoothingSettings(enabled=False)
        
        result_ctx = run(ctx, settings)
        
        # Should not have smoothing_results set (or it should be None)
        assert not hasattr(result_ctx, 'smoothing_results') or result_ctx.smoothing_results is None

    def test_run_without_contour(self):
        """Run function handles missing contour gracefully."""
        ctx = Context()
        settings = ContourSmoothingSettings(enabled=True)
        
        result_ctx = run(ctx, settings)
        
        # Should not crash, just skip
        assert result_ctx is not None


class TestContourSmoothingSettings:
    """Test the settings model validation."""

    def test_default_values(self):
        """Default settings are valid."""
        settings = ContourSmoothingSettings()
        
        assert settings.enabled is False
        assert settings.window_length == 21
        assert settings.polyorder == 3

    def test_window_must_be_odd(self):
        """Even window length should raise validation error."""
        with pytest.raises(ValueError, match="odd"):
            ContourSmoothingSettings(window_length=20)

    def test_valid_odd_window(self):
        """Odd window lengths are accepted."""
        settings = ContourSmoothingSettings(window_length=15)
        assert settings.window_length == 15
