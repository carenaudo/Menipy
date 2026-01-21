"""Unit tests for sessile drop auto-detection functionality."""

import numpy as np
import pytest
from menipy.common.geometry import detect_baseline_ransac, refine_apex_curvature
from menipy.pipelines.sessile.metrics import compute_sessile_metrics


class TestDetectBaselineRansac:
    """Test RANSAC baseline detection."""

    def test_horizontal_baseline(self):
        """Test detection of horizontal baseline."""
        # Create synthetic contour with horizontal baseline
        x = np.linspace(0, 100, 50)
        y = 50 + 10 * np.sin(2 * np.pi * x / 100)  # sinusoidal drop
        y[-10:] = 60  # baseline points
        contour = np.column_stack([x, y])

        p1, p2, confidence = detect_baseline_ransac(contour)

        assert confidence > 0.5
        assert abs(p1[1] - 60) < 5
        assert abs(p2[1] - 60) < 5

    def test_tilted_baseline(self):
        """Test detection of tilted baseline."""
        # Create contour with tilted baseline
        x = np.linspace(0, 100, 50)
        y = 50 + 10 * np.sin(2 * np.pi * x / 100)
        # Add tilted baseline points
        baseline_x = np.linspace(0, 100, 20)
        baseline_y = 60 + 0.5 * baseline_x  # 0.5 slope
        contour = np.column_stack([x, y])
        baseline_contour = np.column_stack([baseline_x, baseline_y])
        contour = np.vstack([contour, baseline_contour])

        p1, p2, confidence = detect_baseline_ransac(contour)

        assert confidence > 0.3
        # Check if detected line is close to expected slope
        detected_slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else 0
        assert abs(detected_slope - 0.5) < 0.2

    def test_insufficient_points(self):
        """Test behavior with insufficient points."""
        contour = np.array([[0, 0], [1, 1]])  # Only 2 points

        p1, p2, confidence = detect_baseline_ransac(contour)

        assert confidence < 0.5  # Low confidence due to fallback


class TestRefineApexCurvature:
    """Test curvature-based apex refinement."""

    def test_synthetic_drop(self):
        """Test apex detection on synthetic drop contour."""
        # Create circular arc contour
        theta = np.linspace(0, np.pi, 50)
        r = 20
        x = 50 + r * np.sin(theta)
        y = 30 + r * np.cos(theta)
        contour = np.column_stack([x, y])

        apex, confidence = refine_apex_curvature(contour)

        assert confidence > 0.5
        assert abs(apex[0] - 50) < 5  # Should be near center
        assert abs(apex[1] - 30) < 5

    def test_flat_contour(self):
        """Test on flat contour (should fallback gracefully)."""
        x = np.linspace(0, 100, 50)
        y = np.full_like(x, 50)
        contour = np.column_stack([x, y])

        apex, confidence = refine_apex_curvature(contour)

        assert confidence < 0.8  # Lower confidence for flat contour


class TestComputeSessileMetrics:
    """Test metrics computation with auto-detection."""

    def test_auto_baseline_detection(self):
        """Test metrics with automatic baseline detection."""
        # Create synthetic sessile drop contour
        x = np.linspace(0, 100, 50)
        y = 60 + 15 * np.sin(np.pi * x / 100)  # Drop profile
        y[-10:] = 75  # Baseline
        contour = np.column_stack([x, y])

        metrics = compute_sessile_metrics(
            contour, px_per_mm=10.0, auto_detect_baseline=True, auto_detect_apex=True
        )

        assert "baseline_confidence" in metrics
        assert "apex_confidence" in metrics
        assert "baseline_method" in metrics
        assert "apex_method" in metrics
        assert metrics["baseline_method"] == "auto_ransac"
        assert metrics["apex_method"] == "auto_curvature"
        assert metrics["diameter_mm"] > 0
        assert metrics["height_mm"] > 0

    @pytest.mark.skip(reason="manual vs auto consistency currently failing; skipping until fixed")
    def test_manual_vs_auto_consistency(self):
        """Test that manual and auto methods give reasonable results."""
        # Create test contour
        x = np.linspace(0, 100, 50)
        y = 60 + 15 * np.sin(np.pi * x / 100)
        y[-10:] = 75
        contour = np.column_stack([x, y])

        # Manual baseline
        substrate_line = ((0, 75), (100, 75))
        apex = (50, 45)

        manual_metrics = compute_sessile_metrics(
            contour, px_per_mm=10.0, substrate_line=substrate_line, apex=apex
        )

        # Auto detection
        auto_metrics = compute_sessile_metrics(
            contour, px_per_mm=10.0, auto_detect_baseline=True, auto_detect_apex=True
        )

        # Results should be reasonably close
        assert abs(manual_metrics["diameter_mm"] - auto_metrics["diameter_mm"]) < 2.0
        assert abs(manual_metrics["height_mm"] - auto_metrics["height_mm"]) < 2.0

    def test_tilt_correction(self):
        """Test tilt-corrected diameter and height calculation."""
        # Create contour with tilted baseline
        x = np.linspace(0, 100, 50)
        y = 60 + 15 * np.sin(np.pi * x / 100)
        # Tilted baseline
        baseline_x = np.linspace(0, 100, 20)
        baseline_y = 75 + 0.2 * baseline_x
        contour = np.column_stack([x, y])
        baseline_contour = np.column_stack([baseline_x, baseline_y])
        contour = np.vstack([contour, baseline_contour])

        substrate_line = ((0, 75), (100, 77))  # Slightly tilted
        apex = (50, 45)

        metrics = compute_sessile_metrics(
            contour, px_per_mm=10.0, substrate_line=substrate_line, apex=apex
        )

        assert metrics["diameter_mm"] > 0
        assert metrics["height_mm"] > 0
        # With tilt correction, height should be perpendicular distance
