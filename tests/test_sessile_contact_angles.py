"""Tests for sessile drop contact angle estimation methods."""

import numpy as np

from menipy.common.geometry import (
    estimate_contact_angle_tangent,
    estimate_contact_angle_circle_fit,
    tangent_angle_at_point,
    circle_fit_angle_at_point,
)
from menipy.pipelines.sessile.metrics import compute_sessile_metrics


def test_estimate_contact_angle_tangent():
    """Test tangent-based contact angle estimation."""
    # Create a simple circular profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100
    contour = np.column_stack([x, y])

    # Substrate line at bottom
    substrate_line = ((50, 100), (150, 100))
    contact_point = np.array([100, 100])  # Bottom point

    angle, rmse = estimate_contact_angle_tangent(contour, contact_point, substrate_line)

    # For a circle, contact angle should be 90 degrees
    assert 85 < angle < 95, f"Expected ~90°, got {angle}°"
    assert rmse < 5.0, f"RMSE too high: {rmse}"


def test_estimate_contact_angle_circle_fit():
    """Test circle-fit contact angle estimation."""
    # Create a simple circular profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100
    contour = np.column_stack([x, y])

    # Substrate line at bottom
    substrate_line = ((50, 100), (150, 100))
    contact_point = np.array([100, 100])  # Bottom point

    angle, rmse = estimate_contact_angle_circle_fit(
        contour, contact_point, substrate_line
    )

    # For a circle, contact angle should be 90 degrees
    assert 85 < angle < 95, f"Expected ~90°, got {angle}°"
    assert rmse < 5.0, f"RMSE too high: {rmse}"


def test_compute_sessile_metrics_tangent_method():
    """Test sessile metrics with tangent method."""
    # Create a simple drop profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100
    contour = np.column_stack([x, y])

    substrate_line = ((50, 100), (150, 100))

    metrics = compute_sessile_metrics(
        contour,
        px_per_mm=1.0,
        substrate_line=substrate_line,
        contact_angle_method="tangent",
    )

    assert "theta_left_deg" in metrics
    assert "theta_right_deg" in metrics
    assert "method" in metrics
    assert "uncertainty_deg" in metrics
    assert metrics["method"] == "tangent"

    # Both angles should be close to 90° for a circle
    assert 80 < metrics["theta_left_deg"] < 100
    assert 80 < metrics["theta_right_deg"] < 100


def test_compute_sessile_metrics_circle_fit_method():
    """Test sessile metrics with circle fit method."""
    # Create a simple drop profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100
    contour = np.column_stack([x, y])

    substrate_line = ((50, 100), (150, 100))

    metrics = compute_sessile_metrics(
        contour,
        px_per_mm=1.0,
        substrate_line=substrate_line,
        contact_angle_method="circle_fit",
    )

    assert "theta_left_deg" in metrics
    assert "theta_right_deg" in metrics
    assert "method" in metrics
    assert "uncertainty_deg" in metrics
    assert metrics["method"] == "circle_fit"

    # Both angles should be close to 90° for a circle
    assert 80 < metrics["theta_left_deg"] < 100
    assert 80 < metrics["theta_right_deg"] < 100


def test_compute_sessile_metrics_spherical_cap_method():
    """Test sessile metrics with spherical cap method (legacy)."""
    # Create a simple drop profile - semicircle above substrate
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100  # Bottom at y=100
    contour = np.column_stack([x, y])

    substrate_line = ((50, 100), (150, 100))

    # Provide apex for spherical cap calculation - top of the drop
    apex = (100.0, 150.0)  # Top of the semicircle

    metrics = compute_sessile_metrics(
        contour,
        px_per_mm=1.0,
        substrate_line=substrate_line,
        apex=apex,
        contact_angle_method="spherical_cap",
    )

    assert "theta_left_deg" in metrics
    assert "theta_right_deg" in metrics
    assert "method" in metrics
    assert "uncertainty_deg" in metrics
    assert metrics["method"] == "spherical_cap"

    # Spherical cap should give some angle estimate
    # For a semicircle, the spherical cap angle should be around 90 degrees
    assert metrics["theta_left_deg"] > 0
    assert metrics["theta_right_deg"] > 0


def test_contact_angle_uncertainty_estimation():
    """Test that uncertainty estimates are reasonable."""
    # Create a noisy profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100 + np.random.normal(0, 2, len(theta))
    y = r * np.sin(theta) + 100 + np.random.normal(0, 2, len(theta))
    contour = np.column_stack([x, y])

    substrate_line = ((50, 100), (150, 100))

    metrics = compute_sessile_metrics(
        contour,
        px_per_mm=1.0,
        substrate_line=substrate_line,
        contact_angle_method="tangent",
    )

    uncertainty = metrics["uncertainty_deg"]
    assert "left" in uncertainty
    assert "right" in uncertainty
    assert uncertainty["left"] > 0  # Should have some uncertainty due to noise
    assert uncertainty["right"] > 0


def test_tangent_angle_at_point():
    """Test the tangent_angle_at_point wrapper function."""
    # Create a simple circular profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100
    contour = np.column_stack([x, y])

    # Substrate line at bottom
    substrate_line = ((50, 100), (150, 100))
    contact_point = np.array([100, 100])  # Bottom point

    angle, uncertainty = tangent_angle_at_point(contour, contact_point, substrate_line)

    # For a circle, contact angle should be 90 degrees
    assert 85 < angle < 95, f"Expected ~90°, got {angle}°"
    assert uncertainty >= 0, f"Uncertainty should be non-negative: {uncertainty}"


def test_circle_fit_angle_at_point():
    """Test the circle_fit_angle_at_point wrapper function."""
    # Create a simple circular profile
    theta = np.linspace(0, np.pi, 100)
    r = 50
    x = r * np.cos(theta) + 100
    y = r * np.sin(theta) + 100
    contour = np.column_stack([x, y])

    # Substrate line at bottom
    substrate_line = ((50, 100), (150, 100))
    contact_point = np.array([100, 100])  # Bottom point

    angle, uncertainty = circle_fit_angle_at_point(
        contour, contact_point, substrate_line
    )

    # For a circle, contact angle should be 90 degrees
    assert 85 < angle < 95, f"Expected ~90°, got {angle}°"
    assert uncertainty >= 0, f"Uncertainty should be non-negative: {uncertainty}"
