"""Unit tests for needle-in-sessile-drop profile fitting algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from menipy.math.needle_profile_fit import (
    NeedleProfileFitResult,
    fit_contact_angle_cdf,
    fit_contact_angle_tangent,
    fit_needle_contact_angles,
)


def _generate_synthetic_circular_arc(
    theta_deg: float,
    radius: float = 100.0,
    n_points: int = 100,
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]:
    """Generate a clean synthetic circular drop profile resting on y=200."""
    sub_y = 200.0
    substrate_line = ((0.0, sub_y), (400.0, sub_y))

    theta_rad = np.radians(theta_deg)
    cx = 200.0
    cy = sub_y + radius * np.cos(theta_rad)

    # Base contact points
    half_width = radius * np.sin(theta_rad)
    p_left = (cx - half_width, sub_y)
    p_right = (cx + half_width, sub_y)

    # Angle alpha from vertical downward axis in [-theta, +theta]
    alphas = np.linspace(-theta_rad, theta_rad, n_points)
    xs = cx + radius * np.sin(alphas)
    ys = cy - radius * np.cos(alphas)
    contour = np.column_stack([xs, ys])

    return contour, (p_left, p_right), substrate_line


@pytest.mark.parametrize("true_theta", [40.0, 60.0, 80.0, 110.0])
def test_tangent_method_accuracy(true_theta: float):
    """Tangent Method 2 should retrieve known circular contact angle within 2.5 deg."""
    contour, contacts, sub_line = _generate_synthetic_circular_arc(true_theta)

    ang_l, rmse_l = fit_contact_angle_tangent(contour, contacts[0], sub_line, is_left=True)
    ang_r, rmse_r = fit_contact_angle_tangent(contour, contacts[1], sub_line, is_left=False)

    assert pytest.approx(ang_l, abs=2.5) == true_theta
    assert pytest.approx(ang_r, abs=2.5) == true_theta
    assert rmse_l < 2.0
    assert rmse_r < 2.0


@pytest.mark.parametrize("true_theta", [45.0, 70.0, 95.0])
def test_cdf_method_accuracy(true_theta: float):
    """CDF method should retrieve known circular contact angle within 2.0 deg."""
    contour, contacts, sub_line = _generate_synthetic_circular_arc(true_theta)

    ang_l, rmse_l = fit_contact_angle_cdf(contour, contacts[0], sub_line, is_left=True)
    ang_r, rmse_r = fit_contact_angle_cdf(contour, contacts[1], sub_line, is_left=False)

    assert pytest.approx(ang_l, abs=2.0) == true_theta
    assert pytest.approx(ang_r, abs=2.0) == true_theta
    assert rmse_l < 2.5
    assert rmse_r < 2.5


def test_fit_needle_contact_angles_auto():
    """Unified fitting function should run in 'auto' mode and populate diagnostics."""
    contour, contacts, sub_line = _generate_synthetic_circular_arc(65.0)

    res = fit_needle_contact_angles(contour, contacts, sub_line, method="auto")

    assert isinstance(res, NeedleProfileFitResult)
    assert pytest.approx(res.theta_left_deg, abs=2.0) == 65.0
    assert pytest.approx(res.theta_right_deg, abs=2.0) == 65.0
    assert res.rmse_left_px < 2.0
    assert res.rmse_right_px < 2.0
    assert "left" in res.diagnostics
    assert "right" in res.diagnostics
