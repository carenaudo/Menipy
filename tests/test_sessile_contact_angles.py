"""Tests for sessile drop contact angle estimation methods."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.common.geometry import (
    circle_fit_angle_at_point,
    estimate_contact_angle_circle_fit,
    estimate_contact_angle_tangent,
    tangent_angle_at_point,
)
from menipy.models.context import Context
from menipy.models.geometry import Contour
from menipy.pipelines.sessile.metrics import compute_sessile_metrics
from menipy.pipelines.sessile.stages import SessilePipeline


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


def test_tangent_method_ignores_closed_baseline_segment():
    theta = np.linspace(np.pi, 0.0, 120)
    dome = np.column_stack([100.0 + 50.0 * np.cos(theta), 100.0 + 50.0 * np.sin(theta)])
    contour = np.vstack([[50.0, 100.0], dome, [150.0, 100.0], [50.0, 100.0]])

    metrics = compute_sessile_metrics(
        contour,
        px_per_mm=10.0,
        substrate_line=((50.0, 100.0), (150.0, 100.0)),
        apex=(100.0, 150.0),
        contact_points=((50, 100), (150, 100)),
        contact_angle_method="tangent",
    )

    assert metrics["theta_left_deg"] == pytest.approx(90.0, abs=5.0)
    assert metrics["theta_right_deg"] == pytest.approx(90.0, abs=5.0)
    assert metrics["contact_angle_fit_rmse_px"]["left"] >= 0.0
    assert metrics["contact_angle_fit_rmse_px"]["right"] >= 0.0


def test_tangent_method_ignores_vertical_contact_closure_edge():
    contact_point = np.array([0.0, 100.0])
    substrate_line = ((0.0, 100.0), (120.0, 100.0))
    target_angle = 52.0
    x = np.linspace(0.0, 40.0, 24)
    branch = np.column_stack([x, 100.0 - x * np.tan(np.deg2rad(target_angle))])
    vertical_closure = np.column_stack([np.zeros(24), np.linspace(100.0, 0.0, 24)])
    contour = np.vstack([vertical_closure, branch])

    angle, rmse = tangent_angle_at_point(contour, contact_point, substrate_line)

    assert angle == pytest.approx(target_angle, abs=2.0)
    assert rmse >= 0.0


def test_sessile_contour_extraction_normalizes_opencv_contour_shape():
    contour = np.array(
        [[50, 100], [75, 140], [100, 150], [125, 140], [150, 100]],
        dtype=float,
    ).reshape(-1, 1, 2)
    ctx = Context(drop_contour=contour)

    SessilePipeline().do_contour_extraction(ctx)

    xy = np.asarray(ctx.contour.xy)
    assert xy.shape == (5, 2)
    np.testing.assert_allclose(xy[0], [50.0, 100.0])


def test_sessile_refinement_builds_separate_calculation_contour():
    contour = np.array(
        [
            [10.0, 20.0],
            [25.0, 10.0],
            [40.0, 20.0],
            [45.0, 35.0],
            [40.0, 45.0],
            [10.0, 45.0],
            [5.0, 35.0],
        ],
        dtype=float,
    )
    ctx = Context(
        contour=Contour(xy=contour),
        substrate_line=((0.0, 50.0), (60.0, 50.0)),
        apex_point=(25, 10),
    )

    pipe = SessilePipeline()
    out = pipe.do_contour_refinement(ctx)

    assert out is not None
    assert out.contour is not None
    assert out.sessile_calc_contour is not None
    assert len(np.asarray(out.sessile_calc_contour).reshape(-1, 2)) >= len(
        np.asarray(out.contour.xy).reshape(-1, 2)
    )
    assert out.sessile_calc_contact_points is not None


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


def test_sessile_3_auto_detection_finds_true_contact_angles():
    sample = "data/samples/sessile_3.jpeg"
    image = cv2.imread(sample)
    assert image is not None

    calibration = AutoCalibrator(image, "sessile").detect_all()

    assert calibration.drop_contour is not None
    assert calibration.substrate_line is not None
    assert calibration.contact_points is not None

    substrate_y = calibration.substrate_line[0][1]
    assert 240 <= substrate_y <= 260

    left_contact, right_contact = calibration.contact_points
    assert right_contact[0] - left_contact[0] > 150
    assert abs(left_contact[1] - substrate_y) <= 2
    assert abs(right_contact[1] - substrate_y) <= 2

    ctx = SessilePipeline().run_with_plan(
        only=["compute_metrics"],
        image=sample,
        drop_contour=calibration.drop_contour,
        contact_points=calibration.contact_points,
        apex_point=calibration.apex_point,
        needle_rect=calibration.needle_rect,
        roi_rect=calibration.roi_rect,
        substrate_line=calibration.substrate_line,
        calibration_params={
            "needle_diameter_mm": 1.83,
            "drop_density_kg_m3": 1000.0,
            "fluid_density_kg_m3": 1.2,
        },
    )

    assert ctx.results["theta_left_deg"] == pytest.approx(52.0, abs=5.0)
    assert ctx.results["theta_right_deg"] == pytest.approx(52.0, abs=5.0)


@pytest.mark.parametrize(
    "sample",
    [
        "data/samples/gota depositada 1.png",
        "data/samples/prueba sesil 2.png",
    ],
)
def test_real_sessile_samples_have_stable_contact_angles_and_diagnostic_fit(sample):
    image = cv2.imread(str(Path(sample)))
    assert image is not None

    calibration = AutoCalibrator(image, "sessile").detect_all()
    assert calibration.drop_contour is not None
    assert calibration.contact_points is not None
    assert calibration.substrate_line is not None

    kwargs = {
        "image": sample,
        "drop_contour": calibration.drop_contour,
        "contact_points": calibration.contact_points,
        "apex_point": calibration.apex_point,
        "needle_rect": calibration.needle_rect,
        "roi_rect": calibration.roi_rect,
        "substrate_line": calibration.substrate_line,
        "calibration_params": {
            "needle_diameter_mm": 1.83,
            "drop_density_kg_m3": 1000.0,
            "fluid_density_kg_m3": 1.2,
        },
    }
    if calibration.needle_rect is not None:
        kwargs["scale"] = {"px_per_mm": calibration.needle_rect[2] / 1.83}

    ctx = SessilePipeline().run_with_plan(only=["compute_metrics"], **kwargs)

    for key in ("diameter_mm", "height_mm", "volume_uL"):
        assert np.isfinite(ctx.results[key])
        assert ctx.results[key] > 0

    assert ctx.results["theta_left_deg"] > 10.0
    assert ctx.results["theta_right_deg"] > 10.0
    assert np.isfinite(ctx.results["contact_angle_deg"])
    assert "fit_R0_mm" in ctx.results
    assert "fit_beta" in ctx.results
    assert "R0_mm" not in ctx.results
    assert ctx.results["fit_warning"] == "profile_fit_unreliable"
