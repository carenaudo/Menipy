"""The liquid region closes on its solid interface, not segmentation artifacts."""

import cv2
import numpy as np
import pytest

from menipy.common.liquid_boundary import liquid_surface_arc
from menipy.common.sessile_detection import detect_sessile_drop_contour


@pytest.mark.parametrize("below", [False, True])
@pytest.mark.parametrize("angle", [0, 0.2, -0.3])
@pytest.mark.parametrize("reverse", [False, True])
def test_outer_arc_removes_internal_return_path(below, angle, reverse):
    theta = np.linspace(np.pi, 0, 101)
    side = 1 if below else -1
    outer = np.column_stack([50 * np.cos(theta), side * 30 * np.sin(theta)])
    inner = np.column_stack([np.linspace(50, -50, 101), side * (2 + 3 * np.sin(theta))])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = np.vstack([outer, inner]) @ rotation.T
    contacts = np.array([[-50, 0], [50, 0]]) @ rotation.T
    if reverse:
        points = points[::-1]
    actual = liquid_surface_arc(points, contacts, liquid_below=below)
    np.testing.assert_allclose(actual, outer @ rotation.T, atol=1e-12)
    np.testing.assert_array_equal(
        actual, liquid_surface_arc(actual, contacts, liquid_below=below)
    )
    assert len(actual) < len(points)


def test_detection_excludes_reflection_and_bright_contact_patch():
    image = np.full((300, 400, 3), 230, np.uint8)
    cv2.ellipse(image, (200, 220), (90, 65), 0, 180, 360, (20, 20, 20), -1)
    cv2.rectangle(image, (0, 220), (399, 299), (100, 100, 100), -1)
    cv2.ellipse(image, (200, 220), (22, 12), 0, 180, 360, (210, 210, 210), -1)
    result = detect_sessile_drop_contour(image, substrate_y=220)
    assert result.contour is not None and result.contact_points is not None
    boundary = liquid_surface_arc(result.contour, result.contact_points)
    assert np.max(boundary[:, 1]) <= 220
    np.testing.assert_array_equal(boundary[[0, -1]], result.contact_points)
    # Near the contact chord, only the outside flanks may remain.
    near_base = boundary[boundary[:, 1] > 210]
    assert np.all(np.abs(near_base[:, 0] - 200) > 60)


def test_cancelled_boundary_cleanup():
    from menipy.common.cancellation import (
        AnalysisCancelled,
        CancellationToken,
        cancellation_scope,
    )

    token = CancellationToken()
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            token.cancel()
            liquid_surface_arc(np.zeros((3, 2)), ((0, 0), (1, 0)))


def test_display_boundary_preserves_measured_inputs_and_tilt():
    from menipy.common.auto_calibrator import CalibrationResult
    from menipy.common.liquid_boundary import update_calibration_boundary

    points = np.array([[-5, 0], [-4, -3], [0, -6], [4, -3], [5, 0], [0, -1]])
    result = CalibrationResult(
        drop_contour=points.copy(),
        contact_points=((-5, 0), (5, 0)),
        substrate_line=((-10, -1), (10, 1)),
    )
    update_calibration_boundary(result, "sessile")
    np.testing.assert_array_equal(result.drop_contour, points)
    assert result.contact_points == ((-5, 0), (5, 0))
    assert result.liquid_boundary is not None
    boundary = result.liquid_boundary
    assert np.all(boundary[:, 1] - 0.1 * boundary[:, 0] <= 1e-9)


def test_exact_collinear_display_simplification():
    points = np.array(
        [[0, 0], [0, -1], [0, -2], [1, -2], [2, -2], [2, -1], [2, 0], [1, 0]]
    )
    result = liquid_surface_arc(points, ((0, 0), (2, 0)))
    np.testing.assert_array_equal(result, [[0, 0], [0, -2], [2, -2], [2, 0]])
    duplicated = liquid_surface_arc(np.repeat(points, 2, axis=0), ((0, 0), (2, 0)))
    np.testing.assert_array_equal(duplicated, result)


def test_calibration_overlay_uses_display_boundary():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from menipy.common.auto_calibrator import CalibrationResult
    from menipy.gui.controllers.overlay_manager import OverlayManager

    measured = np.array([[0, 0], [0, -2], [2, -2], [2, 0], [1, -1]])
    boundary = liquid_surface_arc(measured, ((0, 0), (2, 0)))
    result = CalibrationResult(drop_contour=measured, liquid_boundary=boundary)
    view = Mock()
    OverlayManager(view, SimpleNamespace()).draw_calibration_result(result)
    np.testing.assert_array_equal(view.add_marker_contour.call_args.args[0], boundary)
