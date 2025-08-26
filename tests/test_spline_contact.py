import numpy as np
import pytest

from menipy.analysis.sessile import contact_points_from_spline


def _circle_contour(radius=20.0, center_y=10.0, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta) + center_y], axis=1)


def test_contact_points_basic():
    contour = _circle_contour()
    line = ((-40.0, 0.0), (40.0, 0.0))
    p1, p2 = contact_points_from_spline(contour, line, delta=0.5)
    assert np.allclose(p1[1], 0.0, atol=1e-2)
    assert np.allclose(p2[1], 0.0, atol=1e-2)
    assert p1[0] < 0 < p2[0]


def test_contact_points_rotated():
    contour = _circle_contour()
    angle = np.deg2rad(15.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    contour_r = contour @ rot.T
    line = ((-40.0, 0.0), (40.0, 0.0))
    line_r = (tuple(rot @ np.array(line[0])), tuple(rot @ np.array(line[1])))
    p1, p2 = contact_points_from_spline(contour_r, line_r, delta=0.5)
    from ..pipelines.sessile.geometry_alt import line_params
    a, b, c = line_params(line_r[0], line_r[1])
    assert abs(a * p1[0] + b * p1[1] + c) < 1e-2
    assert abs(a * p2[0] + b * p2[1] + c) < 1e-2
    assert p1[0] < p2[0]


def test_contact_points_clamped_to_segment():
    contour = _circle_contour()
    line = ((-10.0, 0.0), (10.0, 0.0))
    p1, p2 = contact_points_from_spline(contour, line, delta=0.5)
    assert np.allclose(p1, (-10.0, 0.0), atol=1e-6)
    assert np.allclose(p2, (10.0, 0.0), atol=1e-6)
