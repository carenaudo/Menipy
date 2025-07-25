import numpy as np
import pytest

from menipy.physics.contact_geom import (
    geom_metrics,
    contour_line_intersections,
    contour_line_intersection_near,
    line_params,
)
cv2 = __import__('cv2')


def test_circle_geom_metrics():
    px_per_mm = 10.0
    r_mm = 1.0
    r_px = r_mm * px_per_mm
    theta = np.linspace(0, 2 * np.pi, 200)
    contour = np.stack([r_px * np.cos(theta), r_px * np.sin(theta) + r_px], axis=1)
    angle = np.deg2rad(10)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    contour_rot = contour @ rot.T
    p1 = (-2 * r_px, r_px)
    p2 = (2 * r_px, r_px)
    p1 = rot @ p1
    p2 = rot @ p2
    apex_idx = int(np.argmax(contour_rot[:, 1]))
    metrics = geom_metrics(tuple(p1), tuple(p2), contour_rot, apex_idx, px_per_mm)
    assert metrics["rb_mm"] == pytest.approx(1.0, rel=1e-2)
    assert metrics["h_mm"] == pytest.approx(1.0, rel=2e-2)
    poly = metrics["droplet_poly"]
    area_px = cv2.contourArea(poly.astype(np.float32))
    assert area_px == pytest.approx(0.5 * np.pi * r_px ** 2, rel=1e-2)


def test_intersections_wraparound_edge():
    contour = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        float,
    )
    p1 = (-0.5, 0.5)
    p2 = (1.5, 0.5)
    a, b, c = line_params(p1, p2)
    left, right = contour_line_intersections(contour, a, b, c)
    pts = {tuple(np.round(pt, 6)) for pt in (left, right)}
    assert pts == {(0.0, 0.5), (1.0, 0.5)}


def test_intersection_near_selects_correct_point():
    contour = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ],
        float,
    )
    p1 = (-1.0, 0.5)
    p2 = (3.0, 0.5)
    a, b, c = line_params(p1, p2)
    left, _ = contour_line_intersection_near(contour, a, b, c, (0.1, 0.5))
    right, _ = contour_line_intersection_near(contour, a, b, c, (1.9, 0.5))
    assert np.allclose(left, [0.0, 0.5])
    assert np.allclose(right, [2.0, 0.5])
