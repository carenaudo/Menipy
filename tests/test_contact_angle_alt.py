import numpy as np
import pytest


from src.detectors.geometry_alt import (
    trim_poly_between,
    project_pts_onto_poly,
    symmetry_axis,
    polyline_contour_intersections,
    geom_metrics_alt,
)



def test_trim_between_simple():
    poly = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    p1 = np.array([1, 0], float)
    p2 = np.array([2, 0], float)
    seg = trim_poly_between(poly, p1, p2)
    assert seg.shape[0] == 2
    assert np.allclose(seg[0], p1)
    assert np.allclose(seg[-1], p2)


def test_projection_and_axis_tilt():
    poly = np.array([[-10, 0], [10, 0]], float)
    pts = np.array([[0, 5]], float)
    d, foot = project_pts_onto_poly(pts, poly)
    assert pytest.approx(d[0], rel=1e-6) == 5
    assert np.allclose(foot[0], [0, 0])

    rot = np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]])
    poly_tilt = poly @ rot.T
    apex = np.array([0, 5]) @ rot.T
    p1 = poly_tilt[0]
    p2 = poly_tilt[1]
    s1, s2 = symmetry_axis(apex, poly_tilt, p1, p2)
    assert s1.shape == (2,)
    assert s2.shape == (2,)



def test_intersections_and_metrics_alt():
    px_per_mm = 10.0
    theta = np.linspace(0, 2 * np.pi, 200)
    r_px = 20.0
    contour = np.stack([r_px * np.cos(theta), r_px * np.sin(theta) + r_px], axis=1)
    poly = np.array([[-40.0, r_px], [40.0, r_px]], float)
    pts = polyline_contour_intersections(poly, contour)
    assert len(pts) >= 2
    apex_idx = int(np.argmax(contour[:, 1]))
    metrics = geom_metrics_alt(poly, contour, apex_idx, px_per_mm)
    assert metrics["droplet_poly"].shape[0] > 0
    assert pytest.approx(metrics["w_mm"], rel=1e-2) == 4.0

