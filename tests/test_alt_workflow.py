import numpy as np
import cv2
import pytest
try:
    from menipy.pipelines.sessile.geometry_alt import (
        filter_contours_by_size,
        project_onto_line,
        find_contact_points,
        compute_apex,
        analyze,
        HelperBundle,
    )
except Exception as exc:  # pragma: no cover - dependency issue
    filter_contours_by_size = None
    project_onto_line = None
    find_contact_points = None
    compute_apex = None
    analyze = None
    HelperBundle = None
    missing_dependency = exc
else:
    missing_dependency = None


def test_filter_contours_by_size():
    if filter_contours_by_size is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    c1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    c2 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], np.float32)
    res = filter_contours_by_size([c1, c2], 0.5, 2.0)
    assert len(res) == 1
    assert np.allclose(res[0], c1)


def test_project_onto_line():
    if project_onto_line is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    pts = np.array([[1.0, 1.0], [2.0, 2.0]], float)
    dist, foot = project_onto_line(pts, ((0.0, 0.0), (0.0, 3.0)))
    assert np.allclose(dist, [1.0, 2.0])
    assert np.allclose(foot, [[0.0, 1.0], [0.0, 2.0]])


def test_find_apex_and_contact():
    if find_contact_points is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    theta = np.linspace(0, 2 * np.pi, 200)
    contour = np.stack([10 * np.cos(theta) + 50, 10 * np.sin(theta) + 60], axis=1)
    line = ((40.0, 60.0), (60.0, 60.0))
    p1, p2 = find_contact_points(contour, line, (42, 60), (58, 60))
    apex = compute_apex(contour, line, p1, p2)
    assert np.allclose(p1[1], 60.0, atol=1e-6)
    assert np.allclose(p2[1], 60.0, atol=1e-6)
    assert apex[1] < 60.0


def test_analyze_simple_circle():
    if analyze is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    img = np.full((100, 100), 255, np.uint8)
    cv2.circle(img, (50, 50), 10, 0, -1)
    helpers = HelperBundle(px_per_mm=10.0)
    res = analyze(img, helpers, ((40, 60), (60, 60)), contact_points=((45, 60), (55, 60)))
    d_px = np.linalg.norm(np.subtract(res.p1, res.p2))
    assert pytest.approx(d_px, rel=1e-2) == 20.0
