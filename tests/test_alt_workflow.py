import numpy as np
import cv2
import pytest
try:
    from menipy.pipelines.sessile.geometry_alt import (
        filter_contours_by_size,
        project_onto_line,
        find_contact_points,
        compute_apex,
        compute_contact_angles,
        clean_droplet_contour,
        analyze,
        HelperBundle,
    )
except Exception as exc:  # pragma: no cover - dependency issue
    filter_contours_by_size = None
    project_onto_line = None
    find_contact_points = None
    compute_apex = None
    compute_contact_angles = None
    analyze = None
    HelperBundle = None
    clean_droplet_contour = None
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


def test_clean_droplet_contour_basic():
    if clean_droplet_contour is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    img = np.zeros((80, 100), np.uint8)
    cv2.circle(img, (50, 30), 20, 255, -1)
    img[70:, :] = 255  # noise below substrate
    contours = clean_droplet_contour(img, 60, min_area=50)
    assert len(contours) == 1
    cnt = contours[0]
    assert cnt.shape[1] == 2
    assert cnt[:, 1].max() < 60


def test_analyze_removes_substrate_noise():
    if analyze is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    img = np.zeros((80, 100), np.uint8)
    cv2.circle(img, (50, 30), 20, 255, -1)
    img[70:, :] = 255
    helpers = HelperBundle(px_per_mm=10.0)
    res = analyze(img, helpers, ((40, 60), (60, 60)))
    assert res.contour[:, 1].max() < 60


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


def test_find_contact_points_ignores_below_substrate():
    if find_contact_points is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    theta = np.linspace(0, 2 * np.pi, 200)
    base = np.stack([10 * np.cos(theta) + 50, 10 * np.sin(theta) + 60], axis=1)
    noise = np.array([[55.0, 65.0], [56.0, 70.0], [54.0, 62.0]])
    contour = np.vstack([base, noise])
    line = ((40.0, 60.0), (60.0, 60.0))
    p1, p2 = find_contact_points(contour, line, (40, 60), (60, 60))
    assert p1[0] < 41
    assert p2[0] > 59
    assert p1[1] <= 60.0
    assert p2[1] <= 60.0


def test_analyze_simple_circle():
    if analyze is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    img = np.full((100, 100), 255, np.uint8)
    cv2.circle(img, (50, 50), 10, 0, -1)
    helpers = HelperBundle(px_per_mm=10.0)
    res = analyze(img, helpers, ((40, 60), (60, 60)), contact_points=((45, 60), (55, 60)))
    d_px = np.linalg.norm(np.subtract(res.p1, res.p2))
    assert pytest.approx(d_px, rel=1e-2) == 20.0


def test_contact_angle_computation():
    if compute_contact_angles is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    theta = np.linspace(0, 2 * np.pi, 200)
    contour = np.stack([10 * np.cos(theta) + 50, 10 * np.sin(theta) + 60], axis=1)
    p1 = np.array([40.0, 60.0])
    p2 = np.array([60.0, 60.0])
    res = compute_contact_angles(contour, p1, p2, 2.0, 1.0, (p1, p2))
    assert "theta_spherical_p1" in res
    assert pytest.approx(res["theta_spherical_p1"], rel=1e-2) == 90.0
    assert pytest.approx(res["theta_slope_p1"], rel=1e-1) == 90.0
