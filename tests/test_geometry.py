import numpy as np
import pytest

from src.detectors.geometry_alt import (
    mirror_filter,
    find_substrate_intersections,
    apex_point,
    split_contour_by_line,
    symmetry_axis,
)


def _circle_contour(r=10.0, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta) + r], axis=1)


def test_halfplane_and_apex():
    contour = _circle_contour()
    line_pt = np.array([-20.0, 10.0])
    line_dir = np.array([40.0, 0.0])
    filt = mirror_filter(contour, line_pt, line_dir)
    assert filt.shape[0] <= contour.shape[0]
    p1, p2 = find_substrate_intersections(contour, line_pt, line_dir)
    seg = split_contour_by_line(contour, line_pt, line_dir)
    assert any(np.allclose(pt, p1) for pt in seg)
    assert any(np.allclose(pt, p2) for pt in seg)
    apex, _ = apex_point(seg, line_pt, line_dir, "sessile")
    assert apex[1] == pytest.approx(20.0, rel=1e-2)

    # symmetry axis should be vertical
    pt, vec = symmetry_axis(apex, line_dir)
    angle = np.degrees(np.arccos(np.clip(np.dot(vec, [0, 1]), -1.0, 1.0)))
    assert angle <= 1.0


def test_apex_point_pendant():
    contour = _circle_contour()
    line_pt = np.array([-20.0, 20.0])
    line_dir = np.array([40.0, 0.0])
    seg = split_contour_by_line(contour, line_pt, line_dir, keep_above=False)
    apex, _ = apex_point(seg, line_pt, line_dir, "pendant")
    assert apex[1] == pytest.approx(0.0, abs=1e-2)
