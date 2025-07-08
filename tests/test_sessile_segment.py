import numpy as np
from menipy.analysis.sessile import smooth_contour_segment


def _circle(radius=20.0, center=(0.0, -10.0), n=300):
    theta = np.linspace(0, 2 * np.pi, n)
    return np.stack(
        [radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1]],
        axis=1,
    )


def test_segment_above_substrate():
    contour = _circle()
    line = ((-40.0, 0.0), (40.0, 0.0))
    seg, p1, p2 = smooth_contour_segment(contour, line, "left", delta=0.1, min_cluster=5)
    assert np.all(seg[:, 1] < 0.0 + 1e-6)
    assert p1[1] <= 0.0 + 1e-6
    assert p2[1] <= 0.0 + 1e-6


def test_segment_side_filtering():
    contour = _circle()
    line = ((-40.0, 0.0), (40.0, 0.0))
    left, _, _ = smooth_contour_segment(contour, line, "left", delta=0.1, min_cluster=5)
    right, _, _ = smooth_contour_segment(contour, line, "right", delta=0.1, min_cluster=5)
    assert left[:, 0].max() <= 0.0 + 1e-6
    assert right[:, 0].min() >= 0.0 - 1e-6
