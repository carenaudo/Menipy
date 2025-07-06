import numpy as np
import pytest

from menipy.processing.region import close_droplet
from menipy.processing.metrics import metrics_sessile, metrics_pendant


def _circle_contour(r=20.0, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta) + r], axis=1)


def test_close_droplet_clusters():
    pts = [
        [-20, 0],
        [-15, 0.3],
        [-10, -0.3],
        [-5, 0.3],
        [0, -0.3],
        [5, 0.3],
        [10, -0.3],
        [15, 0.3],
        [20, 0],
        [20, 40],
        [-20, 40],
    ]
    contour = np.array(pts, float)
    mask, p1, p2 = close_droplet(contour, (-25, 0), (50, 0), "sessile")
    assert mask.any()
    assert p1[0] < 0 < p2[0]


def test_metrics_sessile_basic():
    contour = _circle_contour()
    line = ((-40.0, 20.0), (40.0, 20.0))
    m = metrics_sessile(contour, line, px_per_mm=10.0)
    assert pytest.approx(m["diameter_mm"], rel=1e-2) == 4.0
    assert pytest.approx(m["height_mm"], rel=1e-2) == 2.0


def test_metrics_pendant_basic():
    contour = _circle_contour()
    line = ((-40.0, 20.0), (40.0, 20.0))
    m = metrics_pendant(contour, line, px_per_mm=10.0)
    assert pytest.approx(m["diameter_mm"], rel=1e-2) == 4.0
    assert pytest.approx(m["height_mm"], rel=1e-2) == 2.0

