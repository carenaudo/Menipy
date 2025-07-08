import numpy as np
from menipy.analysis.sessile import compute_metrics as compute_sessile_metrics, contact_points_from_spline


def _circle_contour(radius=20.0, center_y=10.0, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta) + center_y], axis=1)


def test_compute_metrics_uses_spline_points():
    contour = _circle_contour()
    line = ((-40.0, 0.0), (40.0, 0.0))
    metrics = compute_sessile_metrics(contour, 10.0, substrate_line=line)
    cp1, cp2 = contact_points_from_spline(contour, line, delta=0.5)
    cp1 = (int(round(cp1[0])), int(round(cp1[1])))
    cp2 = (int(round(cp2[0])), int(round(cp2[1])))
    cl = metrics["contact_line"]
    assert cl[0] == cp1
    assert cl[1] == cp2
