import numpy as np
import pytest

from menipy.common.geometry import (
    find_contact_points_from_contour,
    detect_baseline_ransac,
    refine_apex_curvature,
)
from menipy.pipelines.sessile.metrics import compute_sessile_metrics


def _circle_contour(center=(50, 50), radius=10, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    return np.stack([x, y], axis=1)


def test_find_contact_points_from_contour_circle():
    contour = _circle_contour(center=(50, 50), radius=10)
    line = ((40.0, 60.0), (60.0, 60.0))
    p1, p2 = find_contact_points_from_contour(contour, line, tolerance=3.0)
    assert p1 is not None and p2 is not None
    # Should land on opposite sides of the contour and roughly horizontal
    assert p1[0] < p2[0]
    assert 40.0 <= p1[1] <= 60.0
    assert 40.0 <= p2[1] <= 60.0


def test_detect_baseline_ransac_bottom_aligns():
    # Rectangle contour spanning x=0..20, y=0..10
    contour = np.array(
        [[0, 0], [20, 0], [20, 10], [0, 10]], dtype=float
    )
    p1, p2, conf = detect_baseline_ransac(contour, threshold=1.0, min_samples=3)
    assert conf >= 0  # confidence returned
    # Baseline should be within the vertical extent
    assert 0.0 <= p1[1] <= 10.0
    assert 0.0 <= p2[1] <= 10.0


def test_refine_apex_curvature_on_circle():
    contour = _circle_contour(center=(50, 50), radius=10)
    apex, conf = refine_apex_curvature(contour, window=5)
    # Apex should lie on the contour vertical span
    assert 40.0 <= apex[1] <= 60.0
    assert conf >= 0.0


def test_compute_sessile_metrics_circle():
    contour = _circle_contour(center=(50, 50), radius=10)
    px_per_mm = 10.0
    substrate = ((40.0, 60.0), (60.0, 60.0))

    res = compute_sessile_metrics(
        contour,
        px_per_mm=px_per_mm,
        substrate_line=substrate,
        auto_detect_baseline=False,
        auto_detect_apex=True,
        contact_angle_method="spherical_cap",
    )

    # Ensure outputs are finite and non-negative
    for key in ("diameter_mm", "height_mm", "volume_uL",
                "theta_left_deg", "theta_right_deg",
                "contact_angle_deg"):
        assert res[key] >= 0.0
    assert 0.0 <= res["baseline_confidence"] <= 1.0
    assert 0.0 <= res["apex_confidence"] <= 1.0
