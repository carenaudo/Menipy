import numpy as np
import pytest
import math

from menipy.analysis import compute_drop_metrics, find_apex_index


def test_max_diameter_and_radius_apex():
    cv2 = __import__('cv2')
    radius = 10
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(img, (25, 30), radius, 255, -1)
    roi = img[20:45, 10:40]
    contour = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].squeeze(1).astype(float)
    contour += np.array([10, 20])
    metrics = compute_drop_metrics(contour, px_per_mm=5.0, mode="pendant", needle_diam_mm=1.0)
    assert pytest.approx(metrics["diameter_px"], abs=1) == 2 * radius
    assert metrics["diameter_line"][0][1] == metrics["diameter_line"][1][1]
    assert metrics["contact_line"] is not None
    assert metrics["radius_apex_mm"] > 0
    assert metrics["s1"] > 0
    assert metrics["diameter_center"] == (25, 30)
    assert pytest.approx(metrics["apex_to_diam_mm"], abs=0.1) == 2.0
    assert pytest.approx(metrics["contact_to_diam_mm"], abs=0.1) == 2.0
    assert pytest.approx(metrics["apex_to_contact_mm"], abs=0.1) == 4.0
    # projected area of the circular drop
    assert metrics["A_proj_mm2"] > 0
    assert metrics["A_proj_left_mm2"] > 0
    assert metrics["A_proj_right_mm2"] > 0
    assert metrics["needle_area_mm2"] is not None
    assert metrics["needle_area_mm2"] >= 0
    assert metrics["A_surf_mm2"] > 0
    assert metrics["A_surf_left_mm2"] > 0
    assert metrics["A_surf_right_mm2"] > 0
    assert metrics["A_surf_mean_mm2"] == pytest.approx(
        0.5 * (metrics["A_surf_left_mm2"] + metrics["A_surf_right_mm2"]), rel=1e-2
    )


def test_find_apex_index_median():
    contour = np.array([
        [0, 0],
        [5, 0],
        [10, 0],
        [10, 10],
        [0, 10],
    ], float)
    idx = find_apex_index(contour, "contact-angle")
    assert idx == 1
