import numpy as np
import pytest

from src.analysis.drop import compute_drop_metrics, find_apex_index


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
