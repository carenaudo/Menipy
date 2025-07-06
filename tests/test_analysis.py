import numpy as np
import pytest

from menipy.analysis import (
    detect_vertical_edges,
    extract_external_contour,
    compute_drop_metrics,
)


def test_detect_vertical_edges_simple():
    cv2 = __import__('cv2')
    img = np.full((40, 40), 255, dtype=np.uint8)
    cv2.line(img, (15, 5), (15, 35), 0, 2)
    cv2.line(img, (25, 5), (25, 35), 0, 2)
    top, bottom, length = detect_vertical_edges(img)
    assert abs(top[1] - 5) <= 1
    assert abs(bottom[1] - 35) <= 1
    assert abs(top[0] - 20) <= 1
    assert top[0] == bottom[0]
    assert abs(length - 10) <= 4


def test_detect_vertical_edges_failure():
    img = np.full((20, 20), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_vertical_edges(img)


def test_extract_external_contour_with_hole():
    cv2 = __import__('cv2')
    img = np.full((40, 40), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (35, 35), 0, -1)
    cv2.rectangle(img, (10, 10), (30, 30), 255, -1)
    contour = extract_external_contour(img)
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    assert x <= 5 and x + w >= 35
    assert y <= 5 and y + h >= 35


def test_compute_drop_metrics_circle():
    cv2 = __import__('cv2')
    radius = 10
    img = np.zeros((40, 40), dtype=np.uint8)
    center = (20, 20)
    cv2.circle(img, center, radius, 255, -1)
    contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].squeeze(1).astype(float)

    metrics = compute_drop_metrics(contour, px_per_mm=10.0, mode="pendant", needle_diam_mm=1.0)
    assert metrics["apex"] == (20, 30)
    assert pytest.approx(metrics["diameter_mm"], rel=0.05) == 2 * radius / 10.0
    assert pytest.approx(metrics["height_mm"], rel=0.05) == 2 * radius / 10.0
    assert metrics["volume_uL"] is not None


def test_compute_drop_metrics_invalid_mode():
    contour = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    with pytest.raises(ValueError):
        compute_drop_metrics(contour, px_per_mm=1.0, mode="bad", needle_diam_mm=1.0)
