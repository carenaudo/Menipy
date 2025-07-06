import numpy as np
import cv2

from menipy.analysis import (
    compute_drop_metrics as new_compute,
    compute_pendant_metrics,
    compute_sessile_metrics,
    find_apex_index as new_find,
)
from src.analysis.drop import compute_drop_metrics as legacy_compute
from src.analysis.drop import find_apex_index as legacy_find


def _make_circle_contour(radius: int = 10) -> np.ndarray:
    img = np.zeros((radius * 4, radius * 4), dtype=np.uint8)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.circle(img, center, radius, 255, -1)
    contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    return contour.squeeze(1).astype(float)


def test_compute_drop_metrics_parity() -> None:
    contour = _make_circle_contour()
    new = new_compute(contour, px_per_mm=5.0, mode="pendant", needle_diam_mm=1.0)
    old = legacy_compute(contour, px_per_mm=5.0, mode="pendant", needle_diam_mm=1.0)
    assert new["apex"] == old["apex"]
    assert np.isclose(new["diameter_mm"], old["diameter_mm"])
    assert np.isclose(new["height_mm"], old["height_mm"])


def test_wrapper_functions() -> None:
    contour = _make_circle_contour()
    new1 = compute_pendant_metrics(contour, px_per_mm=5.0, needle_diam_mm=1.0)
    old = legacy_compute(contour, px_per_mm=5.0, mode="pendant", needle_diam_mm=1.0)
    assert new1["apex"] == old["apex"]

    new2 = compute_sessile_metrics(contour, px_per_mm=5.0)
    old2 = legacy_compute(contour, px_per_mm=5.0, mode="contact-angle")
    assert np.isclose(new2["diameter_mm"], old2["diameter_mm"])


def test_find_apex_index_parity() -> None:
    contour = np.array([
        [0, 0],
        [5, 0],
        [10, 0],
        [10, 10],
        [0, 10],
    ], float)
    assert new_find(contour, "contact-angle") == legacy_find(contour, "contact-angle")
