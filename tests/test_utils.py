import pytest

from menipy.calibration import (
    set_calibration,
    get_calibration,
    calibrate_from_points,
    pixels_to_mm,
    mm_to_pixels,
    auto_calibrate,
)


def test_calibration_roundtrip():
    set_calibration(2.5)
    cal = get_calibration()
    assert cal.pixels_per_mm == 2.5
    # Ensure getter returns updated value
    assert get_calibration().pixels_per_mm == 2.5


def test_calibration_from_points():
    set_calibration(1.0)
    px_len = calibrate_from_points((0, 0), (0, 10), 5.0)
    assert px_len == 10
    assert get_calibration().pixels_per_mm == 2.0
    assert pixels_to_mm(10) == 5.0
    assert mm_to_pixels(5.0) == 10


def test_auto_calibrate_detects_lines():
    import numpy as np
    import cv2

    img = np.full((20, 20), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 0), (15, 19), 0, -1)

    sep = auto_calibrate(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (0, 0, 20, 20), 1.0)
    assert sep == pytest.approx(10, abs=1)
    assert get_calibration().pixels_per_mm == pytest.approx(10, abs=1)

