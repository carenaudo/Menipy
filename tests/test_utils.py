from src.utils import (
    set_calibration,
    get_calibration,
    calibrate_from_points,
    pixels_to_mm,
    mm_to_pixels,
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

