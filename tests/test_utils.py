from src.utils import set_calibration, get_calibration


def test_calibration_roundtrip():
    set_calibration(2.5)
    cal = get_calibration()
    assert cal.pixels_per_mm == 2.5
    # Ensure getter returns updated value
    assert get_calibration().pixels_per_mm == 2.5
