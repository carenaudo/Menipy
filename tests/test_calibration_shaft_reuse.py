"""Run-local shaft reuse preserves fallback selection and calibration outputs."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common import auto_calibrator as calibration
from menipy.common import sessile_detection as detection
from tests.test_needle_single_run import assert_calibration_equal


def repeated_shaft_drop(image, **kwargs):
    kwargs.pop("needle_shaft_result", None)
    return detection.detect_sessile_drop_contour(image, **kwargs)


@pytest.mark.parametrize("mode,filename", [
    ("sessile", "sessile_needle_reference.png"),
    ("pendant", "pendant_water_reference.png"),
])
def test_exact_full_calibration(mode, filename, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data/samples" / filename
    image = cv2.imread(str(source))
    actual = calibration.run_auto_calibration(image, mode)
    monkeypatch.setattr(calibration, "detect_sessile_drop_contour", repeated_shaft_drop)
    expected = calibration.run_auto_calibration(image, mode)
    assert_calibration_equal(actual, expected)


@pytest.mark.parametrize("expansion", [None, 35])
def test_prepared_result_including_failure(expansion, monkeypatch):
    image = np.full((100, 100), 255, dtype=np.uint8)
    image[10:80, 30:70] = 0
    result = (None, 0.0, expansion)
    monkeypatch.setattr(detection, "detect_sessile_needle_shaft", lambda *a, **k: result)
    expected = detection._segment_sessile_otsu_fallback(image, substrate_y=85)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared shaft must not be recomputed")

    monkeypatch.setattr(detection, "detect_sessile_needle_shaft", forbidden)
    actual = detection._segment_sessile_otsu_fallback(
        image, substrate_y=85, needle_shaft_result=result)
    np.testing.assert_array_equal(actual, expected)


def test_each_calibration_run_recomputes_shaft(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data/samples/sessile_needle_reference.png"
    calibrator = calibration.AutoCalibrator(cv2.imread(str(source)), "sessile")
    calls = []
    original = detection.detect_sessile_needle_shaft

    def counted(*args, **kwargs):
        calls.append(kwargs["substrate_y"])
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration, "detect_sessile_needle_shaft", counted)
    monkeypatch.setattr(detection, "detect_sessile_needle_shaft", counted)
    first = calibrator.detect_all()
    assert len(calls) == 1
    second = calibrator.detect_all()
    assert len(calls) == 2
    assert_calibration_equal(first, second)
