"""Exact foreground-run selection and full calibration regression."""

from itertools import product

import numpy as np
import pytest

from menipy.common import sessile_detection as detection


def original_center_run(row, center_x):
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return None
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, splits)
    best = min(runs, key=lambda run: 0.0
               if int(run[0]) <= center_x <= int(run[-1])
               else min(abs(int(run[0]) - center_x), abs(int(run[-1]) - center_x)))
    return int(best[0]), int(best[-1])


@pytest.mark.parametrize("center", [-2, 0, 3, 7, 10])
def test_exhaustive_small_rows(center):
    for bits in product([0, 255], repeat=8):
        row = np.array(bits, dtype=np.uint8)
        assert detection._center_run(row, center) == original_center_run(row, center)
    assert detection._center_run(np.array([]), center) is None


def test_single_run_avoids_split(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Single runs need no split")

    monkeypatch.setattr(detection.np, "split", forbidden)
    assert detection._center_run(np.array([0, 1, 1, 0]), 0) == (1, 2)


def assert_calibration_equal(actual, expected):
    for field, value in vars(actual).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(expected, field))
        else:
            assert value == getattr(expected, field), field


@pytest.mark.parametrize("mode,filename", [
    ("sessile", "sessile_needle_reference.png"),
    ("pendant", "pendant_water_reference.png"),
])
def test_reference_image_calibration(mode, filename, monkeypatch):
    from pathlib import Path

    import cv2

    from menipy.common.auto_calibrator import run_auto_calibration

    image = cv2.imread(str(Path(__file__).resolve().parents[1] / "data/samples" / filename))
    actual = run_auto_calibration(image, mode)
    monkeypatch.setattr(detection, "_center_run", original_center_run)
    expected = run_auto_calibration(image, mode)
    assert_calibration_equal(actual, expected)
