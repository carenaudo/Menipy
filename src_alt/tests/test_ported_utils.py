from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from menipy.calibration import set_calibration, mm_to_pixels, pixels_to_mm
from menipy.io.loaders import load_image
from menipy.preprocessing.preprocess import clean_frame
from src.processing.segmentation import morphological_cleanup


def test_mm_px_conversion() -> None:
    set_calibration(10.0)
    assert mm_to_pixels(5.0) == 50.0
    assert pixels_to_mm(50.0) == 5.0


def test_load_image_roundtrip(tmp_path: Path) -> None:
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)
    loaded = load_image(path)
    assert loaded.shape == img.shape


def test_clean_frame_matches_legacy() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    noisy = mask.copy()
    noisy[0, 0] = 255
    new = clean_frame(noisy, kernel_size=3, iterations=1)
    old = morphological_cleanup(noisy, kernel_size=3, iterations=1)
    assert np.array_equal(new, old)
