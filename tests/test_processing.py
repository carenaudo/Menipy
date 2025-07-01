import numpy as np

from src.processing.segmentation import (
    otsu_threshold,
    adaptive_threshold,
    morphological_cleanup,
)


def test_otsu_threshold():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[20:30, 20:30] = 255
    mask = otsu_threshold(img)
    assert mask.sum() > 0


def test_adaptive_threshold():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 255
    mask = adaptive_threshold(img, block_size=11, offset=2)
    assert mask.sum() > 0


def test_morphological_cleanup():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[0, 0] = 255  # noise pixel
    cleaned = morphological_cleanup(mask, kernel_size=3, iterations=1)
    assert cleaned[0, 0] == 0
    assert cleaned.sum() >= mask[5:15, 5:15].sum()

