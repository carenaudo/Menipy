import numpy as np

from src.processing.segmentation import (
    otsu_threshold,
    adaptive_threshold,
    morphological_cleanup,
    external_contour_mask,
    find_contours,
    ml_segment,
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


def test_find_contours():
    mask = np.zeros((30, 30), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.rectangle(mask, (5, 5), (25, 25), 255, -1)
    contours = find_contours(mask)
    assert len(contours) == 1
    assert contours[0].shape[1] == 2


def test_ml_segment():
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255
    mask = ml_segment(img)
    assert mask.sum() > 0


def test_external_contour_mask():
    cv2 = __import__('cv2')
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (35, 35), 255, -1)
    cv2.rectangle(mask, (10, 10), (30, 30), 0, -1)  # hole
    cleaned = external_contour_mask(mask)
    contours = find_contours(cleaned)
    assert len(contours) == 1
    # hole should be filled
    assert cleaned[20, 20] == 255

