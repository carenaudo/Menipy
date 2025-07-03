import numpy as np
import pytest

from src.processing.segmentation import (
    otsu_threshold,
    adaptive_threshold,
    morphological_cleanup,
    external_contour_mask,
    find_contours,
    ml_segment,
)
from src.processing import detect_droplet, detect_pendant_droplet


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



def test_detect_droplet_circle():
    cv2 = __import__('cv2')
    frame = np.full((400, 400), 255, dtype=np.uint8)
    x0, y0, w, h = 50, 50, 300, 300
    radius = 100
    center = (x0 + radius, y0 + radius)
    cv2.circle(frame, center, radius, 0, -1)
    px_to_mm = 0.1

    droplet = detect_droplet(frame, (x0, y0, w, h), px_to_mm)

    assert abs(droplet.apex_px[0] - (x0 + radius)) <= 1
    assert abs(droplet.apex_px[1] - (y0 + 2 * radius)) <= 1
    assert droplet.contact_px[1] == y0
    assert droplet.contact_px[3] == y0
    expected_r = radius * px_to_mm
    assert np.isclose(droplet.r_max_mm, expected_r, rtol=5e-3)
    expected_area = np.pi * expected_r**2
    assert np.isclose(droplet.projected_area_mm2, expected_area, rtol=1e-2)


def test_detect_droplet_translation():
    cv2 = __import__('cv2')
    frame = np.full((400, 400), 255, dtype=np.uint8)
    x0, y0, w, h = 50, 50, 300, 300
    radius = 100
    shift = 40
    center1 = (x0 + radius, y0 + radius)
    center2 = (center1[0] + shift, center1[1])
    cv2.circle(frame, center1, radius, 0, -1)
    droplet1 = detect_droplet(frame, (x0, y0, w, h), 0.1)

    frame.fill(255)
    cv2.circle(frame, center2, radius, 0, -1)
    droplet2 = detect_droplet(frame, (x0, y0, w, h), 0.1)

    assert droplet2.apex_px[0] - droplet1.apex_px[0] == shift
    assert np.isclose(droplet1.r_max_mm, droplet2.r_max_mm, rtol=6e-3)
    assert np.isclose(droplet1.projected_area_mm2, droplet2.projected_area_mm2, rtol=6e-3)


def test_detect_droplet_failure():
    frame = np.full((100, 100), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_droplet(frame, (10, 10, 50, 50), 0.1)


def test_detect_pendant_droplet_simple():
    cv2 = __import__("cv2")
    frame = np.full((200, 200), 255, dtype=np.uint8)
    x0, y0, w, h = 60, 40, 80, 120
    radius = 30
    center = (x0 + w // 2, y0 + radius + 20)
    cv2.circle(frame, center, radius, 0, -1)
    y_line = center[1] - radius
    cv2.line(frame, (x0, y_line), (x0 + w - 1, y_line), 0, 4)
    droplet = detect_pendant_droplet(frame, (x0, y0, w, h), 0.1)

    assert abs(droplet.apex_px[1] - (center[1] + radius)) <= 1
    assert abs(droplet.contact_px[1] - y_line) <= 2

