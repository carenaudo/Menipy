from __future__ import annotations

import numpy as np
import cv2

from menipy.detection.needle import detect_vertical_edges as new_detect_needle
from menipy.detection.substrate import detect_substrate_line as new_detect_substrate
from menipy.detection.droplet import (
    detect_droplet as new_detect_droplet,
    detect_pendant_droplet as new_detect_pendant_droplet,
)

from src.analysis.needle import detect_vertical_edges as legacy_detect_needle
from src.processing.substrate import detect_substrate_line as legacy_detect_substrate
from src.processing.detection import (
    detect_droplet as legacy_detect_droplet,
    detect_pendant_droplet as legacy_detect_pendant_droplet,
)


def test_needle_detection_parity() -> None:
    img = np.zeros((80, 40), dtype=np.uint8)
    cv2.line(img, (15, 10), (15, 70), 255, 2)
    cv2.line(img, (25, 10), (25, 70), 255, 2)
    new = new_detect_needle(img)
    old = legacy_detect_needle(img)
    assert new == old


def test_substrate_detection_parity() -> None:
    img = np.zeros((60, 100), dtype=np.uint8)
    cv2.line(img, (0, 45), (99, 45), 255, 2)
    mask = np.zeros_like(img)
    mask[20:40, 20:80] = 255
    new_p1, new_p2 = new_detect_substrate(img, mask, "sessile")
    old_p1, old_p2 = legacy_detect_substrate(img, mask, "sessile")
    assert np.allclose(new_p1, old_p1)
    assert np.allclose(new_p2, old_p2)


def test_detect_droplet_parity() -> None:
    frame = np.full((200, 200), 255, dtype=np.uint8)
    x0, y0, w, h = 50, 50, 100, 100
    radius = 30
    center = (x0 + radius, y0 + radius)
    cv2.circle(frame, center, radius, 0, -1)
    new = new_detect_droplet(frame, (x0, y0, w, h), 0.1)
    old = legacy_detect_droplet(frame, (x0, y0, w, h), 0.1)
    assert new.apex_px == old.apex_px
    assert new.contact_px == old.contact_px
    assert np.allclose(new.contour_px, old.contour_px)
    assert np.isclose(new.r_max_mm, old.r_max_mm)
    assert np.isclose(new.projected_area_mm2, old.projected_area_mm2)


def test_detect_pendant_droplet_parity() -> None:
    frame = np.full((150, 150), 255, dtype=np.uint8)
    x0, y0, w, h = 30, 20, 90, 110
    radius = 25
    center = (x0 + w // 2, y0 + radius + 20)
    cv2.circle(frame, center, radius, 0, -1)
    y_line = center[1] - radius
    cv2.line(frame, (x0, y_line), (x0 + w - 1, y_line), 0, 3)
    new = new_detect_pendant_droplet(frame, (x0, y0, w, h), 0.1)
    old = legacy_detect_pendant_droplet(frame, (x0, y0, w, h), 0.1)
    assert new.apex_px == old.apex_px
    assert np.allclose(new.contour_px, old.contour_px)
    assert np.isclose(new.r_max_mm, old.r_max_mm)
    assert np.isclose(new.projected_area_mm2, old.projected_area_mm2)
